"""SPICE execution boundary for the AutoAnalog CTLE evaluator."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from rl.dfe import optimize_one_tap


class SpiceEvaluator:
    """Map RL actions to legal device values and run gated ngspice analyses."""

    PARAMETER_NAMES = ("W_in", "R_load", "I_bias", "R_s", "C_s")

    # Generic Level-1 NMOS (vto [V], kp [A/V^2]) per process corner. These are
    # placeholder skews so the PVT matrix exercises distinct devices before the
    # IHP PSP deck is available; SF/FS carry only the NMOS half of the skew
    # because the CTLE has no PMOS devices.
    GENERIC_PROCESS_MODELS = {
        "TT": (0.45, 200e-6),
        "SS": (0.51, 170e-6),
        "FF": (0.39, 230e-6),
        "SF": (0.48, 185e-6),
        "FS": (0.42, 215e-6),
    }

    # 5 Gbps NRZ: one unit interval is 200 ps. The transient gate sends the
    # 127-bit PRBS7 pattern PRBS_PERIODS times and measures the final period.
    UNIT_INTERVAL_S = 200e-12
    PRBS_PERIODS = 2
    TRANSIENT_STEP_S = 2e-12

    # Lossy channel used only by the transient gate: two RC sections with a
    # real pole each at CHANNEL_POLE_HZ, i.e. roughly -10 dB at the 2.5 GHz
    # Nyquist frequency and -20 dB at 5 GHz, which is the loss slope of a
    # PCIe Gen 2 class FR-4 trace. The DC/AC gates bypass it so peaking_boost
    # still measures the CTLE alone.
    CHANNEL_POLE_HZ = 1.7e9
    CHANNEL_SECTION_OHMS = (50.0, 500.0)

    # OpenMP threads per ngspice process; see _run_ngspice.
    NGSPICE_THREADS = 1

    # Default eye acceptance for run_transient's tran_valid flag; the RL reward
    # applies CtleSpecifications, which should agree with these. Both were
    # calibrated on the generic Level-1 model; PSP103 devices have roughly
    # half the gain, so IHP training passes a lower eye_height_min_v.
    EYE_HEIGHT_MIN_V = 0.5
    EYE_WIDTH_MIN_UI = 0.7

    # IHP sg13g2 cornerMOSlv.lib section names keyed by the rl.pvt corner names.
    PDK_PROCESS_SECTIONS = {
        "TT": "mos_tt",
        "SS": "mos_ss",
        "FF": "mos_ff",
        "SF": "mos_sf",
        "FS": "mos_fs",
    }

    def __init__(
        self,
        template_path: str | Path | None = None,
        ngspice_binary: str | None = None,
        pdk_model_path: str | Path | None = None,
        pdk_corner_path: str | Path | None = None,
        pdk_corner: str = "mos_tt",
        osdi_model_paths: tuple[str | Path, ...] = (),
        eye_height_min_v: float | None = None,
        eye_width_min_ui: float | None = None,
    ) -> None:
        project_root = Path(__file__).resolve().parents[1]
        self.template_path = Path(
            template_path or project_root / "netlists" / "ctle_template.sp"
        )
        self.template = self.template_path.read_text(encoding="utf-8")

        # Resolve: explicit argument > NGSPICE environment variable > "ngspice" on PATH.
        self.ngspice_binary = ngspice_binary or os.environ.get("NGSPICE", "ngspice")
        self.pdk_model_path = Path(pdk_model_path) if pdk_model_path else None
        self.pdk_corner_path = (
            Path(pdk_corner_path) if pdk_corner_path else None
        )
        self.pdk_corner = pdk_corner
        self.osdi_model_paths = tuple(Path(path) for path in osdi_model_paths)
        # Eye acceptance for tran_valid; keep in step with CtleSpecifications.
        self.eye_height_min_v = self.EYE_HEIGHT_MIN_V if eye_height_min_v is None else float(eye_height_min_v)
        self.eye_width_min_ui = self.EYE_WIDTH_MIN_UI if eye_width_min_ui is None else float(eye_width_min_ui)

        self.bounds = {
            "W_in": (0.5e-6, 50.0e-6),
            "R_load": (100.0, 1000.0),
            "I_bias": (100.0e-6, 2.0e-3),
            "R_s": (10.0, 500.0),
            "C_s": (1.0e-15, 1.0e-12),
        }

    # Files under <pdk_root>/ihp-sg13g2/libs.tech/ngspice that the IHP path needs.
    PDK_MODEL_LIB = "models/sg13g2_moslv_mod.lib"
    PDK_CORNER_LIB = "models/cornerMOSlv.lib"
    PDK_OSDI_MODELS = ("osdi/psp103.osdi", "osdi/psp103_nqs.osdi", "osdi/mosvar.osdi")

    @classmethod
    def for_model_source(
        cls,
        model_source: str = "generic",
        ngspice_binary: str | None = None,
        pdk_root: str | Path | None = None,
        **kwargs: Any,
    ) -> "SpiceEvaluator":
        """Build an evaluator for the generic Level-1 model or the IHP sg13g2 PSP103 OSDI models.

        ``pdk_root`` (or ``$IHP_PDK_ROOT``) is an IHP Open PDK checkout whose
        ``libs.tech/ngspice/osdi`` holds the OpenVAF-compiled models; see
        ``scripts/check_pdk.py``.
        """
        if model_source == "generic":
            return cls(ngspice_binary=ngspice_binary, **kwargs)
        if model_source != "ihp":
            raise ValueError(f"unsupported model source: {model_source}")
        root = pdk_root or os.environ.get("IHP_PDK_ROOT")
        if not root:
            raise ValueError("model_source='ihp' needs pdk_root or $IHP_PDK_ROOT")
        ngspice_dir = Path(root) / "ihp-sg13g2" / "libs.tech" / "ngspice"
        osdi_paths = tuple(ngspice_dir / name for name in cls.PDK_OSDI_MODELS)
        missing = [str(path) for path in (ngspice_dir / cls.PDK_MODEL_LIB, ngspice_dir / cls.PDK_CORNER_LIB, *osdi_paths) if not path.exists()]
        if missing:
            raise FileNotFoundError("IHP PDK files missing (run scripts/check_pdk.py): " + ", ".join(missing))
        return cls(
            ngspice_binary=ngspice_binary,
            pdk_model_path=ngspice_dir / cls.PDK_MODEL_LIB,
            pdk_corner_path=ngspice_dir / cls.PDK_CORNER_LIB,
            osdi_model_paths=osdi_paths,
            **kwargs,
        )

    @property
    def model_source(self) -> str:
        return "ihp_ngspice" if self.osdi_model_paths else "ngspice_generic_level1"

    def map_actions(self, actions: np.ndarray) -> dict[str, float]:
        """Linearly map five normalized actions from [-1, 1] to SI values."""
        values = np.asarray(actions, dtype=float)

        if values.shape != (5,):
            raise ValueError("actions must contain exactly five values")

        if (
            not np.all(np.isfinite(values))
            or np.any(values < -1.0)
            or np.any(values > 1.0)
        ):
            raise ValueError(
                "actions must be finite and bounded by [-1.0, 1.0]"
            )

        return {
            name: lower
            + (float(value) + 1.0) * (upper - lower) / 2.0
            for name, value in zip(self.PARAMETER_NAMES, values)
            for lower, upper in [self.bounds[name]]
        }

    def run_simulation(self, actions: np.ndarray) -> dict[str, Any]:
        """Run .op, then .ac; return a failure result instead of leaking ngspice errors."""
        failure = {
            "dc_valid": False,
            "dc_gain": float("nan"),
            "nyquist_gain": float("nan"),
            "peaking_boost": float("nan"),
            "power": float("nan"),
            "error": None,
            "ac_frequency_hz": np.array([]),
            "ac_gain_db": np.array([]),
        }

        try:
            parameters = self.map_actions(actions)
            netlist = self._inject_parameters(parameters)

            with tempfile.TemporaryDirectory(
                prefix="autoanalog-"
            ) as directory:
                base = Path(directory) / "ctle"

                operating_point = self._run_ngspice(
                    self._with_commands(
                        netlist,
                        self._op_commands(),
                    ),
                    base.with_name("op"),
                )

                output_voltage = self._parse_scalar(
                    operating_point,
                    "v(outp)",
                )

                if (
                    not np.isfinite(output_voltage)
                    or not 0.1 <= output_voltage <= 1.1
                ):
                    return failure

                ac_output = self._run_ngspice(
                    self._with_commands(
                        netlist,
                        self._ac_commands(),
                    ),
                    base.with_name("ac"),
                )

                frequencies, gains = self._parse_ac_data(ac_output)

                dc_gain = self._nearest_value(
                    frequencies,
                    gains,
                    10e6,
                )

                nyquist_gain = self._nearest_value(
                    frequencies,
                    gains,
                    2.5e9,
                )

                if (
                    not np.isfinite(dc_gain)
                    or not np.isfinite(nyquist_gain)
                ):
                    return failure

                return {
                    "dc_valid": True,
                    "dc_gain": dc_gain,
                    "nyquist_gain": nyquist_gain,
                    "peaking_boost": nyquist_gain - dc_gain,
                    "power": 1.2 * parameters["I_bias"],
                    "ac_frequency_hz": frequencies,
                    "ac_gain_db": gains,
                    "error": None,
                }

        except (
            OSError,
            ValueError,
            KeyError,
            subprocess.SubprocessError,
            RuntimeError,
        ) as error:
            failure["error"] = str(error)
            return failure

    def run_pvt_corner(
        self,
        actions: np.ndarray,
        process: str,
        vdd: float,
        temperature_c: float,
    ) -> dict[str, Any]:
        """Run one complete PVT corner using .op followed by .ac."""

        result = {
            "dc_valid": False,
            "dc_gain": float("nan"),
            "nyquist_gain": float("nan"),
            "peaking_boost": float("nan"),
            "power": float("nan"),
            "error": None,
        }

        try:
            parameters = self.map_actions(actions)

            netlist = self._inject_parameters(
                parameters,
                vdd=vdd,
                temperature_c=temperature_c,
                pvt_process=process,
            )

            with tempfile.TemporaryDirectory(
                prefix="autoanalog-pvt-"
            ) as directory:
                base = Path(directory) / "pvt"

                operating_point = self._run_ngspice(
                    self._with_commands(
                        netlist,
                        self._op_commands(),
                    ),
                    base.with_name("op"),
                )

                output_voltage = self._parse_scalar(
                    operating_point,
                    "v(outp)",
                )

                if (
                    not np.isfinite(output_voltage)
                    or not 0.1 <= output_voltage <= vdd
                ):
                    result["error"] = (
                        f"invalid operating point: v(outp)={output_voltage}"
                    )
                    return result

                ac_output = self._run_ngspice(
                    self._with_commands(
                        netlist,
                        self._ac_commands(),
                    ),
                    base.with_name("ac"),
                )

                frequencies, gains = self._parse_ac_data(ac_output)

                dc_gain = self._nearest_value(
                    frequencies,
                    gains,
                    10e6,
                )

                nyquist_gain = self._nearest_value(
                    frequencies,
                    gains,
                    2.5e9,
                )

                peaking_boost = nyquist_gain - dc_gain

                if not (
                    np.isfinite(dc_gain)
                    and np.isfinite(nyquist_gain)
                    and np.isfinite(peaking_boost)
                ):
                    result["error"] = "non-finite AC result"
                    return result

                result.update(
                    {
                        "dc_valid": True,
                        "dc_gain": dc_gain,
                        "nyquist_gain": nyquist_gain,
                        "peaking_boost": peaking_boost,
                        "power": vdd * parameters["I_bias"],
                    }
                )

                return result

        except (
            OSError,
            ValueError,
            KeyError,
            subprocess.SubprocessError,
            RuntimeError,
        ) as error:
            result["error"] = str(error)
            return result

    def run_pvt(
        self,
        actions: np.ndarray,
        corners: tuple[Any, ...],
    ) -> list[dict[str, Any]]:
        """Run all supplied PVT corners and return one row per corner."""

        results: list[dict[str, Any]] = []

        for corner in corners:
            result = self.run_pvt_corner(
                actions=actions,
                process=corner.process,
                vdd=corner.vdd,
                temperature_c=corner.temperature_c,
            )

            peaking = result["peaking_boost"]

            pvt_pass = bool(
                result["dc_valid"]
                and np.isfinite(peaking)
                and 3.0 <= peaking <= 12.0
            )

            results.append(
                {
                    "name": corner.name,
                    "process": corner.process,
                    "vdd": corner.vdd,
                    "temperature_c": corner.temperature_c,
                    "dc_valid": result["dc_valid"],
                    "dc_gain": result["dc_gain"],
                    "nyquist_gain": result["nyquist_gain"],
                    "peaking_boost": peaking,
                    "power": result["power"],
                    "pvt_pass": pvt_pass,
                    "status": "pass" if pvt_pass else "fail",
                    "error": result.get("error"),
                }
            )

        return results

    def run_transient(self, actions: np.ndarray) -> dict[str, Any]:
        """Run the 5 Gbps PRBS7 transient through the lossy channel and measure the eye."""
        failure = {
            "tran_valid": False,
            "time_s": np.array([]),
            "output_v": np.array([]),
            "eye_height_v": float("nan"),
            "eye_width_ui": float("nan"),
            "eye_center_ui": float("nan"),
            "eye_height_pass": False,
            "eye_width_pass": False,
            "error": None,
        }

        try:
            parameters = self.map_actions(actions)
            netlist = self._inject_parameters(
                parameters,
                transient=True,
            )

            with tempfile.TemporaryDirectory(
                prefix="autoanalog-tran-"
            ) as directory:
                output = self._run_ngspice(
                    self._with_commands(
                        netlist,
                        self._tran_commands(),
                    ),
                    Path(directory) / "tran",
                )

            time_s, output_v = self._parse_transient(output)

            eye = self._eye_metrics(
                time_s,
                output_v,
                self._prbs_bits(),
                self.UNIT_INTERVAL_S,
                periods=self.PRBS_PERIODS,
            )

            eye_height_pass = eye["eye_height_v"] >= self.eye_height_min_v
            eye_width_pass = eye["eye_width_ui"] >= self.eye_width_min_ui

            # Behavioural one-tap DFE on the UI-centre samples of the CTLE
            # output; reports the tap that opens the eye the most.
            ui = self.UNIT_INTERVAL_S
            symbol_centers = np.arange(time_s.min() + 0.5 * ui, time_s.max(), ui)
            center_indices = np.searchsorted(time_s, symbol_centers).clip(max=time_s.size - 1)
            center_values = output_v[center_indices]
            dfe = optimize_one_tap(center_values - float(np.median(center_values)))

            return {
                "tran_valid": eye_height_pass and eye_width_pass,
                "time_s": time_s,
                "output_v": output_v,
                **eye,
                "eye_height_pass": eye_height_pass,
                "eye_width_pass": eye_width_pass,
                "dfe_tap": dfe["tap"],
                "dfe_eye_height_v": dfe["eye_height_v"],
                "dfe_output_v": dfe["corrected"],
                "dfe_time_s": symbol_centers,
                "error": None,
            }

        except (
            OSError,
            ValueError,
            KeyError,
            subprocess.SubprocessError,
            RuntimeError,
        ) as error:
            failure["error"] = str(error)
            return failure

    def run_linearity(self, actions: np.ndarray) -> dict[str, Any]:
        """Measure third-harmonic distortion for a 100 MHz differential input."""
        result = {"hd3_db": float("nan"), "linearity_valid": False, "error": None}
        try:
            parameters = self.map_actions(actions)
            netlist = self._inject_parameters(parameters, stimulus="hd3")
            with tempfile.TemporaryDirectory(prefix="autoanalog-hd3-") as directory:
                output = self._run_ngspice(self._with_commands(netlist, self._hd3_commands()), Path(directory) / "hd3")
            time_s, output_v = self._parse_transient(output)
            keep = time_s >= time_s.min() + 0.5 * (time_s.max() - time_s.min())
            sample_time = np.linspace(time_s[keep].min(), time_s[keep].max(), 5001)
            values = np.interp(sample_time, time_s[keep], output_v[keep])
            values -= np.mean(values)
            spectrum = np.abs(np.fft.rfft(values))
            frequencies = np.fft.rfftfreq(values.size, sample_time[1] - sample_time[0])
            fundamental = spectrum[np.argmin(abs(frequencies - 100e6))]
            third = spectrum[np.argmin(abs(frequencies - 300e6))]
            hd3 = float(20.0 * np.log10(max(third, 1e-30) / max(fundamental, 1e-30)))
            return {"hd3_db": hd3, "linearity_valid": bool(np.isfinite(hd3)), "error": None}
        except (OSError, ValueError, KeyError, subprocess.SubprocessError, RuntimeError) as error:
            result["error"] = str(error)
            return result

    def run_noise(self, actions: np.ndarray) -> dict[str, Any]:
        """Integrate output-noise density over the 10 MHz to 5 GHz band."""
        result = {"noise_vrms": float("nan"), "noise_valid": False, "error": None}
        try:
            parameters = self.map_actions(actions)
            netlist = self._inject_parameters(parameters)
            with tempfile.TemporaryDirectory(prefix="autoanalog-noise-") as directory:
                output = self._run_ngspice(self._with_commands(netlist, self._noise_commands()), Path(directory) / "noise")
            rows = []
            for line in output.splitlines():
                fields = line.split()
                if len(fields) >= 3:
                    try:
                        rows.append((float(fields[1]), float(fields[2])))
                    except ValueError:
                        continue
            data = np.asarray(rows, dtype=float)
            if data.size == 0 or np.any(~np.isfinite(data)) or np.any(np.diff(data[:, 0]) <= 0):
                raise ValueError("missing or invalid noise data")
            # ngspice reports spectral density in V/sqrt(Hz); integrate its
            # square over frequency to obtain RMS input-referred noise.
            density = np.maximum(data[:, 1], 0.0)
            noise_vrms = float(np.sqrt(np.trapezoid(density**2, data[:, 0])))
            return {"noise_vrms": noise_vrms, "noise_valid": bool(np.isfinite(noise_vrms)), "error": None}
        except (OSError, ValueError, KeyError, subprocess.SubprocessError, RuntimeError) as error:
            result["error"] = str(error)
            return result

    def estimate_area(self, actions: np.ndarray) -> dict[str, Any]:
        """Return a documented first-order active-device area estimate in mm^2."""
        parameters = self.map_actions(actions)
        width_um = parameters["W_in"] * 1e6
        length_um = 0.13
        area_mm2 = 3.0 * width_um * length_um * 1e-6
        return {"area_mm2": area_mm2, "area_valid": bool(area_mm2 < 0.05), "area_method": "active_mos_geometry_estimate"}

    @staticmethod
    def _eye_metrics(
        time_s: np.ndarray,
        output_v: np.ndarray,
        bits: np.ndarray,
        ui: float,
        periods: int = 2,
        phase_bins: int = 50,
    ) -> dict[str, float]:
        """Measure the eye of ``output_v`` against the transmitted ``bits``.

        The waveform is aligned to the bit pattern by correlation (this also
        resolves the CTLE's inversion), only the final PRBS period is measured
        so start-up transients are excluded, and the eye is folded into one UI:
        at each phase the opening is ``min(ones) - max(zeros)``. Eye height is
        the largest opening, eye width is the fraction of the UI where the
        opening is positive, and eye centre is the phase of the largest opening.
        """
        bits = np.asarray(bits, dtype=int)
        n_bits = bits.size
        pattern = np.tile(bits, periods)
        n_total = pattern.size
        time_s = np.asarray(time_s, dtype=float)
        output_v = np.asarray(output_v, dtype=float)
        closed = {"eye_height_v": 0.0, "eye_width_ui": 0.0, "eye_center_ui": float("nan")}

        if time_s.size < 2 or n_bits < 2:
            return closed

        centred = output_v - float(np.mean(output_v))
        measure_from = (n_total - n_bits) * ui
        best_score = 0.0
        best_delay = 0.0
        best_polarity = 1.0

        # Search the latency over two UI in fine steps; the CTLE inverts, so
        # the sign of the correlation carries the polarity.
        for delay in np.arange(0.0, 2.0 * ui, ui / 40.0):
            shifted = time_s - delay
            index = np.floor(shifted / ui).astype(int)
            valid = (shifted >= measure_from) & (index < n_total)
            if np.count_nonzero(valid) < n_bits:
                continue
            ideal = np.where(pattern[index[valid]] == 1, 1.0, -1.0)
            score = float(np.dot(centred[valid], ideal))
            if abs(score) > abs(best_score):
                best_score = score
                best_delay = float(delay)
                best_polarity = 1.0 if score >= 0.0 else -1.0

        if best_score == 0.0:
            return closed

        shifted = time_s - best_delay
        index = np.floor(shifted / ui).astype(int)
        valid = (shifted >= measure_from) & (index < n_total)
        signal = best_polarity * centred[valid]
        is_one = pattern[index[valid]] == 1
        phase = np.mod(shifted[valid], ui) / ui
        bin_index = np.minimum((phase * phase_bins).astype(int), phase_bins - 1)

        opening = np.full(phase_bins, -np.inf)
        for b in range(phase_bins):
            in_bin = bin_index == b
            ones = signal[in_bin & is_one]
            zeros = signal[in_bin & ~is_one]
            if ones.size and zeros.size:
                opening[b] = float(np.min(ones) - np.max(zeros))

        best_bin = int(np.argmax(opening))
        eye_height = max(0.0, float(opening[best_bin]))
        open_bins = int(np.count_nonzero(opening > 0.0))

        return {
            "eye_height_v": eye_height,
            "eye_width_ui": open_bins / phase_bins,
            "eye_center_ui": (best_bin + 0.5) / phase_bins if eye_height > 0.0 else float("nan"),
        }

    def _inject_parameters(
        self,
        parameters: dict[str, float],
        transient: bool = False,
        vdd: float = 1.2,
        temperature_c: float | None = None,
        pvt_process: str | None = None,
        stimulus: str = "ac",
    ) -> str:
        process = pvt_process if pvt_process is not None else "TT"

        if process not in self.GENERIC_PROCESS_MODELS:
            raise ValueError(f"unsupported process corner: {process}")

        rendered = self.template.replace(
            "{VDD}",
            self._spice_value(vdd),
        )

        if self.osdi_model_paths:
            rendered = rendered.replace(
                "M1 outP inP sourceP 0 ctle_nmos W={W_in} L=0.13u",
                "X1 outP inP sourceP 0 sg13_lv_nmos W={W_in} L=0.13u",
            ).replace(
                "M2 outN inN sourceN 0 ctle_nmos W={W_in} L=0.13u",
                "X2 outN inN sourceN 0 sg13_lv_nmos W={W_in} L=0.13u",
            )
            rendered = re.sub(
                r"^\.model ctle_nmos .*\n",
                "",
                rendered,
                flags=re.MULTILINE,
            )
        else:
            vto, kp = self.GENERIC_PROCESS_MODELS[process]
            rendered = rendered.replace(
                "{VTO}",
                self._spice_value(vto),
            ).replace(
                "{KP}",
                self._spice_value(kp),
            )

        if stimulus == "hd3":
            rendered = rendered.replace("{VINP}", "DC 0.6 SIN(0.0 0.05 100Meg)")
            rendered = rendered.replace("{VINN}", "DC 0.6 SIN(0.0 -0.05 100Meg)")
        elif transient:
            rendered = rendered.replace(
                "{VINP}",
                self._prbs_source(False),
            )
            rendered = rendered.replace(
                "{VINN}",
                self._prbs_source(True),
            )
        else:
            rendered = rendered.replace(
                "{VINP}",
                "DC 0.6 AC 1",
            )
            rendered = rendered.replace(
                "{VINN}",
                "DC 0.6 AC -1",
            )

        rendered = rendered.replace(
            "{CHANNEL}",
            self._channel_block(lossy=transient),
        )

        if self.pdk_model_path:
            model_includes = [
                f".include {self.pdk_model_path}"
            ]

            if self.pdk_corner_path:
                corner = (
                    self.PDK_PROCESS_SECTIONS[process]
                    if pvt_process is not None
                    else self.pdk_corner
                )
                if pvt_process in {"TT", "SS", "FF", "SF", "FS"}:
                    corner = f"mos_{pvt_process.lower()}"

                model_includes.insert(
                    0,
                    f".lib {self.pdk_corner_path} {corner}",
                )

            rendered = rendered.replace(
                "\n.end",
                "\n"
                + "\n".join(model_includes)
                + "\n.end",
                1,
            )

        if temperature_c is not None:
            rendered = rendered.replace(
                "\n.end",
                f"\n.temp {temperature_c}\n.end",
                1,
            )

        for name, value in parameters.items():
            rendered = rendered.replace(
                "{" + name + "}",
                self._spice_value(value),
            )

        return rendered

    @staticmethod
    def _with_commands(
        netlist: str,
        commands: str,
    ) -> str:
        end_marker = "\n.end"

        if end_marker not in netlist:
            raise ValueError(
                "SPICE template is missing .end"
            )

        return netlist.replace(
            end_marker,
            commands + end_marker,
            1,
        )

    @staticmethod
    def _spice_value(value: float) -> str:
        return f"{value:.12g}"

    def _run_ngspice(
        self,
        netlist: str,
        stem: Path,
    ) -> str:
        input_path = stem.with_suffix(".sp")
        output_path = stem.with_suffix(".out")

        input_path.write_text(
            netlist,
            encoding="utf-8",
        )

        # ngspice ignores OMP_NUM_THREADS and spins its own OpenMP pool per
        # process; with several evaluators running in parallel (training) the
        # spin-waiting threads oversubscribe the cores and PSP103 transients
        # go from ~1.5 s to ~90 s each. One thread per process is fastest.
        spiceinit = [f"set num_threads={self.NGSPICE_THREADS}"]
        spiceinit += [f"osdi '{path}'" for path in self.osdi_model_paths]
        (stem.parent / ".spiceinit").write_text("\n".join(spiceinit) + "\n", encoding="utf-8")

        completed = subprocess.run(
            [
                self.ngspice_binary,
                "-b",
                "-o",
                str(output_path),
                str(input_path),
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=stem.parent,
        )

        output = (
            output_path.read_text(
                encoding="utf-8"
            )
            if output_path.exists()
            else completed.stdout + completed.stderr
        )

        if (
            completed.returncode != 0
            or re.search(
                r"matrix is singular|fatal error|error",
                output,
                re.IGNORECASE,
            )
        ):
            raise RuntimeError(output)

        return output

    @staticmethod
    def _op_commands() -> str:
        return (
            "\n.control\n"
            "op\n"
            "print v(outP)\n"
            ".endc\n"
        )

    @staticmethod
    def _ac_commands() -> str:
        return (
            "\n.control\n"
            "ac dec 100 10Meg 10Gig\n"
            "print frequency v(outP,outN)\n"
            ".endc\n"
        )

    @staticmethod
    def _tran_commands() -> str:
        return (
            "\n.control\n"
            f"tran {SpiceEvaluator.TRANSIENT_STEP_S:.12g} "
            f"{SpiceEvaluator.PRBS_PERIODS * 127 * SpiceEvaluator.UNIT_INTERVAL_S:.12g}\n"
            "print time v(outP) v(outN)\n"
            ".endc\n"
        )

    @classmethod
    def _channel_block(cls, lossy: bool) -> str:
        """Return the netlist lines connecting txP/txN to inP/inN."""
        if not lossy:
            return "RchP txP inP 1m\nRchN txN inN 1m"

        lines = []
        for side in ("P", "N"):
            previous = f"tx{side}"
            for stage, ohms in enumerate(cls.CHANNEL_SECTION_OHMS, start=1):
                node = f"in{side}" if stage == len(cls.CHANNEL_SECTION_OHMS) else f"ch{side}{stage}"
                farads = 1.0 / (2.0 * np.pi * cls.CHANNEL_POLE_HZ * ohms)
                lines.append(f"Rch{side}{stage} {previous} {node} {ohms:.12g}")
                lines.append(f"Cch{side}{stage} {node} 0 {farads:.12g}")
                previous = node
        return "\n".join(lines)

    @classmethod
    def channel_loss_db(cls, frequency_hz: float) -> float:
        """Nominal insertion loss of the transient channel (ignores section loading)."""
        ratio = frequency_hz / cls.CHANNEL_POLE_HZ
        return float(-20.0 * len(cls.CHANNEL_SECTION_OHMS) * np.log10(np.sqrt(1.0 + ratio * ratio)))

    @staticmethod
    def _hd3_commands() -> str:
        return "\n.control\ntran 10p 50n\nprint time v(outP) v(outN)\n.endc\n"

    @staticmethod
    def _noise_commands() -> str:
        return "\n.control\nnoise v(outP,outN) Vinp dec 50 10Meg 5Gig\nsetplot noise1\nprint all\n.endc\n"

    @staticmethod
    def _prbs_bits(length: int = 127) -> np.ndarray:
        """Return the PRBS7 (x^7 + x^6 + 1) sequence from the all-ones seed."""
        register = 0x7F
        bits = np.zeros(length, dtype=int)
        for index in range(length):
            bits[index] = register & 1
            feedback = ((register >> 6) ^ (register >> 5)) & 1
            register = ((register << 1) | feedback) & 0x7F
        return bits

    @classmethod
    def _prbs_source(cls, invert: bool) -> str:
        """Return a deterministic PRBS7 PWL source at 5 Gbps, repeated PRBS_PERIODS times."""
        bit_period = cls.UNIT_INTERVAL_S
        points: list[str] = []

        for index, bit in enumerate(np.tile(cls._prbs_bits(), cls.PRBS_PERIODS)):
            level = 0.5 + (0.2 if bit else 0.0)

            if invert:
                level = 1.2 - level

            start = index * bit_period
            end = (index + 1) * bit_period

            points.append(f"{start:.12g} {level:.12g}")
            points.append(f"{end:.12g} {level:.12g}")

        return "PWL(" + " ".join(points) + ")"

    @staticmethod
    def _parse_scalar(
        output: str,
        node: str,
    ) -> float:
        match = re.search(
            rf"{re.escape(node)}\s*=\s*([-+0-9.eE]+)",
            output,
            re.IGNORECASE,
        )

        if not match:
            match = re.search(
                rf"{re.escape(node)}\s+([-+0-9.eE]+)",
                output,
                re.IGNORECASE,
            )

        if not match:
            raise ValueError(
                f"missing operating-point value for {node}"
            )

        return float(match.group(1))

    @staticmethod
    def _parse_ac_data(
        output: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        rows: list[tuple[float, float, float]] = []

        for line in output.splitlines():
            fields = line.split()

            if len(fields) < 4:
                continue

            try:
                frequency = float(fields[1])
                real = float(
                    fields[2].strip(",")
                )
                imaginary = float(
                    fields[3].strip(",")
                )
            except ValueError:
                continue

            rows.append(
                (frequency, real, imaginary)
            )

        if not rows:
            raise ValueError(
                "missing AC data"
            )

        data = np.asarray(
            rows,
            dtype=float,
        )

        if (
            np.any(~np.isfinite(data))
            or np.any(data[:, 0] <= 0)
            or np.any(np.diff(data[:, 0]) <= 0)
        ):
            raise ValueError(
                "invalid AC frequency"
            )

        return (
            data[:, 0],
            20.0
            * np.log10(
                np.abs(
                    data[:, 1]
                    + 1j * data[:, 2]
                )
            ),
        )

    @staticmethod
    def _nearest_value(
        frequencies: np.ndarray,
        values: np.ndarray,
        target: float,
    ) -> float:
        return float(
            values[
                np.argmin(
                    np.abs(
                        frequencies - target
                    )
                )
            ]
        )

    @staticmethod
    def _parse_transient(
        output: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        rows: list[tuple[float, float, float]] = []

        for line in output.splitlines():
            fields = line.split()

            if len(fields) < 4:
                continue

            try:
                rows.append(
                    (
                        float(fields[1]),
                        float(fields[2]),
                        float(fields[3]),
                    )
                )
            except ValueError:
                continue

        if not rows:
            raise ValueError(
                "missing transient data"
            )

        data = np.asarray(
            rows,
            dtype=float,
        )

        if np.any(~np.isfinite(data)) or np.any(np.diff(data[:, 0]) < 0):
            raise ValueError(
                "invalid transient time"
            )

        # ngspice repeats a timepoint at PWL breakpoints; keep the last value.
        keep = np.append(np.diff(data[:, 0]) > 0, True)
        data = data[keep]

        return (
            data[:, 0],
            data[:, 1] - data[:, 2],
        )