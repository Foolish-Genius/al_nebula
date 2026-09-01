"""SPICE execution boundary for the AutoAnalog CTLE evaluator."""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


class SpiceEvaluator:
    """Map RL actions to legal device values and run gated ngspice analyses."""

    PARAMETER_NAMES = ("W_in", "R_load", "I_bias", "R_s", "C_s")

    def __init__(
        self,
        template_path: str | Path | None = None,
        ngspice_binary: str = "ngspice",
        pdk_model_path: str | Path | None = None,
        pdk_corner_path: str | Path | None = None,
        pdk_corner: str = "mos_tt",
        osdi_model_paths: tuple[str | Path, ...] = (),
    ) -> None:
        project_root = Path(__file__).resolve().parents[1]
        self.template_path = Path(
            template_path or project_root / "netlists" / "ctle_template.sp"
        )
        self.template = self.template_path.read_text(encoding="utf-8")
        self.ngspice_binary = ngspice_binary
        self.pdk_model_path = Path(pdk_model_path) if pdk_model_path else None
        self.pdk_corner_path = (
            Path(pdk_corner_path) if pdk_corner_path else None
        )
        self.pdk_corner = pdk_corner
        self.osdi_model_paths = tuple(Path(path) for path in osdi_model_paths)

        self.bounds = {
            "W_in": (0.5e-6, 50.0e-6),
            "R_load": (100.0, 1000.0),
            "I_bias": (100.0e-6, 2.0e-3),
            "R_s": (10.0, 500.0),
            "C_s": (1.0e-15, 1.0e-12),
        }

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
        """Run the 5 Gbps NRZ transient gate and calculate eye-opening metrics."""
        failure = {
            "tran_valid": False,
            "time_s": np.array([]),
            "output_v": np.array([]),
            "eye_height_v": float("nan"),
            "eye_width_ui": float("nan"),
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

            if time_s.size < 2:
                return failure

            # 5 Gbps => 1 UI = 200 ps
            ui = 200e-12

            # Ignore the first UI so that startup/transient effects
            # do not dominate the eye measurement.
            valid = time_s >= time_s.min() + ui

            measurement_v = output_v[valid]

            if measurement_v.size < 2:
                return failure

            # Current vertical eye-opening definition:
            # 95th percentile - 5th percentile
            eye_height = float(
                np.percentile(measurement_v, 95)
                - np.percentile(measurement_v, 5)
            )

            # Determine the voltage range occupied by the waveform.
            low_level = float(
                np.percentile(measurement_v, 5)
            )

            high_level = float(
                np.percentile(measurement_v, 95)
            )

            # Use the midpoint between the two levels as the eye center.
            threshold = (low_level + high_level) / 2.0

            # Determine where the waveform is sufficiently far from
            # the threshold to represent an open eye.
            amplitude = (high_level - low_level) / 2.0
            eye_margin = 0.10 * amplitude

            upper_limit = threshold + eye_margin
            lower_limit = threshold - eye_margin

            # Fold the waveform into one UI.
            phase = np.mod(
                time_s - time_s.min(),
                ui,
            )

            # For each phase position, determine whether the waveform
            # has a valid eye opening.
            phase_bins = np.linspace(
                0.0,
                ui,
                201,
            )

            eye_open = np.zeros(
                len(phase_bins) - 1,
                dtype=bool,
            )

            for index in range(len(phase_bins) - 1):
                mask = (
                    (phase >= phase_bins[index])
                    & (phase < phase_bins[index + 1])
                )

                if not np.any(mask):
                    continue

                values = measurement_v[mask]

                # Eye is considered open if both logical levels
                # remain separated at this phase.
                eye_open[index] = (
                    np.max(values) > upper_limit
                    and np.min(values) < lower_limit
                )

            if np.any(eye_open):
                eye_width_ui = float(
                    np.sum(eye_open) / len(eye_open)
                )
            else:
                eye_width_ui = 0.0

            eye_height_pass = eye_height > 0.10
            eye_width_pass = eye_width_ui > 0.40

            return {
                "tran_valid": eye_height_pass and eye_width_pass,
                "time_s": time_s,
                "output_v": output_v,
                "eye_height_v": eye_height,
                "eye_width_ui": eye_width_ui,
                "eye_height_pass": eye_height_pass,
                "eye_width_pass": eye_width_pass,
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

    def _inject_parameters(
        self,
        parameters: dict[str, float],
        transient: bool = False,
        vdd: float = 1.2,
        temperature_c: float | None = None,
        pvt_process: str | None = None,
    ) -> str:
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
            ).replace(
                ".model ctle_nmos nmos level=1 vto=0.45 kp=200u lambda=0.04 gamma=0.4 phi=0.7\n",
                "",
            )

        if transient:
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

        if self.pdk_model_path:
            model_includes = [
                f".include {self.pdk_model_path}"
            ]

            if self.pdk_corner_path:
                corner = (
                    pvt_process
                    if pvt_process is not None
                    else self.pdk_corner
                )

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

        rendered = rendered.replace(
            "{I_bias/2}",
            self._spice_value(
                parameters["I_bias"] / 2.0
            ),
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

        if self.osdi_model_paths:
            (stem.parent / ".spiceinit").write_text(
                "\n".join(
                    f"osdi '{path}'"
                    for path in self.osdi_model_paths
                )
                + "\n",
                encoding="utf-8",
            )

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
            "tran 1p 25.4n\n"
            "print time v(outP) v(outN)\n"
            ".endc\n"
        )

    @staticmethod
    def _prbs_source(invert: bool) -> str:
        """Return a deterministic PRBS7 PWL source at 5 Gbps."""
        register = 0x7F
        bit_period = 200e-12
        points: list[str] = []

        for index in range(127):
            bit = register & 1
            level = 0.5 + (0.2 if bit else 0.0)

            if invert:
                level = 1.2 - level

            start = index * bit_period
            end = (index + 1) * bit_period

            points.append(
                f"{start:.12g} {level:.12g}"
            )
            points.append(
                f"{end:.12g} {level:.12g}"
            )

            feedback = (
                (register >> 6) ^ (register >> 5)
            ) & 1

            register = (
                (register << 1) | feedback
            ) & 0x7F

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
    def _parse_ac_gain(
        output: str,
        target_frequency: float,
    ) -> float:
        frequencies, gains = SpiceEvaluator._parse_ac_data(
            output
        )

        return SpiceEvaluator._nearest_value(
            frequencies,
            gains,
            target_frequency,
        )

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

        if (
            np.any(~np.isfinite(data))
            or np.any(np.diff(data[:, 0]) <= 0)
        ):
            raise ValueError(
                "invalid transient time"
            )

        return (
            data[:, 0],
            data[:, 1] - data[:, 2],
        )
