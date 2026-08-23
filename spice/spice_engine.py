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

    def __init__(self, template_path: str | Path | None = None, ngspice_binary: str = "ngspice") -> None:
        project_root = Path(__file__).resolve().parents[1]
        self.template_path = Path(template_path or project_root / "netlists" / "ctle_template.sp")
        self.template = self.template_path.read_text(encoding="utf-8")
        self.ngspice_binary = ngspice_binary
        self.bounds = {
            "W_in": (0.5e-6, 50.0e-6),
            "R_load": (10.0, 1000.0),
            "I_bias": (100.0e-6, 10.0e-3),
            "R_s": (10.0, 500.0),
            "C_s": (1.0e-15, 1.0e-12),
        }

    def map_actions(self, actions: np.ndarray) -> dict[str, float]:
        """Linearly map five normalized actions from [-1, 1] to SI values."""
        values = np.asarray(actions, dtype=float)
        if values.shape != (5,):
            raise ValueError("actions must contain exactly five values")
        if not np.all(np.isfinite(values)) or np.any(values < -1.0) or np.any(values > 1.0):
            raise ValueError("actions must be finite and bounded by [-1.0, 1.0]")
        return {
            name: lower + (float(value) + 1.0) * (upper - lower) / 2.0
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
        }
        try:
            parameters = self.map_actions(actions)
            netlist = self._inject_parameters(parameters)
            with tempfile.TemporaryDirectory(prefix="autoanalog-") as directory:
                base = Path(directory) / "ctle"
                operating_point = self._run_ngspice(self._with_commands(netlist, self._op_commands()), base.with_name("op"))
                output_voltage = self._parse_scalar(operating_point, "v(outp)")
                if not np.isfinite(output_voltage) or not 0.1 <= output_voltage <= 1.1:
                    return failure

                ac_output = self._run_ngspice(self._with_commands(netlist, self._ac_commands()), base.with_name("ac"))
                dc_gain = self._parse_ac_gain(ac_output, 10e6)
                nyquist_gain = self._parse_ac_gain(ac_output, 2.5e9)
                if not np.isfinite(dc_gain) or not np.isfinite(nyquist_gain):
                    return failure
                return {
                    "dc_valid": True,
                    "dc_gain": dc_gain,
                    "nyquist_gain": nyquist_gain,
                    "peaking_boost": nyquist_gain - dc_gain,
                    "power": 1.2 * parameters["I_bias"],
                }
        except (OSError, ValueError, KeyError, subprocess.SubprocessError, RuntimeError):
            return failure

    def _inject_parameters(self, parameters: dict[str, float]) -> str:
        rendered = self.template.replace("{VDD}", "1.2")
        for name, value in parameters.items():
            rendered = rendered.replace("{" + name + "}", self._spice_value(value))
        rendered = rendered.replace("{I_bias/2}", self._spice_value(parameters["I_bias"] / 2.0))
        return rendered

    @staticmethod
    def _with_commands(netlist: str, commands: str) -> str:
        end_marker = "\n.end"
        if end_marker not in netlist:
            raise ValueError("SPICE template is missing .end")
        return netlist.replace(end_marker, commands + end_marker, 1)

    @staticmethod
    def _spice_value(value: float) -> str:
        return f"{value:.12g}"

    def _run_ngspice(self, netlist: str, stem: Path) -> str:
        input_path = stem.with_suffix(".sp")
        output_path = stem.with_suffix(".out")
        input_path.write_text(netlist, encoding="utf-8")
        completed = subprocess.run(
            [self.ngspice_binary, "-b", "-o", str(output_path), str(input_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        output = output_path.read_text(encoding="utf-8") if output_path.exists() else completed.stdout + completed.stderr
        if completed.returncode != 0 or re.search(r"matrix is singular|fatal error|error", output, re.IGNORECASE):
            raise RuntimeError(output)
        return output

    @staticmethod
    def _op_commands() -> str:
        return "\n.control\nop\nprint v(outP)\n.endc\n"

    @staticmethod
    def _ac_commands() -> str:
        return "\n.control\nac dec 100 10Meg 10Gig\nprint frequency v(outP)\n.endc\n"

    @staticmethod
    def _parse_scalar(output: str, node: str) -> float:
        match = re.search(rf"{re.escape(node)}\s*=\s*([-+0-9.eE]+)", output, re.IGNORECASE)
        if not match:
            match = re.search(rf"{re.escape(node)}\s+([-+0-9.eE]+)", output, re.IGNORECASE)
        if not match:
            raise ValueError(f"missing operating-point value for {node}")
        return float(match.group(1))

    @staticmethod
    def _parse_ac_gain(output: str, target_frequency: float) -> float:
        rows: list[tuple[float, complex]] = []
        for line in output.splitlines():
            fields = line.split()
            if len(fields) < 3:
                continue
            try:
                frequency = float(fields[0])
                real = float(fields[1].strip(","))
                imaginary = float(fields[2].strip(","))
            except ValueError:
                continue
            rows.append((frequency, complex(real, imaginary)))
        if not rows:
            raise ValueError("missing AC data")
        frequency, value = min(rows, key=lambda row: abs(row[0] - target_frequency))
        if frequency <= 0:
            raise ValueError("invalid AC frequency")
        return float(20.0 * np.log10(abs(value)))
