"""Whole CTLE plus one-tap DFE evaluation boundary."""

from __future__ import annotations

from typing import Any

import numpy as np

from .dfe import apply_one_tap_dfe, eye_height


class EqualizerEvaluator:
    """Expose one six-value action for CTLE sizing plus behavioral DFE tap."""

    PARAMETER_NAMES = ("W_in", "R_load", "I_bias", "R_s", "C_s", "dfe_tap")

    def __init__(self, spice_evaluator: Any) -> None:
        self.spice = spice_evaluator
        self.tap_bounds = (-0.5, 0.5)

    @property
    def model_source(self) -> str:
        return getattr(self.spice, "model_source", "unknown")

    def map_tap(self, value: float) -> float:
        return self.tap_bounds[0] + (float(value) + 1.0) * (self.tap_bounds[1] - self.tap_bounds[0]) / 2.0

    def map_actions(self, action: np.ndarray) -> dict[str, float]:
        """CTLE SI values plus the DFE tap for reports."""
        values = np.asarray(action, dtype=float)
        return {**self.spice.map_actions(values[:5]), "dfe_tap": self.map_tap(values[5])}

    def run_linearity(self, action: np.ndarray) -> dict[str, Any]:
        return self.spice.run_linearity(np.asarray(action, dtype=float)[:5])

    def run(self, action: np.ndarray) -> dict[str, Any]:
        values = np.asarray(action, dtype=float)
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            raise ValueError("equalizer action must contain six finite values")
        if np.any(values < -1.0) or np.any(values > 1.0):
            raise ValueError("equalizer action must be bounded by [-1, 1]")
        ctle_action = values[:5]
        tap = self.map_tap(values[5])
        ac = self.spice.run_simulation(ctle_action)
        if not ac.get("dc_valid", False):
            return {**ac, "dfe_tap": tap, "equalizer_valid": False}
        transient = self.spice.run_transient(ctle_action)
        result = {**ac, **{key: value for key, value in transient.items() if key not in ("time_s", "output_v", "dfe_output_v", "dfe_time_s")}}
        result["dfe_tap"] = tap
        time_s = np.asarray(transient.get("time_s", ()), dtype=float)
        output_v = np.asarray(transient.get("output_v", ()), dtype=float)
        # Apply the agent's tap whenever the CTLE produced a waveform, whether or
        # not the raw eye already passes: the DFE is there to open a closed eye.
        if time_s.size < 2:
            result["equalizer_valid"] = False
            return result
        if hasattr(self.spice, "dfe_eye_metrics"):
            # Bit-referenced eye (see rl.dfe.dfe_eye_against_bits).
            dfe = self.spice.dfe_eye_metrics(time_s, output_v, tap=tap, sample_phase=transient.get("eye_center_ui"))
            result.update({
                "equalizer_valid": bool(dfe["dfe_valid"]),
                "equalizer_tap": tap,
                "dfe_eye_height_v": dfe["dfe_eye_height_v"],
                "dfe_bit_errors": dfe["dfe_bit_errors"],
                "dfe_samples": dfe["dfe_output_v"],
                "dfe_sample_time_s": dfe["dfe_time_s"],
            })
            return result
        # Evaluators without PRBS alignment (test doubles): decision-labelled eye.
        ui = 200e-12
        centers = np.arange(time_s.min() + 0.5 * ui, time_s.max(), ui)
        samples = output_v[np.searchsorted(time_s, centers).clip(max=output_v.size - 1)]
        threshold = float(np.median(samples))
        corrected, decisions = apply_one_tap_dfe(samples - threshold, tap)
        result.update({
            "equalizer_valid": True,
            "equalizer_tap": tap,
            "dfe_eye_height_v": eye_height(corrected, decisions),
            "dfe_samples": corrected + threshold,
            "dfe_sample_time_s": centers,
        })
        return result
