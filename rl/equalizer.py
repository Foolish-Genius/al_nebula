"""Whole CTLE plus one-tap DFE evaluation boundary."""

from __future__ import annotations

from typing import Any

import numpy as np

from .dfe import apply_one_tap_dfe, eye_height


class EqualizerEvaluator:
    """Expose one six-value action for CTLE sizing plus behavioral DFE tap."""

    def __init__(self, spice_evaluator: Any) -> None:
        self.spice = spice_evaluator
        self.tap_bounds = (-0.5, 0.5)

    def run(self, action: np.ndarray) -> dict[str, Any]:
        values = np.asarray(action, dtype=float)
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            raise ValueError("equalizer action must contain six finite values")
        if np.any(values < -1.0) or np.any(values > 1.0):
            raise ValueError("equalizer action must be bounded by [-1, 1]")
        ctle_action = values[:5]
        tap = self.tap_bounds[0] + (values[5] + 1.0) * (self.tap_bounds[1] - self.tap_bounds[0]) / 2.0
        ac = self.spice.run_simulation(ctle_action)
        transient = self.spice.run_transient(ctle_action)
        result = {**ac, **{key: value for key, value in transient.items() if key not in ("time_s", "output_v", "dfe_output_v", "dfe_time_s")}}
        result["dfe_tap"] = tap
        if not transient.get("tran_valid", False):
            result["equalizer_valid"] = False
            return result
        time_s = np.asarray(transient["time_s"], dtype=float)
        output_v = np.asarray(transient["output_v"], dtype=float)
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
