"""Deterministic one-tap DFE post-processing for sampled CTLE waveforms."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def apply_one_tap_dfe(samples: Sequence[float], tap: float, threshold: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Subtract tap times the previous hard decision from a sampled waveform."""
    values = np.asarray(samples, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("samples must be a non-empty one-dimensional sequence")
    if not np.isfinite(tap) or not np.isfinite(threshold):
        raise ValueError("tap and threshold must be finite")
    decisions = np.empty(values.size, dtype=float)
    corrected = np.empty_like(values)
    previous = -1.0 if values[0] < threshold else 1.0
    for index, value in enumerate(values):
        corrected[index] = value - tap * previous
        decisions[index] = 1.0 if corrected[index] >= threshold else -1.0
        previous = decisions[index]
    return corrected, decisions


def eye_height(samples: Sequence[float], decisions: Sequence[float]) -> float:
    """Estimate vertical eye opening from separated hard-decision populations."""
    values = np.asarray(samples, dtype=float)
    labels = np.asarray(decisions, dtype=float)
    if values.shape != labels.shape or values.size == 0:
        raise ValueError("samples and decisions must have equal non-zero shapes")
    high = values[labels > 0]
    low = values[labels < 0]
    if high.size == 0 or low.size == 0:
        return 0.0
    return float(np.percentile(high, 5) - np.percentile(low, 95))


def optimize_one_tap(samples: Sequence[float], tap_bounds: tuple[float, float] = (-0.5, 0.5), steps: int = 41) -> dict[str, object]:
    """Select the tap with the greatest robust vertical eye opening."""
    if steps < 2 or tap_bounds[0] >= tap_bounds[1]:
        raise ValueError("invalid tap search bounds")
    values = np.asarray(samples, dtype=float)
    threshold = float(np.median(values))
    best = {"tap": 0.0, "eye_height_v": 0.0, "corrected": values, "decisions": np.sign(values - threshold)}
    for tap in np.linspace(tap_bounds[0], tap_bounds[1], steps):
        corrected, decisions = apply_one_tap_dfe(values - threshold, float(tap))
        height = eye_height(corrected, decisions)
        if height > float(best["eye_height_v"]):
            best = {"tap": float(tap), "eye_height_v": height, "corrected": corrected + threshold, "decisions": decisions}
    return best
