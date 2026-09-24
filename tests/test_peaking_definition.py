"""Tests for the peaking reporting and the configurable PVT window (mentor points 3 and 5)."""

import numpy as np
import pytest

from rl.pvt import PvtCorner
from spice.spice_engine import SpiceEvaluator


def test_peak_gain_finds_the_maximum_and_its_frequency():
    frequencies = np.array([1e7, 1e8, 1e9, 4e9, 1e10])
    gains = np.array([10.0, 10.5, 14.0, 16.0, 12.0])
    peak_gain, peak_frequency = SpiceEvaluator._peak_gain(frequencies, gains)
    assert peak_gain == 16.0
    assert peak_frequency == 4e9


def test_peak_gain_ignores_non_finite_points():
    frequencies = np.array([1e7, 1e8, 1e9])
    gains = np.array([10.0, np.nan, 12.0])
    peak_gain, peak_frequency = SpiceEvaluator._peak_gain(frequencies, gains)
    assert peak_gain == 12.0 and peak_frequency == 1e9


def test_peak_gain_is_nan_when_nothing_is_measurable():
    peak_gain, peak_frequency = SpiceEvaluator._peak_gain(np.array([]), np.array([]))
    assert not np.isfinite(peak_gain) and not np.isfinite(peak_frequency)
    peak_gain, _ = SpiceEvaluator._peak_gain(np.array([1e7]), np.array([np.nan]))
    assert not np.isfinite(peak_gain)


def test_reported_peaking_definitions_differ_when_the_peak_is_above_nyquist():
    """The mentor's definition (max - DC) exceeds ours (Nyquist - DC) whenever the peak sits higher."""
    frequencies = np.array([1e7, 2.5e9, 4.2e9, 1e10])
    gains = np.array([11.13, 15.61, 16.03, 14.53])
    at_nyquist = float(np.interp(2.5e9, frequencies, gains))
    peak_gain, peak_frequency = SpiceEvaluator._peak_gain(frequencies, gains)
    ours = at_nyquist - gains[0]
    mentor = peak_gain - gains[0]
    assert peak_frequency > 2.5e9
    assert mentor > ours
    assert mentor - ours == pytest.approx(0.42, abs=0.01)


class _StubEvaluator:
    """Drives the real run_pvt loop with a scripted per-corner peaking value."""

    def __init__(self, peaking_by_corner):
        self._peaking = peaking_by_corner

    run_pvt = SpiceEvaluator.run_pvt

    def run_pvt_corner(self, actions, process, vdd, temperature_c):
        peaking = self._peaking[process]
        return {
            "dc_valid": True, "dc_gain": 10.0, "nyquist_gain": 10.0 + peaking,
            "peaking_boost": peaking, "peak_gain": 10.0 + peaking + 0.4,
            "peak_frequency_hz": 4.2e9, "peaking_max": peaking + 0.4,
            "power": 1.2e-3, "error": None,
        }


def _corners():
    return tuple(PvtCorner(process, 1.0, 62.5) for process in ("TT", "SS", "FF"))


def test_run_pvt_defaults_reproduce_the_original_three_to_twelve_window():
    evaluator = _StubEvaluator({"TT": 4.0, "SS": 2.5, "FF": 12.5})
    rows = evaluator.run_pvt(np.zeros(5), _corners())
    assert [row["pvt_pass"] for row in rows] == [True, False, False]


def test_run_pvt_honours_a_retuned_peaking_window():
    """A design at 4 dB must fail a 5 dB floor: the old hardcoded window passed it."""
    evaluator = _StubEvaluator({"TT": 4.0, "SS": 6.0, "FF": 12.5})
    rows = evaluator.run_pvt(np.zeros(5), _corners(), peaking_min_db=5.0, peaking_max_db=12.0)
    assert [row["pvt_pass"] for row in rows] == [False, True, False]

    widened = evaluator.run_pvt(np.zeros(5), _corners(), peaking_min_db=3.0, peaking_max_db=13.0)
    assert [row["pvt_pass"] for row in widened] == [True, True, True]


def test_run_pvt_rows_carry_both_peaking_numbers():
    rows = _StubEvaluator({"TT": 4.0, "SS": 4.0, "FF": 4.0}).run_pvt(np.zeros(5), _corners())
    assert rows[0]["peaking_boost"] == 4.0
    assert rows[0]["peaking_max"] == pytest.approx(4.4)
    assert rows[0]["peak_frequency_hz"] == 4.2e9
