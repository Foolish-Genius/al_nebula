import numpy as np
import pytest

from spice.spice_engine import SpiceEvaluator


def test_map_actions_hits_physical_bounds():
    evaluator = SpiceEvaluator()
    mapped = evaluator.map_actions(np.array([-1.0, 0.0, 1.0, -1.0, 1.0]))
    assert mapped["W_in"] == pytest.approx(0.5e-6)
    assert mapped["R_load"] == pytest.approx(550.0)
    assert mapped["I_bias"] == pytest.approx(2e-3)
    assert mapped["R_s"] == pytest.approx(10.0)
    assert mapped["C_s"] == pytest.approx(1e-12)


def test_map_actions_rejects_bad_shape_and_range():
    evaluator = SpiceEvaluator()
    with pytest.raises(ValueError):
        evaluator.map_actions(np.zeros(4))
    with pytest.raises(ValueError):
        evaluator.map_actions(np.array([0.0, 0.0, 0.0, 0.0, 1.1]))


def test_run_simulation_fails_fast_when_ngspice_is_unavailable():
    evaluator = SpiceEvaluator(ngspice_binary="definitely-not-ngspice")
    result = evaluator.run_simulation(np.zeros(5))
    assert result["dc_valid"] is False
