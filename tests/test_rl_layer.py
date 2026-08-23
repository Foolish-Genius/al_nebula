import numpy as np
import pytest

from analysis.reporting import ValidationReporter
from rl.environment import CtleEnvironment
from rl.pvt import PvtCorner, all_pvt_corners
from rl.search import BoundedDesignSearch
from rl.reward import CtleReward
from rl.specs import CtleSpecifications


def valid_metrics():
    return {
        "dc_valid": True,
        "peaking_boost": 6.0,
        "power": 10e-3,
        "hd3": -35.0,
        "noise": 1e-3,
        "eye_horizontal_ui": 0.5,
        "eye_vertical_v": 0.12,
    }


def test_reward_adds_success_bonus_when_all_specs_pass():
    reward, info = CtleReward().calculate(valid_metrics())
    assert reward == pytest.approx(20.0)
    assert info["all_specs_met"] is True
    assert info["weighted_cost"] == pytest.approx(0.0)


def test_reward_uses_normalized_constraint_violations():
    metrics = valid_metrics()
    metrics["power"] = 18e-3
    reward, info = CtleReward().calculate(metrics)
    assert reward == pytest.approx(-0.2)
    assert info["violations"]["power"] == pytest.approx(0.2)
    assert info["all_specs_met"] is False


def test_invalid_dc_is_immediate_heavy_penalty():
    reward, info = CtleReward().calculate({"dc_valid": False})
    assert reward == -100.0
    assert info["all_specs_met"] is False
    assert len(info["violations"]) == 7


class FakeEvaluator:
    def run_simulation(self, action):
        assert action.shape == (5,)
        return valid_metrics()


def test_environment_returns_gym_style_transition():
    environment = CtleEnvironment(FakeEvaluator(), max_steps=1)
    observation, info = environment.reset(seed=7)
    assert observation.shape == (8,)
    assert info["seed"] == 7

    observation, reward, terminated, truncated, info = environment.step(np.zeros(5))
    assert observation.shape == (8,)
    assert reward == pytest.approx(20.0)
    assert terminated is True
    assert truncated is False
    assert info["metrics"]["dc_valid"] is True


def test_pvt_matrix_contains_45_unique_corners():
    corners = all_pvt_corners()
    assert len(corners) == 45
    assert len({corner.name for corner in corners}) == 45
    assert corners[0].vdd == pytest.approx(1.14)
    assert corners[-1].vdd == pytest.approx(1.26)


def test_pvt_rejects_unsupported_conditions():
    with pytest.raises(ValueError):
        PvtCorner("XX", 1.0, 25.0)


def test_reporter_writes_validation_artifacts(tmp_path):
    reporter = ValidationReporter(tmp_path)
    paths = reporter.write(
        {"dc_valid": True, "power": 0.01},
        ac_frequency_hz=[1e7, 2.5e9],
        ac_gain_db=[1.0, 6.0],
        transient_time_s=[0.0, 200e-12, 400e-12],
        transient_output_v=[-0.05, 0.05, -0.05],
        pvt_results=[{"name": "TT_1.20V_0C", "peaking_boost": 6.0}],
    )
    assert {"json", "ac_csv", "ac_plot", "tran_csv", "tran_plot", "eye_plot", "pvt_csv", "pvt_plot"} <= paths.keys()
    assert all((tmp_path / path.split("/")[-1]).exists() for path in paths.values())


def test_bounded_search_keeps_best_candidate():
    class FakeSearchEvaluator:
        def run_simulation(self, action):
            return {"dc_valid": True, "peaking_boost": float(action[0] + 7.5), "power": 1e-3}

    action, rows = BoundedDesignSearch(FakeSearchEvaluator()).run(4)
    assert len(rows) == 4
    assert np.all((-1.0 <= action) & (action <= 1.0))
