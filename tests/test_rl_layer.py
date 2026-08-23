import numpy as np
import pytest

from rl.environment import CtleEnvironment
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
