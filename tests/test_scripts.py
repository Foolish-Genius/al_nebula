"""Unit tests for the training-side scripts that do not need ngspice."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from rl.environment import CtleEnvironment
from rl.gym_wrapper import make_gym_env
from rl.reward import CtleReward
from rl.specs import CtleSpecifications
from scripts.evaluate_policy import best_reward_curve, latest_checkpoint, rollout_batch
from scripts.train_sac import build_specifications


def test_build_specifications_applies_overrides_and_hd3():
    default = build_specifications({})
    assert default.eye_vertical_min_v == 0.5 and not default.enforce_hd3
    specs = build_specifications({"eye_height_min": 0.25, "eye_width_min": None, "hd3": True})
    assert specs.eye_vertical_min_v == 0.25
    assert specs.eye_horizontal_min_ui == 0.7
    assert specs.enforce_hd3 and specs.constraints[-1].name == "hd3"


def test_latest_checkpoint_prefers_final_then_highest_step(tmp_path: Path):
    (tmp_path / "checkpoints").mkdir()
    for steps in (4000, 12000, 8000):
        (tmp_path / "checkpoints" / f"sac_{steps}_steps.zip").write_bytes(b"")
    assert latest_checkpoint(tmp_path).name == "sac_12000_steps.zip"
    (tmp_path / "sac_final.zip").write_bytes(b"")
    assert latest_checkpoint(tmp_path).name == "sac_final.zip"
    with pytest.raises(SystemExit):
        latest_checkpoint(tmp_path / "missing")


def test_best_reward_curve_is_running_max():
    curve = best_reward_curve(np.array([-1.0, 3.0, 2.0, 5.0, 4.0]))
    assert curve.tolist() == [-1.0, 3.0, 3.0, 5.0, 5.0]


class StepwiseEvaluator:
    """Feasible once the design's first coordinate has been pushed above 0.3."""

    def run_simulation(self, action):
        eye = 0.6 if action[0] > 0.3 else 0.2
        return {"dc_valid": True, "peaking_boost": 6.0, "power": 1e-3, "eye_horizontal_ui": 0.8, "eye_vertical_v": eye}


class PushRightPolicy:
    """Stand-in for SAC: always move the first coordinate up by the full delta."""

    def predict(self, observations, deterministic=True):
        actions = np.zeros((len(observations), 5), dtype=np.float32)
        actions[:, 0] = 1.0
        return actions, None


def test_rollout_batch_reports_steps_to_feasible_and_best_design():
    reward = CtleReward(specifications=CtleSpecifications(), margin_weight=5.0)
    envs = [
        make_gym_env(CtleEnvironment(StepwiseEvaluator(), reward_model=reward, max_steps=10, delta_scale=0.2, run_transient=False))
        for _ in range(3)
    ]
    rng = np.random.default_rng(0)
    with ThreadPoolExecutor(3) as pool:
        records = rollout_batch(PushRightPolicy(), envs, rng, max_steps=10, pool=pool)
    assert len(records) == 3
    for record in records:
        # Starts at the box centre (0), so two +0.2 steps cross 0.3.
        assert record["steps_to_feasible"] == 2
        assert record["evaluations"] == 2  # terminates on success
        assert record["best_reward"] > 20.0
        assert record["best_metrics"]["eye_vertical_v"] == 0.6
        assert record["best_design"].shape == (5,)


def test_keyword_feedback_scales_the_right_weights():
    from rl.feedback import RewardSettings, keyword_settings, settings_from_feedback

    low_power = keyword_settings("power matters much more than anything else")
    assert low_power.weights["power"] == pytest.approx(3.0)
    assert low_power.efficiency_weight == pytest.approx(3.0)
    assert low_power.weights["eye_vertical_v"] == 1.0
    assert low_power.source == "keyword"

    relaxed = keyword_settings("don't care about linearity", current=low_power)
    assert relaxed.weights["hd3"] == pytest.approx(0.5)
    assert relaxed.weights["power"] == pytest.approx(3.0)  # untouched settings carry over

    # Bounds are enforced and the offline path never needs credentials.
    extreme = keyword_settings("power is critical", current=RewardSettings(weights={"power": 9.0}))
    assert extreme.weights["power"] == 10.0
    assert settings_from_feedback("open the eye more", use_llm=False).weights["eye_vertical_v"] == pytest.approx(2.0)

    round_trip = RewardSettings.from_dict(relaxed.to_dict())
    assert round_trip.weights == relaxed.weights


def test_build_reward_applies_reward_settings():
    from scripts.train_sac import build_reward

    config = {"invalid_penalty": -10.0, "success_bonus": 20.0, "margin_weight": 5.0}
    assert build_reward(config).margin_weight == 5.0 and build_reward(config).weights == {}
    config["reward_settings"] = {"weights": {"power": 4.0}, "efficiency_weight": 2.5, "margin_weight": 1.0}
    reward = build_reward(config)
    assert reward.weights["power"] == 4.0 and reward.weights["eye_vertical_v"] == 1.0
    assert reward.efficiency_weight == 2.5 and reward.margin_weight == 1.0
