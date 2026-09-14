import numpy as np
import pytest

from analysis.reporting import ValidationReporter
from rl.environment import CtleEnvironment
from rl.dfe import apply_one_tap_dfe, optimize_one_tap
from rl.equalizer import EqualizerEvaluator
from rl.gym_wrapper import make_gym_env
from rl.pvt import PvtCorner, all_pvt_corners
from rl.search import BoundedDesignSearch
from rl.reward import CtleReward
from rl.specs import CtleSpecifications


def valid_metrics():
    return {
        "dc_valid": True,
        "peaking_boost": 6.0,
        "power": 1e-3,
        "eye_horizontal_ui": 0.8,
        "eye_vertical_v": 0.6,
    }


def test_reward_adds_success_bonus_when_all_specs_pass():
    reward, info = CtleReward().calculate(valid_metrics())
    assert reward == pytest.approx(19.5)  # +20 bonus, -0.5 for 1 mW of the 2 mW budget
    assert info["all_specs_met"] is True
    assert info["weighted_cost"] == pytest.approx(0.0)
    assert info["efficiency_cost"] == pytest.approx(0.5)
    reward, _ = CtleReward(efficiency_weight=0.0).calculate(valid_metrics())
    assert reward == pytest.approx(20.0)


def test_reward_uses_normalized_constraint_violations():
    metrics = valid_metrics()
    metrics["power"] = 2.4e-3
    reward, info = CtleReward(efficiency_weight=0.0).calculate(metrics)
    assert reward == pytest.approx(-0.2)
    assert info["violations"]["power"] == pytest.approx(0.2)
    assert info["all_specs_met"] is False


def test_reward_margin_bonus_rewards_tightest_slack_once_feasible():
    # valid_metrics(): peaking 6 dB has margin 1.0 (min 3 dB, norm 3) and 0.5 (max 12, norm 12),
    # power 0.5, eye width (0.8-0.7)/0.7, eye height (0.6-0.5)/0.5 = 0.2 -> tightest is 0.143.
    reward, info = CtleReward(efficiency_weight=0.0, margin_weight=10.0).calculate(valid_metrics())
    assert info["margin"] == pytest.approx((0.8 - 0.7) / 0.7)
    assert info["margin_bonus"] == pytest.approx(10.0 * (0.8 - 0.7) / 0.7)
    assert reward == pytest.approx(20.0 + info["margin_bonus"])

    metrics = valid_metrics()
    metrics["eye_vertical_v"] = 0.4
    reward, info = CtleReward(efficiency_weight=0.0, margin_weight=10.0).calculate(metrics)
    assert info["all_specs_met"] is False
    assert info["margin"] == pytest.approx(-0.2)
    assert info["margin_bonus"] == 0.0
    assert reward == pytest.approx(-0.2)


def test_invalid_dc_is_immediate_heavy_penalty():
    reward, info = CtleReward().calculate({"dc_valid": False})
    assert reward == -100.0
    assert info["all_specs_met"] is False
    assert len(info["violations"]) == 5


class FakeEvaluator:
    def run_simulation(self, action):
        assert action.shape == (5,)
        return valid_metrics()


def test_hd3_constraint_only_when_enforced():
    assert "hd3" not in {c.name for c in CtleSpecifications().constraints}
    specs = CtleSpecifications(enforce_hd3=True)
    assert [c.name for c in specs.constraints][-1] == "hd3"
    metrics = {**valid_metrics(), "hd3_db": -35.0}
    assert specs.evaluate(metrics)["all_specs_met"] is True
    metrics["hd3_db"] = -25.0
    result = specs.evaluate(metrics)
    assert result["all_specs_met"] is False
    assert result["violations"]["hd3"] == pytest.approx(0.5)  # 5 dB over, normalised by 10 dB
    # Unmeasured HD3 is a full violation, so a feasible-looking design cannot succeed.
    assert specs.evaluate(valid_metrics())["violations"]["hd3"] == 1.0


def test_reward_hard_gate_reads_hd3_by_metric_name():
    reward = CtleReward(specifications=CtleSpecifications(enforce_hd3=True), efficiency_weight=0.0)
    value, info = reward.calculate({**valid_metrics(), "hd3_db": -25.0})
    assert info["hard_gate_failures"] == {"hd3": 1.0}
    assert info["all_specs_met"] is False
    assert value == pytest.approx(-0.5)


class LinearityEvaluator(FakeEvaluator):
    def __init__(self, hd3_db=-35.0):
        self.hd3_db = hd3_db
        self.linearity_calls = 0

    def run_transient(self, action):
        return {"eye_height_v": 0.6, "eye_width_ui": 0.8, "tran_valid": True, "error": None}

    def run_linearity(self, action):
        self.linearity_calls += 1
        return {"hd3_db": self.hd3_db, "linearity_valid": True, "error": None}


def test_environment_runs_linearity_only_when_other_specs_pass():
    evaluator = LinearityEvaluator(hd3_db=-35.0)
    reward = CtleReward(specifications=CtleSpecifications(enforce_hd3=True))
    environment = CtleEnvironment(evaluator, reward_model=reward, max_steps=3, run_linearity=True)
    environment.reset()
    _, _, terminated, _, info = environment.step(np.zeros(5))
    assert evaluator.linearity_calls == 1
    assert info["metrics"]["hd3_db"] == -35.0
    assert info["all_specs_met"] is True and terminated is True

    # Power over budget: the linearity gate is skipped and HD3 stays unmeasured.
    class OverPower(LinearityEvaluator):
        def run_simulation(self, action):
            return {**valid_metrics(), "power": 3e-3}

    evaluator = OverPower()
    environment = CtleEnvironment(evaluator, reward_model=reward, max_steps=3, run_linearity=True)
    environment.reset()
    _, _, _, _, info = environment.step(np.zeros(5))
    assert evaluator.linearity_calls == 0
    assert "hd3_db" not in info["metrics"]
    assert info["violations"]["hd3"] == 1.0

    # Without run_linearity the gate never runs even when enforced.
    evaluator = LinearityEvaluator()
    environment = CtleEnvironment(evaluator, reward_model=reward, max_steps=3)
    environment.reset()
    environment.step(np.zeros(5))
    assert evaluator.linearity_calls == 0


def test_environment_can_hold_on_success_until_max_steps():
    environment = CtleEnvironment(FakeEvaluator(), max_steps=2, terminate_on_success=False)
    environment.reset()
    _, reward, terminated, truncated, info = environment.step(np.zeros(5))
    assert info["all_specs_met"] is True
    assert reward == pytest.approx(19.5)
    assert terminated is False and truncated is False
    _, _, terminated, truncated, _ = environment.step(np.zeros(5))
    assert terminated is False and truncated is True


def test_environment_returns_gym_style_transition():
    environment = CtleEnvironment(FakeEvaluator(), max_steps=1)
    observation, info = environment.reset(seed=7)
    assert observation.shape == (15,)
    assert info["seed"] == 7

    observation, reward, terminated, truncated, info = environment.step(np.zeros(5))
    assert observation.shape == (15,)
    assert reward == pytest.approx(19.5)
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


def test_pvt_report_marks_missing_and_passed_corners(tmp_path):
    reporter = ValidationReporter(tmp_path)
    paths = reporter.write(
        {"dc_valid": False},
        pvt_results=[
            {"name": "TT", "peaking_boost": 6.0, "pvt_pass": True},
            {"name": "SS", "peaking_boost": float("nan"), "pvt_pass": False},
        ],
    )
    assert (tmp_path / "pvt_peaking.png").exists()
    assert paths["pvt_plot"].endswith("pvt_peaking.png")


def test_bounded_search_keeps_best_candidate():
    class FakeSearchEvaluator:
        def run_simulation(self, action):
            return {"dc_valid": True, "peaking_boost": float(action[0] + 7.5), "power": 1e-3}

    action, rows = BoundedDesignSearch(FakeSearchEvaluator()).run(4)
    assert len(rows) == 4
    assert np.all((-1.0 <= action) & (action <= 1.0))


def test_one_tap_dfe_corrects_previous_decision_contribution():
    corrected, decisions = apply_one_tap_dfe(np.array([-1.0, 1.4, -1.0, 1.4]), tap=0.4)
    assert decisions.tolist() == [-1.0, 1.0, -1.0, 1.0]
    assert corrected[1] == pytest.approx(1.8)


def test_one_tap_search_returns_finite_tap_and_eye_metric():
    result = optimize_one_tap(np.tile([-1.0, 1.3, -1.0, 1.3], 8))
    assert -0.5 <= result["tap"] <= 0.5
    assert np.isfinite(result["eye_height_v"])


def test_equalizer_action_controls_ctle_and_dfe_tap():
    class FakeSpice:
        def run_simulation(self, action):
            assert action.shape == (5,)
            return {"dc_valid": True, "peaking_boost": 6.0, "power": 1e-3}

        def run_transient(self, action):
            time = np.arange(0.0, 4.0e-9, 200e-12)
            values = np.tile([-1.0, 1.2], 10)[:time.size]
            return {"tran_valid": True, "time_s": time, "output_v": values}

    result = EqualizerEvaluator(FakeSpice()).run(np.zeros(6))
    assert result["equalizer_valid"] is True
    assert result["equalizer_tap"] == pytest.approx(0.0)


def test_environment_uses_six_action_whole_equalizer():
    class Whole:
        def run(self, action):
            assert action.shape == (6,)
            return {"dc_valid": True, "peaking_boost": 6.0, "power": 1e-3}

    environment = CtleEnvironment(Whole())
    _, _, _, _, info = environment.step(np.zeros(6))
    assert info["metrics"]["peaking_boost"] == 6.0


class FakeTransientEvaluator(FakeEvaluator):
    def __init__(self):
        self.transient_calls = 0

    def run_transient(self, action):
        self.transient_calls += 1
        return {"eye_height_v": 0.6, "eye_width_ui": 0.8, "tran_valid": True, "error": None}


def test_environment_merges_transient_eye_metrics_into_step():
    evaluator = FakeTransientEvaluator()
    environment = CtleEnvironment(evaluator)
    environment.reset()
    _, reward, terminated, _, info = environment.step(np.zeros(5))
    assert evaluator.transient_calls == 1
    assert info["metrics"]["eye_vertical_v"] == pytest.approx(0.6)
    assert info["metrics"]["eye_horizontal_ui"] == pytest.approx(0.8)
    assert reward == pytest.approx(19.5)
    assert terminated is True


def test_delta_actions_move_and_clip_the_design_vector():
    environment = CtleEnvironment(FakeEvaluator(), delta_scale=0.5, run_transient=False)
    environment.reset(options={"initial_design": [0.8, 0.0, 0.0, 0.0, 0.0]})
    _, _, _, _, info = environment.step(np.array([1.0, -1.0, 0.0, 0.0, 0.0]))
    assert info["design"] == pytest.approx([1.0, -0.5, 0.0, 0.0, 0.0])
    observation, *_ = environment.step(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
    assert observation[0] == pytest.approx(1.0)


def test_absolute_actions_replace_the_design_vector():
    environment = CtleEnvironment(FakeEvaluator(), action_mode="absolute", run_transient=False)
    environment.reset()
    _, _, _, _, info = environment.step(np.array([-0.2, 0.4, 0.0, 0.1, -1.0]))
    assert info["design"] == pytest.approx([-0.2, 0.4, 0.0, 0.1, -1.0])


def test_dc_failure_is_penalized_but_does_not_end_the_episode():
    class FailingEvaluator:
        def run_simulation(self, action):
            return {"dc_valid": False, "error": "singular"}

    environment = CtleEnvironment(FailingEvaluator(), max_steps=2)
    environment.reset()
    observation, reward, terminated, truncated, _ = environment.step(np.zeros(5))
    assert reward == -100.0
    assert terminated is False and truncated is False
    assert observation[5] == 0.0
    _, _, terminated, truncated, _ = environment.step(np.zeros(5))
    assert terminated is False and truncated is True


def test_random_reset_is_seeded():
    environment = CtleEnvironment(FakeEvaluator(), random_reset=True, run_transient=False)
    first, _ = environment.reset(seed=3)
    second, _ = environment.reset(seed=3)
    assert np.array_equal(first, second)
    assert np.any(first[:5] != 0.0)


def test_environment_rejects_bad_actions():
    environment = CtleEnvironment(FakeEvaluator(), run_transient=False)
    environment.reset()
    with pytest.raises(ValueError):
        environment.step(np.zeros(4))
    with pytest.raises(ValueError):
        environment.step(np.array([0.0, 0.0, 1.5, 0.0, 0.0]))


def test_gym_wrapper_exposes_box_spaces():
    gymnasium = pytest.importorskip("gymnasium")
    env = CtleEnvironment(FakeEvaluator(), run_transient=False)
    wrapped = make_gym_env(env)
    assert isinstance(wrapped, gymnasium.Env)
    assert wrapped.action_space.shape == (5,)
    assert wrapped.observation_space.shape == (env.observation_size,)
    observation, _ = wrapped.reset(seed=1)
    assert wrapped.observation_space.contains(observation)
    observation, reward, terminated, truncated, info = wrapped.step(np.full(5, 1.0000001, dtype=np.float32))
    assert wrapped.observation_space.contains(observation)
    assert terminated is True
