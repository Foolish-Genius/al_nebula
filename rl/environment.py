"""Gym-style environment boundary for RL agents."""

from __future__ import annotations

from typing import Any

import numpy as np

from .reward import CtleReward


class CtleEnvironment:
    """Small Gym-compatible protocol that keeps RL orchestration simulator-agnostic.

    The environment state is the current normalized design vector in [-1, 1]^5.
    In ``"delta"`` mode (default) each action nudges that vector by
    ``action * delta_scale`` so an agent can size the circuit incrementally;
    in ``"absolute"`` mode each action replaces the design vector outright.

    Every step runs the DC/AC gate and, when it passes, the transient PRBS gate
    so eye metrics reach the reward. With ``run_linearity`` the HD3 gate runs
    too, but only once every other spec passes, so it costs one extra
    simulation per otherwise-feasible step rather than per step. Episodes end on ``all_specs_met`` (unless
    ``terminate_on_success`` is off, in which case feasible steps keep paying
    out until ``max_steps``) or after ``max_steps``; a DC failure is penalized
    but does not end the episode, so a delta-mode agent can step back out of an
    invalid region.
    """

    ACTION_SIZE = 5
    EQUALIZER_ACTION_SIZE = 6
    ACTION_MODES = ("delta", "absolute")
    METRIC_FEATURES = ("peaking_boost", "power", "eye_vertical_v", "eye_horizontal_ui")

    def __init__(
        self,
        evaluator: Any,
        reward_model: CtleReward | None = None,
        max_steps: int = 100,
        action_mode: str = "delta",
        delta_scale: float = 0.2,
        run_transient: bool = True,
        random_reset: bool = False,
        terminate_on_success: bool = True,
        run_linearity: bool = False,
    ) -> None:
        if action_mode not in self.ACTION_MODES:
            raise ValueError(f"unsupported action mode: {action_mode}")
        if delta_scale <= 0.0:
            raise ValueError("delta_scale must be positive")
        self.evaluator = evaluator
        # A whole-equalizer evaluator (see rl.equalizer) takes a six-value action:
        # five CTLE sizing values plus one DFE tap.
        self.whole_equalizer = getattr(evaluator, "run", None)
        self.action_size = self.EQUALIZER_ACTION_SIZE if self.whole_equalizer is not None else self.ACTION_SIZE
        self.reward_model = reward_model or CtleReward()
        self.max_steps = max_steps
        self.action_mode = action_mode
        self.delta_scale = delta_scale
        self.run_transient = run_transient and hasattr(evaluator, "run_transient")
        self.run_linearity = run_linearity and hasattr(evaluator, "run_linearity")
        self.random_reset = random_reset
        self.terminate_on_success = terminate_on_success
        self.step_count = 0
        self.design = np.zeros(self.action_size, dtype=np.float64)
        self.last_metrics: dict[str, Any] = {}
        self._rng = np.random.default_rng()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the design vector and return the initial observation plus diagnostics.

        ``options["initial_design"]`` fixes the starting point; otherwise it is
        the centre of the action box, or a uniform sample when ``random_reset``.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.step_count = 0
        self.last_metrics = {}
        initial = (options or {}).get("initial_design")
        if initial is not None:
            self.design = self._validated(initial)
        elif self.random_reset:
            self.design = self._rng.uniform(-1.0, 1.0, size=self.action_size)
        else:
            self.design = np.zeros(self.action_size, dtype=np.float64)
        blank_violations = {constraint.name: 0.0 for constraint in self.reward_model.specifications.constraints}
        observation = self._observation({}, {"violations": blank_violations})
        return observation, {"seed": seed, "design": self.design.copy()}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply one action, simulate the resulting design, and return the transition."""
        action = self._validated(action)
        if self.action_mode == "delta":
            self.design = np.clip(self.design + action * self.delta_scale, -1.0, 1.0)
        else:
            self.design = action

        metrics = self._simulate(self.design)
        reward, diagnostics = self.reward_model.calculate(metrics)
        self.last_metrics = dict(metrics)
        self.step_count += 1
        terminated = bool(diagnostics["all_specs_met"]) and self.terminate_on_success
        truncated = self.step_count >= self.max_steps and not terminated
        observation = self._observation(metrics, diagnostics)
        info = {"metrics": metrics, "design": self.design.copy(), **diagnostics}
        if hasattr(self.evaluator, "map_actions"):
            info["parameters"] = self.evaluator.map_actions(self.design)
        return observation, float(reward), terminated, truncated, info

    def _simulate(self, design: np.ndarray) -> dict[str, Any]:
        if self.whole_equalizer is not None:
            metrics = dict(self.whole_equalizer(design))
            if metrics.get("dc_valid", False):
                metrics["eye_vertical_v"] = metrics.get("dfe_eye_height_v", metrics.get("eye_height_v", np.nan))
                metrics["eye_horizontal_ui"] = metrics.get("eye_width_ui", np.nan)
                if self.run_linearity and self._others_pass(metrics, except_name="hd3"):
                    linearity = self.evaluator.run_linearity(design)
                    metrics["hd3_db"] = linearity.get("hd3_db", np.nan)
                    metrics["linearity_error"] = linearity.get("error")
            return metrics
        metrics = dict(self.evaluator.run_simulation(design))
        if not self.run_transient or not bool(metrics.get("dc_valid", False)):
            return metrics
        transient = self.evaluator.run_transient(design)
        metrics["eye_vertical_v"] = transient.get("eye_height_v", np.nan)
        metrics["eye_horizontal_ui"] = transient.get("eye_width_ui", np.nan)
        metrics["tran_valid"] = transient.get("tran_valid", False)
        metrics["transient_error"] = transient.get("error")
        if self.run_linearity and self._others_pass(metrics, except_name="hd3"):
            linearity = self.evaluator.run_linearity(design)
            metrics["hd3_db"] = linearity.get("hd3_db", np.nan)
            metrics["linearity_error"] = linearity.get("error")
        return metrics

    def _others_pass(self, metrics: dict[str, Any], except_name: str) -> bool:
        """True when every constraint other than ``except_name`` is satisfied."""
        violations = self.reward_model.specifications.evaluate(metrics)["violations"]
        return all(value <= 0.0 for name, value in violations.items() if name != except_name)

    @property
    def observation_size(self) -> int:
        return self.action_size + 1 + len(self.METRIC_FEATURES) + len(self.reward_model.specifications.constraints)

    def _observation(self, metrics: dict[str, Any], diagnostics: dict[str, Any]) -> np.ndarray:
        """Return [design, dc_valid, normalized metrics, violations] with NaN mapped to zero."""
        specs = self.reward_model.specifications
        dc_valid = 1.0 if metrics.get("dc_valid", False) else 0.0
        scales = {
            "peaking_boost": specs.peaking_max_db,
            "power": specs.power_max_w,
            "eye_vertical_v": 1.0,
            "eye_horizontal_ui": 1.0,
        }
        features = [
            self._finite(metrics.get(name, np.nan)) / scales[name]
            for name in self.METRIC_FEATURES
        ]
        violations = [float(value) for value in diagnostics["violations"].values()]
        return np.asarray([*self.design, dc_valid, *features, *violations], dtype=np.float32)

    def _validated(self, action: Any) -> np.ndarray:
        values = np.asarray(action, dtype=np.float64)
        if values.shape != (self.action_size,):
            raise ValueError(f"action must have shape ({self.action_size},)")
        if not np.all(np.isfinite(values)) or np.any(np.abs(values) > 1.0):
            raise ValueError("action must be finite and bounded by [-1.0, 1.0]")
        return values

    @staticmethod
    def _finite(value: Any) -> float:
        value = float(value)
        return value if np.isfinite(value) else 0.0
