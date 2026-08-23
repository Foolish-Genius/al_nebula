"""Gym-style environment boundary for RL agents."""

from __future__ import annotations

from typing import Any

import numpy as np

from .reward import CtleReward


class CtleEnvironment:
    """Small Gym-compatible protocol that keeps RL orchestration simulator-agnostic."""

    def __init__(self, evaluator: Any, reward_model: CtleReward | None = None, max_steps: int = 100) -> None:
        self.evaluator = evaluator
        self.reward_model = reward_model or CtleReward()
        self.max_steps = max_steps
        self.step_count = 0
        self.last_metrics: dict[str, Any] = {}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the episode and return a zero observation plus diagnostics."""
        del options
        if seed is not None:
            np.random.seed(seed)
        self.step_count = 0
        self.last_metrics = {}
        return np.zeros(self.observation_size, dtype=np.float32), {"seed": seed}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Evaluate one action and return observation, reward, termination, truncation, info."""
        metrics = self.evaluator.run_simulation(np.asarray(action, dtype=np.float64))
        reward, diagnostics = self.reward_model.calculate(metrics)
        self.last_metrics = dict(metrics)
        self.step_count += 1
        terminated = not bool(metrics.get("dc_valid", False)) or bool(diagnostics["all_specs_met"])
        truncated = self.step_count >= self.max_steps and not terminated
        observation = self._observation(metrics, diagnostics)
        info = {"metrics": metrics, **diagnostics}
        return observation, float(reward), terminated, truncated, info

    @property
    def observation_size(self) -> int:
        return 1 + len(self.reward_model.specifications.constraints)

    @staticmethod
    def _observation(metrics: dict[str, Any], diagnostics: dict[str, Any]) -> np.ndarray:
        dc_valid = 1.0 if metrics.get("dc_valid", False) else 0.0
        violations = diagnostics["violations"]
        return np.asarray([dc_valid, *[float(value) for value in violations.values()]], dtype=np.float32)
