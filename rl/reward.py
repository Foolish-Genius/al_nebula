"""Reward shaping for the analog sizing environment."""

from __future__ import annotations

from typing import Mapping

from .specs import CtleSpecifications


class CtleReward:
    """Convert simulator metrics into a transparent scalar reward."""

    def __init__(
        self,
        specifications: CtleSpecifications | None = None,
        weights: Mapping[str, float] | None = None,
        invalid_penalty: float = -100.0,
        success_bonus: float = 20.0,
    ) -> None:
        self.specifications = specifications or CtleSpecifications()
        self.weights = dict(weights or {})
        self.invalid_penalty = invalid_penalty
        self.success_bonus = success_bonus

    def calculate(self, metrics: Mapping[str, float]) -> tuple[float, dict[str, object]]:
        """Return reward and diagnostics without hiding any constraint violations."""
        if not bool(metrics.get("dc_valid", False)):
            violations = {constraint.name: 1.0 for constraint in self.specifications.constraints}
            return self.invalid_penalty, {"all_specs_met": False, "violations": violations}

        evaluation = self.specifications.evaluate(metrics)
        violations = evaluation["violations"]
        weighted_cost = sum(
            self.weights.get(name, 1.0) * float(violation)
            for name, violation in violations.items()
        )
        reward = -weighted_cost
        if evaluation["all_specs_met"]:
            reward += self.success_bonus
        return reward, {**evaluation, "weighted_cost": weighted_cost}
