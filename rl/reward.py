"""Reward shaping for the analog sizing environment."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .specs import CtleSpecifications


class CtleReward:
    """Convert simulator metrics into a transparent scalar reward."""

    def __init__(
        self,
        specifications: CtleSpecifications | None = None,
        weights: Mapping[str, float] | None = None,
        invalid_penalty: float = -100.0,
        success_bonus: float = 20.0,
        efficiency_weight: float = 1.0,
        margin_weight: float = 0.0,
    ) -> None:
        self.specifications = specifications or CtleSpecifications()
        self.weights = dict(weights or {})
        self.invalid_penalty = invalid_penalty
        self.success_bonus = success_bonus
        self.efficiency_weight = efficiency_weight
        self.margin_weight = margin_weight

    def calculate(self, metrics: Mapping[str, float]) -> tuple[float, dict[str, object]]:
        """Return reward and diagnostics without hiding any constraint violations."""
        if not bool(metrics.get("dc_valid", False)):
            violations = {constraint.name: 1.0 for constraint in self.specifications.constraints}
            return self.invalid_penalty, {"all_specs_met": False, "violations": violations}

        evaluation = self.specifications.evaluate(metrics)
        hard_gate_names = ("hd3", "noise", "eye_horizontal_ui", "eye_vertical_v")
        constraints_by_name = {constraint.name: constraint for constraint in self.specifications.constraints}
        hard_gate_failures = {
            name: 1.0
            for name in hard_gate_names
            if name in metrics
            and name in constraints_by_name
            and constraints_by_name[name].violation(float(metrics[name])) > 0.0
        }
        if hard_gate_failures:
            evaluation = {**evaluation, "all_specs_met": False, "hard_gate_failures": hard_gate_failures}
        violations = evaluation["violations"]
        weighted_cost = sum(
            self.weights.get(name, 1.0) * float(violation)
            for name, violation in violations.items()
        )
        # Charge for power continuously so the reward still has a gradient once
        # every constraint is satisfied; the term is at most efficiency_weight.
        power = float(metrics.get("power", np.nan))
        efficiency_cost = (
            self.efficiency_weight * min(1.0, power / self.specifications.power_max_w)
            if np.isfinite(power)
            else self.efficiency_weight
        )
        reward = -weighted_cost - efficiency_cost
        # Reward the tightest spec margin once every constraint passes, so the
        # agent is pulled into the feasible set instead of hugging its boundary.
        margin = min(1.0, min(evaluation["margins"].values()))
        margin_bonus = 0.0
        if evaluation["all_specs_met"]:
            reward += self.success_bonus
            margin_bonus = self.margin_weight * max(0.0, margin)
            reward += margin_bonus
        return reward, {
            **evaluation,
            "weighted_cost": weighted_cost,
            "efficiency_cost": efficiency_cost,
            "margin": margin,
            "margin_bonus": margin_bonus,
        }
