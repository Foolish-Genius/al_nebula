"""Target specifications and normalized constraint evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class Constraint:
    """A metric constraint whose normalized violation is zero when satisfied."""

    name: str
    target: float
    direction: str
    norm_factor: float
    metric_name: str | None = None

    def violation(self, value: float) -> float:
        if not np.isfinite(value):
            return 1.0
        if self.direction == "min":
            return max(0.0, (self.target - value) / self.norm_factor)
        if self.direction == "max":
            return max(0.0, (value - self.target) / self.norm_factor)
        raise ValueError(f"unsupported constraint direction: {self.direction}")


@dataclass(frozen=True)
class CtleSpecifications:
    """PCIe Gen 2 CTLE/DFE targets used by the environment and reports."""

    nyquist_frequency_hz: float = 2.5e9
    peaking_min_db: float = 3.0
    peaking_max_db: float = 12.0
    power_max_w: float = 15e-3
    hd3_max_db: float = -30.0
    noise_max_vrms: float = 1.5e-3
    eye_horizontal_min_ui: float = 0.4
    eye_vertical_min_v: float = 0.1
    constraints: tuple[Constraint, ...] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "constraints",
            (
                Constraint("peaking_boost", self.peaking_min_db, "min", 3.0),
                Constraint("peaking_ceiling", self.peaking_max_db, "max", 12.0, "peaking_boost"),
                Constraint("power", self.power_max_w, "max", self.power_max_w),
                Constraint("hd3", self.hd3_max_db, "max", 30.0),
                Constraint("noise", self.noise_max_vrms, "max", self.noise_max_vrms),
                Constraint("eye_horizontal_ui", self.eye_horizontal_min_ui, "min", self.eye_horizontal_min_ui),
                Constraint("eye_vertical_v", self.eye_vertical_min_v, "min", self.eye_vertical_min_v),
            ),
        )

    def evaluate(self, metrics: Mapping[str, float]) -> dict[str, object]:
        """Return per-spec normalized violations and an all-spec pass flag."""
        violations = {
            constraint.name: constraint.violation(
                float(metrics.get(constraint.metric_name or constraint.name, np.nan))
            )
            for constraint in self.constraints
        }
        return {
            "violations": violations,
            "all_specs_met": bool(all(value <= 0.0 for value in violations.values())),
        }
