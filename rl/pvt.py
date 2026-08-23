"""Explicit PVT corner definitions and deterministic corner generation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product


PROCESS_CORNERS = ("TT", "SS", "FF", "SF", "FS")
VOLTAGE_CORNERS = (0.95, 1.0, 1.05)
TEMPERATURE_CORNERS_C = (0.0, 62.5, 125.0)


@dataclass(frozen=True)
class PvtCorner:
    """One process, supply, and temperature simulation condition."""

    process: str
    supply_scale: float
    temperature_c: float

    def __post_init__(self) -> None:
        if self.process not in PROCESS_CORNERS:
            raise ValueError(f"unsupported process corner: {self.process}")
        if self.supply_scale not in VOLTAGE_CORNERS:
            raise ValueError(f"unsupported supply scale: {self.supply_scale}")
        if self.temperature_c not in TEMPERATURE_CORNERS_C:
            raise ValueError(f"unsupported temperature: {self.temperature_c}")

    @property
    def vdd(self) -> float:
        return 1.2 * self.supply_scale

    @property
    def name(self) -> str:
        temperature = f"{self.temperature_c:g}C"
        voltage = f"{self.supply_scale:.2f}V"
        return f"{self.process}_{voltage}_{temperature}"


def all_pvt_corners() -> tuple[PvtCorner, ...]:
    """Return the deterministic 5 x 3 x 3 = 45-corner verification matrix."""
    return tuple(
        PvtCorner(process, supply_scale, temperature_c)
        for process, supply_scale, temperature_c in product(
            PROCESS_CORNERS, VOLTAGE_CORNERS, TEMPERATURE_CORNERS_C
        )
    )
