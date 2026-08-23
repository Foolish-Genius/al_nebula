"""Sample-efficient bounded search used before SAC is introduced."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


class BoundedDesignSearch:
    """Evaluate deterministic low-discrepancy candidates through the AC gate."""

    def __init__(self, evaluator: Any, seed: int = 7) -> None:
        self.evaluator = evaluator
        self.seed = seed

    def run(self, evaluations: int = 24) -> tuple[np.ndarray, list[dict[str, Any]]]:
        if evaluations < 1:
            raise ValueError("evaluations must be positive")
        candidates = self._candidates(evaluations)
        rows: list[dict[str, Any]] = []
        best_action = candidates[0]
        best_score = -np.inf
        for index, action in enumerate(candidates):
            metrics = self.evaluator.run_simulation(action)
            score = self._score(metrics)
            row = {"evaluation": index + 1, "score": score, "action": action.tolist(), **metrics}
            rows.append(row)
            if score > best_score:
                best_score = score
                best_action = action.copy()
        return best_action, rows

    def write_csv(self, path: str | Path, rows: list[dict[str, Any]]) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fields = ["evaluation", "score", "dc_valid", "dc_gain", "nyquist_gain", "peaking_boost", "power"]
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field) for field in fields})

    @staticmethod
    def _score(metrics: dict[str, Any]) -> float:
        if not metrics.get("dc_valid", False):
            return -100.0
        peaking = float(metrics.get("peaking_boost", np.nan))
        power = float(metrics.get("power", np.nan))
        if not np.isfinite(peaking) or not np.isfinite(power):
            return -100.0
        return -abs(peaking - 7.5) - power / 0.015

    def _candidates(self, evaluations: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        random_candidates = rng.uniform(-1.0, 1.0, size=(evaluations, 5))
        anchors = np.asarray(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [-0.5, 0.25, -0.5, 0.5, 0.0],
                [0.5, -0.25, -0.75, -0.5, 0.5],
            ],
            dtype=float,
        )
        return np.vstack((anchors[:evaluations], random_candidates[max(0, 3 - evaluations) :]))[:evaluations]
