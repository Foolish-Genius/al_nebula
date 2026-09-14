"""Summarise a train_sac.py output directory while it is still running.

Example:
    python scripts/sac_progress.py reports/sac-long-v2
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    parser.add_argument("--window", type=int, default=2000, help="env steps per summary row")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    rows = list(csv.DictReader((output_dir / "steps.csv").open(encoding="utf-8")))
    if not rows:
        raise SystemExit("no steps logged yet")
    reward = np.array([float(row["reward"]) for row in rows])
    feasible = np.array([row["all_specs_met"] == "True" for row in rows])
    dc_valid = np.array([row["dc_valid"] == "True" for row in rows])
    eye_v = np.array([float(row["eye_vertical_v"]) if row["eye_vertical_v"] not in ("", "None", "nan") else np.nan for row in rows])
    total = len(rows)

    print(f"{total} env steps logged; best step reward {reward.max():.2f}")
    print(f"{'steps':>13} {'mean_rew':>9} {'feasible':>9} {'dc_valid':>9} {'med_eye_v':>10}")
    for start in range(0, total, args.window):
        end = min(total, start + args.window)
        window = slice(start, end)
        print(
            f"{start:>6}-{end:<6} {reward[window].mean():9.2f} {feasible[window].mean() * 100:8.1f}% "
            f"{dc_valid[window].mean() * 100:8.1f}% {np.nanmedian(eye_v[window]):10.3f}"
        )

    episodes: list[tuple[float, float, int]] = []
    for path in glob.glob(str(output_dir / "*.monitor.csv")):
        with open(path, encoding="utf-8") as handle:
            handle.readline()
            episodes += [(float(r["t"]), float(r["r"]), int(r["l"])) for r in csv.DictReader(handle)]
    if episodes:
        episodes.sort()
        wall = episodes[-1][0]
        print(f"\n{len(episodes)} episodes, {wall / 60:.0f} min wall, {wall / total:.2f} s/env-step")
        tail = episodes[-max(1, len(episodes) // 5):]
        print(f"last 20% of episodes: mean return {np.mean([e[1] for e in tail]):.1f}, mean length {np.mean([e[2] for e in tail]):.1f}")


if __name__ == "__main__":
    main()
