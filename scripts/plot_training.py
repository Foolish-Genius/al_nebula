"""Plot learning curves from one or more train_sac.py runs (works while a run is live).

Example:
    python scripts/plot_training.py reports/sac-ihp-v1 reports/sac-ihp-hd3 \
        --output reports/screenshots/learning_curves.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_run(run_dir: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader((run_dir / "steps.csv").open(encoding="utf-8")))

    def column(name: str, cast=float) -> np.ndarray:
        values = []
        for row in rows:
            try:
                values.append(cast(row[name]))
            except (ValueError, TypeError):
                values.append(np.nan)
        return np.asarray(values, dtype=float)

    return {
        "reward": column("reward"),
        "feasible": np.asarray([row["all_specs_met"] == "True" for row in rows], dtype=float),
        "eye": column("eye_vertical_v"),
        "power": column("power"),
    }


def rolling(values: np.ndarray, window: int) -> np.ndarray:
    if values.size < window:
        return values
    kernel = np.ones(window) / window
    filled = np.where(np.isfinite(values), values, np.nan)
    # nan-aware moving average
    mask = np.isfinite(filled).astype(float)
    num = np.convolve(np.nan_to_num(filled), kernel, mode="valid")
    den = np.convolve(mask, kernel, mode="valid")
    return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="train_sac.py output directories")
    parser.add_argument("--window", type=int, default=500, help="moving-average window in env steps")
    parser.add_argument("--eye-spec", type=float, default=None, help="draw the eye-height target line")
    parser.add_argument("--output", default="reports/screenshots/learning_curves.png")
    args = parser.parse_args()

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for run in args.runs:
        run_dir = Path(run)
        data = load_run(run_dir)
        steps = np.arange(1, data["reward"].size + 1)
        label = run_dir.name
        offset = args.window - 1
        axes[0].plot(steps[offset:], rolling(data["reward"], args.window), label=label)
        axes[1].plot(steps[offset:], 100 * rolling(data["feasible"], args.window), label=label)
        axes[2].plot(steps[offset:], 1e3 * rolling(data["eye"], args.window), label=label)
    axes[0].set_title("Reward per step (moving avg)")
    axes[0].set_ylabel("reward")
    axes[1].set_title("Steps meeting every spec")
    axes[1].set_ylabel("%")
    axes[1].set_ylim(0, 100)
    axes[2].set_title("Eye height after channel")
    axes[2].set_ylabel("mV")
    if args.eye_spec is not None:
        axes[2].axhline(1e3 * args.eye_spec, color="k", linestyle="--", linewidth=1, label=f"spec {1e3 * args.eye_spec:.0f} mV")
    for axis in axes:
        axis.set_xlabel("environment steps")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    figure.suptitle(f"SAC training on IHP sg13g2 PSP103 models (window {args.window} steps)")
    figure.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
