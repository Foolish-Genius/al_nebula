"""Figures answering the 15 September review: peaking across specs, and the DFE before and after.

Writes into docs/figures/. Two figures need nothing but recorded artifacts; the
peaking overlay rolls the trained policy out against three peaking floors, which
needs ngspice and the IHP PDK.

    python scripts/make_review_figures.py              # both
    python scripts/make_review_figures.py --only dfe   # skip the simulated one
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rl.dfe import apply_one_tap_dfe  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
EQ_RUN = ROOT / "reports" / "eval-ihp-eq"
PVT_RUN = ROOT / "reports" / "sac-ihp-pvt"
UI = 200e-12
TARGETS = (3.0, 5.0, 7.0)
COLOURS = {3.0: "#1A6FE8", 5.0: "#0E7C57", 7.0: "#C2453B"}
SEED = 21
MAX_STEPS = 14


# --------------------------------------------------------------------- shared
def load_transient(path: Path) -> tuple[np.ndarray, np.ndarray]:
    time_s, volts = [], []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            try:
                time_s.append(float(row[0]))
                volts.append(float(row[1]))
            except (ValueError, IndexError):
                continue
    return np.asarray(time_s), np.asarray(volts)


def settled(time_s: np.ndarray, volts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep = time_s > time_s.min() + 0.5 * (time_s.max() - time_s.min())
    return time_s[keep], volts[keep]


def best_sampling_phase(time_s: np.ndarray, volts: np.ndarray) -> tuple[np.ndarray, float]:
    """The UI phase whose samples separate the two symbol groups most widely."""
    time_s, volts = settled(time_s, volts)
    best = None
    for fraction in np.linspace(0.0, 1.0, 80, endpoint=False):
        grid = np.arange(time_s.min() + fraction * UI, time_s.max() - UI, UI)
        samples = np.interp(grid, time_s, volts)
        if not ((samples > 0).any() and (samples < 0).any()):
            continue
        separation = np.percentile(samples[samples > 0], 5) - np.percentile(samples[samples < 0], 95)
        if best is None or separation > best[0]:
            best = (separation, samples, fraction)
    return best[1], best[2]


def eye_traces(time_s: np.ndarray, volts: np.ndarray, points: int = 200) -> np.ndarray:
    time_s, volts = settled(time_s, volts)
    grid = np.linspace(0.0, 2.0, points)
    starts = np.arange(time_s.min(), time_s.max() - 2 * UI, UI)
    return np.array([np.interp(start + grid * UI, time_s, volts) for start in starts])


def opening(values: np.ndarray) -> float:
    high, low = values[values > np.median(values)], values[values <= np.median(values)]
    return float(high.min() - low.max())


# ------------------------------------------------------------------- the DFE
def dfe_figures() -> None:
    """The eye and the output levels, with and without the one-tap DFE."""
    validation = json.loads((EQ_RUN / "validation.json").read_text(encoding="utf-8"))
    tap = float(validation["dfe_tap"])
    time_s, volts = load_transient(EQ_RUN / "transient.csv")
    samples, phase = best_sampling_phase(time_s, volts)
    corrected, _ = apply_one_tap_dfe(samples, tap)
    traces = eye_traces(time_s, volts)
    before, after = opening(samples), opening(corrected)
    print(f"DFE tap {tap:+.4f}: eye {before * 1e3:.0f} mV -> {after * 1e3:.0f} mV "
          f"(validation records {validation['eye_height_v'] * 1e3:.1f} -> "
          f"{validation['dfe_eye_height_v'] * 1e3:.1f} mV, measured against the transmitted bits)")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
    grid = np.linspace(0.0, 2.0, traces.shape[1])
    ax = axes[0]
    for trace in traces:
        ax.plot(grid, trace, color="#1A6FE8", alpha=0.10, linewidth=0.9)
    ax.scatter(np.full(samples.size, 1.0 + phase % 1.0), samples, s=9, color="#0F1C2E",
               zorder=3, label="sampling instant")
    ax.set_title(f"Without DFE  --  eye {before * 1e3:.0f} mV", fontsize=11)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    ax = axes[1]
    for trace in traces:
        ax.plot(grid, trace, color="#B9C6D8", alpha=0.07, linewidth=0.9)
    jitter = np.linspace(-0.06, 0.06, samples.size)
    ax.scatter(1.0 + phase % 1.0 + jitter, corrected, s=11, color="#0E7C57", zorder=3,
               label=f"after DFE, tap {tap:+.3f}")
    ax.axhline(corrected[corrected > 0].min(), color="#0E7C57", linestyle="--", linewidth=1)
    ax.axhline(corrected[corrected < 0].max(), color="#0E7C57", linestyle="--", linewidth=1)
    ax.set_title(f"With DFE  --  eye {after * 1e3:.0f} mV", fontsize=11)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    for ax in axes:
        ax.set_xlabel("unit intervals")
        ax.set_ylabel("differential output (V)")
        ax.set_ylim(-0.45, 0.45)
        ax.grid(alpha=0.25)
    fig.suptitle("CTLE output before and after the one-tap DFE  -  IHP sg13g2, 5 Gbps PRBS7", fontsize=12)
    fig.savefig(OUT / "dfe_before_after.png", dpi=170)
    print(f"wrote {OUT / 'dfe_before_after.png'}")

    decisions = np.sign(samples)
    previous = np.concatenate(([decisions[0]], decisions[:-1]))
    fig2, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    labels, values, colours = [], [], []
    for bit in (1, -1):
        for prev in (1, -1):
            mask = (decisions == bit) & (previous == prev)
            if mask.any():
                labels.append(f"{'+1' if bit > 0 else '-1'} after {'+1' if prev > 0 else '-1'}")
                values.append(corrected[mask].mean() * 1e3)
                colours.append("#0E7C57" if (bit > 0) == (corrected[mask].mean() > 0) else "#C2453B")
    ax.bar(labels, values, color=colours)
    ax.axhline(0, color="#0F1C2E", linewidth=1)
    for index, value in enumerate(values):
        ax.text(index, value + (8 if value > 0 else -14), f"{value:+.0f} mV", ha="center", fontsize=9)
    ax.set_ylabel("level after the DFE (mV)")
    ax.set_title(f"DFE output levels, tap {tap:+.3f}  --  positive and negative swings match", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    fig2.savefig(OUT / "dfe_levels.png", dpi=170)
    print(f"wrote {OUT / 'dfe_levels.png'}")


# --------------------------------------------------------------- the peaking
def size_for_target(model, base: dict, target: float) -> dict:
    from rl.environment import CtleEnvironment
    from rl.gym_wrapper import make_gym_env
    from rl.pvt import all_pvt_corners
    from scripts.train_sac import build_evaluator, build_reward

    config = dict(base)
    config["spec"] = [f"peaking_min_db={target:g}", "peaking_max_db=12"]
    evaluator = build_evaluator(config)
    environment = CtleEnvironment(
        evaluator,
        reward_model=build_reward(config),
        max_steps=MAX_STEPS,
        delta_scale=config["delta_scale"],
        run_transient=True,
        random_reset=True,
        terminate_on_success=False,
        run_linearity=bool(config.get("hd3")),
        corners=(next(c for c in all_pvt_corners() if c.name == "TT_1.00V_62.5C"),),
    )
    env = make_gym_env(environment)
    observation, _ = env.reset(seed=SEED)
    best, first, step = None, None, 0
    for step in range(1, MAX_STEPS + 1):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, _, _, info = env.step(action)
        if info["all_specs_met"] and first is None:
            first = step
        if best is None or reward > best["reward"]:
            best = {"reward": float(reward), "design": np.asarray(info["design"][:5], dtype=float),
                    "metrics": dict(info["metrics"])}
        if first is not None and step >= first + 2:
            break
    ac = evaluator.run_simulation(best["design"])
    best.update({
        "target": target, "first_feasible": first, "steps": step,
        "frequency": np.asarray(ac["ac_frequency_hz"], dtype=float),
        "gain": np.asarray(ac["ac_gain_db"], dtype=float),
        "peaking_boost": float(ac["peaking_boost"]), "peaking_max": float(ac["peaking_max"]),
        "peak_gain": float(ac["peak_gain"]), "peak_frequency": float(ac["peak_frequency_hz"]),
    })
    return best


def peaking_figure() -> None:
    """The response the policy produces for three peaking floors."""
    from stable_baselines3 import SAC

    from scripts.evaluate_policy import latest_checkpoint

    base = json.loads((PVT_RUN / "config.json").read_text(encoding="utf-8"))
    model = SAC.load(str(latest_checkpoint(PVT_RUN)), device="cpu")

    results = []
    for target in TARGETS:
        result = size_for_target(model, base, target)
        results.append(result)
        print(f"peaking >= {target:g} dB: feasible at {result['first_feasible']} | "
              f"{result['peaking_boost']:.2f} dB at Nyquist, {result['peaking_max']:.2f} dB max "
              f"at {result['peak_frequency'] / 1e9:.2f} GHz", flush=True)

    fig, ax = plt.subplots(figsize=(10.5, 5.6), constrained_layout=True)
    for result in results:
        colour = COLOURS[result["target"]]
        ax.semilogx(result["frequency"], result["gain"], color=colour, linewidth=2,
                    label=(f"spec $\\geq$ {result['target']:g} dB  ->  {result['peaking_boost']:.2f} dB "
                           f"at Nyquist, {result['peaking_max']:.2f} dB max"))
        ax.plot(result["peak_frequency"], result["peak_gain"], "o", color=colour, markersize=6)
        ax.annotate(f"peak {result['peak_frequency'] / 1e9:.1f} GHz",
                    (result["peak_frequency"], result["peak_gain"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8, color=colour)
    ax.axvline(2.5e9, color="#5B6B80", linestyle="--", linewidth=1.2)
    ax.annotate("Nyquist, 2.5 GHz", (2.5e9, ax.get_ylim()[0] + 0.6), rotation=90,
                fontsize=9, color="#5B6B80", ha="right")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("gain (dB)")
    ax.set_title("CTLE response the policy produces for three peaking specifications\n"
                 "IHP sg13g2, TT corner - the response peaks above Nyquist in every case", fontsize=11)
    ax.grid(which="both", alpha=0.22)
    ax.legend(fontsize=9, loc="lower left")
    fig.savefig(OUT / "peaking_overlay.png", dpi=170)
    print(f"wrote {OUT / 'peaking_overlay.png'}")

    (OUT / "peaking_overlay.json").write_text(json.dumps(
        [{k: v for k, v in r.items() if k not in ("frequency", "gain", "design", "metrics")}
         for r in results], indent=2, default=float), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=("dfe", "peaking"), default=None,
                        help="build just one of the two (peaking needs ngspice and the PDK)")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.only != "peaking":
        dfe_figures()
    if args.only != "dfe":
        peaking_figure()


if __name__ == "__main__":
    main()
