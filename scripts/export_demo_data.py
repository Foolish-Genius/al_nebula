"""Roll a trained policy out and export every step's waveforms for the visual demo.

Produces one JSON file that docs/demo/index.html animates: per step the design
(normalised and SI), the metrics, the per-spec violations, the reward, the AC
response, and the 2-UI eye traces. Also carries the training curve so the page
can show what the policy learned from.

Example:
    python scripts/export_demo_data.py reports/sac-ihp-pvt --seed 21 \
        --output docs/demo/rollout.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from rl.environment import CtleEnvironment
from rl.equalizer import EqualizerEvaluator
from rl.gym_wrapper import make_gym_env
from rl.pvt import all_pvt_corners
from scripts.train_sac import build_evaluator, build_reward, build_specifications, curriculum_corners
from scripts.evaluate_policy import latest_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def eye_traces(time_s, output_v, ui: float, max_traces: int = 140, points: int = 90) -> list[list[float]]:
    """2-UI slices of the settled half of the waveform, resampled onto a common grid."""
    time_s = np.asarray(time_s, dtype=float)
    output_v = np.asarray(output_v, dtype=float)
    if time_s.size < 4:
        return []
    grid = np.linspace(0.0, 2.0, points)
    first = time_s.min() + 0.5 * (time_s.max() - time_s.min())
    traces = []
    for start in np.arange(first, time_s.max() - 2 * ui, ui):
        sample = np.interp(start + grid * ui, time_s, output_v)
        traces.append([round(float(v), 4) for v in sample])
        if len(traces) >= max_traces:
            break
    return traces


def ac_curve(frequency_hz, gain_db, points: int = 90) -> dict:
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    gain_db = np.asarray(gain_db, dtype=float)
    if frequency_hz.size < 2:
        return {"f": [], "g": []}
    grid = np.logspace(np.log10(frequency_hz.min()), np.log10(frequency_hz.max()), points)
    return {
        "f": [round(float(f), 1) for f in grid],
        "g": [round(float(v), 3) for v in np.interp(grid, frequency_hz, gain_db)],
    }


def training_curve(run_dir: Path, window: int = 400, points: int = 220) -> dict:
    path = run_dir / "steps.csv"
    if not path.exists():
        return {}
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    feasible = np.array([r["all_specs_met"] == "True" for r in rows], dtype=float)
    reward = np.array([float(r["reward"]) for r in rows], dtype=float)
    if feasible.size < window:
        return {}
    kernel = np.ones(window) / window
    smooth_f = np.convolve(feasible, kernel, mode="valid") * 100.0
    smooth_r = np.convolve(reward, kernel, mode="valid")
    index = np.linspace(0, smooth_f.size - 1, min(points, smooth_f.size)).astype(int)
    return {
        "step": [int(i + window) for i in index],
        "feasible": [round(float(smooth_f[i]), 2) for i in index],
        "reward": [round(float(smooth_r[i]), 3) for i in index],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--corners", choices=("nominal", "all", "process"), default=None)
    parser.add_argument("--pvt", action="store_true", help="also run the 45-corner sweep on the final design")
    parser.add_argument("--training-from", default=None, help="run directory whose steps.csv supplies the training curve (default: the rollout run)")
    parser.add_argument("--output", default="docs/demo/rollout.json")
    args = parser.parse_args()

    from stable_baselines3 import SAC

    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    checkpoint = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(run_dir)
    specs = build_specifications(config)
    evaluator = build_evaluator(config)
    equalizer = bool(config.get("equalizer"))

    env_evaluator = EqualizerEvaluator(evaluator) if equalizer else evaluator
    environment = CtleEnvironment(
        env_evaluator,
        reward_model=build_reward(config),
        max_steps=args.max_steps,
        delta_scale=config["delta_scale"],
        run_transient=config["run_transient"],
        random_reset=True,
        terminate_on_success=True,
        run_linearity=bool(config.get("hd3")),
        corners=curriculum_corners(args.corners or config.get("corners", "nominal")),
    )
    env = make_gym_env(environment)
    model = SAC.load(str(checkpoint), device="cpu")

    observation, _ = env.reset(seed=args.seed)
    corner = environment.current_corner.name if environment.current_corner else "TT_1.20V_27C"
    ui = evaluator.unit_interval_s
    steps = []
    final_design = None
    for index in range(1, args.max_steps + 1):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        design = np.asarray(info["design"], dtype=float)
        parameters = (EqualizerEvaluator(evaluator) if equalizer else evaluator).map_actions(design)
        metrics = info["metrics"]
        # Re-run the gates on this design to capture the waveforms for the page.
        ac = evaluator.run_simulation(design[:5])
        transient = evaluator.run_transient(design[:5]) if ac.get("dc_valid") else {}
        steps.append({
            "step": index,
            "design": [round(float(v), 4) for v in design],
            "parameters": {k: float(v) for k, v in parameters.items()},
            "metrics": {k: (None if v is None or (isinstance(v, float) and not np.isfinite(v)) else (float(v) if np.isscalar(v) and not isinstance(v, bool) else bool(v) if isinstance(v, bool) else None))
                        for k, v in metrics.items() if np.isscalar(v) or v is None},
            "violations": {k: round(float(v), 4) for k, v in info["violations"].items()},
            "reward": round(float(reward), 3),
            "all_specs_met": bool(info["all_specs_met"]),
            "ac": ac_curve(ac.get("ac_frequency_hz", []), ac.get("ac_gain_db", [])),
            "eye": eye_traces(transient.get("time_s", []), transient.get("output_v", []), ui),
        })
        final_design = design
        print(f"step {index}: reward {reward:.2f} feasible={info['all_specs_met']}", flush=True)
        if terminated or truncated:
            break

    payload = {
        "run": run_dir.name,
        "checkpoint": checkpoint.name,
        "model_source": evaluator.model_source,
        "corner": corner,
        "seed": args.seed,
        "unit_interval_ps": round(ui * 1e12, 1),
        "data_rate_gbps": round(1e-9 / ui, 2),
        "nyquist_ghz": round(evaluator.nyquist_frequency_hz / 1e9, 3),
        "channel_loss_db": round(evaluator.channel_loss_at_nyquist_db(), 2),
        "bounds": {name: [float(low), float(high)] for name, (low, high) in evaluator.bounds.items()},
        "specs": {
            "peaking_min_db": specs.peaking_min_db, "peaking_max_db": specs.peaking_max_db,
            "power_max_w": specs.power_max_w, "eye_vertical_min_v": specs.eye_vertical_min_v,
            "eye_horizontal_min_ui": specs.eye_horizontal_min_ui, "hd3_max_db": specs.hd3_max_db,
            "enforce_hd3": specs.enforce_hd3,
        },
        "steps": steps,
        "training": training_curve(Path(args.training_from) if args.training_from else run_dir),
    }

    if args.pvt and final_design is not None:
        rows = evaluator.run_pvt(final_design[:5], all_pvt_corners())
        payload["pvt"] = [{"name": r["name"], "peaking": None if not np.isfinite(r.get("peaking_boost", np.nan)) else round(float(r["peaking_boost"]), 3), "pass": bool(r["pvt_pass"])} for r in rows]
        print(f"PVT: {sum(r['pvt_pass'] for r in rows)}/{len(rows)} pass", flush=True)

    if final_design is not None:
        payload["netlist"] = evaluator.sized_netlist(final_design[:5], dfe_tap=parameters.get("dfe_tap"))

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {output} ({output.stat().st_size // 1024} KB, {len(steps)} steps)")


if __name__ == "__main__":
    main()
