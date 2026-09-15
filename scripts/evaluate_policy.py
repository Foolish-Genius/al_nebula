"""Evaluate a trained SAC checkpoint with deterministic rollouts and compare it to random search.

Rebuilds the environment from the run's config.json, rolls the policy out
from N random starting designs without exploration noise, and reports how
often and how quickly it reaches a design that meets every spec. If a random
search run (scripts/baseline_random.py) is given, the best reward found by
each method is compared at the same number of simulated designs.

Example:
    python scripts/evaluate_policy.py reports/sac-ihp-v1 --rollouts 20 \
        --baseline reports/baseline-ihp --output-dir reports/eval-ihp-v1
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from analysis.reporting import ValidationReporter
from rl.environment import CtleEnvironment
from rl.equalizer import EqualizerEvaluator
from rl.gym_wrapper import make_gym_env
from rl.pvt import all_pvt_corners
from scripts.train_sac import build_evaluator, build_reward, build_specifications
from spice.spice_engine import SpiceEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="train_sac.py output directory (config.json, checkpoints/, sac_final.zip)")
    parser.add_argument("--checkpoint", default=None, help="model .zip (default: sac_final.zip, else the latest checkpoint)")
    parser.add_argument("--rollouts", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=None, help="steps per rollout (default: the run's episode length)")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--baseline", default=None, help="baseline_random.py output directory to compare against")
    parser.add_argument("--n-envs", type=int, default=8, help="rollouts simulated in parallel")
    parser.add_argument("--output-dir", default=None, help="default: <run_dir>/evaluation")
    parser.add_argument("--skip-validation", action="store_true", help="do not run PVT/HD3 validation on the best design")
    return parser.parse_args()


def latest_checkpoint(run_dir: Path) -> Path:
    final = run_dir / "sac_final.zip"
    if final.exists():
        return final
    checkpoints = sorted((run_dir / "checkpoints").glob("sac_*_steps.zip"), key=lambda p: int(p.stem.split("_")[1]))
    if not checkpoints:
        raise SystemExit(f"no model found under {run_dir}")
    return checkpoints[-1]


def rollout_batch(model, envs, rng, max_steps: int, pool: ThreadPoolExecutor) -> list[dict]:
    """Roll every env out in lockstep from a random start; return one record per env."""
    observations = []
    for env in envs:
        obs, _ = env.reset(seed=int(rng.integers(2**31 - 1)))
        observations.append(obs)
    observations = np.asarray(observations, dtype=np.float32)
    records = [
        {"steps_to_feasible": None, "best_reward": -np.inf, "best_design": None, "best_metrics": {}, "evaluations": 0, "trace": []}
        for _ in envs
    ]
    active = list(range(len(envs)))
    for _ in range(max_steps):
        if not active:
            break
        actions, _ = model.predict(observations[active], deterministic=True)

        def step(pair):
            index, action = pair
            return index, envs[index].step(action)

        for index, (obs, reward, terminated, truncated, info) in pool.map(step, zip(active, actions)):
            record = records[index]
            record["evaluations"] += 1
            record["trace"].append(float(reward))
            observations[index] = obs
            if reward > record["best_reward"]:
                record["best_reward"] = float(reward)
                record["best_design"] = np.asarray(info["design"], dtype=np.float64)
                record["best_metrics"] = {k: v for k, v in info["metrics"].items() if np.isscalar(v) or v is None}
            if info.get("all_specs_met") and record["steps_to_feasible"] is None:
                record["steps_to_feasible"] = record["evaluations"]
            if terminated or truncated:
                active.remove(index)
    return records


def best_reward_curve(rewards: np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(rewards)


def main() -> None:
    args = parse_args()
    try:
        from stable_baselines3 import SAC
    except ImportError as error:
        raise SystemExit("install the 'rl' extra first: pip install -e .[rl]") from error

    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    checkpoint = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    max_steps = args.max_steps or int(config["max_steps"])

    specifications = build_specifications(config)
    evaluator = build_evaluator(config)

    equalizer = bool(config.get("equalizer"))

    def make_env():
        env_evaluator = build_evaluator(config)
        if equalizer:
            env_evaluator = EqualizerEvaluator(env_evaluator)
        reward = build_reward(config)
        # Terminate on success so steps_to_feasible is well defined; the
        # policy was trained to hold, so the first feasible step is a fair test.
        environment = CtleEnvironment(
            env_evaluator,
            reward_model=reward,
            max_steps=max_steps,
            delta_scale=config["delta_scale"],
            run_transient=config["run_transient"],
            random_reset=True,
            terminate_on_success=True,
            run_linearity=bool(config.get("hd3")),
        )
        return make_gym_env(environment)

    model = SAC.load(str(checkpoint), device="cpu")
    rng = np.random.default_rng(args.seed)
    envs = [make_env() for _ in range(min(args.n_envs, args.rollouts))]
    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=len(envs)) as pool:
        while len(records) < args.rollouts:
            batch = envs[: args.rollouts - len(records)]
            records += rollout_batch(model, batch, rng, max_steps, pool)
            print(f"{len(records)}/{args.rollouts} rollouts", flush=True)

    feasible = [r for r in records if r["steps_to_feasible"] is not None]
    steps = np.array([r["steps_to_feasible"] for r in feasible], dtype=float)
    sac_rewards = np.concatenate([r["trace"] for r in records])
    best = max(records, key=lambda r: r["best_reward"])
    summary = {
        "checkpoint": str(checkpoint),
        "rollouts": len(records),
        "max_steps": max_steps,
        "feasible_rollouts": len(feasible),
        "feasibility_rate": len(feasible) / len(records),
        "steps_to_feasible_median": float(np.median(steps)) if steps.size else None,
        "steps_to_feasible_mean": float(steps.mean()) if steps.size else None,
        "steps_to_feasible_max": float(steps.max()) if steps.size else None,
        "total_evaluations": int(sac_rewards.size),
        "best_reward": best["best_reward"],
        "best_design": best["best_design"].tolist() if best["best_design"] is not None else None,
        "best_parameters": (EqualizerEvaluator(evaluator) if equalizer else evaluator).map_actions(best["best_design"]) if best["best_design"] is not None else None,
        "equalizer": equalizer,
        "best_metrics": best["best_metrics"],
        "model_source": evaluator.model_source,
        "eye_height_min_v": specifications.eye_vertical_min_v,
        "eye_width_min_ui": specifications.eye_horizontal_min_ui,
        "hd3_enforced": bool(config.get("hd3")),
    }

    with (output_dir / "rollouts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["rollout", "steps_to_feasible", "evaluations", "best_reward", "best_eye_height_v", "best_power"])
        for i, r in enumerate(records):
            writer.writerow([i, r["steps_to_feasible"], r["evaluations"], r["best_reward"], r["best_metrics"].get("eye_vertical_v"), r["best_metrics"].get("power")])

    if args.baseline:
        baseline_rows = list(csv.DictReader((Path(args.baseline) / "steps.csv").open(encoding="utf-8")))
        random_rewards = np.array([float(row["reward"]) for row in baseline_rows])
        budget = min(sac_rewards.size, random_rewards.size)
        sac_curve = best_reward_curve(sac_rewards[:budget])
        random_curve = best_reward_curve(random_rewards[:budget])
        summary["baseline"] = {
            "dir": args.baseline,
            "budget": int(budget),
            "sac_best_at_budget": float(sac_curve[-1]),
            "random_best_at_budget": float(random_curve[-1]),
            "sac_feasible_fraction": float(np.mean(sac_rewards[:budget] > 0)),
            "random_feasible_fraction": float(np.mean(random_rewards[:budget] > 0)),
            "evaluations_to_first_feasible": {
                "sac": int(np.argmax(sac_rewards[:budget] > 0) + 1) if np.any(sac_rewards[:budget] > 0) else None,
                "random": int(np.argmax(random_rewards[:budget] > 0) + 1) if np.any(random_rewards[:budget] > 0) else None,
            },
        }
        with (output_dir / "budget_curve.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["evaluations", "sac_best_reward", "random_best_reward"])
            for i in range(budget):
                writer.writerow([i + 1, sac_curve[i], random_curve[i]])
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            figure, axis = plt.subplots(figsize=(7, 4))
            axis.plot(np.arange(1, budget + 1), sac_curve, label="SAC policy (deterministic rollouts)")
            axis.plot(np.arange(1, budget + 1), random_curve, label="random search")
            axis.set_xlabel("simulated designs")
            axis.set_ylabel("best reward so far")
            axis.set_title(f"Best reward vs budget ({evaluator.model_source})")
            axis.grid(True, alpha=0.3)
            axis.legend()
            figure.tight_layout()
            figure.savefig(output_dir / "budget_curve.png", dpi=150)
            plt.close(figure)
        except ImportError:
            pass

    if not args.skip_validation and best["best_design"] is not None:
        full_design = best["best_design"]
        design = full_design[:5]
        equalizer_result = EqualizerEvaluator(evaluator).run(full_design) if equalizer else {}
        ac_result = evaluator.run_simulation(design)
        transient_result = evaluator.run_transient(design)
        linearity_result = evaluator.run_linearity(design)
        pvt_results = evaluator.run_pvt(design, all_pvt_corners())
        pvt_passed = sum(bool(row["pvt_pass"]) for row in pvt_results)
        summary["validation"] = {
            "dc_valid": ac_result["dc_valid"],
            "peaking_boost": ac_result["peaking_boost"],
            "power": ac_result["power"],
            "eye_height_v": transient_result["eye_height_v"],
            "eye_width_ui": transient_result["eye_width_ui"],
            "tran_valid": transient_result["tran_valid"],
            "hd3_db": linearity_result["hd3_db"],
            "hd3_pass": bool(linearity_result["linearity_valid"] and linearity_result["hd3_db"] < specifications.hd3_max_db),
            "dfe_tap": equalizer_result.get("dfe_tap"),
            "dfe_eye_height_v": equalizer_result.get("dfe_eye_height_v"),
            "pvt_pass_count": pvt_passed,
            "pvt_corner_count": len(pvt_results),
            "pvt_all_pass": pvt_passed == len(pvt_results) == 45,
        }
        ValidationReporter(output_dir).write(
            {**summary["validation"], "model_source": evaluator.model_source, "optimizer": "sac_eval", "selected_action": full_design.tolist(), "parameters": summary["best_parameters"], "channel_loss_db_at_nyquist": SpiceEvaluator.channel_loss_db(2.5e9)},
            ac_frequency_hz=ac_result["ac_frequency_hz"],
            ac_gain_db=ac_result["ac_gain_db"],
            transient_time_s=transient_result["time_s"],
            transient_output_v=transient_result["output_v"],
            pvt_results=pvt_results,
        )

    (output_dir / "evaluation.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("best_design", "best_metrics")}, indent=2, default=str))


if __name__ == "__main__":
    main()
