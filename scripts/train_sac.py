"""Train a SAC agent on the CTLE sizing environment and validate its best design.

Each environment step costs two or three ngspice runs (.op, .ac, and .tran when
the DC gate passes), so timesteps are expensive; the defaults are sized for a
first run on a laptop rather than a converged policy.

Example:
    python scripts/train_sac.py --timesteps 2000 --output-dir reports/sac
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from analysis.reporting import ValidationReporter
from rl.environment import CtleEnvironment
from rl.gym_wrapper import make_gym_env
from rl.reward import CtleReward
from rl.specs import CtleSpecifications
from rl.pvt import all_pvt_corners
from spice.spice_engine import SpiceEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timesteps", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=50, help="episode length")
    parser.add_argument("--delta-scale", type=float, default=0.2)
    parser.add_argument("--learning-starts", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-transient", action="store_true", help="skip the .tran gate (faster, eye specs unmeasured)")
    parser.add_argument("--random-reset", action="store_true", help="start each episode from a random design")
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--output-dir", default="reports/sac")
    parser.add_argument("--ngspice", default=None, help="ngspice executable (default: $NGSPICE or ngspice on PATH)")
    parser.add_argument("--model-source", choices=("generic", "ihp"), default="generic")
    parser.add_argument("--pdk-root", default=None, help="IHP Open PDK checkout (default: $IHP_PDK_ROOT)")
    parser.add_argument("--device", default="auto", help="torch device for SAC: auto, cpu, or cuda")
    parser.add_argument("--n-envs", type=int, default=1, help="parallel environments (threads, each driving its own ngspice)")
    parser.add_argument(
        "--hold-on-success",
        action="store_true",
        help="keep the episode running after all specs pass so feasible steps keep earning reward",
    )
    parser.add_argument("--invalid-penalty", type=float, default=-100.0, help="reward for a DC-invalid design")
    parser.add_argument("--success-bonus", type=float, default=20.0)
    parser.add_argument("--margin-weight", type=float, default=0.0, help="bonus per unit of tightest spec margin once feasible")
    parser.add_argument("--eye-height-min", type=float, default=None, help="eye height spec in V (default: CtleSpecifications)")
    parser.add_argument("--eye-width-min", type=float, default=None, help="eye width spec in UI (default: CtleSpecifications)")
    parser.add_argument("--hd3", action="store_true", help="enforce the HD3 spec: run the linearity gate once the other specs pass")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gradient-steps", type=int, default=1, help="-1 matches the number of env steps per rollout")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent-coef", default="auto", help="SAC entropy coefficient, e.g. auto, auto_0.1, or 0.05")
    parser.add_argument("--target-entropy", default="auto", help="SAC target entropy: auto or a float")
    return parser.parse_args()


def build_specifications(config: dict) -> CtleSpecifications:
    overrides = {
        key: config[name]
        for key, name in (("eye_vertical_min_v", "eye_height_min"), ("eye_horizontal_min_ui", "eye_width_min"))
        if config.get(name) is not None
    }
    return CtleSpecifications(enforce_hd3=bool(config.get("hd3")), **overrides)


def build_evaluator(config: dict) -> SpiceEvaluator:
    specs = build_specifications(config)
    return SpiceEvaluator.for_model_source(
        config["model_source"],
        ngspice_binary=config["ngspice"],
        pdk_root=config["pdk_root"],
        eye_height_min_v=specs.eye_vertical_min_v,
        eye_width_min_ui=specs.eye_horizontal_min_ui,
    )


def build_env(config: dict):
    """Build one gym env from plain config so the vec env factory only needs plain data."""
    evaluator = build_evaluator(config)
    reward = CtleReward(
        specifications=build_specifications(config),
        invalid_penalty=config["invalid_penalty"],
        success_bonus=config["success_bonus"],
        margin_weight=config["margin_weight"],
    )
    environment = CtleEnvironment(
        evaluator,
        reward_model=reward,
        max_steps=config["max_steps"],
        delta_scale=config["delta_scale"],
        run_transient=config["run_transient"],
        random_reset=config["random_reset"],
        terminate_on_success=config["terminate_on_success"],
        run_linearity=bool(config.get("hd3")),
    )
    return make_gym_env(environment)


def main() -> None:
    args = parse_args()
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import DummyVecEnv

        from rl.threaded_vec_env import ThreadedVecEnv
    except ImportError as error:
        raise SystemExit("install the 'rl' extra first: pip install -e .[rl]") from error

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env_config = {
        "ngspice": args.ngspice,
        "model_source": args.model_source,
        "pdk_root": args.pdk_root,
        "eye_height_min": args.eye_height_min,
        "eye_width_min": args.eye_width_min,
        "hd3": args.hd3,
        "invalid_penalty": args.invalid_penalty,
        "success_bonus": args.success_bonus,
        "margin_weight": args.margin_weight,
        "max_steps": args.max_steps,
        "delta_scale": args.delta_scale,
        "run_transient": not args.no_transient,
        "random_reset": args.random_reset,
        "terminate_on_success": not args.hold_on_success,
    }
    (output_dir / "config.json").write_text(json.dumps({**vars(args), **env_config}, indent=2), encoding="utf-8")
    evaluator = build_evaluator(env_config)
    specifications = build_specifications(env_config)
    env = make_vec_env(
        functools.partial(build_env, env_config),
        n_envs=args.n_envs,
        seed=args.seed,
        monitor_dir=str(output_dir),
        vec_env_cls=ThreadedVecEnv if args.n_envs > 1 else DummyVecEnv,
    )

    class BestDesignCallback(BaseCallback):
        """Track the best-rewarded design seen during training and log every step."""

        def __init__(self) -> None:
            super().__init__()
            self.best_reward = -np.inf
            self.best_design: np.ndarray | None = None
            self.best_metrics: dict = {}
            self.handle = (output_dir / "steps.csv").open("w", newline="", encoding="utf-8")
            self.writer = csv.writer(self.handle, lineterminator="\n")
            self.writer.writerow(["timestep", "reward", "dc_valid", "peaking_boost", "power", "eye_vertical_v", "eye_horizontal_ui", "all_specs_met"])

        def _on_step(self) -> bool:
            for reward, info in zip(self.locals["rewards"], self.locals["infos"]):
                metrics = info.get("metrics", {})
                self.writer.writerow([
                    self.num_timesteps,
                    float(reward),
                    metrics.get("dc_valid"),
                    metrics.get("peaking_boost"),
                    metrics.get("power"),
                    metrics.get("eye_vertical_v"),
                    metrics.get("eye_horizontal_ui"),
                    info.get("all_specs_met"),
                ])
                if reward > self.best_reward:
                    self.best_reward = float(reward)
                    self.best_design = np.asarray(info["design"], dtype=np.float64)
                    self.best_metrics = {key: value for key, value in metrics.items() if np.isscalar(value) or value is None}
            self.handle.flush()
            return True

        def _on_training_end(self) -> None:
            self.handle.close()

    best = BestDesignCallback()
    checkpoints = CheckpointCallback(save_freq=args.checkpoint_every, save_path=str(output_dir / "checkpoints"), name_prefix="sac")

    target_entropy = "auto" if args.target_entropy == "auto" else float(args.target_entropy)
    model = SAC(
        "MlpPolicy",
        env,
        seed=args.seed,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        train_freq=1,
        gradient_steps=args.gradient_steps,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        target_entropy=target_entropy,
        device=args.device,
        verbose=1,
    )
    model.learn(total_timesteps=args.timesteps, callback=[best, checkpoints])
    model.save(str(output_dir / "sac_final"))

    if best.best_design is None:
        raise SystemExit("no environment step completed; check that ngspice runs")

    print(f"best training reward {best.best_reward:.3f} at design {best.best_design.tolist()}")

    # Validate the best design the same way run_validation.py does, PVT included.
    ac_result = evaluator.run_simulation(best.best_design)
    transient_result = evaluator.run_transient(best.best_design)
    linearity_result = evaluator.run_linearity(best.best_design)
    pvt_results = evaluator.run_pvt(best.best_design, all_pvt_corners())
    pvt_passed = sum(bool(row["pvt_pass"]) for row in pvt_results)
    metrics = {
        "dc_valid": ac_result["dc_valid"],
        "dc_gain": ac_result["dc_gain"],
        "nyquist_gain": ac_result["nyquist_gain"],
        "peaking_boost": ac_result["peaking_boost"],
        "power": ac_result["power"],
        "error": ac_result.get("error"),
        "tran_valid": transient_result["tran_valid"],
        "eye_height_v": transient_result["eye_height_v"],
        "eye_width_ui": transient_result["eye_width_ui"],
        "eye_center_ui": transient_result["eye_center_ui"],
        "channel_loss_db_at_nyquist": SpiceEvaluator.channel_loss_db(2.5e9),
        "hd3_db": linearity_result["hd3_db"],
        "hd3_pass": bool(linearity_result["linearity_valid"] and linearity_result["hd3_db"] < specifications.hd3_max_db),
        "hd3_enforced": args.hd3,
        "transient_error": transient_result.get("error"),
        "model_source": evaluator.model_source,
        "optimizer": "sac",
        "device": str(model.device),
        "timesteps": args.timesteps,
        "n_envs": args.n_envs,
        "hold_on_success": args.hold_on_success,
        "margin_weight": args.margin_weight,
        "eye_height_min_v": specifications.eye_vertical_min_v,
        "eye_width_min_ui": specifications.eye_horizontal_min_ui,
        "best_training_reward": best.best_reward,
        "selected_action": best.best_design.tolist(),
        "parameters": evaluator.map_actions(best.best_design),
        "pvt_corner_count": len(pvt_results),
        "pvt_pass_count": pvt_passed,
        "pvt_all_pass": pvt_passed == len(pvt_results) == 45,
    }
    paths = ValidationReporter(output_dir).write(
        metrics,
        ac_frequency_hz=ac_result["ac_frequency_hz"],
        ac_gain_db=ac_result["ac_gain_db"],
        transient_time_s=transient_result["time_s"],
        transient_output_v=transient_result["output_v"],
        pvt_results=pvt_results,
    )
    print(json.dumps({key: value for key, value in metrics.items() if key != "selected_action"}, indent=2, default=str))
    print(paths)


if __name__ == "__main__":
    main()
