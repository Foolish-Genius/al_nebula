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
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from analysis.reporting import ValidationReporter
from rl.environment import CtleEnvironment
from rl.gym_wrapper import make_gym_env
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
    except ImportError as error:
        raise SystemExit("install the 'rl' extra first: pip install -e .[rl]") from error

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluator = SpiceEvaluator(ngspice_binary=args.ngspice)
    environment = CtleEnvironment(
        evaluator,
        max_steps=args.max_steps,
        delta_scale=args.delta_scale,
        run_transient=not args.no_transient,
        random_reset=args.random_reset,
    )
    env = Monitor(make_gym_env(environment), filename=str(output_dir / "monitor"))

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

    model = SAC(
        "MlpPolicy",
        env,
        seed=args.seed,
        learning_starts=args.learning_starts,
        batch_size=64,
        train_freq=1,
        gradient_steps=1,
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
        "transient_error": transient_result.get("error"),
        "model_source": "ngspice_generic_level1",
        "optimizer": "sac",
        "timesteps": args.timesteps,
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
