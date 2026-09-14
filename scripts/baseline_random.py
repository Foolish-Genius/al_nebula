"""Random-search baseline for the SAC agent at an equal simulation budget.

Samples uniformly random designs, scores each with the same reward the RL
environment uses (DC/AC gate plus the channel + eye transient gate), and writes
the same ``steps.csv`` / validation artifacts as ``train_sac.py`` so the two
can be compared directly.

Example:
    python scripts/baseline_random.py --evaluations 5000 --output-dir reports/baseline-5000
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
from rl.pvt import all_pvt_corners
from spice.spice_engine import SpiceEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evaluations", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="reports/baseline")
    parser.add_argument("--ngspice", default=None, help="ngspice executable (default: $NGSPICE or ngspice on PATH)")
    parser.add_argument("--model-source", choices=("generic", "ihp"), default="generic")
    parser.add_argument("--pdk-root", default=None, help="IHP Open PDK checkout (default: $IHP_PDK_ROOT)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluator = SpiceEvaluator.for_model_source(args.model_source, ngspice_binary=args.ngspice, pdk_root=args.pdk_root)
    environment = CtleEnvironment(evaluator, action_mode="absolute", max_steps=1)
    rng = np.random.default_rng(args.seed)

    best_reward = -np.inf
    best_design: np.ndarray | None = None
    with (output_dir / "steps.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["timestep", "reward", "dc_valid", "peaking_boost", "power", "eye_vertical_v", "eye_horizontal_ui", "all_specs_met"])
        for step in range(1, args.evaluations + 1):
            environment.reset()
            _, reward, _, _, info = environment.step(rng.uniform(-1.0, 1.0, size=CtleEnvironment.ACTION_SIZE))
            metrics = info["metrics"]
            writer.writerow([
                step,
                reward,
                metrics.get("dc_valid"),
                metrics.get("peaking_boost"),
                metrics.get("power"),
                metrics.get("eye_vertical_v"),
                metrics.get("eye_horizontal_ui"),
                info["all_specs_met"],
            ])
            if reward > best_reward:
                best_reward = reward
                best_design = np.asarray(info["design"], dtype=np.float64)
            if step % 100 == 0:
                handle.flush()
                print(f"evaluation {step}/{args.evaluations} best reward {best_reward:.3f}", flush=True)

    if best_design is None:
        raise SystemExit("no evaluation completed; check that ngspice runs")

    print(f"best reward {best_reward:.3f} at design {best_design.tolist()}")

    ac_result = evaluator.run_simulation(best_design)
    transient_result = evaluator.run_transient(best_design)
    pvt_results = evaluator.run_pvt(best_design, all_pvt_corners())
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
        "transient_error": transient_result.get("error"),
        "model_source": evaluator.model_source,
        "optimizer": "random",
        "timesteps": args.evaluations,
        "best_training_reward": float(best_reward),
        "selected_action": best_design.tolist(),
        "parameters": evaluator.map_actions(best_design),
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
