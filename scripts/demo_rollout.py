"""Live demo: watch a trained policy size the CTLE from a random starting design.

Loads a train_sac.py run, draws a random design (and optionally a random PVT
corner), and prints one line per simulation step - device values, the gate
results, and which specs pass - until every spec is met. Meant for a screen
recording; the whole thing takes a few seconds per step.

Example:
    python scripts/demo_rollout.py reports/sac-ihp-pvt --seed 7 --corners all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from rl.environment import CtleEnvironment
from rl.equalizer import EqualizerEvaluator
from rl.gym_wrapper import make_gym_env
from scripts.evaluate_policy import latest_checkpoint
from scripts.train_sac import build_evaluator, build_reward, build_specifications, curriculum_corners

GREEN, RED, DIM, BOLD, RESET = "\033[92m", "\033[91m", "\033[2m", "\033[1m", "\033[0m"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=None, help="starting design (default: random each time)")
    parser.add_argument("--corners", choices=("nominal", "all", "process"), default=None, help="default: the run's setting")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()
    # Classic Windows consoles (powershell.exe / cmd.exe) print ANSI codes literally;
    # Windows Terminal sets WT_SESSION. Colour only where it will render.
    import os
    if args.no_color or (os.name == "nt" and not os.environ.get("WT_SESSION")):
        global GREEN, RED, DIM, BOLD, RESET
        GREEN = RED = DIM = BOLD = RESET = ""

    from stable_baselines3 import SAC

    try:  # Windows consoles default to a legacy code page
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    checkpoint = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(run_dir)
    specs = build_specifications(config)
    evaluator = build_evaluator(config)
    if config.get("equalizer"):
        evaluator = EqualizerEvaluator(evaluator)
    environment = CtleEnvironment(
        evaluator,
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

    seed = args.seed if args.seed is not None else int(np.random.default_rng().integers(1_000_000))
    obs, _ = env.reset(seed=seed)
    corner = environment.current_corner.name if environment.current_corner else "TT_1.20V_27C (nominal)"
    print(f"{BOLD}AutoAnalog-RL policy rollout{RESET}  model={checkpoint.name}  specs: eye >= {specs.eye_vertical_min_v} V, width >= {specs.eye_horizontal_min_ui} UI, "
          f"peaking 3-12 dB, power <= {specs.power_max_w * 1e3:.0f} mW{', HD3 <= -30 dB' if specs.enforce_hd3 else ''}")
    print(f"start: random design (seed {seed}), corner {corner}\n")
    header = f"{'step':>4} {'W_in':>8} {'R_load':>7} {'I_bias':>7} {'R_s':>6} {'C_s':>6}  {'peak dB':>7} {'power mW':>8} {'eye V':>6} {'eye UI':>6}  {'HD3 dB':>7}  {'reward':>7}  specs"
    print(header)
    print(DIM + "-" * len(header) + RESET)
    started = time.time()
    for step in range(1, args.max_steps + 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        p = evaluator.map_actions(info["design"])
        m = info["metrics"]
        v = info["violations"]
        names = {"peaking_boost": "peak", "peaking_ceiling": "peak<12", "power": "power", "eye_horizontal_ui": "width", "eye_vertical_v": "eye", "hd3": "hd3"}
        names = {"peaking_boost": "peak", "peaking_ceiling": "p<12", "power": "pwr", "eye_horizontal_ui": "width", "eye_vertical_v": "eye", "hd3": "hd3"}
        marks = " ".join((GREEN + "ok:" if val <= 0 else RED + "--:") + names.get(k, k) + RESET for k, val in v.items())
        hd3 = m.get("hd3_db")
        line = (f"{step:>4} {p['W_in'] * 1e6:>7.2f}u {p['R_load']:>7.0f} {p['I_bias'] * 1e3:>6.2f}m {p['R_s']:>6.0f} {p['C_s'] * 1e15:>5.0f}f  "
                f"{m.get('peaking_boost', float('nan')):>7.2f} {m.get('power', float('nan')) * 1e3:>8.2f} {m.get('eye_vertical_v', float('nan')):>6.3f} {m.get('eye_horizontal_ui', float('nan')):>6.2f}  "
                f"{(f'{hd3:>7.1f}' if hd3 is not None else '      -')}  {reward:>7.2f}  {marks}")
        if not m.get("dc_valid", False):
            line = f"{step:>4} {p['W_in'] * 1e6:>7.2f}u {p['R_load']:>7.0f} {p['I_bias'] * 1e3:>6.2f}m {p['R_s']:>6.0f} {p['C_s'] * 1e15:>5.0f}f  {RED}DC operating point failed - fail-fast, no AC/transient run{RESET}"
        print(line, flush=True)
        if terminated:
            print(f"\n{GREEN}{BOLD}all specs met after {step} simulation step(s){RESET} in {time.time() - started:.1f} s of wall time")
            print("sized netlist header:")
            base = getattr(evaluator, "spice", evaluator)
            print("\n".join(base.sized_netlist(np.asarray(info['design'])[:5], dfe_tap=p.get("dfe_tap")).splitlines()[:8]))
            break
        if truncated:
            print(f"\n{RED}stopped after {step} steps without meeting every spec{RESET}")
            break


if __name__ == "__main__":
    main()
