"""Event-stream engine behind the studio page.

A run is a sequence of plain dict events. The live server forwards them as
Server-Sent Events and the replay-library builder records them with their
timestamps, so the page consumes exactly one format either way.

    start   the specification, corner and unsimulated starting design
    step    one simulation in one lane ("sac", "cma", "random"); the sac lane
            also carries the AC curve and the eye traces
    done    a lane finished: first feasible simulation, best design, and for an
            unreachable spec the binding constraint and the closest value reached
    pvt     one corner of the 45-corner AC sweep of the sac lane's best design
    end     the whole run, with the sized netlist
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rl.environment import CtleEnvironment  # noqa: E402
from rl.gym_wrapper import make_gym_env  # noqa: E402
from rl.pvt import all_pvt_corners  # noqa: E402
from scripts.evaluate_policy import latest_checkpoint  # noqa: E402
from scripts.train_sac import build_evaluator, build_reward  # noqa: E402

RUN_DIR = ROOT / "reports" / "sac-ihp-pvt"
PARAMS = ("W_in", "R_load", "I_bias", "R_s", "C_s")
CORNERS = {corner.name: corner for corner in all_pvt_corners()}
Emit = Callable[[dict], None]


@dataclass(frozen=True)
class Spec:
    """What the engineer asks for. Everything else comes from the trained run."""

    eye_height_v: float = 0.25
    eye_width_ui: float = 0.70
    power_mw: float = 2.0
    peaking_min_db: float = 3.0
    peaking_max_db: float = 12.0
    corner: str = "random"

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "Spec":
        fields = cls.__dataclass_fields__
        spec = {}
        for name, value in values.items():
            if name in fields and value not in (None, ""):
                spec[name] = str(value) if name == "corner" else float(value)
        return cls(**spec)

    def config(self, base: dict) -> dict:
        config = dict(base)
        config["eye_height_min"] = self.eye_height_v
        config["eye_width_min"] = self.eye_width_ui
        config["spec"] = [
            f"power_max_w={self.power_mw * 1e-3:.6g}",
            f"peaking_min_db={self.peaking_min_db:g}",
            f"peaking_max_db={self.peaking_max_db:g}",
        ]
        return config


class Recording:
    """Forwards to a SpiceEvaluator and keeps the last waveforms for the page."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.last_ac: dict = {}
        self.last_tran: dict = {}

    def run_simulation(self, design):
        self.last_ac, self.last_tran = self._inner.run_simulation(design), {}
        return self.last_ac

    def run_transient(self, design):
        self.last_tran = self._inner.run_transient(design)
        return self.last_tran

    def __getattr__(self, name):
        return getattr(self._inner, name)


class Engine:
    def __init__(self, run_dir: Path = RUN_DIR) -> None:
        from stable_baselines3 import SAC

        self.stop = threading.Event()

        self.base = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        self.checkpoint = latest_checkpoint(run_dir)
        self.model = SAC.load(str(self.checkpoint), device="cpu")
        self._model_lock = threading.Lock()

    # ---------------------------------------------------------------- helpers
    def _corner(self, spec: Spec, seed: int):
        if spec.corner in CORNERS:
            return CORNERS[spec.corner]
        names = sorted(CORNERS)
        return CORNERS[names[int(np.random.default_rng(seed).integers(len(names)))]]

    def _env(self, config: dict, corner, evaluator=None, **kwargs) -> CtleEnvironment:
        return CtleEnvironment(
            evaluator or build_evaluator(config),
            reward_model=build_reward(config),
            delta_scale=config["delta_scale"],
            run_transient=True,
            run_linearity=True,
            corners=(corner,),
            **kwargs,
        )

    @staticmethod
    def _metrics(info: dict) -> dict:
        keep = ("dc_valid", "peaking_boost", "power", "eye_vertical_v", "eye_horizontal_ui", "hd3_db")
        out = {}
        for key in keep:
            value = info["metrics"].get(key)
            if isinstance(value, (bool, np.bool_)):
                out[key] = bool(value)
            elif value is not None and np.isscalar(value) and np.isfinite(value):
                out[key] = round(float(value), 6)
            else:
                out[key] = None
        return out

    @staticmethod
    def _eye(tran: dict, ui: float, traces: int = 30, points: int = 48) -> list[list[int]]:
        """2-UI slices of the settled half of the output, in integer millivolts."""
        time_s = np.asarray(tran.get("time_s", []), dtype=float)
        output_v = np.asarray(tran.get("output_v", []), dtype=float)
        if time_s.size < 4:
            return []
        grid = np.linspace(0.0, 2.0, points)
        first = time_s.min() + 0.5 * (time_s.max() - time_s.min())
        out = []
        for start in np.arange(first, time_s.max() - 2 * ui, ui)[:traces]:
            out.append([int(round(1000 * v)) for v in np.interp(start + grid * ui, time_s, output_v)])
        return out

    @staticmethod
    def _ac(ac: dict, points: int = 60) -> dict:
        f = np.asarray(ac.get("ac_frequency_hz", []), dtype=float)
        g = np.asarray(ac.get("ac_gain_db", []), dtype=float)
        if f.size < 2:
            return {"f": [], "g": []}
        grid = np.logspace(np.log10(f.min()), np.log10(f.max()), points)
        return {"f": [float(f"{v:.4g}") for v in grid], "g": [round(float(v), 2) for v in np.interp(grid, f, g)]}

    @staticmethod
    def _params(evaluator, design) -> dict:
        return {k: float(f"{v:.5g}") for k, v in evaluator.map_actions(np.asarray(design)[:5]).items()}

    # ------------------------------------------------------------------ lanes
    def sac_lane(self, spec: Spec, seed: int, corner, emit: Emit, t0: float, start: np.ndarray,
                 max_sims: int = 12, polish: int = 2) -> dict:
        """The trained policy: first feasible design, then `polish` more steps growing the margin."""
        config = spec.config(self.base)
        recorder = Recording(build_evaluator(config))
        env = make_gym_env(self._env(config, corner, evaluator=recorder, max_steps=max_sims,
                                     random_reset=True, terminate_on_success=False))
        observation, _ = env.reset(seed=seed, options={"initial_design": start})
        ui = recorder.unit_interval_s
        best, first, sim = None, None, 0
        for sim in range(1, max_sims + 1):
            if self.stop.is_set():
                break
            with self._model_lock:
                action, _ = self.model.predict(observation, deterministic=True)
            observation, reward, _, _, info = env.step(action)
            feasible = bool(info["all_specs_met"])
            record = {
                "type": "step", "lane": "sac", "sim": sim, "t": round(time.perf_counter() - t0, 3),
                "design": [round(float(v), 4) for v in info["design"]],
                "params": self._params(recorder, info["design"]),
                "metrics": self._metrics(info),
                "violations": {k: round(float(v), 4) for k, v in info["violations"].items()},
                "reward": round(float(reward), 3), "feasible": feasible,
                "ac": self._ac(recorder.last_ac), "eye": self._eye(recorder.last_tran, ui),
            }
            emit(record)
            if best is None or record["reward"] > best["reward"]:
                best = record
            if feasible and first is None:
                first = sim
            if first is not None and sim >= first + polish:
                break
        if best is None:  # stopped before the first simulation
            summary = {"type": "done", "lane": "sac", "t": round(time.perf_counter() - t0, 3), "sims": 0, "first_feasible": None, "best": None}
            emit(summary)
            return summary
        summary = {"type": "done", "lane": "sac", "t": round(time.perf_counter() - t0, 3), "sims": sim,
                   "first_feasible": first, "best_sim": best["sim"],
                   "best": {k: best[k] for k in ("design", "params", "metrics", "reward", "feasible")},
                   "netlist": recorder.sized_netlist(np.asarray(best["design"])[:5])}
        if first is None:
            summary["binding"] = self._binding(spec, best)
        emit(summary)
        return summary

    def search_lane(self, lane: str, spec: Spec, seed: int, corner, emit: Emit, t0: float, x0: np.ndarray,
                    budget: int = 30) -> dict:
        """CMA-ES or uniform random search on the same reward, stopping at the first feasible design.

        Each lane gets one simulator, like the policy, so the race is equal in
        simulations and in wall-clock time.
        """
        config = spec.config(self.base)
        env = self._env(config, corner, action_mode="absolute", max_steps=1)
        rng = np.random.default_rng(seed + 17)
        optimizer = None
        if lane == "cma":
            import cma

            optimizer = cma.CMAEvolutionStrategy(x0, 0.5, {"bounds": [-1.0, 1.0], "popsize": 8, "seed": seed % 100000 + 1, "verbose": -9})
        sim, first, best = 0, None, None
        while sim < budget and first is None and not self.stop.is_set():
            batch = optimizer.ask() if optimizer else list(rng.uniform(-1.0, 1.0, (8, 5)))
            costs = []
            for design in batch:
                if self.stop.is_set():
                    break
                env.reset()
                _, reward, _, _, info = env.step(np.clip(design, -1.0, 1.0))
                sim += 1
                costs.append(-reward)
                feasible = bool(info["all_specs_met"])
                record = {"type": "step", "lane": lane, "sim": sim, "t": round(time.perf_counter() - t0, 3),
                          "metrics": self._metrics(info), "reward": round(float(reward), 3), "feasible": feasible}
                emit(record)
                if best is None or record["reward"] > best["reward"]:
                    best = record
                if feasible:
                    first = sim
                if first is not None or sim >= budget:
                    break
            if optimizer and len(costs) == len(batch):
                optimizer.tell(batch, costs)
        summary = {"type": "done", "lane": lane, "t": round(time.perf_counter() - t0, 3), "sims": sim, "first_feasible": first,
                   "best": {"metrics": best["metrics"], "reward": best["reward"]} if best else None}
        emit(summary)
        return summary

    def pvt_sweep(self, spec: Spec, design, emit: Emit, t0: float, workers: int = 8) -> int:
        """The 45-corner .op + .ac sweep of one design, streamed corner by corner."""
        config = spec.config(self.base)
        local = threading.local()

        def run(pair):
            index, corner = pair
            if not hasattr(local, "evaluator"):
                local.evaluator = build_evaluator(config)
            row = local.evaluator.run_pvt_corner(np.asarray(design)[:5], corner.process, corner.vdd, corner.temperature_c)
            peaking, power = row.get("peaking_boost", np.nan), row.get("power", np.nan)
            ok = bool(row.get("dc_valid") and np.isfinite(peaking) and spec.peaking_min_db <= peaking <= spec.peaking_max_db
                      and np.isfinite(power) and power <= spec.power_mw * 1e-3)
            return {"type": "pvt", "index": index, "name": corner.name, "pass": ok,
                    "peaking": round(float(peaking), 2) if np.isfinite(peaking) else None,
                    "power_mw": round(float(power) * 1e3, 3) if np.isfinite(power) else None}

        passed = 0
        with ThreadPoolExecutor(workers) as pool:
            for future in as_completed([pool.submit(run, pair) for pair in enumerate(all_pvt_corners())]):
                if self.stop.is_set():
                    continue
                row = future.result()
                row["t"] = round(time.perf_counter() - t0, 3)
                passed += row["pass"]
                emit(row)
        return passed

    @staticmethod
    def _binding(spec: Spec, best: dict) -> dict:
        """The spec that blocked feasibility: what was asked, what was reached, and an achievable ask."""
        names = {
            "eye_vertical_v": ("Eye height", "eye_vertical_v", "eye_height_v", "V", "min"),
            "eye_horizontal_ui": ("Eye width", "eye_horizontal_ui", "eye_width_ui", "UI", "min"),
            "power": ("Power", "power", "power_mw", "mW", "max"),
            "peaking_boost": ("HF peaking", "peaking_boost", "peaking_min_db", "dB", "min"),
            "peaking_ceiling": ("HF peaking ceiling", "peaking_boost", "peaking_max_db", "dB", "max"),
            "hd3": ("HD3", "hd3_db", None, "dB", "max"),
        }
        # A gate that never ran scores a full violation, so ignore anything unmeasured:
        # the binding spec is the worst one the simulator actually reported.
        measured = {name: value for name, value in best["violations"].items()
                    if best["metrics"].get(names.get(name, (None, name))[1]) is not None}
        worst = max(measured or best["violations"], key=lambda name: best["violations"][name])
        label, metric, field, unit, direction = names.get(worst, (worst, worst, None, "", "min"))
        reached = best["metrics"].get(metric)
        if metric == "power" and reached is not None:
            reached *= 1e3
        asked = getattr(spec, field) if field else None
        suggestion = None
        if reached is not None and field:
            step = 0.1 if unit in ("mW", "dB") else 0.01
            suggestion = (np.floor(reached / step) if direction == "min" else np.ceil(reached / step)) * step
            suggestion = round(float(suggestion), 2)
        return {"spec": worst, "label": label, "unit": unit, "asked": asked, "field": field, "suggest": suggestion,
                "reached": None if reached is None else round(float(reached), 3)}

    # -------------------------------------------------------------------- run
    def run(self, spec: Spec, emit: Emit, race: bool = True, seed: int | None = None) -> None:
        seed = int(np.random.default_rng().integers(1, 2**31 - 1)) if seed is None else int(seed)
        corner = self._corner(spec, seed)
        t0 = time.perf_counter()
        # One random start shared by the policy and CMA-ES, shown on the page before sim 1.
        start = np.random.default_rng(seed).uniform(-1.0, 1.0, 5)
        probe = build_evaluator(spec.config(self.base))
        emit({"type": "start", "t": 0.0, "spec": asdict(spec), "corner": corner.name, "seed": seed, "race": race,
              "start": {"design": [round(float(v), 4) for v in start], "params": self._params(probe, start)},
              "bounds": {k: [float(lo), float(hi)] for k, (lo, hi) in probe.bounds.items()},
              "checkpoint": self.checkpoint.name})
        lanes: dict[str, Any] = {}

        def policy_then_verify():
            lanes["sac"] = self.sac_lane(spec, seed, corner, emit, t0, start)
            if not self.stop.is_set() and lanes["sac"]["best"]:
                lanes["pvt"] = self.pvt_sweep(spec, lanes["sac"]["best"]["design"], emit, t0)

        threads = [threading.Thread(target=policy_then_verify)]
        if race:
            for lane in ("cma", "random"):
                threads.append(threading.Thread(target=lambda lane=lane: lanes.__setitem__(
                    lane, self.search_lane(lane, spec, seed, corner, emit, t0, x0=start))))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        emit({"type": "end", "t": round(time.perf_counter() - t0, 3), "stopped": self.stop.is_set(),
              "pvt_passed": lanes.get("pvt"), "pvt_total": len(CORNERS)})
