"""Record real engine runs over a grid of specifications for the studio's replay mode.

Every entry is a genuine run: the trained policy, CMA-ES and random search on
the IHP models, then the 45-corner sweep. Each run is cached as it finishes, so
an interrupted build resumes where it stopped.

    python -m studio.build_library --workers 4
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import json
from pathlib import Path
import time

from studio.engine import ROOT, Engine, Spec

GRID = {
    "eye_height_v": (0.25, 0.30, 0.35, 0.40, 0.45),
    "power_mw": (2.0, 1.5, 1.2, 0.8),
    "eye_width_ui": (0.70, 0.85),
    "peaking_min_db": (3.0, 5.0),
}


def key_of(values: dict) -> str:
    return "e{eye_height_v:.2f}_p{power_mw:.1f}_w{eye_width_ui:.2f}_k{peaking_min_db:.0f}".format(**values)


def thin(event: dict) -> dict:
    """Drop what the replay page does not use; timing is normalised on the page."""
    event = {k: v for k, v in event.items() if k not in ("t", "design")}
    if event.get("type") == "done" and "best" in event:
        event["best"] = {k: v for k, v in event["best"].items() if k != "design"}
    return event


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=4, help="runs recorded in parallel")
    parser.add_argument("--cache", default="reports/studio-library")
    parser.add_argument("--output", default="docs/demo/studio_library.json")
    args = parser.parse_args()

    cache = ROOT / args.cache
    cache.mkdir(parents=True, exist_ok=True)
    engine = Engine()
    combos = [dict(zip(GRID, values)) for values in product(*GRID.values())]
    todo = [(index, values) for index, values in enumerate(combos) if not (cache / f"{key_of(values)}.json").exists()]
    print(f"{len(combos)} specifications, {len(combos) - len(todo)} cached, {len(todo)} to record", flush=True)

    def record(item):
        index, values = item
        events: list[dict] = []
        started = time.time()
        engine.run(Spec(**values), events.append, race=True, seed=1000 + index)
        path = cache / f"{key_of(values)}.json"
        path.write_text(json.dumps([thin(event) for event in events], separators=(",", ":")), encoding="utf-8")
        done = next(e for e in events if e["type"] == "done" and e["lane"] == "sac")
        print(f"{key_of(values)}  policy feasible at {done['first_feasible']}  ({time.time() - started:.0f}s)", flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(record, todo))

    # A partial library is useful too: the page falls back for a combination it has no run for.
    runs = {}
    for values in combos:
        path = cache / f"{key_of(values)}.json"
        if path.exists():
            runs[key_of(values)] = json.loads(path.read_text(encoding="utf-8"))
    missing = len(combos) - len(runs)
    if missing:
        print(f"note: {missing} specifications were not recorded; the page falls back for those")
    output = ROOT / args.output
    output.write_text(json.dumps({"grid": GRID, "runs": runs}, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {output} ({output.stat().st_size // 1024} KB, {len(runs)} runs)")


if __name__ == "__main__":
    main()
