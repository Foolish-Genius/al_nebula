"""Run one candidate and write all available validation artifacts."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from analysis.reporting import ValidationReporter
from rl.pvt import all_pvt_corners
from rl.search import BoundedDesignSearch
from spice.spice_engine import SpiceEvaluator


def main() -> None:
    evaluator = SpiceEvaluator()
    action, search_rows = BoundedDesignSearch(evaluator, seed=7).run(evaluations=24)
    ac_result = evaluator.run_simulation(action)
    transient_result = evaluator.run_transient(action)
    metrics = {
        **ac_result,
        "tran_valid": transient_result["tran_valid"],
        "eye_height_v": transient_result["eye_height_v"],
        "transient_error": transient_result.get("error"),
        "model_source": "ngspice_generic_level1",
        "selected_action": action.tolist(),
    }
    reporter = ValidationReporter("reports")
    paths = reporter.write(
        metrics,
        transient_time_s=transient_result["time_s"],
        transient_output_v=transient_result["output_v"],
        pvt_results=[
            {"name": corner.name, "status": "not_run", "peaking_boost": float("nan")}
            for corner in all_pvt_corners()
        ],
    )
    paths.update(reporter.write_search(search_rows))
    print(metrics)
    print(paths)


if __name__ == "__main__":
    main()