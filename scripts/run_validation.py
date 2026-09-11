"""Run one candidate and write all available validation artifacts."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from analysis.reporting import ValidationReporter
from rl.pvt import all_pvt_corners
from rl.search import BoundedDesignSearch
from spice.spice_engine import SpiceEvaluator


def main(output_dir: str = "reports") -> None:
    evaluator = SpiceEvaluator()
    action, search_rows = BoundedDesignSearch(evaluator, seed=23).run(evaluations=100)
    ac_result = evaluator.run_simulation(action)
    transient_result = evaluator.run_transient(action)
    pvt_results = evaluator.run_pvt(action, all_pvt_corners())
    pvt_simulated = sum(
        bool(row["dc_valid"]) and np.isfinite(row["peaking_boost"])
        for row in pvt_results
    )
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
        "transient_error": transient_result.get("error"),
        "model_source": "ngspice_generic_level1",
        "selected_action": action.tolist(),
        "search_evaluations": len(search_rows),
        "pvt_corner_count": len(pvt_results),
        "pvt_simulated_count": pvt_simulated,
        "pvt_pass_count": pvt_passed,
        "pvt_all_pass": len(pvt_results) == 45 and pvt_simulated == 45 and pvt_passed == 45,
    }
    reporter = ValidationReporter(output_dir)
    paths = reporter.write(
        metrics,
        transient_time_s=transient_result["time_s"],
        transient_output_v=transient_result["output_v"],
        ac_frequency_hz=ac_result["ac_frequency_hz"],
        ac_gain_db=ac_result["ac_gain_db"],
        pvt_results=pvt_results,
    )
    paths.update(reporter.write_search(search_rows))
    print(metrics)
    print(paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="reports")
    main(parser.parse_args().output_dir)