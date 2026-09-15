"""Run one candidate and write all available validation artifacts."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from analysis.reporting import ValidationReporter
from rl.pvt import all_pvt_corners
from rl.equalizer import EqualizerEvaluator
from rl.search import BoundedDesignSearch
from spice.spice_engine import SpiceEvaluator


def main(
    output_dir: str = "reports",
    model_source: str = "generic",
    ngspice: str | None = None,
    pdk_root: str | None = None,
) -> None:
    evaluator = SpiceEvaluator.for_model_source(model_source, ngspice_binary=ngspice, pdk_root=pdk_root)
    action, search_rows = BoundedDesignSearch(evaluator, seed=23).run(evaluations=100)
    ac_result = evaluator.run_simulation(action)
    transient_result = evaluator.run_transient(action)
    selected_tap = float(transient_result.get("dfe_tap", 0.0))
    whole_equalizer = EqualizerEvaluator(evaluator).run(np.r_[action, np.clip(2.0 * selected_tap, -1.0, 1.0)])
    linearity_result = evaluator.run_linearity(action)
    noise_result = evaluator.run_noise(action)
    area_result = evaluator.estimate_area(action)
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
        "eye_width_ui": transient_result.get("eye_width_ui"),
        "eye_height_pass": transient_result.get("eye_height_pass"),
        "eye_width_pass": transient_result.get("eye_width_pass"),
        "dfe_tap": transient_result.get("dfe_tap"),
        "dfe_eye_height_v": transient_result.get("dfe_eye_height_v"),
        "equalizer_tap": whole_equalizer.get("equalizer_tap"),
        "equalizer_eye_height_v": whole_equalizer.get("dfe_eye_height_v"),
        "equalizer_valid": whole_equalizer.get("equalizer_valid", False),
        "hd3_db": linearity_result["hd3_db"],
        "linearity_valid": linearity_result["linearity_valid"],
        "hd3_pass": bool(linearity_result["linearity_valid"] and linearity_result["hd3_db"] < -30.0),
        "noise_vrms": noise_result["noise_vrms"],
        "noise_valid": noise_result["noise_valid"],
        "noise_pass": bool(noise_result["noise_valid"] and noise_result["noise_vrms"] < 1.5e-3),
        **area_result,
        "eye_center_ui": transient_result.get("eye_center_ui"),
        "channel_loss_db_at_nyquist": evaluator.channel_loss_at_nyquist_db(),
        "transient_error": transient_result.get("error"),
        "model_source": evaluator.model_source,
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
        netlist=evaluator.sized_netlist(action, dfe_tap=selected_tap),
    )
    paths.update(reporter.write_search(search_rows))
    paths.update(reporter.write_dfe(transient_result.get("dfe_time_s", transient_result["time_s"]), transient_result.get("dfe_output_v", transient_result["output_v"])))
    print(metrics)
    print(paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--model-source", choices=("generic", "ihp"), default="generic")
    parser.add_argument("--ngspice", default=None, help="ngspice executable (default: $NGSPICE or ngspice on PATH)")
    parser.add_argument("--pdk-root", default=None, help="IHP Open PDK checkout (default: $IHP_PDK_ROOT)")
    arguments = parser.parse_args()
    main(arguments.output_dir, arguments.model_source, arguments.ngspice, arguments.pdk_root)
