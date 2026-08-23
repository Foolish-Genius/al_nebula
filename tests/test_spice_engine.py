import numpy as np
import pytest

from spice.spice_engine import SpiceEvaluator


def test_map_actions_hits_physical_bounds():
    evaluator = SpiceEvaluator()
    mapped = evaluator.map_actions(np.array([-1.0, 0.0, 1.0, -1.0, 1.0]))
    assert mapped["W_in"] == pytest.approx(0.5e-6)
    assert mapped["R_load"] == pytest.approx(550.0)
    assert mapped["I_bias"] == pytest.approx(2e-3)
    assert mapped["R_s"] == pytest.approx(10.0)
    assert mapped["C_s"] == pytest.approx(1e-12)


def test_map_actions_rejects_bad_shape_and_range():
    evaluator = SpiceEvaluator()
    with pytest.raises(ValueError):
        evaluator.map_actions(np.zeros(4))
    with pytest.raises(ValueError):
        evaluator.map_actions(np.array([0.0, 0.0, 0.0, 0.0, 1.1]))


def test_run_simulation_fails_fast_when_ngspice_is_unavailable():
    evaluator = SpiceEvaluator(ngspice_binary="definitely-not-ngspice")
    result = evaluator.run_simulation(np.zeros(5))
    assert result["dc_valid"] is False
    assert result["error"]


def test_mapping_rejects_nan_and_infinity():
    evaluator = SpiceEvaluator()
    with pytest.raises(ValueError):
        evaluator.map_actions(np.array([0.0, 0.0, np.nan, 0.0, 0.0]))
    with pytest.raises(ValueError):
        evaluator.map_actions(np.array([0.0, 0.0, np.inf, 0.0, 0.0]))


def test_prbs_source_is_deterministic():
    first = SpiceEvaluator._prbs_source(False)
    assert first == SpiceEvaluator._prbs_source(False)
    assert first.startswith("PWL(") and first.endswith(")")
    assert first.count(" ") > 200


def test_parsers_reject_malformed_data():
    with pytest.raises(ValueError):
        SpiceEvaluator._parse_ac_data("Index frequency value\n")
    with pytest.raises(ValueError):
        SpiceEvaluator._parse_transient("Index time outp outn\n")
    with pytest.raises(ValueError):
        SpiceEvaluator._parse_transient("0 1e-9 0.5 0.4\n1 0.0 0.6 0.3\n")
    with pytest.raises(ValueError):
        SpiceEvaluator._parse_ac_data("0 1e6 1.0 0.0\n1 1e6 1.0 0.0\n")


def test_osdi_paths_are_configured_without_legacy_model_include(tmp_path):
    model = tmp_path / "psp103.osdi"
    model.write_bytes(b"placeholder")
    evaluator = SpiceEvaluator(osdi_model_paths=(model,))
    rendered = evaluator._inject_parameters(evaluator.map_actions(np.zeros(5)))
    assert ".include" not in rendered
    assert evaluator.osdi_model_paths == (model,)


def test_osdi_injection_selects_ihp_subcircuits(tmp_path):
    model = tmp_path / "psp103.osdi"
    evaluator = SpiceEvaluator(osdi_model_paths=(model,))
    rendered = evaluator._inject_parameters(evaluator.map_actions(np.zeros(5)))
    assert "X1 outP inP sourceP 0 sg13_lv_nmos" in rendered
    assert "M1 outP inP sourceP 0 ctle_nmos" not in rendered
    assert ".model ctle_nmos" not in rendered


def test_successful_transient_result_has_error_field():
    class FakeEvaluator(SpiceEvaluator):
        def _run_ngspice(self, netlist, stem):
            return "Index time v(outp) v(outn)\n0 0 0.5 0.4\n1 1e-9 0.7 0.3\n"

    result = FakeEvaluator().run_transient(np.zeros(5))
    assert result["error"] is None
