"""Unit tests for the studio engine that do not need ngspice or a trained policy."""

import json

import numpy as np
import pytest

from studio.build_library import key_of, thin
from studio.engine import Engine, Spec


def test_spec_from_mapping_ignores_unknown_and_blank_query_keys():
    spec = Spec.from_mapping({"eye_height_v": "0.32", "power_mw": "1.2", "corner": "SS_0.95V_125C",
                              "race": "1", "eye_width_ui": ""})
    assert spec.eye_height_v == 0.32 and spec.power_mw == 1.2
    assert spec.corner == "SS_0.95V_125C"
    assert spec.eye_width_ui == Spec().eye_width_ui  # blank keeps the default


def test_spec_config_overrides_only_the_requested_targets():
    config = Spec(eye_height_v=0.3, power_mw=1.2, peaking_min_db=5).config({"model_source": "ihp", "spec": ["stale=1"]})
    assert config["model_source"] == "ihp"  # the trained run's settings survive
    assert config["eye_height_min"] == 0.3
    assert "power_max_w=0.0012" in config["spec"]
    assert "peaking_min_db=5" in config["spec"]
    assert not any(item.startswith("stale") for item in config["spec"])


def test_binding_ignores_gates_that_never_ran():
    """HD3 only runs once everything else passes, so an unmeasured HD3 must not be named."""
    best = {
        "violations": {"peaking_boost": 0.0, "peaking_ceiling": 0.0, "power": 0.0,
                       "eye_horizontal_ui": 0.0, "eye_vertical_v": 0.2467, "hd3": 1.0},
        "metrics": {"eye_vertical_v": 0.339, "eye_horizontal_ui": 0.90, "power": 0.0014,
                    "peaking_boost": 6.22, "hd3_db": None},
    }
    binding = Engine._binding(Spec(eye_height_v=0.45), best)
    assert binding["spec"] == "eye_vertical_v"
    assert binding["asked"] == 0.45 and binding["reached"] == 0.339
    assert binding["field"] == "eye_height_v"
    assert binding["suggest"] == pytest.approx(0.33)  # relaxed down to something reachable


def test_binding_rounds_a_max_constraint_upwards():
    best = {
        "violations": {"power": 0.35, "eye_vertical_v": 0.0, "eye_horizontal_ui": 0.0,
                       "peaking_boost": 0.0, "peaking_ceiling": 0.0, "hd3": 0.0},
        "metrics": {"eye_vertical_v": 0.33, "eye_horizontal_ui": 0.9, "power": 0.00108,
                    "peaking_boost": 5.5, "hd3_db": -64.0},
    }
    binding = Engine._binding(Spec(power_mw=0.8), best)
    assert binding["spec"] == "power" and binding["unit"] == "mW"
    assert binding["reached"] == pytest.approx(1.08)
    assert binding["suggest"] == pytest.approx(1.1)  # a budget the design would actually meet


def test_metrics_replace_non_finite_values_with_none():
    info = {"metrics": {"dc_valid": np.True_, "peaking_boost": np.float64(4.5), "power": float("nan"),
                        "eye_vertical_v": None, "eye_horizontal_ui": 0.86, "hd3_db": float("inf")}}
    metrics = Engine._metrics(info)
    assert metrics["dc_valid"] is True and isinstance(metrics["dc_valid"], bool)
    assert metrics["peaking_boost"] == 4.5
    assert metrics["power"] is None and metrics["hd3_db"] is None and metrics["eye_vertical_v"] is None
    json.dumps(metrics)  # the event stream must stay serialisable


def test_eye_slices_the_settled_half_into_two_unit_interval_traces():
    ui = 200e-12
    time_s = np.linspace(0.0, 20 * ui, 2000)
    output_v = 0.3 * np.sign(np.sin(2 * np.pi * time_s / (2 * ui)))
    traces = Engine._eye({"time_s": time_s, "output_v": output_v}, ui, traces=5, points=16)
    assert 0 < len(traces) <= 5
    assert all(len(trace) == 16 for trace in traces)
    assert all(isinstance(sample, int) for sample in traces[0])  # millivolts, small to send
    assert max(abs(sample) for sample in traces[0]) == pytest.approx(300, abs=1)


def test_eye_and_ac_return_empty_when_the_gate_did_not_run():
    assert Engine._eye({}, 200e-12) == []
    assert Engine._ac({}) == {"f": [], "g": []}


def test_ac_resamples_onto_a_log_grid_spanning_the_sweep():
    frequency = np.logspace(6, 10, 500)
    curve = Engine._ac({"ac_frequency_hz": frequency, "ac_gain_db": np.linspace(10.0, 4.0, 500)}, points=20)
    assert len(curve["f"]) == 20 and len(curve["g"]) == 20
    assert curve["f"][0] == pytest.approx(1e6, rel=1e-3)
    assert curve["f"][-1] == pytest.approx(1e10, rel=1e-3)
    assert curve["g"][0] > curve["g"][-1]


def test_library_key_is_stable_across_equal_but_differently_typed_specs():
    values = {"eye_height_v": 0.3, "power_mw": 1.2, "eye_width_ui": 0.7, "peaking_min_db": 3.0}
    assert key_of(values) == "e0.30_p1.2_w0.70_k3"
    assert key_of({**values, "eye_height_v": 0.30000001}) == key_of(values)


def test_thin_drops_what_replay_does_not_use():
    event = thin({"type": "done", "lane": "sac", "t": 12.5, "sims": 4, "first_feasible": 3,
                  "best": {"design": [0.1] * 5, "metrics": {"power": 0.001}, "reward": 20.1}})
    assert "t" not in event and "design" not in event["best"]
    assert event["first_feasible"] == 3 and event["best"]["reward"] == 20.1
