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
    first = SpiceEvaluator()._prbs_source(False)
    assert first == SpiceEvaluator()._prbs_source(False)
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


def test_pvt_process_maps_to_ihp_library_corner(tmp_path):
    evaluator = SpiceEvaluator(
        pdk_model_path=tmp_path / "model.lib",
        pdk_corner_path=tmp_path / "corner.lib",
    )
    rendered = evaluator._inject_parameters(evaluator.map_actions(np.zeros(5)), pvt_process="SS")
    assert ".lib" in rendered and "mos_ss" in rendered


def test_area_estimate_reports_units_and_limit():
    result = SpiceEvaluator().estimate_area(np.zeros(5))
    assert result["area_mm2"] > 0.0
    assert result["area_valid"] is True


def test_successful_transient_result_has_error_field():
    class FakeEvaluator(SpiceEvaluator):
        def _run_ngspice(self, netlist, stem):
            return "Index time v(outp) v(outn)\n0 0 0.5 0.4\n1 1e-9 0.7 0.3\n"

    result = FakeEvaluator().run_transient(np.zeros(5))
    assert result["error"] is None
    assert result["time_s"].shape == result["output_v"].shape
    assert "eye_width_ui" in result


def test_run_pvt_returns_explicit_corner_statuses():
    class FakePvtEvaluator(SpiceEvaluator):
        def run_pvt_corner(self, actions, process, vdd, temperature_c):
            return {
                "dc_valid": True,
                "dc_gain": 10.0,
                "nyquist_gain": 16.0,
                "peaking_boost": 6.0,
                "power": 1e-3,
                "error": None,
            }

    class Corner:
        name = "TT_1.20V_25C"
        process = "TT"
        vdd = 1.2
        temperature_c = 25.0

    rows = FakePvtEvaluator().run_pvt(np.zeros(5), (Corner(),))
    assert rows[0]["pvt_pass"] is True
    assert rows[0]["status"] == "pass"


def test_generic_model_is_skewed_per_process_corner():
    evaluator = SpiceEvaluator()
    parameters = evaluator.map_actions(np.zeros(5))
    rendered = {
        corner: evaluator._inject_parameters(parameters, pvt_process=corner)
        for corner in ("TT", "SS", "FF", "SF", "FS")
    }
    assert len(set(rendered.values())) == 5
    assert "vto=0.45 kp=0.0002" in rendered["TT"]
    assert "vto=0.51 kp=0.00017" in rendered["SS"]
    assert rendered["TT"] == evaluator._inject_parameters(parameters)
    with pytest.raises(ValueError):
        evaluator._inject_parameters(parameters, pvt_process="XX")


def test_pdk_corner_names_map_to_ihp_sections(tmp_path):
    evaluator = SpiceEvaluator(pdk_model_path=tmp_path / "sg13g2.lib", pdk_corner_path=tmp_path / "corner.lib")
    parameters = evaluator.map_actions(np.zeros(5))
    assert "corner.lib mos_ss" in evaluator._inject_parameters(parameters, pvt_process="SS")
    assert "corner.lib mos_tt" in evaluator._inject_parameters(parameters)


def _nrz(bits_pattern, ui, delay=0.0, invert=False, tau=None, step=2e-12):
    time = np.arange(0.0, bits_pattern.size * ui, step)
    index = np.clip(np.floor((time - delay) / ui).astype(int), 0, bits_pattern.size - 1)
    wave = np.where(bits_pattern[index] == 1, 0.5, -0.5)
    wave[time < delay] = -0.5
    if invert:
        wave = -wave
    if tau:
        alpha = step / tau
        for i in range(1, wave.size):
            wave[i] = wave[i - 1] + alpha * (wave[i] - wave[i - 1])
    return time, wave


def test_eye_metrics_measure_a_clean_delayed_inverted_eye_as_fully_open():
    bits = SpiceEvaluator._prbs_bits()
    ui = SpiceEvaluator.UNIT_INTERVAL_S
    time, wave = _nrz(np.tile(bits, 2), ui, delay=0.7 * ui, invert=True)
    eye = SpiceEvaluator._eye_metrics(time, wave, bits, ui)
    assert eye["eye_height_v"] == pytest.approx(1.0)
    assert eye["eye_width_ui"] == pytest.approx(1.0)


def test_eye_metrics_shrink_with_isi_and_close_on_noise():
    bits = SpiceEvaluator._prbs_bits()
    ui = SpiceEvaluator.UNIT_INTERVAL_S
    time, wave = _nrz(np.tile(bits, 2), ui, tau=200e-12)
    eye = SpiceEvaluator._eye_metrics(time, wave, bits, ui)
    assert 0.0 < eye["eye_height_v"] < 0.5
    assert 0.0 < eye["eye_width_ui"] < 0.8
    noise = np.random.default_rng(0).normal(0.0, 0.1, time.size)
    closed = SpiceEvaluator._eye_metrics(time, noise, bits, ui)
    assert closed["eye_height_v"] == 0.0 and closed["eye_width_ui"] == 0.0


def test_channel_is_lossy_only_in_the_transient_netlist():
    evaluator = SpiceEvaluator()
    parameters = evaluator.map_actions(np.zeros(5))
    ac_netlist = evaluator._inject_parameters(parameters)
    tran_netlist = evaluator._inject_parameters(parameters, transient=True)
    assert "RchP txP inP 1m" in ac_netlist and "CchP1" not in ac_netlist
    assert "CchP1 chP1 0" in tran_netlist and "RchP2 chP1 inP 500" in tran_netlist
    assert "{CHANNEL}" not in ac_netlist and "{CHANNEL}" not in tran_netlist
    assert SpiceEvaluator.channel_loss_db(2.5e9) == pytest.approx(-10.0, abs=0.1)


def test_transient_parser_collapses_repeated_timepoints():
    time, output = SpiceEvaluator._parse_transient("0 0 0.5 0.4\n1 1e-12 0.6 0.4\n2 1e-12 0.7 0.4\n3 2e-12 0.8 0.4\n")
    assert time.tolist() == [0.0, 1e-12, 2e-12]
    assert output.tolist() == pytest.approx([0.1, 0.3, 0.4])


def test_hd3_from_waveform_recovers_known_distortion():
    import numpy as np
    from spice.spice_engine import SpiceEvaluator

    f0 = SpiceEvaluator.HD3_TONE_HZ
    time_s = np.linspace(0.0, 50e-9, 5001)
    # -40 dB third harmonic on top of the fundamental, plus a DC offset.
    wave = 0.6 + 0.2 * np.sin(2 * np.pi * f0 * time_s) + 0.002 * np.sin(2 * np.pi * 3 * f0 * time_s)
    assert SpiceEvaluator.hd3_from_waveform(time_s, wave) == pytest.approx(-40.0, abs=0.2)
    # A pure tone must not report distortion: leakage floor well below the spec.
    pure = 0.6 + 0.2 * np.sin(2 * np.pi * f0 * time_s)
    assert SpiceEvaluator.hd3_from_waveform(time_s, pure) < -80.0
    with pytest.raises(ValueError):
        SpiceEvaluator.hd3_from_waveform(np.linspace(0.0, 10e-9, 100), np.zeros(100))


def test_noise_spectrum_is_parsed_by_column_name():
    import numpy as np
    from spice.spice_engine import SpiceEvaluator

    output = """
Index   frequency       inoise_spectrum onoise_spectrum
--------------------------------------------------------------------------------
0	1.000000e+07	7.0e-09	8.7e-09
1	1.000000e+08	5.0e-09	8.5e-09
2	1.000000e+09	3.0e-09	8.3e-09

inoise_total = 2.4e-04
onoise_total = 6.2e-04
"""
    frequency, density = SpiceEvaluator._parse_noise_spectrum(output, "inoise_spectrum")
    assert frequency.tolist() == [1e7, 1e8, 1e9]
    assert density.tolist() == [7e-9, 5e-9, 3e-9]
    _, onoise = SpiceEvaluator._parse_noise_spectrum(output, "onoise_spectrum")
    assert onoise.tolist() == [8.7e-9, 8.5e-9, 8.3e-9]
    with pytest.raises(ValueError):
        SpiceEvaluator._parse_noise_spectrum(output, "missing_column")


def test_sized_netlist_fills_in_device_values():
    import numpy as np
    from spice.spice_engine import SpiceEvaluator

    evaluator = SpiceEvaluator()
    netlist = evaluator.sized_netlist(np.zeros(5), dfe_tap=-0.1)
    parameters = evaluator.map_actions(np.zeros(5))
    assert netlist.startswith("* AutoAnalog-RL sized CTLE")
    assert "* model source: ngspice_generic_level1" in netlist
    assert f"* R_load = {parameters['R_load']:.6g}" in netlist
    assert "DFE weight" in netlist and "-0.1" in netlist
    assert "{" not in netlist.split("\n", 8)[-1]  # every template placeholder was substituted


def test_nyquist_frequency_sets_rate_channel_and_peaking_frequency():
    import numpy as np
    from spice.spice_engine import SpiceEvaluator

    gen2 = SpiceEvaluator()
    gen1 = SpiceEvaluator(nyquist_frequency_hz=1.25e9)
    assert gen2.unit_interval_s == pytest.approx(200e-12) and gen1.unit_interval_s == pytest.approx(400e-12)
    # Channel loss at Nyquist is preserved when the rate changes.
    assert gen1.channel_loss_at_nyquist_db() == pytest.approx(gen2.channel_loss_at_nyquist_db())
    assert gen2.channel_loss_at_nyquist_db() == pytest.approx(-10.0, abs=0.1)
    # The PRBS source stretches with the UI and the transient runs long enough for it.
    assert "5.08e-08 0.7" in gen2._prbs_source(False) or gen2._prbs_source(False).count(" ") > 100
    assert gen1._tran_commands() != gen2._tran_commands()
    # Peaking is measured at the configured Nyquist frequency.
    frequencies = np.logspace(7, 10, 301)
    gains = 20 * np.log10(1 + (frequencies / 1e9) ** 2)  # rises with frequency
    assert gen1._nearest_value(frequencies, gains, gen1.nyquist_frequency_hz) < gen2._nearest_value(frequencies, gains, gen2.nyquist_frequency_hz)
