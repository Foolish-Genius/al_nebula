# al_nebula

Phase 1 of AutoAnalog-RL: a parameterized IHP sg13g2 CTLE netlist, fail-fast ngspice evaluator, and RL specification boundary.

## Architecture

- `netlists/` owns circuit topology and injectable SPICE parameters.
- `spice/` owns simulator execution, parsing, and DC/AC gating.
- `rl/specs.py` owns measurable targets and normalized constraint violations.
- `rl/reward.py` owns reward shaping and configurable weights.
- `rl/environment.py` owns the Gym-style `reset`/`step` contract.
- `rl/pvt.py` owns the deterministic 45-corner PVT verification matrix.

The next high-value stages are transient PRBS and eye metrics, PVT-aware simulator
injection, then SAC integration. The simulator adapter should remain unchanged
while the scheduling and optimization layers are added.

## Run

Use an environment with `numpy`, `pytest`, and the `ngspice` executable available on `PATH`.

```bash
python -m pytest
```

The IHP Open PDK is installed locally at `/home/hp/ihp-open-pdk` (revision
`22f2a25`). Its sg13g2 transistor deck uses PSP103, which the packaged ngspice
binary does not support as a built-in model. The PDK includes Verilog-A sources
and an OpenVAF build script. The intended ngspice path is: install OpenVAF with
its LLVM 21.1 runtime, run `libs.tech/verilog-a/openvaf-compile-va.sh`, then
pass the generated `psp103.osdi` and `psp103_nqs.osdi` files using
`SpiceEvaluator(osdi_model_paths=(...))`. Run `python scripts/check_pdk.py` to
check readiness. The default runner uses an explicit ngspice Level-1 model to
validate the architecture and data flow; reports label this model source
clearly. Xyce is not required once the OSDI models are compiled.
Run `python scripts/run_validation.py` to regenerate the complete artifact set;
the PVT CSV lists all 45 corners and marks unavailable simulator runs explicitly.

`SpiceEvaluator.run_simulation()` currently implements the Phase 1 `.op` and `.ac` gates. The transient PRBS gate remains a later pipeline stage.
