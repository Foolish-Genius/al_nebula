# al_nebula

Phase 1 of AutoAnalog-RL: a parameterized IHP sg13g2 CTLE netlist, fail-fast ngspice evaluator, and RL specification boundary.

## Architecture

- `netlists/` owns circuit topology and injectable SPICE parameters.
- `spice/` owns simulator execution, parsing, and DC/AC gating.
- `rl/specs.py` owns measurable targets and normalized constraint violations.
- `rl/reward.py` owns reward shaping and configurable weights.
- `rl/environment.py` owns the Gym-style `reset`/`step` contract. State is the
  normalized design vector; actions are deltas by default (`action_mode="absolute"`
  to replace it). Each step runs the DC/AC gate and, on success, the transient
  PRBS gate so eye metrics reach the reward.
- `rl/pvt.py` owns the deterministic 45-corner PVT verification matrix. With the
  generic Level-1 model, `SpiceEvaluator.GENERIC_PROCESS_MODELS` skews `vto`/`kp`
  per process corner (placeholder values, NMOS-only); with the PDK, corner names
  map to the `mos_tt`/`mos_ss`/... sections of `cornerMOSlv.lib`.

The pre-ML pipeline is complete through CTLE sizing, IHP PSP103 simulation,
PRBS transient/eye validation, behavioral one-tap DFE, HD3/noise/area hooks, and
45-corner PVT. SAC/RL training is the next phase.

HD3 and noise targets are recorded in `CtleSpecifications`; the linearity and
noise gates in `run_validation.py` report them. The simulator adapter should
remain unchanged while the scheduling and optimization layers are added.

## Run

Use an environment with `numpy`, `pytest`, and the `ngspice` executable available on `PATH`.

```bash
python -m pytest
```

To train the SAC agent install the optional extra and run the training script;
it writes checkpoints, a per-step CSV, and the same validation artifact set as
`run_validation.py` for the best design it found:

```bash
pip install -e .[rl]
python scripts/train_sac.py --timesteps 2000 --output-dir reports/sac
```

The IHP Open PDK is installed locally. Its sg13g2 transistor deck uses PSP103, which the packaged ngspice
binary does not support as a built-in model. The PDK includes Verilog-A sources
and an OpenVAF build script. The intended ngspice path is: install OpenVAF with
its LLVM 21.1 runtime, run `libs.tech/verilog-a/openvaf-compile-va.sh`, then
pass the generated `psp103.osdi` and `psp103_nqs.osdi` files using
`SpiceEvaluator(osdi_model_paths=(...))`. Run `python scripts/check_pdk.py` to
check readiness. The default runner uses an explicit ngspice Level-1 model to
validate the architecture and data flow; reports label this model source
clearly. Xyce is not required once the OSDI models are compiled.
Run `python scripts/run_validation.py --output-dir reports/runs/latest` to
regenerate the complete artifact set without overwriting an earlier run. The
PVT CSV and graph contain all 45 simulated corners with pass/fail status. See
`papers/README.md` for the supplied paper's CTLE design takeaways and references.

For final IHP PSP103 validation, use the local OSDI-capable ngspice build:

```bash
export LD_LIBRARY_PATH=/home/hp/miniconda3/envs/autoanalog/lib:/home/hp/ngspice-45.2/install/lib
python scripts/run_validation.py --model-source ihp \
	--ngspice-binary /home/hp/ngspice-45.2/install/bin/ngspice \
	--output-dir reports/runs/ihp-final
```

This uses OpenVAF-compiled `psp103.osdi` models with the sg13g2 MOS corner
libraries. The generic Level-1 path remains available for fast debugging.

The current real-IHP run passes DC, AC peaking, eye, power, area estimate,
input-referred noise, and all 45 PVT corners. HD3 remains the measured strict
failure and is the next optimization target.

`SpiceEvaluator` implements `.op`, `.ac`, transient, PVT, HD3, noise, and area
measurement gates. The active submission report is generated under
`reports/runs/ihp-submission/`.
