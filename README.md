# al_nebula

Phase 1 of AutoAnalog-RL: a parameterized IHP sg13g2 CTLE netlist, fail-fast ngspice evaluator, and RL specification boundary.

## Architecture

- `netlists/` owns circuit topology and injectable SPICE parameters.
- `spice/` owns simulator execution, parsing, and the DC/AC/transient gates. The
  transient gate drives a 2-period PRBS7 at 5 Gbps through a lossy RC channel
  (about -10 dB at 2.5 GHz, see `SpiceEvaluator.CHANNEL_POLE_HZ`) and measures
  the eye by aligning the output to the transmitted bits: height is the largest
  `min(ones) - max(zeros)` over phase, width is the fraction of the UI that is
  open. The DC/AC gates bypass the channel so `peaking_boost` is the CTLE alone.
- `rl/specs.py` owns measurable targets and normalized constraint violations.
- `rl/reward.py` owns reward shaping and configurable weights, including a
  continuous power charge so the reward keeps a gradient inside the feasible set.
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
45-corner PVT. SAC training runs on the generic Level-1 model; the first
5000-step runs matched random search, and the reward/episode changes below
(margin bonus, hold-on-success) are the current fix under evaluation.

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

The defaults reproduce the first runs (terminate on the first feasible design,
one environment). For a converged policy use several environments and let the
episode continue after success so the agent is rewarded for spec margin:

```bash
python scripts/train_sac.py --timesteps 30000 --n-envs 8 --hold-on-success   --random-reset --margin-weight 5 --invalid-penalty -10 --max-steps 30   --batch-size 256 --gradient-steps -1 --ent-coef auto_0.1   --output-dir reports/sac-long
python scripts/sac_progress.py reports/sac-long
```

`--n-envs` uses threads (`rl/threaded_vec_env.py`) rather than `SubprocVecEnv`:
ngspice runs in a subprocess that releases the GIL, and on Windows eight
spawned workers each loading CUDA torch fails DLL initialisation. Training is
simulator-bound; the GPU only runs the SAC networks.

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

For final IHP PSP103 validation, use an OSDI-capable ngspice build and point
`--pdk-root` (or `IHP_PDK_ROOT`) at the IHP Open PDK checkout:

```bash
python scripts/run_validation.py --model-source ihp 	--ngspice /path/to/osdi-capable/ngspice 	--pdk-root /path/to/ihp-open-pdk 	--output-dir reports/runs/ihp-final
```

This uses OpenVAF-compiled `psp103.osdi` models with the sg13g2 MOS corner
libraries. The generic Level-1 path remains available for fast debugging.

The current real-IHP run passes DC, AC peaking, eye, power, area estimate,
input-referred noise, and all 45 PVT corners. HD3 remains the measured strict
failure and is the next optimization target.

`SpiceEvaluator.run_simulation()` implements the `.op` and `.ac` gates;
`run_transient()` implements the channel + PRBS eye gate plus the behavioural
one-tap DFE; `run_linearity()`, `run_noise()`, `estimate_area()`, and
`run_pvt()` cover HD3, noise, area, and the corner matrix. Set `NGSPICE` to the
ngspice executable (on Windows use `ngspice_con.exe`) or pass `--ngspice` to
the scripts. The active submission report is generated under
`reports/runs/ihp-submission/`.
