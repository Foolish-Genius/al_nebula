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
45-corner PVT. SAC training runs on the generic Level-1 model and on the
IHP PSP103 models. The first 5000-step Level-1 runs only matched random
search; with the margin bonus and hold-on-success episodes described below,
the Level-1 policy reaches a fully feasible design from any random start in
a median of 2.5 simulation steps (20/20 rollouts, 45/45 PVT), and the IHP
run is in progress.

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

Add `--hd3` to enforce the HD3 spec in the reward (the linearity gate runs
once the other specs pass) and `--equalizer` to size the whole equalizer:
the five CTLE values plus the one-tap DFE weight as a six-value action, with
the post-DFE eye driving the reward. To evaluate a checkpoint without
exploration noise and compare it with random search at equal budget. For the
PVT curriculum, resume the nominal policy with every episode at a random one
of the 45 corners (`--corners all --resume <checkpoint>`); to steer the reward
from plain-language feedback, write a settings file with
`scripts/reward_from_feedback.py "power matters more than eye margin"` and
pass it as `--reward-settings`. Every validation also writes the sized
netlist `sized_ctle.sp`. To evaluate and compare:

```bash
python scripts/baseline_random.py --evaluations 5000 --output-dir reports/baseline-5000
python scripts/evaluate_policy.py reports/sac-long --rollouts 20 --baseline reports/baseline-5000
```

`--n-envs` uses threads (`rl/threaded_vec_env.py`) rather than `SubprocVecEnv`:
ngspice runs in a subprocess that releases the GIL, and on Windows eight
spawned workers each loading CUDA torch fails DLL initialisation. Training is
simulator-bound; the GPU only runs the SAC networks.

The sg13g2 transistor deck of the IHP Open PDK uses PSP103, which ngspice
loads as OpenVAF-compiled OSDI models. Set `IHP_PDK_ROOT` to a PDK checkout
whose `ihp-sg13g2/libs.tech/ngspice/osdi/` holds `psp103.osdi`,
`psp103_nqs.osdi` and `mosvar.osdi` (compile them with
`libs.tech/verilog-a/openvaf-compile-va.sh`; on Windows without Visual
Studio's C++ tools see `tools/openvaf-link-shim/README.md`). Run
`python scripts/check_pdk.py` to check readiness. ngspice 44+ on Linux and
the ngspice 47 Windows build both load the models directly;
`SpiceEvaluator.for_model_source("ihp")` wires the corner libraries and OSDI
files, and every script takes `--model-source ihp --pdk-root <checkout>`.
The default `generic` runner uses ngspice's Level-1 model to validate the
architecture and data flow; reports label the model source. Xyce is not
required.
Run `python scripts/run_validation.py --output-dir reports/runs/latest` to
regenerate the complete artifact set without overwriting an earlier run. The
PVT CSV and graph contain all 45 simulated corners with pass/fail status. See
`papers/README.md` for the supplied paper's CTLE design takeaways and references.

For IHP PSP103 validation or training:

```bash
python scripts/run_validation.py --model-source ihp --pdk-root /path/to/ihp-open-pdk   --output-dir reports/runs/ihp-final
python scripts/train_sac.py --model-source ihp --eye-height-min 0.25 --n-envs 8 ... \
  --output-dir reports/sac-ihp
```

The eye targets in `CtleSpecifications` (0.5 V, 0.7 UI) were calibrated so
about one in ten random Level-1 designs passes; PSP103 devices have roughly
half the gain and none of 400 random designs reaches 0.5 V, so IHP training
uses `--eye-height-min 0.25` (same calibration, above the ~175 mV PCIe Gen 2
receiver eye).

Each ngspice process is pinned to one OpenMP thread (`set num_threads=1` in
the generated `.spiceinit`); ngspice ignores `OMP_NUM_THREADS`, and with
several simulators in flight its spin-waiting threads made PSP103 transients
sixty times slower. The generic Level-1 path remains available for fast
debugging.

On the IHP models the SAC design passes DC, AC peaking, the post-channel eye,
power, HD3, input-referred noise, the area estimate, and all 45 PVT corners.
HD3 was reported as failing (about -25 dB) until 2026-09-15; that was
spectral leakage from a 2.5-cycle rectangular FFT window in the measurement,
and an integer-cycle Hann window gives about -58 dB for the same designs.

`SpiceEvaluator.run_simulation()` implements the `.op` and `.ac` gates;
`run_transient()` implements the channel + PRBS eye gate plus the behavioural
one-tap DFE; `run_linearity()`, `run_noise()`, `estimate_area()`, and
`run_pvt()` cover HD3, noise, area, and the corner matrix. Set `NGSPICE` to the
ngspice executable (on Windows use `ngspice_con.exe`) or pass `--ngspice` to
the scripts. The active submission report is generated under
`reports/runs/ihp-submission/`.
