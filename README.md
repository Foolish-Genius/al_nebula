
# al_nebula

 AutoAnalog-RL: a parameterized IHP sg13g2 CTLE design and optimization framework combining SPICE simulation, PRBS eye validation, behavioral DFE, PVT verification, and reinforcement-learning-based equalizer sizing.

---

## Key Components

- **Netlists:**
- Parameterized CTLE SPICE topology.
- Injectable transistor and circuit parameters for automated sizing.
- Generated `sized_ctle.sp` artifacts for validated designs.
- **SPICE Evaluation:**
  - `SpiceEvaluator` provides the simulator abstraction and fail-fast evaluation pipeline.
  - DC and AC gates validate operating point and CTLE peaking.
  - Transient evaluation drives a 2-period PRBS7 pattern at 5 Gbps through a lossy RC channel.
  - Eye metrics are extracted by aligning the received waveform with the transmitted bits.
  - Supports both the generic Level-1 model and IHP PSP103 models.
- **RL Specification Layer:**
  - `rl/specs.py` defines measurable CTLE design specifications.
  - Specifications are converted into normalized constraint violations.
  - Eye height, eye width, power, peaking, HD3, noise, and area constraints are represented at the optimization boundary.
- **Reward System:**
  - `rl/reward.py` implements configurable reward shaping.
  - Supports weighted specification objectives and continuous power penalties.
  - Margin bonuses provide a gradient after a design becomes feasible.
  - Natural-language feedback can be converted into reward settings.
- **RL Environment:**
  - `rl/environment.py` provides the Gym-style `reset()` / `step()` interface.
  - The state is the normalized CTLE design vector.
  - Actions are parameter deltas by default.
  - `action_mode="absolute"` allows direct parameter replacement.
  - Each action executes the DC/AC gates and, when valid, the transient PRBS eye evaluation.
- **PVT Verification:**
  - `rl/pvt.py` implements deterministic 45-corner verification.
  - Generic Level-1 models perturb `vto` and `kp` for process-corner analysis.
  - IHP PDK operation maps corner names to the `mos_tt`, `mos_ss`, and related sections of `cornerMOSlv.lib`.
- **Linearity, Noise, and Area:**
  - `run_linearity()` evaluates HD3.
  - `run_noise()` evaluates input-referred noise.
  - `estimate_area()` provides the area estimate used by validation and optimization.
  - These metrics form additional specification gates around the CTLE design.
- **Equalizer Optimization:**
  - The CTLE can be optimized independently or jointly with a behavioral one-tap DFE.
  - With `--equalizer`, the optimization vector contains five CTLE parameters plus the DFE weight.
  - The post-DFE eye is used to evaluate the complete equalizer.

---

## Design and Evaluation Flow

 The complete evaluation pipeline is:

```
CTLE Parameters
      │
      ▼
Parameterized SPICE Netlist
      │
      ▼
   DC Gate
      │
      ▼
   AC Gate
      │
      ▼
Lossy RC Channel + PRBS7
      │
      ▼
   Eye Analysis
      │
      ├── Eye Height
      └── Eye Width
      │
      ▼
 Behavioral One-Tap DFE
      │
      ▼
 HD3 / Noise / Area
      │
      ▼
 45-Corner PVT
      │
      ▼
Specification / Reward
      │
      ▼
 SAC / Optimization
```

 The transient gate drives a 2-period PRBS7 waveform at 5 Gbps through a lossy RC channel. The channel is approximately `-10 dB` at `2.5 GHz`, controlled by `SpiceEvaluator.CHANNEL_POLE_HZ`.

 The eye is measured after aligning the received waveform against the transmitted bits. Eye height is calculated as the largest:

```
min(ones) - max(zeros)
```

 over sampling phase, while eye width is the fraction of the unit interval that remains open.

 The DC and AC gates bypass the channel, allowing `peaking_boost` to represent CTLE-only behavior.

---

## Reinforcement Learning

 The RL pipeline uses a Gym-style environment in `rl/environment.py`.

```
state = env.reset()
next_state, reward, terminated, truncated, info = env.step(action)
```

 The environment state is the normalized CTLE design vector.

 By default, actions represent parameter deltas:

```
new_design = current_design + action
```

 Absolute parameter control can instead be selected with:

```
action_mode="absolute"
```

 The optimization loop evaluates the design through the SPICE gates before calculating the reward.

 The SAC training pipeline supports:

- Multiple simulator-backed environments.
- Random initial designs.
- Hold-on-success episodes.
- Margin-based reward shaping.
- Configurable invalid-design penalties.
- Automatic entropy tuning.
- PVT curriculum training.
- Joint CTLE + DFE optimization.
- Reward settings generated from natural-language feedback.

---

## Reward Shaping

 The reward implementation in `rl/reward.py` combines specification satisfaction with continuous design objectives.

 The reward can include:

- Eye-height margin.
- Eye-width margin.
- Power consumption.
- AC peaking.
- HD3.
- Noise.
- Area.
- Constraint violations.
- Feasibility bonuses.

 A continuous power charge is maintained even after a design becomes feasible so that the agent retains a useful optimization gradient inside the feasible region.

 The margin bonus can similarly distinguish between merely passing a specification and providing additional design margin.

 Natural-language feedback can be converted into reward configuration:

```
python scripts/reward_from_feedback.py \
  "power matters more than eye margin"
```

 The generated settings file can then be supplied to SAC using:

```
--reward-settings <settings-file>
```

---

## PVT Verification

 The PVT subsystem evaluates the design across a deterministic 45-corner matrix.

 For the generic Level-1 model, process variation is represented by modifying:

```
vto
kp
```

 according to the process corner.

 For IHP PSP103 operation, the evaluator uses the corner definitions supplied by the IHP Open PDK and maps the corner names to the corresponding sections of:

```
cornerMOSlv.lib
```

 A nominal SAC policy can also be evaluated using randomized PVT corners:

```
python scripts/train_sac.py \
  --corners all \
  --resume <checkpoint>
```

 This provides a PVT curriculum in which each episode begins at one of the available process corners.

---

## Quick Start

### 1\. Environment

 Use an environment containing:

```
numpy
pytest
ngspice
```

 with the `ngspice` executable available on `PATH`.

 Run the complete test suite:

```
python -m pytest
```

---

### 2\. Run Validation

 The complete validation pipeline can be executed with:

```
python scripts/run_validation.py \
  --output-dir reports/runs/latest
```

 The validation generates the complete artifact set without overwriting an earlier run.

 The PVT report contains all 45 simulated corners with pass/fail status.

 Every validation also writes:

```
sized_ctle.sp
```

 containing the final sized CTLE netlist.

---

### 3\. Train SAC

 Install the optional RL dependencies:

```
pip install -e .[rl]
```

 Run a basic SAC experiment:

```
python scripts/train_sac.py \
  --timesteps 2000 \
  --output-dir reports/sac
```

 The training run produces checkpoints, per-step CSV data, and validation artifacts for the best design found.

 For longer training with multiple simulator environments:

```
python scripts/train_sac.py \
  --timesteps 30000 \
  --n-envs 8 \
  --hold-on-success \
  --random-reset \
  --margin-weight 5 \
  --invalid-penalty -10 \
  --max-steps 30 \
  --batch-size 256 \
  --gradient-steps -1 \
  --ent-coef auto_0.1 \
  --output-dir reports/sac-long
```

 Training progress can then be inspected with:

```
python scripts/sac_progress.py reports/sac-long
```

---

## Baselines and Policy Evaluation

 Random-search performance can be measured using:

```
python scripts/baseline_random.py \
  --evaluations 5000 \
  --output-dir reports/baseline-5000
```

 CMA-ES can be used as an additional optimization baseline:

```
pip install cma

python scripts/baseline_cmaes.py \
  --evaluations 400 \
  --x0 random \
  --output-dir reports/cmaes-400
```

 A trained SAC policy can be evaluated against the random-search results:

```
python scripts/evaluate_policy.py \
  reports/sac-long \
  --rollouts 20 \
  --baseline reports/baseline-5000
```

 Policy evaluation can be performed without exploration noise to measure the learned policy itself.

---

## Equalizer Optimization

 The default optimization problem sizes the CTLE.

 With:

```
--equalizer
```

 the optimization space expands to six parameters:

```
CTLE parameter 1
CTLE parameter 2
CTLE parameter 3
CTLE parameter 4
CTLE parameter 5
DFE weight
```

 The behavioral one-tap DFE is applied after the CTLE and channel simulation, and the post-DFE eye becomes the optimization metric.

 This allows the RL environment to optimize the complete CTLE + DFE equalization path rather than the CTLE in isolation.

 HD3 can be included directly in the reward using:

```
--hd3
```

 The linearity gate is executed once the other required specifications have passed.

---

## IHP sg13g2 PDK Support

 The IHP Open PDK `sg13g2` transistor deck uses PSP103 device models.

 ngspice loads the PSP103 models through OpenVAF-compiled OSDI files.

 Set:

```
IHP_PDK_ROOT=/path/to/ihp-open-pdk
```

 The expected OSDI directory contains:

```
ihp-sg13g2/libs.tech/ngspice/osdi/
```

 with:

```
psp103.osdi
psp103_nqs.osdi
mosvar.osdi
```

 If required, the models can be compiled using:

```
libs.tech/verilog-a/openvaf-compile-va.sh
```

 On Windows, systems without Visual Studio C++ tools can use the support described in:

```
tools/openvaf-link-shim/README.md
```

 Check PDK readiness using:

```
python scripts/check_pdk.py
```

 The evaluator can then be configured for IHP models using:

```
SpiceEvaluator.for_model_source("ihp")
```

 The scripts accept:

```
--model-source ihp
--pdk-root <checkout>
```

 The default `generic` model source uses ngspice's Level-1 model for fast architectural and data-flow validation.

 Xyce is not required.

---

## IHP Validation and Training

 Run complete IHP validation using:

```
python scripts/run_validation.py \
  --model-source ihp \
  --pdk-root /path/to/ihp-open-pdk \
  --output-dir reports/runs/ihp-final
```

 Train SAC against the IHP PSP103 models using:

```
python scripts/train_sac.py \
  --model-source ihp \
  --eye-height-min 0.25 \
  --n-envs 8 \
  ... \
  --output-dir reports/sac-ihp
```

 The default `CtleSpecifications` eye targets are:

```
Eye height: 0.5 V
Eye width:  0.7 UI
```

 These targets were calibrated so that approximately one in ten random Level-1 designs passes.

 PSP103 devices exhibit substantially lower gain in this design and none of 400 random designs reached the `0.5 V` eye-height target. IHP training therefore uses:

```
--eye-height-min 0.25
```

 which remains above the approximately `175 mV` PCIe Gen 2 receiver eye reference used for the calibration.

---

## Simulation Execution

 `SpiceEvaluator.run_simulation()` implements the DC and AC evaluation gates.

```
.op
.ac
```

 `run_transient()` implements:

```
RC channel
   ↓
PRBS7 stimulus
   ↓
output alignment
   ↓
eye measurement
   ↓
one-tap DFE
```

 Additional evaluator methods provide:

```
run_linearity()
run_noise()
estimate_area()
run_pvt()
```

 for HD3, input-referred noise, area estimation, and process-corner verification.

 The ngspice executable can be selected through:

```
NGSPICE=/path/to/ngspice
```

 or passed directly to the scripts:

```
--ngspice <path>
```

 On Windows, use the `ngspice_con.exe` executable where appropriate.

---

## Simulation Parallelism

 The vectorized RL environment uses:

```
rl/threaded_vec_env.py
```

 rather than `SubprocVecEnv`.

 The reason is that each ngspice simulation already executes as a subprocess and releases the Python GIL. On Windows, spawning multiple Python workers can also cause CUDA/Torch DLL initialization failures.

 The optimization is therefore primarily simulator-bound.

 The GPU is used for the SAC neural-network computation rather than for ngspice simulation.

 Each ngspice process is configured with:

```
set num_threads=1
```

 in the generated `.spiceinit`.

 ngspice does not reliably obey `OMP_NUM_THREADS` for this purpose. Allowing multiple OpenMP threads per simulator while several simulations are running caused excessive spin-waiting and made PSP103 transient simulations approximately sixty times slower.

 The generic Level-1 model remains available for rapid development and debugging.

---

## Validation Results

 The pre-ML validation pipeline covers:

- CTLE sizing.
- IHP PSP103 simulation.
- DC operating-point validation.
- AC peaking validation.
- PRBS transient simulation.
- Post-channel eye measurement.
- Behavioral one-tap DFE.
- HD3 measurement.
- Input-referred noise.
- Area estimation.
- 45-corner PVT verification.

 The SAC training flow has been exercised with both the generic Level-1 model and the IHP PSP103 models.

 The initial 5000-step Level-1 experiments were comparable to random search. After introducing the margin bonus and hold-on-success episodes, the Level-1 policy reached a fully feasible design from random starting points in a median of 2.5 simulation steps across 20/20 rollouts, with 45/45 PVT verification.

 The IHP PSP103 training flow is maintained separately because of its substantially higher simulation cost.

---

## Linearity and Measurement

 HD3 and noise requirements are represented in `CtleSpecifications`.

 The validation pipeline reports these requirements through the linearity and noise gates.

 An earlier HD3 measurement reported approximately `-25 dB` for otherwise passing IHP designs. This was traced to spectral leakage caused by a 2.5-cycle rectangular FFT measurement window.

 Using an integer-cycle Hann window produces approximately `-58 dB` for the same designs.

 The measurement methodology therefore uses the corrected integer-cycle window rather than interpreting the earlier leakage-dominated result as circuit distortion.

---

## Reports and Artifacts

 Validation and training runs generate artifacts under the selected output directory.

 Typical outputs include:

```
reports/
├── baseline-5000/
├── cmaes-400/
├── sac/
├── sac-long/
└── runs/
    ├── latest/
    ├── ihp-final/
    └── ihp-submission/
```

 The active submission report is generated under:

```
reports/runs/ihp-submission/
```

 The generated artifacts include validation measurements, PVT results, optimization data, and the final sized SPICE netlist.

---

## Project Status

 The pre-ML pipeline is complete through:

```
CTLE sizing
      ↓
IHP PSP103 simulation
      ↓
PRBS transient / eye validation
      ↓
Behavioral one-tap DFE
      ↓
HD3 / noise / area
      ↓
45-corner PVT
```

 The simulator adapter is intended to remain stable while the scheduling, reward, and optimization layers evolve.

 The current architecture therefore separates:

```
Circuit / Netlist
        │
        ▼
SPICE Evaluator
        │
        ▼
Specifications
        │
        ▼
Reward
        │
        ▼
RL Environment
        │
        ▼
SAC / Optimization
```

 This separation allows optimization algorithms and reward policies to change without modifying the underlying simulator interface.

---

## Report

 Final reports and presentations are maintained under:

```
/astera
```


---

## Authors

 **Team Beluga**

- Bharat Kumar Saxena
- Medha Soni
- Samarth Soni
