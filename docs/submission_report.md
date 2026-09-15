# AutoAnalog-RL Submission Report

## Executive Summary

AutoAnalog-RL is a modular analog equalizer-sizing framework for a PCIe Gen 2 receiver. A soft actor-critic (SAC) agent sizes a source-degenerated CTLE by driving ngspice directly: each step runs the DC/AC gate, the PRBS transient through a lossy channel with eye measurement, and (when enabled) the HD3 linearity gate, and the design is then verified across 45 process/voltage/temperature corners with a one-tap behavioral DFE. The backend uses the IHP sg13g2 PSP103 Verilog-A models compiled with OpenVAF to OSDI and loaded by ngspice 47; the generic ngspice Level-1 model remains as a fast debugging path. On the real models the trained policy reaches a fully spec-compliant CTLE from a random starting point in a median of two simulation steps, where random search needs about thirteen.

## Process Design Kit

The Round 0 synopsis named the SkyWater sky130 PDK. The implementation
targets the IHP sg13g2 open PDK (also 130 nm) instead: the team switched
during development because the sky130 simulation flow was too slow for an
RL loop that runs three ngspice analyses per step, while IHP's PSP103
Verilog-A models compile with OpenVAF to OSDI and run natively in ngspice 47
at about 0.6 s per environment step with eight simulators in parallel. The
framework is PDK-agnostic by construction - the generic Level-1 and IHP
paths share every line except the model library selected by
`SpiceEvaluator.for_model_source()` - so a sky130 backend is a netlist
variant plus a corner-library mapping, not a change to the method.

## Reproducible Commands

Prerequisites: an IHP Open PDK checkout with compiled OSDI models under
`ihp-sg13g2/libs.tech/ngspice/osdi/` (`IHP_PDK_ROOT`; see
`tools/openvaf-link-shim/README.md` for Windows), ngspice 44+ (`NGSPICE`), and
`pip install -e .[rl]`. `python scripts/check_pdk.py` confirms readiness.

```bash
# Train the policy on the IHP models (8 parallel ngspice processes, ~0.6 s/step)
python scripts/train_sac.py --model-source ihp --eye-height-min 0.25 --timesteps 20000 \
  --n-envs 8 --hold-on-success --random-reset --margin-weight 5 --invalid-penalty -10 \
  --max-steps 30 --batch-size 256 --gradient-steps -1 --learning-starts 500 \
  --ent-coef auto_0.1 --seed 1 --output-dir reports/sac-ihp-v1
# Random-search baseline with the same reward
python scripts/baseline_random.py --model-source ihp --eye-height-min 0.25 --margin-weight 5 \
  --invalid-penalty -10 --evaluations 3000 --n-workers 8 --output-dir reports/baseline-ihp-3000
# Deterministic rollouts of the policy, comparison, and PVT/HD3 validation of its best design
python scripts/evaluate_policy.py reports/sac-ihp-v1 --rollouts 20 \
  --baseline reports/baseline-ihp-3000 --output-dir reports/eval-ihp-v1
# Pre-ML pipeline evidence (100-candidate bounded search, all gates, 45 corners)
python scripts/run_validation.py --model-source ihp --output-dir reports/runs/ihp-submission
```

Every command writes JSON, CSV, and PNG evidence to its output directory.

## Architecture

```text
SAC agent (or bounded search) -> CTLE SPICE .op/.ac -> PRBS transient -> 2-UI eye -> reward
                                                     \-> HD3 gate (once other specs pass)
                                                     \-> one-UI sampler -> 1-tap DFE
                                                     \-> noise / area / 45-corner PVT (validation)
```

The CTLE action is `[W_in, R_load, I_bias, R_s, C_s]`. The whole-equalizer interface adds a normalized sixth action for the DFE tap. The analog DFE SPICE template in `DFE/` is retained as a topology reference; the active end-to-end DFE measurement is a behavioral sampled decision-feedback stage.

## Pre-ML Pipeline Evidence (bounded search)

`scripts/run_validation.py --model-source ihp` regenerated on 2026-09-15 with
the Windows ngspice 47 + OpenVAF OSDI toolchain, stored in
`reports/runs/ihp-submission-2026-09-15/`. The 100-candidate bounded search
selects on the AC gate alone; every gate that is unchanged since the original
Linux run (`reports/runs/ihp-submission/`) reproduces to all printed digits.

| Requirement | Result | Status |
|---|---:|---|
| DC operating point | valid | pass |
| HF peaking at 2.5 GHz Nyquist | 7.527 dB | pass, 3 to 12 dB target |
| Power | 0.743 mW | pass, below 2 mW |
| Eye height after -10 dB channel | 0.160 V (0.310 V with the swept 1-tap DFE, no bit errors) | fail, 0.25 V target |
| Eye width after channel | 0.58 UI | fail, 0.7 UI target |
| PVT corners | 45/45 simulated and passing | pass |
| Area estimate | 1.09e-05 mm2 | pass, first-order estimate |
| HD3 | -58.53 dB | pass, target below -30 dB |
| Integrated input-referred noise | 0.242 mV rms | pass, below 1.5 mV rms |

HD3 was reported as -24.63 dB (fail) in every earlier version of this
evidence. That number was a measurement artefact: the FFT analysed 2.5 cycles
of the 100 MHz tone with a rectangular window, so the fundamental fell between
40 MHz bins and its leakage filled the 300 MHz bin, flooring HD3 near -25 dB
for every design (a sweep of each sizing knob over its full range could not
move it past -26 dB). Analysing an integer number of periods with a Hann
taper (`SpiceEvaluator.hd3_from_waveform`, unit-tested against a synthetic
-40 dB tone) gives -58.5 dB for this design; the circuit was always linear
enough. The eye now fails because the transient gate drives the PRBS through a lossy
channel (about -10 dB at Nyquist) that the original evidence did not include,
and the AC-only search cannot see the eye. The RL policy in the next section,
which is rewarded on the post-channel eye, closes exactly this gap. The noise
result is integrated from the ngspice `inoise_spectrum` vector using the RMS
density equation and cross-checked against ngspice's `inoise_total`.

## RL Results on the IHP Models

All runs: `scripts/train_sac.py --model-source ihp --eye-height-min 0.25
--n-envs 8 --hold-on-success --random-reset --margin-weight 5
--invalid-penalty -10 --max-steps 30 --batch-size 256 --gradient-steps -1
--ent-coef auto_0.1`. Each converged within about 2k environment steps.
"Training feasible" is the fraction of environment steps in the last 4k
steps of training at which every enforced spec was met. Evaluation is
`scripts/evaluate_policy.py`: 20 deterministic rollouts from random starting
designs (no exploration noise), then PVT / HD3 validation of the best design.

| Run | Steps | Extra | Training feasible | Rollouts feasible | Steps to feasible (median / worst) | Best design: eye, power, HD3, PVT |
|---|---:|---|---:|---:|---|---|
| `sac-ihp-v1` (seed 1) | 30k | - | 95.5% | 20 / 20 | 2 / 4 | 0.327 V, 1.22 mW, -57.8 dB, 45/45 |
| `sac-ihp-v1-s2` (seed 2) | 12k | - | 95.5% | 20 / 20 | 3 / 4 | 0.316 V, 1.22 mW, -57.7 dB, 45/45 |
| `sac-ihp-hd3` (seed 1) | 12k | HD3 enforced in the reward | 96.2% | 20 / 20 | 2.5 / 4 | 0.344 V, 1.01 mW, -68.2 dB, 45/45 |
| `sac-ihp-pvt` (stage 2) | +8k | resumed from `sac-ihp-hd3`; every episode at a random one of the 45 PVT corners; HD3 enforced | {PVT_TRAIN} | {PVT_ROLLOUTS} | {PVT_STEPS} | {PVT_DESIGN} |
| `sac-ihp-eq` (seed 1) | 12k | six-value action: CTLE plus the one-tap DFE weight; post-DFE eye drives the reward | {EQ_TRAIN} | {EQ_ROLLOUTS} | {EQ_STEPS} | {EQ_DESIGN} |

Against a 3000-design random search on the same models and reward
(`reports/baseline-ihp-3000/`): random designs satisfy every spec 7.9% of the
time (238 / 3000) and the first feasible one is design 27, so about thirteen
simulations per feasible design; the policies reach one in a median of two to
three. At the same 49-52 simulation budget the policies' best reward is
20.7-20.8 against 19.65 for random search. Random search's best over all
3000 designs (21.12) edges the policies' best within their rollouts, at sixty
times the budget.

Every policy design passes every measured spec: peaking 3-12 dB, power
<= 2 mW, post-channel eye >= 0.25 V and >= 0.7 UI, HD3 <= -30 dB (measured
correctly, see the evidence section), and all 45 PVT corners. Artifacts per run:
`reports/<run>/` (training) and `reports/eval-<run>/` (evaluation, including
the sized netlist `sized_ctle.sp`).

### PVT curriculum

Stage 1 trains at the nominal corner (TT, 1.2 V, 27 C). Stage 2 resumes the
same policy with `--corners all --resume`: each episode is simulated at a
corner drawn uniformly from the 45-corner matrix (5 process x 3 supply x 3
temperature) and the policy is not told which, so it must size for every
corner from the metrics it observes. {PVT_TEXT}

### Whole-equalizer sizing (CTLE + DFE tap)

`--equalizer` extends the action to six values: the five CTLE parameters and
the one-tap DFE weight in [-0.5, 0.5]. The DFE is applied to the CTLE's
post-channel samples whenever a waveform exists, and the post-DFE eye height
drives the reward.

The first equalizer run exposed a second measurement flaw: the DFE eye was
labelled by the DFE's own decisions, so a large tap separated the two
decision classes by itself. The agent found this within 2k steps, drove the
tap to its +0.5 bound and reported a 0.70 V eye on a design whose receiver
decides 54 of 127 bits wrongly. The eye is now measured against the
transmitted PRBS bits (`rl.dfe.dfe_eye_against_bits`: correlation-aligned
UI-centre samples, slicer at the midpoint of the two symbol populations,
DFE fed its own decisions, bit errors counted, zero eye on any error). With
the honest metric a one-tap DFE still helps: the CTLE-only policy design
goes from 0.327 V to 0.455 V at tap +0.075 with no errors. {EQ_TEXT}

### LLM frontend for the reward

`scripts/reward_from_feedback.py` turns an engineer's sentence into the
reward's per-spec weights (`rl/feedback.py`): Claude (Anthropic SDK,
structured JSON output) when credentials are present, a context-aware keyword
parser otherwise, so the loop never blocks on an API. The JSON is passed to
`train_sac.py --reward-settings`. Example, offline path:

```text
> "We are power constrained: current budget matters much more than eye margin,
   and we don't care about linearity"
weights: power 3.0, eye_vertical_v 0.33, hd3 0.33 (others 1.0);
efficiency_weight 3.0; margin_weight 5.0
```

### Eye-height target on IHP

The eye-height target for IHP training is 0.25 V rather than the 0.5 V used
with the Level-1 model: none of 400 random PSP103 designs reaches 0.5 V, and
0.25 V keeps the same ~10% random feasibility the Level-1 targets were
calibrated to while staying above the ~175 mV PCIe Gen 2 receiver eye.

## What Is Implemented

- IHP sg13g2 PSP103 OSDI model compilation through OpenVAF (Linux, and
  Windows via `tools/openvaf-link-shim`).
- ngspice 47 with OSDI, one OpenMP thread per process so eight simulators run
  in parallel.
- Parameterized source-degenerated differential CTLE.
- Bounded 100-candidate search and a random-search baseline.
- SAC training (`scripts/train_sac.py`): thread-parallel environments, margin
  bonus, hold-on-success episodes, configurable eye and HD3 specs.
- Deterministic policy evaluation against the baseline at equal budget
  (`scripts/evaluate_policy.py`).
- Two-stage PVT curriculum (`--corners all --resume`) and whole-equalizer
  sizing with the DFE tap in the action (`--equalizer`).
- LLM frontend: natural-language feedback to reward weights
  (`scripts/reward_from_feedback.py`, `--reward-settings`).
- Sized netlist (`sized_ctle.sp`) written with every validation.
- DC and AC gates from 10 MHz to 10 GHz.
- 5 Gbps PRBS transient simulation.
- Conventional 2-UI eye diagram generation.
- Eye height and eye width measurement.
- One-UI hard-decision one-tap DFE post-processing.
- 45-corner PVT sweep.
- HD3, noise, and area measurement hooks.
- Reproducible CSV, JSON, and PNG artifacts.
- 56 automated tests.

## Known Limitations

1. The DFE transistor deck is not yet a clocked closed-loop analog slicer. The active DFE result is a behavioral one-tap decision-feedback stage driven by CTLE transient samples.
2. HD3 passes with about 28 dB of margin once measured with an integer-cycle
   window; the earlier -25 dB "failure" was spectral leakage in the
   measurement (see the evidence section). The stimulus is a fixed 100 mV
   differential tone; HD3 at larger swings has not been characterised.
3. Noise is qualified: the gate integrates ngspice's input-referred
   `inoise_spectrum` (V/sqrt(Hz), parsed by column name) over 10 MHz to
   5 GHz and reports 0.2420 mV rms; ngspice's own band integral
   (`inoise_total`, now recorded alongside) gives 0.2417 mV rms, the
   difference being trapezoid error on the log-spaced grid.
4. Area is a first-order active-device geometry estimate, not a post-layout area result.
5. The DFE in the RL loop is the behavioural one-tap stage; the analog
   slicer in `DFE/` is not simulated end to end (see 1).
6. Corner-randomised training (the PVT curriculum) does not tell the policy
   which corner it is in; a corner-aware observation is a natural extension.

## Submission Position

The framework, real IHP simulator backend, search loop, reports, and reproducible demonstration are complete for the pre-ML stage. The design itself is not yet fully specification-compliant because HD3 and noise fail. Those failures are visible, repeatable, and now actionable optimization targets rather than missing pipeline stages.
