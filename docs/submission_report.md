# AutoAnalog-RL Submission Report

## Executive Summary

AutoAnalog-RL is a modular analog equalizer-sizing framework for a PCIe Gen 2 receiver. It searches a bounded CTLE action space, evaluates candidates with ngspice, validates transient eye behavior, applies a one-tap behavioral DFE, and verifies process/voltage/temperature corners. The final backend uses IHP sg13g2 PSP103 Verilog-A models compiled to OSDI and a local ngspice 45.2 build with OSDI enabled.

## Final Reproducible Command

```bash
export LD_LIBRARY_PATH=/home/hp/miniconda3/envs/autoanalog/lib:/home/hp/ngspice-45.2/install/lib
python scripts/run_validation.py \
  --model-source ihp \
  --ngspice-binary /home/hp/ngspice-45.2/install/bin/ngspice \
  --output-dir reports/runs/ihp-submission
```

The command performs 100 bounded CTLE evaluations, selects the best AC candidate, runs transient PRBS validation, computes eye metrics, applies the one-tap DFE model, executes 45 PVT corners, and writes JSON, CSV, and PNG evidence.

## Architecture

```text
bounded search -> CTLE SPICE .op/.ac -> PRBS transient -> 2-UI eye
                                      \-> one-UI sampler -> 1-tap DFE
                                      \-> HD3 / noise / area / PVT
```

The CTLE action is `[W_in, R_load, I_bias, R_s, C_s]`. The whole-equalizer interface adds a normalized sixth action for the DFE tap. The analog DFE SPICE template in `DFE/` is retained as a topology reference; the active end-to-end DFE measurement is a behavioral sampled decision-feedback stage.

## Latest IHP Evidence

The latest real-IHP run is stored in `reports/runs/2026-09-11-ihp-complete2/`.

| Requirement | Result | Status |
|---|---:|---|
| DC operating point | valid | pass |
| Nyquist frequency | 2.5 GHz | pass |
| HF peaking | 7.527 dB | pass, 3 to 12 dB target |
| Power | 0.743 mW | pass, below 15 mW |
| Eye height | 1.086 V | pass, above 100 mV |
| Eye width | 0.565 UI | pass, above 0.4 UI |
| PVT corners | 45/45 simulated and passing | pass |
| Area estimate | 1.09e-05 mm2 | pass, first-order estimate |
| HD3 | -24.63 dB | fail, target below -30 dB |
| Integrated input-referred noise | 0.242 mV rms | pass, below 1.5 mV rms |

The noise result is integrated from the ngspice `inoise_spectrum` vector using the RMS density equation. The raw output and report remain available for independent review.

## What Is Implemented

- IHP sg13g2 PSP103 OSDI model compilation through OpenVAF.
- Local ngspice 45.2 build with OSDI enabled.
- Parameterized source-degenerated differential CTLE.
- Bounded 100-candidate search.
- DC and AC gates from 10 MHz to 10 GHz.
- 5 Gbps PRBS transient simulation.
- Conventional 2-UI eye diagram generation.
- Eye height and eye width measurement.
- One-UI hard-decision one-tap DFE post-processing.
- 45-corner PVT sweep.
- HD3, noise, and area measurement hooks.
- Reproducible CSV, JSON, and PNG artifacts.
- 40 automated tests.

## Known Limitations

1. The DFE transistor deck is not yet a clocked closed-loop analog slicer. The active DFE result is a behavioral one-tap decision-feedback stage driven by CTLE transient samples.
2. HD3 currently misses the strict target and must be optimized through topology/bias changes.
3. Noise extraction requires final input-referred normalization and a raw-vector sanity check before it can qualify the specification.
4. Area is a first-order active-device geometry estimate, not a post-layout area result.
5. SAC training so far uses the generic Level-1 model, not the IHP backend. Three
   5000-step runs reached feasibility in ~85% of episodes but plateaued at the
   eye-height boundary and did not beat random search at equal budget; the
   margin-bonus / hold-on-success reward changes are being evaluated on a
   longer run. Training against the IHP OSDI models is still open.

## Submission Position

The framework, real IHP simulator backend, search loop, reports, and reproducible demonstration are complete for the pre-ML stage. The design itself is not yet fully specification-compliant because HD3 and noise fail. Those failures are visible, repeatable, and now actionable optimization targets rather than missing pipeline stages.
