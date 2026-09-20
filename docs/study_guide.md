# AutoAnalog-RL study guide: from the transistor to the reward and back

This is the text to know cold before the final-round Q&A. It is organised as one causal chain — circuit → measurement → number → reward → policy → sizing — because that chain is what a cross-questioner probes. Every number here is in the repo; the file that owns each step is named so you can open it during preparation.

---

## 0. The 60-second version (say this when asked "so what does the RL actually do?")

The circuit has five knobs: input width, load resistor, tail current, and the degeneration resistor and capacitor. Every spec is a function of those five knobs, but the function is only available through ngspice, and it is coupled: widening the transistor raises gain and peaking, but also power and input capacitance. Our agent is a policy that looks at the current design and its measured metrics — peaking, power, eye height, eye width, and how far each spec is violated — and outputs a small step on the five knobs. The step is simulated, the new metrics come back, and a reward scores them: a penalty proportional to how far each spec is missed, a continuous charge for power, and a bonus with extra credit for margin once everything passes. Training ran that loop about 12,000 times on the real IHP models; afterwards, from any random starting design, the policy reaches a fully passing design in two to three simulations. That is the connection: the electrical specs *are* the reward, the electrical knobs *are* the actions, and the transistor-level simulator *is* the environment.

---

## 1. The circuit and why each knob matters

**Topology** (`netlists/ctle_template.sp`, Figure 2 of the report): differential NMOS pair M1/M2, resistive loads R_load to VDD = 1.2 V, tail current source I_bias, and *source degeneration*: R_s from each source to a common tail node, with C_s across the two sources.

**Half-circuit intuition.** Each side sees a degeneration impedance Z_s = R_s ∥ (1/(jω·2C_s)) (the capacitor across the pair looks like 2C_s per side). The differential gain is

    A(ω) ≈ gm · R_load / (1 + gm · Z_s(ω))

- **At DC**, Z_s = R_s, so the gain is reduced to gm·R_load / (1 + gm·R_s). Degeneration throws gain away at low frequency on purpose.
- **At high frequency**, C_s shorts R_s, Z_s → 0, and the gain rises toward gm·R_load.
- The result is a **zero** at ω_z ≈ 1/(2·R_s·C_s) and a **pole** at ω_p ≈ (1 + gm·R_s)/(2·R_s·C_s), and the "peaking" between them is up to 20·log10(1 + gm·R_s) dB. That is the equalizer: it amplifies the frequencies the channel attenuated.

**What each knob does** (this is the table to have in your head):

| Knob | Physical effect | Specs it moves |
|---|---|---|
| W_in (0.5–50 µm) | gm ∝ √(W·I) in strong inversion; more W = more gm, more input capacitance, more area | gain and peaking up; bandwidth down at the extreme; noise down (4kTγ/gm) |
| R_load (100–1000 Ω) | gain ∝ gm·R_load; output pole at 1/(R_load·C_out) | gain up, bandwidth down, output swing headroom (I·R drop) |
| I_bias (0.1–2 mA) | sets gm and power = VDD·I_bias; too much current × R_load starves headroom | peaking up, power up linearly, HD3 improves (more overdrive) |
| R_s (10–500 Ω) | degeneration: lowers DC gain, sets peaking amount 1 + gm·R_s, linearises the pair | peaking up, DC gain down, HD3 better, noise slightly worse |
| C_s (1 fF–1 pF) | with R_s sets *where* the zero sits, 1/(2·R_s·C_s) | moves the peaking frequency; too large → peak below Nyquist |

Notice why a manual sweep is painful: peaking depends on the product gm·R_s and on the R_s·C_s product; power on I_bias alone; the eye on all of them through the channel. Nothing is separable.

**The channel** (`SpiceEvaluator._channel_block`): two RC sections, 50 Ω and 500 Ω, each with a real pole at 1.7 GHz, giving about −10 dB at the 2.5 GHz Nyquist frequency and −20 dB at 5 GHz — the loss slope of a PCIe Gen 2 class FR-4 trace. It is only inserted for the transient gate; the DC/AC gates bypass it so "peaking" measures the CTLE alone.

**The DFE** (`rl/dfe.py`): after the CTLE, the waveform is sampled once per unit interval at the eye centre. A one-tap DFE subtracts tap × (previous hard decision) from each sample — it cancels the first post-cursor of the remaining inter-symbol interference. The tap is either swept per design or, in the `--equalizer` variant, part of the agent's action.

---

## 2. What is measured, and how each measurement becomes a number

All of this is in `spice/spice_engine.py`. The order is the fail-fast order.

1. **`.op` — DC operating point** (`run_simulation`). Are all devices biased? If ngspice cannot converge or a node is off the rails, the design is *DC-invalid*. Cost: ~0.1 s. Everything else is skipped and the reward is −10.
2. **`.ac` 10 MHz–10 GHz** (same call). Two numbers: DC gain (at 10 MHz) and the gain at the Nyquist frequency. `peaking_boost = gain(Nyquist) − gain(DC)` in dB. `power = VDD × I(VDD)` from the operating point.
3. **PRBS7 transient through the channel** (`run_transient`). A 127-bit PRBS7 at 5 Gbps (UI = 200 ps), repeated twice, 2 ps time step, ~2 s of ngspice. The output is aligned to the transmitted bits by correlation (this also resolves the CTLE's inversion), only the final period is measured, and the eye is folded into one UI. **Eye height** = the largest over sampling phase of min(ones) − max(zeros); **eye width** = the fraction of the UI over which that opening is positive. Because the labels are the *transmitted* bits, a closed eye reads as zero — it cannot be faked.
4. **HD3** (`run_linearity`, only when `--hd3` and only once every other spec passes). A 100 MHz differential tone of 100 mV peak-to-peak; the last three full cycles of the output are FFT'd with a Hann window; HD3 = 20·log10(|X(300 MHz)| / |X(100 MHz)|).
5. **Validation-only gates** (`run_validation.py`, `evaluate_policy.py`): input-referred noise integrated over 10 MHz–5 GHz from ngspice's `.noise`; a first-order area estimate from device geometry; and the **45-corner PVT sweep** — process {TT, SS, FF, SF, FS} × supply {0.95, 1.00, 1.05} × 1.2 V × temperature {0, 62.5, 125 °C}, checking peaking stays in 3–12 dB at every corner.

**Why "fail fast" is not a detail.** A random design is DC-valid only ~90 % of the time and reaches the transient only when DC passes; HD3 only runs when everything else already passes. Across a 12k-step run this saves roughly a third of all ngspice time, and it is what the brief means by "fail-fast DC operating point checks".

---

## 3. The specifications and their normalisation (`rl/specs.py`)

Each spec is a `Constraint(name, target, direction, norm_factor)`. The **violation** is 0 when satisfied and otherwise the miss divided by the norm factor, so that "1.0" always means "a full spec's worth of miss" regardless of units:

| Constraint | Target | Violation when violated |
|---|---|---|
| peaking_boost | ≥ 3 dB | (3 − peaking) / 3 |
| peaking_ceiling | ≤ 12 dB | (peaking − 12) / 12 |
| power | ≤ 2 mW | (P − 2 mW) / 2 mW |
| eye_horizontal_ui | ≥ 0.7 UI | (0.7 − width) / 0.7 |
| eye_vertical_v | ≥ 0.25 V (IHP) | (0.25 − height) / 0.25 |
| hd3 (with `--hd3`) | ≤ −30 dB | (HD3 + 30) / 10 |

A missing metric (e.g. no eye because the transient did not run) counts as a full violation of 1.0. The **margin** is the signed version of the same quantity (positive inside the spec) and is used for the bonus.

Why our eye targets are tighter than the brief's (100 mV, 0.4 UI): the brief's thresholds are what a receiver *needs*; ours are what makes the agent deliver *margin*. Under the brief's thresholds even the pre-ML AC-only search passes; under ours it fails (0.160 V), which is the honest reason the RL result matters. Why 0.25 V and not 0.5 V: 0.5 V was calibrated on the generic Level-1 model; on the real PSP103 models none of 400 random designs reaches it (the devices have about half the gain), and 0.25 V restores the same ~10 % random feasibility while sitting above PCIe Gen 2's ~175 mV receiver eye.

---

## 4. The RL formulation (`rl/environment.py`, `rl/reward.py`)

**Observation (15 numbers for the 5-knob problem).** The design vector in [−1, 1]⁵ (each knob mapped linearly onto its range), a DC-valid flag, four scaled metrics (peaking/12, power/2 mW, eye height in V, eye width in UI) and the five violations. So the policy sees *where it is* and *how the circuit responded* — it is a feedback controller over the design space, not a blind optimiser.

**Action.** Five values in [−1, 1], scaled by 0.2 and *added* to the design vector (delta mode), clipped to the box. A step therefore moves each knob by at most 20 % of its range — big enough to cross the feasible region in a few steps, small enough that the Q-function can be smooth. (The random-search and CMA-ES baselines use absolute mode.)

**Reward.** With violations vᵢ and weights wᵢ (all 1 by default, adjustable by the LLM frontend):

    r = − Σ wᵢ·vᵢ  − w_eff · min(1, P / P_max)  + [all specs met] · (20 + w_margin · min(1, tightest margin))
    r = −10 if the DC operating point failed

Three ideas in one line: the violation sum is a smooth gradient *toward* feasibility from anywhere; the power term keeps a gradient *inside* the feasible set (otherwise every feasible design would score the same); the success bonus plus margin bonus (weight 5) pulls the agent *into* the region rather than onto its edge.

**Episode.** Up to 30 steps from a random design (`--random-reset`). The episode does **not** end on success (`--hold-on-success`): once feasible, every further step keeps earning the bonus, so the agent learns to stay feasible and grow margin. A DC failure does not end the episode either, so the agent can step back out.

**Algorithm.** Soft Actor-Critic (stable-baselines3): an off-policy actor-critic for continuous actions with an entropy term that keeps exploration alive and tunes itself (initial coefficient 0.1). Batch 256, one gradient step per environment step, 500 random warm-up steps, eight environments in parallel (each with its own ngspice). We chose SAC because the action is continuous and low-dimensional, the simulator is expensive so sample efficiency matters (off-policy reuse of every transition), and it is robust to reward scale.

**Curriculum (`--corners all --resume`).** Stage 1 trains at the nominal corner. Stage 2 resumes the same policy and simulates each episode at a random one of the 45 corners without telling the policy which. The policy must therefore read the corner from the metrics it observes (e.g. lower peaking than expected → slow silicon) and compensate. It transferred at ~95 % feasible from the first window, and in evaluation 24/24 rollouts at random corners reached feasibility in a median of 2 steps.

---

## 5. The seam: what the policy actually learned, in circuit terms

Read a rollout (`scripts/demo_rollout.py`, or slide 4): starting from a random design at the SS / 1.0 V / 62.5 °C corner the policy moved in three steps from (W_in 27.6 µm, R_load 829 Ω, I_bias 0.46 mA, R_s 307 Ω, C_s 885 fF) to (23.2 µm, 977 Ω, 0.84 mA, 232 Ω, 688 fF). Translate that:

- The eye was closed (0.173 V) with plenty of peaking (5.25 dB) but only 0.55 mW. The design was *starved*: not enough gm·R_load, so the post-channel swing was small.
- The policy **raised I_bias** (more gm and swing), **raised R_load** (more gain), **lowered R_s** (less DC-gain loss, since peaking was already sufficient) and **lowered C_s** (keeps the zero near Nyquist as R_s drops), while trimming W_in slightly (less input capacitance).
- Peaking stayed in range (4.97 dB), power rose to 1.01 mW (still half the budget), the eye opened to 0.321 V, and HD3 came in at −57.8 dB.

That is exactly the move an analog designer makes by hand, and it is why the "2–3 simulations" number is meaningful: the policy has internalised the direction of each knob's effect on each spec, at every corner, and it does not need to re-discover it per instance. CMA-ES, by contrast, re-discovers it every time (median 9 evaluations from a random start, 400 to refine); random sampling needs ~13 per feasible hit.

Other things the runs tell you about the circuit:
- **R_load pins at its 1 kΩ bound** in most designs — the agent wants more gain per milliamp than the box allows. Honest answer if asked: widening the bound (or an active/inductive load) is the next design-space step.
- **HD3 is not a binding constraint** once measured correctly (−58 to −72 dB): source degeneration linearises the pair well below the −30 dB target at 100 mV pp.
- **The DFE tap the agent chooses is small** (+0.045 to +0.075): the CTLE already removes most ISI; the tap cleans the first post-cursor and adds ~10–40 % eye.

---

## 6. The two measurement bugs — know these, they will come up

Both were found because the agent optimised against the metric, which is a good story if you tell it first.

**HD3.** The linearity gate FFT'd 25 ns of a 100 MHz tone — 2.5 cycles — with no window. The bin spacing was 40 MHz, so the fundamental sat between bins and its spectral leakage filled the 300 MHz bin: a false floor near −25 dB for every design. Evidence it was the measurement: sweeping each knob over its full range moved HD3 only between −21 and −26 dB, and an HD3-enforced run could not move it either. Fix: integer number of periods (3 cycles) with a Hann taper; unit-tested on a synthetic −40 dB tone. Result: −58 dB for the same designs.

**DFE eye.** The DFE eye was measured as the gap between samples the DFE *decided* were ones and samples it decided were zeros. A large tap subtracts ±tap from alternate samples, so the two decision classes separate by the tap itself — an artificial "eye". The equalizer agent found this within 2k steps, pushed the tap to +0.5 and reported 0.70 V while the receiver decided 54 of 127 bits wrongly. Fix: label samples by the *transmitted* PRBS bits, slice at the midpoint of the two symbol populations, count bit errors, report zero eye on any error. With the honest metric the DFE still helps (+39 % on the seed-1 design).

The general lesson to say out loud: an optimiser that can exploit a metric will find the flaw before you do, so every metric in the reward must be referenced to ground truth (transmitted bits, integer cycles).

---

## 7. Results to have memorised

| | Value |
|---|---|
| Training convergence | ~2k environment steps, every run (7 runs) |
| Feasible steps late in training | 95–97 % (CTLE), 92 % (equalizer) |
| Rollouts feasible | 60/60 over 3 seeds; 24/24 at random corners |
| Steps to feasibility | 2.53 ± 0.06 mean over seeds; median 2–3; worst 5 |
| Best designs | eye 0.32–0.37 V / 0.86–0.96 UI, 0.9–1.4 mW, peaking 4–6 dB, HD3 −58 to −72 dB, 45/45 PVT |
| Random search | 7.9 % feasible, first at #27 |
| CMA-ES from random start | median 9, worst 24 to first feasible; 400 sims to match policy quality |
| AC-only bounded search | 0.160 V / 0.58 UI eye — fails our target |
| Tunability | 2.5 GHz design at 1.25 GHz: 2.87 dB (fail); retuned policy 5.23 dB, 20/20 |
| Throughput | ~0.6 s per environment step, 8 ngspice processes; 12k steps ≈ 2 h |
| Noise | 0.242 mV rms input-referred (ngspice total 0.2417) |

---

## 8. Likely cross-questions and answers

**Why RL and not an optimiser? CMA-ES matched your quality.**
Per instance, yes, given 400 simulations. The policy is *amortised*: one training run, then 2–3 simulations for any new instance — a new start, a new corner, a re-weighted reward. From random starts CMA-ES needs a median of 9 evaluations to reach feasibility, we need 2–3; and CMA-ES has no memory across instances. If you only ever size one circuit once, use CMA-ES; if you size many, or across corners, or re-target specs, the policy pays for itself.

**What does the agent see? Does it see the netlist?**
No. It sees its own five normalised knob values, the DC-valid flag, four scaled metrics and the five violations — 15 numbers. That is enough because the metrics tell it how the circuit responded to its last move.

**Why delta actions rather than outputting the design directly?**
Incremental moves make the value function smooth and let the same policy work from any start. Absolute actions turn each step into an independent guess, which is what random search is.

**Why does the episode not end when the specs are met?**
Our first runs did end on success and plateaued at ~85 % with the design sitting exactly on the eye-height boundary, because nothing rewarded margin. Holding the episode and paying a margin bonus took it to 95 % and moved the median eye from 0.43 to 0.92 V on the Level-1 model.

**How does the reward relate to the electrical specs?**
Directly: each spec is a normalised violation term; the reward is minus their sum, minus a continuous power charge, plus a bonus for meeting everything with margin. The weights are the only free parameters and the LLM frontend sets them from plain language.

**What is "curriculum learning across PVT corners"?**
Stage 1 nominal, stage 2 random corner per episode with the same policy resumed. The policy is corner-blind — it infers the corner from the metrics — and 24/24 rollouts at random corners reach feasibility. A corner-aware observation is the obvious next step.

**Why IHP and not sky130?**
The brief allows both. Our synopsis said sky130; we built on IHP because its PSP103 Verilog-A models compile to OSDI and run natively in ngspice at 0.6 s per step, while the sky130 flow was too slow for an RL loop. PSP103 is a foundry-grade surface-potential model. The pipeline is PDK-agnostic: the generic and IHP paths differ only in the model library.

**Is the DFE a real circuit?**
No — a behavioural one-tap on the sampled CTLE output, measured against the transmitted bits with error propagation modelled. The analog slicer in `DFE/` is a topology reference, not simulated end to end. That is our first listed limitation and next step.

**Your eye spec is 0.25 V but the brief says 100 mV. Why?**
Margin. Under 100 mV even the AC-only search passes. Our reward asks for 2.5× the requirement so the policy lands designs with room for the things we do not model.

**How do you know the policy is not memorising one design?**
Every evaluation starts from a fresh random design (and, for the curriculum policy, a random corner); the sized results differ per rollout, and three seeds agree on the statistics.

**How long does training take, and on what?**
About 2 hours for 12k steps on a 12-core laptop; the GPU is irrelevant, the time is ngspice. Eight simulators in parallel, each pinned to one OpenMP thread — without that pinning eight PSP103 transients took 90 s each instead of 1.5 s.

**What is HD3 and why 100 MHz?**
Third-harmonic distortion of a single tone, a linearity figure: a 100 MHz, 100 mV pp differential tone in, and the ratio of the 300 MHz component to the fundamental at the output. 100 MHz is the brief's stimulus; it sits well inside the passband so the number reflects the pair's nonlinearity rather than filtering.

**Why was HD3 wrong before?**
Spectral leakage — 2.5 cycles in a rectangular window. Section 6.

**How does noise get measured?**
ngspice `.noise` from the input source to the differential output, input-referred spectral density in V/√Hz, squared and integrated over 10 MHz–5 GHz, square-rooted: 0.242 mV rms. Cross-checked against ngspice's own `inoise_total` (0.2417).

**What does the LLM do?**
It maps a sentence ("power matters more than eye margin") to the per-spec reward weights, the power charge and the margin bonus, as schema-validated JSON; a keyword parser is the offline fallback. It applies at the start of a run or resumed stage. It does not touch the circuit or the simulator.

**What happens at a corner where the specs cannot be met?**
The violation terms stay positive and the policy minimises them — it lands on the best available design. In our matrix every corner was reachable (45/45 for every design).

**Why fail-fast ordering matters for RL specifically?**
The agent proposes many infeasible designs early in training. Skipping the 2 s transient for the ~10 % that fail DC, and the HD3 run for everything that is not otherwise feasible, is what keeps a step at 0.6 s.

**What would you do with more time?**
Analog DFE slicer in the transient gate; corner-aware observation; Bayesian-optimisation baseline and a wall-clock comparison to a manual sweep; a closed LLM loop; a sky130 backend if required.

---

## 9. Where to point during the demo

- Circuit: `netlists/ctle_template.sp` (30 lines; read it once).
- Measurements: `spice/spice_engine.py` — `run_simulation`, `run_transient`, `_eye_metrics`, `hd3_from_waveform`, `run_noise`, `run_pvt`.
- Specs and violations: `rl/specs.py`. Reward: `rl/reward.py`. Environment: `rl/environment.py`.
- Training: `scripts/train_sac.py`. Evaluation: `scripts/evaluate_policy.py`. Live demo: `scripts/demo_rollout.py`.
- Numbers: `reports/eval-ihp-*/evaluation.json`; the report `docs/report/AutoAnalog-RL_report.pdf`.
