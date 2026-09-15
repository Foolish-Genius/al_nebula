# AutoAnalog-RL — demo video script

Target length: 5:45 (5:00 without the optional lines marked **[optional]**).
Slide numbers refer to `docs/report/AutoAnalog-RL_deck.pptx`; every slide's speaker notes carry the same text.
Read at a calm pace — about 140 words per minute. Lines in *italics* are stage directions, not spoken.

Recording: everything is on the slides — the repository page (slide 2), a real terminal run (slide 4), the netlist and commands (slide 11) — so the screen recording is the deck from start to finish with no window switching. If you want one live moment, keep a terminal open behind the deck and alt-tab to it on slide 4; otherwise skip that.

---

## 0:00 – 0:25 · Introduction — Slide 1

*Slide 1.*

Hello everyone. This is our project, AutoAnalog-RL — Automated Sizing of High-Speed Interface Circuits via Reinforcement Learning.

Our objective is to automatically size a PCIe Gen 2 receiver equalizer, consisting of a one-stage CTLE and a one-tap DFE, using reinforcement learning and circuit simulation.

Instead of manually sweeping circuit parameters, our SAC agent interacts directly with ngspice and searches for a design that satisfies the required specifications.

## 0:25 – 1:00 · Repository and overall flow — Slide 2

*Slide 2 — the repository page is on the left, the flow diagram on the right.*

This is our project repository.

The `rl` folder contains the specifications, the reward function, the environment, and the DFE and PVT implementations. The `spice` folder handles the simulator interface and every measurement, while `netlists` contains the circuit topology.

The main training and evaluation scripts are inside `scripts`, and we also have a regression test suite of fifty-nine tests that runs without the simulator.

The overall flow is: the SAC agent generates a normalized design action, the evaluator converts it into circuit parameters, generates the SPICE netlist, and runs the simulations. The resulting circuit metrics are converted into a reward and sent back to the agent.

The simulator uses the IHP sg13g2 130-nanometer open PDK with PSP103 transistor models, running through ngspice.

The problem statement allows either IHP or SkyWater sky130. Our synopsis named sky130; we built on IHP because its PSP103 Verilog-A models compile to OSDI and run natively in ngspice at about 0.6 seconds per reinforcement-learning step, with eight simulators in parallel.

## 1:00 – 1:35 · Circuit and parameters — Slide 3

*Show the schematic; point at each labelled element as you name it.*

The circuit being sized is a differential NMOS CTLE with resistive loads and RC source degeneration.

The agent controls five CTLE parameters: the input transistor width, the load resistance, the bias current, the source-degeneration resistance, and the source-degeneration capacitance.

For the complete equalizer, a sixth parameter is added for the one-tap DFE.

At every step, the evaluator checks the design progressively: first the DC operating point, then the AC response and peaking, followed by the PRBS transient through the lossy channel with eye measurement and the DFE, and finally the HD3 linearity check. A design that fails an early gate never reaches the expensive later ones.

## 1:35 – 2:25 · The reinforcement-learning loop, live — Slides 4 and 5

*Slide 4 — a real rollout is captured on the slide. (Optional live moment: alt-tab to a terminal and run `.\.venv\Scripts\python.exe scripts\demo_rollout.py reports\sac-ihp-pvt`.)*

Now we move to the main part of the project — the reinforcement-learning loop.

We use a Soft Actor-Critic, or SAC, agent. The agent starts from a random circuit design and learns which parameter changes improve the circuit's performance.

The reward considers the specification violations as well as power and the margin from the required specifications.

We also use fail-fast simulation, so a design that already fails an early gate does not spend simulation time on the later analyses.

*Point at the lines on the slide:*

What you see here is the trained policy running. It starts from a random design at a random process, voltage and temperature corner. Each line is one ngspice evaluation of a new design, and the checklist on the right shows which specifications pass. Within a few simulations every specification is green, including HD3, and the sized netlist is written. That took about seven seconds on a laptop.

*Switch to slide 5.*

The training itself converges in approximately two thousand environment steps. Across the three base seeds, all sixty out of sixty deterministic rollouts reached a feasible design, with an average of about 2.5 steps to reach feasibility.

These plots show the reward and the fraction of feasible designs improving during training, demonstrating that the agent is learning rather than performing a blind sweep.

## 2:25 – 3:15 · Actual circuit results — Slides 6, 7, 8

*Slide 6 — eye diagram.*

Here is one of the key outputs — the post-channel eye diagram.

For the seed-one design, the measured eye height is about 0.327 volts with an eye width of 0.88 UI, which satisfies our internal target of 0.25 volts and 0.7 UI. Our targets are deliberately tighter than the brief's 100 millivolts and 0.4 UI, so that the policy delivers margin rather than a marginal pass.

*Slide 7 — AC response.*

Next is the AC response. The design shows approximately 4.2 dB of high-frequency peaking at the 2.5-gigahertz Nyquist frequency, which is within the required 3 to 12 dB range.

The Nyquist frequency is itself a specification input. Retargeting the same policy to 1.25 gigahertz — PCIe Gen 1 — gives 5.23 dB of peaking at the new frequency, where the 2.5-gigahertz design gave only 2.87, with twenty out of twenty rollouts feasible.

*Slide 8 — 45 corners and the results table.*

The design is then validated across the complete set of 45 PVT corners: five process corners, supply plus or minus five percent, and zero to 125 degrees. The peaking remains within the required range at every corner.

The policy was also fine-tuned across all 45 corners as a second curriculum stage. In evaluation, 24 out of 24 rollouts that started at random corners reached a feasible design in a median of two steps.

For the curriculum-trained design, the best result has an eye height of 0.370 volts, a power of 1.12 milliwatts, and an HD3 of minus 61.9 dB, while passing all 45 PVT corners.

## 3:15 – 3:45 · DFE — Slide 9

*Slide 9 — the transmitter-to-eye chain and the DFE eye.*

We also extended the system to size the complete equalizer by adding the one-tap DFE to the action.

Importantly, we corrected the eye measurement so that the DFE eye is evaluated against the transmitted PRBS bits rather than the DFE's own decisions.

With the corrected measurement the DFE still provides a real improvement. One example improves the eye height from 0.327 volts to 0.455 volts with a tap of approximately plus 0.075, with no bit errors.

**[optional, 15 s]** We found and fixed two measurement flaws this way, by letting the agent exploit them. An FFT leakage floor had made HD3 look like a failure at minus 25 dB — the real value is minus 58. And the self-labelled DFE eye let the agent push the tap to its bound while the receiver got 54 of 127 bits wrong. Both are fixed and unit-tested.

## 3:45 – 4:20 · Comparison with conventional search — Slide 10

Finally, we compare the RL policy against conventional search methods.

Random search requires about 13 simulations per feasible design, and CMA-ES requires around 9 simulations from random starting points.

In comparison, the trained SAC policy reaches a feasible design in a median of 2 to 3 simulations, without performing a new optimization search for every starting point. Given 400 simulations per design, CMA-ES does reach the same final quality — so the policy's advantage is that it is trained once and then reused.

We also tested the pre-ML AC-only bounded search. Although it optimizes the AC response, its resulting post-channel eye is only 0.160 volts, which shows why optimizing only the frequency-domain characteristics is not sufficient.

## 4:20 – 4:45 · Reproducibility and the final output — Slide 11

*Slide 11 — the commands, the netlist header and the artifact list are all on the slide.*

The entire flow is reproducible through the scripts in our repository.

The training and evaluation scripts generate JSON, CSV and PNG evidence, and every validated design also produces a sized SPICE netlist containing the circuit parameters, the model information and the DFE tap.

So the final output is not just an RL score — it is an actual sized circuit together with its simulation and validation results.

**[optional, 15 s]** *Point at the last command on the slide (reward_from_feedback.py).*
As a bonus, an engineer can steer the reward in plain language: this sentence becomes a set of reward weights — power up, eye margin and linearity down — which the training script takes as input.

## 4:45 – 5:00 · Closing — Slide 12

To summarize, AutoAnalog-RL combines reinforcement learning, SPICE simulation and PVT-aware validation to automatically size a high-speed receiver equalizer.

Our trained policy achieves feasible designs in only a few simulations, passes the required circuit specifications and all 45 PVT corners, and produces the corresponding sized netlist.

Thank you.

---

## Numbers you may be asked about

| Claim | Value | Source |
|---|---|---|
| Steps to feasibility, 3 seeds | 2.53 ± 0.06 (median 2–3, worst 4–5) | `reports/eval-ihp-v1*/evaluation.json` |
| Rollouts feasible | 60/60 nominal; 24/24 at random corners | same, `eval-ihp-pvt` |
| Seed-1 design | 0.327 V / 0.88 UI eye, 4.2 dB, 1.22 mW, HD3 −57.8 dB, 45/45 | `eval-ihp-v1` |
| Curriculum design | 0.370 V / 0.90 UI, 1.12 mW, HD3 −61.9 dB, 45/45 | `eval-ihp-pvt` |
| Retune to 1.25 GHz | 5.23 dB (vs 2.87 untuned), 20/20 | `eval-ihp-gen1` |
| DFE example | 0.327 → 0.455 V at tap +0.075, 0 errors | report §7.2 |
| Random search | 7.9 % feasible, first at #27 | `baseline-ihp-3000` |
| CMA-ES from random starts | median 9, worst 24 | `cmaes-ihp-random-s*` |
| AC-only bounded search | eye 0.160 V / 0.58 UI | `runs/ihp-submission-2026-09-15` |
| Throughput | ~0.6 s per step, 8 ngspice processes | any run's monitor files |
| Noise | 0.242 mV rms (ngspice total 0.2417) | `runs/ihp-submission-2026-09-15` |
