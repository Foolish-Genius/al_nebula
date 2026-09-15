# Demo video shot list

Target length 6-8 minutes. Every command below runs from `Z:\al_nebula` with
the `.venv` Python; `IHP_PDK_ROOT` and `NGSPICE` are set at user scope, so a
fresh PowerShell window works. Figures referenced are in
`reports\screenshots\video\`; live-training captures are in `reports\screenshots\`.

| # | Shot | What to show / say | Source |
|---|------|--------------------|--------|
| 1 | Title | AutoAnalog-RL: sizing a PCIe Gen 2 (5 Gbps) CTLE + 1-tap DFE with SAC, zero human intervention, on the IHP sg13g2 130 nm PDK. One sentence on the PDK: "proposal named sky130; we built on IHP because its PSP103 Verilog-A models run through ngspice's OSDI interface at 0.6 s per RL step - the framework is PDK-agnostic." | report title page |
| 2 | Architecture | Figure 1 of the report: agent -> environment -> SpiceEvaluator gates in fail-fast order (.op, .ac, PRBS transient + channel + DFE, HD3) -> ngspice x8 -> sized netlist. | `docs/report/draft.pdf` p.2 |
| 3 | The circuit | Figure 2 (schematic) + Table 2 (action bounds) + Table 3 (specs). "Five knobs, seven specs, eye measured after a -10 dB channel." | report p.3 |
| 4 | **Live demo** | Run `python scripts\demo_rollout.py reports\sac-ihp-pvt --seed 7` then `--seed 21` (or no seed for a fresh random start). Narrate: random design, random PVT corner, each line is one ngspice evaluation, specs turn green, netlist prints. 2-3 steps, ~5 s. | terminal |
| 5 | Training | `learning_curves_final.png`: five runs converge within ~2k steps; then one of the live captures `training_live_*.png` to show it actually ran (SB3 log, timesteps ticking). | video folder |
| 6 | vs baselines | `budget_curve_seed1.png` (policy vs random search) + Table 5: random needs ~13 sims per feasible design, CMA-ES ~9 from a random start, policy 2-3. Say it honestly: CMA-ES matches the policy's final quality given 400 sims per design; the policy's win is amortisation - one training run, then 2-3 sims per new instance. | report Sec. 6.1 |
| 7 | Eye before/after | `eye_diagram_preML_bounded_search.png` (AC-only search: 0.160 V, fails) next to `eye_diagram_seed1.png` (policy: 0.327 V, passes). | video folder |
| 8 | PVT curriculum | `pvt_45_corners_curriculum.png`: 45/45 corners; 24/24 rollouts feasible at random corners, median 2 steps. Optionally run the demo with `--corners all` again. | video folder |
| 9 | LLM frontend | `python scripts\reward_from_feedback.py "power matters much more than eye margin, and we don't care about linearity"` - show the weights JSON and the rationale; mention `--reward-settings` feeds it to training. (Offline fallback runs without an API key.) | terminal |
| 10 | The two bugs | 20 s: "The agent found two measurement flaws by exploiting them - an FFT leakage floor that faked an HD3 failure, and a DFE eye that rewarded a huge tap while the receiver decided half the bits wrong. Both fixed and unit-tested; reward hacking as a bug detector." | report Sec. 10 |
| 11 | Deliverables | Repo tree, `sized_ctle.sp`, tests (`python -m pytest -q` -> 58 passed), report PDF. | terminal |
| 12 | Close | Limitations (behavioural DFE, first-order area, PDK) and next steps (analog slicer in the loop, corner-aware policy, closed LLM loop). | report Sec. 12 |

## Numbers to quote

- Policy: 20/20 (nominal), 24/24 (random corners) rollouts feasible; median 2-3 simulations, worst 4-5.
- Random search: 7.9% feasible, first feasible design #27. CMA-ES from random start: first feasible median #9.
- Best designs: eye 0.33-0.37 V after the channel, 0.9-1.4 mW, HD3 -58 to -72 dB, 45/45 PVT.
- Whole equalizer: post-DFE eye 0.51 V held during training (CTLE-only ~0.35 V).
- Throughput: 0.6 s per environment step with 8 ngspice processes; 12k steps in ~2 h on a laptop.
- HD3: reported -25 dB (measurement artefact) -> -58 dB corrected.

## Commands

```powershell
Set-Location Z:\al_nebula
.\.venv\Scripts\python.exe scripts\demo_rollout.py reports\sac-ihp-pvt --seed 7
.\.venv\Scripts\python.exe scripts\demo_rollout.py reports\sac-ihp-pvt           # fresh random start each time
.\.venv\Scripts\python.exe scripts\reward_from_feedback.py "power matters much more than eye margin, and we don't care about linearity"
.\.venv\Scripts\python.exe -m pytest -q
```
