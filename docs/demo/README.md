# AutoAnalog-RL demo pages

Three pages, for three different jobs:

| Page | Use it for |
|---|---|
| **`autoanalog_studio.html`** | the **interactive demo** — a judge names a specification and watches the agent size the circuit against real ngspice. See [`studio/README.md`](../../studio/README.md). |
| `autoanalog_demo.html` | a 15-screen walkthrough of the whole project, light blue, advanced with Next |
| `autoanalog_demo_dark.html` | the same walkthrough in a warm dark palette, plus two screens that open the architecture up layer by layer |

The studio is the one to present. The walkthroughs are for explaining the project around it.

---

# AutoAnalog-RL walkthrough site


`autoanalog_demo.html` is a self-contained, 15-screen walkthrough of the whole project — problem,
specification, circuit, method, reward, agent, a **live rollout**, results, baselines, PVT,
extensions, the measurement bugs, deliverables. One screen per idea, advanced with **Next**.
Double-click it: no server, no internet, no Python needed.

Every number, waveform, eye trace and PVT cell on the live screen came out of ngspice on the real
IHP PSP103 models — it is a replayed rollout, not a mock-up.

## The screens

| # | Screen | Content |
|---|---|---|
| 1 | Hero | what the project is, in four numbers |
| 2 | The problem | why sweeping five coupled knobs against seven specs does not work |
| 3 | The target | the specification table, signalling and technology |
| 4 | The circuit | the CTLE schematic and the five sized parameters with their ranges |
| 5 | The method | the closed loop, and the four fail-fast simulation gates |
| 6 | The reward | the reward equation and what each term buys |
| 7 | The agent | SAC, the 15-number observation, the action, and the training curve |
| 8 | **Live** | the rollout — knobs, eye diagram, spec checklist, AC response, reward |
| 9 | Results | the seven runs |
| 10 | Comparison | reward-vs-simulations against CMA-ES and random search |
| 11 | Robustness | the 45-corner grid and the two-stage curriculum |
| 12 | Extensions | tunable Nyquist, the DFE, plain-language reward weights |
| 13 | Findings | the HD3 window bug and the DFE eye bug, both found by being exploited |
| 14 | Deliverables | the sized netlist and what ships with each run |
| 15 | Close | the summary |

## Controls

`→` / `Space` / `Enter` next · `←` back · `Home` / `End` jump to the ends · click any progress dot.

On screen 8 the Next button becomes **Run next simulation** and steps through the ten simulations of
the rollout one at a time; after the last one it moves on to screen 9. Back works the same way in
reverse. Nothing autoplays — the narration drives the pace, which is what you want for a recording.

URL hash options, for screenshots and for restarting a take mid-deck:

```
autoanalog_demo.html#s=live          open on the live screen
autoanalog_demo.html#s=live&sim=3    open on the live screen at simulation 3
autoanalog_demo.html#s=10            open on screen 10
```

## Recording it

1. Open in Chrome, `F11` for full screen (1920×1080 records cleanly).
2. Press `→` to advance. Each screen is one beat of narration.
3. The live screen is the centrepiece: simulation 1 fails on eye height and width, simulation 3
   passes everything, and simulations 4–10 grow the margin. The caption under the knobs says in
   plain language what the policy changed each step.

## Regenerating with a different rollout

```bash
python scripts/export_demo_data.py reports/sac-ihp-pvt --seed 21 --hold --max-steps 10 --pvt \
    --training-from reports/sac-ihp-v1 \
    --baselines "CMA-ES=reports/cmaes-ihp-random-s10" "random search=reports/baseline-ihp-3000" \
    --output docs/demo/rollout.json
python scripts/build_demo.py --output docs/demo/autoanalog_demo.html
```

- `--seed` picks the random starting design and PVT corner.
- `--hold` keeps simulating after the specs pass, exactly as the policy was trained
  (`--hold-on-success`), so the page shows margin growing instead of stopping at the first
  feasible design. Without it the rollout ends in 2–4 simulations.
- `--pvt` runs the 45-corner sweep of the final design for screen 11.
- `--baselines` draws other search runs on the same simulation axis for screen 10.
- `--training-from` picks the run whose learning curve is shown on screen 7 (use a from-scratch
  run; a resumed curriculum run starts already converged and looks flat).

`template.html` is the editable source; `build_demo.py` inlines `rollout.json` into it at the
`const DATA = /*__DATA__*/ null;` marker so the result is one portable file. After editing the
template, syntax-check the built page's script before trusting it.

---

## `autoanalog_demo_dark.html` — the dark variant

Same rollout data, a different presentation: a warm dark palette (brown-black paper, cream text,
clay accent) and **two extra screens built around the architecture**, for a total of 16.

| # | Screen | |
|---|---|---|
| 5 | **The architecture** | the system as a seven-layer device that opens up. Each Next lifts the next layer off the stack in an isometric exploded view; the panel on the right explains that layer and names the file it lives in. Click any layer, or any row of the index, to jump to it. |
| 6 | **The architecture, running** | one environment step traced down the stack and back — specification → policy → environment → evaluator → ngspice → reward → replay buffer — carrying the **real numbers from simulation 3** at every stage: the actions the actor emitted, the device values they mapped to, what each ngspice gate measured, and the reward that came back. |

The seven layers: Specification · SAC policy · Environment · Reward model · Evaluator ·
ngspice + OSDI · The circuit.

Build it from its own template, which shares `rollout.json` with the light page:

```bash
python scripts/build_demo.py --template docs/demo/template_dark.html \
    --output docs/demo/autoanalog_demo_dark.html
```

Hash options are the same, plus `&lay=N` for the exploded view and `&tr=N` for the trace:

```
autoanalog_demo_dark.html#s=stack&lay=7    fully exploded, bottom layer selected
autoanalog_demo_dark.html#s=trace&tr=5     the trace paused at the ngspice gates
```

Use whichever suits the room — the light page for a projector, the dark one for a screen
recording. Both are self-contained and neither depends on the other.

---

## On a phone

Both pages are responsive. Below 860 px the layout collapses to a single column: the multi-column
grids stack, the rollout's knobs / eye / specs become three stacked panels, the exploded device
scales down with its layer index underneath as a two-column list, and the trace rail moves above
the stage content. Below 430 px everything goes to one column and the type steps down again.

Each screen scrolls vertically if its content is taller than the viewport, and **swiping left and
right advances and rewinds** — the same as `→` and `←`, so on the live screen a swipe runs the
next simulation. The Back/Next buttons stay pinned at the bottom.

Verified with no horizontal overflow on every screen at 390 px.
