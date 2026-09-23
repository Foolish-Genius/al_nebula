# AutoAnalog-RL Studio

Type a specification, watch the agent size the circuit, get a netlist.

This is the demo to present: a judge names a target, the trained policy drives real ngspice on the
IHP PSP103 models, and the eye opens simulation by simulation until every specification passes. It
is not a slideshow of a recorded result — in live mode nothing is precomputed.

```
python -m studio.server          # then open http://localhost:8765
```

## What the page does

| Panel | |
|---|---|
| **1 Specify** | four sliders (eye height, eye width, power budget, HF peaking), a PVT corner, five presets, and a plain-language box: *"low-power link, eye at least 0.3 V, worst-case corner"* sets the sliders |
| **2 The agent at work** | the eye diagram redrawn after every simulation with the required opening as a dashed box, the five device values with their deltas, the frequency response with the previous curve ghosted, and a log that says in words what the agent changed and what it is still short on |
| **3 Result** | the specification checklist flipping to pass, the sized design, a 45-corner PVT sweep filling in live, a netlist download, and a tape-out sheet with the annotated schematic, margins and corner heatmap |
| **Race** | the same specification given to CMA-ES and to random search, one simulator each, running alongside the agent |

If a specification is out of reach the page says so: it names the **binding** specification, shows the
closest value reached, and offers to relax it to an achievable number and retry. Ask for a 0.45 V eye
and it will tell you the circuit tops out near 0.35 V at that corner.

## Live or replay

The page checks for a local engine on load and shows which mode it is in.

- **LIVE** — `python -m studio.server` is running. Any specification, any corner; every number comes
  out of ngspice during the demo. About 4 s per simulation, so a run takes 20–60 s.
- **REPLAY** — no server. The page falls back to a library of **real runs recorded ahead of time**
  over a grid of specifications. Sliders snap to the recorded grid, one simulation plays per second.
  Nothing to go wrong on stage, and still genuine ngspice data.

Replay is the safety net: build the library, and `docs/demo/autoanalog_studio.html` works on any
machine by double-clicking it, with no Python, no server and no internet.

## Recording the replay library

```
python -m studio.build_library --workers 4     # ~80 real runs, about 40 minutes
python -m studio.build_page                    # inlines it into the page
```

Each run is cached in `reports/studio-library/` as it finishes, so an interrupted build resumes.
The grid is `GRID` in `build_library.py`: eye height, power, eye width and peaking. Widening it
multiplies the recording time.

## Plain language

With `ANTHROPIC_API_KEY` set, the plain-language box asks Claude to turn the request into
specification targets. Without it the page uses a keyword parser that understands numbers with units
("0.3 V", "1.2 mW", "5 dB") and phrases like *low power*, *wide eye*, *lossy channel*, *worst case*.

This is honest because the agent genuinely reads specification targets — they reach it through the
target-normalised violations in its observation. Reward *weights* are a training-time knob and would
not change anything at inference, so the page never pretends otherwise.

## Why the agent can do this at all

The policy was trained at one specification (eye ≥ 0.25 V, power ≤ 2 mW) but the targets enter its
observation only as normalised violations, so it transfers to targets it never saw. Measured on the
real models, 8/8 rollouts reach feasibility at eye ≥ 0.30 V and ≥ 0.35 V, at eye width ≥ 0.85 UI, at
peaking ≥ 5 dB and at power ≤ 1.2 mW. It fails at eye ≥ 0.40 V (best 0.43 V) and power ≤ 0.8 mW
(best 0.94 mW) — which is what the reality check is for.

## Recording a take

URL options let a take start itself, so there is no mouse in the frame:

```
#preset=1&run          open on "Low power" and start immediately
#preset=4&run          the reality check ("Push it too far")
#race=0&run            skip the baselines, agent only
#ask=low power link, wide eye&run      fill the plain-language box, parse it, run
#tape                  open the tape-out sheet as soon as the run finishes
```

Presets in order: `0` PCIe Gen 2 baseline, `1` Low power, `2` Wide eye, `3` Lossy channel,
`4` Push it too far. `Ctrl+Enter` starts a run, `Esc` closes the tape-out sheet.

## Layout

```
studio/engine.py         the run as an event stream; both modes consume the same events
studio/server.py         live backend: /api/health, /api/run (SSE), /api/parse
studio/build_library.py  records real runs over a specification grid
studio/build_page.py     inlines the library into the page
studio/web/index.html    the page source
docs/demo/autoanalog_studio.html   the built page (served live, or opened directly for replay)
```

One run at a time: starting a new one stops the run in progress, so a judge can interrupt and ask
for something else.
