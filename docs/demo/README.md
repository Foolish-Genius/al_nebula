# Live-sizing demo page

`autoanalog_demo.html` is a self-contained dashboard that replays a **real policy rollout**:
every number, waveform, eye trace and PVT cell on the page came out of ngspice, not a mock-up.
Double-click it — no server, no internet, no Python needed.

## What it shows

| Panel | Content |
|---|---|
| Design under test | the CTLE schematic and the five sized values, each on its bounded range, with the change the policy made this step |
| Eye diagram | 2-UI eye after the −10 dB channel, with the *required opening* drawn as a box, redrawn every simulation |
| Target specification | each spec with its measured value, target and PASS/FAIL; HD3 shows "not run" while fail-fast skips it |
| Verdict + netlist | the reward, and the sized `sized_ctle.sp` once the design passes |
| AC response | CTLE gain with the Nyquist marker and the measured peaking |
| Reward per simulation | the rollout so far; green dot = every spec met |
| Bottom-right | the training curve before the design passes, the 45-corner PVT grid after it |

## Controls

`Space` play/pause · `→` next simulation · `←` previous · Restart · speed 1× / 1.6× / 2.6×.
The page autoplays 1.2 s after loading, which is what you want for a screen recording.

URL hash options (useful for screenshots and for starting a take mid-rollout):

```
autoanalog_demo.html#step=1&auto=0     open on the first simulation, do not autoplay
autoanalog_demo.html#step=3            open on the final design
```

## Recording it

1. Open the file in Chrome, press `F11` for full screen (1920×1080 records cleanly).
2. Start your recorder, then press `R` on the page… actually: click **Restart**, then **Play**.
3. The rollout takes ~8 s at 1× — long enough to narrate "random start → specs failing → two moves → all green".
4. For a slower take, set speed to 1× and use `→` manually so the narration drives the pace.

## Regenerating with a different rollout

```bash
# any trained run, any starting seed; --pvt adds the 45-corner sweep of the final design
python scripts/export_demo_data.py reports/sac-ihp-pvt --seed 21 --pvt \
    --training-from reports/sac-ihp-v1 --output docs/demo/rollout.json
python scripts/build_demo.py --output docs/demo/autoanalog_demo.html
```

`--seed` picks the random starting design and PVT corner; a seed whose first step fails on
more specs makes a better demo. `template.html` is the editable source; `build_demo.py`
inlines `rollout.json` into it so the result is one portable file.
