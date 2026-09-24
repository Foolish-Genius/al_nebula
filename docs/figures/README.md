# Review figures

Figures answering the reviewer feedback of 15 September 2026. They are the source of truth for
the three backup slides in the deck (`scripts/build_deck.py` reads them from here).

| Figure | Answers | What it shows |
|---|---|---|
| `peaking_overlay.png` | "Provide the waveforms for the multiple peaking specs" | The CTLE response the trained policy produces for peaking floors of 3, 5 and 7 dB, with the peak marked on each |
| `dfe_before_after.png` | "Provide the waveform with and without DFE" | The eye leaving the CTLE, with the sampling instants marked, beside the same samples after the one-tap DFE |
| `dfe_levels.png` | "Mismatch between +ve and -ve swing in the DFE waveform" | The four DFE output levels, which mirror exactly once the eye is measured against the transmitted bits |

`peaking_overlay.json` carries the numbers behind the first figure.

## Regenerating

```bash
python scripts/make_review_figures.py              # both
python scripts/make_review_figures.py --only dfe   # no simulator needed
```

The DFE figures are computed from `reports/eval-ihp-eq/` and need nothing but those recorded
artifacts. The peaking overlay rolls the trained policy out against each floor, so it needs ngspice
and the IHP PDK, and takes a couple of minutes.

## What the peaking overlay says

| Specification | Feasible at | At Nyquist | Max − DC | Peak |
|---|---|---|---|---|
| ≥ 3 dB | simulation 2 | 5.38 dB | 5.41 dB | 2.88 GHz |
| ≥ 5 dB | simulation 3 | 6.06 dB | 6.17 dB | 3.31 GHz |
| ≥ 7 dB | not reached | 6.72 dB | 7.09 dB | 3.98 GHz |

The policy was trained only at 3 dB. As the floor rises it trades DC gain away (11.8 → 9.3 dB) and
pushes the peak higher in frequency, which is the correct response to a specification it never saw.

The 7 dB row is the measurement question in one line: that design reaches **7.09 dB by the
maximum-minus-DC definition but only 6.72 dB at Nyquist**, so our specification calls it a failure
while the reviewer's definition would pass it. Peaking is enforced at Nyquist; `peaking_max` is
reported alongside it.

## A caveat on the DFE eye numbers

`dfe_before_after.png` reports 324 → 346 mV, while `reports/eval-ihp-eq/validation.json` records
324.5 → 355.5 mV. The figure picks the sampling phase by maximising separation; the validation path
aligns against the transmitted bit sequence (`rl.dfe.dfe_eye_against_bits`). Same conclusion, about
10 mV apart. Making the figure match exactly would mean saving the transmitted bits alongside the
transient.
