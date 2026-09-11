# Workspace Map

## Active implementation

- `netlists/ctle_template.sp`: active CTLE SPICE topology.
- `spice/spice_engine.py`: active ngspice/IHP evaluator and measurement gates.
- `rl/equalizer.py`: six-action whole-equalizer boundary.
- `rl/dfe.py`: one-tap sampled DFE model.
- `scripts/run_validation.py`: reproducible generic/IHP pipeline entry point.
- `analysis/reporting.py`: CSV, JSON, and PNG artifact writer.
- `tests/`: active regression suite.

## Reference-only material

- `DFE/DFE_template.sp`: analog DFE topology reference; not the closed-loop active evaluator.
- `circuits/`: supplied circuit diagrams.
- `updated codes/`: uploaded historical alternatives; not imported by active code.
- `scripts/run_validation_new.py`: uploaded alternative; use `scripts/run_validation.py`.
- `papers/`: supplied research PDFs and implementation notes.

## Final evidence

- `reports/runs/ihp-submission/`: latest real-IHP pre-ML evidence.
- Older `reports/runs/*` directories are preserved historical runs.