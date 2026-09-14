# Workspace Map

## Active implementation

- `netlists/ctle_template.sp`: active CTLE SPICE topology.
- `spice/spice_engine.py`: active ngspice/IHP evaluator and measurement gates.
- `rl/equalizer.py`: six-action whole-equalizer boundary.
- `rl/dfe.py`: one-tap sampled DFE model.
- `rl/specs.py`, `rl/reward.py`, `rl/environment.py`: spec margins, reward shaping, Gym-style env.
- `rl/gym_wrapper.py`, `rl/threaded_vec_env.py`: gymnasium adapter and thread-parallel VecEnv.
- `scripts/run_validation.py`: reproducible generic/IHP pipeline entry point.
- `scripts/train_sac.py`, `scripts/sac_progress.py`: SAC training and run summaries.
- `scripts/evaluate_policy.py`: deterministic rollouts of a checkpoint plus the random-search comparison.
- `tools/openvaf-link-shim/`: builds the IHP OSDI models on Windows without MSVC.
- `scripts/baseline_random.py`: random-search baseline at the same simulation budget.
- `analysis/reporting.py`: CSV, JSON, and PNG artifact writer.
- `tests/`: active regression suite.

## Reference-only material

- `DFE/DFE_template.sp`: analog DFE topology reference; not the closed-loop active evaluator.
- `circuits/`: supplied circuit diagrams.
- `papers/`: supplied research PDFs and implementation notes.

## Final evidence

- `reports/runs/ihp-submission/`: latest real-IHP pre-ML evidence.
- `reports/sac-*/`, `reports/baseline-*/`: SAC and random-search runs (git-ignored).
- Older `reports/runs/*` directories are preserved historical runs.