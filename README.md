# al_nebula

Phase 1 of AutoAnalog-RL: a parameterized IHP sg13g2 CTLE netlist, fail-fast ngspice evaluator, and RL specification boundary.

## Architecture

- `netlists/` owns circuit topology and injectable SPICE parameters.
- `spice/` owns simulator execution, parsing, and DC/AC gating.
- `rl/specs.py` owns measurable targets and normalized constraint violations.
- `rl/reward.py` owns reward shaping and configurable weights.
- `rl/environment.py` owns the Gym-style `reset`/`step` contract.
- `rl/pvt.py` owns the deterministic 45-corner PVT verification matrix.

The next high-value stages are transient PRBS and eye metrics, PVT-aware simulator
injection, then SAC integration. The simulator adapter should remain unchanged
while the scheduling and optimization layers are added.

## Run

Use an environment with `numpy`, `pytest`, and the `ngspice` executable available on `PATH`.

```bash
python -m pytest
```

`SpiceEvaluator.run_simulation()` currently implements the Phase 1 `.op` and `.ac` gates. The transient PRBS gate remains a later pipeline stage.
