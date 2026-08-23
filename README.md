# al_nebula

Phase 1 of AutoAnalog-RL: a parameterized IHP sg13g2 CTLE netlist and a fail-fast ngspice evaluator.

## Run

Use an environment with `numpy`, `pytest`, and the `ngspice` executable available on `PATH`.

```bash
python -m pytest
```

`SpiceEvaluator.run_simulation()` currently implements the Phase 1 `.op` and `.ac` gates. The transient PRBS gate remains a later pipeline stage.
