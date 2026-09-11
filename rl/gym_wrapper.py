"""Gymnasium adapter so off-the-shelf agents can drive CtleEnvironment."""

from __future__ import annotations

from typing import Any

import numpy as np

from .environment import CtleEnvironment


def make_gym_env(environment: CtleEnvironment) -> Any:
    """Wrap a CtleEnvironment in a gymnasium.Env; imported lazily so the core stays dependency-free."""
    try:
        import gymnasium
    except ImportError as error:  # pragma: no cover - exercised only without the rl extra
        raise ImportError("install the 'rl' extra: pip install -e .[rl]") from error

    class CtleGymEnv(gymnasium.Env):
        metadata = {"render_modes": []}

        def __init__(self, inner: CtleEnvironment) -> None:
            super().__init__()
            self.inner = inner
            self.action_space = gymnasium.spaces.Box(-1.0, 1.0, shape=(inner.ACTION_SIZE,), dtype=np.float32)
            self.observation_space = gymnasium.spaces.Box(
                -np.inf, np.inf, shape=(inner.observation_size,), dtype=np.float32
            )

        def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
            super().reset(seed=seed)
            return self.inner.reset(seed=seed, options=options)

        def step(self, action: np.ndarray):
            # SAC samples float32 actions that can sit a rounding error outside the box.
            clipped = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
            return self.inner.step(clipped)

    return CtleGymEnv(environment)
