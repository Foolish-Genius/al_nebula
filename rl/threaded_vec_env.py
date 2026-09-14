"""Thread-parallel VecEnv: ngspice runs in subprocesses that release the GIL.

SubprocVecEnv spawns one Python per environment, and on Windows every worker
re-imports stable_baselines3 and therefore torch; eight CUDA torch loads at once
exhaust the DLL initialisation budget (WinError 1114). Simulation time here is
spent inside ``subprocess.run`` waiting on ngspice, so threads give the same
parallelism without any extra torch processes.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Sequence

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv


class ThreadedVecEnv(DummyVecEnv):
    """DummyVecEnv whose ``step_wait`` runs each environment's step on a thread."""

    def __init__(self, env_fns: Sequence[Callable[[], Any]]) -> None:
        super().__init__(list(env_fns))
        self._pool = ThreadPoolExecutor(max_workers=len(self.envs))

    def _step_one(self, env_idx: int) -> None:
        obs, self.buf_rews[env_idx], terminated, truncated, self.buf_infos[env_idx] = self.envs[env_idx].step(
            self.actions[env_idx]
        )
        self.buf_dones[env_idx] = terminated or truncated
        self.buf_infos[env_idx]["TimeLimit.truncated"] = truncated and not terminated
        if self.buf_dones[env_idx]:
            self.buf_infos[env_idx]["terminal_observation"] = obs
            obs, self.reset_infos[env_idx] = self.envs[env_idx].reset()
        self._save_obs(env_idx, obs)

    def step_wait(self):
        list(self._pool.map(self._step_one, range(self.num_envs)))
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), list(self.buf_infos))

    def close(self) -> None:
        self._pool.shutdown(wait=False)
        super().close()
