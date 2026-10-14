"""Environment construction and a vector-compatible single-environment adapter."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium.vector import AutoresetMode

from .training_utils import console_print


def create_training_envs(cfg):
    """Construct the configured single, synchronous, or asynchronous environment."""
    # Translates the config arguments into key word arguments for the gymnasium environment
    kwargs = build_env_kwargs_for_gym_id(cfg, cfg.env.gym_id)
    if cfg.training.num_envs == 1:
        env = gym.wrappers.RecordEpisodeStatistics(gym.make(cfg.env.gym_id, **kwargs))
        console_print(cfg, "Using non-vectorized environment path for num_envs=1")
        return SingleEnvAdapter(env)
    vector_kwargs = {"autoreset_mode": AutoresetMode.SAME_STEP}
    if "nopq" in cfg.env.gym_id.lower() and cfg.training.vectorization_mode == "async":
        vector_kwargs["shared_memory"] = False
    return gym.make_vec(
        cfg.env.gym_id,
        num_envs=cfg.training.num_envs,
        vectorization_mode=cfg.training.vectorization_mode,
        vector_kwargs=vector_kwargs,
        wrappers=[gym.wrappers.RecordEpisodeStatistics],
        **kwargs,
    )

# This adapter is included because PPOTrainer expects a vector-style interface.
class SingleEnvAdapter:
    """Minimal vector-env compatible adapter for a single non-vectorized env."""

    def __init__(self, env: gym.Env):
        self.env = env
        self.num_envs = 1
        self.single_observation_space = env.observation_space
        self.single_action_space = env.action_space

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def _batch_obs(self, obs: dict):
        return {k: np.expand_dims(v, axis=0) for k, v in obs.items()}

    def _batch_info(self, info: dict):
        """Use vector-env arrays and presence masks for each info entry."""
        batched = {}
        for key, value in info.items():
            if key == "final_obs":
                batched[key] = np.empty(1, dtype=object)
                batched[key][0] = value
            elif isinstance(value, dict):
                batched[key] = self._batch_info(value)
            elif isinstance(value, np.ndarray):
                batched[key] = np.expand_dims(value, axis=0).copy()
            elif isinstance(value, (int, float, bool, np.number)):
                batched[key] = np.asarray([value])
            else:
                batched[key] = np.empty(1, dtype=object)
                batched[key][0] = value
            batched[f"_{key}"] = np.array([True], dtype=bool)
        return batched

    def reset(self, seed=None, options=None):
        if isinstance(seed, (list, tuple, np.ndarray)):
            seed = seed[0] if len(seed) > 0 else None
        obs, info = self.env.reset(seed=seed, options=options)
        return self._batch_obs(obs), self._batch_info(info)

    def step(self, action):
        if isinstance(action, (list, tuple, np.ndarray)):
            env_action = int(action[0])
        else:
            env_action = int(action)

        obs, reward, terminated, truncated, info = self.env.step(env_action)
        if terminated or truncated:
            # Match SAME_STEP vector autoreset: main info belongs to reset obs.
            final_obs = {key: value.copy() for key, value in obs.items()}
            final_info = info
            obs, info = self.env.reset()
            info = {**info, "final_obs": final_obs, "final_info": final_info}

        return (
            self._batch_obs(obs),
            np.array([reward], dtype=np.float32),
            np.array([terminated], dtype=bool),
            np.array([truncated], dtype=bool),
            self._batch_info(info),
        )

    def value(self, strategy="degree", gamma=0.99):
        return self.env.unwrapped.value(strategy=strategy, gamma=gamma)

    def close(self):
        self.env.close()


def build_env_kwargs_for_gym_id(cfg, gym_id: str) -> dict:
    """Builds env kwargs and drops unsupported options for NoPQ environments."""
    env_kwargs = {
        "ideal_dist": cfg.env.ideal_dist,
        "elimination": cfg.env.elimination,
        "rewards": cfg.env.rewards,
        "pq_strategy": cfg.env.priority_queue_strategy,
        "sort_input": cfg.env.sort_input,
        "sort_reducers": cfg.env.sort_reducers,
        "fix_ideal": cfg.env.fix_ideal,
        "N": cfg.env.N,
        "N_P": cfg.env.N_P,
        "N_G": cfg.env.N_G,
        "k": cfg.env.k,
        "coeff_ring": cfg.env.coeff_ring,
        "symbols": cfg.env.symbols,
        "basis_elements": cfg.env.basis_elements,
    }
    if "nopq" in gym_id.lower():
        env_kwargs.pop("pq_strategy", None)
        env_kwargs.pop("N", None)
        env_kwargs.pop("N_P", None)
        env_kwargs.pop("N_G", None)
    return env_kwargs


def build_fixed_ideal_env_kwargs_for_gym_id(cfg, gym_id: str, corpus_row: dict) -> dict:
    """Builds env kwargs for evaluating a specific fixed ideal corpus row."""
    env_kwargs = {
        "ideal_dist": None,
        "elimination": cfg.env.elimination,
        "rewards": cfg.env.rewards,
        "sort_input": cfg.env.sort_input,
        "sort_reducers": cfg.env.sort_reducers,
        "fix_ideal": True,
        "k": cfg.env.k,
        "coeff_ring": corpus_row["coeff_ring"],
        "symbols": corpus_row["symbols"],
        "basis_elements": corpus_row["basis_elements"],
    }
    if "nopq" not in gym_id.lower():
        env_kwargs["pq_strategy"] = cfg.env.priority_queue_strategy
        env_kwargs["N"] = cfg.env.N
        env_kwargs["N_P"] = cfg.env.N_P
        env_kwargs["N_G"] = cfg.env.N_G
    return env_kwargs
