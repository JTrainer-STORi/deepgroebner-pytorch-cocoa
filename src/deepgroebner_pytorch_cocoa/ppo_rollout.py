"""Rollout storage and minibatch construction for fixed and ragged observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config_dataclasses import ModelInputs
from .model_input_utils import collate_ragged_model_inputs

OBSERVATION_FIELDS = (
    "G",
    "P",
    "G_mask",
    "P_mask",
    "state_features",
)
MASK_FIELDS = {"G_mask", "P_mask"}


def make_buffer(shape, dtype=torch.float32, *, num_steps, num_envs, device):
    """Allocate a rollout tensor without depending on a running trainer."""
    return torch.zeros((num_steps, num_envs) + tuple(shape), dtype=dtype, device=device)


@dataclass
class RolloutBatch:
    observations: dict
    actions: torch.Tensor
    logprobs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor
    heuristic_values: torch.Tensor | None
    ragged: bool
    device: torch.device
    pair_metadata: dict[str, np.ndarray] | None = None

    def model_inputs(self, indices) -> ModelInputs:
        """Select samples in rollout order, padding only the requested ragged samples."""
        obs = self.observations
        if self.ragged:
            inputs = collate_ragged_model_inputs(
                indices=indices,
                device=self.device,
                obs_G_list=obs["G"],
                obs_P_list=obs["P"],
                g_mask_list=obs["G_mask"],
                p_mask_list=obs["P_mask"],
                obs_sf_list=obs["state_features"],
            )
        else:
            inputs = ModelInputs(
                **{
                    name: values[indices] if values is not None else None
                    for name, values in obs.items()
                }
            )
        if self.heuristic_values is not None:
            inputs.h_value = self.heuristic_values[indices].unsqueeze(-1)
        return inputs

    def ranking_metadata(self, indices) -> dict[str, torch.Tensor]:
        """Select diagnostic metadata in the same order as the model minibatch."""
        if self.pair_metadata is None:
            return {}
        if torch.is_tensor(indices):
            indices = indices.detach().cpu().numpy()
        names = {
            "P_idx_raw": "pair_idx_raw",
            "P_sugar": "pair_sugar",
            "P_normal_rank": "pair_normal_rank",
        }
        return {
            names[name]: torch.as_tensor(values[indices], device=self.device)
            for name, values in self.pair_metadata.items()
        }


class RolloutBuffer:
    """This buffer is used to store all of the rollout data necessary for policy updates using ppo for one batch of transitions"""

    def __init__(self, cfg, envs, device, *, ragged):
        # If the environment doesn't use a priority queue then how rollout data is stored differs
        self.ragged = ragged
        self.device = device
        self.observations = {}
        self.action_shape = envs.single_action_space.shape
        space = envs.single_observation_space
        for name in OBSERVATION_FIELDS:
            if ragged:
                # In this case, observations are stored as lists
                self.observations[name] = []
            else:
                # In this case, observations are stored as tensors
                dtype = torch.bool if name in MASK_FIELDS else torch.float32
                self.observations[name] = make_buffer(
                    space[name].shape,
                    dtype,
                    num_steps=cfg.training.num_steps_per_update,
                    num_envs=cfg.training.num_envs,
                    device=device,
                )

        # Keep optional diagnostic snapshots on the CPU, separate from model data.
        self.pair_metadata = None
        if cfg.training.ranking_diagnostic_frequency > 0 and not ragged:
            shape = (cfg.training.num_steps_per_update, cfg.training.num_envs)
            self.pair_metadata = {
                "P_idx_raw": np.zeros(
                    shape + space["P_mask"].shape + (2,), dtype=np.int64
                ),
                "P_sugar": np.zeros(shape + space["P_mask"].shape, dtype=np.float32),
                "P_normal_rank": np.zeros(
                    shape + space["P_mask"].shape, dtype=np.float32
                ),
            }

        # For both the ragged buffer and priority queue buffer, this data will be stored as tensors, so iterate through and initialize the tensors.
        for name, shape in (
            ("actions", self.action_shape),
            ("logprobs", ()),
            ("rewards", ()),
            ("dones", ()),
            ("values", ()),
        ):
            setattr(
                self,
                name,
                make_buffer(
                    shape,
                    num_steps=cfg.training.num_steps_per_update,
                    num_envs=cfg.training.num_envs,
                    device=device,
                ),
            )
        # If we're using a heuristic plus learned residual critic, then the values returned by the heuristic must be stored to add to the residual adjustment.
        self.heuristic_values = (
            make_buffer(
                (),
                num_steps=cfg.training.num_steps_per_update,
                num_envs=cfg.training.num_envs,
                device=device,
            )
            if cfg.critic.critic_strategy == "heuristic-plus-learned-residual"
            else None
        )

    def clear(self):
        if self.ragged:
            for values in self.observations.values():
                if values is not None:
                    values.clear()

    def store_observation(self, step, inputs, dones, *, info=None):
        for name, values in self.observations.items():
            if values is None:
                continue
            value = getattr(inputs, name)
            if self.ragged:
                values.extend(sample.detach().clone() for sample in value)
            else:
                values[step].copy_(value)
        if self.pair_metadata is not None:
            if info is None:
                raise ValueError(
                    "Ranking diagnostics require metadata from environment info."
                )
            for name, values in self.pair_metadata.items():
                np.copyto(values[step], info[name])
        self.dones[step].copy_(dones)

    def store_action(self, step, action, logprob, value, heuristic_value):
        self.actions[step].copy_(action)
        self.logprobs[step].copy_(logprob)
        self.values[step].copy_(value.flatten())
        if self.heuristic_values is not None:
            self.heuristic_values[step].copy_(heuristic_value.flatten())

    def flatten(self, advantages, returns) -> RolloutBatch:
        observations = {
            name: (
                values
                if self.ragged or values is None
                else values.reshape((-1,) + values.shape[2:])
            )
            for name, values in self.observations.items()
        }
        return RolloutBatch(
            observations=observations,
            actions=self.actions.reshape((-1,) + self.action_shape),
            logprobs=self.logprobs.reshape(-1),
            advantages=advantages.reshape(-1),
            returns=returns.reshape(-1),
            values=self.values.reshape(-1),
            heuristic_values=(
                self.heuristic_values.reshape(-1)
                if self.heuristic_values is not None
                else None
            ),
            ragged=self.ragged,
            device=self.device,
            pair_metadata=(
                {
                    name: values.reshape((-1,) + values.shape[2:])
                    for name, values in self.pair_metadata.items()
                }
                if self.pair_metadata is not None
                else None
            ),
        )
