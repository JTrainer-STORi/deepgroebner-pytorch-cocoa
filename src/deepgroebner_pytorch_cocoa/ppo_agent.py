"""Contains functions and classes used to define the actor and critic"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from .combined_networks import (
    CombinedPolicyNetwork,
    CombinedValueNetwork,
)
from .config_dataclasses import ModelInputs


def global_grad_norm(parameters):
    """Computes the global norm of the gradients across all parameters. Used for diagnostics"""
    total_norm_sq = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(2)
        total_norm_sq += param_norm.item() ** 2
    return total_norm_sq**0.5


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal initialization for the weights and constant initialization for the biases of a given layer. By default torch uses Kaiming initialization, but the PPO paper uses orthogonal initialization, so we follow that convention. The std argument is the gain for the orthogonal initialization, and by default is set to sqrt(2) which is commonly used for ReLU activations."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CategoricalMasked(Categorical):
    """A Categorical distribution that applies a mask to the logits, setting the logits of invalid actions to a large negative value (effectively zeroing out their probabilities).
    The masks argument should be a boolean tensor of the same shape as the logits, where True indicates a valid action and False indicates an invalid action.
    Valid action logits must be finite; numerical failures raise ValueError.
    """

    def __init__(
        self, probs=None, logits=None, validate_args=None, masks=None, neg=-1e8
    ):
        if masks is None:
            self.masks = None
            if logits is not None and not torch.isfinite(logits).all():
                raise ValueError(
                    "CategoricalMasked received non-finite logits (NaN or infinity). "
                    "Check the actor inputs, outputs, and parameters for numerical instability."
                )
            super().__init__(probs=probs, logits=logits, validate_args=validate_args)
            return

        m = masks.to(dtype=torch.bool, device=logits.device)
        self.masks = m
        if (m & ~torch.isfinite(logits)).any():
            raise ValueError(
                "CategoricalMasked received non-finite logits (NaN or infinity) for valid actions. "
                "Check the actor inputs, outputs, and parameters for numerical instability."
            )
        masked = torch.where(m, logits, logits.new_full((), neg))
        super().__init__(probs=probs, logits=masked, validate_args=validate_args)

    def entropy(self):
        if self.masks is None:
            return super().entropy()
        p_log_p = self.probs * self.logits
        p_log_p = torch.where(self.masks, p_log_p, p_log_p.new_zeros(()))
        return -p_log_p.sum(-1)

# This is used as an optional diagnostic that tries to account for the fact that the number of available actions varies between time steps and as a result the policies applied to states with more actions may have 
def normalized_masked_entropy(
    entropy: torch.Tensor, action_mask: torch.Tensor, epsilon: float = 1e-8
) -> torch.Tensor:
    """Normalizes masked categorical entropy by log(number of valid actions)."""
    valid_counts = action_mask.to(dtype=entropy.dtype).sum(dim=-1)
    normalizer = torch.log(valid_counts.clamp_min(1.0))
    normalized = torch.where(
        valid_counts > 1.0,
        entropy / normalizer.clamp_min(epsilon),
        torch.zeros_like(entropy),
    )
    return normalized


class Agent(nn.Module):
    """Own independent actor and critic networks for one training run."""

    def __init__(self, cfg, envs=None):
        super().__init__()
        self.cfg = cfg
        self.envs = envs
        self.critic = (
            CombinedValueNetwork(cfg.critic)
            if cfg.critic.critic_strategy != "heuristic-only"
            else None
        )
        self.actor = CombinedPolicyNetwork(cfg.actor)

    def _heuristic_value(self, inputs):
        if inputs.h_value is not None:
            return inputs.h_value
        cfg = self.cfg
        envs = self.envs
        if envs is None:
            raise ValueError(
                "Heuristic values require environments or cached inputs.h_value."
            )
        device = inputs.G.device
        if isinstance(envs, gym.vector.SyncVectorEnv):
            values = torch.zeros(inputs.G.shape[0]).to(device)
            for index, env in enumerate(envs.envs):
                values[index] = torch.tensor(
                    env.unwrapped.value(
                        strategy=cfg.critic.value_function_heuristic,
                        gamma=cfg.training.gamma,
                    )
                ).to(device)
        elif isinstance(envs, gym.vector.AsyncVectorEnv):
            values = torch.tensor(
                envs.call(
                    "value",
                    strategy=cfg.critic.value_function_heuristic,
                    gamma=cfg.training.gamma,
                )
            ).to(device)
        else:
            values = torch.tensor(
                [
                    envs.value(
                        strategy=cfg.critic.value_function_heuristic,
                        gamma=cfg.training.gamma,
                    )
                ],
                device=device,
            )
        return values.unsqueeze(-1)

    def get_value(self, inputs):
        """Return full values and the optional heuristic component for this agent."""
        strategy = self.cfg.critic.critic_strategy
        if strategy == "learned-only":
            values, _, _ = self.critic(inputs, h_value=None)
            return values, None
        heuristic = self._heuristic_value(inputs)
        if strategy == "heuristic-only":
            return heuristic, heuristic
        if strategy == "heuristic-plus-learned-residual":
            values, _, _ = self.critic(inputs, h_value=heuristic)
            return values, heuristic
        raise ValueError(f"Unknown critic strategy: {strategy}")

    def get_action_and_value(self, inputs: ModelInputs, action=None):
        logits = self.actor(inputs)
        logits = logits.squeeze(-1)
        probs = CategoricalMasked(logits=logits, masks=inputs.P_mask)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), *self.get_value(inputs)
