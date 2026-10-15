"""Running statistics and output-preserving normalization for PPO."""

from __future__ import annotations

import torch
import torch.nn as nn
from .config_dataclasses import Config, ModelInputs


class RunningMeanStd:
    """Tracks running mean/variance for vector features using a numerically stable merge."""

    def __init__(self, shape=(), epsilon=1e-4, device=None):
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = torch.tensor(epsilon, dtype=torch.float32, device=device)

    def update(self, x: torch.Tensor):
        x = x.to(dtype=torch.float32)
        if x.numel() == 0:
            return
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = torch.tensor(x.shape[0], dtype=torch.float32, device=x.device)
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x: torch.Tensor, epsilon=1e-8, clip_range: float | None = None):
        std = torch.sqrt(self.var + epsilon)
        y = (x - self.mean) / std
        if clip_range is not None and clip_range > 0:
            y = torch.clamp(y, -clip_range, clip_range)
        return y


class PopArt:
    """PopArt normalizer that preserves raw value outputs when target stats change."""

    def __init__(
        self,
        output_layer: nn.Linear,
        shape=(),
        epsilon=1e-8,
        min_std: float = 1e-4,
        device=None,
    ):
        if not isinstance(output_layer, nn.Linear):
            raise TypeError("PopArt requires an nn.Linear output layer.")

        stat_shape = (
            torch.Size(shape)
            if isinstance(shape, tuple)
            else torch.Size((shape,)) if shape else torch.Size(())
        )
        flat_dim = (
            int(torch.tensor(stat_shape).prod().item()) if len(stat_shape) > 0 else 1
        )
        if output_layer.out_features != flat_dim:
            raise ValueError(
                f"PopArt stats shape {tuple(stat_shape)} does not match output layer "
                f"out_features={output_layer.out_features}."
            )

        self.output_layer = output_layer
        self.epsilon = epsilon
        self.min_std = min_std
        self.mean = torch.zeros(stat_shape, dtype=torch.float32, device=device)
        self.var = torch.ones(stat_shape, dtype=torch.float32, device=device)
        self.count = torch.tensor(1e-4, dtype=torch.float32, device=device)

    def _std(self) -> torch.Tensor:
        return torch.sqrt(self.var + self.epsilon).clamp_min(self.min_std)

    def normalize(
        self, x: torch.Tensor, clip_range: float | None = None
    ) -> torch.Tensor:
        y = (x - self.mean) / self._std()
        if clip_range is not None and clip_range > 0:
            y = torch.clamp(y, -clip_range, clip_range)
        return y

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self._std() + self.mean

    def update(self, targets: torch.Tensor):
        targets = targets.to(dtype=torch.float32)
        if targets.numel() == 0:
            return

        old_mean = self.mean.clone()
        old_std = self._std().clone()

        batch_mean = targets.mean(dim=0)
        batch_var = targets.var(dim=0, unbiased=False)
        batch_count = torch.tensor(
            targets.shape[0], dtype=torch.float32, device=targets.device
        )

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

        new_std = self._std()
        self._preserve_outputs(old_mean, old_std, new_mean, new_std)

    def _preserve_outputs(
        self,
        old_mean: torch.Tensor,
        old_std: torch.Tensor,
        new_mean: torch.Tensor,
        new_std: torch.Tensor,
    ):
        old_mean_flat = old_mean.reshape(-1)
        old_std_flat = old_std.reshape(-1)
        new_mean_flat = new_mean.reshape(-1)
        new_std_flat = new_std.reshape(-1)

        with torch.no_grad():
            scale = (old_std_flat / new_std_flat).unsqueeze(-1)
            self.output_layer.weight.mul_(scale)
            if self.output_layer.bias is not None:
                self.output_layer.bias.copy_(
                    (
                        old_std_flat * self.output_layer.bias
                        + old_mean_flat
                        - new_mean_flat
                    )
                    / new_std_flat
                )


class RewardNormalizer:
    """Normalizes rewards using running statistics of discounted returns."""

    def __init__(self, num_envs, gamma, epsilon=1e-8, clip_range=10.0, device=None):
        self.gamma = gamma
        self.epsilon = epsilon
        self.clip_range = clip_range
        self.returns = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.rms = RunningMeanStd(shape=(), epsilon=1e-4, device=device)

    def normalize(self, rewards: torch.Tensor, dones: torch.Tensor):
        """Scale this transition's rewards, then reset returns for finished episodes.

        ``dones`` describes the transition producing ``rewards``, so its terminal
        reward contributes to the complete episode before the accumulator resets.
        """
        self.returns = self.returns * self.gamma + rewards
        self.rms.update(self.returns)
        self.returns = self.returns * (1.0 - dones.to(dtype=torch.float32))
        std = torch.sqrt(self.rms.var + self.epsilon)
        normalized = rewards / std
        if self.clip_range is not None and self.clip_range > 0:
            normalized = torch.clamp(normalized, -self.clip_range, self.clip_range)
        return normalized


def _find_last_linear(module: nn.Module) -> nn.Linear:
    for child in reversed(list(module.modules())):
        if isinstance(child, nn.Linear):
            return child
    raise ValueError("Could not find a final nn.Linear layer for PopArt.")


def build_value_normalizer(cfg: Config, agent: nn.Module, device=None) -> PopArt | None:
    """Builds and attaches PopArt only when the run has a learned critic."""
    if (
        not cfg.training.normalize_value_function
        or cfg.critic.critic_strategy == "heuristic-only"
    ):
        return None

    value_normalizer = PopArt(
        output_layer=_find_last_linear(agent.critic.value_head),
        shape=(),
        epsilon=cfg.training.value_normalization_epsilon,
        min_std=cfg.training.value_normalization_epsilon,
        device=device,
    )
    agent.critic.popart = value_normalizer
    return value_normalizer


def normalize_eval_state_features(
    model_input: ModelInputs,
    obs_sf_normalizer: RunningMeanStd | dict[str, torch.Tensor] | None,
    cfg,
):
    if obs_sf_normalizer is None:
        return

    if isinstance(obs_sf_normalizer, dict):
        mean = obs_sf_normalizer["mean"].to(
            device=model_input.state_features.device,
            dtype=model_input.state_features.dtype,
        )
        var = obs_sf_normalizer["var"].to(
            device=model_input.state_features.device,
            dtype=model_input.state_features.dtype,
        )
        model_input.state_features = torch.clamp(
            (model_input.state_features - mean)
            / torch.sqrt(var + cfg.training.observation_normalization_epsilon),
            -cfg.training.observation_normalization_clip,
            cfg.training.observation_normalization_clip,
        )
        return

    model_input.state_features = obs_sf_normalizer.normalize(
        model_input.state_features,
        epsilon=cfg.training.observation_normalization_epsilon,
        clip_range=cfg.training.observation_normalization_clip,
    )


def make_obs_sf_normalizer_snapshot(
    obs_sf_normalizer: RunningMeanStd | None,
) -> dict[str, torch.Tensor] | None:
    if obs_sf_normalizer is None:
        return None
    return {
        "mean": obs_sf_normalizer.mean.detach().cpu(),
        "var": obs_sf_normalizer.var.detach().cpu(),
    }
