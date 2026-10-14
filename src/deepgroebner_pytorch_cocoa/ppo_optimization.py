"""PPO returns, clipped policy/value losses, and optimizer updates."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn

from .ppo_agent import global_grad_norm, normalized_masked_entropy
from .training_utils import scalarize_metric


@dataclass
class UpdateMetrics:
    policy_loss: float
    value_loss: float
    entropy_raw: float
    entropy_normalized: float
    entropy_bonus: float
    old_approx_kl: float
    approx_kl: float
    actor_grad_norm: float
    critic_grad_norm: float | None
    explained_variance: float
    clipfracs: list[float]
    value_clipfracs: list[float]
    advantage_means: list[float]
    advantage_stds: list[float]
    last_indices: np.ndarray


@torch.no_grad()
def compute_returns_and_advantages(
    rewards,
    values,
    dones,
    next_value,
    next_done,
    *,
    gamma,
    gae_lambda,
    use_gae,
):
    """Bootstrap returns using done flags from the following rollout transition."""
    next_value = next_value.reshape(1, -1)
    num_steps = rewards.shape[0]
    if use_gae:
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        for step in reversed(range(num_steps)):
            next_nonterminal = 1.0 - (
                next_done if step == num_steps - 1 else dones[step + 1]
            )
            next_values = next_value if step == num_steps - 1 else values[step + 1]
            delta = (
                rewards[step] + gamma * next_values * next_nonterminal - values[step]
            )
            advantages[step] = last_gae = (
                delta + gamma * gae_lambda * next_nonterminal * last_gae
            )
        returns = advantages + values
    else:
        returns = torch.zeros_like(rewards)
        for step in reversed(range(num_steps)):
            next_nonterminal = 1.0 - (
                next_done if step == num_steps - 1 else dones[step + 1]
            )
            next_return = next_value if step == num_steps - 1 else returns[step + 1]
            returns[step] = rewards[step] + gamma * next_nonterminal * next_return
        advantages = returns - values
    return advantages, returns


def optimize_update(
    cfg, agent, optimizer, batch, value_normalizer=None
) -> UpdateMetrics:
    """Optimize one rollout and return scalar metrics without retaining its autograd graph."""
    device = batch.device
    b_actions = batch.actions
    b_logprobs = batch.logprobs
    b_advantages = batch.advantages
    b_returns = batch.returns
    b_values = batch.values
    b_h_values = batch.heuristic_values

    b_inds = np.arange(cfg.training.batch_size)
    clipfracs = []
    v_clipfracs = []
    entropy_raw_mean = torch.tensor(0.0, device=device)
    entropy_normalized_mean = torch.tensor(0.0, device=device)
    entropy_bonus_mean = torch.tensor(0.0, device=device)
    advantages_means = []
    advantages_stds = []
    for epoch in range(cfg.training.num_epochs_per_update):
        np.random.shuffle(b_inds)
        for start in range(0, cfg.training.batch_size, cfg.training.minibatch_size):
            end = start + cfg.training.minibatch_size
            mb_inds = b_inds[start:end]

            minibatch_input = batch.model_inputs(mb_inds)
            if cfg.critic.critic_strategy == "heuristic-only" and not batch.ragged:
                # Optimization does not use heuristic-only value predictions.
                minibatch_input.h_value = torch.zeros((len(mb_inds), 1), device=device)

            mb_value_targets = None
            if (
                value_normalizer is not None
                and cfg.critic.critic_strategy != "heuristic-only"
            ):
                if cfg.critic.critic_strategy == "heuristic-plus-learned-residual":
                    mb_value_targets = b_returns[mb_inds] - b_h_values[mb_inds]
                else:
                    mb_value_targets = b_returns[mb_inds]
                with torch.no_grad():
                    value_normalizer.update(mb_value_targets.detach())

            _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                minibatch_input, action=b_actions[mb_inds].to(dtype=torch.long)
            )

            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [
                    ((ratio - 1.0).abs() > cfg.training.clip_coef).float().mean().item()
                ]

            mb_advantages = b_advantages[mb_inds]
            if cfg.training.normalize_advantage:
                mb_advantages_mean = mb_advantages.mean()
                advantages_means.append(float(mb_advantages_mean.detach().item()))
                mb_advantages_std = mb_advantages.std()
                advantages_stds.append(float(mb_advantages_std.detach().item()))
                mb_advantages = (mb_advantages - mb_advantages_mean) / (
                    mb_advantages_std + 1e-8
                )

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(
                ratio, 1 - cfg.training.clip_coef, 1 + cfg.training.clip_coef
            )
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            if cfg.critic.critic_strategy != "heuristic-only":
                newvalue = newvalue.view(-1)
                if value_normalizer is not None:
                    if cfg.critic.critic_strategy == "heuristic-plus-learned-residual":
                        learned_target = b_returns[mb_inds] - b_h_values[mb_inds]
                        learned_newvalue = newvalue - b_h_values[mb_inds]
                        learned_oldvalue = b_values[mb_inds] - b_h_values[mb_inds]
                    else:
                        learned_target = b_returns[mb_inds]
                        learned_newvalue = newvalue
                        learned_oldvalue = b_values[mb_inds]
                    normalized_newvalue = value_normalizer.normalize(
                        learned_newvalue,
                        clip_range=cfg.training.value_normalization_clip,
                    )
                    normalized_returns = value_normalizer.normalize(
                        learned_target,
                        clip_range=cfg.training.value_normalization_clip,
                    )
                else:
                    normalized_newvalue = newvalue
                    normalized_returns = b_returns[mb_inds]

                if cfg.training.clip_critic_loss:
                    if value_normalizer is not None:
                        normalized_oldvalue = value_normalizer.normalize(
                            learned_oldvalue,
                            clip_range=cfg.training.value_normalization_clip,
                        )
                        v_delta = normalized_newvalue - normalized_oldvalue
                        abs_delta = v_delta.abs()
                        v_clipfrac = (
                            (abs_delta > cfg.training.critic_clip_coef)
                            .float()
                            .mean()
                            .item()
                        )
                        v_clipfracs += [v_clipfrac]
                        v_loss_unclipped = (
                            normalized_newvalue - normalized_returns
                        ) ** 2
                        v_clipped = normalized_oldvalue + torch.clamp(
                            v_delta,
                            -cfg.training.critic_clip_coef,
                            cfg.training.critic_clip_coef,
                        )
                        v_loss_clipped = (v_clipped - normalized_returns) ** 2
                    else:
                        v_delta = newvalue - b_values[mb_inds]
                        abs_delta = v_delta.abs()
                        v_clipfrac = (
                            (abs_delta > cfg.training.critic_clip_coef)
                            .float()
                            .mean()
                            .item()
                        )
                        v_clipfracs += [v_clipfrac]
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            v_delta,
                            -cfg.training.critic_clip_coef,
                            cfg.training.critic_clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = (
                        0.5 * ((normalized_newvalue - normalized_returns) ** 2).mean()
                    )
            else:
                v_loss = torch.tensor(0.0).to(device)

            entropy_raw_mean = entropy.mean()
            entropy_normalized = normalized_masked_entropy(
                entropy, minibatch_input.P_mask
            )
            entropy_normalized_mean = entropy_normalized.mean()
            entropy_bonus_mean = (
                entropy_normalized_mean
                if cfg.training.normalize_masked_action_entropy
                else entropy_raw_mean
            )

            loss = (
                pg_loss
                - cfg.training.entropy_coefficient * entropy_bonus_mean
                + v_loss * cfg.training.critic_loss_coefficient
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.training.clip_grad_norm_actor:
                actor_grad_norm = nn.utils.clip_grad_norm_(
                    agent.actor.parameters(), cfg.training.max_grad_norm_actor
                )
            else:
                actor_grad_norm = global_grad_norm(agent.actor.parameters())
            if cfg.critic.critic_strategy != "heuristic-only":
                if cfg.training.clip_grad_norm_critic:
                    critic_grad_norm = nn.utils.clip_grad_norm_(
                        agent.critic.parameters(), cfg.training.max_grad_norm_critic
                    )
                else:
                    critic_grad_norm = global_grad_norm(agent.critic.parameters())
            optimizer.step()

        if cfg.training.target_kl is not None:
            if approx_kl > cfg.training.target_kl:
                break

    pg_loss_scalar = scalarize_metric(pg_loss)
    entropy_raw_mean_scalar = scalarize_metric(entropy_raw_mean)
    entropy_normalized_mean_scalar = scalarize_metric(entropy_normalized_mean)
    entropy_bonus_mean_scalar = scalarize_metric(entropy_bonus_mean)
    old_approx_kl_scalar = scalarize_metric(old_approx_kl)
    approx_kl_scalar = scalarize_metric(approx_kl)
    actor_grad_norm_scalar = scalarize_metric(actor_grad_norm)
    critic_grad_norm_scalar = (
        scalarize_metric(critic_grad_norm)
        if cfg.critic.critic_strategy != "heuristic-only"
        else None
    )
    v_loss_scalar = (
        scalarize_metric(v_loss)
        if cfg.critic.critic_strategy != "heuristic-only"
        else 0.0
    )

    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_variance = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
    return UpdateMetrics(
        policy_loss=pg_loss_scalar,
        value_loss=v_loss_scalar,
        entropy_raw=entropy_raw_mean_scalar,
        entropy_normalized=entropy_normalized_mean_scalar,
        entropy_bonus=entropy_bonus_mean_scalar,
        old_approx_kl=old_approx_kl_scalar,
        approx_kl=approx_kl_scalar,
        actor_grad_norm=actor_grad_norm_scalar,
        critic_grad_norm=critic_grad_norm_scalar,
        explained_variance=float(explained_variance),
        clipfracs=clipfracs,
        value_clipfracs=v_clipfracs,
        advantage_means=advantages_means,
        advantage_stds=advantages_stds,
        last_indices=mb_inds,
    )
