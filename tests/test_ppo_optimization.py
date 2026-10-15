import math

import numpy as np
import pytest
import torch

from deepgroebner_pytorch_cocoa.config_dataclasses import Config
from deepgroebner_pytorch_cocoa.ppo_agent import CategoricalMasked
from deepgroebner_pytorch_cocoa.ppo_optimization import optimize_update
from deepgroebner_pytorch_cocoa.ppo_rollout import RolloutBatch


@pytest.fixture(autouse=True)
def _fixed_minibatch_order(monkeypatch):
    # The loss checks use explicit sample groups, independent of NumPy's global RNG.
    monkeypatch.setattr(np.random, "shuffle", lambda indices: None)


class _TabularAgent(torch.nn.Module):
    """Small real policy/value networks with independently controllable states."""

    def __init__(self, logits, values):
        super().__init__()
        self.actor = torch.nn.Embedding.from_pretrained(logits.clone(), freeze=False)
        self.critic = torch.nn.Embedding.from_pretrained(
            values.reshape(-1, 1).clone(), freeze=False
        )
        self.calls = 0

    def get_action_and_value(self, inputs, action):
        self.calls += 1
        ids = inputs.state_features[:, 0].long()
        policy = CategoricalMasked(logits=self.actor(ids), masks=inputs.P_mask)
        return (
            action,
            policy.log_prob(action),
            policy.entropy(),
            inputs.h_value if self.critic is None else self.critic(ids),
            None,
        )


def _update_case(
    advantages,
    *,
    ratios=None,
    new_values=None,
    old_values=None,
    returns=None,
    heuristic_values=None,
    logits=None,
    mask=None,
    learning_rate=0.0,
):
    """Use fixed samples and zero learning rate unless a test checks the update."""
    advantages = torch.tensor(advantages, dtype=torch.float32)
    count = len(advantages)
    values = torch.zeros(count) if new_values is None else torch.tensor(new_values)
    if logits is None:
        logits = torch.zeros(count, 4)
    if mask is None:
        mask = torch.ones_like(logits, dtype=torch.bool)
    agent = _TabularAgent(logits, values.float())
    actions = torch.zeros(count, dtype=torch.long)
    with torch.no_grad():
        logprobs = CategoricalMasked(logits=logits, masks=mask).log_prob(actions)
        if ratios is not None:
            logprobs -= torch.tensor(ratios).log()

    batch = RolloutBatch(
        observations={
            "state_features": torch.arange(count).reshape(-1, 1),
            "P_mask": mask,
        },
        actions=actions,
        logprobs=logprobs,
        advantages=advantages,
        returns=(
            torch.arange(count, dtype=torch.float32)
            if returns is None
            else torch.tensor(returns, dtype=torch.float32)
        ),
        values=(
            torch.zeros(count)
            if old_values is None
            else torch.tensor(old_values, dtype=torch.float32)
        ),
        heuristic_values=(
            None
            if heuristic_values is None
            else torch.tensor(heuristic_values, dtype=torch.float32)
        ),
        ragged=False,
        device=torch.device("cpu"),
    )
    cfg = Config()
    cfg.training.batch_size = count
    cfg.training.minibatch_size = count
    cfg.training.num_epochs_per_update = 1
    cfg.training.normalize_advantage = False
    cfg.training.entropy_coefficient = 0.0
    cfg.training.critic_loss_coefficient = 1.0
    cfg.training.clip_grad_norm_actor = False
    cfg.training.clip_grad_norm_critic = False
    cfg.training.clip_coef = 0.2
    cfg.training.critic_clip_coef = 0.5
    optimizer = torch.optim.SGD(agent.parameters(), lr=learning_rate)
    return cfg, agent, optimizer, batch


def test_singleton_minibatch_advantage_normalization_stays_finite():
    #arrange
    cfg, agent, optimizer, batch = _update_case([7.0], learning_rate=0.1)
    cfg.training.normalize_advantage = True

    #act
    metrics = optimize_update(cfg, agent, optimizer, batch)

    #assert
    assert metrics.policy_loss == pytest.approx(0.0)
    assert metrics.advantage_means == pytest.approx([7.0])
    assert metrics.advantage_stds == pytest.approx([0.0])
    assert math.isfinite(metrics.actor_grad_norm)
    assert all(torch.isfinite(parameter).all() for parameter in agent.parameters())


def test_policy_surrogate_clips_only_improvements_for_each_advantage_sign():
    #arrange
    cfg, agent, optimizer, batch = _update_case(
        [2.0, 2.0, -2.0, -2.0], ratios=[1.5, 0.5, 1.5, 0.5]
    )
    expected_kl = -0.5 * (math.log(1.5) + math.log(0.5))
    expected_gradient = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [-0.1875, 0.0625, 0.0625, 0.0625],
            [0.5625, -0.1875, -0.1875, -0.1875],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    #act
    metrics = optimize_update(cfg, agent, optimizer, batch)

    #assert
    # Four losses: -2 * 1.2, -2 * 0.5, 2 * 1.5, 2 * 0.8.
    assert metrics.policy_loss == pytest.approx(0.3)
    assert metrics.clipfracs == pytest.approx([1.0])
    assert metrics.advantage_means == []
    assert metrics.advantage_stds == []
    assert metrics.old_approx_kl == pytest.approx(expected_kl)
    assert metrics.approx_kl == pytest.approx(expected_kl)
    torch.testing.assert_close(agent.actor.weight.grad, expected_gradient)


@pytest.mark.parametrize("clip_value_loss", [False, True])
def test_value_loss_uses_larger_of_clipped_and_unclipped_errors(clip_value_loss):
    #arrange
    cfg, agent, optimizer, batch = _update_case(
        [0.0, 0.0, 0.0],
        new_values=[3.0, -3.0, 0.25],
        old_values=[0.0, 0.0, 0.0],
        returns=[2.0, -2.0, 1.0],
    )
    cfg.training.clip_critic_loss = clip_value_loss
    expected_gradient = [0.0, 0.0, -0.25] if clip_value_loss else [1 / 3, -1 / 3, -0.25]

    #act
    metrics = optimize_update(cfg, agent, optimizer, batch)

    #assert
    if clip_value_loss:
        # Predictions moving toward the target can still incur the old-value penalty.
        assert metrics.value_loss == pytest.approx((2.25 + 2.25 + 0.5625) / 6)
        assert metrics.value_clipfracs == pytest.approx([2 / 3])
    else:
        assert metrics.value_loss == pytest.approx((1.0 + 1.0 + 0.5625) / 6)
        assert metrics.value_clipfracs == []
    torch.testing.assert_close(
        agent.critic.weight.grad[:, 0], torch.tensor(expected_gradient)
    )


def test_advantages_are_normalized_per_minibatch_without_mutating_rollout():
    #arrange
    cfg, agent, optimizer, batch = _update_case(
        [1.0, 3.0, 10.0, 14.0], ratios=[1.1, 0.9, 1.2, 0.8]
    )
    cfg.training.minibatch_size = 2
    cfg.training.normalize_advantage = True
    cfg.training.clip_coef = 0.5
    original_advantages = batch.advantages.clone()

    #act
    metrics = optimize_update(cfg, agent, optimizer, batch)

    #assert
    assert metrics.advantage_means == pytest.approx([2.0, 12.0])
    assert metrics.advantage_stds == pytest.approx([math.sqrt(2), math.sqrt(8)])
    # The last minibatch has advantages [-1/sqrt(2), 1/sqrt(2)].
    assert metrics.policy_loss == pytest.approx(0.2 / math.sqrt(2))
    torch.testing.assert_close(batch.advantages, original_advantages)
    np.testing.assert_array_equal(metrics.last_indices, [2, 3])


def test_constant_advantages_normalize_to_zero():
    #arrange
    cfg, agent, optimizer, batch = _update_case([3.0, 3.0], ratios=[1.1, 0.9])
    cfg.training.normalize_advantage = True

    #act
    metrics = optimize_update(cfg, agent, optimizer, batch)

    #assert
    assert metrics.policy_loss == pytest.approx(0.0)
    assert metrics.advantage_stds == pytest.approx([0.0])
    torch.testing.assert_close(agent.actor.weight.grad, torch.zeros_like(agent.actor.weight))


@pytest.mark.parametrize("normalize_entropy", [False, True])
def test_entropy_bonus_uses_valid_action_counts_and_updates_policy(normalize_entropy):
    #arrange
    probabilities = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.25, 0.75, 0.0, 0.0], [0.1, 0.2, 0.3, 0.4]]
    )
    mask = probabilities > 0
    log_probabilities = probabilities.clamp_min(1e-8).log()
    cfg, agent, optimizer, batch = _update_case(
        [0.0, 0.0, 0.0],
        logits=log_probabilities,
        mask=mask,
        learning_rate=0.1,
    )
    cfg.training.entropy_coefficient = 0.3
    cfg.training.critic_loss_coefficient = 0.0
    cfg.training.normalize_masked_action_entropy = normalize_entropy
    before = agent.actor.weight.detach().clone()
    entropies = -(probabilities * log_probabilities).sum(-1)
    normalized_entropies = entropies / torch.tensor([1.0, math.log(2), math.log(4)])
    expected_bonus = normalized_entropies if normalize_entropy else entropies
    # d(-H)/d(logit_i) = p_i * (log(p_i) + H), with zero masked gradients.
    expected_gradient = (
        0.3 / 3 * probabilities * (log_probabilities + entropies[:, None])
    )
    if normalize_entropy:
        expected_gradient /= torch.tensor([1.0, math.log(2), math.log(4)])[:, None]

    #act
    metrics = optimize_update(cfg, agent, optimizer, batch)

    #assert
    assert metrics.entropy_raw == pytest.approx(entropies.mean().item())
    assert metrics.entropy_normalized == pytest.approx(normalized_entropies.mean().item())
    assert metrics.entropy_bonus == pytest.approx(expected_bonus.mean().item())
    torch.testing.assert_close(agent.actor.weight.grad, expected_gradient)
    torch.testing.assert_close(agent.actor.weight, before - 0.1 * expected_gradient)


class _FixedNormalizer:
    """Record integration calls while fixing units to mean 10 and std 2."""

    def __init__(self):
        self.targets = []
        self.clip_ranges = []

    def update(self, targets):
        assert not torch.is_grad_enabled()
        self.targets.append(targets.clone())

    def normalize(self, values, clip_range):
        self.clip_ranges.append(clip_range)
        return ((values - 10.0) / 2.0).clamp(-clip_range, clip_range)


@pytest.mark.parametrize("strategy", ["learned-only", "heuristic-plus-learned-residual"])
@pytest.mark.parametrize("clip_value_loss", [False, True])
def test_value_normalization_applies_to_learned_component(strategy, clip_value_loss):
    #arrange
    heuristic = torch.tensor([100.0, 200.0])
    offset = heuristic if strategy == "heuristic-plus-learned-residual" else torch.zeros(2)
    cfg, agent, optimizer, batch = _update_case(
        [0.0, 0.0],
        new_values=(offset + torch.tensor([12.0, 13.75])).tolist(),
        old_values=(offset + torch.tensor([11.0, 13.0])).tolist(),
        returns=(offset + torch.tensor([11.0, 14.0])).tolist(),
        heuristic_values=heuristic.tolist(),
    )
    cfg.critic.critic_strategy = strategy
    cfg.training.clip_critic_loss = clip_value_loss
    cfg.training.critic_clip_coef = 0.25
    normalizer = _FixedNormalizer()
    expected_gradient = torch.tensor([0.125, 0.0 if clip_value_loss else -0.03125])

    #act
    metrics = optimize_update(cfg, agent, optimizer, batch, normalizer)

    #assert
    assert len(normalizer.targets) == 1
    torch.testing.assert_close(normalizer.targets[0], torch.tensor([11.0, 14.0]))
    assert normalizer.clip_ranges == [cfg.training.value_normalization_clip] * (
        3 if clip_value_loss else 2
    )
    # Normalized predictions [1, 1.875], targets [.5, 2], old values [.5, 1.5].
    assert metrics.value_loss == pytest.approx(
        0.078125 if clip_value_loss else 0.06640625
    )
    assert metrics.value_clipfracs == (pytest.approx([1.0]) if clip_value_loss else [])
    torch.testing.assert_close(agent.critic.weight.grad[:, 0], expected_gradient)


def test_heuristic_only_critic_skips_value_loss_normalizer_and_critic_gradients():
    #arrange
    cfg, agent, _, batch = _update_case(
        [1.0, -1.0],
        returns=[100.0, -100.0],
        heuristic_values=[20.0, 30.0],
    )
    cfg.critic.critic_strategy = "heuristic-only"
    agent.critic = None
    optimizer = torch.optim.SGD(agent.parameters(), lr=0.1)
    actor_before = agent.actor.weight.detach().clone()

    # A sentinel with no normalization methods proves this branch never uses them.
    normalizer = object()

    #act
    metrics = optimize_update(cfg, agent, optimizer, batch, value_normalizer=normalizer)

    #assert
    assert metrics.value_loss == 0.0
    assert metrics.value_clipfracs == []
    assert metrics.critic_grad_norm is None
    assert not torch.equal(agent.actor.weight, actor_before)


@pytest.mark.parametrize("target_kl,expected_calls", [(None, 8), (0.4, 8), (0.1, 2)])
def test_target_kl_stops_further_epochs_after_finishing_current_epoch(
    target_kl, expected_calls
):
    #arrange
    cfg, agent, optimizer, batch = _update_case(
        [1.0, -1.0, 1.0, -1.0], ratios=[2.0, 2.0, 2.0, 2.0]
    )
    cfg.training.minibatch_size = 2
    cfg.training.num_epochs_per_update = 4
    cfg.training.target_kl = target_kl

    #act
    metrics = optimize_update(cfg, agent, optimizer, batch)

    #assert
    assert agent.calls == expected_calls
    assert len(metrics.clipfracs) == expected_calls
    assert metrics.approx_kl == pytest.approx(1.0 - math.log(2))


@pytest.mark.parametrize("clip_gradients", [False, True])
def test_optimizer_applies_independent_actor_and_critic_gradient_limits(clip_gradients):
    #arrange
    cfg, agent, optimizer, batch = _update_case(
        [4.0, -2.0], returns=[5.0, -4.0], learning_rate=0.1
    )
    cfg.training.clip_grad_norm_actor = clip_gradients
    cfg.training.clip_grad_norm_critic = clip_gradients
    cfg.training.max_grad_norm_actor = 0.3
    cfg.training.max_grad_norm_critic = 0.4
    actor_before = agent.actor.weight.detach().clone()
    critic_before = agent.critic.weight.detach().clone()

    #act
    metrics = optimize_update(cfg, agent, optimizer, batch)
    actor_update_norm = (agent.actor.weight - actor_before).norm().item()
    critic_update_norm = (agent.critic.weight - critic_before).norm().item()

    #assert
    assert metrics.actor_grad_norm == pytest.approx(math.sqrt(3.75))
    assert metrics.critic_grad_norm == pytest.approx(math.sqrt(10.25))
    assert actor_update_norm == pytest.approx(
        0.1 * (0.3 if clip_gradients else math.sqrt(3.75)), rel=1e-5
    )
    assert critic_update_norm == pytest.approx(
        0.1 * (0.4 if clip_gradients else math.sqrt(10.25)), rel=1e-5
    )
