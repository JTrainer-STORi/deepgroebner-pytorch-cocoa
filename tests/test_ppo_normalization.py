"""Deterministic checks for PPO running statistics and normalization invariants."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from deepgroebner_pytorch_cocoa.config_dataclasses import Config, ModelInputs
from deepgroebner_pytorch_cocoa.ppo_normalization import (
    PopArt,
    RewardNormalizer,
    RunningMeanStd,
    build_value_normalizer,
    make_obs_sf_normalizer_snapshot,
    normalize_eval_state_features,
)


@pytest.mark.parametrize("batch_sizes", [(4,), (1, 3), (2, 1, 1)])
def test_running_moments_match_population_statistics_across_unequal_batches(batch_sizes):
    #arrange
    samples = torch.tensor([[1, 10], [3, 14], [5, 18], [7, 22]])
    normalizer = RunningMeanStd(shape=(2,), epsilon=0)

    #act
    for batch in samples.split(batch_sizes):
        normalizer.update(batch)

    #assert
    torch.testing.assert_close(normalizer.mean, torch.tensor([4.0, 16.0]))
    torch.testing.assert_close(normalizer.var, torch.tensor([5.0, 20.0]))
    assert normalizer.count.item() == 4
    assert normalizer.mean.dtype == torch.float32


def test_running_moments_include_initial_prior_weight():
    #arrange
    normalizer = RunningMeanStd(epsilon=2)
    samples = torch.tensor([2.0, 6.0], dtype=torch.float64)

    #act
    normalizer.update(samples)

    #assert
    # Two prior observations with mean 0 and variance 1, plus samples 2 and 6.
    torch.testing.assert_close(normalizer.mean, torch.tensor(2.0))
    torch.testing.assert_close(normalizer.var, torch.tensor(6.5))
    assert normalizer.count.item() == 4


def test_running_moments_empty_update_is_a_noop_and_singletons_have_zero_variance():
    #arrange
    normalizer = RunningMeanStd(shape=(2,), epsilon=0)
    normalizer.update(torch.tensor([[3.0, -2.0]]))
    before = tuple(
        value.clone() for value in (normalizer.mean, normalizer.var, normalizer.count)
    )

    empty_batch = torch.empty(0, 2)
    singleton = torch.tensor([[3.0, -2.0]])

    #act
    normalizer.update(empty_batch)

    #assert
    for actual, expected in zip(
        (normalizer.mean, normalizer.var, normalizer.count), before
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(normalizer.var, torch.zeros(2))

    #act
    normalized = normalizer.normalize(singleton)

    #assert
    torch.testing.assert_close(normalized, torch.zeros(1, 2))


@pytest.mark.parametrize("clip_range", [None, 2.0, 0.0, -1.0])
def test_running_normalization_centers_scales_and_optionally_clips(clip_range):
    #arrange
    normalizer = RunningMeanStd(shape=(2,))
    normalizer.mean = torch.tensor([2.0, -2.0])
    normalizer.var = torch.tensor([4.0, 16.0])
    samples = torch.tensor([[-4.0, -20.0], [2.0, -2.0], [8.0, 16.0]])
    expected = torch.tensor([[-3.0, -4.5], [0.0, 0.0], [3.0, 4.5]])
    if clip_range == 2.0:
        expected = torch.tensor([[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]])

    #act
    actual = normalizer.normalize(samples, epsilon=0, clip_range=clip_range)

    #assert
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(normalizer.mean, torch.tensor([2.0, -2.0]))
    assert normalizer.count.item() == pytest.approx(1e-4)


def test_running_normalization_uses_configured_epsilon_for_constant_features():
    #arrange
    normalizer = RunningMeanStd(epsilon=0)
    normalizer.update(torch.tensor([5.0, 5.0]))

    samples = torch.tensor([3.0, 5.0, 7.0])

    #act
    actual = normalizer.normalize(samples, epsilon=0.25)

    #assert
    torch.testing.assert_close(actual, torch.tensor([-4.0, 0.0, 4.0]))
    assert torch.isfinite(actual).all()


def test_reward_normalization_keeps_terminal_history_then_resets_each_environment():
    #arrange
    normalizer = RewardNormalizer(num_envs=2, gamma=0.5, clip_range=None)
    transitions = [
        # rewards, transition done flags, accumulators after resetting terminals
        ([1.0, 10.0], [False, False], [1.0, 10.0]),
        ([3.0, 20.0], [True, False], [0.0, 25.0]),
        ([5.0, 30.0], [False, True], [5.0, 0.0]),
    ]
    # The terminal transition includes the same episode's earlier rewards. Its
    # return must enter the statistics before that environment starts afresh.
    discounted_samples = [[1.0, 10.0], [3.5, 25.0], [5.0, 42.5]]
    seen_returns = []
    for (rewards, dones, expected_returns), samples in zip(
        transitions, discounted_samples
    ):
        #arrange
        rewards = torch.tensor(rewards)
        dones = torch.tensor(dones)
        seen_returns.extend(samples)

        # Account for RunningMeanStd's small initial mean-zero, variance-one prior.
        history = torch.tensor(seen_returns, dtype=torch.float64)
        count = history.numel() + 1e-4
        expected_mean = history.sum() / count
        expected_var = (
            (history.square().sum() + 1e-4) / count - expected_mean.square()
        )
        expected_reward = rewards / torch.sqrt(expected_var + normalizer.epsilon)

        #act
        actual = normalizer.normalize(rewards, dones)

        #assert
        torch.testing.assert_close(actual, expected_reward)
        torch.testing.assert_close(normalizer.returns, torch.tensor(expected_returns))
        torch.testing.assert_close(normalizer.rms.mean, expected_mean.float())
        torch.testing.assert_close(normalizer.rms.var, expected_var.float())
        assert normalizer.rms.count.item() == pytest.approx(count)


@pytest.mark.parametrize("clip_range", [None, 0.0, 2.0])
def test_reward_normalization_scales_without_centering_and_honors_clipping(clip_range):
    #arrange
    normalizer = RewardNormalizer(num_envs=2, gamma=0, clip_range=clip_range)
    rewards = torch.tensor([10.0, 11.0])
    dones = torch.zeros(2)

    #act
    actual = normalizer.normalize(rewards, dones)

    #assert
    expected = rewards / torch.sqrt(normalizer.rms.var + normalizer.epsilon)
    if clip_range == 2.0:
        expected = torch.tensor([2.0, 2.0])
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(rewards, torch.tensor([10.0, 11.0]))
    assert (actual > 0).all()


@pytest.mark.parametrize("num_outputs", [1, 2])
def test_popart_preserves_raw_predictions_across_sequential_statistic_updates(num_outputs):
    #arrange
    layer = nn.Linear(3, num_outputs)
    with torch.no_grad():
        layer.weight.copy_(
            torch.arange(1, 3 * num_outputs + 1).reshape(num_outputs, 3) / 10
        )
        layer.bias.copy_(torch.arange(num_outputs) + 0.25)
    normalizer = PopArt(layer, shape=() if num_outputs == 1 else (num_outputs,))
    features = torch.tensor([[1.0, -2.0, 3.0], [-3.0, 1.0, 0.5]])
    raw_before = normalizer.denormalize(layer(features)).detach().clone()
    original_parameters = tuple(layer.parameters())
    targets = [
        torch.tensor([[10.0, -4.0], [20.0, -8.0], [30.0, -12.0]]),
        torch.tensor([[-100.0, 40.0], [-80.0, 20.0]]),
    ]

    for batch in targets:
        #act
        normalizer.update(batch[:, 0] if num_outputs == 1 else batch)
        raw_after = normalizer.denormalize(layer(features))

        #assert
        torch.testing.assert_close(
            raw_after, raw_before, atol=1e-5, rtol=1e-5
        )

    assert all(
        before is after
        for before, after in zip(original_parameters, layer.parameters())
    )

    #act
    layer(features).sum().backward()

    #assert
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in layer.parameters()
    )


def test_popart_running_statistics_match_combined_targets_and_initial_prior():
    #arrange
    normalizer = PopArt(nn.Linear(2, 2), shape=(2,))
    targets = torch.tensor([[2.0, -3.0], [4.0, 5.0], [9.0, 7.0]], dtype=torch.float64)
    expected_count = 3 + 1e-4
    expected_mean = targets.sum(dim=0) / expected_count
    expected_var = (
        (targets.square().sum(dim=0) + 1e-4) / expected_count - expected_mean.square()
    )

    #act
    normalizer.update(targets[:1])
    normalizer.update(targets[1:])

    #assert
    torch.testing.assert_close(normalizer.mean, expected_mean.float())
    torch.testing.assert_close(normalizer.var, expected_var.float())
    assert normalizer.count.item() == pytest.approx(expected_count)


def test_popart_round_trip_std_floor_and_clipping():
    #arrange
    normalizer = PopArt(nn.Linear(2, 1), epsilon=0, min_std=0.25)
    normalizer.mean = torch.tensor(5.0)
    normalizer.var = torch.tensor(0.0)
    samples = torch.tensor([4.0, 5.0, 6.0])

    #act
    normalized = normalizer.normalize(samples)
    restored = normalizer.denormalize(normalized)
    clipped = normalizer.normalize(samples, clip_range=2)

    #assert
    torch.testing.assert_close(normalized, torch.tensor([-4.0, 0.0, 4.0]))
    torch.testing.assert_close(restored, samples)
    torch.testing.assert_close(clipped, torch.tensor([-2.0, 0.0, 2.0]))
    assert torch.isfinite(normalized).all()


def test_popart_empty_update_changes_neither_statistics_nor_parameters():
    #arrange
    layer = nn.Linear(2, 1)
    normalizer = PopArt(layer)
    before = tuple(
        value.clone()
        for value in (
            normalizer.mean, normalizer.var, normalizer.count, layer.weight, layer.bias
        )
    )

    empty_targets = torch.empty(0)

    #act
    normalizer.update(empty_targets)

    #assert
    for actual, expected in zip(
        (normalizer.mean, normalizer.var, normalizer.count, layer.weight, layer.bias), before
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_popart_rejects_nonlinear_output_layers_and_incompatible_stat_shapes():
    #arrange
    nonlinear_layer = nn.ReLU()
    incompatible_layer = nn.Linear(3, 2)

    #act
    #assert
    with pytest.raises(TypeError, match="nn.Linear"):
        PopArt(nonlinear_layer)
    with pytest.raises(ValueError, match="out_features=2"):
        PopArt(incompatible_layer, shape=())


@pytest.mark.parametrize(
    "enabled,strategy",
    [(False, "learned-only"), (True, "heuristic-only")],
)
def test_value_normalization_is_disabled_without_a_learned_normalized_critic(enabled, strategy):
    #arrange
    cfg = Config()
    cfg.training.normalize_value_function = enabled
    cfg.critic.critic_strategy = strategy

    agent = SimpleNamespace()

    #act
    normalizer = build_value_normalizer(cfg, agent)

    #assert
    assert normalizer is None


@pytest.mark.parametrize("strategy", ["learned-only", "heuristic-plus-learned-residual"])
def test_value_normalizer_attaches_to_final_nested_critic_output_layer(strategy):
    #arrange
    cfg = Config()
    cfg.training.normalize_value_function = True
    cfg.training.value_normalization_epsilon = 0.01
    cfg.critic.critic_strategy = strategy
    output_layer = nn.Linear(3, 1)
    value_head = nn.Sequential(
        nn.Linear(2, 3), nn.ReLU(), nn.Sequential(output_layer)
    )
    agent = SimpleNamespace(critic=SimpleNamespace(value_head=value_head))

    #act
    normalizer = build_value_normalizer(cfg, agent)

    #assert
    assert agent.critic.popart is normalizer
    assert normalizer.output_layer is output_layer
    assert normalizer.epsilon == 0.01
    assert normalizer.min_std == 0.01


def test_value_normalizer_requires_a_linear_critic_output_layer():
    #arrange
    cfg = Config()
    cfg.training.normalize_value_function = True
    cfg.critic.critic_strategy = "learned-only"
    agent = SimpleNamespace(critic=SimpleNamespace(value_head=nn.ReLU()))

    #act
    #assert
    with pytest.raises(ValueError, match="final nn.Linear"):
        build_value_normalizer(cfg, agent)


@pytest.mark.parametrize("use_snapshot", [False, True])
def test_evaluation_state_normalization_matches_frozen_training_statistics(use_snapshot):
    #arrange
    cfg = Config()
    cfg.training.observation_normalization_epsilon = 0.25
    cfg.training.observation_normalization_clip = 1.5
    normalizer = RunningMeanStd(shape=(2,), epsilon=0)
    normalizer.update(torch.tensor([[1.0, -1.0], [3.0, 3.0]]))
    snapshot = make_obs_sf_normalizer_snapshot(normalizer)
    inputs = ModelInputs(
        state_features=torch.tensor([[2.0, 1.0], [2.5, 0.0], [7.0, -9.0]])
    )

    #act
    normalize_eval_state_features(inputs, snapshot if use_snapshot else normalizer, cfg)

    #assert
    torch.testing.assert_close(
        inputs.state_features,
        torch.tensor([[0.0, 0.0], [0.4472136, -0.4850713], [1.5, -1.5]]),
    )
    torch.testing.assert_close(normalizer.mean, torch.tensor([2.0, 1.0]))
    torch.testing.assert_close(normalizer.var, torch.tensor([1.0, 4.0]))
    assert normalizer.count.item() == 2
    assert all(
        value.device.type == "cpu" and not value.requires_grad
        for value in snapshot.values()
    )


def test_observation_snapshot_retains_statistics_after_later_training_updates():
    #arrange
    normalizer = RunningMeanStd(shape=(2,), epsilon=0)
    normalizer.update(
        torch.tensor([[1.0, 3.0], [3.0, 7.0]], requires_grad=True)
    )
    #act
    snapshot = make_obs_sf_normalizer_snapshot(normalizer)

    #assert
    assert normalizer.mean.requires_grad
    assert normalizer.var.requires_grad
    assert all(not value.requires_grad for value in snapshot.values())

    #act
    normalizer.update(torch.tensor([[100.0, 200.0]]))

    #assert
    torch.testing.assert_close(snapshot["mean"], torch.tensor([2.0, 5.0]))
    torch.testing.assert_close(snapshot["var"], torch.tensor([1.0, 4.0]))


def test_missing_observation_normalizer_is_a_noop():
    #arrange
    inputs = ModelInputs(state_features=torch.tensor([[1.0, 2.0]]))
    original = inputs.state_features
    cfg = Config()

    #act
    normalize_eval_state_features(inputs, None, cfg)
    snapshot = make_obs_sf_normalizer_snapshot(None)

    #assert
    assert inputs.state_features is original
    assert snapshot is None
