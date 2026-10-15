"""Keep parallel environment transitions aligned through PPO minibatching."""

from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch

from deepgroebner_pytorch_cocoa.config_dataclasses import Config, ModelInputs
from deepgroebner_pytorch_cocoa.ppo_rollout import RolloutBuffer


def _buffer(*, ragged, residual=True, num_envs=2):
    cfg = Config()
    cfg.training.num_steps_per_update = 2
    cfg.training.num_envs = num_envs
    cfg.training.ranking_diagnostic_frequency = 0
    cfg.critic.critic_strategy = (
        "heuristic-plus-learned-residual" if residual else "learned-only"
    )
    spaces = {
        name: gym.spaces.Box(low=0, high=100, shape=shape)
        for name, shape in {
            "G": (3, 2),
            "P": (3, 4),
            "G_mask": (3,),
            "P_mask": (3,),
            "state_features": (2,),
        }.items()
    }
    envs = SimpleNamespace(
        single_observation_space=spaces,
        single_action_space=gym.spaces.Discrete(3),
    )
    return RolloutBuffer(cfg, envs, torch.device("cpu"), ragged=ragged)


def _sample(index, *, ragged):
    n_g = (4, 1, 2, 3)[index] if ragged else 3
    n_p = (5, 2, 3, 1)[index] if ragged else 3
    sample = {
        "G": torch.arange(n_g * 2).reshape(n_g, 2).float() + index * 10,
        "P": torch.arange(n_p * 4).reshape(n_p, 4).float() + index * 10,
        "G_mask": torch.ones(n_g, dtype=torch.bool),
        "P_mask": torch.ones(n_p, dtype=torch.bool),
        "state_features": torch.tensor([index, index + 0.5]),
    }
    if not ragged:
        sample["G_mask"][-1] = False
        sample["P_mask"][-1] = False
    return sample


@pytest.mark.parametrize("ragged", [False, True], ids=["fixed", "ragged"])
@pytest.mark.parametrize("residual", [False, True], ids=["learned", "residual"])
def test_parallel_rollout_minibatches_keep_observations_and_targets_aligned(
    ragged, residual
):
    #arrange
    buffer = _buffer(ragged=ragged, residual=residual)
    samples = [_sample(index, ragged=ragged) for index in range(4)]
    actions = torch.tensor([[0, 1], [1, 0]])
    logprobs = torch.tensor([[-0.1, -0.2], [-0.3, -0.4]])
    values = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    heuristic_values = values + 100
    advantages = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    returns = torch.tensor([[11.0, 22.0], [33.0, 44.0]])
    # Exclude sample 0, whose ragged observation is larger than this minibatch.
    indices = torch.tensor([3, 1, 2])
    observations = []
    for step in range(2):
        records = samples[step * 2 : step * 2 + 2]
        fields = {
            name: [record[name].clone() for record in records]
            for name in records[0]
        }
        if not ragged:
            fields = {name: torch.stack(value) for name, value in fields.items()}
        observations.append((ModelInputs(**fields), fields))

    #act
    for step, (inputs, fields) in enumerate(observations):
        buffer.store_observation(step, inputs, torch.tensor([step, 0.0]))
        buffer.store_action(
            step, actions[step], logprobs[step], values[step], heuristic_values[step]
        )
        # Reusing inference storage must not alter previously collected samples.
        for field in fields.values():
            tensors = field if isinstance(field, list) else [field]
            for tensor in tensors:
                tensor.zero_()

    batch = buffer.flatten(advantages, returns)
    inputs = batch.model_inputs(indices)

    #assert
    for row, index in enumerate(indices.tolist()):
        for name in ("G", "P", "G_mask", "P_mask"):
            expected = samples[index][name]
            actual = getattr(inputs, name)[row]
            torch.testing.assert_close(actual[: len(expected)], expected)
            assert torch.count_nonzero(actual[len(expected) :]) == 0
        torch.testing.assert_close(
            inputs.state_features[row], samples[index]["state_features"]
        )
    assert inputs.G.shape == (3, 3, 2)
    assert inputs.P.shape == (3, 3, 4)
    assert inputs.G_mask.dtype == inputs.P_mask.dtype == torch.bool
    torch.testing.assert_close(batch.actions.long(), actions.flatten())
    torch.testing.assert_close(batch.logprobs, logprobs.flatten())
    torch.testing.assert_close(batch.values, values.flatten())
    torch.testing.assert_close(batch.advantages, advantages.flatten())
    torch.testing.assert_close(batch.returns, returns.flatten())
    if residual:
        torch.testing.assert_close(batch.heuristic_values, heuristic_values.flatten())
        torch.testing.assert_close(
            inputs.h_value, heuristic_values.flatten()[indices, None]
        )
    else:
        assert batch.heuristic_values is None
        assert inputs.h_value is None


def test_clearing_ragged_rollout_discards_observations_from_the_previous_update():
    #arrange
    buffer = _buffer(ragged=True, num_envs=1)
    sample = _sample(0, ragged=True)
    inputs = ModelInputs(**{name: [value] for name, value in sample.items()})
    buffer.store_observation(0, inputs, torch.zeros(1))

    #act
    buffer.clear()

    #assert
    assert all(value == [] for value in buffer.observations.values())

    #arrange
    replacement = _sample(1, ragged=True)
    inputs = ModelInputs(**{name: [value] for name, value in replacement.items()})

    #act
    buffer.store_observation(0, inputs, torch.zeros(1))
    batch = buffer.flatten(torch.zeros(2, 1), torch.zeros(2, 1))
    minibatch = batch.model_inputs([0])

    #assert
    torch.testing.assert_close(minibatch.G[0], replacement["G"])
    torch.testing.assert_close(minibatch.P[0], replacement["P"])
    torch.testing.assert_close(minibatch.state_features[0], replacement["state_features"])
