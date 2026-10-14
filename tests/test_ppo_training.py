from dataclasses import asdict
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from deepgroebner_pytorch_cocoa.config_dataclasses import Config, ModelInputs
from deepgroebner_pytorch_cocoa.model_input_utils import (
    build_model_inputs_from_single_obs,
)
from deepgroebner_pytorch_cocoa.ppo_agent import Agent
from deepgroebner_pytorch_cocoa.ppo_normalization import PopArt
from deepgroebner_pytorch_cocoa.ppo_optimization import compute_returns_and_advantages
from deepgroebner_pytorch_cocoa.ppo_rollout import RolloutBuffer
from deepgroebner_pytorch_cocoa import ppo_training


def _small_config(tmp_path):
    cfg = Config()
    cfg.env.coeff_ring = "QQ"
    cfg.env.symbols = "x,y"
    cfg.env.basis_elements = "x^2 - y,x*y - 1,y^2 - x"
    cfg.env.elimination = "none"
    cfg.training.num_envs = 1
    cfg.training.num_steps_per_update = 4
    cfg.training.total_timesteps = 4
    cfg.training.num_epochs_per_update = 1
    cfg.training.num_minibatches = 2
    cfg.training.hparam_eval_num_examples = 0
    cfg.training.cuda = False
    cfg.training.track = False
    cfg.training.pytorch_num_threads = 1
    cfg.training.system_metrics_enabled = False
    cfg.training.output_directory = str(tmp_path)
    return cfg


@pytest.mark.parametrize(
    "use_gae,expected_advantages,expected_returns",
    [
        (True, [0.5, 4.334, 3.2], [1.0, 4.934, 3.9]),
        (False, [0.5, 4.91, 3.2], [1.0, 5.51, 3.9]),
    ],
)
def test_returns_stop_at_episode_boundaries_and_bootstrap_unfinished_episodes(
    use_gae,
    expected_advantages,
    expected_returns,
):
    advantages, returns = compute_returns_and_advantages(
        rewards=torch.tensor([[1.0], [2.0], [3.0]]),
        values=torch.tensor([[0.5], [0.6], [0.7]]),
        dones=torch.tensor([[0.0], [1.0], [0.0]]),
        next_value=torch.tensor([[1.0]]),
        next_done=torch.tensor([0.0]),
        gamma=0.9,
        gae_lambda=0.8,
        use_gae=use_gae,
    )
    torch.testing.assert_close(advantages[:, 0], torch.tensor(expected_advantages))
    torch.testing.assert_close(returns[:, 0], torch.tensor(expected_returns))


def test_popart_preserves_raw_predictions_when_target_statistics_change():
    layer = torch.nn.Linear(3, 1)
    normalizer = PopArt(layer)
    features = torch.randn(4, 3)
    before = normalizer.denormalize(layer(features)).detach()
    normalizer.update(torch.tensor([10.0, 20.0, 30.0, 40.0]))
    torch.testing.assert_close(
        normalizer.denormalize(layer(features)), before, atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize(
    "gym_id", ["BuchbergerExponents-v1", "BuchbergerExponentsNoPQ-v1"]
)
def test_rollout_snapshots_and_minibatch_order_survive_environment_steps(
    gym_id, tmp_path
):
    cfg = _small_config(tmp_path)
    cfg.training.num_steps_per_update = 2
    env = gym.make(
        gym_id,
        coeff_ring="QQ",
        symbols="x,y",
        basis_elements=cfg.env.basis_elements.split(","),
        elimination="none",
    )
    try:
        envs = SimpleNamespace(
            single_observation_space=env.observation_space,
            single_action_space=env.action_space,
        )
        buffer = RolloutBuffer(cfg, envs, torch.device("cpu"), ragged="NoPQ" in gym_id)
        obs, _ = env.reset(seed=123)
        expected = []
        for step in range(2):
            inputs = build_model_inputs_from_single_obs(obs, torch.device("cpu"))
            expected.append(
                {
                    name: getattr(inputs, name).clone()
                    for name in ("G", "P", "G_mask", "P_mask")
                }
            )
            buffer.store_observation(step, inputs, torch.zeros(1))
            action = int(np.flatnonzero(obs["P_mask"])[0])
            buffer.store_action(
                step,
                torch.tensor([action]),
                torch.zeros(1),
                torch.ones(1),
                torch.tensor([step + 1.0]),
            )
            obs, _, _, _, _ = env.step(action)
        batch = buffer.flatten(torch.zeros(2, 1), torch.ones(2, 1))
        minibatch = batch.model_inputs([1, 0])
        for row, record in enumerate(reversed(expected)):
            for name, tensor in record.items():
                tensor = tensor[0]
                actual = getattr(minibatch, name)[row]
                slices = tuple(slice(0, size) for size in tensor.shape)
                torch.testing.assert_close(actual[slices], tensor)
        torch.testing.assert_close(minibatch.h_value, torch.tensor([[2.0], [1.0]]))
    finally:
        env.close()


def test_agents_keep_their_own_critic_configuration_and_cached_values():
    cfg = Config()
    cfg.actor.pair_embedding_head_in_dim = 6
    cfg.critic.state_embedding_head_in_dim = 3
    cfg.critic.state_features_embedding_head_in_dim = 4
    cfg.critic.critic_strategy = "heuristic-only"
    heuristic_agent = Agent(cfg)
    cfg = Config()
    cfg.actor.pair_embedding_head_in_dim = 6
    cfg.critic.state_embedding_head_in_dim = 3
    cfg.critic.state_features_embedding_head_in_dim = 4
    cfg.critic.critic_strategy = "learned-only"
    learned_agent = Agent(cfg)
    inputs = ModelInputs(
        G=torch.ones(2, 2, 3),
        G_mask=torch.ones(2, 2, dtype=torch.bool),
        state_features=torch.ones(2, 4),
        h_value=torch.tensor([[7.0], [9.0]]),
    )
    learned_values, learned_heuristic = learned_agent.get_value(inputs)
    assert learned_values.shape == (2, 1)
    assert learned_heuristic is None
    heuristic_values, heuristic = heuristic_agent.get_value(inputs)
    torch.testing.assert_close(heuristic_values, inputs.h_value)
    torch.testing.assert_close(heuristic, inputs.h_value)


@pytest.mark.parametrize("state_combiner", ["film", "concat"])
def test_actor_and_critic_updates_leave_the_other_network_unchanged(state_combiner):
    cfg = Config()
    cfg.actor.state_embedding_head = "deep-set"
    cfg.actor.state_features_embedding_head = "mlp"
    cfg.actor.pair_embedding_head_in_dim = 6
    cfg.actor.state_embedding_head_out_dim = 8
    cfg.actor.state_features_embedding_head_out_dim = 6
    cfg.critic.state_embedding_head_out_dim = 12
    cfg.critic.state_features_embedding_head_out_dim = 10
    cfg.critic.critic_strategy = "learned-only"
    for model_cfg in (cfg.actor, cfg.critic):
        model_cfg.state_embedding_head_in_dim = 3
        model_cfg.state_features_embedding_head_in_dim = 4
        model_cfg.state_combiner = state_combiner

    agent = Agent(cfg)
    assert {id(p) for p in agent.actor.parameters()}.isdisjoint(
        id(p) for p in agent.critic.parameters()
    )
    inputs = ModelInputs(
        P=torch.randn(2, 3, 6),
        P_mask=torch.tensor([[True, True, False], [True, False, False]]),
        G=torch.randn(2, 2, 3),
        G_mask=torch.ones(2, 2, dtype=torch.bool),
        state_features=torch.randn(2, 4),
    )
    optimizer = torch.optim.SGD(agent.parameters(), lr=0.01)
    actor_before = [p.detach().clone() for p in agent.actor.parameters()]
    critic_before = [p.detach().clone() for p in agent.critic.parameters()]

    agent.actor(inputs)[inputs.P_mask].sum().backward()
    assert all(p.grad is None for p in agent.critic.parameters())
    optimizer.step()
    assert any(
        not torch.equal(before, after)
        for before, after in zip(actor_before, agent.actor.parameters())
    )
    for before, after in zip(critic_before, agent.critic.parameters()):
        torch.testing.assert_close(before, after, rtol=0, atol=0)

    optimizer.zero_grad(set_to_none=True)
    actor_before = [p.detach().clone() for p in agent.actor.parameters()]
    values, _ = agent.get_value(inputs)
    values.sum().backward()
    assert all(p.grad is None for p in agent.actor.parameters())
    optimizer.step()
    assert any(
        not torch.equal(before, after)
        for before, after in zip(critic_before, agent.critic.parameters())
    )
    for before, after in zip(actor_before, agent.actor.parameters()):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_training_can_run_twice_without_mutating_the_supplied_config(tmp_path):
    cfg = _small_config(tmp_path)
    before = asdict(cfg)
    results = [ppo_training.train(cfg), ppo_training.train(cfg)]
    assert asdict(cfg) == before
    for result in results:
        assert result.summary["global_step"] == 4
        assert result.summary["policy_updates"] == 1
    checkpoints = [
        torch.load(result.checkpoint_paths["actor"], weights_only=True)
        for result in results
    ]
    assert checkpoints[0].keys() == checkpoints[1].keys()
    for name in checkpoints[0]:
        torch.testing.assert_close(
            checkpoints[0][name], checkpoints[1][name], rtol=0, atol=0
        )


def test_training_closes_environments_and_log_files_when_rollout_fails(
    tmp_path, monkeypatch
):
    closed = []
    writers = []
    make_envs = ppo_training.create_training_envs
    make_writer = ppo_training.create_run_writer

    def capture_envs(cfg):
        envs = make_envs(cfg)
        original_close = envs.close

        def close():
            closed.append("envs")
            original_close()

        monkeypatch.setattr(envs, "close", close)
        return envs

    def capture_writer(cfg):
        name, writer = make_writer(cfg)
        writers.append(writer)
        return name, writer

    def fail_rollout(self):
        raise RuntimeError("injected rollout failure")

    monkeypatch.setattr(ppo_training, "create_training_envs", capture_envs)
    monkeypatch.setattr(ppo_training, "create_run_writer", capture_writer)
    monkeypatch.setattr(ppo_training.PPOTrainer, "_collect_rollout", fail_rollout)
    with pytest.raises(RuntimeError, match="injected rollout failure"):
        ppo_training.train(_small_config(tmp_path))
    assert closed == ["envs"]
    assert writers[0]._scalar_handle.closed
    assert writers[0]._scalar_csv_handle.closed
