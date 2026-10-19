from dataclasses import asdict
import json
from pathlib import Path
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
    cfg.training.greedy_eval_num_examples = 0
    cfg.training.cuda = False
    cfg.training.track = False
    cfg.training.pytorch_num_threads = 1
    cfg.training.system_metrics_enabled = False
    cfg.training.output_directory = str(tmp_path)
    return cfg


@pytest.mark.parametrize(
    "pair_head", ["alphastar-buchberger-pair-transformer", "symmetric-per-entity-mlp"]
)
@pytest.mark.parametrize("state_head", [None, "deep-set"])
def test_training_attention_diagnostics_follow_their_own_completed_update_frequency(
    pair_head, state_head, tmp_path,
):
    #arrange
    cfg = _small_config(tmp_path)
    cfg.env.N = 4
    cfg.actor.pair_embedding_head = pair_head
    cfg.actor.state_embedding_head = state_head
    cfg.training.total_timesteps = 12
    cfg.training.attention_diagnostic_frequency = 2
    cfg.training.counterfactual_diagnostic_frequency = 3

    #act
    result = ppo_training.train(cfg)
    records = [
        json.loads(line)
        for line in (Path(result.run_directory) / "scalars.jsonl").read_text().splitlines()
    ]
    attention_records = [
        record for record in records if record["tag"].startswith("policy_function/diagnostic_attn_")
    ]
    counterfactual_records = [
        record for record in records if record["tag"].startswith("policy_function/diagnostic_cf_")
    ]

    #assert
    assert result.summary["policy_updates"] == 3
    if pair_head == "alphastar-buchberger-pair-transformer":
        assert attention_records
        assert {record["step"] for record in attention_records} == {8}
        assert all(np.isfinite(record["value"]) for record in attention_records)
    else:
        assert not attention_records
    if state_head is not None:
        assert counterfactual_records
        assert {record["step"] for record in counterfactual_records} == {12}
    else:
        assert not counterfactual_records


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
    #arrange
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.tensor([[0.5], [0.6], [0.7]])
    dones = torch.tensor([[0.0], [1.0], [0.0]])
    next_value = torch.tensor([[1.0]])
    next_done = torch.tensor([0.0])

    #act
    advantages, returns = compute_returns_and_advantages(
        rewards=rewards,
        values=values,
        dones=dones,
        next_value=next_value,
        next_done=next_done,
        gamma=0.9,
        gae_lambda=0.8,
        use_gae=use_gae,
    )

    #assert
    torch.testing.assert_close(advantages[:, 0], torch.tensor(expected_advantages))
    torch.testing.assert_close(returns[:, 0], torch.tensor(expected_returns))


def test_popart_preserves_raw_predictions_when_target_statistics_change():
    #arrange
    layer = torch.nn.Linear(3, 1)
    normalizer = PopArt(layer)
    features = torch.randn(4, 3)
    before = normalizer.denormalize(layer(features)).detach()

    #act
    normalizer.update(torch.tensor([10.0, 20.0, 30.0, 40.0]))

    #assert
    torch.testing.assert_close(
        normalizer.denormalize(layer(features)), before, atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize(
    "gym_id", ["BuchbergerExponents-v1", "BuchbergerExponentsNoPQ-v1"]
)
def test_rollout_snapshots_and_minibatch_order_survive_environment_steps(
    gym_id, tmp_path
):
    #arrange
    cfg = _small_config(tmp_path)
    cfg.critic.critic_strategy = "heuristic-plus-learned-residual"
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

        #act
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

        #assert
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
    #arrange
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

    #act
    learned_values, learned_heuristic = learned_agent.get_value(inputs)
    heuristic_values, heuristic = heuristic_agent.get_value(inputs)

    #assert
    assert learned_values.shape == (2, 1)
    assert learned_heuristic is None
    torch.testing.assert_close(heuristic_values, inputs.h_value)
    torch.testing.assert_close(heuristic, inputs.h_value)


@pytest.mark.parametrize("state_combiner", ["film", "concat"])
def test_actor_and_critic_updates_leave_the_other_network_unchanged(state_combiner):
    #arrange
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

    #act
    agent.actor(inputs)[inputs.P_mask].sum().backward()
    optimizer.step()

    #assert
    assert {id(p) for p in agent.actor.parameters()}.isdisjoint(
        id(p) for p in agent.critic.parameters()
    )
    assert all(p.grad is None for p in agent.critic.parameters())
    assert any(
        not torch.equal(before, after)
        for before, after in zip(actor_before, agent.actor.parameters())
    )
    for before, after in zip(critic_before, agent.critic.parameters()):
        torch.testing.assert_close(before, after, rtol=0, atol=0)

    #arrange
    optimizer.zero_grad(set_to_none=True)
    actor_before = [p.detach().clone() for p in agent.actor.parameters()]

    #act
    values, _ = agent.get_value(inputs)
    values.sum().backward()
    optimizer.step()

    #assert
    assert all(p.grad is None for p in agent.actor.parameters())
    assert any(
        not torch.equal(before, after)
        for before, after in zip(critic_before, agent.critic.parameters())
    )
    for before, after in zip(actor_before, agent.actor.parameters()):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_training_can_run_twice_without_mutating_the_supplied_config(tmp_path):
    #arrange
    cfg = _small_config(tmp_path)
    before = asdict(cfg)

    #act
    results = [ppo_training.train(cfg), ppo_training.train(cfg)]

    #assert
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


@pytest.mark.parametrize("configured_output", [False, True])
def test_grouped_training_keeps_logs_and_final_and_best_models_together(
    configured_output, monkeypatch, tmp_path
):
    #arrange
    monkeypatch.chdir(tmp_path)
    cfg = _small_config(tmp_path / "configured-output")
    if not configured_output:
        cfg.training.output_directory = None
    cfg.training.group = "sweep-001"
    cfg.training.save_actor = True
    cfg.training.save_critic = True
    cfg.training.greedy_eval_interval_steps = 4
    cfg.training.greedy_eval_num_examples = 1
    cfg.training.greedy_eval_on_train_env = True
    cfg.training.greedy_eval_on_nopq_env = False
    cfg.training.system_metrics_enabled = True
    cfg.training.system_metrics_interval_seconds = 0
    cfg.training.hparam_metrics_out = str(tmp_path / "summaries" / "trial.json")
    original_config = asdict(cfg)
    output_root = tmp_path / "configured-output" if configured_output else tmp_path
    group_root = output_root / "sweep-001"

    #act
    result = ppo_training.train(cfg)

    #assert
    assert asdict(cfg) == original_config
    assert result.summary["global_step"] == 4
    assert result.summary["greedy_eval_best_step"] == 4
    assert result.summary["greedy_eval_last_selection_metric"] is not None
    run_directory = Path(result.run_directory).resolve()
    assert run_directory.parent == group_root / "runs"
    for filename in ("scalars.jsonl", "scalars.csv", "system_metrics.jsonl"):
        assert (run_directory / filename).is_file()
    scalar_records = [
        json.loads(line)
        for line in (run_directory / "scalars.jsonl").read_text().splitlines()
    ]
    scalar_values = {record["tag"]: record["value"] for record in scalar_records}
    allowed_sections = {
        "training_rollouts",
        "evaluation_greedy",
        "policy_function",
        "value_function",
        "algorithm_general",
        "performance",
        "system_cpu",
        "system_gpu",
        "system_memory",
        "system_general",
        "priority_queue",
    }
    assert all(
        tag.count("/") == 1 and tag.split("/", 1)[0] in allowed_sections
        for tag in scalar_values
    )
    assert scalar_values["algorithm_general/policy_updates"] == 1
    assert "algorithm_general/learning_rate" not in scalar_values
    assert scalar_values["performance/steps_per_second"] >= 0
    assert scalar_values["evaluation_greedy/selection_metric"] == (
        result.summary["greedy_eval_last_selection_metric"]
    )
    assert scalar_values["evaluation_greedy/train_env_mean_return"] == (
        result.summary["greedy_eval_last_mean_return"]
    )
    assert scalar_values["value_function/weighted_loss"] == pytest.approx(
        cfg.training.critic_loss_coefficient * scalar_values["value_function/loss"]
    )
    assert scalar_values["policy_function/weighted_entropy_bonus"] == pytest.approx(
        cfg.training.entropy_coefficient * scalar_values["policy_function/entropy_bonus"]
    )
    assert {
        tag for tag in scalar_values if tag.startswith("policy_function/") and "entropy" in tag
    } == {
        "policy_function/entropy_raw",
        "policy_function/entropy_normalized",
        "policy_function/entropy_bonus",
        "policy_function/weighted_entropy_bonus",
    }
    if result.summary["episodes_completed"]:
        assert scalar_values["training_rollouts/episodes_completed"] == (
            result.summary["episodes_completed"]
        )
    assert (run_directory / "text" / "hyperparameters.md").is_file()
    assert (run_directory / "system_metrics.jsonl").read_text().strip()
    assert set(result.checkpoint_paths) == {"actor", "critic"}

    for kind in ("actor", "critic"):
        model_directory = group_root / "models" / f"{kind}s"
        final_path = Path(result.checkpoint_paths[kind]).resolve()
        best_path = Path(result.summary[f"greedy_eval_best_{kind}_path"]).resolve()
        assert final_path != best_path
        for checkpoint in (final_path, best_path):
            assert checkpoint.parent == model_directory
            assert checkpoint.suffix == ".pt"
            assert checkpoint.is_file()
            assert torch.load(checkpoint, weights_only=True)
            config_path = checkpoint.with_name(f"{checkpoint.stem}__config.json")
            saved_config = json.loads(config_path.read_text())
            assert saved_config["training"]["output_directory"] == cfg.training.output_directory
            assert saved_config["training"]["group"] == "sweep-001"

    summary_path = tmp_path / "summaries" / "trial.json"
    assert json.loads(summary_path.read_text()) == result.summary
    corpus_path = Path(result.summary["greedy_eval_corpus_path"]).resolve()
    assert corpus_path.parent == tmp_path / "eval_corpora" / "greedy_eval"


@pytest.mark.parametrize("normalization_enabled", [False, True])
@pytest.mark.parametrize("anneal_learning_rate", [False, True])
def test_update_metrics_keep_values_and_group_optional_normalizers(
    normalization_enabled, anneal_learning_rate, monkeypatch, tmp_path
):
    #arrange
    cfg = _small_config(tmp_path)
    cfg.training.console_verbosity = "quiet"
    cfg.training.anneal_learning_rate = anneal_learning_rate
    expected_general_tags = {"algorithm_general/policy_updates"}
    if anneal_learning_rate:
        expected_general_tags.add("algorithm_general/learning_rate")
    if normalization_enabled:
        expected_general_tags.update(
            {
                "algorithm_general/reward_running_return_mean",
                "algorithm_general/reward_running_return_std",
            }
        )
    records = []
    trainer = ppo_training.PPOTrainer.__new__(ppo_training.PPOTrainer)
    trainer.cfg = cfg
    trainer.writer = SimpleNamespace(
        add_scalar=lambda tag, value, step: records.append((tag, value, step))
    )
    trainer.optimizer = SimpleNamespace(param_groups=[{"lr": 0.0002}])
    trainer.policy_updates = 2
    trainer.global_step = 128
    trainer.start_time = 10.0
    trainer.num_updates = 10
    trainer.recent_returns = []
    trainer.last_k_avg_return = None
    trainer._record_system_metrics = lambda **kwargs: None
    monkeypatch.setattr(ppo_training.time, "time", lambda: 12.0)
    trainer.obs_sf_normalizer = (
        SimpleNamespace(mean=torch.tensor([2.0, 4.0]), var=torch.tensor([1.0, 9.0]))
        if normalization_enabled else None
    )
    trainer.reward_normalizer = (
        SimpleNamespace(rms=SimpleNamespace(
            mean=torch.tensor([6.0, 10.0]), var=torch.tensor([4.0, 16.0])
        ))
        if normalization_enabled else None
    )
    trainer.value_normalizer = (
        SimpleNamespace(mean=torch.tensor([3.0, 7.0]), var=torch.tensor([9.0, 25.0]))
        if normalization_enabled else None
    )
    metrics = SimpleNamespace(
        value_loss=2.5,
        value_clipfracs=[0.25, 0.5],
        critic_grad_norm=3.0,
        explained_variance=0.75,
        policy_loss=-0.2,
        entropy_bonus=0.4,
        entropy_raw=0.8,
        entropy_normalized=0.4,
        old_approx_kl=0.03,
        approx_kl=0.02,
        advantage_means=[1.0, 3.0],
        advantage_stds=[2.0, 4.0],
        actor_grad_norm=1.5,
        clipfracs=[0.1, 0.3],
    )
    batch = SimpleNamespace(observations={"state_features": torch.zeros(2, 14)})

    #act
    trainer._log_update(metrics, batch, update=3)

    #assert
    assert all(step == 128 for _, _, step in records)
    assert len({tag for tag, _, _ in records}) == len(records)
    values = {tag: value for tag, value, _ in records}
    assert {
        tag for tag in values if tag.startswith("algorithm_general/")
    } == expected_general_tags
    assert values["performance/steps_per_second"] == 64
    assert values["algorithm_general/policy_updates"] == 3
    if anneal_learning_rate:
        assert values["algorithm_general/learning_rate"] == 0.0002
    else:
        assert "algorithm_general/learning_rate" not in values
    assert values["value_function/advantage_mean"] == 2
    assert values["value_function/advantage_std"] == pytest.approx(np.sqrt(10))
    assert values["policy_function/loss"] == -0.2
    assert values["policy_function/weighted_entropy_bonus"] == pytest.approx(0.004)
    assert values["policy_function/clip_fraction"] == pytest.approx(0.2)
    assert values["value_function/weighted_loss"] == pytest.approx(0.25)
    assert values["value_function/clip_fraction"] == pytest.approx(0.375)
    assert values["value_function/explained_variance"] == 0.75
    normalization_values = {
        "policy_function/state_features_running_mean": 3.0,
        "policy_function/state_features_running_std": 2.0,
        "policy_function/state_features_running_std_max": 3.0,
        "policy_function/state_features_running_std_min": 1.0,
        "algorithm_general/reward_running_return_mean": 8.0,
        "algorithm_general/reward_running_return_std": 3.0,
        "value_function/running_target_mean": 5.0,
        "value_function/running_target_std": 4.0,
    }
    if normalization_enabled:
        for tag, expected in normalization_values.items():
            assert values[tag] == pytest.approx(expected)
    else:
        assert normalization_values.keys().isdisjoint(values)


@pytest.mark.parametrize("use_final_info", [False, True])
def test_completed_episodes_log_only_training_rollout_metrics(use_final_info):
    #arrange
    records = []
    trainer = ppo_training.PPOTrainer.__new__(ppo_training.PPOTrainer)
    trainer.writer = SimpleNamespace(
        add_scalar=lambda tag, value, step: records.append((tag, value, step))
    )
    trainer.global_step = 128
    trainer.episode_counter = 0
    trainer.recent_returns = []
    trainer.max_episodic_return = float("-inf")
    trainer.min_episodic_return = float("inf")
    episode_info = {
        "_episode": np.array([True, False, True]),
        "episode": {
            "r": np.array([-6.0, 0.0, -2.0]),
            "l": np.array([4, 0, 2]),
        },
    }
    infos = {"final_info": episode_info} if use_final_info else episode_info
    expected_episodes = [
        {
            "episodic_return": -6.0,
            "max_episodic_return": -6.0,
            "min_episodic_return": -6.0,
            "episodic_return_last_100_avg": -6.0,
            "episodic_length": 4,
            "episodes_completed": 1,
        },
        {
            "episodic_return": -2.0,
            "max_episodic_return": -2.0,
            "min_episodic_return": -6.0,
            "episodic_return_last_100_avg": -4.0,
            "episodic_length": 2,
            "episodes_completed": 2,
        },
    ]
    expected_records = [
        (f"training_rollouts/{key}", value, 128)
        for episode in expected_episodes
        for key, value in episode.items()
    ]

    #act
    trainer._log_completed_episodes(infos)

    #assert
    assert records == expected_records
    assert trainer.episode_counter == 2
    assert trainer.recent_returns == [-6.0, -2.0]


def test_training_closes_environments_and_log_files_when_rollout_fails(
    tmp_path, monkeypatch
):
    #arrange
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

    #act
    with pytest.raises(RuntimeError, match="injected rollout failure"):
        ppo_training.train(_small_config(tmp_path))

    #assert
    assert closed == ["envs"]
    assert writers[0]._scalar_handle.closed
    assert writers[0]._scalar_csv_handle.closed
