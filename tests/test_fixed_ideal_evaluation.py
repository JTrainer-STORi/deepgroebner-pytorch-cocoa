"""Greedy evaluation must replay the single ideal used for training."""

from copy import deepcopy

import gymnasium as gym
import numpy as np
import pytest

from deepgroebner_pytorch_cocoa.config_dataclasses import Config
from deepgroebner_pytorch_cocoa.greedy_eval import (
    build_greedy_eval_corpus,
    build_greedy_eval_seeds,
    greedy_eval_corpus_cache_path,
    resolve_greedy_eval_corpus,
)
from deepgroebner_pytorch_cocoa.ideal_corpus import split_ring_symbols
from deepgroebner_pytorch_cocoa.ppo_training import PPOTrainer
from deepgroebner_pytorch_cocoa.training_env import (
    build_env_kwargs_for_gym_id,
    build_fixed_ideal_env_kwargs_for_gym_id,
)
from deepgroebner_pytorch_cocoa.training_utils import resolve_training_config
from deepgroebner_pytorch_cocoa.wrapped import CGLeadMonomialsEnv


ENVIRONMENTS = ["BuchbergerExponents-v1", "BuchbergerExponentsNoPQ-v1"]


def _config(tmp_path, gym_id=ENVIRONMENTS[0]):
    cfg = Config()
    cfg.env.gym_id = gym_id
    cfg.env.ideal_dist = "2-3-2-weighted"
    cfg.env.fix_ideal = True
    cfg.env.env_seed = 12
    cfg.env.elimination = "none"
    cfg.env.N = 8
    cfg.training.num_envs = 1
    cfg.training.num_steps_per_update = 2
    cfg.training.total_timesteps = 0
    cfg.training.num_epochs_per_update = 1
    cfg.training.num_minibatches = 1
    cfg.training.cuda = False
    cfg.training.track = False
    cfg.training.pytorch_num_threads = 1
    cfg.training.normalize_observation_state_features = False
    cfg.training.system_metrics_enabled = False
    cfg.training.console_verbosity = "quiet"
    cfg.training.save_actor = False
    cfg.training.save_critic = False
    cfg.training.output_directory = str(tmp_path / "output")
    cfg.training.greedy_eval_first_step = 0
    cfg.training.greedy_eval_interval_steps = 1
    cfg.training.greedy_eval_num_examples = 10
    cfg.training.greedy_eval_seed = 9876
    cfg.training.greedy_eval_on_train_env = True
    cfg.training.greedy_eval_on_nopq_env = True
    return cfg


def _assert_observations_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for name in actual:
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)


def _seed_for_variable_count(ideal_dist, n_vars):
    # C++ standard libraries can map the same seed to different ranged samples.
    # Find a seed locally so each case exercises its intended ring size.
    sampler = CGLeadMonomialsEnv(ideal_dist=ideal_dist, elimination="none", fix_ideal=False)
    for seed in range(128):
        sampler.seed(seed)
        sampler.reset()
        if len(split_ring_symbols(sampler.initial_ideal()["symbols"])) == n_vars:
            return seed
    pytest.fail(f"Could not sample a {n_vars}-variable ring from {ideal_dist}")


@pytest.mark.parametrize("gym_id", ENVIRONMENTS)
@pytest.mark.parametrize(
    "source,expected_ring,expected_nvars",
    [
        ("explicit", "ZZmod(7)", 2),
        ("2-3-2-weighted", "ZZmod(32003)", 2),
        ("(2-3)-3-2-weighted", "ZZmod(32003)", 2),
        ("geom-(3-4)", "ZZmod(32003)", 3),
    ],
    ids=["explicit", "binomial", "variable-range", "geometry-range"],
)
def test_greedy_corpus_replays_the_training_ideal_and_ring(
    tmp_path, gym_id, source, expected_ring, expected_nvars
):
    #arrange
    cfg = _config(tmp_path, gym_id)
    if source == "explicit":
        cfg.env.coeff_ring = expected_ring
        cfg.env.symbols = "x,y"
        cfg.env.basis_elements = "x^2-y,x*y-1"
    else:
        cfg.env.ideal_dist = source
        if source in {"(2-3)-3-2-weighted", "geom-(3-4)"}:
            cfg.env.env_seed = _seed_for_variable_count(source, expected_nvars)
    cfg = resolve_training_config(cfg)
    seeds = build_greedy_eval_seeds(cfg, True)

    #act
    rows = build_greedy_eval_corpus(cfg, seeds)

    #assert
    assert len(rows) == 1
    row = rows[0]
    assert row["episode_seed"] == cfg.training.greedy_eval_seed
    assert row["coeff_ring"] == expected_ring
    assert len(row["symbols"].split(",")) == expected_nvars
    training_env = gym.make(gym_id, **build_env_kwargs_for_gym_id(cfg, gym_id))
    evaluation_env = gym.make(
        gym_id, **build_fixed_ideal_env_kwargs_for_gym_id(cfg, gym_id, row)
    )
    try:
        training_obs, _ = training_env.reset(seed=cfg.env.env_seed)
        evaluation_obs, _ = evaluation_env.reset(seed=row["episode_seed"])
        for _ in range(300):
            _assert_observations_equal(training_obs, evaluation_obs)
            action = int(np.flatnonzero(training_obs["P_mask"])[0])

            #act
            training_obs, reward, terminated, truncated, _ = training_env.step(action)
            evaluation_obs, eval_reward, eval_terminated, eval_truncated, _ = (
                evaluation_env.step(action)
            )

            #assert
            assert (reward, terminated, truncated) == (
                eval_reward, eval_terminated, eval_truncated
            )
            if terminated or truncated:
                break
        assert terminated and not truncated
        _assert_observations_equal(training_obs, evaluation_obs)
        assert training_env.unwrapped.env.is_groebner_basis()
        assert evaluation_env.unwrapped.env.is_groebner_basis()

        # A new episode seed cannot change either environment's fixed input.
        training_obs, _ = training_env.reset(seed=cfg.env.env_seed + 1)
        evaluation_obs, _ = evaluation_env.reset(seed=row["episode_seed"] + 1)
        _assert_observations_equal(training_obs, evaluation_obs)
    finally:
        training_env.close()
        evaluation_env.close()


@pytest.mark.parametrize("gym_id", ENVIRONMENTS)
@pytest.mark.parametrize(
    "num_envs,vectorization", [(1, "sync"), (2, "sync"), (2, "async")],
    ids=["single", "sync-workers", "async-workers"],
)
def test_training_workers_and_both_greedy_targets_share_one_fixed_ideal(
    tmp_path, monkeypatch, gym_id, num_envs, vectorization
):
    #arrange
    monkeypatch.chdir(tmp_path)
    cfg = _config(tmp_path, gym_id)
    cfg.training.num_envs = num_envs
    cfg.training.vectorization_mode = vectorization
    trainer = PPOTrainer(cfg)

    #act
    trainer.run()

    #assert
    assert len(trainer.evaluation.corpus_rows) == 1
    assert trainer.evaluation.last_train["num_examples"] == 1
    assert trainer.evaluation.last_train["num_completed"] == 1
    if "nopq" not in gym_id.lower():
        assert trainer.evaluation.last_nopq["num_examples"] == 1
        assert trainer.evaluation.last_nopq["num_completed"] == 1
    row = trainer.evaluation.corpus_rows[0]
    evaluation_env = gym.make(
        gym_id, **build_fixed_ideal_env_kwargs_for_gym_id(trainer.cfg, gym_id, row)
    )
    try:
        expected, _ = evaluation_env.reset(seed=row["episode_seed"])
        for worker in range(num_envs):
            actual = {
                name: getattr(trainer.inputs, name)[worker].detach().cpu().numpy()
                for name in expected
            }
            _assert_observations_equal(actual, expected)
    finally:
        evaluation_env.close()


def test_generated_fixed_corpus_uses_training_seed_and_invalidates_cache(tmp_path):
    #arrange
    cfg = resolve_training_config(_config(tmp_path))
    seeds = build_greedy_eval_seeds(cfg, True)
    cache_root = tmp_path / "corpora"

    #act
    rows, path, generated = resolve_greedy_eval_corpus(cfg, seeds, cache_root=cache_root)
    cached_rows, cached_path, cached_generated = resolve_greedy_eval_corpus(
        cfg, seeds, cache_root=cache_root
    )
    other_training_seed = deepcopy(cfg)
    other_training_seed.env.env_seed += 1
    other_rows, other_path, other_generated = resolve_greedy_eval_corpus(
        other_training_seed, seeds, cache_root=cache_root
    )
    other_evaluation_seed = deepcopy(cfg)
    other_evaluation_seed.training.greedy_eval_seed += 100
    other_seeds = build_greedy_eval_seeds(other_evaluation_seed, True)
    replay_rows = build_greedy_eval_corpus(other_evaluation_seed, other_seeds)

    #assert
    assert generated and not cached_generated and other_generated
    assert rows == cached_rows and path == cached_path
    assert path != other_path
    assert rows[0]["basis_elements"] != other_rows[0]["basis_elements"]
    for name in ("coeff_ring", "symbols", "basis_elements"):
        assert rows[0][name] == replay_rows[0][name]
    assert rows[0]["episode_seed"] != replay_rows[0]["episode_seed"]


@pytest.mark.parametrize("field,value", [("coeff_ring", "ZZmod(7)"), ("symbols", "y,x")])
def test_explicit_fixed_cache_distinguishes_the_input_ring(tmp_path, field, value):
    #arrange
    cfg = _config(tmp_path)
    cfg.env.coeff_ring = "QQ"
    cfg.env.symbols = "x,y"
    cfg.env.basis_elements = "x^2-y,x*y-1"
    cfg = resolve_training_config(cfg)
    seeds = build_greedy_eval_seeds(cfg, True)
    other = deepcopy(cfg)
    setattr(other.env, field, value)

    #act
    path = greedy_eval_corpus_cache_path(cfg, seeds, cache_root=tmp_path)
    other_path = greedy_eval_corpus_cache_path(other, seeds, cache_root=tmp_path)

    #assert
    assert path != other_path
