"""Observation, diagnostic info, and rollout alignment across episode boundaries."""

from dataclasses import fields
from functools import partial
from types import SimpleNamespace
import warnings

import gymnasium as gym
import numpy as np
import pytest
import torch

from deepgroebner_pytorch_cocoa.config_dataclasses import Config, ModelInputs
from deepgroebner_pytorch_cocoa.model_input_utils import (
    build_model_inputs_from_batched_obs,
    build_model_inputs_from_single_obs,
)
from deepgroebner_pytorch_cocoa.ppo_rollout import RolloutBuffer
from deepgroebner_pytorch_cocoa.training_env import (
    SingleEnvAdapter,
    create_training_envs,
)
from deepgroebner_pytorch_cocoa.wrapped import CGLeadMonomialsEnv

OBSERVATION_KEYS = {"G", "P", "G_mask", "P_mask", "state_features"}
METADATA_KEYS = {"P_idx_raw", "P_sugar", "P_normal_rank"}
RANKING_KEYS = {
    "P_idx_raw": "pair_idx_raw",
    "P_sugar": "pair_sugar",
    "P_normal_rank": "pair_normal_rank",
}
FIXED_BASIS = ["x^2 - y", "x*y - 1", "y^2 - x"]


def _assert_pair_info_shapes(obs, info):
    assert METADATA_KEYS <= info.keys()
    assert info["P_idx_raw"].shape == obs["P_mask"].shape + (2,)
    assert info["P_sugar"].shape == obs["P_mask"].shape
    assert info["P_normal_rank"].shape == obs["P_mask"].shape
    assert np.issubdtype(info["P_idx_raw"].dtype, np.integer)


@pytest.mark.parametrize(
    "gym_id", ["BuchbergerExponents-v1", "BuchbergerExponentsNoPQ-v1"]
)
def test_gym_observations_contain_only_model_inputs(gym_id):
    #arrange
    env = gym.make(
        gym_id,
        coeff_ring="QQ",
        symbols="x,y",
        basis_elements=FIXED_BASIS,
        elimination="none",
    )
    try:
        #act
        obs, info = env.reset(seed=123)
        inputs = build_model_inputs_from_single_obs(obs, torch.device("cpu"))

        #assert
        assert set(obs) == OBSERVATION_KEYS
        assert set(env.observation_space.spaces) == OBSERVATION_KEYS
        assert (METADATA_KEYS | {"P_idx"}).isdisjoint(vars(inputs))
        if "NoPQ" not in gym_id:
            _assert_pair_info_shapes(obs, info)

        #act
        next_obs, _, _, _, next_info = env.step(int(np.flatnonzero(obs["P_mask"])[0]))

        #assert
        assert set(next_obs) == OBSERVATION_KEYS
        if "NoPQ" not in gym_id:
            _assert_pair_info_shapes(next_obs, next_info)
    finally:
        env.close()


@pytest.mark.parametrize(
    "gym_id", ["BuchbergerExponents-v1", "BuchbergerExponentsNoPQ-v1"]
)
def test_gym_observations_remain_snapshots_after_steps_and_reset(gym_id):
    #arrange
    env = gym.make(
        gym_id,
        coeff_ring="QQ",
        symbols="x,y",
        basis_elements=FIXED_BASIS,
        elimination="none",
        k=1,
    )
    history = []

    def retain_and_check(obs):
        assert set(obs) == OBSERVATION_KEYS
        for old_obs, expected in history:
            for name in OBSERVATION_KEYS:
                assert not np.shares_memory(old_obs[name], obs[name]), name
                np.testing.assert_array_equal(old_obs[name], expected[name])
        history.append((obs, {name: obs[name].copy() for name in OBSERVATION_KEYS}))

    try:
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always")

            #act
            obs, _ = env.reset(seed=123)

            #assert
            retain_and_check(obs)
            for _ in range(2):
                #arrange
                action = int(np.flatnonzero(obs["P_mask"])[0])

                #act
                obs, _, terminated, truncated, _ = env.step(action)

                #assert
                assert not terminated and not truncated
                retain_and_check(obs)

            #arrange
            action = int(np.flatnonzero(obs["P_mask"])[0])

            #act
            obs, _, _, _, _ = env.step(action)

            #assert
            retain_and_check(obs)

            #act
            obs, _ = env.reset(seed=123)

            #assert
            retain_and_check(obs)

        #assert
        alias_warnings = [
            str(warning.message)
            for warning in emitted
            if "observation" in str(warning.message).lower()
            and (
                "memory" in str(warning.message).lower()
                or "alias" in str(warning.message).lower()
                or "share an object" in str(warning.message).lower()
            )
        ]
        assert not alias_warnings
    finally:
        env.close()


@pytest.mark.parametrize(
    "gym_id", ["BuchbergerExponents-v1", "BuchbergerExponentsNoPQ-v1"]
)
def test_caller_mutations_of_old_observations_do_not_change_native_state(gym_id):
    #arrange
    kwargs = dict(
        coeff_ring="QQ",
        symbols="x,y",
        basis_elements=FIXED_BASIS,
        elimination="none",
        k=1,
    )
    env = gym.make(gym_id, **kwargs)
    reference_env = gym.make(gym_id, **kwargs)
    try:
        retained_obs, _ = env.reset(seed=123)
        reference_obs, _ = reference_env.reset(seed=123)
        action = int(np.flatnonzero(reference_obs["P_mask"])[0])

        #act
        obs, _, terminated, truncated, _ = env.step(action)
        reference_obs, _, _, _, _ = reference_env.step(action)

        #assert
        assert not terminated and not truncated

        # Poison each earlier returned array after a later observation is available.
        for name in OBSERVATION_KEYS:
            #act
            retained_obs[name][...] = 0

            #assert
            np.testing.assert_array_equal(obs[name], reference_obs[name])

        #arrange
        action = int(np.flatnonzero(reference_obs["P_mask"])[0])
        for name in OBSERVATION_KEYS:
            obs[name][...] = 0

        # The native algorithm must still produce the same next state and reward.
        #act
        actual = env.step(action)
        expected = reference_env.step(action)

        #assert
        for name in OBSERVATION_KEYS:
            np.testing.assert_array_equal(actual[0][name], expected[0][name])
        assert actual[1:4] == expected[1:4]

        #act
        reset_obs, _ = env.reset(seed=123)
        reference_reset_obs, _ = reference_env.reset(seed=123)

        #assert
        for name in OBSERVATION_KEYS:
            np.testing.assert_array_equal(reset_obs[name], reference_reset_obs[name])
    finally:
        env.close()
        reference_env.close()


def test_native_pair_info_is_a_snapshot_and_reset_preserves_corpus_basis():
    #arrange
    env = CGLeadMonomialsEnv(
        coeff_ring="QQ",
        symbols="x,y",
        basis=FIXED_BASIS,
        elimination="none",
        N=8,
        k=1,
    )

    #act
    obs, info = env.reset()

    #assert
    assert set(obs) == (OBSERVATION_KEYS - {"state_features"}) | {"state_feats"}
    assert [value.replace(" ", "") for value in info["basis_elements"]] == [
        value.replace(" ", "") for value in FIXED_BASIS
    ]

    #arrange
    snapshots = []
    pair_counts = []
    for _ in range(4):
        #assert
        _assert_pair_info_shapes(obs, info)

        #arrange
        snapshots.append((info, {name: info[name].copy() for name in METADATA_KEYS}))
        pair_counts.append(int(obs["P_mask"].sum()))

        #act
        if obs["P_mask"].any():
            obs, _, _, info = env.step(int(np.flatnonzero(obs["P_mask"])[0]))
        else:
            obs, info = env.reset()

    #assert
    assert len(set(pair_counts)) > 1
    for old_info, expected in snapshots:
        for name, value in expected.items():
            np.testing.assert_array_equal(old_info[name], value)
            assert not np.shares_memory(old_info[name], info[name])


class MetadataEnv(gym.Env):
    """Give each episode/state distinct metadata to detect stale autoreset info."""

    def __init__(self, episode_length=2, truncate=False):
        self.episode_length = episode_length
        self.truncate = truncate
        self.episode = 0
        self.step_number = 0
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Dict(
            {
                "G": gym.spaces.Box(-np.inf, np.inf, (2, 1), np.float32),
                "P": gym.spaces.Box(-np.inf, np.inf, (2, 2), np.float32),
                "G_mask": gym.spaces.Box(0, 1, (2,), np.bool_),
                "P_mask": gym.spaces.Box(0, 1, (2,), np.bool_),
                "state_features": gym.spaces.Box(-np.inf, np.inf, (1,), np.float32),
            }
        )

    def _state(self):
        marker = 100 * self.episode + self.step_number
        obs = {
            "G": np.full((2, 1), marker, np.float32),
            "P": np.full((2, 2), marker, np.float32),
            "G_mask": np.ones(2, np.bool_),
            "P_mask": np.ones(2, np.bool_),
            "state_features": np.array([marker], np.float32),
        }
        info = {
            "P_idx_raw": np.full((2, 2), marker + 10, np.int64),
            "P_sugar": np.full(2, marker, np.float32),
            "P_normal_rank": np.full(2, marker + 20, np.float32),
        }
        return obs, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode += 1
        self.step_number = 0
        return self._state()

    def step(self, action):
        self.step_number += 1
        obs, info = self._state()
        finished = self.step_number == self.episode_length
        if finished:
            info["episode"] = {
                "r": float(self.step_number),
                "l": self.step_number,
                "t": 0.01,
            }
        return (
            obs,
            1.0,
            finished and not self.truncate,
            finished and self.truncate,
            info,
        )


def _assert_batched_metadata_matches_observation(obs, info):
    markers = obs["state_features"][:, 0]
    for key, offset in (("P_idx_raw", 10), ("P_sugar", 0), ("P_normal_rank", 20)):
        assert info["_" + key].all()
        values = info[key]
        expected = np.broadcast_to(
            (markers + offset).reshape((-1,) + (1,) * (values.ndim - 1)), values.shape
        )
        np.testing.assert_array_equal(values, expected)


@pytest.mark.parametrize("truncate", [False, True])
def test_single_adapter_keeps_reset_and_terminal_metadata_separate(truncate):
    #arrange
    env = SingleEnvAdapter(MetadataEnv(truncate=truncate))
    try:
        #act
        obs, info = env.reset(seed=[123])

        #assert
        _assert_batched_metadata_matches_observation(obs, info)

        #act
        obs, _, terminated, truncated, info = env.step([0])

        #assert
        assert not terminated.any() and not truncated.any()
        _assert_batched_metadata_matches_observation(obs, info)

        #act
        obs, rewards, terminated, truncated, info = env.step([0])

        #assert
        assert bool(terminated[0]) == (not truncate)
        assert bool(truncated[0]) == truncate
        np.testing.assert_array_equal(rewards, [1.0])
        np.testing.assert_array_equal(obs["state_features"], [[200]])
        _assert_batched_metadata_matches_observation(obs, info)
        assert info["_final_info"].tolist() == [True]
        assert info["_final_obs"].tolist() == [True]
        np.testing.assert_array_equal(info["final_obs"][0]["state_features"], [102])
        final_info = info["final_info"]
        np.testing.assert_array_equal(final_info["P_sugar"], [[102, 102]])
        np.testing.assert_array_equal(final_info["episode"]["l"], [2])
        assert "episode" not in info
    finally:
        env.close()


@pytest.mark.parametrize(
    "vector_class", [gym.vector.SyncVectorEnv, gym.vector.AsyncVectorEnv]
)
def test_vector_info_masks_and_snapshots_survive_partial_autoreset(vector_class):
    #arrange
    envs = vector_class(
        [
            partial(MetadataEnv, episode_length=2),
            partial(MetadataEnv, episode_length=3),
        ],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    try:
        #act
        obs, info = envs.reset(seed=[1, 2])

        #assert
        _assert_batched_metadata_matches_observation(obs, info)

        #arrange
        history = [(info, {name: info[name].copy() for name in METADATA_KEYS})]

        #act
        obs, _, _, _, info = envs.step([0, 0])

        #assert
        _assert_batched_metadata_matches_observation(obs, info)

        #act
        obs, _, terminated, _, info = envs.step([0, 0])

        #assert
        np.testing.assert_array_equal(terminated, [True, False])
        np.testing.assert_array_equal(obs["state_features"], [[200], [102]])
        _assert_batched_metadata_matches_observation(obs, info)
        np.testing.assert_array_equal(info["_final_info"], [True, False])
        np.testing.assert_array_equal(info["final_info"]["_P_sugar"], [True, False])
        np.testing.assert_array_equal(info["final_info"]["P_sugar"][0], [102, 102])

        #arrange
        history.append((info, {name: info[name].copy() for name in METADATA_KEYS}))

        #act
        obs, _, terminated, _, info = envs.step([0, 0])

        #assert
        np.testing.assert_array_equal(terminated, [False, True])
        np.testing.assert_array_equal(obs["state_features"], [[201], [200]])
        _assert_batched_metadata_matches_observation(obs, info)
        for old_info, expected in history:
            for name in METADATA_KEYS:
                np.testing.assert_array_equal(old_info[name], expected[name])
    finally:
        envs.close()


@pytest.mark.parametrize("vectorization_mode", ["sync", "async"])
@pytest.mark.parametrize(
    "gym_id", ["BuchbergerExponents-v1", "BuchbergerExponentsNoPQ-v1"]
)
def test_training_vectors_return_reset_info_after_completed_native_episode(
    vectorization_mode, gym_id
):
    #arrange
    cfg = Config()
    cfg.env.gym_id = gym_id
    cfg.env.coeff_ring = "QQ"
    cfg.env.symbols = "x,y"
    cfg.env.basis_elements = FIXED_BASIS
    cfg.env.elimination = "none"
    cfg.env.N = 8
    cfg.env.k = 1
    cfg.training.num_envs = 2
    cfg.training.vectorization_mode = vectorization_mode
    envs = create_training_envs(cfg)
    try:
        obs, info = envs.reset(seed=[123, 124])
        expected_reset = {
            name: info[name].copy() for name in METADATA_KEYS if name in info
        }
        completed = np.zeros(2, dtype=bool)
        for _ in range(20):
            #arrange
            assert all(np.any(mask) for mask in obs["P_mask"])
            actions = np.array([np.flatnonzero(mask)[0] for mask in obs["P_mask"]])

            #act
            obs, _, terminated, truncated, info = envs.step(actions)

            #assert
            if "NoPQ" not in gym_id:
                _assert_pair_info_shapes(obs, info)
            else:
                assert METADATA_KEYS.isdisjoint(info)
            finished = terminated | truncated
            completed |= finished
            for index in np.flatnonzero(finished):
                assert obs["P_mask"][index].any()
                assert not info["final_obs"][index]["P_mask"].any()
                assert info["final_info"]["_episode"][index]
                for name in expected_reset:
                    np.testing.assert_array_equal(
                        info[name][index], expected_reset[name][index]
                    )
            if completed.all():
                break

        #assert
        assert completed.all()
    finally:
        envs.close()


def _rollout_config(*, frequency=1):
    cfg = Config()
    cfg.training.num_envs = 2
    cfg.training.num_steps_per_update = 2
    cfg.training.ranking_diagnostic_frequency = frequency
    return cfg


def _rollout_spaces():
    env = MetadataEnv()
    return SimpleNamespace(
        single_observation_space=env.observation_space,
        single_action_space=env.action_space,
    )


def test_rollout_ranking_metadata_snapshots_follow_shuffled_minibatches():
    #arrange
    cfg = _rollout_config()

    #act
    buffer = RolloutBuffer(cfg, _rollout_spaces(), torch.device("cpu"), ragged=False)

    #assert
    assert set(buffer.observations) == OBSERVATION_KEYS
    assert set(buffer.pair_metadata) == METADATA_KEYS
    assert all(isinstance(value, np.ndarray) for value in buffer.pair_metadata.values())

    #arrange
    expected = {name: [] for name in METADATA_KEYS}
    sources = [MetadataEnv(), MetadataEnv()]
    sources[1].episode = 1
    records = [source.reset() for source in sources]

    #act
    for step in range(2):
        obs = {
            name: np.stack([record[0][name] for record in records])
            for name in OBSERVATION_KEYS
        }
        info = {
            name: np.stack([record[1][name] for record in records])
            for name in METADATA_KEYS
        }
        for name in METADATA_KEYS:
            info["_" + name] = np.ones(2, dtype=bool)
            expected[name].append(info[name].copy())
        inputs = build_model_inputs_from_batched_obs(obs, torch.device("cpu"))
        buffer.store_observation(step, inputs, torch.zeros(2), info=info)
        for name in METADATA_KEYS:
            info[name].fill(-999)
        next_records = [source.step(0) for source in sources]
        records = [(record[0], record[4]) for record in next_records]

    batch = buffer.flatten(torch.zeros(2, 2), torch.zeros(2, 2))
    indices = torch.tensor([3, 0, 2, 1])
    metadata = batch.ranking_metadata(indices)
    inputs = batch.model_inputs(indices)

    #assert
    assert set(metadata) == set(RANKING_KEYS.values())
    for source_name, ranking_name in RANKING_KEYS.items():
        values = np.stack(expected[source_name]).reshape(
            (-1,) + expected[source_name][0].shape[1:]
        )
        actual = metadata[ranking_name]
        assert actual.device == batch.device
        np.testing.assert_array_equal(actual.numpy(), values[indices.numpy()])
    assert (METADATA_KEYS | {"P_idx"}).isdisjoint(vars(inputs))
    np.testing.assert_array_equal(inputs.state_features[:, 0], [201, 100, 101, 200])


@pytest.mark.parametrize("frequency,ragged", [(0, False), (1, True)])
def test_rollout_omits_pair_metadata_when_disabled_or_unavailable(frequency, ragged):
    #arrange
    cfg = _rollout_config(frequency=frequency)
    spaces = _rollout_spaces()

    #act
    buffer = RolloutBuffer(
        cfg,
        spaces,
        torch.device("cpu"),
        ragged=ragged,
    )

    #assert
    assert buffer.pair_metadata is None

    #act
    batch = buffer.flatten(torch.zeros(2, 2), torch.zeros(2, 2))
    metadata = batch.ranking_metadata([0])

    #assert
    assert batch.pair_metadata is None
    assert metadata == {}
    assert (METADATA_KEYS | {"P_idx"}).isdisjoint(
        field.name for field in fields(ModelInputs)
    )
