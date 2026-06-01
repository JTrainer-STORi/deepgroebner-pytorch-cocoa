from __future__ import annotations

import gymnasium as gym
import pytest

import deepgroebner_pytorch_cocoa
from deepgroebner_pytorch_cocoa.runner import TrainConfig, build_training_cli_args, build_training_command


def test_symmetric_per_entity_mlp_preset_uses_pair_only_scoring():
    args = build_training_cli_args(TrainConfig(architecture="symmetric-per-entity-mlp"))

    assert "--actor.state_embedding_head=None" in args
    assert "--actor.state_features_embedding_head=None" in args
    assert "--actor.pair_embedding_head=symmetric-per-entity-mlp" in args
    assert "--actor.pair_combiner=pair-only" in args
    assert "--training.track=False" in args
    assert "--training.console_verbosity=normal" in args


def test_spem_deep_set_preset_uses_pair_and_state_encoders():
    args = build_training_cli_args(TrainConfig(architecture="spem-deep-set"))

    assert "--actor.state_embedding_head=deep-set" in args
    assert "--actor.state_features_embedding_head=None" in args
    assert "--actor.pair_embedding_head=symmetric-per-entity-mlp" in args
    assert "--actor.pair_combiner=film" in args


def test_spem_deep_set_state_features_preset_enables_both_state_paths():
    args = build_training_cli_args(TrainConfig(architecture="spem-deep-set-state-features"))

    assert "--actor.state_embedding_head=deep-set" in args
    assert "--actor.state_features_embedding_head=mlp" in args
    assert "--actor.state_combiner=film" in args
    assert "--actor.pair_embedding_head=symmetric-per-entity-mlp" in args
    assert "--actor.pair_combiner=film" in args


def test_build_training_command_points_at_trainer_entrypoint():
    command = build_training_command(TrainConfig(dry_run=True))

    assert command[0]
    assert command[1].endswith("ppo_masked.py")
    assert "--env.gym_id=BuchbergerExponents-v1" in command


def test_experimental_heuristic_choice_env_is_not_registered():
    with pytest.raises(gym.error.NameNotFound):
        gym.spec("BuchbergerHeuristicChoice-v1")
