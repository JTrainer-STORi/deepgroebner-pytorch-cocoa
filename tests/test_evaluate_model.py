from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import torch

from deepgroebner_pytorch_cocoa.combined_networks import CombinedPolicyNetwork
from deepgroebner_pytorch_cocoa.config_dataclasses import ActorConfig, ModelInputs
from evaluate_model import (
    _build_queue_usage_summary,
    _compute_queue_usage_stats,
    _extract_run_id,
    _get_state_feature_names,
    _load_actor,
    _normalize_gym_id,
    _resolve_config_path,
    _resolve_corpus_symbols,
    _resolve_model_path,
    _unwrap_config_section,
)


def test_resolve_corpus_symbols_keeps_single_width_corpus():
    rows = [
        {"symbols": "x[0],x[1],x[2]"},
        {"symbols": "x[0],x[1],x[2]"},
    ]
    assert _resolve_corpus_symbols(rows) == "x[0],x[1],x[2]"


def test_resolve_corpus_symbols_expands_to_widest_union():
    rows = [
        {"symbols": "x[0],x[1],x[2]"},
        {"symbols": "x[0],x[1],x[2],x[3]"},
        {"symbols": "x[0],x[1]"},
    ]
    assert _resolve_corpus_symbols(rows) == "x[0],x[1],x[2],x[3]"


def test_resolve_corpus_symbols_handles_geom_style_names():
    rows = [
        {"symbols": "x2,x3,x4,y3,y4"},
        {"symbols": "x2,x3,x4,x5,y3,y4,y5"},
    ]
    assert _resolve_corpus_symbols(rows) == "x2,x3,x4,x5,y3,y4,y5"


def test_load_actor_round_trips_current_pair_scoring_head(tmp_path):
    actor_cfg_dict = {
        "state_embedding_head": None,
        "state_features_embedding_head": None,
        "pair_embedding_head": "symmetric-per-entity-mlp",
        "use_lcm_gcd": False,
        "pair_embedding_head_out_dim": 8,
        "item_embedding_head_out_dim": 6,
    }
    actor_cfg = ActorConfig(**actor_cfg_dict)
    actor_cfg.state_embedding_head_in_dim = 3
    actor_cfg.state_features_embedding_head_in_dim = 4
    actor_cfg.pair_embedding_head_in_dim = 6

    original_actor = CombinedPolicyNetwork(actor_cfg).eval()
    checkpoint_path = tmp_path / "actor.pt"
    torch.save(original_actor.state_dict(), checkpoint_path)

    env = SimpleNamespace(
        observation_space={
            "G": gym.spaces.Box(low=-1.0, high=1.0, shape=(5, 3)),
            "state_features": gym.spaces.Box(low=-1.0, high=1.0, shape=(4,)),
            "P": gym.spaces.Box(low=-1.0, high=1.0, shape=(7, 6)),
        }
    )
    loaded_actor = _load_actor(
        actor_cfg_dict,
        env,
        checkpoint_path,
        torch.device("cpu"),
    )

    assert isinstance(loaded_actor.final_module, torch.nn.Sequential)

    inputs = ModelInputs(
        P=torch.tensor(
            [
                [
                    [1.0, 0.0, 2.0, 0.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0, 1.0, 0.0, 2.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ]
        ),
        P_mask=torch.tensor([[True, True, False]]),
        G=torch.tensor([[[1.0, 0.0, 2.0], [0.0, 1.0, 1.0], [2.0, 1.0, 0.0]]]),
        G_mask=torch.tensor([[True, True, True]]),
        state_features=torch.randn(1, 4),
    )

    with torch.no_grad():
        torch.testing.assert_close(loaded_actor(inputs), original_actor(inputs))


def test_resolve_model_path_filters_best_greedy_eval_vs_final(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    best_path = models_dir / "run__abc123__best_greedy_eval.pt"
    final_path = models_dir / "run__abc123.pt"
    best_path.touch()
    final_path.touch()

    assert (
        _resolve_model_path("abc123", models_dir, checkpoint_type="best_greedy_eval")
        == best_path.resolve()
    )
    assert (
        _resolve_model_path("abc123", models_dir, checkpoint_type="final")
        == final_path.resolve()
    )


def test_extract_run_id_handles_best_greedy_eval_suffix():
    best_path = Path(
        "BuchbergerExponents__3-20-10-weighted__20260524_012455__j77pf3uu__best_greedy_eval.pt"
    )
    final_path = Path(
        "BuchbergerExponents__3-20-10-weighted__20260524_012455__j77pf3uu.pt"
    )

    assert _extract_run_id(best_path) == "j77pf3uu"
    assert _extract_run_id(final_path) == "j77pf3uu"


def test_normalize_gym_id_maps_legacy_aliases():
    assert _normalize_gym_id("BuchbergerExponents") == "BuchbergerExponents-v1"
    assert _normalize_gym_id("BuchbergerExponentsNoPQ") == "BuchbergerExponentsNoPQ-v1"
    assert _normalize_gym_id("BuchbergerExponents-v1") == "BuchbergerExponents-v1"
    assert _normalize_gym_id(None) is None


def test_resolve_config_path_prefers_sidecar_config_json(tmp_path):
    model_path = tmp_path / "run__abc123.pt"
    model_path.touch()
    config_path = tmp_path / "run__abc123__config.json"
    config_path.write_text("{}", encoding="utf-8")

    resolved = _resolve_config_path(None, model_path, tmp_path / "wandb", "abc123")
    assert resolved == config_path.resolve()


def test_unwrap_config_section_supports_nested_and_plain_dicts():
    nested = {"env": {"value": {"gym_id": "BuchbergerExponents-v1"}}}
    plain = {"env": {"gym_id": "BuchbergerExponents-v1"}}

    assert _unwrap_config_section(nested, "env") == {"gym_id": "BuchbergerExponents-v1"}
    assert _unwrap_config_section(plain, "env") == {"gym_id": "BuchbergerExponents-v1"}


def test_get_state_feature_names_falls_back_to_default():
    env = SimpleNamespace(unwrapped=SimpleNamespace())
    names = _get_state_feature_names(env)

    assert names[1] == "P_cardinality"
    assert names[12] == "n_pairs_processed"


def test_compute_queue_usage_stats_tracks_binding_and_overflow():
    stats = _compute_queue_usage_stats([3.0, 7.0, 9.0], pq_budget=5)

    assert stats["pair_count_mean"] == 19.0 / 3.0
    assert stats["pair_count_median"] == 7.0
    assert stats["pair_count_min"] == 3.0
    assert stats["pair_count_max"] == 9.0
    assert stats["pq_budget"] == 5
    assert stats["pq_binding_step_fraction"] == 2.0 / 3.0
    assert stats["pq_binding_ever"] is True
    assert stats["pair_overflow_mean"] == 2.0
    assert stats["pair_overflow_max"] == 4.0


def test_build_queue_usage_summary_aggregates_episode_stats():
    rows = [
        {
            "pair_count_mean": 4.0,
            "pair_count_median": 4.0,
            "pair_count_min": 2.0,
            "pair_count_max": 7.0,
            "pq_budget": 5,
            "pq_binding_step_fraction": 0.25,
            "pq_binding_ever": True,
            "pair_overflow_mean": 0.5,
            "pair_overflow_max": 2.0,
        },
        {
            "pair_count_mean": 6.0,
            "pair_count_median": 6.0,
            "pair_count_min": 3.0,
            "pair_count_max": 9.0,
            "pq_budget": 5,
            "pq_binding_step_fraction": 0.75,
            "pq_binding_ever": False,
            "pair_overflow_mean": 1.5,
            "pair_overflow_max": 4.0,
        },
    ]

    summary = _build_queue_usage_summary(rows)

    assert summary["pq_budget"] == 5
    assert summary["pair_count_mean"]["mean"] == 5.0
    assert summary["pair_count_max"]["max"] == 9.0
    assert summary["pq_binding_step_fraction"]["median"] == 0.5
    assert summary["pq_binding_ever_rate"] == 0.5
    assert summary["pq_binding_ever_count"] == 1
    assert summary["pq_binding_ever_total"] == 2
