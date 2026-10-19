from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch

from deepgroebner_pytorch_cocoa.combined_networks import CombinedPolicyNetwork
from deepgroebner_pytorch_cocoa.config_dataclasses import ActorConfig, EnvConfig, ModelInputs
from deepgroebner_pytorch_cocoa.model_evaluation import (
    EvalCorpusConfig,
    TestCorpusConfig as CorpusTestConfig,
    _build_env_kwargs,
    _build_queue_usage_summary,
    _compute_queue_usage_stats,
    _extract_run_id,
    _get_state_feature_names,
    _load_actor,
    _normalize_gym_id,
    _resolve_config_path,
    _resolve_corpus_symbols,
    _resolve_model_path,
    _resolve_observation_nvars,
    _unwrap_config_section,
    evaluate,
)


def test_legacy_config_name_aliases_test_corpus_config():
    assert EvalCorpusConfig is CorpusTestConfig


def test_resolve_corpus_symbols_keeps_single_width_corpus():
    #arrange
    rows = [
        {"symbols": "x[0],x[1],x[2]"},
        {"symbols": "x[0],x[1],x[2]"},
    ]
    #act
    symbols = _resolve_corpus_symbols(rows)

    #assert
    assert symbols == "x[0],x[1],x[2]"


def test_resolve_corpus_symbols_selects_widest_ring():
    #arrange
    rows = [
        {"symbols": "x[0],x[1],x[2]"},
        {"symbols": "x[0],x[1],x[2],x[3]"},
        {"symbols": "x[0],x[1]"},
    ]
    #act
    symbols = _resolve_corpus_symbols(rows)

    #assert
    assert symbols == "x[0],x[1],x[2],x[3]"


def test_resolve_corpus_symbols_handles_geom_style_names():
    #arrange
    rows = [
        {"symbols": "x2,x3,x4,y3,y4"},
        {"symbols": "x2,x3,x4,x5,y3,y4,y5"},
    ]
    #act
    symbols = _resolve_corpus_symbols(rows)

    #assert
    assert symbols == "x2,x3,x4,x5,y3,y4,y5"


def test_resolve_corpus_symbols_keeps_different_ring_names_separate():
    rows = [{"symbols": "a,b"}, {"symbols": "x,y,z"}, {"symbols": "u,v"}]

    assert _resolve_corpus_symbols(rows) == "x,y,z"


def test_resolve_observation_nvars_uses_training_width_for_smaller_corpus():
    env_cfg = EnvConfig(symbols="x,y,z,w")
    rows = [{"symbols": "a,b"}, {"symbols": "u,v,t"}]

    assert _resolve_observation_nvars(env_cfg, rows) == 4


def test_resolve_observation_nvars_falls_back_to_widest_corpus_ring():
    rows = [{"symbols": "a,b"}, {"symbols": "u,v,t"}]

    assert _resolve_observation_nvars(EnvConfig(), rows) == 3


def test_resolve_observation_nvars_counts_indexed_symbols():
    env_cfg = EnvConfig(symbols="x[1,2],x[2,3],x[3,4]")
    rows = [{"symbols": "u[1,2],v[2,3]"}]

    assert _resolve_observation_nvars(env_cfg, rows) == 3


def test_resolve_observation_nvars_rejects_corpus_wider_than_checkpoint():
    with pytest.raises(ValueError, match="requires 3 variables.*at most 2"):
        _resolve_observation_nvars(EnvConfig(symbols="x,y"), [{"symbols": "a,b,c"}])


def test_build_env_kwargs_preserves_row_ring_and_sets_training_width():
    row = {"coeff_ring": "ZZmod(101)", "symbols": "a,b", "basis_elements": ["a+b"]}

    kwargs = _build_env_kwargs(EnvConfig(), 4, row)

    assert kwargs["coeff_ring"] == "ZZmod(101)"
    assert kwargs["symbols"] == "a,b"
    assert kwargs["observation_nvars"] == 4
    assert kwargs["basis_elements"] == ["a+b"]


@pytest.mark.parametrize(
    ("gym_id", "verbosity"),
    [
        ("BuchbergerExponents-v1", "normal"),
        ("BuchbergerExponentsNoPQ-v1", "verbose"),
        ("BuchbergerExponentsNoPQ-v1", "quiet"),
    ],
)
def test_evaluate_checkpoint_on_smaller_ideals_from_different_rings(
    tmp_path, capsys, gym_id, verbosity
):
    env_cfg = EnvConfig(gym_id=gym_id, symbols="x,y,z", N=16, k=2, elimination="none")
    actor_cfg = ActorConfig(
        state_embedding_head=None,
        state_features_embedding_head=None,
        pair_embedding_head="symmetric-per-entity-mlp",
        use_lcm_gcd=False,
        pair_embedding_head_out_dim=8,
        item_embedding_head_out_dim=6,
    )
    actor_cfg.state_embedding_head_in_dim = 6
    actor_cfg.state_features_embedding_head_in_dim = 14
    actor_cfg.pair_embedding_head_in_dim = 12
    model_path = tmp_path / "mixed_rings.pt"
    torch.save(CombinedPolicyNetwork(actor_cfg).state_dict(), model_path)
    config_path = tmp_path / "mixed_rings__config.json"
    config_path.write_text(
        json.dumps({"env": asdict(env_cfg), "actor": asdict(actor_cfg)}), encoding="utf-8"
    )
    corpus_path = tmp_path / "mixed_rings.jsonl"
    corpus_rows = [
        {"coeff_ring": "QQ", "symbols": "u,v", "basis_elements": ["u^2-v", "u*v-1"]},
        {"coeff_ring": "ZZmod(101)", "symbols": "a", "basis_elements": ["a^2+1", "a+1"]},
    ]
    corpus_path.write_text("\n".join(json.dumps(row) for row in corpus_rows), encoding="utf-8")

    result = evaluate(CorpusTestConfig(
        corpus=str(corpus_path),
        model_name=str(model_path),
        gym_id=gym_id,
        output_root=str(tmp_path / "results"),
        device="cpu",
        verbosity=verbosity,
    ))

    assert result.summary["num_evals"] == 2
    output_dir = Path(result.output_directory)
    assert output_dir.parent == tmp_path / "results"
    assert output_dir.name.startswith(f"test_corpus__{gym_id}__")
    assert all(row["length"] > 0 for row in result.episode_results)
    metadata = json.loads(Path(result.artifact_paths["metadata"]).read_text(encoding="utf-8"))
    assert metadata["observation_nvars"] == 3
    assert metadata["resource_measurements"]["time_unit"] == "seconds"
    assert metadata["resource_measurements"]["memory_unit"] == "bytes"
    assert "after reset" in metadata["resource_measurements"]["episode_timing_scope"]
    assert "excludes report writing" in metadata["resource_measurements"]["total_timing_scope"]

    saved_summary = json.loads(Path(result.artifact_paths["summary"]).read_text(encoding="utf-8"))
    saved_rows = [
        json.loads(line)
        for line in Path(result.artifact_paths["episode_results_jsonl"]).read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert saved_summary == result.summary
    assert saved_rows == result.episode_results
    for measurement in [saved_summary["resources"], *saved_rows]:
        assert measurement["elapsed_seconds"] >= 0.0
        assert measurement["cpu_seconds"] >= 0.0
        for field in ["rss_start_bytes", "rss_end_bytes", "process_peak_rss_bytes"]:
            assert measurement[field] is None or (
                isinstance(measurement[field], int) and measurement[field] > 0
            )
        assert measurement["rss_delta_bytes"] is None or isinstance(
            measurement["rss_delta_bytes"], int
        )
        for field in [
            "cuda_allocated_start_bytes",
            "cuda_allocated_end_bytes",
            "cuda_allocated_delta_bytes",
            "cuda_reserved_start_bytes",
            "cuda_reserved_end_bytes",
            "cuda_reserved_delta_bytes",
            "cuda_peak_allocated_bytes",
            "cuda_peak_reserved_bytes",
        ]:
            assert measurement[field] is None

    for field in ["elapsed_seconds", "cpu_seconds"]:
        assert saved_summary[f"{field}_sum"] == pytest.approx(
            sum(row[field] for row in saved_rows)
        )
        assert saved_summary[f"{field}_mean"] >= 0.0
        assert saved_summary[f"{field}_min"] <= saved_summary[f"{field}_median"]
        assert saved_summary[f"{field}_median"] <= saved_summary[f"{field}_max"]
        assert saved_summary["resources"][field] >= saved_summary[f"{field}_sum"]

    output = capsys.readouterr().out
    if verbosity == "quiet":
        assert output == f"Wrote test corpus evaluation to {result.output_directory}\n"
    else:
        assert "Mean return:" in output
        assert "CPU" in output
        assert "RSS" in output
        if verbosity == "verbose":
            assert "[1/2] return=" in output


def test_load_actor_round_trips_current_pair_scoring_head(tmp_path):
    #arrange
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

    #act
    loaded_actor = _load_actor(
        actor_cfg_dict,
        env,
        checkpoint_path,
        torch.device("cpu"),
    )
    with torch.no_grad():
        loaded_predictions = loaded_actor(inputs)
        original_predictions = original_actor(inputs)

    #assert
    assert isinstance(loaded_actor.final_module, torch.nn.Sequential)
    torch.testing.assert_close(loaded_predictions, original_predictions)


def test_resolve_model_path_filters_best_greedy_eval_vs_final(tmp_path):
    #arrange
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    best_path = models_dir / "run__abc123__best_greedy_eval.pt"
    final_path = models_dir / "run__abc123.pt"
    best_path.touch()
    final_path.touch()

    #act
    resolved_best = _resolve_model_path(
        "abc123", models_dir, checkpoint_type="best_greedy_eval"
    )
    resolved_final = _resolve_model_path("abc123", models_dir, checkpoint_type="final")

    #assert
    assert resolved_best == best_path.resolve()
    assert resolved_final == final_path.resolve()


def test_extract_run_id_handles_best_greedy_eval_suffix():
    #arrange
    best_path = Path(
        "BuchbergerExponents__3-20-10-weighted__20260524_012455__j77pf3uu__best_greedy_eval.pt"
    )
    final_path = Path(
        "BuchbergerExponents__3-20-10-weighted__20260524_012455__j77pf3uu.pt"
    )

    #act
    best_run_id = _extract_run_id(best_path)
    final_run_id = _extract_run_id(final_path)

    #assert
    assert best_run_id == "j77pf3uu"
    assert final_run_id == "j77pf3uu"


def test_normalize_gym_id_maps_legacy_aliases():
    #arrange
    legacy_pq = "BuchbergerExponents"
    legacy_no_pq = "BuchbergerExponentsNoPQ"
    current = "BuchbergerExponents-v1"

    #act
    normalized_pq = _normalize_gym_id(legacy_pq)
    normalized_no_pq = _normalize_gym_id(legacy_no_pq)
    normalized_current = _normalize_gym_id(current)
    normalized_missing = _normalize_gym_id(None)

    #assert
    assert normalized_pq == "BuchbergerExponents-v1"
    assert normalized_no_pq == "BuchbergerExponentsNoPQ-v1"
    assert normalized_current == "BuchbergerExponents-v1"
    assert normalized_missing is None


def test_resolve_config_path_prefers_sidecar_config_json(tmp_path):
    #arrange
    model_path = tmp_path / "run__abc123.pt"
    model_path.touch()
    config_path = tmp_path / "run__abc123__config.json"
    config_path.write_text("{}", encoding="utf-8")

    #act
    resolved = _resolve_config_path(None, model_path, tmp_path / "wandb", "abc123")

    #assert
    assert resolved == config_path.resolve()


def test_unwrap_config_section_supports_nested_and_plain_dicts():
    #arrange
    nested = {"env": {"value": {"gym_id": "BuchbergerExponents-v1"}}}
    plain = {"env": {"gym_id": "BuchbergerExponents-v1"}}

    #act
    nested_env = _unwrap_config_section(nested, "env")
    plain_env = _unwrap_config_section(plain, "env")

    #assert
    assert nested_env == {"gym_id": "BuchbergerExponents-v1"}
    assert plain_env == {"gym_id": "BuchbergerExponents-v1"}


def test_get_state_feature_names_falls_back_to_default():
    #arrange
    env = SimpleNamespace(unwrapped=SimpleNamespace())

    #act
    names = _get_state_feature_names(env)

    #assert
    assert names[1] == "P_cardinality"
    assert names[12] == "n_pairs_processed"


def test_compute_queue_usage_stats_tracks_binding_and_overflow():
    #arrange
    pair_counts = [3.0, 7.0, 9.0]

    #act
    stats = _compute_queue_usage_stats(pair_counts, pq_budget=5)

    #assert
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
    #arrange
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

    #act
    summary = _build_queue_usage_summary(rows)

    #assert
    assert summary["pq_budget"] == 5
    assert summary["pair_count_mean"]["mean"] == 5.0
    assert summary["pair_count_max"]["max"] == 9.0
    assert summary["pq_binding_step_fraction"]["median"] == 0.5
    assert summary["pq_binding_ever_rate"] == 0.5
    assert summary["pq_binding_ever_count"] == 1
    assert summary["pq_binding_ever_total"] == 2
