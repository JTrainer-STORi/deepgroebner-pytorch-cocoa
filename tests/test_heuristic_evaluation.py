"""Check test corpus reports against actual Buchberger environment trajectories."""

import csv
from dataclasses import asdict
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from deepgroebner_pytorch_cocoa import heuristic_evaluation
from deepgroebner_pytorch_cocoa.corpus_generation import CorpusConfig, generate_corpus
from deepgroebner_pytorch_cocoa.evaluation_resources import RESOURCE_FIELDS
from deepgroebner_pytorch_cocoa.heuristic_evaluation import (
    HEURISTICS,
    HeuristicEvalConfig,
    evaluate_heuristics,
)
from deepgroebner_pytorch_cocoa.wrapped import CNoPQEnv


BASIS = ["x*y-z", "x*z-y", "y*z-x"]


def _write_corpus(path, rows=None):
    if rows is None:
        rows = [
            {"ideal_id": "first", "episode_seed": 100, "distribution": "custom",
             "coeff_ring": "QQ", "symbols": "x,y,z", "basis_elements": BASIS},
            {"coeff_ring": "ZZmod(7)", "symbols": "a,b",
             "basis_elements": ["a^2-b", "a*b-1"]},
        ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


@pytest.mark.parametrize("strategy", [name for name in HEURISTICS if name != "random"])
@pytest.mark.parametrize("elimination", ["none", "lcm", "gebauermoeller"])
@pytest.mark.parametrize("rewards", ["additions", "reductions"])
def test_native_heuristic_stats_match_counts_from_environment_steps(strategy, elimination, rewards):
    #arrange
    native = CNoPQEnv(
        coeff_ring="QQ", symbols="x,y,z", basis=BASIS,
        elimination=elimination, rewards=rewards,
    )
    initial, _ = native.reset()
    before = native.value(strategy="first", gamma=1.0)
    env = gym.make(
        "BuchbergerExponents-v1", coeff_ring="QQ", symbols="x,y,z",
        basis_elements=BASIS, pq_strategy=strategy, elimination=elimination,
        rewards="additions", N_P=1, N_G=1,
    )
    try:
        obs, _ = env.reset(seed=12)
        additions = reductions = nonzero = 0
        truncations = []

        #act
        stats = native.heuristic_stats(strategy=strategy, seed=91)
        while obs["P_mask"].any():
            basis_size = obs["state_features"][0]
            obs, reward, _, truncated, _ = env.step(0)
            truncations.append(truncated)
            additions -= int(reward)
            reductions += 1
            nonzero += int(obs["state_features"][0] > basis_size)
        after = native.value(strategy="first", gamma=1.0)
        following, _ = native.reset()

        #assert
        assert not any(truncations)
        assert stats == {
            "polynomial_additions": additions,
            "reductions": reductions,
            "nonzero_reductions": nonzero,
            "zero_reductions": reductions - nonzero,
            "reward_return": -float(additions if rewards == "additions" else reductions),
        }
        assert before == after
        np.testing.assert_array_equal(initial, following)
        assert env.unwrapped.env.is_groebner_basis()
    finally:
        env.close()


def test_native_random_stats_repeat_with_the_same_seed_and_do_not_advance_state():
    #arrange
    env = CNoPQEnv(coeff_ring="QQ", symbols="x,y,z", basis=BASIS, elimination="none")
    env.reset()

    #act
    first = env.heuristic_stats(strategy="random", seed=123)
    env.heuristic_stats(strategy="random", seed=456)
    repeated = env.heuristic_stats(strategy="random", seed=123)
    still_unfinished = not env.is_groebner_basis()

    #assert
    assert first == repeated
    assert still_unfinished
    assert first["reductions"] == first["zero_reductions"] + first["nonzero_reductions"]
    assert first["reward_return"] == -first["polynomial_additions"]


def test_native_heuristic_stats_require_reset_and_reject_removed_strategies():
    #arrange
    env = CNoPQEnv(coeff_ring="QQ", symbols="x,y,z", basis=BASIS)

    #act
    #assert
    with pytest.raises(RuntimeError, match="Call reset"):
        env.heuristic_stats()

    #arrange
    env.reset()

    #act
    #assert
    with pytest.raises(IndexError):
        env.heuristic_stats(strategy="last")


@pytest.mark.parametrize("rewards", ["additions", "reductions"])
def test_corpus_reports_preserve_each_ring_metadata_and_aggregate_actual_costs(rewards, tmp_path):
    #arrange
    corpus_path = _write_corpus(tmp_path / "corpus.jsonl")
    cfg = HeuristicEvalConfig(
        corpus=str(corpus_path), rewards=rewards, elimination="none", repetitions=2,
        output_dir=str(tmp_path / "reports"), verbosity="quiet",
    )
    before = asdict(cfg)

    #act
    result = evaluate_heuristics(cfg)
    jsonl_rows = [
        json.loads(line) for line in Path(result.artifact_paths["episode_results_jsonl"]).read_text().splitlines()
    ]
    summary = json.loads(Path(result.artifact_paths["summary"]).read_text())
    metadata = json.loads(Path(result.artifact_paths["metadata"]).read_text())
    with Path(result.artifact_paths["episode_results_csv"]).open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))

    #assert
    assert asdict(cfg) == before
    assert len(result.episode_results) == len(csv_rows) == 20
    assert jsonl_rows == result.episode_results
    assert summary == result.summary
    assert summary["num_ideals"] == 2
    assert summary["num_evals"] == 20
    assert set(summary["strategies"]) == set(HEURISTICS)
    for index, ring, symbols, seeds in ((0, "QQ", "x,y,z", {100, 101}), (1, "ZZmod(7)", "a,b", {2, 3})):
        rows = [row for row in jsonl_rows if row["corpus_index"] == index]
        assert {row["coeff_ring"] for row in rows} == {ring}
        assert {row["symbols"] for row in rows} == {symbols}
        assert {row["seed"] for row in rows} == seeds
    assert jsonl_rows[0]["ideal_id"] == "first"
    assert jsonl_rows[0]["episode_seed"] == 100
    assert jsonl_rows[0]["distribution"] == "custom"
    for row in jsonl_rows:
        count = row["polynomial_additions"] if rewards == "additions" else row["reductions"]
        assert row["cost"] == count == -row["reward_return"]
        assert row["reductions"] == row["zero_reductions"] + row["nonzero_reductions"]
        assert row["elapsed_seconds"] >= 0
        assert row["cpu_seconds"] >= 0
        assert set(RESOURCE_FIELDS).issubset(row)
        for name in ("rss_start_bytes", "rss_end_bytes", "process_peak_rss_bytes"):
            assert row[name] is None or row[name] > 0
        assert row["cuda_peak_allocated_bytes"] is None
    assert summary["resources"]["elapsed_seconds"] >= summary["elapsed_seconds_sum"]
    assert summary["resources"]["cpu_seconds"] >= 0
    assert metadata["resource_measurements"]["memory_unit"] == "bytes"
    for strategy in HEURISTICS:
        costs = [row["cost"] for row in jsonl_rows if row["strategy"] == strategy]
        metrics = summary["strategies"][strategy]
        mean = sum(costs) / len(costs)
        assert metrics["num_evals"] == 4
        assert metrics["cost_mean"] == mean
        assert metrics["cost_min"] == min(costs)
        assert metrics["cost_max"] == max(costs)
        assert metrics["cost_std"] == pytest.approx((sum((cost - mean)**2 for cost in costs) / len(costs))**0.5)
        runtimes = [row["elapsed_seconds"] for row in jsonl_rows if row["strategy"] == strategy]
        assert metrics["elapsed_seconds_sum"] == pytest.approx(sum(runtimes))
        assert metrics["elapsed_seconds_mean"] == pytest.approx(sum(runtimes) / len(runtimes))
    assert metadata["corpus"] == str(corpus_path)
    assert metadata["pair_selection"] == "full_pending_pair_set"
    assert metadata["rewards"] == rewards
    assert metadata["reward_discount"] == 1.0


def test_random_corpus_runs_are_reproducible_and_seed_override_is_independent_of_strategy_order(tmp_path):
    #arrange
    path = _write_corpus(tmp_path / "corpus.jsonl")
    configs = [
        HeuristicEvalConfig(
            corpus=str(path), strategies=strategies, seed=50, repetitions=3,
            output_dir=str(tmp_path / f"report-{index}"), verbosity="quiet",
        )
        for index, strategies in enumerate((("random",), ("sugar", "random")))
    ]

    #act
    results = [evaluate_heuristics(cfg) for cfg in configs]
    random_rows = [
        [{key: value for key, value in row.items() if key not in RESOURCE_FIELDS}
         for row in result.episode_results if row["strategy"] == "random"]
        for result in results
    ]

    #assert
    assert random_rows[0] == random_rows[1]
    assert [row["seed"] for row in random_rows[0]] == [50, 51, 52, 53, 54, 55]


@pytest.mark.parametrize("basis", [["x^2"], ["x^2", "y^3"], ["1"]])
def test_already_complete_ideals_have_zero_operation_cost(basis, tmp_path):
    #arrange
    path = _write_corpus(tmp_path / "corpus.jsonl", [
        {"coeff_ring": "QQ", "symbols": "x,y", "basis_elements": basis},
    ])

    #act
    result = evaluate_heuristics(HeuristicEvalConfig(
        corpus=str(path), output_dir=str(tmp_path / "reports"), verbosity="quiet",
    ))

    #assert
    assert len(result.episode_results) == 5
    for row in result.episode_results:
        assert row["cost"] == row["polynomial_additions"] == row["reductions"] == 0
        assert row["reward_return"] == 0


@pytest.mark.parametrize("options", [
    {"strategies": ()}, {"strategies": ("last",)}, {"strategies": ("sugar", "sugar")},
    {"repetitions": 0}, {"repetitions": 1.5}, {"repetitions": True},
    {"seed": -1}, {"seed": 2**31}, {"seed": True},
    {"elimination": "invalid"}, {"rewards": "invalid"},
])
def test_invalid_evaluation_options_fail_before_creating_outputs(options, tmp_path):
    #arrange
    path = _write_corpus(tmp_path / "corpus.jsonl")
    output_dir = tmp_path / "reports"
    cfg = HeuristicEvalConfig(corpus=str(path), output_dir=str(output_dir), **options)

    #act
    #assert
    with pytest.raises(ValueError):
        evaluate_heuristics(cfg)
    assert not output_dir.exists()


@pytest.mark.parametrize("content", [
    "", "\n", "[]\n", "{\n", "{}\n",
    '{"coeff_ring":"QQ","symbols":"x,y","basis_elements":[]}\n',
    '{"coeff_ring":"QQ","symbols":"x,y","basis_elements":[0]}\n',
    '{"coeff_ring":"QQ","symbols":"x,y","basis_elements":["x"],"episode_seed":-1}\n',
])
def test_invalid_corpus_schema_fails_before_creating_outputs(content, tmp_path):
    #arrange
    path = tmp_path / "bad.jsonl"
    path.write_text(content)
    output_dir = tmp_path / "reports"

    #act
    #assert
    with pytest.raises(ValueError, match="bad.jsonl"):
        evaluate_heuristics(HeuristicEvalConfig(corpus=str(path), output_dir=str(output_dir)))
    assert not output_dir.exists()


@pytest.mark.parametrize("polynomial", ["z", "0", "x-x", "x^^2"])
def test_invalid_native_input_reports_its_corpus_index_without_writing_results(polynomial, tmp_path):
    #arrange
    path = _write_corpus(tmp_path / "bad.jsonl", [
        {"coeff_ring": "QQ", "symbols": "x,y", "basis_elements": [polynomial]},
    ])
    output_dir = tmp_path / "reports"

    #act
    #assert
    with pytest.raises(ValueError, match="ideal 0"):
        evaluate_heuristics(HeuristicEvalConfig(corpus=str(path), output_dir=str(output_dir)))
    assert not output_dir.exists()


def test_existing_reports_are_protected_until_overwrite_is_requested(tmp_path):
    #arrange
    path = _write_corpus(tmp_path / "corpus.jsonl")
    output_dir = tmp_path / "reports"
    output_dir.mkdir()
    report = output_dir / "summary.json"
    report.write_text("existing report")
    cfg = HeuristicEvalConfig(
        corpus=str(path), strategies=("degree",), output_dir=str(output_dir), verbosity="quiet",
    )

    #act
    #assert
    with pytest.raises(FileExistsError, match="overwrite"):
        evaluate_heuristics(cfg)
    assert report.read_text() == "existing report"

    #arrange
    cfg.overwrite = True

    #act
    result = evaluate_heuristics(cfg)

    #assert
    assert json.loads(report.read_text()) == result.summary


def test_cli_and_python_api_evaluate_a_generated_corpus_from_another_directory(monkeypatch, tmp_path):
    #arrange
    monkeypatch.chdir(tmp_path)
    generated = generate_corpus(CorpusConfig(
        ideal_distribution="2-4-3-weighted", examples=2, seed=123, elimination="none",
        output_path="inputs/ideals.jsonl",
    ))
    cfg = HeuristicEvalConfig(
        corpus=generated.corpus_path, strategies=("degree", "random"), elimination="none",
        sort_input=True, sort_reducers=False, repetitions=2,
        output_dir="api-reports", verbosity="quiet",
    )

    #act
    api_result = evaluate_heuristics(cfg)
    cli_result = heuristic_evaluation.main([
        "--corpus", "inputs/ideals.jsonl", "--strategies", "degree", "random",
        "--elimination", "none", "--sort-input", "True", "--sort-reducers", "False",
        "--repetitions", "2", "--output-dir", "cli-reports", "--verbosity", "quiet",
    ])
    expected = [{k: v for k, v in row.items() if k not in RESOURCE_FIELDS} for row in api_result.episode_results]
    actual = [{k: v for k, v in row.items() if k not in RESOURCE_FIELDS} for row in cli_result.episode_results]

    #assert
    assert actual == expected
    assert len(actual) == 8
    assert Path(cli_result.output_directory) == tmp_path / "cli-reports"
    assert all(Path(path).is_file() for path in cli_result.artifact_paths.values())


@pytest.mark.parametrize("verbosity", ["normal", "quiet"])
def test_resource_statistics_are_printed_unless_quiet(tmp_path, capsys, verbosity):
    path = _write_corpus(tmp_path / "corpus.jsonl")
    evaluate_heuristics(HeuristicEvalConfig(
        corpus=str(path), strategies=("degree",), elimination="none",
        output_dir=str(tmp_path / "reports"), verbosity=verbosity,
    ))

    output = capsys.readouterr().out
    if verbosity == "quiet":
        assert output == ""
    else:
        assert "Total evaluation:" in output
        assert "wall" in output and "CPU" in output and "peak RSS" in output
