"""Generate reusable evaluation corpora through the installed package API."""

from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from deepgroebner_pytorch_cocoa.corpus_generation import (
    CorpusConfig,
    CorpusGenerationResult,
    generate_corpus,
    main,
)


FIXED_BASIS = "x^2 - y,x*y - 1,y^2 - x"


def _fixed_config(**overrides):
    values = {
        "ideal_distribution": None,
        "coeff_ring": "QQ",
        "symbols": "x,y",
        "basis_elements": FIXED_BASIS,
        "examples": 1,
        "seed": 123,
        "elimination": "none",
        "observation_budget": 8,
        "top_monomials": 1,
    }
    values.update(overrides)
    return CorpusConfig(**values)


def _read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def test_fixed_corpus_api_returns_written_rows_from_another_directory(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    cfg = _fixed_config(output_path="corpora/fixed.jsonl")
    original_config = asdict(cfg)

    result = generate_corpus(cfg)

    assert isinstance(result, CorpusGenerationResult)
    assert result.generated is True
    assert result.corpus_path == str(tmp_path / "corpora" / "fixed.jsonl")
    assert result.corpus_rows == _read_rows(result.corpus_path)
    assert len(result.corpus_rows) == 1
    row = result.corpus_rows[0]
    assert row["ideal_id"] == 0
    assert row["episode_seed"] == 123
    assert row["distribution"] is None
    assert row["coeff_ring"] == "QQ"
    assert row["symbols"] == "x,y"
    assert [entry.replace(" ", "") for entry in row["basis_elements"]] == [
        entry.replace(" ", "") for entry in FIXED_BASIS.split(",")
    ]
    assert asdict(cfg) == original_config
    assert Path.cwd() == tmp_path


def test_geometric_corpus_api_uses_bundled_graphs_from_another_directory(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    result = generate_corpus(
        CorpusConfig(
            ideal_distribution="geom-3",
            examples=1,
            seed=123,
            elimination="none",
            output_path="corpora/triangle.jsonl",
        )
    )

    assert result.generated is True
    assert result.corpus_path == str(tmp_path / "corpora" / "triangle.jsonl")
    assert result.corpus_rows == _read_rows(result.corpus_path)
    assert len(result.corpus_rows) == 1
    row = result.corpus_rows[0]
    assert row["distribution"] == "geom-3"
    assert row["episode_seed"] == 123
    assert row["symbols"] == "x2,x3,y3"
    assert len(row["basis_elements"]) == 3
    assert all(row["basis_elements"])
    assert Path.cwd() == tmp_path


@pytest.mark.parametrize("distribution", [None, "geom-3"])
def test_corpus_cli_and_api_generate_identical_examples(
    distribution, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    cfg = (
        _fixed_config(output_path="api.jsonl")
        if distribution is None
        else CorpusConfig(
            ideal_distribution=distribution,
            examples=1,
            seed=123,
            elimination="none",
            observation_budget=8,
            top_monomials=1,
            output_path="api.jsonl",
        )
    )
    args = [
        f"--ideal-distribution={distribution}",
        "--examples=1",
        "--seed=123",
        "--elimination=none",
        "--observation-budget=8",
        "--top-monomials=1",
        "--output-path=cli.jsonl",
    ]
    if distribution is None:
        args.extend(
            ["--coeff-ring=QQ", "--symbols=x,y", f"--basis-elements={FIXED_BASIS}"]
        )

    api_result = generate_corpus(cfg)
    cli_result = main(args)

    assert isinstance(cli_result, CorpusGenerationResult)
    assert api_result.generated is cli_result.generated is True
    assert cli_result.corpus_path == str(tmp_path / "cli.jsonl")
    assert api_result.corpus_rows == cli_result.corpus_rows
    assert cli_result.corpus_rows == _read_rows(cli_result.corpus_path)


def test_corpus_api_reuses_existing_cache(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    cfg = _fixed_config()

    first = generate_corpus(cfg)
    corpus_path = Path(first.corpus_path)
    original_contents = corpus_path.read_bytes()
    original_mtime = corpus_path.stat().st_mtime_ns
    second = generate_corpus(cfg)

    assert first.generated is True
    assert second.generated is False
    assert first.corpus_path == second.corpus_path
    assert corpus_path.is_absolute()
    assert corpus_path.parent == tmp_path / "eval_corpora" / "greedy_eval"
    assert first.corpus_rows == second.corpus_rows == _read_rows(corpus_path)
    assert corpus_path.read_bytes() == original_contents
    assert corpus_path.stat().st_mtime_ns == original_mtime


def test_corpus_api_requires_explicit_overwrite(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    cfg = _fixed_config(output_path="corpus.jsonl")
    first = generate_corpus(cfg)
    corpus_path = Path(first.corpus_path)
    original_contents = corpus_path.read_bytes()
    replacement_cfg = replace(cfg, seed=124)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        generate_corpus(replacement_cfg)
    assert corpus_path.read_bytes() == original_contents

    replacement = generate_corpus(replace(replacement_cfg, overwrite=True))

    assert replacement.generated is True
    assert replacement.corpus_path == first.corpus_path
    assert replacement.corpus_rows == _read_rows(corpus_path)
    assert replacement.corpus_rows[0]["episode_seed"] == 124
    assert replacement.corpus_rows != first.corpus_rows


@pytest.mark.parametrize("examples", [0, -1])
def test_corpus_api_rejects_nonpositive_example_counts(
    examples, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="positive"):
        generate_corpus(
            _fixed_config(examples=examples, output_path="invalid/corpus.jsonl")
        )

    assert not (tmp_path / "invalid").exists()
    assert not (tmp_path / "eval_corpora").exists()
