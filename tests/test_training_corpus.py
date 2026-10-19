"""Train on fixed JSONL ideals with independent rings and shared feature widths."""

from dataclasses import asdict
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from deepgroebner_pytorch_cocoa.config_dataclasses import Config
from deepgroebner_pytorch_cocoa.greedy_eval import (
    build_greedy_eval_seeds,
    resolve_greedy_eval_corpus,
)
from deepgroebner_pytorch_cocoa.ideal_corpus import load_ideal_corpus
from deepgroebner_pytorch_cocoa.ppo_training import train
from deepgroebner_pytorch_cocoa.training_env import create_training_envs
from deepgroebner_pytorch_cocoa.training_utils import resolve_training_config


ENVIRONMENTS = ["BuchbergerExponents-v1", "BuchbergerExponentsNoPQ-v1"]


def _rows():
    return [
        {"coeff_ring": "QQ", "symbols": "x,y", "basis_elements": basis}
        for basis in (
            ["x^2-y", "x*y-1"],
            ["x^2-y^2", "x*y-1"],
            ["x^3-y", "x*y-1"],
        )
    ]


def _mixed_rows():
    # The widest ring comes last, and the union of variable names is wider still.
    return [
        {"coeff_ring": "QQ", "symbols": "x,y", "basis_elements": ["x^2-y", "x*y-1"]},
        {"coeff_ring": "ZZmod(7)", "symbols": "y,x", "basis_elements": ["x^2-2*y", "x*y-1"]},
        {"coeff_ring": "QQ", "symbols": "t", "basis_elements": ["t^3-t", "t^2-1"]},
        {"coeff_ring": "ZZmod(7)", "symbols": "u,v,w", "basis_elements": ["u^2-v", "u*v-w", "v*w-1"]},
    ]


def _write(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def _config(tmp_path, gym_id=ENVIRONMENTS[0], rows=None):
    cfg = Config()
    cfg.env.gym_id = gym_id
    cfg.env.ideal_corpus = str(_write(tmp_path / "training.jsonl", _rows() if rows is None else rows))
    cfg.env.elimination = "none"
    cfg.env.N = 8
    cfg.training.num_envs = 1
    cfg.training.num_steps_per_update = 4
    cfg.training.total_timesteps = 4
    cfg.training.num_epochs_per_update = 1
    cfg.training.num_minibatches = 2
    cfg.training.cuda = False
    cfg.training.track = False
    cfg.training.pytorch_num_threads = 1
    cfg.training.console_verbosity = "quiet"
    cfg.training.system_metrics_enabled = False
    cfg.training.output_directory = str(tmp_path / "output")
    return cfg


@pytest.mark.parametrize(
    "field,value",
    [("coeff_ring", "ZZmod(7)"), ("symbols", "y,x"), ("symbols", "x,y,z")],
)
def test_corpus_accepts_mixed_coefficient_rings_and_ordered_variables(tmp_path, field, value):
    #arrange
    rows = _rows()
    rows[1][field] = value
    path = _write(tmp_path / "mixed.jsonl", rows)

    #act
    corpus = load_ideal_corpus(path)

    #assert
    assert corpus.rows() == rows
    assert corpus.coeff_rings == tuple(row["coeff_ring"] for row in rows)
    assert corpus.ring_symbols == tuple(row["symbols"] for row in rows)
    assert corpus.n_vars == max(len(row["symbols"].split(",")) for row in rows)


def test_corpus_width_uses_largest_ring_and_preserves_each_rows_names(tmp_path):
    #arrange
    rows = _mixed_rows()
    path = _write(tmp_path / "mixed.jsonl", rows)

    #act
    corpus = load_ideal_corpus(path)

    #assert
    assert corpus.n_vars == 3
    assert corpus.coeff_ring == rows[-1]["coeff_ring"]
    assert corpus.symbols == rows[-1]["symbols"]
    assert corpus.rows() == rows
    assert corpus.bases == tuple(tuple(row["basis_elements"]) for row in rows)


@pytest.mark.parametrize(
    "row",
    [
        [],
        {},
        {"coeff_ring": "QQ", "symbols": "x,y", "basis_elements": []},
        {"coeff_ring": "QQ", "symbols": "x,y", "basis_elements": "x^2-y,x*y-1"},
        {"coeff_ring": "QQ", "symbols": "x,y", "basis_elements": [""]},
        {"coeff_ring": "QQ", "symbols": "x,y", "basis_elements": [1]},
        {"coeff_ring": "QQ", "symbols": "x,x", "basis_elements": ["x"]},
        {"coeff_ring": "QQ", "symbols": "x,", "basis_elements": ["x"]},
        {"coeff_ring": "invalid", "symbols": "x,y", "basis_elements": ["x"]},
        {"coeff_ring": "QQ", "symbols": "x[1,2", "basis_elements": ["x"]},
    ],
)
def test_invalid_corpus_rows_report_the_source_line(tmp_path, row):
    #arrange
    path = _write(tmp_path / "invalid.jsonl", [_rows()[0], row])

    #act
    #assert
    with pytest.raises(ValueError, match="line 2"):
        load_ideal_corpus(path)


@pytest.mark.parametrize("text,match", [("\n", "at least one ideal"), ("\n{bad}\n", "line 2")])
def test_empty_and_malformed_corpora_fail_clearly(tmp_path, text, match):
    #arrange
    path = tmp_path / "invalid.jsonl"
    path.write_text(text)

    #act
    #assert
    with pytest.raises(ValueError, match=match):
        load_ideal_corpus(path)


def test_equivalent_ring_formatting_and_evaluation_metadata_are_accepted(tmp_path):
    #arrange
    rows = _rows()
    for row in rows:
        row["coeff_ring"] = " ZZmod( 007 ) "
        row["symbols"] = " x, y "
        row["episode_seed"] = 123
    rows[1]["coeff_ring"] = "ZZmod(7)"
    path = _write(tmp_path / "formatted.jsonl", rows)

    #act
    corpus = load_ideal_corpus(path)

    #assert
    assert corpus.coeff_ring == "ZZmod(7)"
    assert corpus.symbols == "x,y"
    assert len(corpus.bases) == 3
    assert corpus.bases[0] == tuple(rows[0]["basis_elements"])
    assert corpus.coeff_rings == ("ZZmod(7)",) * 3
    assert corpus.ring_symbols == ("x,y",) * 3
    assert all(row["coeff_ring"] == "ZZmod(7)" and row["symbols"] == "x,y" for row in corpus.rows())


@pytest.mark.parametrize(
    "overrides",
    [
        {"basis_elements": "x^2-y,x*y-1"},
        {"fix_ideal": True},
        {"coeff_ring": "ZZmod(7)"},
        {"symbols": "y,x"},
    ],
)
def test_conflicting_input_settings_are_rejected(tmp_path, overrides):
    #arrange
    cfg = _config(tmp_path)
    for name, value in overrides.items():
        setattr(cfg.env, name, value)

    #act
    #assert
    with pytest.raises(ValueError, match="ideal_corpus|ideal corpus"):
        resolve_training_config(cfg)


def test_invalid_corpus_fails_before_creating_training_outputs(tmp_path):
    #arrange
    cfg = _config(tmp_path)
    rows = _rows()
    rows[-1]["coeff_ring"] = "invalid"
    _write(Path(cfg.env.ideal_corpus), rows)

    #act
    #assert
    with pytest.raises(ValueError, match="line 3.*coeff_ring"):
        train(cfg)
    assert not Path(cfg.training.output_directory).exists()


@pytest.mark.parametrize("gym_id", ENVIRONMENTS)
def test_mixed_ring_observations_pad_each_monomial_and_match_standalone_ideals(tmp_path, gym_id):
    #arrange
    rows = _mixed_rows()
    path = _write(tmp_path / "mixed.jsonl", rows)
    env = gym.make(gym_id, ideal_corpus=str(path), elimination="none", k=2)
    try:
        seen = set()
        for seed in range(80):
            obs, info = env.reset(seed=seed)
            index = info["ideal_corpus_index"]
            if index in seen:
                continue
            seen.add(index)
            row = rows[index]
            n_vars = len(row["symbols"].split(","))
            reference = gym.make(gym_id, elimination="none", k=2, **row)
            try:
                expected, _ = reference.reset(seed=seed)
                for _ in range(300):
                    # Every monomial gets its own three-coordinate slot, including
                    # both endpoints of an S-pair. Padding the flattened row fails.
                    for name, monomials in (("G", 2), ("P", 4)):
                        source = expected[name].reshape(expected[name].shape[0], monomials, n_vars)
                        padded = np.pad(source, ((0, 0), (0, 0), (0, 3 - n_vars)))
                        np.testing.assert_array_equal(
                            obs[name], padded.reshape(source.shape[0], monomials * 3)
                        )
                    for name in ("G_mask", "P_mask", "state_features"):
                        np.testing.assert_array_equal(obs[name], expected[name])
                    assert env.unwrapped.n_vars == 3
                    assert info["ideal_corpus_index"] == index
                    if not obs["P_mask"].any():
                        break

                    #act
                    action = int(np.flatnonzero(obs["P_mask"])[0])
                    obs, reward, terminated, truncated, info = env.step(action)
                    expected, expected_reward, expected_done, expected_truncated, _ = reference.step(action)

                    #assert
                    assert (reward, terminated, truncated) == (
                        expected_reward, expected_done, expected_truncated
                    )
                assert terminated and not truncated
                assert env.unwrapped.env.is_groebner_basis()
                assert reference.unwrapped.env.is_groebner_basis()
            finally:
                reference.close()
            if len(seen) == len(rows):
                break
        assert seen == set(range(len(rows)))
    finally:
        env.close()


@pytest.mark.parametrize("gym_id", ENVIRONMENTS)
@pytest.mark.parametrize("elimination", ["none", "lcm", "gebauermoeller"])
@pytest.mark.parametrize("coeff_ring", ["QQ", "ZZmod(7)"])
def test_corpus_resets_sample_exact_inputs_reproducibly_and_finish_correctly(
    tmp_path, gym_id, elimination, coeff_ring
):
    #arrange
    rows = _rows()
    for row in rows:
        row["coeff_ring"] = coeff_ring
    path = _write(tmp_path / "ideals.jsonl", rows)
    env = gym.make(gym_id, ideal_corpus=str(path), elimination=elimination, k=1)
    try:
        #act
        first, first_info = env.reset(seed=123)
        indices = [first_info["ideal_corpus_index"]]
        for _ in range(20):
            obs, info = env.reset()
            indices.append(info["ideal_corpus_index"])
        repeated, repeated_info = env.reset(seed=123)
        reference = gym.make(gym_id, elimination=elimination, k=1, **rows[indices[0]])
        try:
            expected, _ = reference.reset(seed=123)
        finally:
            reference.close()

        #assert
        assert set(indices) == {0, 1, 2}
        assert repeated_info["ideal_corpus_index"] == indices[0]
        for name in first:
            np.testing.assert_array_equal(first[name], repeated[name])
            np.testing.assert_array_equal(first[name], expected[name])

        #act
        obs = repeated
        for _ in range(300):
            action = int(np.flatnonzero(obs["P_mask"])[0])
            obs, _, terminated, truncated, info = env.step(action)
            if terminated:
                break

        #assert
        assert terminated and not truncated
        assert not obs["P_mask"].any()
        assert info["ideal_corpus_index"] == indices[0]
        assert env.unwrapped.env.is_groebner_basis()
    finally:
        env.close()


@pytest.mark.parametrize("gym_id", ENVIRONMENTS)
def test_parsed_corpus_and_sampling_rng_survive_native_copy_and_file_removal(tmp_path, gym_id):
    #arrange
    path = _write(tmp_path / "ideals.jsonl", _mixed_rows())
    env = gym.make(gym_id, ideal_corpus=str(path), elimination="none")
    try:
        env.reset(seed=123)
        native = env.unwrapped.env
        copy = native.copy()
        path.unlink()

        #act
        original_indices = []
        copied_indices = []
        for _ in range(8):
            native.reset()
            copy.reset()
            original_indices.append(native.corpus_index())
            copied_indices.append(copy.corpus_index())

        #assert
        assert original_indices == copied_indices
        assert len(set(original_indices)) > 1
    finally:
        env.close()


@pytest.mark.parametrize("gym_id", ENVIRONMENTS)
@pytest.mark.parametrize("vectorization", ["sync", "async"])
def test_parallel_autoresets_preserve_corpus_indices_and_seed_reproducibility(
    tmp_path, gym_id, vectorization
):
    #arrange
    cfg = _config(tmp_path, gym_id, _mixed_rows())
    cfg.training.num_envs = 2
    cfg.training.vectorization_mode = vectorization
    envs = create_training_envs(resolve_training_config(cfg))
    try:
        traces = []

        #act
        for _ in range(2):
            obs, info = envs.reset(seed=[123, 456])
            trace = [info["ideal_corpus_index"].copy()]
            for _ in range(12):
                actions = np.array([np.flatnonzero(mask)[0] for mask in obs["P_mask"]])
                previous_indices = info["ideal_corpus_index"].copy()
                obs, _, terminated, truncated, info = envs.step(actions)
                finished = terminated | truncated

                #assert
                assert all(mask.any() for mask in obs["P_mask"])
                if finished.any():
                    np.testing.assert_array_equal(
                        info["final_info"]["ideal_corpus_index"][finished], previous_indices[finished]
                    )

                #act
                trace.append(info["ideal_corpus_index"].copy())
            traces.append(np.stack(trace))

        #assert
        np.testing.assert_array_equal(traces[0], traces[1])
        assert np.unique(traces[0]).size > 1
        assert np.all((traces[0] >= 0) & (traces[0] < len(_mixed_rows())))
        assert not np.array_equal(traces[0][:, 0], traces[0][:, 1])
    finally:
        envs.close()


@pytest.mark.parametrize("gym_id", ENVIRONMENTS)
@pytest.mark.parametrize("basis", [["x"], ["x-x", "x*y-1"], ["z", "x*y-1"]])
def test_unusable_ideals_fail_during_construction_with_their_corpus_index(
    tmp_path, gym_id, basis
):
    #arrange
    rows = _rows()
    rows[1]["basis_elements"] = basis
    path = _write(tmp_path / "invalid.jsonl", rows)

    #act
    #assert
    with pytest.raises(ValueError, match="Corpus ideal 1"):
        gym.make(gym_id, ideal_corpus=str(path))


def test_greedy_evaluation_uses_a_fixed_corpus_subset_and_invalidates_edited_cache(tmp_path):
    #arrange
    rows = _mixed_rows()
    cfg = resolve_training_config(_config(tmp_path, rows=rows))
    cfg.training.greedy_eval_num_examples = 2
    seeds = build_greedy_eval_seeds(cfg, True)

    #act
    first, first_path, generated = resolve_greedy_eval_corpus(cfg, seeds, cache_root=tmp_path / "cache")
    repeated, repeated_path, regenerated = resolve_greedy_eval_corpus(
        cfg, seeds, cache_root=tmp_path / "cache"
    )

    #assert
    assert generated and not regenerated
    assert first == repeated
    assert first_path == repeated_path
    assert len({row["ideal_corpus_index"] for row in first}) == 2
    assert [row["episode_seed"] for row in first] == seeds
    for row in first:
        original = rows[row["ideal_corpus_index"]]
        assert all(row[name] == original[name] for name in ("coeff_ring", "symbols", "basis_elements"))

    #arrange
    rows[0]["basis_elements"] = ["x^4-y", "x*y-1"]
    _write(Path(cfg.env.ideal_corpus), rows)

    #act
    _, changed_path, generated = resolve_greedy_eval_corpus(cfg, seeds, cache_root=tmp_path / "cache")

    #assert
    assert generated
    assert changed_path != first_path


@pytest.mark.parametrize("gym_id", ENVIRONMENTS)
@pytest.mark.parametrize("num_envs,vectorization", [(1, "sync"), (2, "sync"), (2, "async")])
def test_training_from_a_corpus_includes_greedy_evaluation_and_config_snapshots(
    tmp_path, monkeypatch, gym_id, num_envs, vectorization
):
    #arrange
    monkeypatch.chdir(tmp_path)
    rows = _mixed_rows()
    cfg = _config(tmp_path, gym_id, rows)
    cfg.env.k = 2
    cfg.training.num_envs = num_envs
    cfg.training.total_timesteps = cfg.training.num_steps_per_update * num_envs
    cfg.training.vectorization_mode = vectorization
    cfg.training.greedy_eval_first_step = 0
    cfg.training.greedy_eval_interval_steps = 4
    cfg.training.greedy_eval_num_examples = 1
    cfg.training.greedy_eval_seed = 1
    before = asdict(cfg)

    #act
    result = train(cfg)
    actor = Path(result.checkpoint_paths["actor"])
    snapshot = json.loads(actor.with_name(f"{actor.stem}__config.json").read_text())

    #assert
    assert asdict(cfg) == before
    assert result.summary["global_step"] == 4 * num_envs
    assert result.summary["greedy_eval_num_examples"] == 1
    assert result.summary["greedy_eval_last_completion_rate"] == 1.0
    assert "corpus-training" in result.run_directory
    assert snapshot["env"]["ideal_corpus"] == cfg.env.ideal_corpus
    assert snapshot["env"]["coeff_ring"] == rows[-1]["coeff_ring"]
    assert snapshot["env"]["symbols"] == rows[-1]["symbols"]
    assert snapshot["env"]["ideal_dist"] is None
    assert snapshot["actor"]["state_embedding_head_in_dim"] == 6
    assert snapshot["actor"]["pair_embedding_head_in_dim"] == 12
    assert snapshot["critic"]["state_embedding_head_in_dim"] == 6
    eval_rows = [json.loads(line) for line in Path(result.summary["greedy_eval_corpus_path"]).read_text().splitlines()]
    assert len(eval_rows) == 1
    selected = rows[eval_rows[0]["ideal_corpus_index"]]
    assert len(selected["symbols"].split(",")) < 3
    assert all(eval_rows[0][name] == selected[name] for name in ("coeff_ring", "symbols", "basis_elements"))
