"""Replay user-selected greedy evaluation corpora without sampling or rewriting."""

from dataclasses import asdict
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from deepgroebner_pytorch_cocoa import greedy_eval
from deepgroebner_pytorch_cocoa.config_dataclasses import Config
from deepgroebner_pytorch_cocoa.corpus_generation import CorpusConfig, generate_corpus
from deepgroebner_pytorch_cocoa.ppo_evaluation import TrainingEvaluation
from deepgroebner_pytorch_cocoa.ppo_training import PPOTrainer
from deepgroebner_pytorch_cocoa.training_env import build_fixed_ideal_env_kwargs_for_gym_id
from deepgroebner_pytorch_cocoa.training_utils import resolve_training_config


ENVIRONMENTS = ["BuchbergerExponents-v1", "BuchbergerExponentsNoPQ-v1"]


def _rows():
    return [
        {
            "ideal_id": "first",
            "episode_seed": 40,
            "coeff_ring": "QQ",
            "symbols": "t",
            "basis_elements": ["t^3-t", "t^2-1"],
        },
        {
            "ideal_id": "second",
            "coeff_ring": "ZZmod(7)",
            "symbols": "y,x",
            "basis_elements": ["x^2-2*y", "x*y-1"],
        },
        {
            "ideal_id": "third",
            "episode_seed": None,
            "coeff_ring": "QQ",
            "symbols": "u,v,w",
            "basis_elements": ["u^2-v", "u*v-w", "v*w-1"],
        },
    ]


def _write(path, rows):
    path.write_text("\n\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _config(tmp_path, gym_id=ENVIRONMENTS[0]):
    cfg = Config()
    cfg.env.gym_id = gym_id
    cfg.env.ideal_dist = "3-3-2-weighted"
    cfg.env.env_seed = 12
    cfg.env.elimination = "none"
    cfg.env.N = 8
    cfg.env.k = 2
    cfg.training.num_envs = 1
    cfg.training.num_steps_per_update = 2
    cfg.training.total_timesteps = 2
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
    cfg.training.greedy_eval_interval_steps = 2
    cfg.training.greedy_eval_num_examples = 1
    cfg.training.greedy_eval_seed = 10000
    cfg.training.greedy_eval_corpus = str(_write(tmp_path / "evaluation.jsonl", _rows()))
    return cfg


def _resolve(cfg, cache_root):
    cfg = resolve_training_config(cfg)
    seeds = greedy_eval.build_greedy_eval_seeds(cfg, True)
    return seeds, greedy_eval.resolve_greedy_eval_corpus(cfg, seeds, cache_root=cache_root)


def _forbid_sampling(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("An explicit evaluation corpus must bypass automatic corpus sampling")

    monkeypatch.setattr(greedy_eval, "build_greedy_eval_corpus", unexpected)
    monkeypatch.setattr(greedy_eval, "collect_unique_ideal_eval_seeds", unexpected)


@pytest.mark.parametrize("num_examples", [1, 20])
def test_explicit_corpus_replays_all_rows_in_file_order_without_a_cache(
    tmp_path, monkeypatch, num_examples
):
    cfg = _config(tmp_path)
    # A finite distribution would otherwise sample native environments to
    # discover its distinct ideals before materializing the evaluation cache.
    cfg.env.ideal_dist = "cyclic-3"
    cfg.training.greedy_eval_num_examples = num_examples
    source = Path(cfg.training.greedy_eval_corpus)
    original = source.read_bytes()
    _forbid_sampling(monkeypatch)

    seeds, (rows, path, generated) = _resolve(cfg, tmp_path / "automatic-cache")

    assert seeds == [40, 10001, 10002]
    assert [row["ideal_id"] for row in rows] == ["first", "second", "third"]
    assert [row["episode_seed"] for row in rows] == seeds
    for actual, expected in zip(rows, _rows()):
        for name in ("coeff_ring", "symbols", "basis_elements"):
            assert actual[name] == expected[name]
    assert path == str(source.resolve())
    assert generated is False
    assert not (tmp_path / "automatic-cache").exists()
    assert source.read_bytes() == original


@pytest.mark.parametrize("gym_id", ENVIRONMENTS)
def test_explicit_corpus_preserves_rings_and_pads_each_monomial(tmp_path, gym_id):
    cfg = resolve_training_config(_config(tmp_path, gym_id))
    seeds = greedy_eval.build_greedy_eval_seeds(cfg, True)
    rows, _, _ = greedy_eval.resolve_greedy_eval_corpus(cfg, seeds)
    row = rows[1]
    kwargs = build_fixed_ideal_env_kwargs_for_gym_id(cfg, gym_id, row)
    padded = gym.make(gym_id, **kwargs)
    standalone_kwargs = {name: value for name, value in kwargs.items() if name != "observation_nvars"}
    standalone = gym.make(gym_id, **standalone_kwargs)
    try:
        actual, _ = padded.reset(seed=row["episode_seed"])
        expected, _ = standalone.reset(seed=row["episode_seed"])
        for _ in range(100):
            for name, monomials in (("G", 2), ("P", 4)):
                native = expected[name].reshape(expected[name].shape[0], monomials, 2)
                wanted = np.pad(native, ((0, 0), (0, 0), (0, 1)))
                np.testing.assert_array_equal(actual[name], wanted.reshape(native.shape[0], monomials * 3))
            for name in ("G_mask", "P_mask", "state_features"):
                np.testing.assert_array_equal(actual[name], expected[name])
            if not actual["P_mask"].any():
                break
            action = int(np.flatnonzero(actual["P_mask"])[0])
            actual, reward, done, truncated, _ = padded.step(action)
            expected, expected_reward, expected_done, expected_truncated, _ = standalone.step(action)
            assert (reward, done, truncated) == (expected_reward, expected_done, expected_truncated)
        assert padded.unwrapped.env.is_groebner_basis()
        assert standalone.unwrapped.env.is_groebner_basis()
    finally:
        padded.close()
        standalone.close()


@pytest.mark.parametrize(
    "gym_id,train_target,nopq_target",
    [
        (ENVIRONMENTS[0], True, False),
        (ENVIRONMENTS[0], True, True),
        (ENVIRONMENTS[0], False, True),
        (ENVIRONMENTS[1], True, True),
    ],
    ids=["pq", "pq-and-nopq", "nopq-only", "nopq-training"],
)
def test_short_training_uses_explicit_corpus_and_reports_its_source(
    tmp_path, monkeypatch, gym_id, train_target, nopq_target
):
    monkeypatch.chdir(tmp_path)
    cfg = _config(tmp_path, gym_id)
    cfg.training.greedy_eval_corpus = "evaluation.jsonl"
    cfg.training.greedy_eval_on_train_env = train_target
    cfg.training.greedy_eval_on_nopq_env = nopq_target
    before = asdict(cfg)
    original = (tmp_path / "evaluation.jsonl").read_bytes()
    _forbid_sampling(monkeypatch)

    trainer = PPOTrainer(cfg)
    result = trainer.run()

    assert asdict(cfg) == before
    assert result.summary["global_step"] == 2
    assert result.summary["policy_updates"] == 1
    assert result.summary["greedy_eval_first_step"] == 0
    assert result.summary["greedy_eval_corpus_path"] == str(tmp_path / "evaluation.jsonl")
    assert result.summary["greedy_eval_num_examples"] == 3
    assert result.summary["greedy_eval_last_num_completed"] == 3
    assert trainer.evaluation.corpus_generated is False
    for stats in (trainer.evaluation.last_train, trainer.evaluation.last_nopq):
        if stats is not None:
            assert stats["num_examples"] == 3
            assert stats["num_completed"] == 3
            assert [row["ideal_id"] for row in stats["episode_results"]] == ["first", "second", "third"]
            assert [row["episode_seed"] for row in stats["episode_results"]] == [40, 10001, 10002]
    assert not (tmp_path / "eval_corpora").exists()
    assert (tmp_path / "evaluation.jsonl").read_bytes() == original


@pytest.mark.parametrize(
    "contents,match",
    [
        ("\n", "at least one|empty"),
        ("\n{broken}\n", "line 2"),
        ("[]\n", "line 1"),
        ("{}\n", "line 1"),
        (json.dumps({**_rows()[0], "basis_elements": []}), "line 1.*basis_elements"),
        (json.dumps({**_rows()[0], "coeff_ring": "invalid"}), "line 1.*coeff_ring"),
        (json.dumps({**_rows()[0], "symbols": "t,t"}), "line 1.*symbols"),
        (json.dumps({**_rows()[0], "symbols": "t,u,v,w"}), "variables|width|policy"),
    ],
    ids=["empty", "malformed", "non-object", "missing-fields", "empty-basis", "bad-ring", "bad-symbols", "too-wide"],
)
def test_invalid_explicit_corpora_fail_before_training_creates_outputs(tmp_path, contents, match):
    cfg = _config(tmp_path)
    Path(cfg.training.greedy_eval_corpus).write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        PPOTrainer(cfg).run()

    assert not Path(cfg.training.output_directory).exists()


@pytest.mark.parametrize("seed", [-1, 2**31, True, 2.5, "123", [], {}])
def test_invalid_episode_seeds_report_the_file_line(tmp_path, seed):
    cfg = _config(tmp_path)
    _write(Path(cfg.training.greedy_eval_corpus), [{**_rows()[0], "episode_seed": seed}])

    with pytest.raises(ValueError, match="line 1.*episode_seed"):
        _resolve(cfg, tmp_path / "automatic-cache")


def test_missing_explicit_corpus_fails_before_training_creates_outputs(tmp_path):
    cfg = _config(tmp_path)
    cfg.training.greedy_eval_corpus = str(tmp_path / "missing.jsonl")

    with pytest.raises(FileNotFoundError, match="missing.jsonl"):
        PPOTrainer(cfg).run()

    assert not Path(cfg.training.output_directory).exists()


def test_repeated_episode_seeds_preserve_distinct_ideals(tmp_path):
    cfg = _config(tmp_path)
    rows = [{**row, "episode_seed": 40} for row in _rows()]
    _write(Path(cfg.training.greedy_eval_corpus), rows)

    seeds, (loaded, _, generated) = _resolve(cfg, tmp_path / "automatic-cache")

    assert seeds == [40, 40, 40]
    assert loaded == rows
    assert generated is False


def test_repeated_seeds_evaluate_all_distinct_ideals_on_both_targets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _config(tmp_path)
    cfg.training.total_timesteps = 0
    cfg.training.greedy_eval_on_nopq_env = True
    _write(Path(cfg.training.greedy_eval_corpus), [{**row, "episode_seed": 40} for row in _rows()])
    trainer = PPOTrainer(cfg)
    trainer.run()

    for stats in (trainer.evaluation.last_train, trainer.evaluation.last_nopq):
        assert stats["num_completed"] == 3
        assert [row["ideal_id"] for row in stats["episode_results"]] == ["first", "second", "third"]
        assert [row["episode_seed"] for row in stats["episode_results"]] == [40, 40, 40]


def test_generated_corpus_replays_the_same_sampled_fixed_training_ideal(tmp_path):
    cfg = _config(tmp_path)
    cfg.env.ideal_dist = "2-3-2-weighted"
    cfg.env.fix_ideal = True
    generated = generate_corpus(
        CorpusConfig(
            ideal_distribution=cfg.env.ideal_dist,
            examples=1,
            seed=cfg.env.env_seed,
            elimination="none",
            output_path=cfg.training.greedy_eval_corpus,
            overwrite=True,
        )
    )
    original = Path(generated.corpus_path).read_bytes()

    _, (loaded, path, was_generated) = _resolve(cfg, tmp_path / "automatic-cache")

    assert loaded == generated.corpus_rows
    assert loaded[0]["coeff_ring"] == "ZZmod(32003)"
    assert loaded[0]["episode_seed"] == cfg.env.env_seed
    assert path == generated.corpus_path
    assert was_generated is False
    assert Path(path).read_bytes() == original


@pytest.mark.parametrize("source", ["explicit", "sampled"])
def test_fixed_training_accepts_only_its_single_ideal(tmp_path, source):
    cfg = _config(tmp_path)
    cfg.training.greedy_eval_corpus = None
    if source == "explicit":
        cfg.env.coeff_ring = "QQ"
        cfg.env.symbols = "x,y"
        cfg.env.basis_elements = "x^2-y,x*y-1"
    else:
        cfg.env.fix_ideal = True
        cfg.env.ideal_dist = "(2-3)-3-2-weighted"
    resolved = resolve_training_config(cfg)
    seeds = greedy_eval.build_greedy_eval_seeds(resolved, True)
    fixed_row = greedy_eval.build_greedy_eval_corpus(resolved, seeds)[0]
    source_path = tmp_path / "fixed-evaluation.jsonl"
    cfg.training.greedy_eval_corpus = str(_write(source_path, [fixed_row]))

    _, (loaded, _, generated) = _resolve(cfg, tmp_path / "automatic-cache")

    assert loaded[0]["basis_elements"] == fixed_row["basis_elements"]
    assert generated is False

    _write(source_path, [fixed_row, fixed_row])
    with pytest.raises(ValueError, match="single|one|fixed"):
        _resolve(cfg, tmp_path / "automatic-cache")

    _write(source_path, [{**fixed_row, "basis_elements": ["1"]}])
    with pytest.raises(ValueError, match="same|match|fixed"):
        _resolve(cfg, tmp_path / "automatic-cache")


@pytest.mark.parametrize("disable", ["interval", "examples", "targets"])
def test_disabled_evaluation_ignores_missing_explicit_corpus(tmp_path, disable):
    cfg = _config(tmp_path)
    cfg.training.greedy_eval_corpus = str(tmp_path / "missing.jsonl")
    if disable == "interval":
        cfg.training.greedy_eval_interval_steps = 0
    elif disable == "examples":
        cfg.training.greedy_eval_num_examples = 0
    else:
        cfg.training.greedy_eval_on_train_env = False
        cfg.training.greedy_eval_on_nopq_env = False

    resolved = resolve_training_config(cfg)
    evaluation = TrainingEvaluation(resolved)

    assert evaluation.enabled is False
    assert evaluation.corpus_rows == []
    assert evaluation.corpus_path is None
    assert greedy_eval.build_greedy_eval_seeds(resolved, False) == []
