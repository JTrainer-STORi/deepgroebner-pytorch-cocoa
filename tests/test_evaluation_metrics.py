"""Greedy evaluation metrics retain target identity and checkpoint semantics."""

from collections import Counter
from types import SimpleNamespace

import pytest

from deepgroebner_pytorch_cocoa import ppo_evaluation
from deepgroebner_pytorch_cocoa.config_dataclasses import Config
from deepgroebner_pytorch_cocoa.training_summary import build_training_summary


def _stats(mean_return, elapsed_seconds):
    return {
        "mean_return": mean_return,
        "mean_return_completed_only": mean_return + 1,
        "median_return": mean_return - 1,
        "min_return": mean_return - 5,
        "max_return": mean_return + 2,
        "mean_length": 6.5,
        "max_length": 9,
        "num_examples": 2,
        "num_completed": 2,
        "completion_rate": 1.0,
        "num_truncated_total": 0,
        "num_benchmark_truncated": 0,
        "num_timed_out": 0,
        "num_worker_exited": 0,
        "elapsed_seconds": elapsed_seconds,
        "episode_results": [{"episode_seed": 10000}, {"episode_seed": 10001}],
    }


def _evaluation(monkeypatch, *, train, nopq):
    cfg = Config()
    cfg.training.hparam_eval_interval_steps = 10
    cfg.training.hparam_eval_num_examples = 2
    cfg.training.hparam_eval_on_train_env = train
    cfg.training.hparam_eval_on_nopq_env = nopq
    cfg.training.console_verbosity = "quiet"
    stats = {
        cfg.env.gym_id: _stats(-25.0, 1.25),
        cfg.training.hparam_eval_nopq_gym_id: _stats(-14.0, 2.5),
    }
    corpus = [{"episode_seed": 10000}, {"episode_seed": 10001}]
    monkeypatch.setattr(
        ppo_evaluation,
        "resolve_greedy_eval_corpus",
        lambda cfg, seeds: (corpus, "shared-corpus.jsonl", False),
    )
    calls = []

    def evaluate(**kwargs):
        calls.append(kwargs)
        return stats[kwargs["eval_gym_id"]]

    saved_checkpoints = []

    def save(cfg, agent, name, checkpoint_suffix):
        saved_checkpoints.append((name, checkpoint_suffix))
        return {"actor": "best-actor.pt", "critic": "best-critic.pt"}

    monkeypatch.setattr(ppo_evaluation, "run_fixed_greedy_eval", evaluate)
    monkeypatch.setattr(ppo_evaluation, "save_model_checkpoints", save)
    records = []
    writer = SimpleNamespace(
        add_scalar=lambda tag, value, step: records.append((tag, value, step))
    )
    return SimpleNamespace(
        cfg=cfg,
        evaluation=ppo_evaluation.TrainingEvaluation(cfg),
        stats=stats,
        calls=calls,
        saved_checkpoints=saved_checkpoints,
        records=records,
        writer=writer,
        agent=SimpleNamespace(actor=object()),
    )


def _run(context, step=10):
    context.evaluation.evaluate(
        agent=context.agent,
        writer=context.writer,
        global_step=step,
        run_name="test-run",
        device="cpu",
        obs_sf_normalizer=None,
    )


@pytest.mark.parametrize("train,nopq", [(True, False), (False, True), (True, True)])
def test_greedy_metrics_log_each_target_once_without_primary_aliases(
    train, nopq, monkeypatch
):
    #arrange
    context = _evaluation(monkeypatch, train=train, nopq=nopq)

    #act
    _run(context)

    #assert
    metrics = {tag: value for tag, value, _ in context.records}
    counts = Counter(tag for tag, _, _ in context.records)
    assert all(count == 1 for count in counts.values())
    assert all(step == 10 for _, _, step in context.records)
    assert all(tag.count("/") == 1 and tag == tag.lower() for tag in metrics)

    common_leaves = {
        "mean_return", "median_return", "min_return", "max_return",
        "mean_length", "max_length", "num_completed", "completion_rate",
        "num_truncated", "num_timed_out", "num_worker_exited",
        "mean_return_completed_only",
    }
    targets = []
    if train:
        targets.append(("train_env", context.stats[context.cfg.env.gym_id]))
    if nopq:
        targets.append(
            ("nopq", context.stats[context.cfg.training.hparam_eval_nopq_gym_id])
        )
    expected_tags = {"evaluation_greedy/selection_metric"}
    for target, stats in targets:
        expected_tags.update(f"evaluation_greedy/{target}_{leaf}" for leaf in common_leaves)
        expected_tags.add(f"performance/greedy_{target}_seconds")
        for leaf in common_leaves:
            field = "num_truncated_total" if leaf == "num_truncated" else leaf
            assert metrics[f"evaluation_greedy/{target}_{leaf}"] == stats[field]
        assert metrics[f"performance/greedy_{target}_seconds"] == stats["elapsed_seconds"]
    if nopq:
        expected_tags.add("evaluation_greedy/nopq_num_benchmark_truncated")
        assert metrics["evaluation_greedy/nopq_num_benchmark_truncated"] == 0
    if train and nopq:
        expected_tags.add("evaluation_greedy/nopq_minus_train_mean_return")
        assert metrics["evaluation_greedy/nopq_minus_train_mean_return"] == 11.0
        assert context.calls[1]["benchmark_episode_results_by_seed"] == {
            10000: {"episode_seed": 10000}, 10001: {"episode_seed": 10001}
        }
    else:
        assert context.calls[0]["benchmark_episode_results_by_seed"] is None
    assert set(metrics) == expected_tags

    primary_gym = (
        context.cfg.training.hparam_eval_nopq_gym_id if nopq else context.cfg.env.gym_id
    )
    primary_stats = context.stats[primary_gym]
    initial_score = primary_stats["mean_return_completed_only"]
    assert metrics["evaluation_greedy/selection_metric"] == initial_score
    assert context.evaluation.last is primary_stats
    assert context.evaluation.last_metric == initial_score
    assert context.evaluation.best_metric == initial_score
    assert context.evaluation.best_step == 10
    assert context.evaluation.next_step == 20
    assert context.saved_checkpoints == [("test-run", "best_greedy_eval")]

    # A later lower score changes the latest summary, while retaining the best model.
    #arrange
    for gym_id, stats in list(context.stats.items()):
        context.stats[gym_id] = _stats(stats["mean_return"] - 10, stats["elapsed_seconds"])

    #act
    _run(context, step=25)
    summary = build_training_summary(
        context.cfg, context.evaluation,
        run_name="test-run", global_step=25, policy_updates=5,
        episodes_completed=3, max_episodic_return=-5.0,
        last_100_avg_return=-30.0, timed_out=False, elapsed_seconds=4.0,
    )

    #assert
    assert context.evaluation.next_step == 30
    assert context.saved_checkpoints == [("test-run", "best_greedy_eval")]
    assert summary["greedy_eval_last_mean_return"] == context.stats[primary_gym]["mean_return"]
    assert summary["greedy_eval_last_selection_metric"] == initial_score - 10
    assert summary["selection_metric"] == initial_score - 10
    assert summary["greedy_eval_best_selection_metric"] == initial_score
    assert summary["greedy_eval_best_step"] == 10
    assert summary["greedy_eval_best_actor_path"] == "best-actor.pt"
    assert summary["greedy_eval_best_critic_path"] == "best-critic.pt"


@pytest.mark.parametrize("completed_mean", [None, float("nan"), float("inf")])
@pytest.mark.parametrize("train,nopq", [(True, False), (False, True)])
def test_completed_only_mean_is_logged_only_when_finite(
    completed_mean, train, nopq, monkeypatch
):
    #arrange
    context = _evaluation(monkeypatch, train=train, nopq=nopq)
    for stats in context.stats.values():
        stats["mean_return_completed_only"] = completed_mean
        stats["num_truncated_total"] = 1

    #act
    _run(context)

    #assert
    tags = {tag for tag, _, _ in context.records}
    target = "train_env" if train else "nopq"
    assert f"evaluation_greedy/{target}_mean_return_completed_only" not in tags
    assert f"evaluation_greedy/{target}_mean_return" in tags
    assert "evaluation_greedy/selection_metric" not in tags
    assert context.evaluation.last_metric is None
    assert context.evaluation.best_step is None
    assert context.evaluation.next_step == 20
    assert not context.saved_checkpoints
