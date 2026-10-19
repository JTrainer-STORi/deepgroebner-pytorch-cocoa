"""Final training-run metrics assembled from PPO statistics and evaluation results."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .greedy_eval import compute_greedy_eval_selection_metric

if TYPE_CHECKING:
    from .config_dataclasses import Config
    from .ppo_evaluation import TrainingEvaluation


def build_training_summary(
    cfg: Config,
    evaluation: TrainingEvaluation,
    *,
    run_name: str,
    global_step: int,
    policy_updates: int,
    episodes_completed: int,
    max_episodic_return: float,
    last_100_avg_return: float | None,
    timed_out: bool,
    elapsed_seconds: float,
) -> dict:
    """Build final run metrics from training counters and evaluation results."""
    greedy_metric = (
        None if evaluation.last is None else float(evaluation.last["mean_return"])
    )
    greedy_metric_completed_only = (
        None
        if evaluation.last is None
        or evaluation.last["mean_return_completed_only"] is None
        else float(evaluation.last["mean_return_completed_only"])
    )
    selection_metric_kind = compute_greedy_eval_selection_metric(
        evaluation.last_train,
        evaluation.last_nopq,
        run_train_env=evaluation.run_train_env,
        run_nopq_env=evaluation.run_nopq_env,
    )[1]
    summary = {
        "run_name": run_name,
        "global_step": int(global_step),
        "policy_updates": int(policy_updates),
        "episodes_completed": int(episodes_completed),
        "max_episodic_return": (
            None if episodes_completed == 0 else float(max_episodic_return)
        ),
        "last_100_avg_return": (
            None if last_100_avg_return is None else float(last_100_avg_return)
        ),
        "greedy_eval_last_mean_return": greedy_metric,
        "greedy_eval_last_mean_return_completed_only": greedy_metric_completed_only,
        "greedy_eval_last_median_return": (
            None if evaluation.last is None else float(evaluation.last["median_return"])
        ),
        "greedy_eval_last_min_return": (
            None if evaluation.last is None else float(evaluation.last["min_return"])
        ),
        "greedy_eval_last_max_return": (
            None if evaluation.last is None else float(evaluation.last["max_return"])
        ),
        "greedy_eval_last_mean_length": (
            None if evaluation.last is None else float(evaluation.last["mean_length"])
        ),
        "greedy_eval_last_max_length": (
            None if evaluation.last is None else int(evaluation.last["max_length"])
        ),
        "greedy_eval_last_num_completed": (
            None if evaluation.last is None else int(evaluation.last["num_completed"])
        ),
        "greedy_eval_last_completion_rate": (
            None
            if evaluation.last is None
            else float(evaluation.last["completion_rate"])
        ),
        "greedy_eval_last_num_truncated": (
            None
            if evaluation.last is None
            else int(evaluation.last["num_truncated_total"])
        ),
        "greedy_eval_last_num_timed_out": (
            None if evaluation.last is None else int(evaluation.last["num_timed_out"])
        ),
        "greedy_eval_last_num_worker_exited": (
            None
            if evaluation.last is None
            else int(evaluation.last["num_worker_exited"])
        ),
        "greedy_eval_num_examples": (
            int(len(evaluation.corpus_rows)) if evaluation.enabled else 0
        ),
        "greedy_eval_interval_steps": (
            int(cfg.training.greedy_eval_interval_steps) if evaluation.enabled else 0
        ),
        "greedy_eval_first_step": evaluation.first_step,
        "greedy_eval_seed": (
            int(cfg.training.greedy_eval_seed) if evaluation.enabled else None
        ),
        "greedy_eval_corpus_path": evaluation.corpus_path,
        "greedy_eval_last_selection_metric": (
            None if evaluation.last_metric is None else float(evaluation.last_metric)
        ),
        "greedy_eval_selection_metric_kind": selection_metric_kind,
        "greedy_eval_best_selection_metric": (
            None if evaluation.best_step is None else float(evaluation.best_metric)
        ),
        "greedy_eval_best_mean_return": (
            None
            if evaluation.best_step is None
            else (
                float(evaluation.best_nopq_mean)
                if evaluation.primary_target == "nopq"
                and evaluation.best_nopq_step is not None
                else float(evaluation.best_metric)
            )
        ),
        "greedy_eval_best_step": (
            None if evaluation.best_step is None else int(evaluation.best_step)
        ),
        "greedy_eval_best_actor_path": evaluation.best_checkpoints.get("actor"),
        "greedy_eval_best_critic_path": evaluation.best_checkpoints.get("critic"),
        "greedy_eval_gym_id": (evaluation.gym_id if evaluation.enabled else None),
        "greedy_eval_compare_nopq": bool(evaluation.run_nopq_env),
        "greedy_eval_train_is_benchmark": bool(
            evaluation.run_train_env and evaluation.run_nopq_env
        ),
        "greedy_eval_nopq_uses_train_benchmark": bool(
            evaluation.run_train_env and evaluation.run_nopq_env
        ),
        "greedy_eval_nopq_gym_id": (
            evaluation.nopq_gym_id if evaluation.run_nopq_env else None
        ),
        "greedy_eval_nopq_last_mean_return": (
            None
            if evaluation.last_nopq is None
            else float(evaluation.last_nopq["mean_return"])
        ),
        "greedy_eval_nopq_last_mean_return_completed_only": (
            None
            if evaluation.last_nopq is None
            or evaluation.last_nopq["mean_return_completed_only"] is None
            else float(evaluation.last_nopq["mean_return_completed_only"])
        ),
        "greedy_eval_nopq_last_median_return": (
            None
            if evaluation.last_nopq is None
            else float(evaluation.last_nopq["median_return"])
        ),
        "greedy_eval_nopq_last_min_return": (
            None
            if evaluation.last_nopq is None
            else float(evaluation.last_nopq["min_return"])
        ),
        "greedy_eval_nopq_last_max_return": (
            None
            if evaluation.last_nopq is None
            else float(evaluation.last_nopq["max_return"])
        ),
        "greedy_eval_nopq_last_mean_length": (
            None
            if evaluation.last_nopq is None
            else float(evaluation.last_nopq["mean_length"])
        ),
        "greedy_eval_nopq_last_max_length": (
            None
            if evaluation.last_nopq is None
            else int(evaluation.last_nopq["max_length"])
        ),
        "greedy_eval_nopq_last_num_completed": (
            None
            if evaluation.last_nopq is None
            else int(evaluation.last_nopq["num_completed"])
        ),
        "greedy_eval_nopq_last_completion_rate": (
            None
            if evaluation.last_nopq is None
            else float(evaluation.last_nopq["completion_rate"])
        ),
        "greedy_eval_nopq_last_num_truncated": (
            None
            if evaluation.last_nopq is None
            else int(evaluation.last_nopq["num_truncated_total"])
        ),
        "greedy_eval_nopq_last_num_benchmark_truncated": (
            None
            if evaluation.last_nopq is None
            else int(evaluation.last_nopq["num_benchmark_truncated"])
        ),
        "greedy_eval_nopq_last_num_timed_out": (
            None
            if evaluation.last_nopq is None
            else int(evaluation.last_nopq["num_timed_out"])
        ),
        "greedy_eval_nopq_last_num_worker_exited": (
            None
            if evaluation.last_nopq is None
            else int(evaluation.last_nopq["num_worker_exited"])
        ),
        "greedy_eval_nopq_best_mean_return": (
            None
            if evaluation.best_nopq_step is None
            else float(evaluation.best_nopq_mean)
        ),
        "greedy_eval_nopq_best_step": (
            None
            if evaluation.best_nopq_step is None
            else int(evaluation.best_nopq_step)
        ),
        "greedy_eval_nopq_minus_base_last_mean_return": (
            None
            if (evaluation.last_train is None or evaluation.last_nopq is None)
            else float(
                evaluation.last_nopq["mean_return"]
                - evaluation.last_train["mean_return"]
            )
        ),
        "selection_metric": (
            float(evaluation.last_metric)
            if evaluation.last_metric is not None
            else (None if last_100_avg_return is None else float(last_100_avg_return))
        ),
        "timed_out": bool(timed_out),
        "elapsed_seconds": float(elapsed_seconds),
    }
    return summary
