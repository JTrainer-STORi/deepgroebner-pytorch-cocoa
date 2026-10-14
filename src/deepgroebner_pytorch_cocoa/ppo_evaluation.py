"""Periodic greedy evaluation and best-checkpoint selection for a training run."""

from __future__ import annotations

from .greedy_eval import (
    resolve_greedy_eval_targets,
    build_greedy_eval_seeds,
    resolve_greedy_eval_corpus,
    run_fixed_greedy_eval,
    compute_greedy_eval_selection_metric,
)
from .training_utils import console_print, save_model_checkpoints


class TrainingEvaluation:
    """Handles greedy evaluation of the policy during training if enabled"""

    def __init__(self, cfg):
        self.cfg = cfg
        greedy_eval_plan = resolve_greedy_eval_targets(cfg)
        self.enabled = greedy_eval_plan["enabled"]
        self.run_train_env = greedy_eval_plan["run_train_env"]
        self.run_nopq_env = greedy_eval_plan["run_nopq_env"]
        self.primary_target = greedy_eval_plan["primary_target"]
        self.nopq_gym_id = greedy_eval_plan["nopq_gym_id"]
        self.gym_id = (
            cfg.env.gym_id
            if self.primary_target == "train"
            else self.nopq_gym_id if self.primary_target == "nopq" else None
        )
        self.targets = []
        if self.run_train_env:
            self.targets.append(
                {
                    "key": "train",
                    "label": "",
                    "gym_id": cfg.env.gym_id,
                    "is_primary": self.primary_target == "train",
                }
            )
        if self.run_nopq_env:
            self.targets.append(
                {
                    "key": "nopq",
                    "label": " NoPQ",
                    "gym_id": self.nopq_gym_id,
                    "is_primary": self.primary_target == "nopq",
                }
            )
        self.seeds = build_greedy_eval_seeds(cfg, self.enabled)
        if self.enabled:
            self.corpus_rows, self.corpus_path, self.corpus_generated = (
                resolve_greedy_eval_corpus(
                    cfg,
                    self.seeds,
                )
            )
        else:
            self.corpus_rows = []
            self.corpus_path = None
            self.corpus_generated = False
        if self.corpus_path is not None:
            if self.corpus_generated:
                console_print(
                    cfg, f"Generated greedy eval corpus at {self.corpus_path}"
                )
            else:
                console_print(
                    cfg, f"Loaded greedy eval corpus from {self.corpus_path}"
                )
        self.next_step = (
            cfg.training.hparam_eval_interval_steps if self.enabled else None
        )
        self.last = None
        self.last_train = None
        self.last_nopq = None
        self.last_metric = None
        self.best_metric = float("-inf")
        self.best_step = None
        self.best_nopq_mean = float("-inf")
        self.best_nopq_step = None
        self.best_checkpoints = {}

    def evaluate(
        self, *, agent, writer, global_step, run_name, device, obs_sf_normalizer
    ):
        """Run evaluation; the caller decides when to evaluate."""
        cfg = self.cfg
        greedy_eval_round_stats = {}
        for greedy_eval_target in self.targets:
            benchmark_episode_results_by_seed = None
            if (
                greedy_eval_target["key"] == "nopq"
                and self.run_train_env
                and "train" in greedy_eval_round_stats
            ):
                benchmark_episode_results_by_seed = {
                    int(result["episode_seed"]): result
                    for result in greedy_eval_round_stats["train"].get(
                        "episode_results", []
                    )
                }
            console_print(
                cfg,
                f"Starting greedy eval{greedy_eval_target['label']} @ step {global_step}: "
                f"examples={len(self.corpus_rows)}",
                flush=True,
            )
            greedy_eval_stats = run_fixed_greedy_eval(
                actor=agent.actor,
                eval_gym_id=greedy_eval_target["gym_id"],
                eval_corpus_rows=self.corpus_rows,
                device=device,
                obs_sf_normalizer=obs_sf_normalizer,
                cfg=cfg,
                benchmark_episode_results_by_seed=benchmark_episode_results_by_seed,
            )
            greedy_eval_round_stats[greedy_eval_target["key"]] = greedy_eval_stats

            if greedy_eval_target["key"] == "train":
                self.last_train = greedy_eval_stats
            elif greedy_eval_target["key"] == "nopq":
                self.last_nopq = greedy_eval_stats
                if (
                    greedy_eval_stats["num_truncated_total"] == 0
                    and greedy_eval_stats["mean_return_completed_only"] is not None
                    and greedy_eval_stats["mean_return_completed_only"]
                    > self.best_nopq_mean
                ):
                    self.best_nopq_mean = greedy_eval_stats[
                        "mean_return_completed_only"
                    ]
                    self.best_nopq_step = global_step

                writer.add_scalar(
                    "charts/greedy_eval_nopq_mean_return",
                    greedy_eval_stats["mean_return"],
                    global_step,
                )
                writer.add_scalar(
                    "charts/greedy_eval_nopq_median_return",
                    greedy_eval_stats["median_return"],
                    global_step,
                )
                writer.add_scalar(
                    "charts/greedy_eval_nopq_min_return",
                    greedy_eval_stats["min_return"],
                    global_step,
                )
                writer.add_scalar(
                    "charts/greedy_eval_nopq_max_return",
                    greedy_eval_stats["max_return"],
                    global_step,
                )
                writer.add_scalar(
                    "charts/greedy_eval_nopq_mean_length",
                    greedy_eval_stats["mean_length"],
                    global_step,
                )
                writer.add_scalar(
                    "charts/greedy_eval_nopq_max_length",
                    greedy_eval_stats["max_length"],
                    global_step,
                )
                writer.add_scalar(
                    "charts/greedy_eval_nopq_num_completed",
                    greedy_eval_stats["num_completed"],
                    global_step,
                )
                writer.add_scalar(
                    "charts/greedy_eval_nopq_completion_rate",
                    greedy_eval_stats["completion_rate"],
                    global_step,
                )
                writer.add_scalar(
                    "charts/greedy_eval_nopq_num_truncated",
                    greedy_eval_stats["num_truncated_total"],
                    global_step,
                )
                writer.add_scalar(
                    "charts/greedy_eval_nopq_num_benchmark_truncated",
                    greedy_eval_stats["num_benchmark_truncated"],
                    global_step,
                )
                writer.add_scalar(
                    "charts/greedy_eval_nopq_num_timed_out",
                    greedy_eval_stats["num_timed_out"],
                    global_step,
                )
                writer.add_scalar(
                    "charts/greedy_eval_nopq_num_worker_exited",
                    greedy_eval_stats["num_worker_exited"],
                    global_step,
                )
                if greedy_eval_stats["mean_return_completed_only"] is not None:
                    writer.add_scalar(
                        "charts/greedy_eval_nopq_mean_return_completed_only",
                        greedy_eval_stats["mean_return_completed_only"],
                        global_step,
                    )

            if greedy_eval_target["is_primary"]:
                self.last = greedy_eval_stats
                writer.add_scalar(
                    "charts/greedy_eval_mean_return",
                    greedy_eval_stats["mean_return"],
                    global_step,
                )
                writer.add_scalar(
                    "charts/greedy_eval_median_return",
                    greedy_eval_stats["median_return"],
                    global_step,
                )
                writer.add_scalar(
                    "charts/greedy_eval_min_return",
                    greedy_eval_stats["min_return"],
                    global_step,
                )
                writer.add_scalar(
                    "charts/greedy_eval_max_return",
                    greedy_eval_stats["max_return"],
                    global_step,
                )
                writer.add_scalar(
                    "charts/greedy_eval_mean_length",
                    greedy_eval_stats["mean_length"],
                    global_step,
                )
                writer.add_scalar(
                    "charts/greedy_eval_max_length",
                    greedy_eval_stats["max_length"],
                    global_step,
                )
                writer.add_scalar(
                    "charts/greedy_eval_num_completed",
                    greedy_eval_stats["num_completed"],
                    global_step,
                )
                writer.add_scalar(
                    "charts/greedy_eval_completion_rate",
                    greedy_eval_stats["completion_rate"],
                    global_step,
                )
                writer.add_scalar(
                    "charts/greedy_eval_num_truncated",
                    greedy_eval_stats["num_truncated_total"],
                    global_step,
                )
                writer.add_scalar(
                    "charts/greedy_eval_num_timed_out",
                    greedy_eval_stats["num_timed_out"],
                    global_step,
                )
                writer.add_scalar(
                    "charts/greedy_eval_num_worker_exited",
                    greedy_eval_stats["num_worker_exited"],
                    global_step,
                )
                if greedy_eval_stats["mean_return_completed_only"] is not None:
                    writer.add_scalar(
                        "charts/greedy_eval_mean_return_completed_only",
                        greedy_eval_stats["mean_return_completed_only"],
                        global_step,
                    )

            console_print(
                cfg,
                f"Greedy eval{greedy_eval_target['label']} @ step {global_step}: "
                f"mean={greedy_eval_stats['mean_return']:.4f}, "
                f"median={greedy_eval_stats['median_return']:.4f}, "
                f"examples={greedy_eval_stats['num_examples']}, "
                f"completed={greedy_eval_stats['num_completed']}, "
                f"truncated={greedy_eval_stats['num_truncated_total']}, "
                f"timed_out={greedy_eval_stats['num_timed_out']}, "
                f"worker_exited={greedy_eval_stats['num_worker_exited']}, "
                f"elapsed={greedy_eval_stats['elapsed_seconds']:.1f}s",
            )

        self.last = (
            greedy_eval_round_stats.get(self.primary_target)
            if self.primary_target is not None
            else None
        )
        selection_metric, selection_metric_kind = (
            compute_greedy_eval_selection_metric(
                greedy_eval_round_stats.get("train"),
                greedy_eval_round_stats.get("nopq"),
                run_train_env=self.run_train_env,
                run_nopq_env=self.run_nopq_env,
            )
        )
        self.last_metric = selection_metric
        if selection_metric is not None:
            writer.add_scalar(
                "charts/greedy_eval_selection_metric", selection_metric, global_step
            )
        if self.run_train_env and self.run_nopq_env:
            train_stats = greedy_eval_round_stats.get("train")
            nopq_stats = greedy_eval_round_stats.get("nopq")
            if train_stats is not None and nopq_stats is not None:
                writer.add_scalar(
                    "charts/greedy_eval_nopq_minus_train_mean_return",
                    float(nopq_stats["mean_return"] - train_stats["mean_return"]),
                    global_step,
                )
                writer.add_scalar(
                    "charts/greedy_eval_nopq_num_benchmark_truncated",
                    int(nopq_stats["num_benchmark_truncated"]),
                    global_step,
                )

        if selection_metric is not None and selection_metric > self.best_metric:
            self.best_metric = selection_metric
            self.best_step = global_step
            self.best_checkpoints = save_model_checkpoints(
                cfg,
                agent,
                run_name,
                checkpoint_suffix="best_greedy_eval",
            )
            for checkpoint_kind, checkpoint_path in sorted(
                self.best_checkpoints.items()
            ):
                console_print(
                    cfg,
                    f"Saved best greedy eval {checkpoint_kind} checkpoint "
                    f"@ step {global_step} using {selection_metric_kind}={selection_metric:.4f} "
                    f"to {checkpoint_path}",
                )

        if selection_metric is not None:
            console_print(
                cfg,
                f"Greedy eval selection metric @ step {global_step}: "
                f"{selection_metric_kind}={selection_metric:.4f}",
                flush=True,
            )
        elif selection_metric_kind is not None:
            console_print(
                cfg,
                f"Greedy eval selection metric @ step {global_step} skipped: "
                f"{selection_metric_kind}",
                flush=True,
            )

        while self.next_step is not None and self.next_step <= global_step:
            self.next_step += cfg.training.hparam_eval_interval_steps
