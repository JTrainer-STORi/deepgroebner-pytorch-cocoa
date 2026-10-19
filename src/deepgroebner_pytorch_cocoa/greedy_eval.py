"""Fixed-corpus greedy evaluation, corpus caching, and subprocess timeout handling."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import pathlib
import re
import time
from collections import deque
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from .buchberger_env import (
    parse_class_list_token,
    parse_int_or_range_token,
    parse_nvars_from_ideal_dist,
    split_top_level_dash,
)
from .combined_networks import CombinedPolicyNetwork
from .evaluation_corpus import load_explicit_greedy_eval_corpus
from .ideal_corpus import load_ideal_corpus
from .model_input_utils import build_model_inputs_from_single_obs
from .package_resources import laman_graph_path
from .ppo_normalization import (
    RunningMeanStd,
    normalize_eval_state_features,
    make_obs_sf_normalizer_snapshot,
)
from .training_env import build_fixed_ideal_env_kwargs_for_gym_id
from .training_utils import console_print, format_optional_metric
from .wrapped import CGLeadMonomialsEnv

_NOPQ_GYM_ID = "BuchbergerExponentsNoPQ-v1"

STATE_FEATURE_P_CARDINALITY_IDX = 1
STATE_FEATURE_N_PAIRS_PROCESSED_IDX = 12
STATE_FEATURE_STEPS_SINCE_BASIS_UPDATE_IDX = 13
EVAL_TIMEOUT_GRACE_SECONDS = 30.0
EVAL_TIMEOUT_MAX_EXTENSIONS = 2
EVAL_TIMEOUT_PROGRESS_HISTORY_MAXLEN = 32
EVAL_TIMEOUT_MIN_PROGRESS_SAMPLES = 8
EVAL_TIMEOUT_RECENT_ACTIVITY_SECONDS = 10.0
EVAL_TIMEOUT_MIN_PAIR_CARDINALITY_DROP = 2


def _greedy_eval_episode_progress_state(
    obs: dict,
    ep_return: float,
    ep_length: int,
    truncated_reason: str | None,
    hit_zero_action_state: bool,
) -> dict:
    state_features = np.asarray(obs["state_features"], dtype=np.float32).reshape(-1)
    return {
        "return": float(ep_return),
        "length": int(ep_length),
        "truncated_reason": truncated_reason,
        "hit_zero_action_state": bool(hit_zero_action_state),
        "pair_cardinality": int(
            round(float(state_features[STATE_FEATURE_P_CARDINALITY_IDX]))
        ),
        "n_pairs_processed": int(
            round(float(state_features[STATE_FEATURE_N_PAIRS_PROCESSED_IDX]))
        ),
        "steps_since_basis_update": int(
            round(float(state_features[STATE_FEATURE_STEPS_SINCE_BASIS_UPDATE_IDX]))
        ),
        "n_pairs": int(np.asarray(obs["P"]).shape[0]),
        "n_basis": int(np.asarray(obs["G"]).shape[0]),
    }


def _summarize_greedy_eval_scalar_array(
    values: list[float] | list[int],
) -> dict[str, float | None]:
    """Returns mean/median/min/max for eval aggregates, or None when empty."""
    if not values:
        return {
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
    values_arr = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(values_arr.mean()),
        "median": float(np.median(values_arr)),
        "min": float(values_arr.min()),
        "max": float(values_arr.max()),
    }


def _should_extend_greedy_eval_timeout(
    progress_history: deque[dict],
    *,
    now: float,
) -> bool:
    """Returns True when recent progress suggests the episode is still converging."""
    if len(progress_history) < EVAL_TIMEOUT_MIN_PROGRESS_SAMPLES:
        return False

    recent = list(progress_history)
    last_update_time = float(recent[-1]["received_at"])
    if now - last_update_time > EVAL_TIMEOUT_RECENT_ACTIVITY_SECONDS:
        return False

    pair_cardinalities = [
        int(entry["pair_cardinality"])
        for entry in recent
        if entry.get("pair_cardinality") is not None
    ]
    if len(pair_cardinalities) < EVAL_TIMEOUT_MIN_PROGRESS_SAMPLES:
        return False

    start_pair_count = pair_cardinalities[0]
    end_pair_count = pair_cardinalities[-1]
    if (start_pair_count - end_pair_count) < EVAL_TIMEOUT_MIN_PAIR_CARDINALITY_DROP:
        return False

    decreasing_steps = sum(
        1
        for prev_value, next_value in zip(pair_cardinalities, pair_cardinalities[1:])
        if next_value < prev_value
    )
    return decreasing_steps >= max(3, len(pair_cardinalities) // 3)


def _build_greedy_eval_watchdog_result(
    last_progress: dict,
    *,
    truncated_reason: str,
    elapsed_seconds: float,
    timed_out: bool,
    timeout_extensions_used: int,
    worker_exitcode: int | None = None,
    worker_error: str | None = None,
) -> dict:
    result = dict(last_progress)
    result["timed_out"] = bool(timed_out)
    result["truncated_reason"] = truncated_reason
    result["elapsed_seconds"] = float(elapsed_seconds)
    result["initial_pair_count"] = None
    result["max_pairs_processed_limit"] = None
    result["benchmark_pairs_processed_limit"] = None
    result["max_steps_since_basis_update_limit"] = None
    result["timeout_extensions_used"] = int(timeout_extensions_used)
    if worker_exitcode is not None:
        result["worker_exitcode"] = int(worker_exitcode)
    if worker_error is not None:
        result["worker_error"] = str(worker_error)
    return result


def _print_greedy_eval_progress_line(
    cfg,
    progress_prefix: str,
    seed: int,
    progress_state: dict,
    *,
    elapsed_seconds: float,
) -> None:
    console_print(
        cfg,
        f"{progress_prefix}: seed={seed}, length={int(progress_state.get('length', 0))}, "
        f"n_pairs={progress_state.get('n_pairs')}, n_basis={progress_state.get('n_basis')}, "
        f"processed={int(progress_state.get('n_pairs_processed', 0))}, "
        f"stagnation={int(progress_state.get('steps_since_basis_update', 0))}, "
        f"elapsed={elapsed_seconds:.1f}s",
        flush=True,
        level="verbose",
    )


def _run_single_greedy_eval_episode(
    actor: nn.Module,
    eval_env: gym.Env,
    seed: int,
    device: torch.device,
    obs_sf_normalizer: RunningMeanStd | dict[str, torch.Tensor] | None,
    cfg,
    benchmark_episode_result: dict | None = None,
    *,
    progress_callback=None,
    progress_interval_seconds: float | None = None,
    progress_prefix: str = "Greedy eval progress",
) -> dict:
    obs, _ = eval_env.reset(seed=seed)
    state_features = np.asarray(obs["state_features"], dtype=np.float32).reshape(-1)
    initial_pair_count = max(
        0, int(round(float(state_features[STATE_FEATURE_P_CARDINALITY_IDX])))
    )

    done = False
    truncated = False
    ep_return = 0.0
    ep_length = 0
    hit_zero_action_state = False
    truncated_reason = None
    episode_start_time = time.monotonic()
    next_progress_time = (
        episode_start_time + progress_interval_seconds
        if progress_interval_seconds is not None and progress_interval_seconds > 0
        else None
    )

    while not (done or truncated):
        model_input = build_model_inputs_from_single_obs(obs, device)
        normalize_eval_state_features(model_input, obs_sf_normalizer, cfg)

        valid_action_count = int(model_input.P_mask.sum().item())
        if valid_action_count == 0:
            done = True
            hit_zero_action_state = True
            break

        with torch.no_grad():
            logits = actor(model_input).squeeze(-1)
            masked_logits = torch.where(
                model_input.P_mask,
                logits,
                logits.new_full((), -1e8),
            )
            action = int(torch.argmax(masked_logits, dim=-1).item())

        obs, reward, done, truncated, _ = eval_env.step(action)
        ep_return += float(reward)
        ep_length += 1

        progress_state = _greedy_eval_episode_progress_state(
            obs,
            ep_return,
            ep_length,
            truncated_reason,
            hit_zero_action_state,
        )
        if progress_callback is not None:
            progress_callback(progress_state)

        if next_progress_time is not None:
            now = time.monotonic()
            if now >= next_progress_time:
                console_print(
                    cfg,
                    f"{progress_prefix}: seed={seed}, length={ep_length}, "
                    f"n_pairs={progress_state['n_pairs']}, n_basis={progress_state['n_basis']}, "
                    f"processed={progress_state['n_pairs_processed']}, "
                    f"stagnation={progress_state['steps_since_basis_update']}, "
                    f"elapsed={now - episode_start_time:.1f}s",
                    flush=True,
                    level="verbose",
                )
                next_progress_time = now + progress_interval_seconds

    result = _greedy_eval_episode_progress_state(
        obs,
        ep_return,
        ep_length,
        truncated_reason,
        hit_zero_action_state,
    )
    result["elapsed_seconds"] = float(time.monotonic() - episode_start_time)
    result["timed_out"] = False
    result["timeout_extensions_used"] = 0
    result["initial_pair_count"] = int(initial_pair_count)
    result["max_pairs_processed_limit"] = None
    result["benchmark_pairs_processed_limit"] = None
    result["max_steps_since_basis_update_limit"] = None
    return result


def _run_greedy_eval_episode_worker(
    conn,
    actor_state_dict: dict[str, torch.Tensor],
    actor_cfg,
    eval_gym_id: str,
    env_kwargs: dict,
    seed: int,
    obs_sf_normalizer_state: dict[str, torch.Tensor] | None,
    cfg,
    benchmark_episode_result: dict | None,
    progress_interval_seconds: float | None,
    progress_prefix: str,
):
    env = None
    try:
        env = gym.make(eval_gym_id, **env_kwargs)
        actor = CombinedPolicyNetwork(actor_cfg).to(device="cpu")
        actor.load_state_dict(actor_state_dict)
        actor.eval()

        def _send_progress(progress_state: dict):
            conn.send({"kind": "progress", "state": progress_state})

        result = _run_single_greedy_eval_episode(
            actor=actor,
            eval_env=env,
            seed=seed,
            device=torch.device("cpu"),
            obs_sf_normalizer=obs_sf_normalizer_state,
            cfg=cfg,
            benchmark_episode_result=benchmark_episode_result,
            progress_callback=_send_progress,
            progress_interval_seconds=progress_interval_seconds,
            progress_prefix=progress_prefix,
        )
        conn.send({"kind": "final", "result": result})
    except BaseException as exc:
        conn.send({"kind": "error", "error": repr(exc)})
    finally:
        if env is not None:
            env.close()
        conn.close()


def _run_single_greedy_eval_episode_with_timeout(
    actor: nn.Module,
    eval_gym_id: str,
    env_kwargs: dict,
    seed: int,
    obs_sf_normalizer: RunningMeanStd | None,
    cfg,
    timeout_seconds: float,
    benchmark_episode_result: dict | None = None,
    progress_interval_seconds: float | None = None,
    progress_prefix: str = "Greedy eval progress",
) -> dict:
    ctx = mp.get_context("spawn")
    recv_conn, send_conn = ctx.Pipe(duplex=False)
    actor_state_dict = {
        key: value.detach().cpu() for key, value in actor.state_dict().items()
    }
    process = ctx.Process(
        target=_run_greedy_eval_episode_worker,
        args=(
            send_conn,
            actor_state_dict,
            cfg.actor,
            eval_gym_id,
            env_kwargs,
            seed,
            make_obs_sf_normalizer_snapshot(obs_sf_normalizer),
            cfg,
            benchmark_episode_result,
            None,
            progress_prefix,
        ),
    )
    process.start()
    send_conn.close()

    watchdog_start_time = time.monotonic()
    deadline = time.monotonic() + timeout_seconds
    last_progress = {
        "return": 0.0,
        "length": 0,
        "truncated_reason": None,
        "hit_zero_action_state": False,
        "pair_cardinality": None,
        "n_pairs_processed": 0,
        "steps_since_basis_update": 0,
        "n_pairs": None,
        "n_basis": None,
    }
    progress_history: deque[dict] = deque(maxlen=EVAL_TIMEOUT_PROGRESS_HISTORY_MAXLEN)
    timeout_extensions_used = 0
    next_progress_print_time = (
        watchdog_start_time + progress_interval_seconds
        if progress_interval_seconds is not None and progress_interval_seconds > 0
        else None
    )
    try:
        while True:
            now = time.monotonic()
            remaining = deadline - now
            if remaining <= 0:
                if (
                    timeout_extensions_used < EVAL_TIMEOUT_MAX_EXTENSIONS
                    and _should_extend_greedy_eval_timeout(progress_history, now=now)
                ):
                    timeout_extensions_used += 1
                    deadline = now + EVAL_TIMEOUT_GRACE_SECONDS
                    console_print(
                        cfg,
                        f"{progress_prefix}: seed={seed}, extending timeout by "
                        f"{EVAL_TIMEOUT_GRACE_SECONDS:.1f}s because pair_cardinality "
                        f"continues to shrink; current_pair_cardinality="
                        f"{last_progress.get('pair_cardinality')}, "
                        f"extensions_used={timeout_extensions_used}/{EVAL_TIMEOUT_MAX_EXTENSIONS}",
                        flush=True,
                        level="verbose",
                    )
                    continue

                process.terminate()
                process.join(timeout=1.0)
                return _build_greedy_eval_watchdog_result(
                    last_progress,
                    truncated_reason="episode_seconds",
                    elapsed_seconds=float(now - watchdog_start_time),
                    timed_out=True,
                    timeout_extensions_used=timeout_extensions_used,
                    worker_exitcode=process.exitcode,
                )

            poll_timeout = remaining
            if next_progress_print_time is not None:
                poll_timeout = min(
                    poll_timeout, max(0.0, next_progress_print_time - now)
                )

            if recv_conn.poll(timeout=poll_timeout):
                try:
                    message = recv_conn.recv()
                except (EOFError, OSError) as exc:
                    process.join(timeout=1.0)
                    return _build_greedy_eval_watchdog_result(
                        last_progress,
                        truncated_reason="worker_exit",
                        elapsed_seconds=float(time.monotonic() - watchdog_start_time),
                        timed_out=False,
                        timeout_extensions_used=timeout_extensions_used,
                        worker_exitcode=process.exitcode,
                        worker_error=f"{type(exc).__name__}: {exc}",
                    )
                kind = message.get("kind")
                if kind == "progress":
                    last_progress = dict(message["state"])
                    progress_history.append(
                        {
                            "received_at": time.monotonic(),
                            "pair_cardinality": last_progress.get("pair_cardinality"),
                        }
                    )
                    continue
                if kind == "final":
                    result = dict(message["result"])
                    result["timed_out"] = False
                    result["timeout_extensions_used"] = int(timeout_extensions_used)
                    process.join(timeout=1.0)
                    return result
                if kind == "error":
                    process.join(timeout=1.0)
                    raise RuntimeError(
                        f"Greedy eval worker failed for seed {seed}: {message['error']}"
                    )
            else:
                now = time.monotonic()
                if (
                    next_progress_print_time is not None
                    and now >= next_progress_print_time
                ):
                    _print_greedy_eval_progress_line(
                        cfg,
                        progress_prefix,
                        seed,
                        last_progress,
                        elapsed_seconds=float(now - watchdog_start_time),
                    )
                    next_progress_print_time = now + progress_interval_seconds
                    continue
                if not process.is_alive():
                    break
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
        recv_conn.close()

    raise RuntimeError(
        f"Greedy eval worker exited without reporting a result for seed {seed}."
    )


def run_fixed_greedy_eval(
    actor: nn.Module,
    eval_gym_id: str,
    eval_corpus_rows: list[dict],
    device: torch.device,
    obs_sf_normalizer: RunningMeanStd | None,
    cfg,
    benchmark_episode_results_by_seed: dict[int, dict] | None = None,
) -> dict:
    """Runs greedy policy evaluation on a fixed ideal corpus and returns aggregate stats."""
    was_training = actor.training
    actor.eval()

    returns = []
    lengths = []
    completed_returns = []
    completed_lengths = []
    zero_action_episodes = 0
    timed_out_episodes = 0
    worker_exited_episodes = 0
    truncated_max_pairs_processed = 0
    truncated_steps_since_basis_update = 0
    truncated_episode_seconds = 0
    truncated_worker_exit = 0
    benchmark_truncated_episodes = 0
    episode_results = []
    eval_start_time = time.monotonic()
    progress_interval = float(
        getattr(cfg.training, "greedy_eval_progress_interval_seconds", 0.0) or 0.0
    )
    timeout_seconds = getattr(cfg.training, "greedy_eval_max_episode_seconds", None)
    if timeout_seconds is not None:
        timeout_seconds = float(timeout_seconds)
        if timeout_seconds <= 0:
            timeout_seconds = None

    for row_idx, row in enumerate(eval_corpus_rows, start=1):
        episode_seed = int(row["episode_seed"])
        env_kwargs = build_fixed_ideal_env_kwargs_for_gym_id(cfg, eval_gym_id, row)
        benchmark_episode_result = None
        if benchmark_episode_results_by_seed is not None:
            benchmark_episode_result = benchmark_episode_results_by_seed.get(
                episode_seed
            )

        if timeout_seconds is not None:
            result = _run_single_greedy_eval_episode_with_timeout(
                actor=actor,
                eval_gym_id=eval_gym_id,
                env_kwargs=env_kwargs,
                seed=episode_seed,
                obs_sf_normalizer=obs_sf_normalizer,
                cfg=cfg,
                timeout_seconds=timeout_seconds,
                benchmark_episode_result=benchmark_episode_result,
                progress_interval_seconds=progress_interval,
                progress_prefix=f"Greedy eval progress: example={row_idx}/{len(eval_corpus_rows)}",
            )
        else:
            eval_env = gym.make(eval_gym_id, **env_kwargs)
            try:
                result = _run_single_greedy_eval_episode(
                    actor=actor,
                    eval_env=eval_env,
                    seed=episode_seed,
                    device=device,
                    obs_sf_normalizer=obs_sf_normalizer,
                    cfg=cfg,
                    benchmark_episode_result=benchmark_episode_result,
                    progress_interval_seconds=progress_interval,
                    progress_prefix=f"Greedy eval progress: example={row_idx}/{len(eval_corpus_rows)}",
                )
            finally:
                eval_env.close()

        if result["hit_zero_action_state"]:
            zero_action_episodes += 1
        if result.get("timed_out", False):
            timed_out_episodes += 1

        truncated_reason = result.get("truncated_reason")
        if truncated_reason == "max_pairs_processed":
            truncated_max_pairs_processed += 1
        elif truncated_reason == "steps_since_basis_update":
            truncated_steps_since_basis_update += 1
        elif truncated_reason == "episode_seconds":
            truncated_episode_seconds += 1
        elif truncated_reason == "worker_exit":
            truncated_worker_exit += 1
            worker_exited_episodes += 1

        if (
            result.get("benchmark_pairs_processed_limit") is not None
            and truncated_reason == "max_pairs_processed"
        ):
            benchmark_truncated_episodes += 1

        returns.append(float(result["return"]))
        lengths.append(int(result["length"]))
        if truncated_reason is None:
            completed_returns.append(float(result["return"]))
            completed_lengths.append(int(result["length"]))
        episode_results.append(
            {
                "ideal_id": row.get("ideal_id"),
                "episode_seed": episode_seed,
                **result,
            }
        )

    if was_training:
        actor.train()

    returns_summary = _summarize_greedy_eval_scalar_array(returns)
    lengths_summary = _summarize_greedy_eval_scalar_array(lengths)
    completed_returns_summary = _summarize_greedy_eval_scalar_array(completed_returns)
    completed_lengths_summary = _summarize_greedy_eval_scalar_array(completed_lengths)
    num_examples = int(len(eval_corpus_rows))
    num_completed = int(len(completed_returns))
    truncated_total = int(
        sum(
            1
            for episode_result in episode_results
            if episode_result.get("truncated_reason") is not None
        )
    )
    return {
        "mean_return": returns_summary["mean"],
        "median_return": returns_summary["median"],
        "min_return": returns_summary["min"],
        "max_return": returns_summary["max"],
        "mean_length": lengths_summary["mean"],
        "max_length": (
            0 if lengths_summary["max"] is None else int(lengths_summary["max"])
        ),
        "mean_return_completed_only": completed_returns_summary["mean"],
        "median_return_completed_only": completed_returns_summary["median"],
        "min_return_completed_only": completed_returns_summary["min"],
        "max_return_completed_only": completed_returns_summary["max"],
        "mean_length_completed_only": completed_lengths_summary["mean"],
        "max_length_completed_only": (
            0
            if completed_lengths_summary["max"] is None
            else int(completed_lengths_summary["max"])
        ),
        "num_examples": num_examples,
        "num_completed": num_completed,
        "completion_rate": (
            float(num_completed / num_examples) if num_examples > 0 else None
        ),
        "num_zero_action": int(zero_action_episodes),
        "num_timed_out": int(timed_out_episodes),
        "num_worker_exited": int(worker_exited_episodes),
        "num_truncated_max_pairs_processed": int(truncated_max_pairs_processed),
        "num_truncated_steps_since_basis_update": int(
            truncated_steps_since_basis_update
        ),
        "num_truncated_episode_seconds": int(truncated_episode_seconds),
        "num_truncated_worker_exit": int(truncated_worker_exit),
        "num_benchmark_truncated": int(benchmark_truncated_episodes),
        "num_truncated_total": truncated_total,
        "elapsed_seconds": float(time.monotonic() - eval_start_time),
        "episode_results": episode_results,
    }


def benchmark_supports_n(ideal_class: str, n: int) -> bool:
    if ideal_class == "katsura":
        return n >= 1
    return n >= 2


def enumerate_benchmark_distributions(ideal_dist: str) -> list[str] | None:
    dist_args = split_top_level_dash(ideal_dist.strip())
    if len(dist_args) != 2 or not dist_args[0]:
        return None

    head = dist_args[0]
    if head == "geom":
        return None
    if not (
        head[0].isalpha() or (head[0] == "(" and len(head) > 1 and head[1].isalpha())
    ):
        return None

    classes = parse_class_list_token(head)
    n_lo, n_hi = parse_int_or_range_token(dist_args[1], "n")

    members = [
        f"{ideal_class}-{n}"
        for ideal_class in classes
        for n in range(n_lo, n_hi + 1)
        if benchmark_supports_n(ideal_class, n)
    ]
    if not members:
        raise ValueError(f"No valid benchmark ideals for ideal_dist: {ideal_dist}")
    return members


def count_laman_graphs(n: int) -> int:
    text = laman_graph_path(n).read_text(encoding="utf-8")
    return len(re.findall(r"-?\d+", text))


def geom_support_size(ideal_dist: str) -> int | None:
    dist_args = split_top_level_dash(ideal_dist.strip())
    if len(dist_args) != 2 or dist_args[0] != "geom":
        return None

    n_lo, n_hi = parse_int_or_range_token(dist_args[1], "n")
    if n_lo < 3 or n_hi > 10:
        raise ValueError(
            f"Malformed geom ideal_dist: {ideal_dist}. Expected 3 <= n <= 10"
        )

    return sum(count_laman_graphs(n) for n in range(n_lo, n_hi + 1))


def finite_ideal_support_size(ideal_dist: str | None) -> int | None:
    if ideal_dist is None:
        return 1

    benchmark_members = enumerate_benchmark_distributions(ideal_dist)
    if benchmark_members is not None:
        return len(benchmark_members)

    return geom_support_size(ideal_dist)


def collect_unique_ideal_eval_seeds(
    cfg, target_count: int, *, base_seed: int | None = None
) -> list[int]:
    """Finds seeds that produce unique ideals for finite-support distributions."""
    if target_count <= 0:
        return []

    env = CGLeadMonomialsEnv(
        ideal_dist=cfg.env.ideal_dist,
        elimination=cfg.env.elimination,
        rewards=cfg.env.rewards,
        pq_strategy=cfg.env.priority_queue_strategy,
        sort_input=cfg.env.sort_input,
        sort_reducers=cfg.env.sort_reducers,
        fix_ideal=cfg.env.fix_ideal,
        N=cfg.env.N,
        k=cfg.env.k,
        N_P=cfg.env.N_P,
        N_G=cfg.env.N_G,
    )

    seen: set[tuple[str, ...]] = set()
    seeds: list[int] = []
    max_attempts = max(target_count * 20, target_count + 1_000)
    if base_seed is None:
        base_seed = int(cfg.training.greedy_eval_seed)

    for attempt in range(max_attempts):
        seed = int(base_seed) + attempt
        env.seed(seed)
        _, info = env.reset()
        basis_key = tuple(info.get("basis_elements", []))
        if basis_key in seen:
            continue
        seen.add(basis_key)
        seeds.append(seed)
        if len(seeds) >= target_count:
            break

    if len(seeds) < target_count:
        console_print(
            cfg,
            "Greedy eval could only find "
            f"{len(seeds)} unique ideals out of requested finite support {target_count} "
            f"for ideal_dist={cfg.env.ideal_dist!r}.",
            flush=True,
        )

    return seeds


def build_greedy_eval_seeds(cfg, greedy_eval_enabled: bool) -> list[int]:
    """Returns eval seeds, avoiding duplicate work for finite-support ideal sets."""
    if not greedy_eval_enabled:
        return []

    num_examples = int(cfg.training.greedy_eval_num_examples)
    if num_examples <= 0:
        return []

    if getattr(cfg.training, "greedy_eval_corpus", None) is not None:
        return [row["episode_seed"] for row in load_explicit_greedy_eval_corpus(cfg)]

    if cfg.env.ideal_corpus is not None:
        num_examples = min(num_examples, len(load_ideal_corpus(cfg.env.ideal_corpus).bases))
        return [cfg.training.greedy_eval_seed + i for i in range(num_examples)]

    fixed_ideal_eval = (
        bool(cfg.env.fix_ideal)
        or cfg.env.ideal_dist is None
        or cfg.env.basis_elements is not None
    )
    if fixed_ideal_eval:
        if num_examples > 1:
            console_print(
                cfg,
                "Greedy eval is using a fixed ideal; evaluating one unique seed "
                f"instead of repeating {num_examples} identical episodes.",
                flush=True,
            )
        num_examples = 1

    finite_support = finite_ideal_support_size(cfg.env.ideal_dist)
    if finite_support is not None and not fixed_ideal_eval:
        target_count = min(num_examples, finite_support)
        if target_count < num_examples:
            console_print(
                cfg,
                "Greedy eval is using finite ideal support "
                f"({finite_support} ideals for {cfg.env.ideal_dist!r}); "
                f"evaluating {target_count} unique ideals instead of {num_examples}.",
                flush=True,
            )
        return collect_unique_ideal_eval_seeds(
            cfg,
            target_count,
            base_seed=int(cfg.training.greedy_eval_seed),
        )

    return [cfg.training.greedy_eval_seed + i for i in range(num_examples)]


def _split_symbols(symbols: str) -> list[str]:
    return [tok.strip() for tok in symbols.split(",") if tok.strip()]


def _resolve_corpus_symbols(corpus_rows: list[dict]) -> str:
    if not corpus_rows:
        raise ValueError("Empty greedy eval corpus.")

    parsed_rows: list[list[str]] = []
    widest_symbols: list[str] = []
    for row_idx, row in enumerate(corpus_rows):
        symbols = row.get("symbols")
        if not isinstance(symbols, str) or not symbols.strip():
            raise ValueError(
                f"Greedy eval corpus row {row_idx} is missing a non-empty 'symbols' field."
            )
        row_symbols = _split_symbols(symbols)
        parsed_rows.append(row_symbols)
        if len(row_symbols) > len(widest_symbols):
            widest_symbols = row_symbols.copy()

    resolved = widest_symbols.copy()
    seen = set(resolved)
    for row_symbols in parsed_rows:
        for symbol in row_symbols:
            if symbol not in seen:
                seen.add(symbol)
                resolved.append(symbol)
    return ",".join(resolved)


def _infer_symbols_from_basis_elements(basis_elements: list[str]) -> str:
    seen: set[str] = set()
    ordered_symbols: list[str] = []
    for basis_element in basis_elements:
        for match in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*(?:\[\d+\])?", basis_element):
            symbol = match.group(0)
            if symbol not in seen:
                seen.add(symbol)
                ordered_symbols.append(symbol)
    if not ordered_symbols:
        raise ValueError("Unable to infer any symbols from greedy eval basis elements.")
    return ",".join(ordered_symbols)


def _resolve_symbols_for_eval_corpus(cfg, basis_elements: list[str]) -> str:
    if cfg.env.symbols is not None:
        return cfg.env.symbols
    if cfg.env.ideal_dist is None:
        return _infer_symbols_from_basis_elements(basis_elements)

    dist_args = split_top_level_dash(cfg.env.ideal_dist.strip())
    if len(dist_args) == 2 and dist_args[0] == "geom":
        _, hi = parse_int_or_range_token(dist_args[1], "n")
        symbols = [f"x{i}" for i in range(2, hi + 1)]
        symbols.extend(f"y{i}" for i in range(3, hi + 1))
        return ",".join(symbols)

    n_vars = parse_nvars_from_ideal_dist(cfg.env.ideal_dist)
    return ",".join(f"x[{i}]" for i in range(n_vars))


def build_greedy_eval_corpus(cfg, eval_seeds: list[int]) -> list[dict]:
    """Materializes a fixed ideal corpus so all eval targets share the same examples."""
    if not eval_seeds:
        return []

    if getattr(cfg.training, "greedy_eval_corpus", None) is not None:
        return load_explicit_greedy_eval_corpus(cfg)

    if cfg.env.ideal_corpus is not None:
        corpus = load_ideal_corpus(cfg.env.ideal_corpus)
        if len(eval_seeds) > len(corpus.bases):
            raise ValueError("Greedy evaluation cannot select more ideals than the training corpus.")
        indices = np.random.default_rng(cfg.training.greedy_eval_seed).choice(
            len(corpus.bases), size=len(eval_seeds), replace=False
        )
        return [
            {
                "ideal_id": ideal_id,
                "episode_seed": int(seed),
                "distribution": "ideal_corpus",
                "ideal_corpus_index": int(index),
                "coeff_ring": corpus.coeff_rings[int(index)],
                "symbols": corpus.ring_symbols[int(index)],
                "basis_elements": list(corpus.bases[int(index)]),
            }
            for ideal_id, (index, seed) in enumerate(zip(indices, eval_seeds))
        ]

    env = CGLeadMonomialsEnv(
        ideal_dist=cfg.env.ideal_dist,
        elimination=cfg.env.elimination,
        rewards=cfg.env.rewards,
        pq_strategy=cfg.env.priority_queue_strategy,
        sort_input=cfg.env.sort_input,
        sort_reducers=cfg.env.sort_reducers,
        fix_ideal=cfg.env.fix_ideal,
        N=cfg.env.N,
        k=cfg.env.k,
        N_P=cfg.env.N_P,
        N_G=cfg.env.N_G,
        coeff_ring=cfg.env.coeff_ring,
        symbols=cfg.env.symbols,
        basis=cfg.env.basis_elements,
    )

    generated_fixed_ideal = bool(cfg.env.fix_ideal) and cfg.env.basis_elements is None
    corpus_rows = []
    for ideal_id, seed in enumerate(eval_seeds):
        # Fixed training workers all sample with env_seed. Evaluation seeds still
        # control episode randomness, but must not select a different ideal.
        env.seed(cfg.env.env_seed if generated_fixed_ideal else seed)
        _, info = env.reset()
        basis_elements = list(info.get("basis_elements", []))
        if len(basis_elements) == 0:
            raise ValueError(
                f"Greedy eval corpus row {ideal_id} had no basis_elements for seed {seed}."
            )
        initial_ideal = env.initial_ideal()
        coeff_ring = initial_ideal["coeff_ring"]
        if generated_fixed_ideal:
            symbols = initial_ideal["symbols"]
            basis_elements = initial_ideal["basis_elements"]
        else:
            symbols = _resolve_symbols_for_eval_corpus(cfg, basis_elements)
        corpus_rows.append(
            {
                "ideal_id": ideal_id,
                "episode_seed": int(seed),
                "distribution": cfg.env.ideal_dist,
                "coeff_ring": coeff_ring,
                "symbols": symbols,
                "basis_elements": basis_elements,
            }
        )

    corpus_symbols = _resolve_corpus_symbols(corpus_rows)
    for row in corpus_rows:
        row["symbols"] = corpus_symbols

    return corpus_rows


def _greedy_eval_corpus_cache_root(
    cache_root: str | pathlib.Path | None = None,
) -> pathlib.Path:
    if cache_root is not None:
        return pathlib.Path(cache_root)
    return pathlib.Path(".") / "eval_corpora" / "greedy_eval"


def _greedy_eval_corpus_cache_key(cfg, eval_seeds: list[int]) -> str:
    descriptor = {
        "corpus_format_version": 2,
        "ideal_dist": cfg.env.ideal_dist,
        "base_seed": int(cfg.training.greedy_eval_seed),
        "eval_seeds": [int(seed) for seed in eval_seeds],
        "fix_ideal": bool(cfg.env.fix_ideal),
        "basis_elements": cfg.env.basis_elements,
    }
    if cfg.env.ideal_corpus is not None:
        descriptor["ideal_corpus"] = load_ideal_corpus(cfg.env.ideal_corpus).rows()
    if cfg.env.basis_elements is not None:
        descriptor["coeff_ring"] = cfg.env.coeff_ring
        descriptor["symbols"] = cfg.env.symbols
    elif cfg.env.fix_ideal:
        descriptor["fixed_ideal_env_seed"] = int(cfg.env.env_seed)
    descriptor_json = json.dumps(descriptor, sort_keys=True)
    digest = hashlib.sha1(descriptor_json.encode("utf-8")).hexdigest()[:12]
    ideal_label = (
        f"corpus_{pathlib.Path(cfg.env.ideal_corpus).stem}"
        if cfg.env.ideal_corpus is not None
        else cfg.env.ideal_dist if cfg.env.ideal_dist is not None else "fixed_ideal"
    )
    ideal_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", ideal_label).strip("_") or "ideal"
    ideal_label = ideal_label[:80]
    return (
        f"{ideal_label}__seed_{int(cfg.training.greedy_eval_seed)}"
        f"__n_{len(eval_seeds)}__{digest}"
    )


def greedy_eval_corpus_cache_path(
    cfg,
    eval_seeds: list[int],
    *,
    cache_root: str | pathlib.Path | None = None,
) -> pathlib.Path | None:
    if not eval_seeds:
        return None
    if getattr(cfg.training, "greedy_eval_corpus", None) is not None:
        return pathlib.Path(cfg.training.greedy_eval_corpus)
    root = _greedy_eval_corpus_cache_root(cache_root)
    return root / f"{_greedy_eval_corpus_cache_key(cfg, eval_seeds)}.jsonl"


def load_greedy_eval_corpus(corpus_path: str | pathlib.Path) -> list[dict]:
    rows = []
    with pathlib.Path(corpus_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_greedy_eval_corpus(
    corpus_rows: list[dict],
    corpus_path: str | pathlib.Path,
) -> str | None:
    """Persists the materialized greedy-eval corpus as JSONL for exact replay."""
    if not corpus_rows:
        return None

    path = pathlib.Path(corpus_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in corpus_rows:
            f.write(json.dumps(row))
            f.write("\n")
    return str(path)


def resolve_greedy_eval_corpus(
    cfg,
    eval_seeds: list[int],
    *,
    cache_root: str | pathlib.Path | None = None,
) -> tuple[list[dict], str | None, bool]:
    """Load an explicit or cached corpus, generating only when no file is supplied."""
    if not eval_seeds:
        return [], None, False
    if getattr(cfg.training, "greedy_eval_corpus", None) is not None:
        corpus_rows = load_explicit_greedy_eval_corpus(cfg)
        return corpus_rows, str(cfg.training.greedy_eval_corpus), False
    corpus_path = greedy_eval_corpus_cache_path(cfg, eval_seeds, cache_root=cache_root)
    if corpus_path is None:
        return [], None, False

    expected_seeds = [int(seed) for seed in eval_seeds]
    if corpus_path.exists():
        corpus_rows = load_greedy_eval_corpus(corpus_path)
        loaded_seeds = [int(row.get("episode_seed")) for row in corpus_rows]
        if loaded_seeds != expected_seeds:
            raise ValueError(
                "Cached greedy eval corpus seeds do not match requested eval seeds: "
                f"path={corpus_path}, loaded={loaded_seeds}, expected={expected_seeds}"
            )
        return corpus_rows, str(corpus_path), False

    corpus_rows = build_greedy_eval_corpus(cfg, eval_seeds)
    save_greedy_eval_corpus(corpus_rows, corpus_path)
    return corpus_rows, str(corpus_path), True


def resolve_greedy_eval_targets(cfg) -> dict:
    """Resolves which greedy-eval envs to run from the explicit config toggles."""
    eval_enabled = (
        cfg.training.greedy_eval_interval_steps > 0
        and cfg.training.greedy_eval_num_examples > 0
    )
    if not eval_enabled:
        return {
            "enabled": False,
            "run_train_env": False,
            "run_nopq_env": False,
            "primary_target": None,
            "nopq_gym_id": _NOPQ_GYM_ID,
        }

    run_train_env = bool(cfg.training.greedy_eval_on_train_env)
    requested_run_nopq_env = bool(cfg.training.greedy_eval_on_nopq_env)
    nopq_matches_train_env = _NOPQ_GYM_ID.lower() == cfg.env.gym_id.lower()
    run_nopq_env = requested_run_nopq_env and not nopq_matches_train_env
    if run_train_env and run_nopq_env:
        primary_target = "nopq"
    else:
        primary_target = (
            "train" if run_train_env else ("nopq" if run_nopq_env else None)
        )

    return {
        "enabled": run_train_env or run_nopq_env,
        "run_train_env": run_train_env,
        "run_nopq_env": run_nopq_env,
        "primary_target": primary_target,
        "nopq_gym_id": _NOPQ_GYM_ID,
    }


def compute_greedy_eval_selection_metric(
    train_stats: dict | None,
    nopq_stats: dict | None,
    *,
    run_train_env: bool,
    run_nopq_env: bool,
) -> tuple[float | None, str | None]:
    """Returns the checkpoint-selection metric and a label describing it."""

    def _metric_from_stats(
        stats: dict | None,
        label: str,
    ) -> tuple[float | None, str]:
        if stats is None:
            return None, label
        if int(stats.get("num_truncated_total", 0)) > 0:
            return None, f"{label}_requires_untruncated_eval"
        metric = stats.get("mean_return_completed_only", stats.get("mean_return"))
        if metric is None:
            return None, f"{label}_requires_completed_episodes"
        return float(metric), label

    if run_nopq_env:
        return _metric_from_stats(nopq_stats, "nopq_mean_return")
    if run_train_env:
        return _metric_from_stats(train_stats, "train_mean_return")
    return None, None
