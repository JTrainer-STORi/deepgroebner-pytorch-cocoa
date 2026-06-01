from __future__ import annotations

from dataclasses import dataclass, fields
import csv
import datetime
import importlib.util
import json
from pathlib import Path
import random
from typing import Any, Literal

import gymnasium as gym
import numpy as np
import torch
import tyro

from .combined_networks import CombinedPolicyNetwork, checkpoint_uses_legacy_pair_scoring_head
from .config_dataclasses import ActorConfig, EnvConfig, ModelInputs
from .runner import resolve_training_entrypoint


DEFAULT_STATE_FEATURE_NAMES = [
    "G_cardinality",
    "P_cardinality",
    "pair_density",
    "mean_degree",
    "max_degree",
    "min_degree",
    "st_dev_degree",
    "mean_lcm_degree",
    "max_lcm_degree",
    "min_lcm_degree",
    "std_dev_lcm_degree",
    "n_pairs_pruned",
    "n_pairs_processed",
    "steps_since_basis_update",
]

BEST_GREEDY_EVAL_SUFFIX = "__best_greedy_eval"
CHECKPOINT_TYPES = ("auto", "best_greedy_eval", "final")


@dataclass
class EvalCorpusConfig:
    """Evaluate a saved actor checkpoint on a fixed ideal corpus."""

    corpus: str
    model_name: str
    checkpoint_type: Literal["auto", "best_greedy_eval", "final"] = "auto"
    models_dir: str = "models"
    config_path: str | None = None
    wandb_dir: str = "wandb"
    output_root: str = "eval_results"
    output_dir: str | None = None
    gym_id: str | None = "BuchbergerExponentsNoPQ-v1"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    legacy_actor_network: tyro.conf.FlagConversionOff[bool] = False
    verbosity: Literal["quiet", "normal", "verbose"] = "normal"


def _console_print(cfg: EvalCorpusConfig, *parts: object, level: Literal["normal", "verbose"] = "normal") -> None:
    if cfg.verbosity == "quiet":
        return
    if level == "verbose" and cfg.verbosity != "verbose":
        return
    print(*parts)


def _load_trainer_module():
    entrypoint = resolve_training_entrypoint()
    spec = importlib.util.spec_from_file_location("deepgroebner_pytorch_cocoa._trainer_module", entrypoint)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load trainer module from {entrypoint}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _space_last_dim(space: Any) -> int:
    if hasattr(space, "shape") and space.shape is not None:
        return int(space.shape[-1])
    if hasattr(space, "feature_space") and hasattr(space.feature_space, "shape"):
        return int(space.feature_space.shape[-1])
    raise ValueError(f"Unable to infer last dimension from space: {space}")


def _polynomial_term_input_dim(env: gym.Env) -> int | None:
    obs_space = env.observation_space
    spaces = getattr(obs_space, "spaces", obs_space)
    if "G_terms" not in spaces:
        return None
    dim = _space_last_dim(spaces["G_terms"])
    if "G_term_coeffs" in spaces:
        dim += 1
    return dim


def _obs_to_model_inputs(obs: dict[str, Any], device: torch.device, trainer_module) -> ModelInputs:
    return trainer_module.build_model_inputs_from_obs(obs, device)


def _split_symbols(symbols: str) -> list[str]:
    return [tok.strip() for tok in symbols.split(",") if tok.strip()]


def _resolve_corpus_symbols(corpus_rows: list[dict[str, Any]]) -> str:
    if not corpus_rows:
        raise ValueError("Empty corpus")

    parsed_rows: list[list[str]] = []
    widest_symbols: list[str] = []
    for row_idx, row in enumerate(corpus_rows):
        symbols = row.get("symbols")
        if not isinstance(symbols, str) or not symbols.strip():
            raise ValueError(f"Corpus row {row_idx} is missing a non-empty 'symbols' field.")
        row_symbols = _split_symbols(symbols)
        parsed_rows.append(row_symbols)
        if len(row_symbols) > len(widest_symbols):
            widest_symbols = row_symbols.copy()

    if not widest_symbols:
        raise ValueError("Corpus does not contain any symbols.")

    resolved = widest_symbols.copy()
    seen = set(resolved)
    for row_symbols in parsed_rows:
        for symbol in row_symbols:
            if symbol not in seen:
                seen.add(symbol)
                resolved.append(symbol)
    return ",".join(resolved)


def _normalize_gym_id(gym_id: str | None) -> str | None:
    aliases = {
        "BuchbergerExponents": "BuchbergerExponents-v1",
        "BuchbergerExponentsNoPQ": "BuchbergerExponentsNoPQ-v1",
    }
    if gym_id is None:
        return None
    return aliases.get(gym_id, gym_id)


def _unwrap_config_section(config: dict[str, Any], key: str) -> dict[str, Any]:
    section = config[key]
    if isinstance(section, dict) and "value" in section and isinstance(section["value"], dict):
        return section["value"]
    if isinstance(section, dict):
        return section
    raise TypeError(f"Config section {key!r} is not a dict: {type(section).__name__}")


def _get_state_feature_names(env: gym.Env) -> list[str]:
    names = getattr(getattr(env, "unwrapped", env), "state_feature_names", None)
    if names is None:
        return DEFAULT_STATE_FEATURE_NAMES.copy()
    return list(names)


def _extract_state_feature(obs: dict[str, Any], feature_names: list[str], feature_name: str) -> float:
    if feature_name not in feature_names:
        raise KeyError(f"State feature {feature_name!r} not found in feature names: {feature_names}")
    state_features = np.asarray(obs["state_features"], dtype=np.float32).reshape(-1)
    idx = feature_names.index(feature_name)
    if idx >= state_features.shape[0]:
        raise IndexError(
            f"State feature index {idx} for {feature_name!r} is out of bounds for "
            f"state_features with shape {state_features.shape}"
        )
    return float(state_features[idx])


def _compute_queue_usage_stats(pair_counts: list[float], pq_budget: int | None) -> dict[str, Any]:
    if pair_counts:
        pair_counts_arr = np.asarray(pair_counts, dtype=np.float64)
        stats: dict[str, Any] = {
            "pair_count_mean": float(pair_counts_arr.mean()),
            "pair_count_median": float(np.median(pair_counts_arr)),
            "pair_count_min": float(pair_counts_arr.min()),
            "pair_count_max": float(pair_counts_arr.max()),
        }
    else:
        pair_counts_arr = np.asarray([], dtype=np.float64)
        stats = {
            "pair_count_mean": 0.0,
            "pair_count_median": 0.0,
            "pair_count_min": 0.0,
            "pair_count_max": 0.0,
        }

    if pq_budget is None:
        stats.update(
            {
                "pq_budget": None,
                "pq_binding_step_fraction": None,
                "pq_binding_ever": None,
                "pair_overflow_mean": None,
                "pair_overflow_max": None,
            }
        )
        return stats

    overflow = np.maximum(pair_counts_arr - float(pq_budget), 0.0)
    binding = overflow > 0.0
    stats.update(
        {
            "pq_budget": int(pq_budget),
            "pq_binding_step_fraction": float(binding.mean()) if binding.size else 0.0,
            "pq_binding_ever": bool(binding.any()) if binding.size else False,
            "pair_overflow_mean": float(overflow.mean()) if overflow.size else 0.0,
            "pair_overflow_max": float(overflow.max()) if overflow.size else 0.0,
        }
    )
    return stats


def _summarize_numeric_values(values: list[Any]) -> dict[str, float] | None:
    clean = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if clean.size == 0:
        return None
    return {
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=0)),
        "min": float(clean.min()),
        "median": float(np.median(clean)),
        "max": float(clean.max()),
    }


def _build_queue_usage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for col in [
        "pair_count_mean",
        "pair_count_median",
        "pair_count_min",
        "pair_count_max",
        "pq_binding_step_fraction",
        "pair_overflow_mean",
        "pair_overflow_max",
    ]:
        col_summary = _summarize_numeric_values([row.get(col) for row in rows])
        if col_summary is not None:
            summary[col] = col_summary

    budgets = sorted({int(budget) for budget in [row.get("pq_budget") for row in rows] if budget is not None})
    if len(budgets) == 1:
        summary["pq_budget"] = budgets[0]

    binding_values = [row.get("pq_binding_ever") for row in rows if row.get("pq_binding_ever") is not None]
    if binding_values:
        binding_arr = np.asarray(binding_values, dtype=np.float64)
        summary["pq_binding_ever_rate"] = float(binding_arr.mean())
        summary["pq_binding_ever_count"] = int(binding_arr.sum())
        summary["pq_binding_ever_total"] = int(binding_arr.shape[0])

    return summary


def _is_best_greedy_eval_checkpoint(model_path: Path) -> bool:
    return model_path.stem.endswith(BEST_GREEDY_EVAL_SUFFIX)


def _resolve_model_path(model_name: str, models_dir: Path, checkpoint_type: str = "auto") -> Path:
    path = Path(model_name).expanduser()
    if path.is_file():
        return path.resolve()

    candidates = list(models_dir.expanduser().glob(f"**/*{model_name}*.pt"))
    if checkpoint_type == "best_greedy_eval":
        candidates = [candidate for candidate in candidates if _is_best_greedy_eval_checkpoint(candidate)]
    elif checkpoint_type == "final":
        candidates = [candidate for candidate in candidates if not _is_best_greedy_eval_checkpoint(candidate)]
    elif checkpoint_type != "auto":
        raise ValueError(f"Unknown checkpoint_type={checkpoint_type!r}")

    if not candidates:
        raise FileNotFoundError(
            f"No model path found for {model_name!r} with checkpoint_type={checkpoint_type!r}"
        )
    if len(candidates) > 1:
        candidate_names = ", ".join(sorted(path.name for path in candidates[:5]))
        if len(candidates) > 5:
            candidate_names += ", ..."
        raise ValueError(
            "Multiple model matches found; pass a full path or a more specific "
            f"--checkpoint-type. Matches: {candidate_names}"
        )
    return candidates[0].resolve()


def _extract_run_id(model_path: Path) -> str | None:
    parts = model_path.stem.split("__")
    if not parts:
        return None
    if parts[-1] == "best_greedy_eval":
        return parts[-2] if len(parts) >= 2 else None
    return parts[-1]


def _resolve_config_path(
    config_path: str | None,
    model_path: Path,
    wandb_dir: Path,
    run_id: str | None,
) -> Path:
    if config_path:
        path = Path(config_path).expanduser()
        if path.is_file():
            return path.resolve()
        json_path = path.with_suffix(".json")
        if json_path.is_file():
            return json_path.resolve()
        raise FileNotFoundError(f"Config file not found: {path} or {json_path}")

    colocated_json_candidates = [
        model_path.with_name(f"{model_path.stem}__config.json"),
        model_path.with_suffix(".json"),
    ]
    for candidate in colocated_json_candidates:
        if candidate.is_file():
            return candidate.resolve()

    if run_id is None:
        raise ValueError("Could not infer run id from model path; pass --config-path")

    matches = sorted(wandb_dir.expanduser().glob(f"**/*{run_id}*/files/config.yaml"))
    if not matches:
        raise FileNotFoundError(
            "No local config snapshot found next to the checkpoint and no matching "
            f"wandb config.yaml found for run id {run_id}. Pass --config-path explicitly."
        )
    return matches[0].resolve()


def _load_config_file(config_path: Path) -> dict[str, Any]:
    suffix = config_path.suffix.lower()
    with config_path.open("r", encoding="utf-8") as handle:
        if suffix == ".json":
            return json.load(handle)
        if suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise ImportError(
                    "Loading YAML configs requires PyYAML. "
                    "Install it or use the JSON sidecar config saved next to checkpoints."
                ) from exc
            return yaml.safe_load(handle)
    raise ValueError(f"Unsupported config format: {config_path}")


def _load_actor(
    actor_cfg_dict: dict[str, Any],
    env: gym.Env,
    model_path: Path,
    device: torch.device,
    legacy_actor_network: bool = False,
) -> CombinedPolicyNetwork:
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict):
        actor_state = {k.replace("actor.", "", 1): v for k, v in state.items() if k.startswith("actor.")}
        if actor_state:
            state = actor_state

    valid_init_fields = {field.name for field in fields(ActorConfig) if field.init}
    filtered_cfg = {key: value for key, value in actor_cfg_dict.items() if key in valid_init_fields}
    actor_cfg = ActorConfig(**filtered_cfg)
    actor_cfg.state_embedding_head_in_dim = _space_last_dim(env.observation_space["G"])
    actor_cfg.state_features_embedding_head_in_dim = _space_last_dim(env.observation_space["state_features"])
    actor_cfg.polynomial_term_in_dim = _polynomial_term_input_dim(env)
    actor_cfg.pair_embedding_head_in_dim = _space_last_dim(env.observation_space["P"])

    uses_pair_scoring_head = (
        actor_cfg.pair_combiner == "pair-only"
        or (
            actor_cfg.pair_combiner != "basis-cross-attn"
            and actor_cfg.state_embedding_head is None
            and actor_cfg.state_features_embedding_head is None
            and not actor_cfg.share_polynomial_encoder_between_heads
        )
    )
    use_legacy_pair_scoring_head = (
        uses_pair_scoring_head
        and isinstance(state, dict)
        and (legacy_actor_network or checkpoint_uses_legacy_pair_scoring_head(state))
    )

    actor = CombinedPolicyNetwork(
        actor_cfg,
        legacy_pair_scoring_head=use_legacy_pair_scoring_head,
    ).to(device)
    actor.load_state_dict(state)
    actor.eval()
    return actor


def _load_corpus_rows(corpus_path: Path) -> list[dict[str, Any]]:
    rows = []
    with corpus_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError("Empty corpus")
    return rows


def _build_env_kwargs(env_cfg: EnvConfig, corpus_symbols: str, row: dict[str, Any]) -> dict[str, Any]:
    env_kwargs = {
        "ideal_dist": None,
        "elimination": env_cfg.elimination,
        "rewards": env_cfg.rewards,
        "sort_input": env_cfg.sort_input,
        "sort_reducers": env_cfg.sort_reducers,
        "fix_ideal": True,
        "k": env_cfg.k,
        "coeff_ring": row["coeff_ring"],
        "symbols": corpus_symbols,
        "basis_elements": row["basis_elements"],
    }
    if env_cfg.gym_id != "BuchbergerExponentsNoPQ-v1":
        env_kwargs["pq_strategy"] = env_cfg.priority_queue_strategy
        env_kwargs["N"] = env_cfg.N
        env_kwargs["N_P"] = env_cfg.N_P
        env_kwargs["N_G"] = env_cfg.N_G
    return env_kwargs


def _write_episode_results(rows: list[dict[str, Any]], out_dir: Path) -> None:
    csv_path = out_dir / "episode_results.csv"
    jsonl_path = out_dir / "episode_results.jsonl"
    fieldnames = list(rows[0].keys())

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def _build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    returns = np.asarray([row["return"] for row in rows], dtype=np.float64)
    lengths = np.asarray([row["length"] for row in rows], dtype=np.float64)
    return {
        "num_evals": int(len(rows)),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std(ddof=0)),
        "return_min": float(returns.min()),
        "return_median": float(np.median(returns)),
        "return_max": float(returns.max()),
        "length_mean": float(lengths.mean()),
        "length_std": float(lengths.std(ddof=0)),
        "length_min": float(lengths.min()),
        "length_median": float(np.median(lengths)),
        "length_max": float(lengths.max()),
        "queue_usage": _build_queue_usage_summary(rows),
    }


def main() -> None:
    cli_cfg = tyro.cli(EvalCorpusConfig)
    trainer_module = _load_trainer_module()
    device = torch.device(
        "cuda" if (cli_cfg.device == "auto" and torch.cuda.is_available()) else cli_cfg.device
        if cli_cfg.device != "auto" else "cpu"
    )

    model_path = _resolve_model_path(
        cli_cfg.model_name,
        Path(cli_cfg.models_dir),
        checkpoint_type=cli_cfg.checkpoint_type,
    )
    run_id = _extract_run_id(model_path)
    config_path = _resolve_config_path(
        cli_cfg.config_path,
        model_path,
        Path(cli_cfg.wandb_dir),
        run_id,
    )
    train_cfg = _load_config_file(config_path)

    env_cfg = EnvConfig(**_unwrap_config_section(train_cfg, "env"))
    env_cfg.gym_id = _normalize_gym_id(cli_cfg.gym_id or env_cfg.gym_id)

    corpus_path = Path(cli_cfg.corpus).expanduser().resolve()
    corpus_rows = _load_corpus_rows(corpus_path)
    corpus_symbols = _resolve_corpus_symbols(corpus_rows)

    _console_print(cli_cfg, f"Model checkpoint: {model_path}")
    _console_print(cli_cfg, f"Config snapshot: {config_path}")
    _console_print(cli_cfg, f"Corpus: {corpus_path} ({len(corpus_rows)} ideals)")
    _console_print(cli_cfg, f"Evaluation gym id: {env_cfg.gym_id}")

    base_env = gym.make(env_cfg.gym_id, **_build_env_kwargs(env_cfg, corpus_symbols, corpus_rows[0]))
    actor = _load_actor(
        _unwrap_config_section(train_cfg, "actor"),
        base_env,
        model_path,
        device,
        legacy_actor_network=cli_cfg.legacy_actor_network,
    )
    state_feature_names = _get_state_feature_names(base_env)
    pq_budget = None
    if env_cfg.gym_id != "BuchbergerExponentsNoPQ-v1":
        pq_budget = int(env_cfg.N_P if env_cfg.N_P is not None else env_cfg.N)

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for row_idx, row in enumerate(corpus_rows):
            env = gym.make(env_cfg.gym_id, **_build_env_kwargs(env_cfg, corpus_symbols, row))
            obs, _ = env.reset(seed=row.get("episode_seed"))
            done = False
            total_return = 0.0
            steps = 0
            pair_counts: list[float] = []

            while not done:
                pair_counts.append(_extract_state_feature(obs, state_feature_names, "P_cardinality"))
                inputs = _obs_to_model_inputs(obs, device, trainer_module)
                logits = actor(inputs).squeeze(-1)
                probs = trainer_module.CategoricalMasked(logits=logits, masks=inputs.P_mask).probs
                action = int(torch.argmax(probs, dim=-1).item())
                obs, reward, terminated, truncated, _ = env.step(np.int32(action))
                done = bool(terminated or truncated)
                total_return -= float(reward)
                steps += 1

            result_row = {
                "ideal_id": row.get("ideal_id"),
                "episode_seed": row.get("episode_seed"),
                "model_name": model_path.stem,
                "distribution": row.get("distribution"),
                "return": float(total_return),
                "length": int(steps),
                **_compute_queue_usage_stats(pair_counts, pq_budget),
            }
            rows.append(result_row)
            _console_print(
                cli_cfg,
                f"[{row_idx + 1}/{len(corpus_rows)}] return={result_row['return']:.3f} "
                f"length={result_row['length']}",
                level="verbose",
            )

    if cli_cfg.output_dir:
        out_dir = Path(cli_cfg.output_dir).expanduser()
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(cli_cfg.output_root).expanduser() / (
            f"eval_corpus__{env_cfg.gym_id}__{timestamp}__{model_path.stem}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_episode_results(rows, out_dir)

    summary = _build_summary(rows)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    metadata = {
        "model_path": str(model_path),
        "config_path": str(config_path),
        "corpus": str(corpus_path),
        "gym_id": env_cfg.gym_id,
        "device": str(device),
        "checkpoint_type": cli_cfg.checkpoint_type,
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    if cli_cfg.verbosity == "quiet":
        print(f"Wrote corpus evaluation to {out_dir}")
    else:
        print(f"Wrote corpus evaluation to {out_dir}")
        print(f"Mean return: {summary['return_mean']:.3f}")


if __name__ == "__main__":
    main()
