"""Evaluate native pair-selection heuristics on a fixed test corpus."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import datetime
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import tyro

from .evaluation_resources import (
    ResourceMeasurement,
    format_resource_summary,
    resource_metadata,
    resource_statistics,
)
from .ideal_corpus import normalize_coefficient_ring, split_ring_symbols
from .wrapped import CNoPQEnv


Heuristic = Literal["first", "degree", "normal", "sugar", "random"]
HEURISTICS = ("first", "degree", "normal", "sugar", "random")
REPORT_FILES = ("episode_results.csv", "episode_results.jsonl", "summary.json", "metadata.json")
MAX_SEED = 2**31 - 1


@dataclass
class HeuristicEvalConfig:
    """Benchmark heuristic costs on each JSONL test corpus entry, without a checkpoint."""

    corpus: str # Path to a JSONL test corpus.
    strategies: tuple[Heuristic, ...] = HEURISTICS # Pair-selection strategies; each sees the full pending pair set.
    elimination: Literal["gebauermoeller", "lcm", "none"] = "gebauermoeller"
    rewards: Literal["additions", "reductions"] = "additions" # Select which positive operation count becomes cost; reward_return is its negative.
    sort_input: tyro.conf.FlagConversionOff[bool] = False
    sort_reducers: tyro.conf.FlagConversionOff[bool] = True
    seed: int | None = None # Override entry seeds with seed + corpus_index * repetitions. None uses episode_seed, falling back to corpus_index * repetitions.
    repetitions: int = 1 # Runs per ideal and strategy; repetition r uses the entry's base seed plus r.
    output_root: str = "eval_results"
    output_dir: str | None = None
    overwrite: tyro.conf.FlagConversionOff[bool] = False # Allow replacing existing reports in output_dir.
    verbosity: Literal["quiet", "normal", "verbose"] = "normal"


@dataclass
class HeuristicEvaluationResult:
    """Aggregate statistics, per-run costs, and locations of saved reports."""

    summary: dict[str, Any]
    episode_results: list[dict[str, Any]]
    output_directory: str
    artifact_paths: dict[str, str]


def _validate_seed(seed: Any) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_SEED:
        raise ValueError(f"Seed must be an integer between 0 and {MAX_SEED}.")
    return seed


def _load_corpus(path: Path, cfg: HeuristicEvalConfig) -> list[dict[str, Any]]:
    """Validate each input independently, preserving its ring and optional metadata."""
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Each ideal must be a JSON object.")
                row["coeff_ring"] = normalize_coefficient_ring(row.get("coeff_ring"))
                row["symbols"] = ",".join(split_ring_symbols(row.get("symbols")))
                basis = row.get("basis_elements")
                if not isinstance(basis, list) or not basis or not all(
                    isinstance(polynomial, str) and polynomial.strip() for polynomial in basis
                ):
                    raise ValueError("basis_elements must be a non-empty list of polynomial strings.")
                row["basis_elements"] = [polynomial.strip() for polynomial in basis]
                base_seed = (
                    cfg.seed + len(rows) * cfg.repetitions
                    if cfg.seed is not None else row.get("episode_seed")
                )
                if base_seed is None:
                    base_seed = len(rows) * cfg.repetitions
                row["evaluation_seed"] = _validate_seed(base_seed)
                _validate_seed(base_seed + cfg.repetitions - 1)
            except ValueError as error:
                raise ValueError(f"Invalid test corpus {path}, line {line_number}: {error}") from error
            rows.append(row)
    if not rows:
        raise ValueError(f"Test corpus {path} must contain at least one ideal.")
    return rows


def _statistics(values: list[int | float]) -> dict[str, int | float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "max": float(array.max()),
    }


def evaluate_heuristics(cfg: HeuristicEvalConfig) -> HeuristicEvaluationResult:
    """Run every requested strategy on every test corpus ideal using native Buchberger.

    Each run starts from the same initialized basis and full pending pair set.
    Counts cover S-pair processing; elapsed time also includes final basis
    interreduction, but excludes input parsing and environment initialization.
    Relative paths resolve against the caller's working directory.
    """
    if not cfg.strategies or any(strategy not in HEURISTICS for strategy in cfg.strategies):
        raise ValueError(f"strategies must be a non-empty selection from {HEURISTICS}.")
    if len(set(cfg.strategies)) != len(cfg.strategies):
        raise ValueError("strategies must not contain duplicates.")
    if isinstance(cfg.repetitions, bool) or not isinstance(cfg.repetitions, int) or cfg.repetitions < 1:
        raise ValueError("repetitions must be a positive integer.")
    if cfg.elimination not in {"gebauermoeller", "lcm", "none"}:
        raise ValueError("Unknown elimination strategy.")
    if cfg.rewards not in {"additions", "reductions"}:
        raise ValueError("rewards must be additions or reductions.")
    if cfg.seed is not None:
        _validate_seed(cfg.seed)

    total_measurement = ResourceMeasurement()
    corpus_path = Path(cfg.corpus).expanduser().resolve()
    corpus_rows = _load_corpus(corpus_path, cfg)
    out_dir = (
        Path(cfg.output_dir).expanduser()
        if cfg.output_dir is not None
        else Path(cfg.output_root).expanduser() / (
            "heuristics__" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
    ).resolve()
    if not cfg.overwrite:
        for filename in REPORT_FILES:
            if (out_dir / filename).exists():
                raise FileExistsError(
                    f"Refusing to overwrite {out_dir / filename}. Pass --overwrite True to replace reports."
                )

    if cfg.verbosity != "quiet":
        print(f"Test corpus: {corpus_path} ({len(corpus_rows)} ideals)")
        print(f"Heuristics: {', '.join(cfg.strategies)}; full pending pair set")

    results = []
    for corpus_index, row in enumerate(corpus_rows):
        try:
            env = CNoPQEnv(
                coeff_ring=row["coeff_ring"], symbols=row["symbols"],
                basis=row["basis_elements"], elimination=cfg.elimination,
                rewards=cfg.rewards, sort_input=cfg.sort_input,
                sort_reducers=cfg.sort_reducers, k=1, return_dict=True,
            )
            observation, _ = env.reset()
            if not len(observation["G"]):
                raise ValueError("The input must contain at least one nonzero generator.")
        except Exception as error:
            raise ValueError(
                f"Could not initialize test corpus {corpus_path}, ideal {corpus_index}: {error}"
            ) from error

        for strategy in cfg.strategies:
            for repetition in range(cfg.repetitions):
                seed = row["evaluation_seed"] + repetition
                measurement = ResourceMeasurement()
                try:
                    stats = env.heuristic_stats(strategy=strategy, seed=seed)
                except Exception as error:
                    raise RuntimeError(
                        f"Heuristic {strategy} failed for test corpus ideal {corpus_index}, "
                        f"repetition {repetition}, seed {seed}: {error}"
                    ) from error
                resources = measurement.finish()
                cost = (
                    stats["polynomial_additions"] if cfg.rewards == "additions" else stats["reductions"]
                )
                results.append({
                    "corpus_index": corpus_index,
                    "ideal_id": row.get("ideal_id", corpus_index),
                    "episode_seed": row.get("episode_seed"),
                    "distribution": row.get("distribution"),
                    "coeff_ring": row["coeff_ring"],
                    "symbols": row["symbols"],
                    "strategy": strategy,
                    "repetition": repetition,
                    "seed": seed,
                    **stats,
                    "cost": cost,
                    **resources,
                })
                if cfg.verbosity == "verbose":
                    print(
                        f"[{corpus_index + 1}/{len(corpus_rows)}] {strategy} "
                        f"run={repetition + 1} additions={stats['polynomial_additions']} "
                        f"reductions={stats['reductions']} cost={cost}; "
                        f"{format_resource_summary(resources)}"
                    )

    summary = {"num_ideals": len(corpus_rows), "num_evals": len(results), "strategies": {}}
    for strategy in cfg.strategies:
        strategy_rows = [row for row in results if row["strategy"] == strategy]
        metrics = {"num_evals": len(strategy_rows)}
        for name in (
            "cost", "reward_return", "polynomial_additions", "reductions",
            "zero_reductions", "nonzero_reductions",
        ):
            for statistic, value in _statistics([row[name] for row in strategy_rows]).items():
                metrics[f"{name}_{statistic}"] = value
        metrics.update(resource_statistics(strategy_rows))
        summary["strategies"][strategy] = metrics
    summary.update(resource_statistics(results))
    summary["resources"] = total_measurement.finish()

    metadata = {
        **asdict(cfg),
        "corpus": str(corpus_path),
        "pair_selection": "full_pending_pair_set",
        "cost_metric": "polynomial_additions" if cfg.rewards == "additions" else "reductions",
        "reward_discount": 1.0,
        "timing": "native_solve_including_final_interreduction_excluding_initialization",
        "resource_measurements": {
            **resource_metadata(),
            "total_timing_scope": (
                "Corpus loading, environment initialization, native solves, and aggregation; "
                "excludes report writing."
            ),
            "episode_timing_scope": (
                "Native solve including final basis interreduction; excludes corpus parsing "
                "and environment initialization."
            ),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "episode_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    with (out_dir / "episode_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    for filename, payload in (("summary.json", summary), ("metadata.json", metadata)):
        with (out_dir / filename).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    if cfg.verbosity != "quiet":
        for strategy, metrics in summary["strategies"].items():
            print(
                f"{strategy}: mean cost={metrics['cost_mean']:.3f}, "
                f"mean reductions={metrics['reductions_mean']:.3f}; "
                f"{format_resource_summary(metrics)}"
            )
        print(f"Total evaluation: {format_resource_summary(summary['resources'])}")
        print(f"Wrote test corpus heuristic evaluation to {out_dir}")
    return HeuristicEvaluationResult(
        summary=summary,
        episode_results=results,
        output_directory=str(out_dir),
        artifact_paths={
            "episode_results_csv": str(out_dir / "episode_results.csv"),
            "episode_results_jsonl": str(out_dir / "episode_results.jsonl"),
            "summary": str(out_dir / "summary.json"),
            "metadata": str(out_dir / "metadata.json"),
        },
    )


def main(args: list[str] | None = None) -> HeuristicEvaluationResult:
    """Parse CLI options and benchmark the requested heuristic strategies."""
    return evaluate_heuristics(tyro.cli(HeuristicEvalConfig, args=args))


if __name__ == "__main__":
    main()
