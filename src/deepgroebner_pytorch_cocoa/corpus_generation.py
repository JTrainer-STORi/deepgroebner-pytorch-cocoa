"""Generate ideal corpora for greedy evaluation or testing after training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import tyro

from .config_dataclasses import Config
from .greedy_eval import (
    build_greedy_eval_seeds,
    build_greedy_eval_corpus,
    save_greedy_eval_corpus,
    resolve_greedy_eval_corpus,
)


@dataclass
class CorpusConfig:
    """Generate an ideal corpus for greedy evaluation or post-training testing."""

    ideal_distribution: str | None = "3-20-10-weighted"
    examples: int = 100
    seed: int = 10_000
    elimination: Literal["gebauermoeller", "lcm", "none"] = "gebauermoeller"
    rewards: Literal["additions", "reductions"] = "additions"
    coeff_ring: str | None = None
    symbols: str | None = None
    basis_elements: str | None = None
    output_path: str | None = None # JSONL destination, e.g. test_corpora/ideals.jsonl for testing after training. None uses the training evaluation cache.
    overwrite: tyro.conf.FlagConversionOff[bool] = False


@dataclass
class CorpusGenerationResult:
    """Materialized corpus and its location, including whether it was generated."""

    corpus_rows: list[dict[str, Any]]
    corpus_path: str
    generated: bool


def _build_internal_config(cli_cfg: CorpusConfig) -> Config:
    cfg = Config()
    cfg.env.ideal_dist = cli_cfg.ideal_distribution
    cfg.env.elimination = cli_cfg.elimination
    cfg.env.rewards = cli_cfg.rewards
    cfg.env.coeff_ring = cli_cfg.coeff_ring
    cfg.env.symbols = cli_cfg.symbols
    cfg.env.basis_elements = (
        cli_cfg.basis_elements.split(",") if cli_cfg.basis_elements else None
    )
    cfg.training.greedy_eval_num_examples = cli_cfg.examples
    cfg.training.greedy_eval_seed = cli_cfg.seed
    cfg.training.greedy_eval_interval_steps = 1
    cfg.training.console_verbosity = "normal"
    return cfg


def generate_corpus(cfg: CorpusConfig) -> CorpusGenerationResult:
    """Generate or load a corpus and return its rows and absolute file path.

    Use an explicit output path such as ``test_corpora/ideals.jsonl`` for a
    separate test corpus. Explicit paths honor ``overwrite``. Without one,
    matching evaluation corpora are reused from the training evaluation cache.
    """
    internal_cfg = _build_internal_config(cfg)
    eval_seeds = build_greedy_eval_seeds(internal_cfg, greedy_eval_enabled=True)

    if not eval_seeds:
        raise ValueError(
            "No corpus seeds were produced. Set `examples` to a positive integer."
        )

    if cfg.output_path is not None:
        output_path = Path(cfg.output_path).expanduser()
        if output_path.exists() and not cfg.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing corpus file: {output_path}. "
                "Pass --overwrite True to replace it."
            )
        corpus_rows = build_greedy_eval_corpus(internal_cfg, eval_seeds)
        save_greedy_eval_corpus(corpus_rows, output_path)
        print(
            f"Wrote ideal corpus with {len(corpus_rows)} ideals to {output_path}"
        )
        return CorpusGenerationResult(
            corpus_rows=corpus_rows,
            corpus_path=str(output_path.resolve()),
            generated=True,
        )

    corpus_rows, corpus_path, generated = resolve_greedy_eval_corpus(
        internal_cfg, eval_seeds
    )
    status = "Generated" if generated else "Loaded"
    print(
        f"{status} evaluation corpus with {len(corpus_rows)} ideals at {corpus_path}"
    )
    return CorpusGenerationResult(
        corpus_rows=corpus_rows,
        corpus_path=str(Path(corpus_path).resolve()),
        generated=generated,
    )


def main(args: list[str] | None = None) -> CorpusGenerationResult:
    """Parse command-line options and generate or load the requested corpus."""
    return generate_corpus(tyro.cli(CorpusConfig, args=args))


if __name__ == "__main__":
    main()
