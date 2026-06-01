from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro

from deepgroebner_pytorch_cocoa.config_dataclasses import Config
from deepgroebner_pytorch_cocoa.runner import load_training_module


@dataclass
class CorpusConfig:
    """Public helper for building fixed greedy-eval corpora."""

    ideal_distribution: str | None = "3-20-10-weighted"
    examples: int = 100
    seed: int = 10_000
    priority_queue_strategy: Literal["degree", "normal", "sugar", "first"] = "degree"
    observation_budget: int = 100
    top_monomials: int = 2
    elimination: Literal["gebauermoeller", "lcm", "none"] = "gebauermoeller"
    rewards: Literal["additions", "reductions"] = "additions"
    coeff_ring: str | None = None
    symbols: str | None = None
    basis_elements: str | None = None
    output_path: str | None = None
    overwrite: tyro.conf.FlagConversionOff[bool] = False


def _build_internal_config(cli_cfg: CorpusConfig) -> Config:
    cfg = Config()
    cfg.env.ideal_dist = cli_cfg.ideal_distribution
    cfg.env.priority_queue_strategy = cli_cfg.priority_queue_strategy
    cfg.env.N = cli_cfg.observation_budget
    cfg.env.k = cli_cfg.top_monomials
    cfg.env.elimination = cli_cfg.elimination
    cfg.env.rewards = cli_cfg.rewards
    cfg.env.coeff_ring = cli_cfg.coeff_ring
    cfg.env.symbols = cli_cfg.symbols
    cfg.env.basis_elements = cli_cfg.basis_elements.split(",") if cli_cfg.basis_elements else None
    cfg.training.hparam_eval_num_examples = cli_cfg.examples
    cfg.training.hparam_eval_seed = cli_cfg.seed
    cfg.training.hparam_eval_interval_steps = 1
    cfg.training.console_verbosity = "normal"
    return cfg


def _load_trainer_module():
    return load_training_module()


def main() -> None:
    cli_cfg = tyro.cli(CorpusConfig)
    cfg = _build_internal_config(cli_cfg)
    trainer_module = _load_trainer_module()
    eval_seeds = trainer_module.build_greedy_eval_seeds(cfg, greedy_eval_enabled=True)

    if not eval_seeds:
        raise ValueError("No eval seeds were produced. Set `examples` to a positive integer.")

    if cli_cfg.output_path is not None:
        output_path = Path(cli_cfg.output_path)
        if output_path.exists() and not cli_cfg.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing corpus file: {output_path}. "
                "Pass --overwrite True to replace it."
            )
        corpus_rows = trainer_module.build_greedy_eval_corpus(cfg, eval_seeds)
        trainer_module.save_greedy_eval_corpus(corpus_rows, output_path)
        print(f"Wrote greedy eval corpus with {len(corpus_rows)} ideals to {output_path}")
        return

    corpus_rows, corpus_path, generated = trainer_module.resolve_greedy_eval_corpus(cfg, eval_seeds)
    status = "Generated" if generated else "Loaded"
    print(f"{status} greedy eval corpus with {len(corpus_rows)} ideals at {corpus_path}")


if __name__ == "__main__":
    main()
