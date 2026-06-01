from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Literal

import tyro

from .presets import get_architecture_preset


@dataclass
class TrainConfig:
    """Small public training surface for the standalone trainer."""

    architecture: Literal[
        "symmetric-per-entity-mlp",
        "spem-deep-set",
        "spem-deep-set-state-features",
    ] = "spem-deep-set"
    ideal_distribution: str = "3-20-10-weighted"
    total_timesteps: int = 1_000_000
    num_envs: int = 12
    rollout_steps: int = 1024
    seed: int = 24
    priority_queue_strategy: Literal["degree", "normal", "sugar", "first"] = "degree"
    observation_budget: int = 100
    top_monomials: int = 2
    greedy_eval_interval_steps: int = 250_000
    greedy_eval_examples: int = 100
    device: Literal["auto", "cpu", "cuda"] = "auto"
    output_directory: str | None = None
    track: tyro.conf.FlagConversionOff[bool] = False
    verbosity: Literal["quiet", "normal", "verbose"] = "normal"
    save_critic: tyro.conf.FlagConversionOff[bool] = False
    dry_run: tyro.conf.FlagConversionOff[bool] = False


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_training_entrypoint() -> Path:
    configured_root = os.environ.get("DEEPGROEBNER_TRAINER_ROOT")
    repo_root = Path(configured_root).expanduser() if configured_root else _default_repo_root()
    entrypoint = repo_root / "ppo_masked.py"
    if not entrypoint.exists():
        raise FileNotFoundError(
            "Could not find `ppo_masked.py`. "
            "Set DEEPGROEBNER_TRAINER_ROOT to the repository root if the trainer lives elsewhere."
        )
    return entrypoint


def build_training_cli_args(cfg: TrainConfig) -> list[str]:
    preset = get_architecture_preset(cfg.architecture)
    args = [
        "--env.gym_id=BuchbergerExponents-v1",
        f"--env.ideal_dist={cfg.ideal_distribution}",
        f"--env.priority_queue_strategy={cfg.priority_queue_strategy}",
        f"--env.N={cfg.observation_budget}",
        f"--env.k={cfg.top_monomials}",
        f"--training.total_timesteps={cfg.total_timesteps}",
        f"--training.num_envs={cfg.num_envs}",
        f"--training.num_steps_per_update={cfg.rollout_steps}",
        f"--training.training_seed={cfg.seed}",
        f"--training.track={cfg.track}",
        f"--training.console_verbosity={cfg.verbosity}",
        f"--training.cuda={cfg.device != 'cpu'}",
        f"--training.save_critic={cfg.save_critic}",
        f"--training.hparam_eval_interval_steps={cfg.greedy_eval_interval_steps}",
        f"--training.hparam_eval_num_examples={cfg.greedy_eval_examples}",
        "--training.hparam_eval_on_train_env=True",
        "--training.hparam_eval_on_nopq_env=False",
        "--training.counterfactual_diagnostic_frequency=0",
        "--training.gradient_sensitivity_diagnostic_frequency=0",
        "--training.ranking_diagnostic_frequency=0",
    ]
    if cfg.output_directory is not None:
        args.append(f"--training.output_directory={cfg.output_directory}")
    for key, value in preset.trainer_overrides:
        args.append(f"--{key}={value}")
    return args


def build_training_command(cfg: TrainConfig) -> list[str]:
    return [sys.executable, str(resolve_training_entrypoint()), *build_training_cli_args(cfg)]


def format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_training(cfg: TrainConfig) -> list[str]:
    command = build_training_command(cfg)
    if cfg.dry_run:
        print(format_command(command))
        return command
    subprocess.run(command, check=True)
    return command
