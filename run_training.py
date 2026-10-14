"""Train a PPO policy with the package's configuration defaults and overrides."""

from __future__ import annotations

import tyro

from deepgroebner_pytorch_cocoa.config_dataclasses import Config
from deepgroebner_pytorch_cocoa.ppo_training import TrainingResult, train


def main(args: list[str] | None = None) -> TrainingResult:
    """Parse training options and run PPO in this process."""
    return train(tyro.cli(Config, args=args))


if __name__ == "__main__":
    main()
