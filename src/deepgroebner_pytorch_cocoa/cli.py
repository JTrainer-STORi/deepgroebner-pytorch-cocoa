from __future__ import annotations

import tyro

from .presets import get_architecture_preset
from .runner import TrainConfig, run_training


def main() -> None:
    cfg = tyro.cli(TrainConfig)
    preset = get_architecture_preset(cfg.architecture)
    print(f"Architecture preset: {preset.name}")
    print(preset.summary)
    run_training(cfg)


if __name__ == "__main__":
    main()
