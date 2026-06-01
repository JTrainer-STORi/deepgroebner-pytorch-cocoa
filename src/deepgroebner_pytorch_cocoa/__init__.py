"""Public package for the standalone DeepGroebner PyTorch + CoCoALib release."""

from gymnasium.envs.registration import register

from .presets import ARCHITECTURE_PRESETS, ArchitecturePreset, get_architecture_preset
from .runner import TrainConfig, build_training_cli_args, build_training_command


register(
    id="BuchbergerExponents-v1",
    entry_point="deepgroebner_pytorch_cocoa.buchberger_env:BuchbergerEnvExponentsv1",
)

register(
    id="BuchbergerExponentsNoPQ-v1",
    entry_point="deepgroebner_pytorch_cocoa.buchberger_env:BuchbergerEnvExponentsNoPQv1",
)

__all__ = [
    "ARCHITECTURE_PRESETS",
    "ArchitecturePreset",
    "TrainConfig",
    "build_training_cli_args",
    "build_training_command",
    "get_architecture_preset",
]
