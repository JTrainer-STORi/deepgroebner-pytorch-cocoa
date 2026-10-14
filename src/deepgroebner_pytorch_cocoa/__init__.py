"""Public package for the standalone DeepGroebner PyTorch + CoCoALib release."""

from gymnasium.envs.registration import register

register(
    id="BuchbergerExponents-v1",
    entry_point="deepgroebner_pytorch_cocoa.buchberger_env:BuchbergerEnvExponentsv1",
)

register(
    id="BuchbergerExponentsNoPQ-v1",
    entry_point="deepgroebner_pytorch_cocoa.buchberger_env:BuchbergerEnvExponentsNoPQv1",
)
