from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import tyro

from .runner import TrainConfig, build_training_command, format_command


@dataclass
class VerifyConfig:
    """Verify that the public package and its native environments are usable."""

    verbosity: str = "normal"


def _print(cfg: VerifyConfig, message: str) -> None:
    if cfg.verbosity != "quiet":
        print(message)


def _find_first_valid_action(obs: dict[str, Any]) -> int:
    valid = np.flatnonzero(np.asarray(obs["P_mask"], dtype=bool))
    if valid.size == 0:
        raise ValueError("BuchbergerExponents-v1 reset produced no valid actions.")
    return int(valid[0])


def run_verification(cfg: VerifyConfig) -> list[str]:
    messages: list[str] = []

    import deepgroebner_pytorch_cocoa

    messages.append(f"OK package import: {deepgroebner_pytorch_cocoa.__name__}")

    import deepgroebner_pytorch_cocoa.wrapped as wrapped

    messages.append(f"OK compiled extension import: {wrapped.__name__}")

    pq_env = gym.make("BuchbergerExponents-v1")
    pq_obs, _ = pq_env.reset(seed=123)
    pq_action = _find_first_valid_action(pq_obs)
    _, pq_reward, pq_terminated, pq_truncated, _ = pq_env.step(pq_action)
    pq_env.close()
    messages.append(
        "OK BuchbergerExponents-v1 reset/step: "
        f"reward={float(pq_reward)} terminated={bool(pq_terminated)} truncated={bool(pq_truncated)}"
    )

    nopq_env = gym.make("BuchbergerExponentsNoPQ-v1")
    _, _ = nopq_env.reset(seed=123)
    _, nopq_reward, nopq_terminated, nopq_truncated, _ = nopq_env.step(0)
    nopq_env.close()
    messages.append(
        "OK BuchbergerExponentsNoPQ-v1 reset/step: "
        f"reward={float(nopq_reward)} terminated={bool(nopq_terminated)} truncated={bool(nopq_truncated)}"
    )

    train_command = build_training_command(
        TrainConfig(
            dry_run=True,
            track=False,
            total_timesteps=5_000,
        )
    )
    messages.append(f"OK public train command: {format_command(train_command)}")

    return messages


def main() -> None:
    cfg = tyro.cli(VerifyConfig)
    messages = run_verification(cfg)
    for message in messages:
        _print(cfg, message)
    print("Installation verification succeeded.")


if __name__ == "__main__":
    main()
