"""Quick installation checks, available from any working directory."""

from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Literal

import tyro


@dataclass
class VerifyConfig:
    """Verify the native environments and PyTorch models without training."""

    verbosity: Literal["normal", "quiet"] = "normal"


def _verify_models(obs: dict) -> None:
    import torch

    from .combined_networks import CombinedPolicyNetwork, CombinedValueNetwork
    from .config_dataclasses import ActorConfig, CriticConfig
    from .model_input_utils import build_model_inputs_from_single_obs

    actor_cfg = ActorConfig()
    critic_cfg = CriticConfig()
    for cfg in (actor_cfg, critic_cfg):
        cfg.state_embedding_head_in_dim = obs["G"].shape[-1]
        cfg.state_features_embedding_head_in_dim = obs["state_features"].shape[-1]
    actor_cfg.pair_embedding_head_in_dim = obs["P"].shape[-1]

    inputs = build_model_inputs_from_single_obs(obs, torch.device("cpu"))
    with torch.inference_mode():
        logits = CombinedPolicyNetwork(actor_cfg).eval()(inputs)
        values, _, _ = CombinedValueNetwork(critic_cfg).eval()(inputs)
    if not torch.isfinite(logits[inputs.P_mask]).all() or not torch.isfinite(values).all():
        raise RuntimeError("PyTorch actor/critic produced non-finite outputs.")


def run_verification() -> list[str]:
    """Exercise installed package components using small, fixed input ideals."""
    import gymnasium as gym
    import numpy as np

    import deepgroebner_pytorch_cocoa
    import deepgroebner_pytorch_cocoa.wrapped as wrapped

    messages = [
        f"OK package import: {deepgroebner_pytorch_cocoa.__name__}",
        f"OK compiled extension import: {wrapped.__name__}",
    ]
    for gym_id in ("BuchbergerExponents-v1", "BuchbergerExponentsNoPQ-v1"):
        env = gym.make(
            gym_id,
            # NoPQ's sequence masks trigger Gym's scalar-to-array warnings.
            disable_env_checker=True,
            coeff_ring="QQ",
            symbols="x,y",
            basis_elements=["x^2-y", "x*y-1"],
            elimination="none",
        )
        try:
            obs, _ = env.reset(seed=123)
            valid = np.flatnonzero(obs["P_mask"])
            if valid.size == 0:
                raise RuntimeError(f"{gym_id} reset produced no valid actions.")
            _verify_models(obs)
            messages.append(f"OK PyTorch actor/critic (CPU): {gym_id}")
            _, reward, terminated, truncated, _ = env.step(int(valid[0]))
            if not np.isfinite(reward) or truncated:
                raise RuntimeError(f"{gym_id} step returned an invalid reward or truncated.")
            messages.append(
                f"OK {gym_id} reset/step: reward={float(reward)} "
                f"terminated={bool(terminated)} truncated={bool(truncated)}"
            )
        finally:
            env.close()
    return messages


def main(args: list[str] | None = None) -> None:
    cfg = tyro.cli(VerifyConfig, args=args)
    try:
        messages = run_verification()
    except Exception as exc:
        print(f"Installation verification failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    if cfg.verbosity != "quiet":
        for message in messages:
            print(message)
    print("Installation verification succeeded.")


if __name__ == "__main__":
    main()
