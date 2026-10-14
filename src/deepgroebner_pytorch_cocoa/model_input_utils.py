"""Functions for converting from an environment observation to a suitable model input"""

from __future__ import annotations

import numpy as np
import torch

from deepgroebner_pytorch_cocoa.config_dataclasses import ModelInputs


def build_model_inputs_from_single_obs(obs: dict, device: torch.device) -> ModelInputs:
    """Converts a single observation dict into batched `ModelInputs` on `device`."""

    return ModelInputs(
        P=torch.from_numpy(np.asarray(obs["P"]))
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32),

        G=torch.from_numpy(np.asarray(obs["G"]))
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32),

        G_mask=torch.from_numpy(np.asarray(obs["G_mask"]))
        .unsqueeze(0)
        .to(device=device, dtype=torch.bool),

        P_mask=torch.from_numpy(np.asarray(obs["P_mask"]))
        .unsqueeze(0)
        .to(device=device, dtype=torch.bool),

        state_features=torch.from_numpy(np.asarray(obs["state_features"]))
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32),
    )


def copy_obs_dict_to_model_inputs(obs: dict, dst: ModelInputs):
    """Copy NumPy observation arrays into preallocated ModelInputs tensors."""
    dst.P.copy_(torch.from_numpy(obs["P"]))
    dst.G.copy_(torch.from_numpy(obs["G"]))
    dst.G_mask.copy_(torch.from_numpy(obs["G_mask"]))
    dst.P_mask.copy_(torch.from_numpy(obs["P_mask"]))
    dst.state_features.copy_(torch.from_numpy(obs["state_features"]))


def bind_obs_dict_as_model_inputs_views(obs: dict) -> ModelInputs:
    """Create ModelInputs as zero-copy torch views over NumPy observation arrays (CPU only)."""
    return ModelInputs(
        P=torch.from_numpy(obs["P"]),
        G=torch.from_numpy(obs["G"]),
        G_mask=torch.from_numpy(obs["G_mask"]),
        P_mask=torch.from_numpy(obs["P_mask"]),
        state_features=torch.from_numpy(obs["state_features"]),
    )


def build_model_inputs_from_batched_obs(obs: dict, device: torch.device) -> ModelInputs:
    """Converts a batched observation dict (B, ...) into ModelInputs on the target device."""
    # Sequence spaces from Gym vector envs are represented as tuples of per-env arrays.
    # Collate and pad them into dense tensors so the model can process minibatches.
    if isinstance(obs["P"], (list, tuple)):
        batch_size = len(obs["P"])

        max_p = (
            max(int(np.asarray(p).shape[0]) for p in obs["P"]) if batch_size > 0 else 0
        )
        max_g = (
            max(int(np.asarray(g).shape[0]) for g in obs["G"]) if batch_size > 0 else 0
        )

        p_dim = (
            max(
                (int(np.asarray(p).shape[-1]) if np.asarray(p).ndim >= 2 else 0)
                for p in obs["P"]
            )
            if batch_size > 0
            else 0
        )
        g_dim = (
            max(
                (int(np.asarray(g).shape[-1]) if np.asarray(g).ndim >= 2 else 0)
                for g in obs["G"]
            )
            if batch_size > 0
            else 0
        )

        P = torch.zeros((batch_size, max_p, p_dim), dtype=torch.float32, device=device)
        G = torch.zeros((batch_size, max_g, g_dim), dtype=torch.float32, device=device)
        G_mask = torch.zeros((batch_size, max_g), dtype=torch.bool, device=device)
        P_mask = torch.zeros((batch_size, max_p), dtype=torch.bool, device=device)

        for i in range(batch_size):
            p_arr = np.asarray(obs["P"][i])
            g_arr = np.asarray(obs["G"][i])
            p_mask_arr = np.asarray(obs["P_mask"][i])
            g_mask_arr = np.asarray(obs["G_mask"][i])

            n_p = int(p_arr.shape[0])
            n_g = int(g_arr.shape[0])

            if n_p > 0:
                p_width = int(p_arr.shape[-1]) if p_arr.ndim >= 2 else 0
                P[i, :n_p, :p_width].copy_(
                    torch.from_numpy(p_arr).to(device=device, dtype=torch.float32)
                )
                P_mask[i, :n_p].copy_(
                    torch.from_numpy(p_mask_arr).to(device=device, dtype=torch.bool)
                )
            if n_g > 0:
                g_width = int(g_arr.shape[-1]) if g_arr.ndim >= 2 else 0
                G[i, :n_g, :g_width].copy_(
                    torch.from_numpy(g_arr).to(device=device, dtype=torch.float32)
                )
                G_mask[i, :n_g].copy_(
                    torch.from_numpy(g_mask_arr).to(device=device, dtype=torch.bool)
                )

        return ModelInputs(
            P=P,
            G=G,
            G_mask=G_mask,
            P_mask=P_mask,
            state_features=torch.from_numpy(obs["state_features"]).to(
                device=device, dtype=torch.float32
            ),
        )

    return ModelInputs(
        P=torch.from_numpy(obs["P"]).to(device=device, dtype=torch.float32),
        G=torch.from_numpy(obs["G"]).to(device=device, dtype=torch.float32),
        G_mask=torch.from_numpy(obs["G_mask"]).to(device=device, dtype=torch.bool),
        P_mask=torch.from_numpy(obs["P_mask"]).to(device=device, dtype=torch.bool),
        state_features=torch.from_numpy(obs["state_features"]).to(
            device=device, dtype=torch.float32
        ),
    )


def collate_ragged_model_inputs(
    indices,
    obs_G_list,
    obs_P_list,
    g_mask_list,
    p_mask_list,
    obs_sf_list,
    device: torch.device,
) -> ModelInputs:
    """Pads ragged per-sample tensors to per-minibatch maxima and returns a dense ModelInputs."""
    mb_size = len(indices)
    max_g = max(int(obs_G_list[i].shape[0]) for i in indices)
    max_p = max(int(obs_P_list[i].shape[0]) for i in indices)
    g_dim = max(int(obs_G_list[i].shape[-1]) for i in indices)
    p_dim = max(int(obs_P_list[i].shape[-1]) for i in indices)

    G = torch.zeros((mb_size, max_g, g_dim), dtype=torch.float32, device=device)
    P = torch.zeros((mb_size, max_p, p_dim), dtype=torch.float32, device=device)
    G_mask = torch.zeros((mb_size, max_g), dtype=torch.bool, device=device)
    P_mask = torch.zeros((mb_size, max_p), dtype=torch.bool, device=device)
    state_features = torch.stack([obs_sf_list[i] for i in indices], dim=0).to(
        device=device, dtype=torch.float32
    )

    for row, idx in enumerate(indices):
        g = obs_G_list[idx]
        p = obs_P_list[idx]
        g_m = g_mask_list[idx]
        p_m = p_mask_list[idx]

        n_g = int(g.shape[0])
        n_p = int(p.shape[0])
        g_width = int(g.shape[-1])
        p_width = int(p.shape[-1])

        G[row, :n_g, :g_width].copy_(g)
        P[row, :n_p, :p_width].copy_(p)
        G_mask[row, :n_g].copy_(g_m)
        P_mask[row, :n_p].copy_(p_m)

    return ModelInputs(
        P=P,
        G=G,
        G_mask=G_mask,
        P_mask=P_mask,
        state_features=state_features,
    )


def get_space_last_dim(obs_space, key: str) -> int:
    """Returns the last feature dimension for Box or Sequence observation subspaces."""
    space = obs_space[key]
    shape = getattr(space, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[-1])
    feature_space = getattr(space, "feature_space", None)
    if feature_space is not None:
        feature_shape = getattr(feature_space, "shape", None)
        if feature_shape is not None and len(feature_shape) > 0:
            return int(feature_shape[-1])
    raise ValueError(f"Could not infer feature dimension for observation key '{key}'.")
