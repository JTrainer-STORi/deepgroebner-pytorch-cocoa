from __future__ import annotations

import numpy as np
import torch

from deepgroebner_pytorch_cocoa.config_dataclasses import ModelInputs


def build_padded_polynomial_term_tensors(
    g_terms_list,
    g_term_counts_list,
    device: torch.device,
    fallback_term_dim: int | None = None,
):
    """Converts flattened full-term polynomial observations into padded [B, M, T, D] tensors."""
    batch_size = len(g_terms_list)
    max_polys = max(int(np.asarray(counts).shape[0]) for counts in g_term_counts_list) if batch_size > 0 else 0
    max_terms = 0
    term_dim = fallback_term_dim if fallback_term_dim is not None else 0

    for terms_arr, counts_arr in zip(g_terms_list, g_term_counts_list):
        terms_np = np.asarray(terms_arr)
        counts_np = np.asarray(counts_arr).reshape(-1)
        if counts_np.size > 0:
            max_terms = max(max_terms, int(np.max(counts_np)))
        if terms_np.ndim >= 2 and terms_np.shape[-1] > 0:
            term_dim = max(term_dim, int(terms_np.shape[-1]))

    G_poly_terms = torch.zeros((batch_size, max_polys, max_terms, term_dim), dtype=torch.float32, device=device)
    G_poly_term_mask = torch.zeros((batch_size, max_polys, max_terms), dtype=torch.bool, device=device)

    for batch_idx, (terms_arr, counts_arr) in enumerate(zip(g_terms_list, g_term_counts_list)):
        terms_np = np.asarray(terms_arr, dtype=np.float32)
        counts_np = np.asarray(counts_arr, dtype=np.int64).reshape(-1)
        offset = 0
        for poly_idx, term_count in enumerate(counts_np.tolist()):
            if term_count > 0:
                poly_terms = terms_np[offset:offset + term_count]
                poly_term_width = int(poly_terms.shape[-1]) if poly_terms.ndim >= 2 else 0
                G_poly_terms[batch_idx, poly_idx, :term_count, :poly_term_width].copy_(
                    torch.from_numpy(poly_terms).to(device=device, dtype=torch.float32)
                )
                G_poly_term_mask[batch_idx, poly_idx, :term_count] = True
            offset += term_count

    return G_poly_terms, G_poly_term_mask


def build_model_inputs_from_single_obs(obs: dict, device: torch.device) -> ModelInputs:
    """Converts a single observation dict into batched `ModelInputs` on `device`."""
    g_poly_terms = None
    g_poly_term_mask = None

    if "G_terms" in obs and "G_term_counts" in obs:
        fallback_term_dim = int(np.asarray(obs["G"]).shape[-1]) if np.asarray(obs["G"]).ndim >= 2 else None
        g_poly_terms, g_poly_term_mask = build_padded_polynomial_term_tensors(
            [obs["G_terms"]],
            [obs["G_term_counts"]],
            device=device,
            fallback_term_dim=fallback_term_dim,
        )

    return ModelInputs(
        P=torch.from_numpy(np.asarray(obs["P"])).unsqueeze(0).to(device=device, dtype=torch.float32),
        P_idx=torch.from_numpy(np.asarray(obs["P_idx"])).unsqueeze(0).to(device=device, dtype=torch.long),
        P_idx_raw=(
            torch.from_numpy(np.asarray(obs["P_idx_raw"])).unsqueeze(0).to(device=device, dtype=torch.long)
            if "P_idx_raw" in obs
            else None
        ),
        G=torch.from_numpy(np.asarray(obs["G"])).unsqueeze(0).to(device=device, dtype=torch.float32),
        G_mask=torch.from_numpy(np.asarray(obs["G_mask"])).unsqueeze(0).to(device=device, dtype=torch.bool),
        P_mask=torch.from_numpy(np.asarray(obs["P_mask"])).unsqueeze(0).to(device=device, dtype=torch.bool),
        P_sugar=(
            torch.from_numpy(np.asarray(obs["P_sugar"])).unsqueeze(0).to(device=device, dtype=torch.float32)
            if "P_sugar" in obs
            else None
        ),
        P_normal_rank=(
            torch.from_numpy(np.asarray(obs["P_normal_rank"])).unsqueeze(0).to(device=device, dtype=torch.float32)
            if "P_normal_rank" in obs
            else None
        ),
        G_poly_terms=g_poly_terms,
        G_poly_term_mask=g_poly_term_mask,
        state_features=torch.from_numpy(np.asarray(obs["state_features"])).unsqueeze(0).to(device=device, dtype=torch.float32),
    )
