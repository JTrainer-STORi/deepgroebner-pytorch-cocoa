"""Diagnostic functions for monitoring the correlation between the policies decisions and those that would have been made by heuristics, the influence of the state and state features on policy decisions using counterfactuals, diagnostics related to the attention weights in the alphastar transformer"""

import torch
from torch.distributions import Categorical

from deepgroebner_pytorch_cocoa.model_input_utils import build_model_inputs_from_single_obs


def _diagnostic_metric_name(key: str) -> str:
    return f"policy_function/diagnostic_{key}"


def _log_diagnostic_metrics(writer, metrics: dict, global_step: int):
    for key, value in metrics.items():
        writer.add_scalar(_diagnostic_metric_name(key), value, global_step)


def _spearman_from_rank_vectors(policy_ranks: torch.Tensor, heuristic_ranks: torch.Tensor) -> float:
    """Computes Spearman correlation from 0-indexed rank vectors of equal length."""
    n = int(policy_ranks.numel())
    if n < 2:
        return float("nan")
    d2 = (policy_ranks - heuristic_ranks).pow(2).sum().item()
    denom = n * (n * n - 1)
    return 1.0 - (6.0 * d2) / float(denom)


def _heuristic_orders_from_pair_obs(
    pair_obs: torch.Tensor,
    pair_idx_raw: torch.Tensor | None = None,
    pair_sugar: torch.Tensor | None = None,
    pair_normal_rank: torch.Tensor | None = None,
):
    """
    pair_obs: (M, 2*D) tensor containing concatenated lead monomial exponent vectors for each pair.
    Returns ordering indices for heuristics over the M valid pairs.
    """
    half = pair_obs.shape[-1] // 2
    left = pair_obs[:, :half]
    right = pair_obs[:, half:]
    lcm = torch.maximum(left, right)
    lcm_degree = lcm.sum(dim=-1)

    m = int(pair_obs.shape[0])
    idxs = list(range(m))

    if pair_idx_raw is not None and pair_idx_raw.ndim == 2 and pair_idx_raw.shape == (m, 2):
        first_order = sorted(
            idxs,
            key=lambda i: (
                int(pair_idx_raw[i, 1].item()),
                int(pair_idx_raw[i, 0].item()),
                i,
            ),
        )
    else:
        first_order = idxs
    degree_order = sorted(idxs, key=lambda i: (float(lcm_degree[i].item()), i))
    if pair_normal_rank is not None and int(pair_normal_rank.numel()) == m:
        normal_order = sorted(idxs, key=lambda i: (float(pair_normal_rank[i].item()), i))
    else:
        # Fallback approximation when exact normal ranks are not provided.
        normal_order = sorted(idxs, key=lambda i: (float(lcm_degree[i].item()), tuple(int(v) for v in lcm[i].tolist()), i))
    if pair_sugar is not None and int(pair_sugar.numel()) == m:
        sugar_order = sorted(idxs, key=lambda i: (float(pair_sugar[i].item()), i))
    else:
        sugar_order = degree_order

    return {
        "first": first_order,
        "degree": degree_order,
        "normal": normal_order,
        "sugar": sugar_order,
    }


@torch.no_grad()
def _policy_vs_heuristic_ranking_metrics(
    logits: torch.Tensor,
    pair_obs: torch.Tensor,
    pair_mask: torch.Tensor,
    pair_idx_raw: torch.Tensor | None = None,
    pair_sugar: torch.Tensor | None = None,
    pair_normal_rank: torch.Tensor | None = None,
):
    """
    Compare policy ranking on valid pairs against heuristic rankings.
    Returns batch-averaged Spearman and top-1 agreement metrics.
    """
    if logits.ndim != 2 or pair_obs.ndim != 3 or pair_mask.ndim != 2:
        return {}
    if logits.shape != pair_mask.shape:
        return {}
    if logits.shape[1] != pair_obs.shape[1]:
        return {}
    if pair_obs.shape[-1] % 2 != 0:
        return {}

    spearman = {"first": [], "degree": [], "normal": [], "sugar": []}
    top1 = {"first": [], "degree": [], "normal": [], "sugar": []}

    batch_size = logits.shape[0]
    for b in range(batch_size):
        valid = pair_mask[b].bool()
        n_valid = int(valid.sum().item())
        if n_valid < 2:
            continue

        valid_logits = logits[b, valid]
        valid_pairs = pair_obs[b, valid]
        valid_pair_idx_raw = None
        valid_sugar = None
        valid_normal_rank = None
        if pair_idx_raw is not None and pair_idx_raw.ndim == 3 and pair_idx_raw.shape[:2] == pair_mask.shape and pair_idx_raw.shape[-1] == 2:
            valid_pair_idx_raw = pair_idx_raw[b, valid]
        if pair_sugar is not None and pair_sugar.ndim == 2 and pair_sugar.shape == pair_mask.shape:
            valid_sugar = pair_sugar[b, valid]
        if pair_normal_rank is not None and pair_normal_rank.ndim == 2 and pair_normal_rank.shape == pair_mask.shape:
            valid_normal_rank = pair_normal_rank[b, valid]

        policy_order = torch.argsort(valid_logits, descending=True)
        policy_ranks = torch.empty_like(policy_order, dtype=torch.float32)
        policy_ranks[policy_order] = torch.arange(n_valid, device=policy_order.device, dtype=torch.float32)

        heuristic_orders = _heuristic_orders_from_pair_obs(
            valid_pairs,
            pair_idx_raw=valid_pair_idx_raw,
            pair_sugar=valid_sugar,
            pair_normal_rank=valid_normal_rank,
        )
        policy_top_idx = int(policy_order[0].item())

        for name, order in heuristic_orders.items():
            h_order = torch.tensor(order, device=policy_order.device, dtype=torch.long)
            h_ranks = torch.empty_like(h_order, dtype=torch.float32)
            h_ranks[h_order] = torch.arange(n_valid, device=policy_order.device, dtype=torch.float32)

            spearman[name].append(_spearman_from_rank_vectors(policy_ranks, h_ranks))
            top1[name].append(1.0 if policy_top_idx == int(h_order[0].item()) else 0.0)

    out = {}
    for name in ["first", "degree", "normal", "sugar"]:
        if len(spearman[name]) > 0:
            out[f"policy_vs_{name}_spearman"] = float(sum(spearman[name]) / len(spearman[name]))
            out[f"policy_vs_{name}_top1_agreement"] = float(sum(top1[name]) / len(top1[name]))
    return out


def run_actor_ranking_diagnostics(
    actor,
    input,
    writer,
    update_idx: int,
    global_step: int,
    ranking_every: int = 20,
    *,
    pair_idx_raw: torch.Tensor | None = None,
    pair_sugar: torch.Tensor | None = None,
    pair_normal_rank: torch.Tensor | None = None,
):
    """Compare policy rankings with heuristics using optional diagnostic metadata."""
    if ranking_every <= 0 or (update_idx % ranking_every) != 0:
        return {}

    actor.eval()
    with torch.no_grad():
        logits = actor(input).squeeze(-1)
        metrics = _policy_vs_heuristic_ranking_metrics(
            logits,
            input.P,
            input.P_mask,
            pair_idx_raw=pair_idx_raw,
            pair_sugar=pair_sugar,
            pair_normal_rank=pair_normal_rank,
        )

    writer.add_scalar(_diagnostic_metric_name("ranking_metrics_present"), 1.0 if metrics else 0.0, global_step)
    _log_diagnostic_metrics(writer, metrics, global_step)

    actor.train()
    return metrics


def _normalize_cached_state_features(
    model_input,
    obs_sf_normalizer,
    *,
    epsilon: float,
    clip_range: float,
):
    if obs_sf_normalizer is None or model_input.state_features is None:
        return

    if isinstance(obs_sf_normalizer, dict):
        mean = obs_sf_normalizer["mean"].to(
            device=model_input.state_features.device,
            dtype=model_input.state_features.dtype,
        )
        var = obs_sf_normalizer["var"].to(
            device=model_input.state_features.device,
            dtype=model_input.state_features.dtype,
        )
        model_input.state_features = torch.clamp(
            (model_input.state_features - mean) / torch.sqrt(var + epsilon),
            -clip_range,
            clip_range,
        )
        return

    model_input.state_features = obs_sf_normalizer.normalize(
        model_input.state_features,
        epsilon=epsilon,
        clip_range=clip_range,
    )


def _masked_dist_from_logits(logits: torch.Tensor, mask: torch.Tensor, neg: float = -1e8) -> Categorical:
    # logits: (B, N), mask: (B, N) bool
    logits_m = torch.where(mask, logits, logits.new_full((), neg))
    return Categorical(logits=logits_m)

@torch.no_grad()
def _cf_metrics(logits_base: torch.Tensor, logits_cf: torch.Tensor, mask: torch.Tensor, prefix: str):
    dist = _masked_dist_from_logits(logits_base, mask)
    dist_cf = _masked_dist_from_logits(logits_cf, mask)

    # KL(dist || dist_cf)
    kl = (dist.probs * (dist.logits - dist_cf.logits)).sum(-1).mean()

    # argmax flip rate (under masked support)
    flip = (dist.probs.argmax(-1) != dist_cf.probs.argmax(-1)).float().mean()

    # L1 distance in prob space
    l1 = (dist.probs - dist_cf.probs).abs().sum(-1).mean()

    return {
        f"{prefix}_kl": kl.item(),
        f"{prefix}_flip": flip.item(),
        f"{prefix}_l1": l1.item(),
    }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Use where(...) so masked-out NaNs do not leak into the reduction.
    masked_values = torch.where(mask > 0, values, torch.zeros_like(values))
    denom = mask.sum().clamp_min(1.0)
    return masked_values.sum() / denom


def _mha_head_diversity_stats(probs: torch.Tensor, q_valid: torch.Tensor, prefix: str):
    """
    probs: (B, H, Nq, Nk), normalized attention probabilities.
    q_valid: (B, 1, Nq), float mask in {0,1} for valid query rows.
    """
    eps = 1e-12
    B, H, Nq, Nk = probs.shape

    if H < 2:
        return {}

    # Build all unordered head pairs.
    pair_idx = torch.triu_indices(H, H, offset=1, device=probs.device)
    hi, hj = pair_idx[0], pair_idx[1]

    # Reorder to (B, Nq, H, Nk) so head-pair slicing is straightforward.
    p = probs.permute(0, 2, 1, 3)
    pi = p[:, :, hi, :]  # (B, Nq, P, Nk)
    pj = p[:, :, hj, :]  # (B, Nq, P, Nk)

    # Pairwise L1 distance between heads.
    l1 = (pi - pj).abs().sum(dim=-1)

    # Pairwise cosine similarity between heads.
    dot = (pi * pj).sum(dim=-1)
    ni = pi.norm(dim=-1)
    nj = pj.norm(dim=-1)
    cosine = dot / (ni * nj).clamp_min(eps)

    # Pairwise Jensen-Shannon divergence.
    m = 0.5 * (pi + pj)
    js = 0.5 * (
        (pi * (pi.clamp_min(eps).log() - m.clamp_min(eps).log())).sum(dim=-1)
        + (pj * (pj.clamp_min(eps).log() - m.clamp_min(eps).log())).sum(dim=-1)
    )

    # Agreement on most-attended key.
    top_agree = (pi.argmax(dim=-1) == pj.argmax(dim=-1)).to(probs.dtype)

    # Valid query rows mask (broadcast over head-pair dim).
    row_mask = q_valid.squeeze(1).unsqueeze(-1)

    # Per-row spread of entropy across heads (another diversity signal).
    row_entropy = -(probs * probs.clamp_min(eps).log()).sum(dim=-1)  # (B, H, Nq)
    entropy_head_std = row_entropy.permute(0, 2, 1).std(dim=-1, unbiased=False)  # (B, Nq)

    l1 = torch.nan_to_num(l1, nan=0.0, posinf=0.0, neginf=0.0)
    cosine = torch.nan_to_num(cosine, nan=0.0, posinf=0.0, neginf=0.0)
    js = torch.nan_to_num(js, nan=0.0, posinf=0.0, neginf=0.0)
    top_agree = torch.nan_to_num(top_agree, nan=0.0, posinf=0.0, neginf=0.0)
    entropy_head_std = torch.nan_to_num(entropy_head_std, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        f"{prefix}_head_pair_l1": _masked_mean(l1, row_mask).item(),
        f"{prefix}_head_pair_cosine": _masked_mean(cosine, row_mask).item(),
        f"{prefix}_head_pair_js": _masked_mean(js, row_mask).item(),
        f"{prefix}_head_pair_top_agree": _masked_mean(top_agree, row_mask).item(),
        f"{prefix}_head_entropy_std": _masked_mean(entropy_head_std, q_valid.squeeze(1)).item(),
    }


def _attention_stats(weights: torch.Tensor, q_mask: torch.Tensor | None, k_mask: torch.Tensor | None, prefix: str):
    """
    weights: (B, H, Nq, Nk) attention probabilities.
    q_mask: (B, Nq) bool mask for valid queries or None.
    k_mask: (B, Nk) bool mask for valid keys or None.
    """
    eps = 1e-12
    B, H, Nq, Nk = weights.shape

    q_valid = torch.ones((B, 1, Nq), device=weights.device, dtype=weights.dtype)
    if q_mask is not None:
        q_valid = q_mask.to(weights.dtype).unsqueeze(1)

    k_valid = torch.ones((B, 1, 1, Nk), device=weights.device, dtype=weights.dtype)
    if k_mask is not None:
        k_valid = k_mask.to(weights.dtype).unsqueeze(1).unsqueeze(1)

    # Guard against non-finite attention rows (can happen when all keys are masked).
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

    probs = weights * k_valid
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

    # If a sample has no valid keys, exclude its queries from diagnostics.
    has_valid_keys = (k_valid.sum(dim=-1) > 0).to(weights.dtype)  # (B,1,1)
    q_valid = q_valid * has_valid_keys

    entropy = -(probs * probs.clamp_min(eps).log()).sum(dim=-1)  # (B,H,Nq)
    entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)
    entropy_mean = _masked_mean(entropy, q_valid)

    # Normalize by max possible entropy for current valid key count.
    if k_mask is not None:
        key_count = k_mask.to(weights.dtype).sum(dim=-1).clamp_min(1.0)  # (B,)
    else:
        key_count = torch.full((B,), Nk, device=weights.device, dtype=weights.dtype)
    entropy_norm = entropy / key_count.log().clamp_min(eps).view(B, 1, 1)
    entropy_norm = torch.nan_to_num(entropy_norm, nan=0.0, posinf=0.0, neginf=0.0)
    entropy_norm_mean = _masked_mean(entropy_norm, q_valid)

    stats = {
        f"{prefix}_entropy": entropy_mean.item(),
        f"{prefix}_entropy_norm": entropy_norm_mean.item(),
    }

    stats.update(_mha_head_diversity_stats(probs, q_valid, prefix=prefix))

    # Diagonal vs off-diagonal only makes sense for square attention maps.
    if Nq == Nk:
        diag_idx = torch.arange(Nq, device=weights.device)
        diag_vals = probs[:, :, diag_idx, diag_idx]  # (B,H,Nq)

        if q_mask is not None and k_mask is not None:
            diag_valid = (q_mask & k_mask).to(weights.dtype).unsqueeze(1)
        elif q_mask is not None:
            diag_valid = q_mask.to(weights.dtype).unsqueeze(1)
        elif k_mask is not None:
            diag_valid = k_mask.to(weights.dtype).unsqueeze(1)
        else:
            diag_valid = torch.ones((B, 1, Nq), device=weights.device, dtype=weights.dtype)

        diag_vals = torch.nan_to_num(diag_vals, nan=0.0, posinf=0.0, neginf=0.0)
        diag_mean = _masked_mean(diag_vals, diag_valid)

        off_mask = q_valid.unsqueeze(-1) * k_valid
        eye = torch.eye(Nq, device=weights.device, dtype=weights.dtype).view(1, 1, Nq, Nk)
        off_mask = off_mask * (1.0 - eye)

        off_mean = _masked_mean(probs, off_mask)

        stats[f"{prefix}_diag_mean"] = diag_mean.item()
        stats[f"{prefix}_offdiag_mean"] = off_mean.item()
        stats[f"{prefix}_diag_over_offdiag"] = (diag_mean / (off_mean + eps)).item()

    return stats


def _attention_payload_metrics(attention_payload):
    metrics = {}

    if not attention_payload:
        return metrics

    for branch_name, branch_maps in attention_payload.items():
        if not branch_maps:
            continue

        for layer_name, layer_data in branch_maps.items():
            weights = layer_data.get("weights")
            if weights is None:
                continue

            prefix = f"attn_{branch_name}_{layer_name}"
            q_mask = layer_data.get("q_mask")
            k_mask = layer_data.get("k_mask")
            metrics.update(_attention_stats(weights, q_mask, k_mask, prefix=prefix))

    return metrics

def _head_logits_from_state(actor, pair_embedding, state, mask, state_feats=None):
    if state_feats is not None:
        state = actor.state_feature_combiner(state, state_feats)

    logits = actor.final_module(state, pair_embedding, mask)

    return logits.squeeze(-1)

def _head_logits_from_state_feats(actor, pair_embedding, state_feats, mask, state=None):
    if state is not None:
        state = actor.state_feature_combiner(state, state_feats)
        logits = actor.final_module(state, pair_embedding, mask)
    else:
        logits = actor.final_module(state_feats, pair_embedding, mask)

    return logits.squeeze(-1)

def _counterfactual_metrics(
    actor, input, logits, pair_embedding, state_embedding, state_features_embedding,
):
    metrics = {}

    if state_embedding is not None and state_features_embedding is not None:

        # cf states
        state_zero = torch.zeros_like(state_embedding)
        state_mean = state_embedding.mean(dim=0, keepdim=True).expand_as(state_embedding)
        perm = torch.randperm(state_embedding.shape[0], device=state_embedding.device)
        state_swap = state_embedding[perm]

        # cf state features
        state_features_zero = torch.zeros_like(state_features_embedding)
        state_features_mean = state_features_embedding.mean(dim=0, keepdim=True).expand_as(state_features_embedding)
        perm = torch.randperm(state_features_embedding.shape[0], device=state_features_embedding.device)
        state_features_swap = state_features_embedding[perm]

        # compute the counterfactual diagnostics for the state by using the original state features
        state_logits_zero = _head_logits_from_state(actor, pair_embedding, state_zero, mask=input.P_mask, state_feats=state_features_embedding)
        state_logits_mean = _head_logits_from_state(actor, pair_embedding, state_mean, mask=input.P_mask, state_feats=state_features_embedding)
        state_logits_swap = _head_logits_from_state(actor, pair_embedding, state_swap, mask=input.P_mask, state_feats=state_features_embedding)
        metrics.update(_cf_metrics(logits, state_logits_zero, input.P_mask, prefix="cf_state_zero"))
        metrics.update(_cf_metrics(logits, state_logits_mean, input.P_mask, prefix="cf_state_mean"))
        metrics.update(_cf_metrics(logits, state_logits_swap, input.P_mask, prefix="cf_state_swap"))
        # compute the counterfactual diagnostics for the state features by using the original state features
        state_features_logits_zero = _head_logits_from_state_feats(actor, pair_embedding, state_features_zero, input.P_mask, state=state_embedding)
        state_features_logits_mean = _head_logits_from_state_feats(actor, pair_embedding, state_features_mean, input.P_mask, state=state_embedding)
        state_features_logits_swap = _head_logits_from_state_feats(actor, pair_embedding, state_features_swap, input.P_mask, state=state_embedding)
        metrics.update(_cf_metrics(logits, state_features_logits_zero, input.P_mask, prefix="cf_state_feats_zero"))
        metrics.update(_cf_metrics(logits, state_features_logits_mean, input.P_mask, prefix="cf_state_feats_mean"))
        metrics.update(_cf_metrics(logits, state_features_logits_swap, input.P_mask, prefix="cf_state_feats_swap"))

        # norms (optional)
        metrics["state_norm"] = state_embedding.norm(dim=-1).mean().item()
        metrics["pair_embed_norm"] = pair_embedding.norm(dim=-1).masked_select(input.P_mask).mean().item()
        metrics["state_over_pair_norm_ratio"] = metrics["state_norm"] / (metrics["pair_embed_norm"] + 1e-12)
        metrics["state_feats_norm"] = state_features_embedding.norm(dim=-1).mean().item()
        metrics["pair_embed_norm"] = pair_embedding.norm(dim=-1).masked_select(input.P_mask).mean().item()
        metrics["state_feats_over_pair_norm_ratio"] = metrics["state_feats_norm"] / (metrics["pair_embed_norm"] + 1e-12)

    elif state_embedding is not None:
        state_zero = torch.zeros_like(state_embedding)
        state_mean = state_embedding.mean(dim=0, keepdim=True).expand_as(state_embedding)
        perm = torch.randperm(state_embedding.shape[0], device=state_embedding.device)
        state_swap = state_embedding[perm]

        # recompute logits under counterfactuals using the SAME head logic as the actor
        logits_zero = _head_logits_from_state(actor, pair_embedding, state_zero, mask = input.P_mask)
        logits_mean = _head_logits_from_state(actor, pair_embedding, state_mean, mask = input.P_mask)
        logits_swap = _head_logits_from_state(actor, pair_embedding, state_swap, mask = input.P_mask)
        metrics.update(_cf_metrics(logits, logits_zero, input.P_mask, prefix="cf_state_zero"))
        metrics.update(_cf_metrics(logits, logits_mean, input.P_mask, prefix="cf_state_mean"))
        metrics.update(_cf_metrics(logits, logits_swap, input.P_mask, prefix="cf_state_swap"))

        # norms (optional)
        metrics["state_norm"] = state_embedding.norm(dim=-1).mean().item()
        metrics["pair_embed_norm"] = pair_embedding.norm(dim=-1).masked_select(input.P_mask).mean().item()
        metrics["state_over_pair_norm_ratio"] = metrics["state_norm"] / (metrics["pair_embed_norm"] + 1e-12)

    elif state_features_embedding is not None:
        state_features_zero = torch.zeros_like(state_features_embedding)
        state_features_mean = state_features_embedding.mean(dim=0, keepdim=True).expand_as(state_features_embedding)
        perm = torch.randperm(state_features_embedding.shape[0], device=state_features_embedding.device)
        state_features_swap = state_features_embedding[perm]

        # recompute logits under counterfactuals using the SAME head logic as the actor
        logits_zero = _head_logits_from_state_feats(actor, pair_embedding, state_features_zero, input.P_mask)
        logits_mean = _head_logits_from_state_feats(actor, pair_embedding, state_features_mean, input.P_mask)
        logits_swap = _head_logits_from_state_feats(actor, pair_embedding, state_features_swap, input.P_mask)
        metrics.update(_cf_metrics(logits, logits_zero, input.P_mask, prefix="cf_state_feats_zero"))
        metrics.update(_cf_metrics(logits, logits_mean, input.P_mask, prefix="cf_state_feats_mean"))
        metrics.update(_cf_metrics(logits, logits_swap, input.P_mask, prefix="cf_state_feats_swap"))

        # norms (optional)
        metrics["state_feats_norm"] = state_features_embedding.norm(dim=-1).mean().item()
        metrics["pair_embed_norm"] = pair_embedding.norm(dim=-1).masked_select(input.P_mask).mean().item()
        metrics["state_feats_over_pair_norm_ratio"] = metrics["state_feats_norm"] / (metrics["pair_embed_norm"] + 1e-12)

    elif actor.pair_only_scoring:
        metrics["pair_embed_norm"] = pair_embedding.norm(dim=-1).masked_select(input.P_mask).mean().item()
        metrics["pair_only_scoring_no_global_state_embedding"] = 1.0


    else:
        raise ValueError("Invalid network structure for diagnostics.")

    return metrics


def run_actor_global_diagnostics(
    actor,
    input,
    writer,
    update_idx: int,
    global_step: int,
    diag_every: int = 20,
    grad_every: int = 100,
    return_attention: bool = True,
    attention_every: int = 0,
):
    """
    Call AFTER finishing a PPO update, before collecting the next rollout.
    batch must contain: P, G, G_mask, P_mask
    Frequencies count completed PPO updates; nonpositive values disable a group.
    return_attention indicates whether the actor supports attention diagnostics.
    """

    counterfactual_due = diag_every > 0 and update_idx % diag_every == 0
    attention_due = (
        return_attention and attention_every > 0 and update_idx % attention_every == 0
    )
    gradient_due = grad_every > 0 and update_idx % grad_every == 0
    if not (counterfactual_due or attention_due or gradient_due):
        return

    actor.eval()

    metrics = {}

    if counterfactual_due or attention_due:

        # Share the initial forward pass when both diagnostic groups are due.
        with torch.no_grad():
            output = actor(input, return_internals=True, return_attention=attention_due)
            logits, pair_embedding, state_embedding, state_features_embedding = output[:4]
            logits = logits.squeeze(-1)
            if attention_due:
                metrics.update(_attention_payload_metrics(output[4]))

            if counterfactual_due:
                metrics.update(_counterfactual_metrics(
                    actor, input, logits, pair_embedding, state_embedding,
                    state_features_embedding,
                ))

    # ---- gradient sensitivity (optional) ----
    if gradient_due:

        logits, pair_embedding, state_embedding, state_features_embedding = actor(input, return_internals=True)
        logits = logits.squeeze(-1)

        pair_leaf = pair_embedding.detach().requires_grad_(True)

        if state_embedding is not None and state_features_embedding is not None:
            state_leaf = state_embedding.detach().requires_grad_(True)
            state_features_leaf = state_features_embedding.detach().requires_grad_(True)

            logits2 = _head_logits_from_state(actor, pair_leaf, state_leaf, mask=input.P_mask, state_feats=state_features_leaf)  # (B,N)

        elif state_embedding is not None:
            state_leaf = state_embedding.detach().requires_grad_(True)
            logits2 = _head_logits_from_state(actor, pair_leaf, state_leaf, mask=input.P_mask)  # (B,N)
        elif state_features_embedding is not None:
            state_features_leaf = state_features_embedding.detach().requires_grad_(True)
            logits2 = _head_logits_from_state_feats(actor, pair_leaf, state_features_leaf, mask=input.P_mask)  # (B,N)
        elif actor.pair_only_scoring:
            logits2 = actor.score_from_embeddings(
                pair_leaf,
                input,
                None,
                None,
            ).squeeze(-1)
        else:
            raise ValueError("Invalid network structure for diagnostics.")


        logits2_masked = logits2.masked_fill(~input.P_mask, float("-inf"))
        scalar = logits2_masked.max(dim=-1).values.mean()

        # Compute gradients w.r.t. both leaves
        actor.zero_grad(set_to_none=True)
        if state_embedding is not None:
            if state_leaf.grad is not None:
                state_leaf.grad.zero_()
        if pair_leaf.grad is not None:
            pair_leaf.grad.zero_()
        if state_features_embedding is not None:
            if state_features_leaf.grad is not None:
                state_features_leaf.grad.zero_()

        scalar.backward()

        # --- pair sensitivity: grad is (B,N,E). Reduce over E then average over valid pairs then over batch ---
        pair_grad = pair_leaf.grad  # (B,N,E)
        if pair_grad is None:
            pair_grad_sens = 0.0
        else:
            pair_grad_norm_per_pair = pair_grad.norm(dim=-1)  # (B,N)

            valid = input.P_mask
            # mean over valid pairs per batch item, then mean over batch
            denom = valid.sum(dim=-1).clamp_min(1).to(pair_grad_norm_per_pair.dtype)  # (B,)
            pair_grad_sens = ((pair_grad_norm_per_pair * valid.to(pair_grad_norm_per_pair.dtype)).sum(dim=-1) / denom).mean().item()

        metrics["pair_embed_sens"] = pair_grad_sens

        if state_embedding is not None:
            state_grad_sens = 0.0 if state_leaf.grad is None else state_leaf.grad.norm(dim=-1).mean().item()
            metrics["state_grad_sens"] = state_grad_sens

            metrics["state_over_pair_sens_ratio"] = state_grad_sens / (pair_grad_sens + 1e-12)
        if state_features_embedding is not None:
            state_features_grad_sens = 0.0 if state_features_leaf.grad is None else state_features_leaf.grad.norm(dim=-1).mean().item()
            metrics["state_features_grad_sens"] = state_features_grad_sens

            metrics["state_feats_over_pair_sens_ratio"] = state_features_grad_sens / (pair_grad_sens + 1e-12)

        # Optional: keep training grads pristine (especially if you call this mid-loop by mistake)
        actor.zero_grad(set_to_none=True)

    # ---- log ----
    _log_diagnostic_metrics(writer, metrics, global_step)

    actor.train()
