from typing import List
import torch
import torch.nn as nn

def make_mlp(
    in_dim: int,
    n_layers: int,
    hidden_dims: int,
    out_dim: int,
    activation: str = "relu",
    layer_norm: bool = False,
) -> nn.Sequential:
    """Utility: fully-connected MLP with optional LayerNorm ending in a linear layer."""
    acts = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    if activation not in acts:
        raise ValueError(f"Unknown activation '{activation}'. Options: {list(acts.keys())}")
    if n_layers < 0:
        raise ValueError("n_layers must be non-negative")

    Act = acts[activation]

    layers: List[nn.Module] = []
    prev = in_dim

    for _ in range(n_layers):
        layers.append(nn.Linear(prev, hidden_dims))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dims))
        layers.append(Act())
        prev = hidden_dims

    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)

def scatter_sum(src, index, dim_size):
    out = torch.zeros(dim_size, src.size(-1), device=src.device)
    out.index_add_(0, index, src)
    return out


def flatten_valid_nodes(node_feat, node_mask):
    B, N_max, d = node_feat.shape
    device = node_feat.device

    valid = node_mask.view(-1).bool()
    flat_nodes = node_feat.view(-1, d)[valid]

    node_batch = (
        torch.arange(B, device=device)
        .unsqueeze(1)
        .expand(B, N_max)
        .reshape(-1)[valid]
    )

    return flat_nodes, node_batch


def build_batched_edge_index(
    pair_index,
    pair_mask,
    num_nodes,
    make_undirected=True,
):
    device = pair_index.device
    B, E_max, _ = pair_index.shape

    # Node offsets
    node_offsets = torch.cumsum(
        torch.cat([
            torch.zeros(1, device=device, dtype=num_nodes.dtype),
            num_nodes[:-1]
        ]),
        dim=0
    )

    valid_pairs = pair_mask.view(-1)
    flat_pairs = pair_index.view(-1, 2)[valid_pairs]

    batch_ids = (
        torch.arange(B, device=device)
        .unsqueeze(1)
        .expand(B, E_max)
        .reshape(-1)[valid_pairs]
    )

    offsets = node_offsets[batch_ids].unsqueeze(1)
    global_pairs = flat_pairs + offsets

    src = global_pairs[:, 0]
    dst = global_pairs[:, 1]

    edge_index = torch.stack([src, dst], dim=0)

    if make_undirected:
        rev = torch.stack([dst, src], dim=0)
        edge_index = torch.cat([edge_index, rev], dim=1)
        batch_ids = torch.cat([batch_ids, batch_ids], dim=0)

    return edge_index, batch_ids

def build_pair_graph_edge_index(
    pair_indices: torch.Tensor,   # [B, P_max, 2]
    pair_mask: torch.Tensor,      # [B, P_max]
    make_undirected: bool = True,
):
    """
    Build edge_index for pair graph where nodes are pairs and
    edges exist if two pairs share at least one index.

    Returns:
        edge_index: [2, E_total]
    """

    B, P_max, _ = pair_indices.shape
    device = pair_indices.device

    edge_list = []
    node_offset = 0

    for b in range(B):
        mask = pair_mask[b]                  # [P_max]
        pairs = pair_indices[b]             # [P_max, 2]

        valid_idx = mask.nonzero(as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            continue

        pairs = pairs[valid_idx]            # [P, 2]
        P = pairs.shape[0]

        # Expand for pairwise comparison
        p1 = pairs.unsqueeze(1)             # [P, 1, 2]
        p2 = pairs.unsqueeze(0)             # [1, P, 2]

        # Check shared indices
        # Condition: any element matches
        shared = (
            ((p1[..., 0] >= 0) & (p2[..., 0] >= 0) & (p1[..., 0] == p2[..., 0])) |
            ((p1[..., 0] >= 0) & (p2[..., 1] >= 0) & (p1[..., 0] == p2[..., 1])) |
            ((p1[..., 1] >= 0) & (p2[..., 0] >= 0) & (p1[..., 1] == p2[..., 0])) |
            ((p1[..., 1] >= 0) & (p2[..., 1] >= 0) & (p1[..., 1] == p2[..., 1]))
        )                                   # [P, P]

        # Remove self-loops if desired
        shared.fill_diagonal_(False)

        src, dst = shared.nonzero(as_tuple=True)  # indices in [0, P)

        # Apply node offset
        src = src + node_offset
        dst = dst + node_offset

        edges = torch.stack([src, dst], dim=0)

        edge_list.append(edges)

        node_offset += P

    if len(edge_list) == 0:
        return torch.zeros(2, 0, dtype=torch.long, device=device)

    edge_index = torch.cat(edge_list, dim=1)

    if make_undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    return edge_index

def global_pool_batch(x, batch_index):
    B = int(batch_index.max().item()) + 1
    d = x.size(-1)
    device = x.device

    sum_pool = torch.zeros(B, d, device=device)
    sum_pool.index_add_(0, batch_index, x)

    count = torch.zeros(B, device=device)
    count.index_add_(0, batch_index, torch.ones_like(batch_index, dtype=torch.float))
    mean_pool = sum_pool / count.clamp(min=1).unsqueeze(-1)

    max_pool = torch.full((B, d), -torch.inf, device=device)
    max_pool = max_pool.index_reduce(0, batch_index, x, reduce="amax")

    return torch.cat([mean_pool, max_pool, sum_pool], dim=-1)
