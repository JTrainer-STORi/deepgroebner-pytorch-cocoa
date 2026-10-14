from typing import List
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
        "selu": nn.SELU,
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
