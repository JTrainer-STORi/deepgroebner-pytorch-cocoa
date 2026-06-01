import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from typing import Optional

from deepgroebner_pytorch_cocoa.network_helpers import make_mlp, scatter_sum, flatten_valid_nodes, build_batched_edge_index, global_pool_batch, build_pair_graph_edge_index
from deepgroebner_pytorch_cocoa.config_dataclasses import ModelConfig, ModelInputs


def _polynomial_projection_hparams(config: ModelConfig):
    n_layers = getattr(config, "item_embedding_head_n_layers", 1)
    hidden_dims = getattr(config, "item_embedding_head_hidden_dims", config.polynomial_rnn_hidden_dim)
    activation = getattr(config, "item_embedding_head_activation", "relu")
    layer_norm = getattr(config, "item_embedding_head_layer_norm", False)
    out_dim = getattr(
        config,
        "item_embedding_head_out_dim",
        config.polynomial_rnn_hidden_dim * (2 if config.polynomial_rnn_bidirectional else 1),
    )
    return n_layers, hidden_dims, activation, layer_norm, out_dim


def _slice_optional_batch_tensor(tensor: Optional[torch.Tensor], batch_indices: torch.Tensor):
    if tensor is None:
        return None
    return tensor.index_select(0, batch_indices)


def _slice_model_inputs(inputs: ModelInputs, batch_indices: torch.Tensor) -> ModelInputs:
    """Selects a batch subset while preserving optional observation fields."""
    return ModelInputs(
        P=inputs.P.index_select(0, batch_indices),
        P_idx=inputs.P_idx.index_select(0, batch_indices),
        P_idx_raw=_slice_optional_batch_tensor(inputs.P_idx_raw, batch_indices),
        P_mask=inputs.P_mask.index_select(0, batch_indices),
        P_sugar=_slice_optional_batch_tensor(inputs.P_sugar, batch_indices),
        P_normal_rank=_slice_optional_batch_tensor(inputs.P_normal_rank, batch_indices),
        G=None if inputs.G is None else inputs.G.index_select(0, batch_indices),
        G_mask=None if inputs.G_mask is None else inputs.G_mask.index_select(0, batch_indices),
        G_poly_terms=_slice_optional_batch_tensor(inputs.G_poly_terms, batch_indices),
        G_poly_term_mask=_slice_optional_batch_tensor(inputs.G_poly_term_mask, batch_indices),
        state_features=inputs.state_features.index_select(0, batch_indices),
        h_value=_slice_optional_batch_tensor(inputs.h_value, batch_indices),
    )

class PerEntityMLP(nn.Module):
    """Basic MLP to apply to exponent vectors of pairs in the pairset P. Takes the pair exponent vector and passes it through an MLP. If use_lcm_gcd is true in the ModelConfig then the lcm and gcd will be appended to the input vector. 

    Options to add layer normalization or use any of the following activations: 'relu', 'gelu', 'silu', 'tanh'

    To mimic the network used in D. Peifer et al 2020 set layer_norm False, activation = relu, n_layers = 1, and hidden_dims = 128"""

    def __init__(self, config: ModelConfig):
        super().__init__()

        assert config.pair_embedding_head_in_dim % 2 == 0, \
        "pair_embedding_head_in_dim must be even"

        self.D = config.pair_embedding_head_in_dim // 2

        self.use_lcm_gcd = config.use_lcm_gcd

        in_dim = config.pair_embedding_head_in_dim * 2 if config.use_lcm_gcd else config.pair_embedding_head_in_dim

        self.pair_mlp = make_mlp(
            in_dim=in_dim,
            n_layers=config.pair_embedding_head_n_layers,
            hidden_dims=config.pair_embedding_head_hidden_dims,
            out_dim=config.pair_embedding_head_out_dim,
            activation=config.pair_embedding_head_activation,
            layer_norm=config.pair_embedding_head_layer_norm,
        )

    def forward(self, inputs: ModelInputs):

        if self.use_lcm_gcd:
            p1 = inputs.P[..., :self.D]
            p2 = inputs.P[..., self.D:]
            p_lcm = torch.maximum(p1, p2)
            p_gcd = torch.minimum(p1, p2)
            pair_feat = torch.cat([inputs.P, p_lcm, p_gcd], dim=-1)
            return self.pair_mlp(pair_feat)
        
        return self.pair_mlp(inputs.P)
        
class SymmetricPerEntityMLP(nn.Module):
    """This per-entity MLP is invariant to the ordering of pairs in a S-Pair. This means that it will evaluate pair (i,j) identically to pair (j,i) unlike the PerEntityMLP which will produce different outputs for (i,j) and (j,i)
    
    It does this by putting each pair through the same `item_encoder` and then combining the outputs using symmetric operations like sum and absolute difference. 

    If config.use_lcm_gcd is True then the lcm and gcd of the two pairs will be calculated and passed through the item encoder before the outputs for each item are concatenated and passed to a final MLP. 
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()

        assert config.pair_embedding_head_in_dim % 2 == 0, \
        "pair_embedding_head_in_dim must be even"

        self.D = config.pair_embedding_head_in_dim // 2 

        self.use_lcm_gcd = config.use_lcm_gcd

        self.item_encoder = make_mlp(
            in_dim=self.D,
            n_layers=config.item_embedding_head_n_layers,
            hidden_dims=config.item_embedding_head_hidden_dims,
            out_dim=config.item_embedding_head_out_dim,
            activation=config.item_embedding_head_activation,
            layer_norm=config.item_embedding_head_layer_norm
        )
        
        in_dim = 4 * config.item_embedding_head_out_dim if self.use_lcm_gcd else 2 * config.item_embedding_head_out_dim

        self.pair_mlp = make_mlp(
            in_dim=in_dim,
            n_layers=config.pair_embedding_head_n_layers,
            hidden_dims=config.pair_embedding_head_hidden_dims,
            out_dim=config.pair_embedding_head_out_dim,
            activation=config.pair_embedding_head_activation,
            layer_norm=config.pair_embedding_head_layer_norm
        )

    def forward(self, inputs: ModelInputs) -> torch.Tensor:

        p1 = inputs.P[..., :self.D]
        p2 = inputs.P[..., self.D:]

        h1 = self.item_encoder(p1)          # (B, N, E)
        h2 = self.item_encoder(p2)          # (B, N, E)

        s = h1 + h2
        d = torch.abs(h1 - h2)

        if self.use_lcm_gcd:
          p_lcm = torch.maximum(p1, p2)
          p_gcd = torch.minimum(p1, p2)

          hlcm = self.item_encoder(p_lcm)
          hgcd = self.item_encoder(p_gcd)

          pair_feat = torch.cat([s, d, hlcm, hgcd], dim=-1)  # (B, N, 4E)
        else:
            pair_feat = torch.cat([s, d], dim=-1) # (B, N, 2E)

        return self.pair_mlp(pair_feat)        # (B, N, pair_dim)


class RoutedMoESymmetricPerEntityMLP(nn.Module):
    """Routes each state through a soft mixture of symmetric pair experts."""

    def __init__(self, config: ModelConfig):
        super().__init__()

        if config.state_embedding_head is not None:
            router_in_dim = config.state_embedding_head_out_dim
        elif config.state_features_embedding_head is not None:
            router_in_dim = config.state_features_embedding_head_out_dim
        else:
            raise ValueError(
                "moe-symmetric-per-entity-mlp requires a state_embedding_head "
                "and/or state_features_embedding_head for routing."
            )

        if config.moe_num_experts <= 0:
            raise ValueError("moe_num_experts must be positive.")
        if config.moe_top_k <= 0:
            raise ValueError("moe_top_k must be positive.")
        if config.moe_top_k > config.moe_num_experts:
            raise ValueError("moe_top_k cannot exceed moe_num_experts.")

        self.experts = nn.ModuleList(
            [SymmetricPerEntityMLP(config) for _ in range(config.moe_num_experts)]
        )
        self.num_experts = int(config.moe_num_experts)
        self.top_k = int(config.moe_top_k)
        self.out_dim = int(config.pair_embedding_head_out_dim)
        self.router = make_mlp(
            in_dim=router_in_dim,
            n_layers=config.moe_router_n_layers,
            hidden_dims=config.moe_router_hidden_dims,
            out_dim=config.moe_num_experts,
            activation=config.moe_router_activation,
            layer_norm=config.moe_router_layer_norm,
        )

    def forward(self, inputs: ModelInputs, router_context: torch.Tensor) -> torch.Tensor:
        if router_context is None:
            raise ValueError(
                "moe-symmetric-per-entity-mlp requires router_context from the "
                "state and/or state feature heads."
            )
        router_logits = self.router(router_context)
        topk_logits, topk_indices = torch.topk(
            router_logits,
            k=self.top_k,
            dim=-1,
        )
        topk_weights = F.softmax(topk_logits, dim=-1)

        batch_size, n_pairs = inputs.P.shape[:2]
        mixed_output = inputs.P.new_zeros((batch_size, n_pairs, self.out_dim))

        for expert_idx, expert in enumerate(self.experts):
            expert_selection = (topk_indices == expert_idx)
            if not bool(expert_selection.any()):
                continue

            batch_mask = expert_selection.any(dim=1)
            batch_indices = torch.nonzero(batch_mask, as_tuple=False).reshape(-1)
            expert_inputs = _slice_model_inputs(inputs, batch_indices)
            expert_output = expert(expert_inputs)
            expert_weights = (
                topk_weights.index_select(0, batch_indices)
                * expert_selection.index_select(0, batch_indices).to(topk_weights.dtype)
            ).sum(dim=-1)
            mixed_output[batch_indices] += expert_output * expert_weights[:, None, None]

        return mixed_output


class PolynomialSequenceEncoder(nn.Module):
    """Encodes one embedding per polynomial from either fixed-width vectors or term sequences."""

    def __init__(self, config: ModelConfig, encoder_type: str, out_dim: int):
        super().__init__()
        self.encoder_type = encoder_type
        self.out_dim = out_dim
        self.input_dim = config.state_embedding_head_in_dim
        proj_n_layers, proj_hidden_dims, proj_activation, proj_layer_norm, _ = _polynomial_projection_hparams(config)

        if self.encoder_type == "mlp":
            self.encoder = make_mlp(
                in_dim=self.input_dim,
                n_layers=proj_n_layers,
                hidden_dims=proj_hidden_dims,
                out_dim=out_dim,
                activation=proj_activation,
                layer_norm=proj_layer_norm,
            )
        elif self.encoder_type == "rnn":
            rnn_input_dim = self.input_dim
            hidden_dim = config.polynomial_rnn_hidden_dim
            num_directions = 2 if config.polynomial_rnn_bidirectional else 1
            dropout = (
                config.polynomial_rnn_dropout
                if config.polynomial_rnn_num_layers > 1
                else 0.0
            )
            rnn_cls = nn.GRU if config.polynomial_rnn_cell == "gru" else nn.LSTM
            self.encoder = rnn_cls(
                input_size=rnn_input_dim,
                hidden_size=hidden_dim,
                num_layers=config.polynomial_rnn_num_layers,
                batch_first=True,
                bidirectional=config.polynomial_rnn_bidirectional,
                dropout=dropout,
            )
            self.readout = make_mlp(
                in_dim=hidden_dim * num_directions,
                n_layers=proj_n_layers,
                hidden_dims=proj_hidden_dims,
                out_dim=out_dim,
                activation=proj_activation,
                layer_norm=proj_layer_norm,
            )
            self.post_readout_norm = (
                nn.LayerNorm(out_dim)
                if config.polynomial_rnn_layer_norm
                else nn.Identity()
            )
        else:
            raise ValueError(f"Unknown polynomial encoder type: {self.encoder_type}")

    def forward(self, inputs: ModelInputs) -> torch.Tensor:
        if self.encoder_type == "mlp":
            if inputs.G is None:
                raise ValueError("MLP polynomial encoder requires G in ModelInputs.")
            return self.encoder(inputs.G)

        if inputs.G_poly_terms is None or inputs.G_poly_term_mask is None:
            raise ValueError(
                "RNN polynomial encoder requires G_poly_terms and G_poly_term_mask in ModelInputs."
            )
        terms = inputs.G_poly_terms
        term_mask = inputs.G_poly_term_mask.bool()
        batch_size, num_polys, max_terms, term_dim = terms.shape

        flat_terms = terms.reshape(batch_size * num_polys, max_terms, term_dim)
        flat_mask = term_mask.reshape(batch_size * num_polys, max_terms)
        lengths = flat_mask.sum(dim=-1)
        valid = lengths > 0

        flat_encoded = flat_terms.new_zeros((batch_size * num_polys, self.out_dim))
        if not valid.any():
            return flat_encoded.reshape(batch_size, num_polys, self.out_dim)

        valid_terms = flat_terms[valid]
        valid_lengths = lengths[valid].to(dtype=torch.long)
        packed = pack_padded_sequence(
            valid_terms,
            valid_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        rnn_out = self.encoder(packed)
        hidden = rnn_out[1][0] if isinstance(rnn_out[1], tuple) else rnn_out[1]
        num_layers_times_dirs, valid_batch, hidden_dim = hidden.shape
        num_directions = num_layers_times_dirs // self.encoder.num_layers
        hidden = hidden.view(self.encoder.num_layers, num_directions, valid_batch, hidden_dim)
        last_hidden = hidden[-1].transpose(0, 1).reshape(valid_batch, num_directions * hidden_dim)
        encoded = self.post_readout_norm(self.readout(last_hidden))
        flat_encoded[valid] = encoded
        return flat_encoded.reshape(batch_size, num_polys, self.out_dim)


class PolynomialSetStateEncoder(nn.Module):
    """Pools polynomial embeddings into a global state embedding."""

    def __init__(self, config: ModelConfig, encoder_type: str):
        super().__init__()
        self.pooling = config.deep_set_pooling
        _, _, _, _, poly_embedding_dim = _polynomial_projection_hparams(config)
        self.polynomial_encoder = PolynomialSequenceEncoder(
            config=config,
            encoder_type=encoder_type,
            out_dim=poly_embedding_dim,
        )
        self.pool_norm = (
            nn.LayerNorm(poly_embedding_dim)
            if config.deep_set_post_pool_layer_norm
            else nn.Identity()
        )
        self.state_readout = make_mlp(
            in_dim=poly_embedding_dim,
            n_layers=config.deep_set_rho_n_layers,
            hidden_dims=config.deep_set_rho_hidden_dims,
            out_dim=config.state_embedding_head_out_dim,
            activation=config.deep_set_rho_activation,
            layer_norm=config.deep_set_rho_layer_norm,
        )

    def _pool(self, encoded_polynomials: torch.Tensor, g_mask: torch.Tensor) -> torch.Tensor:
        masked_polynomials = encoded_polynomials * g_mask.unsqueeze(-1).float()
        pooled_sum = masked_polynomials.sum(dim=1)

        if self.pooling == "sum":
            pooled = pooled_sum
        else:
            counts = g_mask.unsqueeze(-1).float().sum(dim=1).clamp_min(1.0)
            if self.pooling == "mean":
                pooled = pooled_sum / counts
            elif self.pooling == "sqrt":
                pooled = pooled_sum / counts.sqrt()
            else:
                raise RuntimeError("Unreachable pooling mode")

        return self.state_readout(self.pool_norm(pooled))

    def forward(self, inputs: ModelInputs):
        if inputs.G_mask is None:
            raise ValueError("Polynomial set state encoder requires G_mask in ModelInputs.")
        g_mask = inputs.G_mask.bool()
        encoded_polynomials = self.polynomial_encoder(inputs)
        encoded_polynomials = encoded_polynomials * g_mask.unsqueeze(-1).float()
        return self._pool(encoded_polynomials, g_mask)


class SharedPolynomialActorHead(nn.Module):
    """Shared polynomial/item encoder for the actor's state and pair heads.

    This reuses a single per-polynomial encoder over the basis set `G`, then:
    - pools those encoded polynomials into a global state embedding
    - gathers encoded endpoints via `P_idx` to build per-pair embeddings
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        if config.state_embedding_head_in_dim is None:
            raise ValueError("Shared polynomial actor head requires state_embedding_head_in_dim.")
        if config.pair_embedding_head_in_dim is not None and config.pair_embedding_head_in_dim % 2 != 0:
            raise ValueError("pair_embedding_head_in_dim must be even.")

        self.encoder_type = config.shared_polynomial_encoder_type
        self.poly_dim = config.state_embedding_head_in_dim
        self.item_out_dim = config.item_embedding_head_out_dim
        self.use_lcm_gcd = config.use_lcm_gcd
        self.pooling = config.deep_set_pooling

        self.polynomial_encoder = PolynomialSequenceEncoder(
            config=config,
            encoder_type=self.encoder_type,
            out_dim=self.item_out_dim,
        )

        self.pool_norm = (
            nn.LayerNorm(self.item_out_dim)
            if config.deep_set_post_pool_layer_norm
            else nn.Identity()
        )

        self.state_readout = make_mlp(
            in_dim=self.item_out_dim,
            n_layers=config.deep_set_rho_n_layers,
            hidden_dims=config.deep_set_rho_hidden_dims,
            out_dim=config.state_embedding_head_out_dim,
            activation=config.deep_set_rho_activation,
            layer_norm=config.deep_set_rho_layer_norm,
        )

        self.raw_term_encoder = None
        if self.use_lcm_gcd:
            self.raw_term_encoder = make_mlp(
                in_dim=self.poly_dim,
                n_layers=config.item_embedding_head_n_layers,
                hidden_dims=config.item_embedding_head_hidden_dims,
                out_dim=self.item_out_dim,
                activation=config.item_embedding_head_activation,
                layer_norm=config.item_embedding_head_layer_norm,
            )

        pair_in_dim = 4 * self.item_out_dim if self.use_lcm_gcd else 2 * self.item_out_dim
        self.pair_readout = make_mlp(
            in_dim=pair_in_dim,
            n_layers=config.pair_embedding_head_n_layers,
            hidden_dims=config.pair_embedding_head_hidden_dims,
            out_dim=config.pair_embedding_head_out_dim,
            activation=config.pair_embedding_head_activation,
            layer_norm=config.pair_embedding_head_layer_norm,
        )

    def _pool_state_embedding(
        self,
        encoded_polynomials: torch.Tensor,
        g_mask: torch.Tensor,
    ) -> torch.Tensor:
        masked_polynomials = encoded_polynomials * g_mask.unsqueeze(-1).float()
        pooled_sum = masked_polynomials.sum(dim=1)

        if self.pooling == "sum":
            pooled = pooled_sum
        else:
            counts = g_mask.unsqueeze(-1).float().sum(dim=1).clamp_min(1.0)
            if self.pooling == "mean":
                pooled = pooled_sum / counts
            elif self.pooling == "sqrt":
                pooled = pooled_sum / counts.sqrt()
            else:
                raise RuntimeError("Unreachable pooling mode")

        pooled = self.pool_norm(pooled)
        return self.state_readout(pooled)

    def _build_pair_embedding(
        self,
        inputs: ModelInputs,
        encoded_polynomials: torch.Tensor,
        g_mask: torch.Tensor,
    ) -> torch.Tensor:
        if inputs.P_idx is None:
            raise ValueError("Shared polynomial actor head requires P_idx in ModelInputs.")
        if inputs.P_mask is None:
            raise ValueError("Shared polynomial actor head requires P_mask in ModelInputs.")

        _, num_basis, item_dim = encoded_polynomials.shape
        idx_i_raw = inputs.P_idx[..., 0].long()
        idx_j_raw = inputs.P_idx[..., 1].long()

        idx_i = idx_i_raw.clamp(0, max(num_basis - 1, 0))
        idx_j = idx_j_raw.clamp(0, max(num_basis - 1, 0))

        gather_shape = (-1, -1, item_dim)
        poly_i = torch.gather(
            encoded_polynomials,
            1,
            idx_i.unsqueeze(-1).expand(*gather_shape),
        )
        poly_j = torch.gather(
            encoded_polynomials,
            1,
            idx_j.unsqueeze(-1).expand(*gather_shape),
        )

        p_mask = inputs.P_mask.bool()
        idx_i_in_bounds = (idx_i_raw >= 0) & (idx_i_raw < num_basis)
        idx_j_in_bounds = (idx_j_raw >= 0) & (idx_j_raw < num_basis)
        poly_i_valid = torch.gather(g_mask, 1, idx_i)
        poly_j_valid = torch.gather(g_mask, 1, idx_j)
        valid_pair = p_mask & idx_i_in_bounds & idx_j_in_bounds & poly_i_valid & poly_j_valid

        pair_features = [poly_i + poly_j, torch.abs(poly_i - poly_j)]

        if self.use_lcm_gcd:
            if inputs.P is None:
                raise ValueError(
                    "Shared polynomial actor head with use_lcm_gcd=True requires P in ModelInputs."
                )
            p1 = inputs.P[..., :self.poly_dim]
            p2 = inputs.P[..., self.poly_dim:]
            pair_features.append(self.raw_term_encoder(torch.maximum(p1, p2)))
            pair_features.append(self.raw_term_encoder(torch.minimum(p1, p2)))

        pair_embedding = self.pair_readout(torch.cat(pair_features, dim=-1))
        return pair_embedding * valid_pair.unsqueeze(-1).float()

    def forward(self, inputs: ModelInputs):
        if inputs.G is None or inputs.G_mask is None:
            raise ValueError("Shared polynomial actor head requires G and G_mask in ModelInputs.")

        g_mask = inputs.G_mask.bool()
        encoded_polynomials = self.polynomial_encoder(inputs)
        encoded_polynomials = encoded_polynomials * g_mask.unsqueeze(-1).float()

        state_embedding = self._pool_state_embedding(encoded_polynomials, g_mask)
        pair_embedding = self._build_pair_embedding(inputs, encoded_polynomials, g_mask)

        return state_embedding, pair_embedding


class PolynomialRNNDeepSetStateEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = PolynomialSetStateEncoder(config, encoder_type="rnn")

    def forward(self, inputs: ModelInputs):
        return self.net(inputs)

class MAB(nn.Module): # Multiheaded Attention Block
    def __init__(self, dim_Q, dim_KV, dim_hidden, num_heads, ln=True):
        super().__init__()
        self.num_heads = num_heads
        self.dim_hidden = dim_hidden

        self.fc_q = nn.Linear(dim_Q, dim_hidden)
        self.fc_k = nn.Linear(dim_KV, dim_hidden)
        self.fc_v = nn.Linear(dim_KV, dim_hidden)

        self.attn = nn.MultiheadAttention(
            dim_hidden,
            num_heads,
            batch_first=True,
        )

        self.fc_o = nn.Linear(dim_hidden, dim_hidden)

        self.ln0 = nn.LayerNorm(dim_hidden) if ln else nn.Identity()
        self.ln1 = nn.LayerNorm(dim_hidden) if ln else nn.Identity()

    def forward(self, Q, K, mask=None, need_weights=False):
        """
        Q: [B, Nq, d]
        K: [B, Nk, d]
        mask: [B, Nk] True = valid
        """
        Qh = self.fc_q(Q)
        Kh = self.fc_k(K)
        Vh = self.fc_v(K)

        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask  # PyTorch expects True = ignore

        attn_out, attn_weights = self.attn(
            Qh, Kh, Vh,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=False,
        )

        H = self.ln0(Qh + attn_out)
        O = self.ln1(H + self.fc_o(F.relu(H)))
        if need_weights:
            return O, attn_weights
        return O

class ISAB(nn.Module): # Induced Set Attention Block
    def __init__(self, dim_in, dim_hidden, num_heads, num_inds, layer_norm):
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(1, num_inds, dim_hidden))

        self.mab0 = MAB(dim_hidden, dim_in, dim_hidden, num_heads, ln=layer_norm)
        self.mab1 = MAB(dim_in, dim_hidden, dim_hidden, num_heads, ln=layer_norm)

    def forward(self, X, mask=None, need_weights=False):
        B = X.size(0)
        I = self.inducing.expand(B, -1, -1)

        if need_weights:
            H, attn0 = self.mab0(I, X, mask=mask, need_weights=True)
            O, attn1 = self.mab1(X, H, need_weights=True)
            return O, {"mab0": attn0, "mab1": attn1}

        H = self.mab0(I, X, mask=mask)
        return self.mab1(X, H)
    
class EquivariantSetTransformer(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
    ):
        super().__init__()

        self.input_proj = nn.Linear(config.pair_embedding_head_in_dim, config.equivariant_st_hidden_dim)

        if config.equivariant_st_num_inds is not None:
            self.encoder = nn.ModuleList([
                ISAB(config.equivariant_st_hidden_dim, config.equivariant_st_hidden_dim, config.equivariant_st_num_heads, config.equivariant_st_num_inds, layer_norm=config.equivariant_st_layer_norm)
                for _ in range(config.equivariant_st_num_ab)
            ])
        else:
            self.encoder = nn.ModuleList([
                MAB(config.equivariant_st_hidden_dim, config.equivariant_st_hidden_dim, config.equivariant_st_hidden_dim, config.equivariant_st_num_heads, ln=config.equivariant_st_layer_norm)
                for _ in range(config.equivariant_st_num_ab)
            ])
        self.readout = make_mlp(
            in_dim=config.equivariant_st_hidden_dim,
            n_layers=config.equivariant_st_final_n_layers,
            hidden_dims=config.equivariant_st_final_hidden_dims,
            out_dim=config.pair_embedding_head_out_dim,
            activation=config.equivariant_st_final_activation,
            layer_norm=config.equivariant_st_final_layer_norm
        )

    def forward(self, inputs: ModelInputs, return_attention: bool = False):
        """
        node_feat: [B, N_max, raw_pair_dim]
        node_mask: [B, N_max] (True = valid)

        Returns:
            state_emb: [B, N_max, pair_embedding_dim]
        """

        H = self.input_proj(inputs.P)

        H = H * inputs.P_mask.unsqueeze(-1)  # Mask out invalid pairs

        p_mask = inputs.P_mask.bool()
        attention_maps = {}

        for idx, layer in enumerate(self.encoder):
            if return_attention:
                if isinstance(layer, ISAB):
                    layer_out = layer(H, p_mask, need_weights=True)
                    H, layer_attn = layer_out
                    attention_maps[f"encoder_{idx}_mab0"] = {
                        "weights": layer_attn["mab0"],
                        "q_mask": None,
                        "k_mask": p_mask,
                    }
                    attention_maps[f"encoder_{idx}_mab1"] = {
                        "weights": layer_attn["mab1"],
                        "q_mask": p_mask,
                        "k_mask": None,
                    }
                else:
                    H, layer_attn = layer(H, H, mask=p_mask, need_weights=True)
                    attention_maps[f"encoder_{idx}"] = {
                        "weights": layer_attn,
                        "q_mask": p_mask,
                        "k_mask": p_mask,
                    }
            else:
                if isinstance(layer, ISAB):
                    H = layer(H, p_mask)
                else:
                    H = layer(H, H, mask=p_mask)

        out = self.readout(H)
        if return_attention:
            return out, attention_maps
        return out

class PairEmbeddingHead(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        if config.pair_embedding_head == 'per-entity-mlp':
            self.net = PerEntityMLP(config=config)
        elif config.pair_embedding_head == 'symmetric-per-entity-mlp':
            self.net = SymmetricPerEntityMLP(config=config)
        elif config.pair_embedding_head == 'moe-symmetric-per-entity-mlp':
            self.net = RoutedMoESymmetricPerEntityMLP(config=config)
        elif config.pair_embedding_head == 'equivariant-set-transformer':
            self.net = EquivariantSetTransformer(config=config)
        else:
            raise ValueError(f"Unknown pair_embedding_head: {config.pair_embedding_head}")

    def forward(
        self,
        inputs: ModelInputs,
        return_attention: bool = False,
        router_context: torch.Tensor | None = None,
    ):

        attention_maps = None
        if return_attention and isinstance(self.net, EquivariantSetTransformer):
            network_output, attention_maps = self.net(inputs, return_attention=True)
        elif isinstance(self.net, RoutedMoESymmetricPerEntityMLP):
            network_output = self.net(inputs, router_context=router_context)
        else:
            network_output = self.net(inputs)

        # Do masking here
        P_mask = inputs.P_mask.bool()
        mask_p = P_mask.unsqueeze(-1).float()
        masked_output = network_output * mask_p

        if return_attention:
            return masked_output, attention_maps
        return masked_output
    
class StateFeaturesEmbeddingHead(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        if config.state_features_embedding_head == 'mlp':
            self.net = make_mlp(
                in_dim=config.state_features_embedding_head_in_dim,
                n_layers=config.state_features_embedding_head_n_layers,
                hidden_dims=config.state_features_embedding_head_hidden_dims,
                out_dim=config.state_features_embedding_head_out_dim,
                activation=config.state_features_embedding_head_activation,
                layer_norm=config.state_features_embedding_head_layer_norm
            )
        else:
            raise ValueError(
                f"Unknown state_features_embedding_head: "
                f"{config.state_features_embedding_head}"
            )


    def forward(self, inputs: ModelInputs):
        return self.net(inputs.state_features)
    
class DeepSetNetwork(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.phi_out_dim = config.deep_set_phi_out_dim
        self.pooling = config.deep_set_pooling

        self.phi_mlp = make_mlp(
            in_dim=config.state_embedding_head_in_dim,
            n_layers=config.deep_set_phi_n_layers,
            hidden_dims=config.deep_set_phi_hidden_dims,
            out_dim=config.deep_set_phi_out_dim,
            activation=config.deep_set_phi_activation,
            layer_norm=config.deep_set_phi_layer_norm,
        )

        if config.deep_set_post_pool_layer_norm:
            self.pool_norm = nn.LayerNorm(config.deep_set_phi_out_dim)
        else:
            self.pool_norm = nn.Identity()

        self.rho_mlp = make_mlp(
            in_dim=config.deep_set_phi_out_dim,
            n_layers=config.deep_set_rho_n_layers,
            hidden_dims=config.deep_set_rho_hidden_dims,
            out_dim=config.state_embedding_head_out_dim,
            activation=config.deep_set_rho_activation,
            layer_norm=config.deep_set_rho_layer_norm,
        )

    def forward(self, inputs: ModelInputs):
        B, N, D = inputs.G.shape

        g_flat = inputs.G.reshape(B * N, D)

        phi = self.phi_mlp(g_flat).reshape(B, N, self.phi_out_dim)

        G_mask = inputs.G_mask.bool()
        mask_g = G_mask.unsqueeze(-1).float()

        phi_masked = phi * mask_g

        pooled_sum = phi_masked.sum(dim=1)

        if self.pooling == "sum":
            pooled = pooled_sum
        else:
            counts = mask_g.sum(dim=1).clamp_min(1.0)  # (B,1)
            if self.pooling == "mean":
                pooled = pooled_sum / counts
            elif self.pooling == "sqrt":
                pooled = pooled_sum / counts.sqrt()
            else:
                raise RuntimeError("Unreachable pooling mode")
            
        pooled = self.pool_norm(pooled)

        rho = self.rho_mlp(pooled)

        return rho

class EdgeGatedBlock(nn.Module):
    def __init__(
        self,
        config: ModelConfig
    ):
        super().__init__()

        self.msg_lin = nn.Linear(config.gnn_node_encoding_dim, config.gnn_node_encoding_dim)

        self.gate_mlp = make_mlp(
            in_dim=config.gnn_node_encoding_dim * 2 + config.gnn_edge_encoding_dim,
            n_layers=config.gnn_n_layers,
            hidden_dims=config.gnn_hidden_dims,
            out_dim=1,
            activation=config.gnn_activation,
            layer_norm=config.gnn_layer_norm_mlp,
        )

        self.node_mlp = make_mlp(
            in_dim=config.gnn_node_encoding_dim * 2,
            n_layers=config.gnn_n_layers,
            hidden_dims=config.gnn_hidden_dims,
            out_dim=config.gnn_node_encoding_dim,
            activation=config.gnn_activation,
            layer_norm=config.gnn_layer_norm_mlp,
        )

        self.norm = nn.LayerNorm(config.gnn_node_encoding_dim)

    def forward(self, h, edge_index, e):
        src, dst = edge_index.long()

        h_src = h[src]
        h_dst = h[dst]

        m = self.msg_lin(h_src)

        gate_input = torch.cat([h_src + h_dst, torch.abs(h_src - h_dst), e], dim=-1) # gating is symmetric
        alpha = torch.sigmoid(self.gate_mlp(gate_input))

        m = m * alpha
        agg = scatter_sum(m, dst, h.size(0))

        h_update = self.node_mlp(torch.cat([h, agg], dim=-1))
        h = self.norm(h + h_update)

        return h, alpha

class EdgeGatedGNN(nn.Module):
    def __init__(
        self,
        config: ModelConfig
    ):
        super().__init__()

        self.make_undirected = config.gnn_make_undirected

        self.node_encoder = make_mlp(
            in_dim=config.state_embedding_head_in_dim,
            n_layers=config.gnn_n_layers,
            hidden_dims=config.gnn_hidden_dims,
            out_dim=config.gnn_node_encoding_dim,
            activation=config.gnn_activation,
            layer_norm=config.gnn_layer_norm_mlp,
        )

        self.edge_encoder = make_mlp(
            in_dim=config.state_embedding_head_in_dim,
            n_layers=config.gnn_n_layers,
            hidden_dims=config.gnn_hidden_dims,
            out_dim=config.gnn_edge_encoding_dim,
            activation=config.gnn_activation,
            layer_norm=config.gnn_layer_norm_mlp,
        )

        self.layers = nn.ModuleList([
            EdgeGatedBlock(
                config
            )
            for _ in range(config.gnn_message_passing_layers)
        ])

        self.readout = make_mlp(
            in_dim=config.gnn_node_encoding_dim * 3,
            n_layers=config.gnn_n_layers,
            hidden_dims=config.gnn_hidden_dims,
            out_dim=config.state_embedding_head_out_dim,
            activation=config.gnn_activation,
            layer_norm=config.gnn_layer_norm_mlp,
        )

    def forward(
        self,
        inputs: ModelInputs,
        return_last_alpha: bool = False,
        return_all_alpha: bool = False,
    ):
        # -------------------------------
        # Remove padded nodes
        # -------------------------------
        h, node_batch = flatten_valid_nodes(inputs.G, inputs.G_mask)

        # -------------------------------
        # Construct mask 
        # -------------------------------
        invalid_pairs = torch.any((inputs.P_idx < 0) | (inputs.P_idx >= inputs.G.shape[1]), dim=2)
        pair_mask = torch.logical_and(inputs.P_mask, ~invalid_pairs)
        # -------------------------------
        # Build edges
        # -------------------------------
        num_nodes = inputs.G_mask.sum(dim=1)

        edge_index, edge_batch = build_batched_edge_index(
            inputs.P_idx,
            pair_mask,
            num_nodes,
            make_undirected=self.make_undirected,
        )

        # -------------------------------
        # Construct pair features for edges
        # -------------------------------

        x1 = inputs.P[..., :inputs.G.shape[-1]]
        x2 = inputs.P[..., inputs.G.shape[-1]:]
        pair_feat = torch.maximum(x1, x2)

        # -------------------------------
        # Flatten edge features
        # -------------------------------
        e = pair_feat.view(-1, pair_feat.size(-1))[pair_mask.view(-1)]
        if self.make_undirected:
            e = torch.cat([e, e], dim=0)

        # -------------------------------
        # Encode
        # -------------------------------
        h = self.node_encoder(h)
        e = self.edge_encoder(e)

        # -------------------------------
        # Message passing
        # -------------------------------
        last_alpha = None
        all_alpha = []
        for layer in self.layers:
            h, alpha = layer(h, edge_index, e)
            last_alpha = alpha
            if return_all_alpha:
                all_alpha.append(alpha)

        # -------------------------------
        # Global pooling
        # -------------------------------
        graph_emb = global_pool_batch(h, node_batch)
        state_emb = self.readout(graph_emb)

        if return_all_alpha:
            return state_emb, all_alpha
        if return_last_alpha:
            return state_emb, last_alpha
        return state_emb

class SimpleGATLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.pair_gnn_node_encoding_dim
        self.n_heads = config.pair_gnn_n_heads
        self.d_head = self.d_model // self.n_heads

        assert self.d_model % self.n_heads == 0

        self.W_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_v = nn.Linear(self.d_model, self.d_model, bias=False)

        self.out_proj = nn.Linear(self.d_model, self.d_model)

    def forward(self, h, edge_index):
        """
        h: (N, d_model)
        edge_index: (2, E) — COO format
        """

        N = h.size(0)
        src, dst = edge_index  # j -> i

        Q = self.W_q(h).view(N, self.n_heads, self.d_head)
        K = self.W_k(h).view(N, self.n_heads, self.d_head)
        V = self.W_v(h).view(N, self.n_heads, self.d_head)

        # gather edge endpoints
        q_i = Q[dst]  # (E, heads, d_head)
        k_j = K[src]
        v_j = V[src]

        # attention scores
        scores = (q_i * k_j).sum(-1) / (self.d_head ** 0.5)

        # softmax per destination node
        alpha = torch.zeros_like(scores)
        for i in range(N):
            mask = (dst == i)
            if mask.any():
                alpha[mask] = F.softmax(scores[mask], dim=0)

        # aggregate
        messages = alpha.unsqueeze(-1) * v_j

        out = torch.zeros(N, self.n_heads, self.d_head, device=h.device)
        out.index_add_(0, dst, messages)

        out = out.reshape(N, self.d_model)
        return self.out_proj(out)
    
class EquivariantPairsetGNN(nn.Module):
    def __init__(
        self,
        config: ModelConfig
    ):
        super().__init__()

        self.make_undirected = config.pair_gnn_make_undirected

        self.node_encoder = make_mlp(
            in_dim=config.pair_embedding_head_in_dim,
            n_layers=config.pair_gnn_n_layers,
            hidden_dims=config.pair_gnn_hidden_dims,
            out_dim=config.pair_gnn_node_embedding_dim,
            activation=config.pair_gnn_activation,
            layer_norm=config.pair_gnn_layer_norm_mlp,
        )

        self.layers = nn.ModuleList([
            SimpleGATLayer(config)
            for _ in range(config.pair_gnn_message_passing_layers)
        ])

    def forward(
        self,
        inputs: ModelInputs,
        return_last_alpha: bool = False,
        return_all_alpha: bool = False,
    ):
        # -------------------------------
        # Remove padded nodes
        # -------------------------------
        h, node_batch = flatten_valid_nodes(inputs.P, inputs.P_mask)

        # -------------------------------
        # Build edges
        # -------------------------------

        edge_index = build_pair_graph_edge_index(
            inputs.P_idx,
            inputs.P_mask,
            make_undirected=self.make_undirected,
        )

        # -------------------------------
        # Encode
        # -------------------------------
        h = self.node_encoder(h)

        # -------------------------------
        # Message passing
        # -------------------------------
        last_alpha = None
        all_alpha = []
        for layer in self.layers:
            h, alpha = layer(h, edge_index)
            last_alpha = alpha
            if return_all_alpha:
                all_alpha.append(alpha)

        if return_all_alpha:
            return h, all_alpha
        if return_last_alpha:
            return h, last_alpha
        return h
    
class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds=1):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, num_seeds, dim))
        self.mab = MAB(dim, dim, dim, num_heads)

    def forward(self, X, mask=None, return_attention: bool = False):
        B = X.size(0)
        S = self.seed.expand(B, -1, -1)
        if return_attention:
            return self.mab(S, X, mask=mask, need_weights=True)
        return self.mab(S, X, mask=mask)
    
class InvariantSetTransformer(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
    ):
        super().__init__()

        self.input_proj = nn.Linear(config.state_embedding_head_in_dim, config.invariant_st_hidden_dim)

        if config.invariant_st_num_inds is not None:
            self.encoder = nn.ModuleList([
                ISAB(config.invariant_st_hidden_dim, config.invariant_st_hidden_dim, config.invariant_st_num_heads, config.invariant_st_num_inds, layer_norm=config.invariant_st_layer_norm)
                for _ in range(config.invariant_st_num_ab)
            ])
        else:
            self.encoder = nn.ModuleList([
                MAB(config.invariant_st_hidden_dim, config.invariant_st_hidden_dim, config.invariant_st_hidden_dim, config.invariant_st_num_heads, ln=config.invariant_st_layer_norm)
                for _ in range(config.invariant_st_num_ab)
            ])

        # Allow encoder hidden size and PMA hidden size to be tuned independently.
        self.pre_pma_proj = (
            nn.Linear(config.invariant_st_hidden_dim, config.invariant_st_pma_hidden_dim)
            if config.invariant_st_hidden_dim != config.invariant_st_pma_hidden_dim
            else nn.Identity()
        )

        self.pma = PMA(config.invariant_st_pma_hidden_dim, config.invariant_st_pma_num_heads, num_seeds=1)

        self.readout = make_mlp(
            in_dim=config.invariant_st_pma_hidden_dim,
            n_layers=config.invariant_st_final_n_layers,
            hidden_dims=config.invariant_st_final_hidden_dims,
            out_dim=config.state_embedding_head_out_dim,
            activation=config.invariant_st_final_activation,
            layer_norm=config.invariant_st_final_layer_norm
        )

    def forward(self, inputs: ModelInputs, return_attention: bool = False):
        """
        node_feat: [B, N_max, node_dim]
        node_mask: [B, N_max] (True = valid)

        Returns:
            state_emb: [B, state_dim]
        """

        g_mask = inputs.G_mask.bool()
        H = self.input_proj(inputs.G)

        attention_maps = {}
        for idx, layer in enumerate(self.encoder):
            if return_attention:
                if isinstance(layer, ISAB):
                    layer_out = layer(H, g_mask, need_weights=True)
                    H, layer_attn = layer_out
                    attention_maps[f"encoder_{idx}_mab0"] = {
                        "weights": layer_attn["mab0"],
                        "q_mask": None,
                        "k_mask": g_mask,
                    }
                    attention_maps[f"encoder_{idx}_mab1"] = {
                        "weights": layer_attn["mab1"],
                        "q_mask": g_mask,
                        "k_mask": None,
                    }
                else:
                    H, layer_attn = layer(H, H, mask=g_mask, need_weights=True)
                    attention_maps[f"encoder_{idx}"] = {
                        "weights": layer_attn,
                        "q_mask": g_mask,
                        "k_mask": g_mask,
                    }
            else:
                if isinstance(layer, ISAB):
                    H = layer(H, g_mask)
                else:
                    H = layer(H, H, mask=g_mask)

        H = self.pre_pma_proj(H)

        if return_attention:
            pooled, pma_attn = self.pma(H, g_mask, return_attention=True)  # [B, 1, d]
            attention_maps["pma"] = {
                "weights": pma_attn,
                "q_mask": None,
                "k_mask": g_mask,
            }
        else:
            pooled = self.pma(H, g_mask)  # [B, 1, d]
        pooled = pooled.squeeze(1)

        out = self.readout(pooled)
        if return_attention:
            return out, attention_maps
        return out

class StateEmbeddingHead(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        if config.state_embedding_head == 'deep-set':
            self.net = DeepSetNetwork(config)
        elif config.state_embedding_head == 'gnn':
            self.net = EdgeGatedGNN(config)
        elif config.state_embedding_head == 'invariant-set-transformer':
            self.net = InvariantSetTransformer(config)
        elif config.state_embedding_head == 'polynomial-rnn-deep-set':
            self.net = PolynomialRNNDeepSetStateEncoder(config)
        else:
            raise ValueError(
                f"Unknown state_embedding_head: "
                f"{config.state_embedding_head}"
            )

    def forward(self, inputs: ModelInputs, return_attention: bool = False):
        if not return_attention:
            return self.net(inputs)

        if isinstance(self.net, InvariantSetTransformer):
            state_emb, attention_maps = self.net(inputs, return_attention=True)
            return state_emb, attention_maps

        if isinstance(self.net, EdgeGatedGNN):
            state_emb, all_alpha = self.net(inputs, return_all_alpha=True)
            attention_maps = {}
            for layer_idx, layer_alpha in enumerate(all_alpha):
                attention_maps[f"edge_gated_layer_{layer_idx}"] = {
                    "type": "edge_gate",
                    "weights": layer_alpha,
                }
            return state_emb, attention_maps

        # No attention-like map for deep-set, but keep return signature consistent.
        state_emb = self.net(inputs)
        return state_emb, None

class FiLMConditioner(nn.Module):
    """
    Produces FiLM scale/shift from handcrafted state features.

    PPO-safe initialization:
    - gamma starts near 1
    - beta starts near 0
    """

    def __init__(self, feature_dim: int, n_layers: int, hidden_dims: int, out_dim: int, activation: str, layer_norm: bool):
        super().__init__()

        self.net = make_mlp(
            in_dim=feature_dim,
            n_layers=n_layers,
            hidden_dims=hidden_dims,
            out_dim=2 * out_dim,
            activation=activation,
            layer_norm=layer_norm
        )

        # PPO-safe init: start near identity modulation
        last_linear = next(
            module for module in reversed(self.net) if isinstance(module, nn.Linear)
        )
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

        self.out_dim = out_dim

    def forward(self, f):
        """
        f: (B, feature_dim)
        returns gamma, beta each (B, target_dim)
        """
        film = self.net(f)
        gamma, beta = film.chunk(2, dim=-1)

        gamma = 1.0 + gamma  # start near identity
        return gamma, beta

class Glimpse(nn.Module):
    def __init__(self, key_dim: int, query_dim: int, d: int = 128, attention_mode: str = "additive"):
        super().__init__()
        self.W_h = nn.Linear(key_dim, d, bias=False)
        self.W_q = nn.Linear(query_dim, d, bias=False)
        self.v = nn.Parameter(torch.randn(d))
        self.d = d
        self.attention_mode = attention_mode

    def forward(self, q, H, mask=None):
        """
        q    : (batch, query_dim)
        H    : (batch, n, key_dim)
        mask : (batch, n)  — True = keep, False = mask out
        """

        # Expand query over pair dimension while preserving its own feature width.
        q_expanded = q.unsqueeze(1).expand(-1, H.size(1), -1)
        h_proj = self.W_h(H)
        q_proj = self.W_q(q_expanded)

        if self.attention_mode == "additive":
            u = torch.tanh(h_proj + q_proj)
            scores = torch.matmul(u, self.v)  # (batch, n)
        elif self.attention_mode == "scaled dot-product":
            scores = (h_proj * q_proj).sum(dim=-1) / math.sqrt(self.d)

        if mask is not None:
            mask_bool = mask.to(dtype=torch.bool)
            has_valid = mask_bool.any(dim=-1, keepdim=True)
            # Use a large finite negative value to avoid softmax(-inf, -inf, ...) NaNs.
            scores = scores.masked_fill(~mask_bool, -1e9)
            attn = F.softmax(scores, dim=-1)  # (batch, n)
            # If an entire row is invalid, return a zero attention row.
            attn = torch.where(has_valid, attn, torch.zeros_like(attn))
        else:
            attn = F.softmax(scores, dim=-1)  # (batch, n)

        # Weighted sum over keys keeps key feature width.
        g = torch.bmm(attn.unsqueeze(1), H).squeeze(1)  # (batch, key_dim)

        return g
    
class MultiGlimpse(nn.Module):
    def __init__(self, key_dim: int, query_dim: int, d: int = 128, n_glimpses=1, attention_mode: str = "additive"):
        super().__init__()
        self.n_glimpses = n_glimpses
        self.glimpses = nn.ModuleList()

        if n_glimpses > 0:
            # First glimpse consumes the original query; subsequent glimpses consume key-sized vectors.
            self.glimpses.append(Glimpse(key_dim, query_dim, d, attention_mode))
            for _ in range(1, n_glimpses):
                self.glimpses.append(Glimpse(key_dim, key_dim, d, attention_mode))

    def forward(self, q, H, mask=None):
        """
        Iteratively refines the query.

        """
        if self.n_glimpses == 0:
            return q
        else:
            for glimpse in self.glimpses:
                q = glimpse(q, H, mask)
            return q

class QueryKeyPairScorer(nn.Module):
    def __init__(self, key_dim: int, query_dim: int, d: int = 128, bias: bool = False, norm_qk: bool = False, final_attention: str = "scaled dot-product", n_glimpses: int = 0, glimpse_attention: str = "additive"):
        super().__init__()
        self.d = d
        self.eps = 1e-8
        self.norm_qk = norm_qk
        q_dim_for_scoring = key_dim if n_glimpses > 0 else query_dim
        self.q_net = nn.Linear(q_dim_for_scoring, d, bias=True)
        self.k_net = nn.Linear(key_dim, d, bias=True)
        self.bias = nn.Parameter(torch.zeros(1)) if bias else None
        self.final_attention = final_attention
        self.glimpse = MultiGlimpse(key_dim, query_dim, d, n_glimpses, glimpse_attention)

        if self.final_attention == "additive":
            self.v = nn.Parameter(torch.randn(d))

    def forward(self, keys: torch.Tensor, query: torch.Tensor, mask: torch.tensor) -> torch.Tensor:

        q = self.glimpse(query, keys, mask)

        q = self.q_net(q)          # (B,d)
        k = self.k_net(keys)     # (B,N,d)

        if self.norm_qk:
            q = q / (q.norm(dim=-1, keepdim=True).clamp_min(self.eps))
            k = k / (k.norm(dim=-1, keepdim=True).clamp_min(self.eps))

        if self.final_attention == "scaled dot-product":
            logits = (k * q[:, None, :]).sum(dim=-1) if self.norm_qk else (k * q[:, None, :]).sum(dim=-1) / math.sqrt(self.d)
        elif self.final_attention == "additive":
            q_expanded = q.unsqueeze(1).expand_as(k)
            u = torch.tanh(k + q_expanded)
            logits = torch.matmul(u, self.v)
        
        if self.bias is not None:
            logits = logits + self.bias
        return logits.unsqueeze(-1)


class CrossConditioner(nn.Module):
    def __init__(
        self,
        pair_dim: int,
        context_dim: int,
        heads: int = 4,
        post_attn_layer_norm: bool = True,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=pair_dim,
            num_heads=heads,
            batch_first=True,
            kdim=context_dim,
            vdim=context_dim,
        )
        self.post_attn_norm = nn.LayerNorm(pair_dim) if post_attn_layer_norm else nn.Identity()

    def forward(self, pair_emb: torch.Tensor, global_emb: torch.Tensor) -> torch.Tensor:
        # pair_emb: [B, N, pair_dim], global_emb: [B, context_dim]
        context = global_emb.unsqueeze(1)
        attn_out, _ = self.attn(pair_emb, context, context)
        return self.post_attn_norm(pair_emb + attn_out)


class BasisCrossConditioner(nn.Module):
    """Lets each pair attend over the full visible basis set."""

    def __init__(
        self,
        pair_dim: int,
        basis_dim: int,
        heads: int = 4,
        post_attn_layer_norm: bool = True,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=pair_dim,
            num_heads=heads,
            batch_first=True,
            kdim=basis_dim,
            vdim=basis_dim,
        )
        self.post_attn_norm = (
            nn.LayerNorm(pair_dim) if post_attn_layer_norm else nn.Identity()
        )

    def forward(
        self,
        pair_emb: torch.Tensor,
        basis_nodes: torch.Tensor,
        basis_mask: torch.Tensor,
    ) -> torch.Tensor:
        basis_mask_bool = basis_mask.bool()
        key_padding_mask = ~basis_mask_bool

        if key_padding_mask.shape[1] == 0:
            return self.post_attn_norm(pair_emb)

        # Avoid all-masked rows causing NaNs inside attention softmax.
        has_valid_basis = basis_mask_bool.any(dim=-1)
        if not has_valid_basis.all():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[~has_valid_basis, 0] = False
            basis_nodes = basis_nodes.clone()
            basis_nodes[~has_valid_basis] = 0.0

        attn_out, _ = self.attn(
            pair_emb,
            basis_nodes,
            basis_nodes,
            key_padding_mask=key_padding_mask,
        )
        return self.post_attn_norm(pair_emb + attn_out)


class BipartiteMP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.g_to_p = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.p_to_g = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        self.g_norm = nn.LayerNorm(dim)
        self.p_norm = nn.LayerNorm(dim)

    def forward(
        self,
        g_emb: torch.Tensor,
        p_emb: torch.Tensor,
        pair_index: torch.Tensor,
        g_mask: torch.Tensor,
        p_mask: torch.Tensor,
    ):
        # g_emb: [B, G, d], p_emb: [B, P, d], pair_index: [B, P, 2]
        B, G, d = g_emb.shape

        idx_i_raw = pair_index[..., 0].long()
        idx_j_raw = pair_index[..., 1].long()

        idx_i = idx_i_raw.clamp(0, max(G - 1, 0))
        idx_j = idx_j_raw.clamp(0, max(G - 1, 0))

        g_i = torch.gather(g_emb, 1, idx_i.unsqueeze(-1).expand(-1, -1, d))
        g_j = torch.gather(g_emb, 1, idx_j.unsqueeze(-1).expand(-1, -1, d))

        g_mask_bool = g_mask.bool()
        p_mask_bool = p_mask.bool()

        idx_i_in_bounds = (idx_i_raw >= 0) & (idx_i_raw < G)
        idx_j_in_bounds = (idx_j_raw >= 0) & (idx_j_raw < G)
        gi_valid = torch.gather(g_mask_bool, 1, idx_i)
        gj_valid = torch.gather(g_mask_bool, 1, idx_j)
        valid_pair = p_mask_bool & idx_i_in_bounds & idx_j_in_bounds & gi_valid & gj_valid
        valid_pair_f = valid_pair.unsqueeze(-1).float()

        # Let the pair update depend on its current state as well as both endpoints.
        delta_p = self.g_to_p(torch.cat([p_emb, g_i, g_j], dim=-1)) * valid_pair_f
        p_emb = self.p_norm(p_emb + delta_p)
        p_emb = p_emb * p_mask_bool.unsqueeze(-1).float()

        msg_i = self.p_to_g(torch.cat([p_emb, g_i], dim=-1)) * valid_pair_f
        msg_j = self.p_to_g(torch.cat([p_emb, g_j], dim=-1)) * valid_pair_f

        agg = torch.zeros_like(g_emb)
        agg.scatter_add_(1, idx_i.unsqueeze(-1).expand(-1, -1, d), msg_i)
        agg.scatter_add_(1, idx_j.unsqueeze(-1).expand(-1, -1, d), msg_j)

        g_emb = self.g_norm(g_emb + agg)
        g_emb = g_emb * g_mask_bool.unsqueeze(-1).float()

        return g_emb, p_emb


class BuchbergerBipartite(nn.Module):
    def __init__(self, node_dim: int, pair_dim: int, hidden_dim: int = 128, layers: int = 3):
        super().__init__()

        self.g_proj = nn.Linear(node_dim, hidden_dim)
        self.p_proj = nn.Linear(pair_dim, hidden_dim)
        self.layers = nn.ModuleList([BipartiteMP(hidden_dim) for _ in range(layers)])
        self.pair_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        G_nodes: torch.Tensor,
        P_pairs: torch.Tensor,
        pair_index: torch.Tensor,
        G_mask: torch.Tensor,
        P_mask: torch.Tensor,
    ) -> torch.Tensor:
        g = self.g_proj(G_nodes)
        p = self.p_proj(P_pairs)

        for layer in self.layers:
            g, p = layer(g, p, pair_index, G_mask, P_mask)

        logits = self.pair_head(p)
        return logits * P_mask.unsqueeze(-1).float()


class BasisPairCombiner(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.conditioner = BasisCrossConditioner(
            pair_dim=config.pair_embedding_head_out_dim,
            basis_dim=config.state_embedding_head_in_dim,
            heads=config.cross_attn_num_heads,
            post_attn_layer_norm=config.cross_attn_post_attn_layer_norm,
        )
        self.final_mlp = make_mlp(
            in_dim=config.pair_embedding_head_out_dim,
            n_layers=config.concat_n_layers,
            hidden_dims=config.concat_hidden_dims,
            out_dim=1,
            activation=config.concat_activation,
            layer_norm=config.concat_layer_norm,
        )

    def forward(
        self,
        pair_embedding: torch.Tensor,
        model_inputs: ModelInputs,
    ) -> torch.Tensor:
        if model_inputs.G is None or model_inputs.G_mask is None or model_inputs.P_mask is None:
            raise ValueError("basis-cross-attn requires G, G_mask, and P_mask in ModelInputs.")

        pair_mod = self.conditioner(
            pair_embedding,
            model_inputs.G,
            model_inputs.G_mask,
        )
        logits = self.final_mlp(pair_mod)
        return logits * model_inputs.P_mask.unsqueeze(-1).float()

class StatePairCombiner(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.pair_combiner = config.pair_combiner
        
        if config.pair_combiner == 'sqa':
            self.sqa = QueryKeyPairScorer(
                key_dim=config.pair_embedding_head_out_dim,
                query_dim=config.state_embedding_head_out_dim,
                d=config.single_query_attention_dim,
                norm_qk=config.single_query_attention_norm_qk,
                final_attention=config.single_query_attention_final_attention,
                n_glimpses=config.single_query_attention_n_glimpses,
                glimpse_attention=config.single_query_glimpse_attention_mode)
        elif config.pair_combiner == 'film':
            self.film = FiLMConditioner(
                feature_dim=config.state_embedding_head_out_dim,
                n_layers=config.film_n_layers,
                hidden_dims=config.film_hidden_dims,
                out_dim=config.pair_embedding_head_out_dim,
                activation=config.film_activation,
                layer_norm=config.film_layer_norm
            )
            self.pair_mod_norm = nn.LayerNorm(config.pair_embedding_head_out_dim) if config.film_pair_mod_layer_norm else None
            self.final_mlp = make_mlp(
                in_dim=config.pair_embedding_head_out_dim,
                n_layers=config.film_final_n_layers,
                hidden_dims=config.film_final_hidden_dims,
                out_dim=1,
                activation=config.film_final_activation,
                layer_norm=config.film_final_layer_norm,
            )
        elif config.pair_combiner == 'concat':
            self.final_mlp = make_mlp(
                in_dim=config.pair_embedding_head_out_dim + config.state_embedding_head_out_dim,
                n_layers=config.concat_n_layers,
                hidden_dims=config.concat_hidden_dims,
                out_dim=1,
                activation=config.concat_activation,
                layer_norm=config.concat_layer_norm,
            )
        elif config.pair_combiner == 'cross-attn':
            self.cross_attn = CrossConditioner(
                pair_dim=config.pair_embedding_head_out_dim,
                context_dim=config.state_embedding_head_out_dim,
                heads=config.cross_attn_num_heads,
                post_attn_layer_norm=config.cross_attn_post_attn_layer_norm,
            )
            self.final_mlp = make_mlp(
                in_dim=config.pair_embedding_head_out_dim,
                n_layers=config.concat_n_layers,
                hidden_dims=config.concat_hidden_dims,
                out_dim=1,
                activation=config.concat_activation,
                layer_norm=config.concat_layer_norm,
            )
        elif config.pair_combiner == 'bipartite-mp':
            self.bipartite = BuchbergerBipartite(
                node_dim=config.state_embedding_head_in_dim,
                pair_dim=config.pair_embedding_head_out_dim,
                hidden_dim=config.bipartite_hidden_dim,
                layers=config.bipartite_layers,
            )
        else: 
            raise ValueError(
                f"Unknown combiner method: "
                f"{config.pair_combiner}"
            )
        
    def forward(self, state_embedding, pair_embedding, mask, model_inputs: Optional[ModelInputs] = None):

        mask_bool = mask.bool()
        mask = mask_bool.unsqueeze(-1).float()

        if self.pair_combiner == 'film':
            gamma, beta = self.film(state_embedding)  # (B, S)

            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)

            pair_mod = gamma * pair_embedding + beta  # (B, S)

            if self.pair_mod_norm is not None:
                pair_mod = self.pair_mod_norm(pair_mod)

            pair_out = self.final_mlp(pair_mod)

            logits = pair_out * mask
        elif self.pair_combiner == 'sqa':
            logits = self.sqa(pair_embedding, state_embedding, mask_bool)
        elif self.pair_combiner == 'concat':
            state_exp = state_embedding.unsqueeze(1).expand(pair_embedding.shape[0], pair_embedding.shape[1], state_embedding.shape[-1])            # (B,N,S)
            head_in = torch.cat([pair_embedding, state_exp], dim=-1)     
            pair_out = self.final_mlp(head_in)

            logits = pair_out * mask
        elif self.pair_combiner == 'cross-attn':
            pair_mod = self.cross_attn(pair_embedding, state_embedding)
            pair_out = self.final_mlp(pair_mod)
            logits = pair_out * mask
        elif self.pair_combiner == 'bipartite-mp':
            if model_inputs is None:
                raise ValueError("bipartite-mp combiner requires model_inputs.")
            if model_inputs.G is None or model_inputs.G_mask is None or model_inputs.P_idx is None:
                raise ValueError("bipartite-mp requires G, G_mask, and P_idx in ModelInputs.")
            logits = self.bipartite(
                G_nodes=model_inputs.G,
                P_pairs=pair_embedding,
                pair_index=model_inputs.P_idx,
                G_mask=model_inputs.G_mask,
                P_mask=mask_bool,
            )

        return logits

class FeatureStateCombiner(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.state_combiner = config.state_combiner
        
        if config.state_combiner == 'film':
            self.film = FiLMConditioner(
                feature_dim=config.state_features_embedding_head_out_dim,
                n_layers=config.film_n_layers,
                hidden_dims=config.film_hidden_dims,
                out_dim=config.state_embedding_head_out_dim,
                activation=config.film_activation,
                layer_norm=config.film_layer_norm
            )
            self.state_mod_norm = nn.LayerNorm(config.state_embedding_head_out_dim) if config.film_state_mod_layer_norm else None
        elif config.state_combiner == 'concat':
            self.final_mlp = make_mlp(
                in_dim=config.state_features_embedding_head_out_dim + config.state_embedding_head_out_dim,
                n_layers=config.concat_n_layers,
                hidden_dims=config.concat_hidden_dims,
                out_dim=config.state_embedding_head_out_dim,
                activation=config.concat_activation,
                layer_norm=config.concat_layer_norm,
            )
        else: 
            raise ValueError(
                f"Unknown combiner method: "
                f"{config.state_combiner}"
            )
        
    def forward(self, state_embedding, state_feature_embedding):
        if self.state_combiner == 'film':
            gamma, beta = self.film(state_feature_embedding)  # (B, S)

            state_embedding = gamma * state_embedding + beta  # (B, S)

            if self.state_mod_norm is not None:
                state_embedding = self.state_mod_norm(state_embedding)
        elif self.state_combiner == 'concat':
            head_in = torch.cat([state_embedding, state_feature_embedding], dim = -1)
            state_embedding = self.final_mlp(head_in)
        return state_embedding

class FeaturePairCombiner(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.pair_combiner = config.pair_combiner
        
        if config.pair_combiner == 'sqa':
            self.sqa = QueryKeyPairScorer(
                key_dim=config.pair_embedding_head_out_dim,
                query_dim=config.state_features_embedding_head_out_dim,
                d=config.single_query_attention_dim,
                norm_qk=config.single_query_attention_norm_qk,
                final_attention=config.single_query_attention_final_attention,
                n_glimpses=config.single_query_attention_n_glimpses,
                glimpse_attention=config.single_query_glimpse_attention_mode)
        elif config.pair_combiner == 'film':
            self.film = FiLMConditioner(
                feature_dim=config.state_features_embedding_head_out_dim,
                n_layers=config.film_n_layers,
                hidden_dims=config.film_hidden_dims,
                out_dim=config.pair_embedding_head_out_dim,
                activation=config.film_activation,
                layer_norm=config.film_layer_norm
            )
            self.pair_mod_norm = nn.LayerNorm(config.pair_embedding_head_out_dim) if config.film_pair_mod_layer_norm else None
            self.final_mlp = make_mlp(
                in_dim=config.pair_embedding_head_out_dim,
                n_layers=config.film_n_layers,
                hidden_dims=config.film_hidden_dims,
                out_dim=1,
                activation=config.film_activation,
                layer_norm=config.film_layer_norm,
            )
        elif config.pair_combiner == 'concat':
            self.final_mlp = make_mlp(
                in_dim=config.pair_embedding_head_out_dim + config.state_features_embedding_head_out_dim,
                n_layers=config.concat_n_layers,
                hidden_dims=config.concat_hidden_dims,
                out_dim=1,
                activation=config.concat_activation,
                layer_norm=config.concat_layer_norm,
            )
        elif config.pair_combiner == 'cross-attn':
            self.cross_attn = CrossConditioner(
                pair_dim=config.pair_embedding_head_out_dim,
                context_dim=config.state_features_embedding_head_out_dim,
                heads=config.cross_attn_num_heads,
                post_attn_layer_norm=config.cross_attn_post_attn_layer_norm,
            )
            self.final_mlp = make_mlp(
                in_dim=config.pair_embedding_head_out_dim,
                n_layers=config.concat_n_layers,
                hidden_dims=config.concat_hidden_dims,
                out_dim=1,
                activation=config.concat_activation,
                layer_norm=config.concat_layer_norm,
            )
        elif config.pair_combiner == 'bipartite-mp':
            self.bipartite = BuchbergerBipartite(
                node_dim=config.state_embedding_head_in_dim,
                pair_dim=config.pair_embedding_head_out_dim,
                hidden_dim=config.bipartite_hidden_dim,
                layers=config.bipartite_layers,
            )
        else: 
            raise ValueError(
                f"Unknown combiner method: "
                f"{config.pair_combiner}"
            )
        
    def forward(self, state_feature_embedding, pair_embedding, mask, model_inputs: Optional[ModelInputs] = None):
        mask_bool = mask.bool()
        mask_exp = mask_bool.unsqueeze(-1).float()

        if self.pair_combiner == 'film':
            gamma, beta = self.film(state_feature_embedding)  # (B, S)

            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)

            pair_mod = gamma * pair_embedding + beta  # (B, S)

            if self.pair_mod_norm is not None:
                pair_mod = self.pair_mod_norm(pair_mod)

            pair_out = self.final_mlp(pair_mod)

            logits = pair_out * mask_exp
        elif self.pair_combiner == 'sqa':
            logits = self.sqa(pair_embedding, state_feature_embedding, mask_bool)
        elif self.pair_combiner == 'concat':
            state_exp = state_feature_embedding.unsqueeze(1).expand(pair_embedding.shape[0], pair_embedding.shape[1], state_feature_embedding.shape[-1])            # (B,N,S)
            head_in = torch.cat([pair_embedding, state_exp], dim=-1)     
            pair_out = self.final_mlp(head_in)
            logits = pair_out * mask_exp
        elif self.pair_combiner == 'cross-attn':
            pair_mod = self.cross_attn(pair_embedding, state_feature_embedding)
            pair_out = self.final_mlp(pair_mod)
            logits = pair_out * mask_exp
        elif self.pair_combiner == 'bipartite-mp':
            if model_inputs is None:
                raise ValueError("bipartite-mp combiner requires model_inputs.")
            if model_inputs.G is None or model_inputs.G_mask is None or model_inputs.P_idx is None:
                raise ValueError("bipartite-mp requires G, G_mask, and P_idx in ModelInputs.")
            logits = self.bipartite(
                G_nodes=model_inputs.G,
                P_pairs=pair_embedding,
                pair_index=model_inputs.P_idx,
                G_mask=model_inputs.G_mask,
                P_mask=mask_bool,
            )

        return logits
