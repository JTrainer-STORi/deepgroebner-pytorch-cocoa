import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from deepgroebner_pytorch_cocoa.network_helpers import make_mlp
from deepgroebner_pytorch_cocoa.config_dataclasses import ModelConfig, ModelInputs

class PerEntityMLP(nn.Module):
    """Basic MLP to apply to exponent vectors of pairs in the pairset P. Takes the pair exponent vector and passes it through an MLP. If use_lcm_gcd is true in the ModelConfig then the lcm and gcd will be appended to the input vector. 

    Options to add layer normalization or use any of the following activations: 'relu', 'gelu', 'selu', 'tanh'

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
class PerPairResblock(nn.Module):
    """Residual MLP block applied independently to each visible pair token."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        *,
        use_layer_norm: bool,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if use_layer_norm else nn.Identity()
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        delta = self.ff(F.relu(self.norm(x)))
        x = x + delta
        return x * mask.unsqueeze(-1).float()


class AlphaStarPairSelfAttentionBlock(nn.Module):
    """Masked self-attention block over the visible pair stream."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        use_layer_norm: bool,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if use_layer_norm else nn.Identity()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        *,
        return_attention: bool = False,
    ):
        # The mask is necessary for states in which the number of pairs is less than the priority queue size and padding is used. Masking ensures that the padded pairs are ignored.
        mask_bool = mask.bool()
        key_padding_mask = ~mask_bool
        if x.shape[1] == 0:
            if return_attention:
                return x, x.new_zeros((x.shape[0], self.attn.num_heads, 0, 0))
            return x
        attn_input = F.relu(self.norm(x))
        # Terminal observations can be entirely padded before vector-env autoreset.
        # Give those rows a dummy key to avoid an all-masked attention softmax.
        has_valid = mask_bool.any(dim=-1)
        if not has_valid.all():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[~has_valid, 0] = False
            attn_input = attn_input.clone()
            attn_input[~has_valid] = 0.0

        attn_out, attn_weights = self.attn(
            attn_input,
            attn_input,
            attn_input,
            key_padding_mask=key_padding_mask if key_padding_mask.numel() > 0 else None,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        # Explicitly zero padded pairs output embeddings
        x = (x + attn_out) * mask_bool.unsqueeze(-1).float()
        if return_attention:
            attn_weights = attn_weights.masked_fill(
                ~(mask_bool[:, None, :, None] & mask_bool[:, None, None, :]), 0.0
            )
            return x, attn_weights
        return x

class AlphaStarBuchbergerPairTransformer(nn.Module):
    """Buchberger-specific pair transformer seeded from symmetric endpoint pair tokens."""

    def __init__(self, config: ModelConfig):
        super().__init__()

        if config.pair_embedding_head_in_dim is None:
            raise ValueError(
                "alphastar-buchberger-pair-transformer requires pair_embedding_head_in_dim."
            )
        if config.pair_embedding_head_out_dim <= 0:
            raise ValueError(
                "alphastar-buchberger-pair-transformer requires positive pair_embedding_head_out_dim."
            )
        if config.pair_embedding_head_in_dim % 2 != 0:
            raise ValueError("pair_embedding_head_in_dim must be even.")

        self.raw_pair_dim = int(config.pair_embedding_head_in_dim)
        self.hid_dim = int(config.pair_embedding_head_out_dim)
        self.use_lcm_gcd = bool(config.use_lcm_gcd)
        self.D = self.raw_pair_dim // 2

        self.basis_encoder = make_mlp(
            in_dim=self.D,
            n_layers=config.alphastar_buchberger_basis_encoder_n_layers,
            hidden_dims=config.alphastar_buchberger_basis_encoder_hidden_dims,
            out_dim=self.hid_dim,
            activation=config.alphastar_buchberger_basis_encoder_activation,
            layer_norm=config.alphastar_buchberger_basis_encoder_layer_norm,
        )
        pair_feature_dim = 4 * self.hid_dim if self.use_lcm_gcd else 2 * self.hid_dim
        self.pair_token_mlp = make_mlp(
            in_dim=pair_feature_dim,
            n_layers=config.pair_embedding_head_n_layers,
            hidden_dims=config.pair_embedding_head_hidden_dims,
            out_dim=self.hid_dim,
            activation=config.pair_embedding_head_activation,
            layer_norm=config.pair_embedding_head_layer_norm,
        )
        self.pre_blocks = nn.ModuleList(
            [
                PerPairResblock(
                    self.hid_dim,
                    int(config.alphastar_pair_transformer_resblock_hidden_dim),
                    use_layer_norm=bool(config.alphastar_pair_transformer_layer_norm),
                )
                for _ in range(int(config.alphastar_pair_transformer_num_resblocks_before))
            ]
        )
        self.attn_blocks = nn.ModuleList(
            [
                AlphaStarPairSelfAttentionBlock(
                    self.hid_dim,
                    int(config.alphastar_pair_transformer_num_heads),
                    use_layer_norm=bool(config.alphastar_pair_transformer_layer_norm),
                )
                for _ in range(int(config.alphastar_pair_transformer_num_layers))
            ]
        )
        self.post_blocks = nn.ModuleList(
            [
                PerPairResblock(
                    self.hid_dim,
                    int(config.alphastar_pair_transformer_resblock_hidden_dim),
                    use_layer_norm=bool(config.alphastar_pair_transformer_layer_norm),
                )
                for _ in range(int(config.alphastar_pair_transformer_num_resblocks_after))
            ]
        )

    def _build_pair_tokens(self, inputs: ModelInputs) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.P is None or inputs.P_mask is None:
            raise ValueError(
                "alphastar-buchberger-pair-transformer requires P and P_mask in ModelInputs."
            )
        pair_mask = inputs.P_mask.bool()
        # P retains both endpoints even when the visible basis G is truncated.
        p1 = inputs.P[..., :self.D]
        p2 = inputs.P[..., self.D:]
        g_i = self.basis_encoder(p1)
        g_j = self.basis_encoder(p2)
        pair_features = [g_i + g_j, torch.abs(g_i - g_j)]

        if self.use_lcm_gcd:
            pair_features.extend(
                [
                    self.basis_encoder(torch.maximum(p1, p2)),
                    self.basis_encoder(torch.minimum(p1, p2)),
                ]
            )

        pair_tokens = self.pair_token_mlp(torch.cat(pair_features, dim=-1))
        return pair_tokens * pair_mask.unsqueeze(-1).float(), pair_mask

    def forward(self, inputs: ModelInputs, return_attention: bool = False):
        H, pair_mask = self._build_pair_tokens(inputs)

        for block in self.pre_blocks:
            H = block(H, pair_mask)

        attention_maps = {}
        for idx, block in enumerate(self.attn_blocks):
            if return_attention:
                H, attn_weights = block(H, pair_mask, return_attention=True)
                attention_maps[f"encoder_{idx}"] = {
                    "weights": attn_weights,
                    "q_mask": pair_mask,
                    "k_mask": pair_mask,
                }
            else:
                H = block(H, pair_mask)

        for block in self.post_blocks:
            H = block(H, pair_mask)

        if return_attention:
            return H, attention_maps
        return H


class PairEmbeddingHead(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        if config.pair_embedding_head == 'per-entity-mlp':
            self.net = PerEntityMLP(config=config)
        elif config.pair_embedding_head == 'symmetric-per-entity-mlp':
            self.net = SymmetricPerEntityMLP(config=config)
        elif config.pair_embedding_head == 'alphastar-buchberger-pair-transformer':
            self.net = AlphaStarBuchbergerPairTransformer(config=config)
        else:
            raise ValueError(f"Unknown pair_embedding_head: {config.pair_embedding_head}")

    def forward(
        self,
        inputs: ModelInputs,
        return_attention: bool = False,
    ):

        attention_maps = None
        if return_attention and isinstance(self.net, AlphaStarBuchbergerPairTransformer):
            network_output, attention_maps = self.net(inputs, return_attention=True)
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

class StateEmbeddingHead(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        if config.state_embedding_head == 'deep-set':
            self.net = DeepSetNetwork(config)
        else:
            raise ValueError(
                f"Unknown state_embedding_head: "
                f"{config.state_embedding_head}"
            )

    def forward(self, inputs: ModelInputs):
        return self.net(inputs)

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
        else: 
            raise ValueError(
                f"Unknown combiner method: "
                f"{config.pair_combiner}"
            )
        
    def forward(self, state_embedding, pair_embedding, mask):

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
        else: 
            raise ValueError(
                f"Unknown combiner method: "
                f"{config.pair_combiner}"
            )
        
    def forward(self, state_feature_embedding, pair_embedding, mask):
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

        return logits
