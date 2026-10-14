""" The purpose of this script is to house the final combined network which will bring together all the networks that have been implemented thus far

This network will use differnt components from the network_components.py
file based on which pair embedding head, state embedding head, state
feature embedding head etc. was specified by the configuration dataclass

The network's forward method will take as arguments:
- The pair set `P` in the form of an array of exponent vectors with shape (B, N, 2*D).
- The basis set `G` in the form of an array of exponent vectors with shape (B, M, D) or None.
- The pair set mask `P_mask` in the form of a boolean or 0/1 array with shape (B, N). This pair mask tells us which (if any) of the pair set rows correspond to a padded element.
- The basis set mask `G_mask` in the form of a boolean or 0/1 array with shape (B, M). This basis set mask tells us which (if any) of the basis set rows correspond to a padded element or None.
- State features `state_features` an array of shape (B, n_state_features) containing the 'handcrafted' state features or None.

"""


from __future__ import annotations

from deepgroebner_pytorch_cocoa.config_dataclasses import ModelConfig, ModelInputs, ActorConfig, CriticConfig
from deepgroebner_pytorch_cocoa.network_helpers import make_mlp
from deepgroebner_pytorch_cocoa.network_components import PairEmbeddingHead, StateEmbeddingHead, StateFeaturesEmbeddingHead, StatePairCombiner, FeatureStateCombiner, FeaturePairCombiner

import torch
import torch.nn as nn

def _make_activation(name: str) -> nn.Module:
    activations = {
        "identity": nn.Identity,
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "selu": nn.SELU,
        "tanh": nn.Tanh,
    }
    if name not in activations:
        raise ValueError(f"Unknown activation '{name}'. Options: {list(activations.keys())}")
    return activations[name]()

def _build_pair_scoring_head(config: ActorConfig) -> nn.Module:
    layers: list[nn.Module] = []
    if config.pair_scoring_head_layer_norm:
        layers.append(nn.LayerNorm(config.pair_embedding_head_out_dim))
    layers.append(_make_activation(config.pair_scoring_head_activation))
    layers.append(nn.Linear(config.pair_embedding_head_out_dim, 1))
    return nn.Sequential(*layers)

def _build_state_modules(config: ModelConfig):
    """Construct new state encoders and a combiner for one network."""
    state_embedding_head = (
        StateEmbeddingHead(config) if config.state_embedding_head is not None else None
    )
    state_features_embedding_head = (
        StateFeaturesEmbeddingHead(config)
        if config.state_features_embedding_head is not None
        else None
    )
    state_feature_combiner = (
        FeatureStateCombiner(config)
        if state_embedding_head is not None and state_features_embedding_head is not None
        else None
    )
    return state_embedding_head, state_features_embedding_head, state_feature_combiner


class CombinedPolicyNetwork(nn.Module):

    def __init__(self, config: ActorConfig):
        super().__init__()

        if config.pair_combiner not in {"film", "sqa", "concat"}:
            raise ValueError(f"Unknown pair_combiner: {config.pair_combiner}")
        self.pair_embedding_head = PairEmbeddingHead(config)
        (
            self.state_embedding_head,
            self.state_features_embedding_head,
            self.state_feature_combiner,
        ) = _build_state_modules(config)
        has_state_embedding = self.state_embedding_head is not None
        # Keep this derived flag available to diagnostics.
        self.pair_only_scoring = (
            not has_state_embedding and self.state_features_embedding_head is None
        )

        if self.pair_only_scoring:
            self.final_module = _build_pair_scoring_head(config)
        elif has_state_embedding:
            self.final_module = StatePairCombiner(config)
        else:
            self.final_module = FeaturePairCombiner(config)

    def _compute_state_embeddings(
        self,
        inputs: ModelInputs,
    ):
        state_embedding = (
            self.state_embedding_head(inputs)
            if self.state_embedding_head is not None
            else None
        )

        state_features_embedding = (
            self.state_features_embedding_head(inputs)
            if self.state_features_embedding_head is not None
            else None
        )

        return state_embedding, state_features_embedding

    def _combine_state_conditioning(
        self,
        state_embedding: torch.Tensor,
        state_features_embedding: torch.Tensor | None,
    ) -> torch.Tensor:
        if state_features_embedding is not None:
            return self.state_feature_combiner(
                state_embedding,
                state_features_embedding,
            )
        return state_embedding

    def _compute_pair_embedding(
        self,
        inputs: ModelInputs,
        return_attention: bool = False,
    ):
        if return_attention:
            return self.pair_embedding_head(
                inputs,
                return_attention=True,
            )
        return (
            self.pair_embedding_head(
                inputs,
            ),
            None,
        )

    def score_from_embeddings(
        self,
        pair_embedding: torch.Tensor,
        inputs: ModelInputs,
        state_embedding: torch.Tensor | None,
        state_features_embedding: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.pair_only_scoring:
            logits = self.final_module(pair_embedding)
            return logits * inputs.P_mask.unsqueeze(-1)

        if state_embedding is not None:
            combined_state = self._combine_state_conditioning(
                state_embedding,
                state_features_embedding,
            )
            return self.final_module(
                combined_state,
                pair_embedding,
                inputs.P_mask,
            )

        return self.final_module(
            state_features_embedding,
            pair_embedding,
            inputs.P_mask,
        )

    def forward(
        self,
        inputs: ModelInputs,
        return_internals: bool = False,
        return_attention: bool = False,
    ):

        (
            raw_state_embedding,
            state_features_embedding,
        ) = self._compute_state_embeddings(inputs)

        pair_embedding, pair_attention = self._compute_pair_embedding(
            inputs,
            return_attention=return_attention,
        )

        masked_logits = self.score_from_embeddings(
            pair_embedding,
            inputs,
            raw_state_embedding,
            state_features_embedding,
        )

        if return_internals:
            if return_attention:
                attention_payload = {
                    "pair_embedding_head": pair_attention,
                }
                return (
                    masked_logits,
                    pair_embedding,
                    raw_state_embedding,
                    state_features_embedding,
                    attention_payload,
                )
            return masked_logits, pair_embedding, raw_state_embedding, state_features_embedding
        else:
            return masked_logits

class CombinedValueNetwork(nn.Module):

    def __init__(self, config: CriticConfig):
        super().__init__()
        self.popart = None

        if config.state_embedding_head is None:
            raise ValueError("Value network requires state_embedding_head.")

        (
            self.state_embedding_head,
            self.state_features_embedding_head,
            self.state_feature_combiner,
        ) = _build_state_modules(config)

        self.value_head = make_mlp(
            in_dim=config.state_embedding_head_out_dim,
            n_layers=config.value_head_n_layers,
            hidden_dims=config.value_head_hidden_dims,
            out_dim=1,
            activation=config.value_head_activation,
            layer_norm=config.value_head_layer_norm
        )

    def forward(self, inputs: ModelInputs, h_value = None):

        state_embedding = self.state_embedding_head(inputs)

        if self.state_features_embedding_head is not None:
            state_features_embedding = \
                self.state_features_embedding_head(inputs)

            state_embedding = self.state_feature_combiner(
                state_embedding,
                state_features_embedding
            )

        delta_normalized = self.value_head(state_embedding)
        delta = (
            self.popart.denormalize(delta_normalized)
            if self.popart is not None
            else delta_normalized
        )

        if h_value is not None:
            V = h_value + delta
        else:
            V = delta

        return V, delta, state_embedding
