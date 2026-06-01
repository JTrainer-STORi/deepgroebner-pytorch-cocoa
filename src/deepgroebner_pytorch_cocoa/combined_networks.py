""" The purpose of this script is to house the final combined network which will bring together all the networks that have been implemented thus far

The options which we will supply to this network are as follows:

The network's forward method will take as arguments:
- The pair set `P` in the form of an array of exponent vectors with shape (B, N, 2*D).
- The basis set `G` in the form of an array of exponent vectors with shape (B, M, D) or None.
- The pair set mask `P_mask` in the form of a boolean or 0/1 array with shape (B, N). This pair mask tells us which (if any) of the pair set rows correspond to a padded element.
- The basis set mask `G_mask` in the form of a boolean or 0/1 array with shape (B, M). This basis set mask tells us which (if any) of the basis set rows correspond to a padded element or None.
- State features `state_features` an array of shape (B, n_state_features) containing the 'handcrafted' state features or None.

"""


from __future__ import annotations

from dataclasses import fields
from typing import Optional
from deepgroebner_pytorch_cocoa.config_dataclasses import ModelConfig, ModelInputs, ActorConfig, CriticConfig
from deepgroebner_pytorch_cocoa.network_helpers import make_mlp
from deepgroebner_pytorch_cocoa.network_components import PairEmbeddingHead, StateEmbeddingHead, StateFeaturesEmbeddingHead, StatePairCombiner, FeatureStateCombiner, FeaturePairCombiner, SharedPolynomialActorHead, BasisPairCombiner

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

def checkpoint_uses_legacy_pair_scoring_head(
    state_dict: dict[str, torch.Tensor],
) -> bool:
    return (
        "final_module.weight" in state_dict
        and "final_module.bias" in state_dict
        and not any(key.startswith("final_module.0.") for key in state_dict)
    )


def _build_pair_scoring_head(
    config: ActorConfig,
    legacy_pair_scoring_head: bool = False,
) -> nn.Module:
    if legacy_pair_scoring_head:
        return nn.Linear(config.pair_embedding_head_out_dim, 1)

    layers: list[nn.Module] = []
    if config.pair_scoring_head_layer_norm:
        layers.append(nn.LayerNorm(config.pair_embedding_head_out_dim))
    layers.append(_make_activation(config.pair_scoring_head_activation))
    layers.append(nn.Linear(config.pair_embedding_head_out_dim, 1))
    return nn.Sequential(*layers)

def _resolve_shared_state_modules(
    config: ModelConfig,
    state_embedding_head: nn.Module | None,
    state_features_embedding_head: nn.Module | None,
    state_feature_combiner: nn.Module | None,
):
    resolved_state_embedding_head = (
        state_embedding_head
        if state_embedding_head is not None
        else (
            StateEmbeddingHead(config)
            if config.state_embedding_head is not None
            else None
        )
    )
    resolved_state_features_embedding_head = (
        state_features_embedding_head
        if state_features_embedding_head is not None
        else (
            StateFeaturesEmbeddingHead(config)
            if config.state_features_embedding_head is not None
            else None
        )
    )

    if resolved_state_embedding_head is None and config.state_embedding_head is not None:
        raise ValueError("State embedding head config requires a state_embedding_head module.")

    if resolved_state_embedding_head is not None and resolved_state_features_embedding_head is not None:
        resolved_state_feature_combiner = (
            state_feature_combiner
            if state_feature_combiner is not None
            else FeatureStateCombiner(config)
        )
    else:
        if state_feature_combiner is not None:
            raise ValueError(
                "state_feature_combiner can only be provided when both state heads are present."
            )
        resolved_state_feature_combiner = None

    return (
        resolved_state_embedding_head,
        resolved_state_features_embedding_head,
        resolved_state_feature_combiner,
    )


def _validate_shared_actor_critic_state_configs(
    actor_config: ActorConfig,
    critic_config: CriticConfig,
):
    if actor_config.share_polynomial_encoder_between_heads:
        raise ValueError(
            "Shared actor/critic trunk is not supported when "
            "actor.share_polynomial_encoder_between_heads=True."
        )

    if actor_config.state_embedding_head is None:
        raise ValueError("Shared actor/critic trunk requires actor.state_embedding_head.")

    if critic_config.state_embedding_head is None:
        raise ValueError("Shared actor/critic trunk requires critic.state_embedding_head.")

    mismatched_fields = []
    for field in fields(ModelConfig):
        if not field.init:
            continue
        actor_value = getattr(actor_config, field.name)
        critic_value = getattr(critic_config, field.name)
        if actor_value != critic_value:
            mismatched_fields.append(field.name)

    runtime_dim_fields = [
        "state_embedding_head_in_dim",
        "state_features_embedding_head_in_dim",
    ]
    for field_name in runtime_dim_fields:
        if getattr(actor_config, field_name) != getattr(critic_config, field_name):
            mismatched_fields.append(field_name)

    if mismatched_fields:
        mismatched_fields_str = ", ".join(sorted(set(mismatched_fields)))
        raise ValueError(
            "Shared actor/critic trunk requires matching actor/critic state trunk "
            f"configuration. Mismatched fields: {mismatched_fields_str}"
        )


class SharedActorCriticStateTrunk(nn.Module):
    """Reusable state-side modules shared between actor and critic."""

    def __init__(self, config: ModelConfig):
        super().__init__()

        if config.state_embedding_head is None:
            raise ValueError("Shared actor/critic trunk requires state_embedding_head.")

        (
            self.state_embedding_head,
            self.state_features_embedding_head,
            self.state_feature_combiner,
        ) = _resolve_shared_state_modules(
            config,
            state_embedding_head=None,
            state_features_embedding_head=None,
            state_feature_combiner=None,
        )


def build_shared_actor_critic_state_trunk(
    actor_config: ActorConfig,
    critic_config: CriticConfig,
) -> SharedActorCriticStateTrunk:
    _validate_shared_actor_critic_state_configs(actor_config, critic_config)
    return SharedActorCriticStateTrunk(actor_config)


class CombinedPolicyNetwork(nn.Module):

    def __init__(
        self,
        config: ActorConfig,
        state_embedding_head: nn.Module | None = None,
        state_features_embedding_head: nn.Module | None = None,
        state_feature_combiner: nn.Module | None = None,
        legacy_pair_scoring_head: bool = False,
    ):
        super().__init__()

        self.share_polynomial_encoder_between_heads = (
            config.share_polynomial_encoder_between_heads
        )
        self.state_routed_pair_embedding = (
            config.pair_embedding_head == "moe-symmetric-per-entity-mlp"
        )
        self.pair_only_scoring = config.pair_combiner == "pair-only"
        self.basis_conditioned_scoring = config.pair_combiner == "basis-cross-attn"
        self.legacy_pair_scoring_head = legacy_pair_scoring_head

        if self.share_polynomial_encoder_between_heads:
            if config.state_embedding_head not in {"deep-set", "polynomial-rnn-deep-set"}:
                raise ValueError(
                    "Shared polynomial actor head currently requires "
                    "state_embedding_head in {'deep-set', 'polynomial-rnn-deep-set'}."
                )
            if config.pair_embedding_head != "symmetric-per-entity-mlp":
                raise ValueError(
                    "Shared polynomial actor head currently requires "
                    "pair_embedding_head='symmetric-per-entity-mlp'."
                )
            if self.state_routed_pair_embedding:
                raise ValueError(
                    "Shared polynomial actor head is not supported with "
                    "pair_embedding_head='moe-symmetric-per-entity-mlp'."
                )
            self.shared_polynomial_head = SharedPolynomialActorHead(config)
            self.pair_embedding_head = None
            self.state_embedding_head = None
            self.state_features_embedding_head = (
                StateFeaturesEmbeddingHead(config)
                if config.state_features_embedding_head is not None
                else None
            )
            self.state_feature_combiner = (
                FeatureStateCombiner(config)
                if self.state_features_embedding_head is not None
                else None
            )
        else:
            self.shared_polynomial_head = None
            self.pair_embedding_head = (
                PairEmbeddingHead(config)
                if config.pair_embedding_head is not None
                else None
            )

            (
                self.state_embedding_head,
                self.state_features_embedding_head,
                self.state_feature_combiner,
            ) = _resolve_shared_state_modules(
                config,
                state_embedding_head=state_embedding_head,
                state_features_embedding_head=state_features_embedding_head,
                state_feature_combiner=state_feature_combiner,
            )

        has_pair_embedding = (
            self.shared_polynomial_head is not None or
            self.pair_embedding_head is not None
        )
        has_state_embedding = (
            self.shared_polynomial_head is not None or
            self.state_embedding_head is not None
        )

        if not has_pair_embedding:
            raise ValueError("Policy requires pair_embedding_head.")

        if self.pair_only_scoring:
            self.final_module = _build_pair_scoring_head(
                config,
                legacy_pair_scoring_head=self.legacy_pair_scoring_head,
            )
        elif self.basis_conditioned_scoring:
            self.final_module = BasisPairCombiner(config)
        elif has_state_embedding:
            self.final_module = StatePairCombiner(config)
        elif self.state_features_embedding_head is not None:
            self.final_module = FeaturePairCombiner(config)
        else:
            self.final_module = _build_pair_scoring_head(
                config,
                legacy_pair_scoring_head=self.legacy_pair_scoring_head,
            )

    def _compute_state_embeddings(
        self,
        inputs: ModelInputs,
        return_attention: bool = False,
    ):
        state_attention = None
        if self.shared_polynomial_head is not None:
            state_embedding, pair_embedding = self.shared_polynomial_head(inputs)
            state_features_embedding = (
                self.state_features_embedding_head(inputs)
                if self.state_features_embedding_head is not None
                else None
            )
            return state_embedding, state_features_embedding, pair_embedding, state_attention

        if self.state_embedding_head is not None:
            if return_attention:
                state_embedding, state_attention = self.state_embedding_head(
                    inputs,
                    return_attention=True,
                )
            else:
                state_embedding = self.state_embedding_head(inputs)
        else:
            state_embedding = None

        state_features_embedding = (
            self.state_features_embedding_head(inputs)
            if self.state_features_embedding_head is not None
            else None
        )

        return state_embedding, state_features_embedding, None, state_attention

    def _combine_state_conditioning(
        self,
        state_embedding: torch.Tensor | None,
        state_features_embedding: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if state_embedding is not None and state_features_embedding is not None:
            return self.state_feature_combiner(
                state_embedding,
                state_features_embedding,
            )
        if state_embedding is not None:
            return state_embedding
        return state_features_embedding

    def _compute_pair_embedding(
        self,
        inputs: ModelInputs,
        state_embedding: torch.Tensor | None,
        state_features_embedding: torch.Tensor | None,
        return_attention: bool = False,
        precomputed_pair_embedding: torch.Tensor | None = None,
    ):
        if precomputed_pair_embedding is not None:
            return precomputed_pair_embedding, None

        router_context = None
        if self.state_routed_pair_embedding:
            router_context = self._combine_state_conditioning(
                state_embedding,
                state_features_embedding,
            )

        if return_attention:
            return self.pair_embedding_head(
                inputs,
                return_attention=True,
                router_context=router_context,
            )
        return (
            self.pair_embedding_head(
                inputs,
                router_context=router_context,
            ),
            None,
        )

    def _score_from_embeddings(
        self,
        pair_embedding: torch.Tensor,
        inputs: ModelInputs,
        state_embedding: torch.Tensor | None,
        state_features_embedding: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.pair_only_scoring or (
            state_embedding is None and self.state_features_embedding_head is None
            and not self.basis_conditioned_scoring
        ):
            logits = self.final_module(pair_embedding)
            return logits * inputs.P_mask.unsqueeze(-1)

        if self.basis_conditioned_scoring:
            return self.final_module(
                pair_embedding,
                model_inputs=inputs,
            )

        if state_embedding is not None:
            combined_state = self._combine_state_conditioning(
                state_embedding,
                state_features_embedding,
            )
            return self.final_module(
                combined_state,
                pair_embedding,
                inputs.P_mask,
                model_inputs=inputs,
            )

        return self.final_module(
            state_features_embedding,
            pair_embedding,
            inputs.P_mask,
            model_inputs=inputs,
        )

    def forward(
        self,
        inputs: ModelInputs,
        return_internals: bool = False,
        return_attention: bool = False,
    ):

        pair_attention = None
        (
            raw_state_embedding,
            state_features_embedding,
            precomputed_pair_embedding,
            state_attention,
        ) = self._compute_state_embeddings(inputs, return_attention=return_attention)

        pair_embedding, pair_attention = self._compute_pair_embedding(
            inputs,
            raw_state_embedding,
            state_features_embedding,
            return_attention=return_attention,
            precomputed_pair_embedding=precomputed_pair_embedding,
        )

        masked_logits = self._score_from_embeddings(
            pair_embedding,
            inputs,
            raw_state_embedding,
            state_features_embedding,
        )

        if return_internals:
            if return_attention:
                attention_payload = {
                    "pair_embedding_head": pair_attention,
                    "state_embedding_head": state_attention,
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

    def __init__(
        self,
        config: CriticConfig,
        state_embedding_head: nn.Module | None = None,
        state_features_embedding_head: nn.Module | None = None,
        state_feature_combiner: nn.Module | None = None,
    ):
        super().__init__()
        self.popart = None

        if config.state_embedding_head is None:
            raise ValueError("Value network requires state_embedding_head.")

        (
            self.state_embedding_head,
            self.state_features_embedding_head,
            self.state_feature_combiner,
        ) = _resolve_shared_state_modules(
            config,
            state_embedding_head=state_embedding_head,
            state_features_embedding_head=state_features_embedding_head,
            state_feature_combiner=state_feature_combiner,
        )

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
