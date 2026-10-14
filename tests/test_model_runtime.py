from dataclasses import fields
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from deepgroebner_pytorch_cocoa.combined_networks import CombinedPolicyNetwork, CombinedValueNetwork
from deepgroebner_pytorch_cocoa.config_dataclasses import ActorConfig, CriticConfig, ModelInputs
from deepgroebner_pytorch_cocoa.diagnostics import run_actor_global_diagnostics
from deepgroebner_pytorch_cocoa.model_input_utils import (
    build_model_inputs_from_single_obs,
    build_model_inputs_from_batched_obs,
    collate_ragged_model_inputs,
)


def test_experimental_heuristic_choice_env_is_not_registered():
    with pytest.raises(gym.error.NameNotFound):
        gym.spec("BuchbergerHeuristicChoice-v1")


def _inputs():
    return ModelInputs(
        P=torch.randn(2, 3, 6),
        P_mask=torch.tensor([[True, True, False], [True, False, False]]),
        G=torch.randn(2, 2, 3),
        G_mask=torch.ones(2, 2, dtype=torch.bool),
        state_features=torch.randn(2, 4),
    )


def _config(config_type, activation="gelu", **kwargs):
    config = config_type(**kwargs)
    config.state_embedding_head_in_dim = 3
    config.state_features_embedding_head_in_dim = 4
    if isinstance(config, ActorConfig):
        config.pair_embedding_head_in_dim = 6
    for field in fields(config):
        if field.name.endswith("_activation"):
            setattr(config, field.name, activation)
    return config


@pytest.mark.parametrize("activation", ["relu", "gelu", "selu", "tanh"])
@pytest.mark.parametrize("combiner", ["film", "concat"])
def test_configured_activations_support_actor_and_critic_forward_backward(activation, combiner):
    inputs = _inputs()
    actor = CombinedPolicyNetwork(_config(
        ActorConfig, activation,
        state_embedding_head="deep-set", state_features_embedding_head="mlp",
        state_combiner=combiner, pair_combiner=combiner,
    ))
    critic = CombinedValueNetwork(_config(CriticConfig, activation, state_combiner=combiner))
    logits = actor(inputs)
    values, delta, _ = critic(inputs)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(values).all()
    torch.testing.assert_close(values, delta)
    (logits[inputs.P_mask].sum() + values.sum()).backward()
    for model in (actor, critic):
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads
        assert all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("combiner", ["film", "sqa", "concat"])
@pytest.mark.parametrize("state_head,feature_head", [
    (None, None), ("deep-set", None), (None, "mlp"), ("deep-set", "mlp"),
])
def test_current_actor_heads_emit_finite_counterfactual_gradient_and_attention_metrics(
    combiner, state_head, feature_head,
):
    actor = CombinedPolicyNetwork(_config(
        ActorConfig, state_embedding_head=state_head,
        state_features_embedding_head=feature_head,
        pair_embedding_head="alphastar-buchberger-pair-transformer", pair_combiner=combiner,
    ))
    metrics = {}
    writer = SimpleNamespace(add_scalar=lambda tag, value, step: metrics.update({tag: value}))
    run_actor_global_diagnostics(actor, _inputs(), writer, 1, 10, diag_every=1, grad_every=1)
    assert metrics
    assert all(np.isfinite(value) for value in metrics.values())
    assert "diagnostics/pair_embed_sens" in metrics
    assert any(key.startswith("diagnostics/attn_pair_embedding_head_") for key in metrics)
    if state_head:
        assert "diagnostics/cf_state_zero_kl" in metrics
        assert "diagnostics/state_grad_sens" in metrics
    if feature_head:
        assert "diagnostics/cf_state_feats_zero_kl" in metrics
        assert "diagnostics/state_features_grad_sens" in metrics
    assert actor.training
    assert all(p.grad is None for p in actor.parameters())


@pytest.mark.parametrize("gym_id", ["BuchbergerExponents-v1", "BuchbergerExponentsNoPQ-v1"])
def test_native_observations_survive_single_batched_and_minibatch_conversion(gym_id):
    device = torch.device("cpu")
    env = gym.make(
        gym_id, coeff_ring="QQ", symbols="x,y",
        basis_elements=["x^2 - y", "x*y - 1"], elimination="none",
    )
    try:
        obs, _ = env.reset(seed=123)
        first = build_model_inputs_from_single_obs(obs, device)
        action = int(np.flatnonzero(obs["P_mask"])[0])
        next_obs, reward, _, _, _ = env.step(action)
        assert np.isfinite(reward)
        second = build_model_inputs_from_single_obs(next_obs, device)
        batched_obs = {
            key: (value,) if "NoPQ" in gym_id and key != "state_features" else value[None]
            for key, value in next_obs.items()
        }
        batched = build_model_inputs_from_batched_obs(batched_obs, device)
        for key in ("P", "G", "P_mask", "G_mask", "state_features"):
            torch.testing.assert_close(getattr(batched, key), getattr(second, key))

        records = [first, second]
        collated = collate_ragged_model_inputs(
            indices=[1, 0], device=device,
            obs_G_list=[r.G[0] for r in records], obs_P_list=[r.P[0] for r in records],
            g_mask_list=[r.G_mask[0] for r in records], p_mask_list=[r.P_mask[0] for r in records],
            obs_sf_list=[r.state_features[0] for r in records],
        )
        for row, record in enumerate(reversed(records)):
            for key in ("P", "G", "P_mask", "G_mask", "state_features"):
                expected = getattr(record, key)[0]
                actual = getattr(collated, key)[row]
                slices = tuple(slice(0, size) for size in expected.shape)
                torch.testing.assert_close(actual[slices], expected)
            assert not collated.P_mask[row, record.P.shape[1]:].any()
            assert not collated.G_mask[row, record.G.shape[1]:].any()
    finally:
        env.close()
