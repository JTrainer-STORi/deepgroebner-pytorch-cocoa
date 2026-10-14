import gymnasium as gym
import pytest
import torch

from deepgroebner_pytorch_cocoa.combined_networks import CombinedPolicyNetwork
from deepgroebner_pytorch_cocoa.config_dataclasses import ActorConfig, ModelInputs
from deepgroebner_pytorch_cocoa.model_input_utils import (
    build_model_inputs_from_single_obs,
)


def _config(use_lcm_gcd):
    cfg = ActorConfig(
        pair_embedding_head="alphastar-buchberger-pair-transformer",
        use_lcm_gcd=use_lcm_gcd,
        pair_embedding_head_hidden_dims=16,
        pair_embedding_head_out_dim=8,
        alphastar_buchberger_basis_encoder_hidden_dims=16,
        alphastar_pair_transformer_num_layers=1,
        alphastar_pair_transformer_num_resblocks_before=1,
        alphastar_pair_transformer_num_resblocks_after=1,
        alphastar_pair_transformer_resblock_hidden_dim=16,
    )
    cfg.pair_embedding_head_in_dim = 4
    return cfg


@pytest.mark.parametrize("basis_budget", [1, 2])
@pytest.mark.parametrize("use_lcm_gcd", [False, True])
def test_native_pairs_with_hidden_endpoints_keep_attention_and_gradient_signal(
    basis_budget, use_lcm_gcd
):
    torch.manual_seed(123)
    env = gym.make(
        "BuchbergerExponents-v1",
        coeff_ring="QQ",
        symbols="x,y",
        basis_elements=["x^2 - y", "x*y - 1", "y^2 - x"],
        elimination="none",
        N_P=3,
        N_G=basis_budget,
        k=1,
    )
    try:
        obs, _ = env.reset(seed=123)
        inputs = build_model_inputs_from_single_obs(obs, torch.device("cpu"))
        inputs.P = inputs.P.clone().requires_grad_()
        pair_endpoints = inputs.P.reshape(*inputs.P.shape[:2], 2, -1)
        endpoint_matches = (
            pair_endpoints.unsqueeze(-2) == inputs.G[:, None, None, :, :]
        ).all(dim=-1)
        visible_endpoints = (
            endpoint_matches & inputs.G_mask[:, None, None, :]
        ).any(dim=-1)
        missing_endpoints = inputs.P_mask & ~visible_endpoints.all(dim=-1)
        assert missing_endpoints.any()

        actor = CombinedPolicyNetwork(_config(use_lcm_gcd))
        tokens, mask = actor.pair_embedding_head.net._build_pair_tokens(inputs)
        torch.testing.assert_close(mask, inputs.P_mask)
        assert (tokens[missing_endpoints].abs().sum(dim=-1) > 0).all()

        logits, embeddings, _, _, attention = actor(
            inputs, return_internals=True, return_attention=True
        )
        assert (embeddings[missing_endpoints].abs().sum(dim=-1) > 0).all()
        for block in attention["pair_embedding_head"].values():
            torch.testing.assert_close(block["q_mask"], inputs.P_mask)
            torch.testing.assert_close(block["k_mask"], inputs.P_mask)
            row_sums = block["weights"][0, :, inputs.P_mask[0]].sum(dim=-1)
            torch.testing.assert_close(row_sums, torch.ones_like(row_sums))

        logits[missing_endpoints].sum().backward()
        assert torch.isfinite(inputs.P.grad).all()
        assert (inputs.P.grad[missing_endpoints].abs().sum(dim=-1) > 0).all()
    finally:
        env.close()


@pytest.mark.parametrize("use_lcm_gcd", [False, True])
@pytest.mark.parametrize("basis_context", ["absent", "inconsistent", "empty"])
def test_pair_only_policy_requires_only_pair_vectors_and_the_action_mask(
    use_lcm_gcd, basis_context
):
    torch.manual_seed(123)
    actor = CombinedPolicyNetwork(_config(use_lcm_gcd)).eval()
    pairs = torch.tensor([[[2.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 2.0]]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    inputs = ModelInputs(P=pairs, P_mask=mask)
    reference = actor(inputs)

    if basis_context == "inconsistent":
        inputs.G = torch.randn(1, 1, 7)
        inputs.G_mask = torch.zeros(1, 1, dtype=torch.bool)
    elif basis_context == "empty":
        inputs.G = torch.empty(1, 0, 2)
        inputs.G_mask = torch.empty(1, 0, dtype=torch.bool)

    actual = actor(inputs)
    assert actor.pair_only_scoring
    assert actual.shape == (1, 2, 1)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize("use_lcm_gcd", [False, True])
def test_pair_scores_are_symmetric_and_ignore_padding(use_lcm_gcd):
    torch.manual_seed(123)
    actor = CombinedPolicyNetwork(_config(use_lcm_gcd)).eval()
    pairs = torch.tensor([[[2.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 2.0]]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    reference = actor(ModelInputs(P=pairs, P_mask=mask))

    swapped = torch.cat((pairs[..., 2:], pairs[..., :2]), dim=-1)
    torch.testing.assert_close(actor(ModelInputs(P=swapped, P_mask=mask)), reference)

    padded_pairs = torch.cat((pairs, torch.full((1, 1, 4), 100.0)), dim=1)
    inputs = ModelInputs(
        P=torch.cat((padded_pairs, torch.full_like(padded_pairs, 200.0)), dim=0),
        P_mask=torch.tensor([[True, True, False], [False, False, False]]),
    )
    logits = actor(inputs)
    assert torch.isfinite(logits).all()
    torch.testing.assert_close(logits[:1, :2], reference)
    assert torch.count_nonzero(logits[~inputs.P_mask]) == 0
