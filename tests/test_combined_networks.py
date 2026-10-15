import pytest
import torch
import tyro

from deepgroebner_pytorch_cocoa.combined_networks import CombinedPolicyNetwork
from deepgroebner_pytorch_cocoa.config_dataclasses import ActorConfig, ModelInputs


@pytest.mark.parametrize("pair_head", ["per-entity-mlp", "symmetric-per-entity-mlp", "alphastar-buchberger-pair-transformer"])
@pytest.mark.parametrize("combiner", ["film", "sqa", "concat"])
@pytest.mark.parametrize("state_head,feature_head", [(None, None), ("deep-set", None), (None, "mlp"), ("deep-set", "mlp")])
def test_scoring_uses_available_state_heads(combiner, state_head, feature_head, pair_head):
    #arrange
    config = ActorConfig(
        state_embedding_head=state_head,
        state_features_embedding_head=feature_head,
        pair_embedding_head=pair_head,
        pair_combiner=combiner,
        use_lcm_gcd=False,
    )
    config.pair_embedding_head_in_dim = 6
    config.state_embedding_head_in_dim = 3
    config.state_features_embedding_head_in_dim = 4
    actor = CombinedPolicyNetwork(config)
    needs_basis = state_head is not None
    inputs = ModelInputs(
        P=torch.ones(2, 3, 6),
        P_mask=torch.tensor([[True, True, False], [True, False, False]]),
        G=torch.ones(2, 2, 3) if needs_basis else None,
        G_mask=torch.ones(2, 2, dtype=torch.bool) if needs_basis else None,
        state_features=torch.ones(2, 4) if feature_head else None,
    )
    #act
    logits = actor(inputs)

    #assert
    assert logits.shape == (2, 3, 1)
    assert torch.isfinite(logits[inputs.P_mask]).all()
    assert actor.pair_only_scoring == (state_head is None and feature_head is None)
    if actor.pair_only_scoring:
        #act
        expected = actor.final_module(actor.pair_embedding_head(inputs))
        expected = expected * inputs.P_mask.unsqueeze(-1)

        #assert
        torch.testing.assert_close(logits, expected)

    #act
    logits[inputs.P_mask].sum().backward()

    #assert
    for module in (actor.pair_embedding_head, actor.state_embedding_head, actor.state_features_embedding_head):
        if module is not None:
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            assert grads
            assert all(torch.isfinite(g).all() for g in grads)


def test_cli_accepts_disabled_state_heads_without_pair_combiner_override():
    #arrange
    args = ["--state-embedding-head=None", "--state-features-embedding-head=None"]

    #act
    config = tyro.cli(ActorConfig, args=args)

    #assert
    assert config.state_embedding_head is None
    assert config.state_features_embedding_head is None
    assert config.pair_combiner == "film"


def test_cli_rejects_removed_pair_only_option():
    #arrange
    args = ["--pair-combiner=pair-only"]

    #act
    #assert
    with pytest.raises(SystemExit) as exc:
        tyro.cli(ActorConfig, args=args)
    assert exc.value.code != 0


def test_model_rejects_removed_pair_only_option():
    #arrange
    config = ActorConfig(pair_combiner="pair-only")

    #act
    #assert
    with pytest.raises(ValueError, match="Unknown pair_combiner"):
        CombinedPolicyNetwork(config)


def test_asb_attention_handles_terminal_padding_without_affecting_valid_rows():
    #arrange
    from deepgroebner_pytorch_cocoa.network_components import AlphaStarPairSelfAttentionBlock

    block = AlphaStarPairSelfAttentionBlock(8, 2, use_layer_norm=True).eval()
    x = torch.randn(2, 3, 8, requires_grad=True)
    mask = torch.tensor([[True, True, False], [False, False, False]])
    #act
    output, weights = block(x, mask, return_attention=True)
    reference = block(x[:1, :2], mask[:1, :2])

    #assert
    torch.testing.assert_close(output[:1, :2], reference)
    assert torch.isfinite(output).all()
    assert torch.isfinite(weights).all()
    assert torch.count_nonzero(output[~mask]) == 0
    assert torch.count_nonzero(weights[1]) == 0

    #act
    output.sum().backward()

    #assert
    assert torch.isfinite(x.grad).all()

    #act
    empty, empty_weights = block(x[:, :0], mask[:, :0], return_attention=True)

    #assert
    assert empty.shape == (2, 0, 8)
    assert empty_weights.shape == (2, 2, 0, 0)
