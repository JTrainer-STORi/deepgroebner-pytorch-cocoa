import pytest
import torch
from torch.distributions import Categorical

from deepgroebner_pytorch_cocoa.ppo_agent import CategoricalMasked


@pytest.mark.parametrize("bad_logit", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("masked", [False, True])
def test_nonfinite_valid_logits_raise_even_when_torch_validation_is_disabled(
    bad_logit, masked
):
    #arrange
    logits = torch.tensor([[0.0, bad_logit, 1.0]])
    masks = torch.tensor([[True, True, False]]) if masked else None

    #act
    #assert
    with pytest.raises(ValueError, match="non-finite logits") as exc:
        CategoricalMasked(logits=logits, masks=masks, validate_args=False)
    if masked:
        assert "valid actions" in str(exc.value)


def test_all_nonfinite_valid_logits_raise_instead_of_sampling_invalid_actions():
    #arrange
    logits = torch.tensor([[float("nan"), float("nan"), 0.0]])
    masks = torch.tensor([[True, True, False]])

    #act
    #assert
    with pytest.raises(ValueError, match="non-finite logits.*valid actions"):
        CategoricalMasked(logits=logits, masks=masks)


@pytest.mark.parametrize("bad_logit", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_masked_out_logits_do_not_affect_probabilities_or_gradients(
    bad_logit,
):
    #arrange
    logits = torch.tensor([[0.0, bad_logit, 0.0]], requires_grad=True)

    #act
    distribution = CategoricalMasked(
        logits=logits,
        masks=torch.tensor([[True, False, True]]),
    )
    entropy = distribution.entropy()

    #assert
    torch.testing.assert_close(distribution.probs, torch.tensor([[0.5, 0.0, 0.5]]))
    torch.testing.assert_close(entropy, torch.tensor([2.0]).log())

    #act
    (-distribution.log_prob(torch.tensor([0]))).sum().backward()

    #assert
    torch.testing.assert_close(logits.grad, torch.tensor([[-0.5, 0.0, 0.5]]))


def test_finite_unmasked_logits_match_standard_categorical():
    #arrange
    logits = torch.tensor([[0.0, 2.0, -1.0], [3.0, -2.0, 1.0]])

    #act
    actual = CategoricalMasked(logits=logits)
    expected = Categorical(logits=logits)
    actual_entropy = actual.entropy()
    expected_entropy = expected.entropy()
    actual_log_prob = actual.log_prob(torch.tensor([1, 0]))
    expected_log_prob = expected.log_prob(torch.tensor([1, 0]))

    #assert
    torch.testing.assert_close(actual.probs, expected.probs)
    torch.testing.assert_close(actual_entropy, expected_entropy)
    torch.testing.assert_close(actual_log_prob, expected_log_prob)


def test_broadcast_action_masks_exclude_invalid_actions_for_every_batch_row():
    #arrange
    logits = torch.tensor([[0.0, 100.0, 0.0], [1.0, 100.0, 1.0]])
    masks = torch.tensor([True, False, True])

    #act
    distribution = CategoricalMasked(logits=logits, masks=masks)
    samples = distribution.sample((64,))

    #assert
    torch.testing.assert_close(
        distribution.probs, torch.tensor([[0.5, 0.0, 0.5], [0.5, 0.0, 0.5]])
    )
    assert masks[samples].all()
