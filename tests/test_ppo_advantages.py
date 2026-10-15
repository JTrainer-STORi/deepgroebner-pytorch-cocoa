"""PPO return estimates checked against explicit forward sums."""

import pytest
import torch

from deepgroebner_pytorch_cocoa.ppo_optimization import compute_returns_and_advantages


@pytest.mark.parametrize(
    "use_gae,expected_advantages,expected_returns",
    [
        (
            True,
            [[1.85, 9.0], [1.0, 43.218], [4.2, 50.04], [2.0, 43.2]],
            [[2.35, 10.0], [2.0, 45.218], [5.7, 53.04], [4.0, 47.2]],
        ),
        (
            False,
            [[2.3, 9.0], [1.0, 83.232], [5.1, 69.48], [2.0, 43.2]],
            [[2.8, 10.0], [2.0, 85.232], [6.6, 72.48], [4.0, 47.2]],
        ),
    ],
)
def test_asynchronous_episode_boundaries_and_bootstrap_match_hand_calculations(
    use_gae, expected_advantages, expected_returns
):
    #arrange
    rewards = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    values = torch.tensor([[0.5, 1.0], [1.0, 2.0], [1.5, 3.0], [2.0, 4.0]])
    # dones[t] describes the transition preceding observation t.
    dones = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    next_value = torch.tensor([[99.0], [8.0]])
    next_done = torch.tensor([1.0, 0.0])
    expected_advantages = torch.tensor(expected_advantages)
    expected_returns = torch.tensor(expected_returns)

    #act
    advantages, returns = compute_returns_and_advantages(
        rewards=rewards,
        values=values,
        dones=dones,
        next_value=next_value,
        next_done=next_done,
        gamma=0.9,
        gae_lambda=0.5,
        use_gae=use_gae,
    )

    #assert
    torch.testing.assert_close(advantages, expected_advantages)
    torch.testing.assert_close(returns, expected_returns)


def _forward_reference(
    rewards, values, dones, next_value, next_done, gamma, gae_lambda, use_gae
):
    """Sum future rewards or TD errors directly, stopping at each episode end.

    This deliberately uses a separate forward sum for every starting time rather
    than the implementation's backwards recurrence.
    """
    num_steps, num_envs = rewards.shape
    advantages = torch.empty_like(rewards)
    returns = torch.empty_like(rewards)
    for env in range(num_envs):
        for start in range(num_steps):
            estimate = 0.0
            for future in range(start, num_steps):
                final_step = future == num_steps - 1
                done = (
                    next_done[env].item() if final_step else dones[future + 1, env].item()
                )
                next_v = (
                    next_value[env].item() if final_step else values[future + 1, env].item()
                )
                reward = rewards[future, env].item()
                if use_gae:
                    td_error = (
                        reward + gamma * next_v * (1 - done) - values[future, env].item()
                    )
                    estimate += (gamma * gae_lambda) ** (future - start) * td_error
                else:
                    estimate += gamma ** (future - start) * reward
                if done:
                    break
                if final_step and not use_gae:
                    estimate += gamma ** (future - start + 1) * next_v
            if use_gae:
                advantages[start, env] = estimate
                returns[start, env] = estimate + values[start, env]
            else:
                returns[start, env] = estimate
                advantages[start, env] = estimate - values[start, env]
    return advantages, returns


@pytest.mark.parametrize("use_gae", [False, True])
@pytest.mark.parametrize(
    "gamma,gae_lambda",
    [(0.0, 0.7), (0.73, 0.0), (0.91, 0.37), (0.97, 1.0), (1.0, 1.0)],
)
@pytest.mark.parametrize("num_steps,num_envs", [(1, 1), (5, 3)])
def test_estimates_match_forward_sums_for_discount_and_lambda_edge_cases(
    use_gae, gamma, gae_lambda, num_steps, num_envs
):
    #arrange
    generator = torch.Generator().manual_seed(1234)
    rewards = torch.randn(num_steps, num_envs, generator=generator, dtype=torch.float64)
    values = torch.randn(num_steps, num_envs, generator=generator, dtype=torch.float64)
    dones = torch.randint(0, 2, (num_steps, num_envs), generator=generator).double()
    next_value = torch.randn(num_envs, generator=generator, dtype=torch.float64)
    next_done = torch.arange(num_envs).remainder(2).double()
    expected = _forward_reference(
        rewards, values, dones, next_value, next_done, gamma, gae_lambda, use_gae
    )

    #act
    actual = compute_returns_and_advantages(
        rewards, values, dones, next_value, next_done,
        gamma=gamma, gae_lambda=gae_lambda, use_gae=use_gae,
    )

    #assert
    for calculated, reference in zip(actual, expected):
        torch.testing.assert_close(calculated, reference, rtol=1e-12, atol=1e-12)
        assert calculated.shape == (num_steps, num_envs)
        assert calculated.dtype == rewards.dtype


@pytest.mark.parametrize("next_value_shape", [(3,), (3, 1), (1, 3)])
@pytest.mark.parametrize("use_gae", [False, True])
def test_one_step_rollout_bootstraps_only_nonterminal_environments(next_value_shape, use_gae):
    #arrange
    rewards = torch.tensor([[1.0, 2.0, 3.0]])
    values = torch.tensor([[0.5, -1.0, 4.0]])
    dones = torch.tensor([[1.0, 0.0, 1.0]])
    next_value = torch.tensor([10.0, 2000.0, -5.0]).reshape(next_value_shape)
    next_done = torch.tensor([0.0, 1.0, 0.0])
    expected_returns = torch.tensor([[9.0, 2.0, -1.0]])
    expected_advantages = torch.tensor([[8.5, 3.0, -5.0]])

    #act
    advantages, returns = compute_returns_and_advantages(
        rewards=rewards,
        values=values,
        dones=dones,
        next_value=next_value,
        next_done=next_done,
        gamma=0.8,
        gae_lambda=0.6,
        use_gae=use_gae,
    )

    #assert
    torch.testing.assert_close(returns, expected_returns)
    torch.testing.assert_close(advantages, expected_advantages)


@pytest.mark.parametrize("use_gae", [False, True])
def test_estimates_do_not_mutate_rollouts_or_retain_autograd_graphs(use_gae):
    #arrange
    rewards = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    values = torch.tensor([[0.1, 0.2], [0.3, 0.4]], requires_grad=True)
    dones = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    next_value = torch.tensor([5.0, 6.0], requires_grad=True)
    next_done = torch.tensor([1.0, 0.0])
    inputs = (rewards, values, dones, next_value, next_done)
    before = tuple(value.detach().clone() for value in inputs)

    #act
    advantages, returns = compute_returns_and_advantages(
        *inputs, gamma=0.9, gae_lambda=0.8, use_gae=use_gae,
    )

    #assert
    for actual, expected in zip(inputs, before):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not advantages.requires_grad
    assert not returns.requires_grad
    assert advantages.grad_fn is None
    assert returns.grad_fn is None
