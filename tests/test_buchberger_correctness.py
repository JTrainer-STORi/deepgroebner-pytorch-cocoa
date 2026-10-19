"""Check completed RL trajectories with CoCoALib's independent GBasis routine.

The native diagnostic compares the *actual*, potentially unfinished basis to a
reference computed from the original input. Recomputing GBasis from the final
candidate instead would let an incomplete Buchberger implementation pass.
"""

from contextlib import contextmanager

import gymnasium as gym
import numpy as np
import pytest

import deepgroebner_pytorch_cocoa  # noqa: F401: registers the Gym environments


ENVIRONMENTS = ["BuchbergerExponents-v1", "BuchbergerExponentsNoPQ-v1"]
ELIMINATIONS = ["none", "lcm", "gebauermoeller"]
SELECTORS = ["first", "degree", "normal", "sugar", "random"]
SORTING = [(False, False), (False, True), (True, False), (True, True)]
NON_GROEBNER_BASIS = ["x^2-y", "x*y-1"]
SYSTEMS = [
    pytest.param("QQ", "x,y", NON_GROEBNER_BASIS, id="rational-bivariate"),
    pytest.param(
        "QQ", "x,y", ["2*x^2-3*y", "5*x*y-2"], id="rational-nonmonic"
    ),
    pytest.param(
        "ZZmod(7)", "x,y,z", ["x*y-z", "x*z-y", "y*z-x"],
        id="finite-field-trivariate",
    ),
    pytest.param(
        "ZZmod(32003)", "x,y,z",
        ["x+y+z", "x*y+y*z+z*x", "x*y*z-1"], id="finite-field-cyclic",
    ),
]


@contextmanager
def _environment(gym_id, **kwargs):
    env = gym.make(gym_id, **kwargs)
    try:
        yield env
    finally:
        env.close()


def _finish_episode(env, obs, action_order="first"):
    rewards = []
    for step in range(300):
        valid = np.flatnonzero(obs["P_mask"])
        assert valid.size, "Expected an unfinished state with at least one valid action"
        if action_order == "last" or (action_order == "alternating" and step % 2):
            action = int(valid[-1])
        else:
            action = int(valid[0])
        obs, reward, terminated, truncated, _ = env.step(action)
        assert not truncated
        assert np.isfinite(reward) and reward <= -1
        assert terminated == (obs["state_features"][1] == 0)
        assert terminated == (not np.any(obs["P_mask"]))
        rewards.append(reward)
        if terminated:
            return obs, rewards
    pytest.fail("Small deterministic ideal did not terminate within 300 steps")


@pytest.mark.parametrize("fixed", [True, False], ids=["fixed-ideal", "generated-ideal"])
def test_removed_last_priority_queue_strategy_is_rejected(fixed):
    #arrange
    kwargs = (
        dict(coeff_ring="QQ", symbols="x,y", basis_elements=NON_GROEBNER_BASIS)
        if fixed else dict(ideal_dist="2-3-3-weighted")
    )

    #act
    #assert
    with pytest.raises(IndexError):
        with _environment("BuchbergerExponents-v1", pq_strategy="last", **kwargs):
            pytest.fail("Removed priority queue strategy was accepted")


@pytest.mark.parametrize("gym_id", ENVIRONMENTS)
def test_removed_last_heuristic_raises_a_python_exception(gym_id):
    #arrange
    with _environment(
        gym_id, coeff_ring="QQ", symbols="x,y", basis_elements=NON_GROEBNER_BASIS,
    ) as env:
        env.reset(seed=81)

        #act
        #assert
        with pytest.raises(IndexError):
            env.unwrapped.value(strategy="last")


@pytest.mark.parametrize("gym_id", ENVIRONMENTS)
@pytest.mark.parametrize("elimination", ELIMINATIONS)
@pytest.mark.parametrize("sort_input,sort_reducers", SORTING)
@pytest.mark.parametrize("coeff_ring,symbols,basis", SYSTEMS)
def test_completed_basis_matches_cocoa(
    gym_id, elimination, sort_input, sort_reducers, coeff_ring, symbols, basis
):
    #arrange
    with _environment(
        gym_id,
        coeff_ring=coeff_ring,
        symbols=symbols,
        basis_elements=basis,
        elimination=elimination,
        sort_input=sort_input,
        sort_reducers=sort_reducers,
        k=2,
    ) as env:
        obs, _ = env.reset(seed=124)

        #act
        final_obs, _ = _finish_episode(env, obs, "alternating")
        is_groebner_basis = env.unwrapped.env.is_groebner_basis()

        #assert
        assert not np.any(final_obs["P_mask"])
        assert is_groebner_basis


@pytest.mark.parametrize("elimination", ELIMINATIONS)
@pytest.mark.parametrize("selector", SELECTORS)
@pytest.mark.parametrize("action_order", ["first", "last", "alternating"])
def test_pq_selectors_and_hidden_basis_match_cocoa(elimination, selector, action_order):
    #arrange
    with _environment(
        "BuchbergerExponents-v1",
        coeff_ring="QQ",
        symbols="x,y,z",
        basis_elements=["x*y-z", "x*z-y", "y*z-x"],
        elimination=elimination,
        pq_strategy=selector,
        N_P=2,
        N_G=1,
        k=1,
    ) as env:
        obs, _ = env.reset(seed=53)

        #act
        final_obs, _ = _finish_episode(env, obs, action_order)
        is_groebner_basis = env.unwrapped.env.is_groebner_basis()
        copied_is_groebner_basis = env.unwrapped.env.copy().is_groebner_basis()

        #assert
        assert final_obs["G_mask"].sum() == 1
        assert final_obs["state_features"][0] > final_obs["G_mask"].sum()
        # The oracle inspects the complete native basis, including rows hidden
        # by the observation budget, and remains valid on a copied environment.
        assert is_groebner_basis
        assert copied_is_groebner_basis


@pytest.mark.parametrize("gym_id", ENVIRONMENTS)
@pytest.mark.parametrize("elimination", ELIMINATIONS)
def test_cocoa_oracle_rejects_unfinished_basis_without_mutating_it(gym_id, elimination):
    #arrange
    kwargs = dict(
        coeff_ring="QQ", symbols="x,y", basis_elements=NON_GROEBNER_BASIS,
        elimination=elimination,
    )
    with _environment(gym_id, **kwargs) as env, _environment(gym_id, **kwargs) as control:
        obs, _ = env.reset(seed=81)
        control_obs, _ = control.reset(seed=81)
        native = env.unwrapped.env
        fresh_native = type(native)()

        #act
        #assert
        with pytest.raises(RuntimeError, match="Call reset"):
            # A fresh native instance must not silently validate an empty basis.
            fresh_native.is_groebner_basis()

        #arrange
        expected_value = env.unwrapped.value(gamma=0.7)
        saved = {name: value.copy() for name, value in obs.items()}

        #act
        initial_is_groebner_basis = native.is_groebner_basis()
        copied_is_groebner_basis = native.copy().is_groebner_basis()
        repeated_is_groebner_basis = native.is_groebner_basis()
        actual_value = env.unwrapped.value(gamma=0.7)

        #assert
        # xy-1 and x^2-y generate an ideal whose Groebner basis also needs y^2-x.
        # An oracle that completed the candidate itself would return True here.
        assert not initial_is_groebner_basis
        assert not copied_is_groebner_basis
        assert not repeated_is_groebner_basis
        assert actual_value == pytest.approx(expected_value)
        for name, value in saved.items():
            np.testing.assert_array_equal(obs[name], value)

        #act
        final_obs, rewards = _finish_episode(env, obs)
        control_final, control_rewards = _finish_episode(control, control_obs)
        final_is_groebner_basis = native.is_groebner_basis()

        #assert
        assert rewards == control_rewards
        for name, value in final_obs.items():
            np.testing.assert_array_equal(value, control_final[name])
        assert final_is_groebner_basis
        assert not np.any(final_obs["P_mask"])


@pytest.mark.parametrize("gym_id", ENVIRONMENTS)
@pytest.mark.parametrize("elimination", ["lcm", "gebauermoeller"])
@pytest.mark.parametrize("basis", [["1"], ["x^2"], ["x^2", "y^3"]])
def test_fixed_already_groebner_ideals_reset_to_zero_pairs(gym_id, elimination, basis):
    #arrange
    with _environment(
        gym_id, coeff_ring="QQ", symbols="x,y", basis_elements=basis,
        elimination=elimination,
    ) as env:
        for _ in range(2):
            #act
            obs, _ = env.reset(seed=9)
            is_groebner_basis = env.unwrapped.env.is_groebner_basis()

            #assert
            assert not np.any(obs["P_mask"])
            assert obs["state_features"][1] == 0
            assert obs["state_features"][12] == 0
            assert is_groebner_basis
            for strategy in SELECTORS:
                #act
                value = env.unwrapped.value(strategy=strategy, gamma=0.5)

                #assert
                assert value == 0


@pytest.mark.parametrize("gym_id", ENVIRONMENTS)
@pytest.mark.parametrize("fixed", [True, False], ids=["fixed-ideal", "generated-ideal"])
def test_reward_modes_share_trajectory_and_count_reductions(gym_id, fixed):
    #arrange
    if fixed:
        kwargs = dict(
            coeff_ring="QQ", symbols="x,y",
            basis_elements=["x^2-y", "x*y-1", "y^2-x"],
        )
    else:
        kwargs = dict(ideal_dist="2-3-3-weighted")
    kwargs["elimination"] = "none"
    with _environment(gym_id, rewards="additions", **kwargs) as additions, _environment(
        gym_id, rewards="reductions", **kwargs
    ) as reductions:
        add_obs, _ = additions.reset(seed=14)
        red_obs, _ = reductions.reset(seed=14)

        #act
        add_final, add_rewards = _finish_episode(additions, add_obs)
        red_final, red_rewards = _finish_episode(reductions, red_obs)
        additions_is_groebner_basis = additions.unwrapped.env.is_groebner_basis()
        reductions_is_groebner_basis = reductions.unwrapped.env.is_groebner_basis()

        #assert
        assert len(add_rewards) == len(red_rewards)
        assert red_rewards == [-1.0] * len(red_rewards)
        assert sum(add_rewards) < sum(red_rewards)
        for name, value in add_final.items():
            np.testing.assert_array_equal(value, red_final[name])
        assert additions_is_groebner_basis
        assert reductions_is_groebner_basis


@pytest.mark.parametrize("selector", [name for name in SELECTORS if name != "random"])
@pytest.mark.parametrize("rewards", ["additions", "reductions"])
@pytest.mark.parametrize("gamma", [0.0, 0.7, 1.0])
def test_heuristic_value_matches_discounted_environment_rewards(selector, rewards, gamma):
    #arrange
    with _environment(
        "BuchbergerExponents-v1",
        coeff_ring="QQ", symbols="x,y,z",
        basis_elements=["x*y-z", "x*z-y", "y*z-x"],
        pq_strategy=selector, elimination="none", rewards=rewards, N_P=2,
    ) as env:
        obs, _ = env.reset(seed=51)
        expected = env.unwrapped.value(strategy=selector, gamma=gamma)

        #act
        _, step_rewards = _finish_episode(env, obs)
        actual = sum(gamma**step * reward for step, reward in enumerate(step_rewards))
        final_value = env.unwrapped.value(strategy=selector, gamma=gamma)
        is_groebner_basis = env.unwrapped.env.is_groebner_basis()

        #assert
        assert actual == pytest.approx(expected)
        assert final_value == 0
        assert is_groebner_basis
