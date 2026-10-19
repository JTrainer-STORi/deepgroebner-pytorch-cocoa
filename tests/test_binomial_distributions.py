"""Binomial distribution strings expose exactly four fields."""

from functools import partial

import gymnasium as gym
import numpy as np
import pytest

from deepgroebner_pytorch_cocoa.buchberger_env import parse_nvars_from_ideal_dist
from deepgroebner_pytorch_cocoa.wrapped import CGLeadMonomialsEnv, CNoPQEnv


ENVIRONMENTS = ["BuchbergerExponents-v1", "BuchbergerExponentsNoPQ-v1"]
DISTRIBUTIONS = ["3-3-3-weighted", "(2-3)-(2-3)-(2-3)-weighted"]
UNSUPPORTED_SUFFIXES = [
    "-consts",
    "-homog",
    "-pure",
    "-consts-homog",
    "-consts-pure",
    "-homog-pure",
    "-consts-homog-pure",
    "-unknown",
    "-",
]
CONSTRUCTORS = [
    pytest.param(CGLeadMonomialsEnv, id="native-pq"),
    pytest.param(CNoPQEnv, id="native-nopq"),
    pytest.param(partial(gym.make, ENVIRONMENTS[0]), id="gym-pq"),
    pytest.param(partial(gym.make, ENVIRONMENTS[1]), id="gym-nopq"),
]


@pytest.mark.parametrize("constructor", CONSTRUCTORS)
@pytest.mark.parametrize("ideal_dist", DISTRIBUTIONS, ids=["scalar", "ranged"])
@pytest.mark.parametrize("suffix", UNSUPPORTED_SUFFIXES)
def test_binomial_constructors_reject_optional_suffixes(constructor, ideal_dist, suffix):
    with pytest.raises(ValueError, match="exactly 4 fields"):
        constructor(ideal_dist=ideal_dist + suffix)


@pytest.mark.parametrize("ideal_dist", DISTRIBUTIONS, ids=["scalar", "ranged"])
@pytest.mark.parametrize("suffix", UNSUPPORTED_SUFFIXES)
def test_policy_variable_count_rejects_optional_suffixes(ideal_dist, suffix):
    with pytest.raises(ValueError, match="exactly 4 fields"):
        parse_nvars_from_ideal_dist(ideal_dist + suffix)


@pytest.mark.parametrize("gym_id", ENVIRONMENTS)
@pytest.mark.parametrize("distribution_type", ["weighted", "uniform", "maximum"])
@pytest.mark.parametrize(
    "parameters", ["3-3-3", "(2-3)-(2-3)-(2-3)"], ids=["scalar", "ranged"]
)
def test_four_field_binomial_distributions_support_complete_episodes(
    gym_id, distribution_type, parameters
):
    ideal_dist = f"{parameters}-{distribution_type}"
    assert parse_nvars_from_ideal_dist(ideal_dist) == 3
    env = gym.make(gym_id, ideal_dist=ideal_dist, elimination="none", k=2)
    try:
        obs, _ = env.reset(seed=12)
        for _ in range(300):
            assert obs["G"].shape[1] == 6
            assert obs["P"].shape[1] == 12
            valid_actions = np.flatnonzero(obs["P_mask"])
            assert valid_actions.size
            obs, reward, terminated, truncated, _ = env.step(int(valid_actions[0]))
            assert np.isfinite(reward)
            assert not truncated
            if terminated:
                break
        assert terminated, "Small binomial ideal did not complete within 300 steps"
        assert env.unwrapped.env.is_groebner_basis()
        assert obs["G"].shape[1] == 6
        assert obs["P"].shape[1] == 12
    finally:
        env.close()
