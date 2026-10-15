# Testing

Install the package and development dependencies as described in
[INSTALL.md](../INSTALL.md), then run the suite from the repository root:

```bash
python -m pytest -q
```

Run just the PPO numerical tests and Buchberger correctness tests with:

```bash
python -m pytest -q tests/test_ppo_normalization.py tests/test_ppo_advantages.py tests/test_ppo_optimization.py tests/test_ppo_rollout.py tests/test_buchberger_correctness.py
```

## PPO checks

The numerical tests use deterministic inputs and independently calculated
expectations for running moments, reward and observation normalization, PopArt,
discounted returns, and generalized advantage estimation. Episode boundaries
and bootstrap values are checked across parallel environments. Optimization
tests check the clipped policy and value objectives and their gradients using
a small trainable policy and critic. Rollout tests check that observations,
actions, values, advantages, returns, and heuristic predictions stay aligned
when fixed and variable length observations are shuffled into minibatches.

Reward normalization receives the completion flags for the transition that
produced the rewards. Its running statistics include each terminal reward
with its episode's discounted history, and the completed environment's
accumulator resets before the next transition. Advantage normalization also
supports a minibatch containing a single transition.

## CoCoALib reference checks

The Buchberger tests run the actual Gym environments to completion with
deterministic pair choices and check their native polynomial basis against
CoCoALib's `GBasis` for the original input ideal. The check compares leading
ideals and verifies polynomial membership using CoCoALib's normal remainder.
Computing a new Groebner basis from the environment's final generators alone
would hide missing S-pair processing, since an unfinished generating set can
still generate the correct ideal. Negative controls therefore also check
that an unfinished basis fails validation.

The native wrappers expose this check as `is_groebner_basis()`. For a Gym
environment `env`, use `env.unwrapped.env.is_groebner_basis()` after reset or
after stepping. This diagnostic is read-only and invokes CoCoALib on demand;
it is not part of training or observation construction. A basis can pass
before the episode terminates if only redundant pairs remain.

The reference tests use the CoCoALib already linked into the extension and
need no separate CoCoA executable or Python algebra package. After changing
the Cython or C++ sources, rebuild the installed extension before testing:

```bash
python -m pip install -e .
python -m pytest -q
```

Keep the `COCOA_INCLUDE_DIR`, `COCOA_LIB_DIR`, and any GMP settings from the
original installation when rebuilding against a custom CoCoALib location.
