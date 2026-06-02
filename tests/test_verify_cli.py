from __future__ import annotations

import numpy as np

from deepgroebner_pytorch_cocoa.verify_cli import _find_first_valid_action


def test_find_first_valid_action_returns_first_true_index():
    obs = {"P_mask": np.array([False, False, True, True])}
    assert _find_first_valid_action(obs) == 2
