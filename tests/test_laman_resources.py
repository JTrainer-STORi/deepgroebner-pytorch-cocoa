"""Bundled Laman graph tables work independently of the working directory."""

from pathlib import Path

import numpy as np
import pytest

from deepgroebner_pytorch_cocoa.greedy_eval import count_laman_graphs, geom_support_size
from deepgroebner_pytorch_cocoa.package_resources import (
    laman_graph_directory,
    laman_graph_path,
)
from deepgroebner_pytorch_cocoa.wrapped import CNoPQEnv, CGLeadMonomialsEnv


def test_python_laman_resources_from_another_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    assert laman_graph_directory().is_absolute()
    assert laman_graph_path(3).parent == laman_graph_directory()
    assert laman_graph_path(3).read_text(encoding="utf-8").strip() == "{7}"
    assert count_laman_graphs(3) == 1
    assert count_laman_graphs(4) == 1
    assert geom_support_size("geom-3") == 1
    assert geom_support_size("geom-(3-4)") == 2
    assert Path.cwd() == tmp_path


@pytest.mark.parametrize("n", [2, 11])
def test_laman_graph_path_rejects_unsupported_vertex_counts(n):
    with pytest.raises(ValueError, match="3 to 10"):
        laman_graph_path(n)


@pytest.mark.parametrize("env_type", [CNoPQEnv, CGLeadMonomialsEnv])
@pytest.mark.parametrize("ideal_dist", ["geom-3", "geom-(3-4)"])
def test_native_laman_sampling_from_another_directory(
    env_type, ideal_dist, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    env = env_type(ideal_dist=ideal_dist, elimination="none", k=1)
    env.seed(123)
    sampled_pair_counts = set()
    max_nvars = 3 if ideal_dist == "geom-3" else 5

    for _ in range(16):
        obs, info = env.reset()
        if isinstance(obs, dict):
            assert obs["G"].shape[1] == max_nvars
            assert obs["P"].shape[1] == 2 * max_nvars
            pair_count = int(obs["P_mask"].sum())
            assert len(info["basis_elements"]) in {3, 5}
        else:
            assert obs.shape[1] == 2 * max_nvars
            pair_count = info["N_Pairs"]
            assert obs.shape[0] == pair_count
        sampled_pair_counts.add(pair_count)
        assert Path.cwd() == tmp_path

    assert sampled_pair_counts == ({3} if ideal_dist == "geom-3" else {3, 10})


@pytest.mark.parametrize("env_type", [CNoPQEnv, CGLeadMonomialsEnv])
@pytest.mark.parametrize("ideal_dist", ["geom-3", "geom-(3-4)"])
def test_copied_native_laman_generator_keeps_resources_and_rng(
    env_type, ideal_dist, monkeypatch, tmp_path
):
    env = env_type(ideal_dist=ideal_dist, elimination="none", k=1)
    env.seed(123)
    env.reset()
    monkeypatch.chdir(tmp_path)
    copied_env = env.copy()

    for _ in range(4):
        obs, info = env.reset()
        copied_obs, copied_info = copied_env.reset()
        if isinstance(obs, dict):
            assert obs.keys() == copied_obs.keys()
            for key in obs:
                np.testing.assert_array_equal(obs[key], copied_obs[key])
            assert info["basis_elements"] == copied_info["basis_elements"]
        else:
            np.testing.assert_array_equal(obs, copied_obs)
            assert info == copied_info
        assert Path.cwd() == tmp_path
