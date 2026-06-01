from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
import os
import pathlib
import random
import re
import time
import datetime
import pprint as pp
import uuid
from collections import deque

import gymnasium as gym
import numpy as np
try:
    import wandb
except ImportError:  # pragma: no cover - exercised in installs without wandb
    wandb = None
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import tyro

from deepgroebner_pytorch_cocoa.buchberger_env import (
    _parse_class_list_token,
    _parse_int_or_range_token,
    _parse_nvars_from_ideal_dist,
    _split_top_level_dash,
)
from deepgroebner_pytorch_cocoa.combined_networks import (
    CombinedPolicyNetwork,
    CombinedValueNetwork,
    build_shared_actor_critic_state_trunk,
)

from deepgroebner_pytorch_cocoa.config_dataclasses import ModelInputs, Config
from deepgroebner_pytorch_cocoa.local_logging import make_summary_writer
from deepgroebner_pytorch_cocoa.system_metrics import (
    SystemMetricsRecorder,
    summarize_state_feature_rollout,
)
from deepgroebner_pytorch_cocoa.wrapped import CGLeadMonomialsEnv

from deepgroebner_pytorch_cocoa.diagnostics import run_actor_global_diagnostics, run_actor_ranking_diagnostics


# ------------------------------
# ------ Helper functions ------
# ------------------------------

def make_buffer(shape, dtype=torch.float32):
    return torch.zeros((cfg.training.num_steps_per_update, cfg.training.num_envs) + shape, dtype=dtype).to(device)


def _wandb_run_active() -> bool:
    return wandb is not None and wandb.run is not None


def _generate_run_id() -> str:
    if wandb is not None:
        return wandb.util.generate_id()
    return uuid.uuid4().hex[:8]


def _scalarize_metric(value):
    if torch.is_tensor(value):
        return float(value.detach().item())
    return float(value)


_VERBOSITY_LEVELS = {
    "quiet": 0,
    "normal": 1,
    "verbose": 2,
}


def _console_allows(cfg, level: str = "normal") -> bool:
    configured_level = getattr(cfg.training, "console_verbosity", "normal")
    return _VERBOSITY_LEVELS.get(configured_level, 1) >= _VERBOSITY_LEVELS[level]


def _console_print(cfg, *args, level: str = "normal", **kwargs) -> None:
    if _console_allows(cfg, level):
        print(*args, **kwargs)


def _format_optional_metric(value, *, precision: int = 4) -> str:
    if value is None:
        return "n/a"
    numeric_value = float(value)
    if not np.isfinite(numeric_value):
        return "n/a"
    return f"{numeric_value:.{precision}f}"


def _print_update_summary(
    *,
    cfg,
    update_idx: int,
    num_updates: int,
    global_step: int,
    sps: int,
    recent_returns,
    last_k_avg_return: float,
    policy_loss: float,
    entropy: float,
    approx_kl: float,
    critic_loss: float | None,
):
    last_episode_return = recent_returns[-1] if len(recent_returns) > 0 else None
    parts = [
        f"[update {update_idx}/{num_updates}]",
        f"step={global_step}",
        f"sps={sps}",
        f"avg100={_format_optional_metric(last_k_avg_return)}",
        f"last_ep={_format_optional_metric(last_episode_return)}",
        f"policy={_format_optional_metric(policy_loss)}",
        f"entropy={_format_optional_metric(entropy)}",
        f"approx_kl={_format_optional_metric(approx_kl)}",
    ]
    if critic_loss is not None:
        parts.append(f"critic={_format_optional_metric(critic_loss)}")
    _console_print(cfg, " ".join(parts), flush=True)


def _log_system_metrics(writer, record: dict[str, object], global_step: int):
    for key, value in record.items():
        if key in {"global_step", "policy_updates", "timestamp_utc"}:
            continue
        if isinstance(value, bool):
            writer.add_scalar(f"system/{key}", 1.0 if value else 0.0, global_step)
        elif isinstance(value, (int, float)) and np.isfinite(value):
            writer.add_scalar(f"system/{key}", value, global_step)


class SingleEnvAdapter:
    """Minimal vector-env compatible adapter for a single non-vectorized env."""

    def __init__(self, env: gym.Env):
        self.env = env
        self.num_envs = 1
        self.single_observation_space = env.observation_space
        self.single_action_space = env.action_space

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def _batch_obs(self, obs: dict):
        return {k: np.expand_dims(v, axis=0) for k, v in obs.items()}

    def reset(self, seed=None, options=None):
        if isinstance(seed, (list, tuple, np.ndarray)):
            seed = seed[0] if len(seed) > 0 else None
        obs, info = self.env.reset(seed=seed, options=options)
        return self._batch_obs(obs), info

    def step(self, action):
        if isinstance(action, (list, tuple, np.ndarray)):
            env_action = int(action[0])
        else:
            env_action = int(action)

        obs, reward, terminated, truncated, info = self.env.step(env_action)
        reset_obs = None
        if terminated or truncated:
            # Match vector env behavior: return done flags for this transition,
            # but provide the next (reset) observation for subsequent rollout steps.
            reset_obs, _ = self.env.reset()

        infos = {}
        if "episode" in info:
            infos["_episode"] = np.array([True], dtype=bool)
            infos["episode"] = {
                key: np.array([value])
                for key, value in info["episode"].items()
            }

        return (
            self._batch_obs(reset_obs if reset_obs is not None else obs),
            np.array([reward], dtype=np.float32),
            np.array([terminated], dtype=bool),
            np.array([truncated], dtype=bool),
            infos,
        )

    def value(self, strategy='degree', gamma=0.99):
        return self.env.unwrapped.value(strategy=strategy, gamma=gamma)

    def close(self):
        self.env.close()


class RunningMeanStd:
    """Tracks running mean/variance for vector features using a numerically stable merge."""

    def __init__(self, shape=(), epsilon=1e-4, device=None):
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = torch.tensor(epsilon, dtype=torch.float32, device=device)

    def update(self, x: torch.Tensor):
        x = x.to(dtype=torch.float32)
        if x.numel() == 0:
            return
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = torch.tensor(x.shape[0], dtype=torch.float32, device=x.device)
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x: torch.Tensor, epsilon=1e-8, clip_range: float | None = None):
        std = torch.sqrt(self.var + epsilon)
        y = (x - self.mean) / std
        if clip_range is not None and clip_range > 0:
            y = torch.clamp(y, -clip_range, clip_range)
        return y


class PopArt:
    """PopArt normalizer that preserves raw value outputs when target stats change."""

    def __init__(
        self,
        output_layer: nn.Linear,
        shape=(),
        epsilon=1e-8,
        min_std: float = 1e-4,
        device=None,
    ):
        if not isinstance(output_layer, nn.Linear):
            raise TypeError("PopArt requires an nn.Linear output layer.")

        stat_shape = torch.Size(shape) if isinstance(shape, tuple) else torch.Size((shape,)) if shape else torch.Size(())
        flat_dim = int(torch.tensor(stat_shape).prod().item()) if len(stat_shape) > 0 else 1
        if output_layer.out_features != flat_dim:
            raise ValueError(
                f"PopArt stats shape {tuple(stat_shape)} does not match output layer "
                f"out_features={output_layer.out_features}."
            )

        self.output_layer = output_layer
        self.epsilon = epsilon
        self.min_std = min_std
        self.mean = torch.zeros(stat_shape, dtype=torch.float32, device=device)
        self.var = torch.ones(stat_shape, dtype=torch.float32, device=device)
        self.count = torch.tensor(1e-4, dtype=torch.float32, device=device)

    def _std(self) -> torch.Tensor:
        return torch.sqrt(self.var + self.epsilon).clamp_min(self.min_std)

    def normalize(self, x: torch.Tensor, clip_range: float | None = None) -> torch.Tensor:
        y = (x - self.mean) / self._std()
        if clip_range is not None and clip_range > 0:
            y = torch.clamp(y, -clip_range, clip_range)
        return y

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self._std() + self.mean

    def update(self, targets: torch.Tensor):
        targets = targets.to(dtype=torch.float32)
        if targets.numel() == 0:
            return

        old_mean = self.mean.clone()
        old_std = self._std().clone()

        batch_mean = targets.mean(dim=0)
        batch_var = targets.var(dim=0, unbiased=False)
        batch_count = torch.tensor(targets.shape[0], dtype=torch.float32, device=targets.device)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

        new_std = self._std()
        self._preserve_outputs(old_mean, old_std, new_mean, new_std)

    def _preserve_outputs(
        self,
        old_mean: torch.Tensor,
        old_std: torch.Tensor,
        new_mean: torch.Tensor,
        new_std: torch.Tensor,
    ):
        old_mean_flat = old_mean.reshape(-1)
        old_std_flat = old_std.reshape(-1)
        new_mean_flat = new_mean.reshape(-1)
        new_std_flat = new_std.reshape(-1)

        with torch.no_grad():
            scale = (old_std_flat / new_std_flat).unsqueeze(-1)
            self.output_layer.weight.mul_(scale)
            if self.output_layer.bias is not None:
                self.output_layer.bias.copy_(
                    (old_std_flat * self.output_layer.bias + old_mean_flat - new_mean_flat)
                    / new_std_flat
                )


class RewardNormalizer:
    """Normalizes rewards using running statistics of discounted returns."""

    def __init__(self, num_envs, gamma, epsilon=1e-8, clip_range=10.0, device=None):
        self.gamma = gamma
        self.epsilon = epsilon
        self.clip_range = clip_range
        self.returns = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.rms = RunningMeanStd(shape=(), epsilon=1e-4, device=device)

    def normalize(self, rewards: torch.Tensor, dones: torch.Tensor):
        dones = dones.to(dtype=torch.float32)
        self.returns = self.returns * self.gamma * (1.0 - dones) + rewards
        self.rms.update(self.returns)
        std = torch.sqrt(self.rms.var + self.epsilon)
        normalized = rewards / std
        if self.clip_range is not None and self.clip_range > 0:
            normalized = torch.clamp(normalized, -self.clip_range, self.clip_range)
        return normalized


def normalized_masked_entropy(entropy: torch.Tensor, action_mask: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Normalizes masked categorical entropy by log(number of valid actions)."""
    valid_counts = action_mask.to(dtype=entropy.dtype).sum(dim=-1)
    normalizer = torch.log(valid_counts.clamp_min(1.0))
    normalized = torch.where(
        valid_counts > 1.0,
        entropy / normalizer.clamp_min(epsilon),
        torch.zeros_like(entropy),
    )
    return normalized


def _tensor_stat_to_float(x: torch.Tensor):
    """Converts scalar/vector tensors to a stable float for scalar logging."""
    return x.detach().mean().item()


def _find_last_linear(module: nn.Module) -> nn.Linear:
    for child in reversed(list(module.modules())):
        if isinstance(child, nn.Linear):
            return child
    raise ValueError("Could not find a final nn.Linear layer for PopArt.")


def build_value_normalizer(cfg: Config, agent: nn.Module, device=None) -> PopArt | None:
    """Builds and attaches PopArt only when the run has a learned critic."""
    if (
        not cfg.training.normalize_value_function
        or cfg.critic.critic_strategy == "heuristic-only"
    ):
        return None

    value_normalizer = PopArt(
        output_layer=_find_last_linear(agent.critic.value_head),
        shape=(),
        epsilon=cfg.training.value_normalization_epsilon,
        min_std=cfg.training.value_normalization_epsilon,
        device=device,
    )
    agent.critic.popart = value_normalizer
    return value_normalizer


def _jsonify_config_value(value):
    """Recursively converts config values into JSON-serializable Python types."""
    if isinstance(value, dict):
        return {k: _jsonify_config_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonify_config_value(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.device):
        return str(value)
    return value


def save_config_snapshot(cfg: Config, output_path: str):
    """Saves the full resolved run config to JSON for reproducible checkpoint loading."""
    cfg_dict = _jsonify_config_value(asdict(cfg))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=2, sort_keys=True)


def save_model_checkpoints(cfg, agent, run_name: str, checkpoint_suffix: str | None = None) -> dict[str, str]:
    """Saves configured actor/critic checkpoints and companion config snapshots."""
    model_group_subdir = cfg.training.wandb_group if cfg.training.wandb_group is not None else ""
    output_dir = cfg.training.output_directory if cfg.training.output_directory is not None else ""
    checkpoint_stem = run_name if checkpoint_suffix is None else f"{run_name}__{checkpoint_suffix}"
    saved_paths = {}

    if cfg.training.save_actor:
        actor_dir = os.path.join(output_dir, "models", "actors", model_group_subdir)
        os.makedirs(actor_dir, exist_ok=True)
        actor_model_path = os.path.join(actor_dir, f"{checkpoint_stem}.pt")
        torch.save(agent.actor.state_dict(), actor_model_path)
        save_config_snapshot(cfg, os.path.join(actor_dir, f"{checkpoint_stem}__config.json"))
        saved_paths["actor"] = actor_model_path

    if cfg.critic.critic_strategy != "heuristic-only" and cfg.training.save_critic:
        critic_dir = os.path.join(output_dir, "models", "critics", model_group_subdir)
        os.makedirs(critic_dir, exist_ok=True)
        critic_model_path = os.path.join(critic_dir, f"{checkpoint_stem}.pt")
        torch.save(agent.critic.state_dict(), critic_model_path)
        save_config_snapshot(cfg, os.path.join(critic_dir, f"{checkpoint_stem}__config.json"))
        saved_paths["critic"] = critic_model_path

    return saved_paths


def init_wandb_run(cfg: Config, run_name: str, wandb_id: str):
    """Initializes W&B with a configurable timeout and optional offline fallback."""
    if wandb is None:
        raise RuntimeError(
            "Weights & Biases tracking was requested, but `wandb` is not installed. "
            "Install wandb or run with `--training.track=False`."
        )
    is_sweep_run = os.environ.get("WANDB_SWEEP_ID") is not None
    init_kwargs = dict(
        id=wandb_id,
        project=cfg.training.wandb_project,
        entity=cfg.training.wandb_entity,
        sync_tensorboard=False,
        name=run_name,
        monitor_gym=True,
        save_code=True,
        settings=wandb.Settings(init_timeout=cfg.training.wandb_init_timeout),
    )
    if cfg.training.wandb_group is not None:
        base_wandb_dir = cfg.training.output_directory if cfg.training.output_directory is not None else "wandb"
        grouped_wandb_dir = os.path.join(base_wandb_dir, cfg.training.wandb_group)
        os.makedirs(grouped_wandb_dir, exist_ok=True)
        init_kwargs["dir"] = grouped_wandb_dir
    elif cfg.training.output_directory is not None:
        os.makedirs(cfg.training.output_directory, exist_ok=True)
        init_kwargs["dir"] = cfg.training.output_directory
    if cfg.training.wandb_group is not None:
        init_kwargs["group"] = cfg.training.wandb_group
    # In sweep runs, W&B injects parameter config already; avoid duplicate keys.
    if not is_sweep_run:
        init_kwargs["config"] = _jsonify_config_value(asdict(cfg))

    try:
        wandb.init(**init_kwargs)
    except Exception as err:
        timeout_msg = "Run initialization has timed out"
        if cfg.training.wandb_offline_on_timeout and timeout_msg in str(err):
            print(
                "wandb.init() timed out after "
                f"{cfg.training.wandb_init_timeout} seconds; retrying in offline mode."
            )
            init_kwargs["mode"] = "offline"
            wandb.init(**init_kwargs)
            return
        raise


def print_runtime_cpu_thread_info(envs, cfg):
    """Prints launch-time CPU threading/concurrency info for PyTorch and Gymnasium."""
    torch_intra_threads = torch.get_num_threads()
    torch_interop_threads = torch.get_num_interop_threads()
    omp_threads = os.environ.get("OMP_NUM_THREADS", "unset")
    mkl_threads = os.environ.get("MKL_NUM_THREADS", "unset")

    # Gymnasium vector envs parallelize over environments. For AsyncVectorEnv this
    # maps to worker processes; for SyncVectorEnv everything runs in the main process.
    gym_num_envs = int(getattr(envs, "num_envs", cfg.training.num_envs))
    gym_vector_mode = str(cfg.training.vectorization_mode)
    gym_worker_units = gym_num_envs if "async" in gym_vector_mode.lower() else 1

    _console_print(cfg, "\n=== Runtime CPU Concurrency Info ===", level="verbose")
    _console_print(
        cfg,
        "PyTorch threads: "
        f"intra_op={torch_intra_threads}, inter_op={torch_interop_threads}, "
        f"OMP_NUM_THREADS={omp_threads}, MKL_NUM_THREADS={mkl_threads}",
        level="verbose",
    )
    _console_print(
        cfg,
        "Gymnasium vectorization: "
        f"mode={gym_vector_mode}, num_envs={gym_num_envs}, worker_units={gym_worker_units}",
        level="verbose",
    )
    _console_print(cfg, "====================================\n", level="verbose")

def global_grad_norm(parameters):
    """Computes the global norm of the gradients across all parameters. Used for diagnostics"""
    total_norm_sq = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(2)
        total_norm_sq += param_norm.item() ** 2
    return total_norm_sq ** 0.5

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal initialization for the weights and constant initialization for the biases of a given layer. By default torch uses Kaiming initialization, but the PPO paper uses orthogonal initialization, so we follow that convention. The std argument is the gain for the orthogonal initialization, and by default is set to sqrt(2) which is commonly used for ReLU activations.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class CategoricalMasked(Categorical):
    """A Categorical distribution that applies a mask to the logits, setting the logits of invalid actions to a large negative value (effectively zeroing out their probabilities). 
    The masks argument should be a boolean tensor of the same shape as the logits, where True indicates a valid action and False indicates an invalid action."""
    _warned_nonfinite_logits = False

    def __init__(self, probs=None, logits=None, validate_args=None, masks=None, neg=-1e8):
        if masks is None:
            self.masks = None
            sanitized_logits = logits
            if logits is not None and not torch.isfinite(logits).all():
                if not CategoricalMasked._warned_nonfinite_logits:
                    print("[WARN] Non-finite policy logits detected; sanitizing to finite fallback values.")
                    CategoricalMasked._warned_nonfinite_logits = True
                sanitized_logits = torch.nan_to_num(logits, nan=neg, posinf=neg, neginf=neg)
            super().__init__(probs=probs, logits=sanitized_logits, validate_args=validate_args)
            return

        m = masks.to(dtype=torch.bool, device=logits.device)
        self.masks = m
        masked = torch.where(m, logits, logits.new_full((), neg))
        if not torch.isfinite(masked).all():
            if not CategoricalMasked._warned_nonfinite_logits:
                print("[WARN] Non-finite masked policy logits detected; sanitizing to finite fallback values.")
                CategoricalMasked._warned_nonfinite_logits = True
            masked = torch.nan_to_num(masked, nan=neg, posinf=neg, neginf=neg)
        super().__init__(probs=probs, logits=masked, validate_args=validate_args)

    def entropy(self):
        if self.masks is None:
            return super().entropy()
        p_log_p = self.probs * self.logits
        p_log_p = torch.where(self.masks, p_log_p, p_log_p.new_zeros(()))
        return -p_log_p.sum(-1)


def build_model_inputs_from_obs(obs: dict, device: torch.device) -> ModelInputs:
    """Converts a single-environment observation dict into batched ModelInputs."""
    G_poly_terms = None
    G_poly_term_mask = None
    if "G_terms" in obs and "G_term_counts" in obs:
        fallback_term_dim = int(np.asarray(obs["G"]).shape[-1]) if np.asarray(obs["G"]).ndim >= 2 else None
        G_poly_terms, G_poly_term_mask = build_padded_polynomial_term_tensors(
            [obs["G_terms"]],
            [obs["G_term_counts"]],
            device=device,
            fallback_term_dim=fallback_term_dim,
        )
    return ModelInputs(
        P=torch.from_numpy(obs["P"]).unsqueeze(0).to(device=device, dtype=torch.float32),
        P_idx=torch.from_numpy(obs["P_idx"]).unsqueeze(0).to(device=device, dtype=torch.long),
        P_idx_raw=(
            torch.from_numpy(obs["P_idx_raw"]).unsqueeze(0).to(device=device, dtype=torch.long)
            if "P_idx_raw" in obs
            else None
        ),
        G=torch.from_numpy(obs["G"]).unsqueeze(0).to(device=device, dtype=torch.float32),
        G_mask=torch.from_numpy(obs["G_mask"]).unsqueeze(0).to(device=device, dtype=torch.bool),
        P_mask=torch.from_numpy(obs["P_mask"]).unsqueeze(0).to(device=device, dtype=torch.bool),
        P_sugar=(
            torch.from_numpy(obs["P_sugar"]).unsqueeze(0).to(device=device, dtype=torch.float32)
            if "P_sugar" in obs
            else None
        ),
        P_normal_rank=(
            torch.from_numpy(obs["P_normal_rank"]).unsqueeze(0).to(device=device, dtype=torch.float32)
            if "P_normal_rank" in obs
            else None
        ),
        G_poly_terms=G_poly_terms,
        G_poly_term_mask=G_poly_term_mask,
        state_features=torch.from_numpy(obs["state_features"]).unsqueeze(0).to(device=device, dtype=torch.float32),
    )


STATE_FEATURE_P_CARDINALITY_IDX = 1
STATE_FEATURE_N_PAIRS_PROCESSED_IDX = 12
STATE_FEATURE_STEPS_SINCE_BASIS_UPDATE_IDX = 13

EVAL_TIMEOUT_GRACE_SECONDS = 30.0
EVAL_TIMEOUT_MAX_EXTENSIONS = 2
EVAL_TIMEOUT_PROGRESS_HISTORY_MAXLEN = 32
EVAL_TIMEOUT_MIN_PROGRESS_SAMPLES = 8
EVAL_TIMEOUT_RECENT_ACTIVITY_SECONDS = 10.0
EVAL_TIMEOUT_MIN_PAIR_CARDINALITY_DROP = 2


def _normalize_eval_state_features(
    model_input: ModelInputs,
    obs_sf_normalizer: RunningMeanStd | dict[str, torch.Tensor] | None,
    cfg,
):
    if obs_sf_normalizer is None:
        return

    if isinstance(obs_sf_normalizer, dict):
        mean = obs_sf_normalizer["mean"].to(
            device=model_input.state_features.device,
            dtype=model_input.state_features.dtype,
        )
        var = obs_sf_normalizer["var"].to(
            device=model_input.state_features.device,
            dtype=model_input.state_features.dtype,
        )
        model_input.state_features = torch.clamp(
            (model_input.state_features - mean) / torch.sqrt(var + cfg.training.observation_normalization_epsilon),
            -cfg.training.observation_normalization_clip,
            cfg.training.observation_normalization_clip,
        )
        return

    model_input.state_features = obs_sf_normalizer.normalize(
        model_input.state_features,
        epsilon=cfg.training.observation_normalization_epsilon,
        clip_range=cfg.training.observation_normalization_clip,
    )


def copy_obs_dict_to_model_inputs(obs: dict, dst: ModelInputs):
    """Copy NumPy observation arrays into preallocated ModelInputs tensors."""
    dst.P.copy_(torch.from_numpy(obs["P"]))
    dst.P_idx.copy_(torch.from_numpy(obs["P_idx"]))
    if dst.P_idx_raw is not None and "P_idx_raw" in obs:
        dst.P_idx_raw.copy_(torch.from_numpy(obs["P_idx_raw"]))
    dst.G.copy_(torch.from_numpy(obs["G"]))
    dst.G_mask.copy_(torch.from_numpy(obs["G_mask"]))
    dst.P_mask.copy_(torch.from_numpy(obs["P_mask"]))
    if dst.P_sugar is not None and "P_sugar" in obs:
        dst.P_sugar.copy_(torch.from_numpy(obs["P_sugar"]))
    if dst.P_normal_rank is not None and "P_normal_rank" in obs:
        dst.P_normal_rank.copy_(torch.from_numpy(obs["P_normal_rank"]))
    dst.state_features.copy_(torch.from_numpy(obs["state_features"]))


def bind_obs_dict_as_model_inputs_views(obs: dict) -> ModelInputs:
    """Create ModelInputs as zero-copy torch views over NumPy observation arrays (CPU only)."""
    return ModelInputs(
        P=torch.from_numpy(obs["P"]),
        P_idx=torch.from_numpy(obs["P_idx"]),
        P_idx_raw=torch.from_numpy(obs["P_idx_raw"]) if "P_idx_raw" in obs else None,
        G=torch.from_numpy(obs["G"]),
        G_mask=torch.from_numpy(obs["G_mask"]),
        P_mask=torch.from_numpy(obs["P_mask"]),
        P_sugar=torch.from_numpy(obs["P_sugar"]) if "P_sugar" in obs else None,
        P_normal_rank=torch.from_numpy(obs["P_normal_rank"]) if "P_normal_rank" in obs else None,
        state_features=torch.from_numpy(obs["state_features"]),
    )


def build_padded_polynomial_term_tensors(
    g_terms_list,
    g_term_counts_list,
    device: torch.device,
    fallback_term_dim: int | None = None,
):
    """Converts flattened full-term polynomial observations into padded [B, M, T, D] tensors."""
    batch_size = len(g_terms_list)
    max_polys = max(int(np.asarray(counts).shape[0]) for counts in g_term_counts_list) if batch_size > 0 else 0
    max_terms = 0
    term_dim = fallback_term_dim if fallback_term_dim is not None else 0

    for terms_arr, counts_arr in zip(g_terms_list, g_term_counts_list):
        terms_np = np.asarray(terms_arr)
        counts_np = np.asarray(counts_arr).reshape(-1)
        if counts_np.size > 0:
            max_terms = max(max_terms, int(np.max(counts_np)))
        if terms_np.ndim >= 2 and terms_np.shape[-1] > 0:
            term_dim = max(term_dim, int(terms_np.shape[-1]))

    G_poly_terms = torch.zeros((batch_size, max_polys, max_terms, term_dim), dtype=torch.float32, device=device)
    G_poly_term_mask = torch.zeros((batch_size, max_polys, max_terms), dtype=torch.bool, device=device)

    for batch_idx, (terms_arr, counts_arr) in enumerate(zip(g_terms_list, g_term_counts_list)):
        terms_np = np.asarray(terms_arr, dtype=np.float32)
        counts_np = np.asarray(counts_arr, dtype=np.int64).reshape(-1)
        offset = 0
        for poly_idx, term_count in enumerate(counts_np.tolist()):
            if term_count > 0:
                poly_terms = terms_np[offset:offset + term_count]
                poly_term_width = int(poly_terms.shape[-1]) if poly_terms.ndim >= 2 else 0
                G_poly_terms[batch_idx, poly_idx, :term_count, :poly_term_width].copy_(
                    torch.from_numpy(poly_terms).to(device=device, dtype=torch.float32)
                )
                G_poly_term_mask[batch_idx, poly_idx, :term_count] = True
            offset += term_count

    return G_poly_terms, G_poly_term_mask


def build_model_inputs_from_batched_obs(obs: dict, device: torch.device) -> ModelInputs:
    """Converts a batched observation dict (B, ...) into ModelInputs on the target device."""
    # Sequence spaces from Gym vector envs are represented as tuples of per-env arrays.
    # Collate and pad them into dense tensors so the model can process minibatches.
    if isinstance(obs["P"], (list, tuple)):
        batch_size = len(obs["P"])

        max_p = max(int(np.asarray(p).shape[0]) for p in obs["P"]) if batch_size > 0 else 0
        max_g = max(int(np.asarray(g).shape[0]) for g in obs["G"]) if batch_size > 0 else 0

        p_dim = max(
            (int(np.asarray(p).shape[-1]) if np.asarray(p).ndim >= 2 else 0)
            for p in obs["P"]
        ) if batch_size > 0 else 0
        g_dim = max(
            (int(np.asarray(g).shape[-1]) if np.asarray(g).ndim >= 2 else 0)
            for g in obs["G"]
        ) if batch_size > 0 else 0

        P = torch.zeros((batch_size, max_p, p_dim), dtype=torch.float32, device=device)
        P_idx = torch.zeros((batch_size, max_p, 2), dtype=torch.long, device=device)
        P_idx_raw = torch.zeros((batch_size, max_p, 2), dtype=torch.long, device=device) if "P_idx_raw" in obs else None
        G = torch.zeros((batch_size, max_g, g_dim), dtype=torch.float32, device=device)
        G_mask = torch.zeros((batch_size, max_g), dtype=torch.bool, device=device)
        P_mask = torch.zeros((batch_size, max_p), dtype=torch.bool, device=device)
        P_sugar = torch.zeros((batch_size, max_p), dtype=torch.float32, device=device) if "P_sugar" in obs else None
        P_normal_rank = torch.zeros((batch_size, max_p), dtype=torch.float32, device=device) if "P_normal_rank" in obs else None

        for i in range(batch_size):
            p_arr = np.asarray(obs["P"][i])
            p_idx_arr = np.asarray(obs["P_idx"][i])
            p_idx_raw_arr = np.asarray(obs["P_idx_raw"][i]) if "P_idx_raw" in obs else None
            g_arr = np.asarray(obs["G"][i])
            p_mask_arr = np.asarray(obs["P_mask"][i])
            g_mask_arr = np.asarray(obs["G_mask"][i])
            p_sugar_arr = np.asarray(obs["P_sugar"][i]) if "P_sugar" in obs else None
            p_normal_rank_arr = np.asarray(obs["P_normal_rank"][i]) if "P_normal_rank" in obs else None

            n_p = int(p_arr.shape[0])
            n_g = int(g_arr.shape[0])

            if n_p > 0:
                p_width = int(p_arr.shape[-1]) if p_arr.ndim >= 2 else 0
                P[i, :n_p, :p_width].copy_(torch.from_numpy(p_arr).to(device=device, dtype=torch.float32))
                P_idx[i, :n_p].copy_(torch.from_numpy(p_idx_arr).to(device=device, dtype=torch.long))
                if P_idx_raw is not None:
                    P_idx_raw[i, :n_p].copy_(torch.from_numpy(p_idx_raw_arr).to(device=device, dtype=torch.long))
                P_mask[i, :n_p].copy_(torch.from_numpy(p_mask_arr).to(device=device, dtype=torch.bool))
                if P_sugar is not None:
                    P_sugar[i, :n_p].copy_(torch.from_numpy(p_sugar_arr).to(device=device, dtype=torch.float32))
                if P_normal_rank is not None:
                    P_normal_rank[i, :n_p].copy_(torch.from_numpy(p_normal_rank_arr).to(device=device, dtype=torch.float32))
            if n_g > 0:
                g_width = int(g_arr.shape[-1]) if g_arr.ndim >= 2 else 0
                G[i, :n_g, :g_width].copy_(torch.from_numpy(g_arr).to(device=device, dtype=torch.float32))
                G_mask[i, :n_g].copy_(torch.from_numpy(g_mask_arr).to(device=device, dtype=torch.bool))

        G_poly_terms = None
        G_poly_term_mask = None
        if "G_terms" in obs and "G_term_counts" in obs:
            G_poly_terms, G_poly_term_mask = build_padded_polynomial_term_tensors(
                obs["G_terms"],
                obs["G_term_counts"],
                device=device,
                fallback_term_dim=g_dim,
            )

        return ModelInputs(
            P=P,
            P_idx=P_idx,
            P_idx_raw=P_idx_raw,
            G=G,
            G_mask=G_mask,
            P_mask=P_mask,
            P_sugar=P_sugar,
            P_normal_rank=P_normal_rank,
            G_poly_terms=G_poly_terms,
            G_poly_term_mask=G_poly_term_mask,
            state_features=torch.from_numpy(obs["state_features"]).to(device=device, dtype=torch.float32),
        )

    G_poly_terms = None
    G_poly_term_mask = None
    if "G_terms" in obs and "G_term_counts" in obs:
        g_terms_arr = np.asarray(obs["G_terms"])
        g_term_counts_arr = np.asarray(obs["G_term_counts"])
        if g_terms_arr.ndim >= 3 and g_term_counts_arr.ndim >= 2:
            g_terms_list = [g_terms_arr[i] for i in range(g_terms_arr.shape[0])]
            g_term_counts_list = [g_term_counts_arr[i] for i in range(g_term_counts_arr.shape[0])]
        else:
            g_terms_list = [obs["G_terms"]]
            g_term_counts_list = [obs["G_term_counts"]]
        G_poly_terms, G_poly_term_mask = build_padded_polynomial_term_tensors(
            g_terms_list,
            g_term_counts_list,
            device=device,
            fallback_term_dim=int(np.asarray(obs["G"]).shape[-1]) if np.asarray(obs["G"]).ndim >= 2 else None,
        )

    return ModelInputs(
        P=torch.from_numpy(obs["P"]).to(device=device, dtype=torch.float32),
        P_idx=torch.from_numpy(obs["P_idx"]).to(device=device, dtype=torch.long),
        P_idx_raw=(
            torch.from_numpy(obs["P_idx_raw"]).to(device=device, dtype=torch.long)
            if "P_idx_raw" in obs
            else None
        ),
        G=torch.from_numpy(obs["G"]).to(device=device, dtype=torch.float32),
        G_mask=torch.from_numpy(obs["G_mask"]).to(device=device, dtype=torch.bool),
        P_mask=torch.from_numpy(obs["P_mask"]).to(device=device, dtype=torch.bool),
        P_sugar=(
            torch.from_numpy(obs["P_sugar"]).to(device=device, dtype=torch.float32)
            if "P_sugar" in obs
            else None
        ),
        P_normal_rank=(
            torch.from_numpy(obs["P_normal_rank"]).to(device=device, dtype=torch.float32)
            if "P_normal_rank" in obs
            else None
        ),
        G_poly_terms=G_poly_terms,
        G_poly_term_mask=G_poly_term_mask,
        state_features=torch.from_numpy(obs["state_features"]).to(device=device, dtype=torch.float32),
    )


def collate_ragged_model_inputs(
    indices,
    obs_G_list,
    obs_P_list,
    p_idx_list,
    p_idx_raw_list,
    g_mask_list,
    p_mask_list,
    obs_sf_list,
    device: torch.device,
    p_sugar_list=None,
    p_normal_rank_list=None,
    g_poly_terms_list=None,
    g_poly_term_mask_list=None,
) -> ModelInputs:
    """Pads ragged per-sample tensors to per-minibatch maxima and returns a dense ModelInputs."""
    mb_size = len(indices)
    max_g = max(int(obs_G_list[i].shape[0]) for i in indices)
    max_p = max(int(obs_P_list[i].shape[0]) for i in indices)
    g_dim = max(int(obs_G_list[i].shape[-1]) for i in indices)
    p_dim = max(int(obs_P_list[i].shape[-1]) for i in indices)

    G = torch.zeros((mb_size, max_g, g_dim), dtype=torch.float32, device=device)
    P = torch.zeros((mb_size, max_p, p_dim), dtype=torch.float32, device=device)
    P_idx = torch.zeros((mb_size, max_p, 2), dtype=torch.long, device=device)
    P_idx_raw = torch.zeros((mb_size, max_p, 2), dtype=torch.long, device=device) if p_idx_raw_list is not None else None
    G_mask = torch.zeros((mb_size, max_g), dtype=torch.bool, device=device)
    P_mask = torch.zeros((mb_size, max_p), dtype=torch.bool, device=device)
    state_features = torch.stack([obs_sf_list[i] for i in indices], dim=0).to(device=device, dtype=torch.float32)

    P_sugar = None
    if p_sugar_list is not None:
        P_sugar = torch.zeros((mb_size, max_p), dtype=torch.float32, device=device)
    P_normal_rank = None
    if p_normal_rank_list is not None:
        P_normal_rank = torch.zeros((mb_size, max_p), dtype=torch.float32, device=device)
    G_poly_terms = None
    G_poly_term_mask = None
    if g_poly_terms_list is not None and g_poly_term_mask_list is not None:
        max_poly_terms = max(int(g_poly_terms_list[i].shape[1]) for i in indices)
        term_dim = max(int(g_poly_terms_list[i].shape[-1]) for i in indices)
        G_poly_terms = torch.zeros((mb_size, max_g, max_poly_terms, term_dim), dtype=torch.float32, device=device)
        G_poly_term_mask = torch.zeros((mb_size, max_g, max_poly_terms), dtype=torch.bool, device=device)

    for row, idx in enumerate(indices):
        g = obs_G_list[idx]
        p = obs_P_list[idx]
        p_idx = p_idx_list[idx]
        p_idx_raw = p_idx_raw_list[idx] if p_idx_raw_list is not None else None
        g_m = g_mask_list[idx]
        p_m = p_mask_list[idx]

        n_g = int(g.shape[0])
        n_p = int(p.shape[0])
        g_width = int(g.shape[-1])
        p_width = int(p.shape[-1])

        G[row, :n_g, :g_width].copy_(g)
        P[row, :n_p, :p_width].copy_(p)
        P_idx[row, :n_p].copy_(p_idx)
        if P_idx_raw is not None:
            P_idx_raw[row, :n_p].copy_(p_idx_raw)
        G_mask[row, :n_g].copy_(g_m)
        P_mask[row, :n_p].copy_(p_m)

        if P_sugar is not None:
            P_sugar[row, :n_p].copy_(p_sugar_list[idx])
        if P_normal_rank is not None:
            P_normal_rank[row, :n_p].copy_(p_normal_rank_list[idx])
        if G_poly_terms is not None:
            poly_terms = g_poly_terms_list[idx]
            poly_term_mask = g_poly_term_mask_list[idx]
            G_poly_terms[row, :poly_terms.shape[0], :poly_terms.shape[1], :poly_terms.shape[2]].copy_(poly_terms)
            G_poly_term_mask[row, :poly_term_mask.shape[0], :poly_term_mask.shape[1]].copy_(poly_term_mask)
        elif G_poly_term_mask is not None:
            poly_term_mask = g_poly_term_mask_list[idx]
            G_poly_term_mask[row, :poly_term_mask.shape[0], :poly_term_mask.shape[1]].copy_(poly_term_mask)

    return ModelInputs(
        P=P,
        P_idx=P_idx,
        P_idx_raw=P_idx_raw,
        G=G,
        G_mask=G_mask,
        P_mask=P_mask,
        P_sugar=P_sugar,
        P_normal_rank=P_normal_rank,
        G_poly_terms=G_poly_terms,
        G_poly_term_mask=G_poly_term_mask,
        state_features=state_features,
    )


def get_space_last_dim(obs_space, key: str) -> int:
    """Returns the last feature dimension for Box or Sequence observation subspaces."""
    space = obs_space[key]
    shape = getattr(space, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[-1])
    feature_space = getattr(space, "feature_space", None)
    if feature_space is not None:
        feature_shape = getattr(feature_space, "shape", None)
        if feature_shape is not None and len(feature_shape) > 0:
            return int(feature_shape[-1])
    raise ValueError(f"Could not infer feature dimension for observation key '{key}'.")


def _build_greedy_eval_episode_limits(
    obs: dict,
    cfg,
    *,
    benchmark_episode_result: dict | None = None,
) -> dict[str, int | None]:
    state_features = np.asarray(obs["state_features"], dtype=np.float32).reshape(-1)
    initial_pair_count = max(0, int(round(float(state_features[STATE_FEATURE_P_CARDINALITY_IDX]))))

    max_pairs_processed = None
    pairs_multiplier = getattr(cfg.training, "hparam_eval_max_pairs_processed_multiplier", None)
    if pairs_multiplier is not None:
        pairs_multiplier = float(pairs_multiplier)
        if pairs_multiplier > 0:
            slack = int(getattr(cfg.training, "hparam_eval_max_pairs_processed_slack", 0) or 0)
            max_pairs_processed = int(np.ceil((pairs_multiplier * max(1, initial_pair_count)) + slack))

    benchmark_pairs_processed_limit = None
    if benchmark_episode_result is not None:
        benchmark_multiplier = getattr(
            cfg.training,
            "hparam_eval_nopq_benchmark_pairs_processed_multiplier",
            None,
        )
        if benchmark_multiplier is not None:
            benchmark_multiplier = float(benchmark_multiplier)
            if benchmark_multiplier > 0:
                benchmark_slack = int(
                    getattr(cfg.training, "hparam_eval_nopq_benchmark_pairs_processed_slack", 0) or 0
                )
                benchmark_length = max(1, int(benchmark_episode_result.get("length", 0)))
                benchmark_pairs_processed_limit = int(
                    np.ceil((benchmark_multiplier * benchmark_length) + benchmark_slack)
                )
                if max_pairs_processed is None:
                    max_pairs_processed = benchmark_pairs_processed_limit
                else:
                    max_pairs_processed = min(max_pairs_processed, benchmark_pairs_processed_limit)

    max_steps_since_basis_update = getattr(
        cfg.training,
        "hparam_eval_max_steps_since_basis_update",
        None,
    )
    if max_steps_since_basis_update is not None:
        max_steps_since_basis_update = int(max_steps_since_basis_update)
        if max_steps_since_basis_update <= 0:
            max_steps_since_basis_update = None

    return {
        "initial_pair_count": initial_pair_count,
        "max_pairs_processed": max_pairs_processed,
        "benchmark_pairs_processed_limit": benchmark_pairs_processed_limit,
        "max_steps_since_basis_update": max_steps_since_basis_update,
    }


def _greedy_eval_episode_progress_state(
    obs: dict,
    ep_return: float,
    ep_length: int,
    truncated_reason: str | None,
    hit_zero_action_state: bool,
    ) -> dict:
    state_features = np.asarray(obs["state_features"], dtype=np.float32).reshape(-1)
    return {
        "return": float(ep_return),
        "length": int(ep_length),
        "truncated_reason": truncated_reason,
        "hit_zero_action_state": bool(hit_zero_action_state),
        "pair_cardinality": int(round(float(state_features[STATE_FEATURE_P_CARDINALITY_IDX]))),
        "n_pairs_processed": int(round(float(state_features[STATE_FEATURE_N_PAIRS_PROCESSED_IDX]))),
        "steps_since_basis_update": int(round(float(state_features[STATE_FEATURE_STEPS_SINCE_BASIS_UPDATE_IDX]))),
        "n_pairs": int(np.asarray(obs["P"]).shape[0]),
        "n_basis": int(np.asarray(obs["G"]).shape[0]),
    }


def _summarize_greedy_eval_scalar_array(values: list[float] | list[int]) -> dict[str, float | None]:
    """Returns mean/median/min/max for eval aggregates, or None when empty."""
    if not values:
        return {
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
    values_arr = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(values_arr.mean()),
        "median": float(np.median(values_arr)),
        "min": float(values_arr.min()),
        "max": float(values_arr.max()),
    }


def _should_extend_greedy_eval_timeout(
    progress_history: deque[dict],
    *,
    now: float,
) -> bool:
    """Returns True when recent progress suggests the episode is still converging."""
    if len(progress_history) < EVAL_TIMEOUT_MIN_PROGRESS_SAMPLES:
        return False

    recent = list(progress_history)
    last_update_time = float(recent[-1]["received_at"])
    if now - last_update_time > EVAL_TIMEOUT_RECENT_ACTIVITY_SECONDS:
        return False

    pair_cardinalities = [
        int(entry["pair_cardinality"])
        for entry in recent
        if entry.get("pair_cardinality") is not None
    ]
    if len(pair_cardinalities) < EVAL_TIMEOUT_MIN_PROGRESS_SAMPLES:
        return False

    start_pair_count = pair_cardinalities[0]
    end_pair_count = pair_cardinalities[-1]
    if (start_pair_count - end_pair_count) < EVAL_TIMEOUT_MIN_PAIR_CARDINALITY_DROP:
        return False

    decreasing_steps = sum(
        1
        for prev_value, next_value in zip(pair_cardinalities, pair_cardinalities[1:])
        if next_value < prev_value
    )
    return decreasing_steps >= max(3, len(pair_cardinalities) // 3)


def _build_greedy_eval_watchdog_result(
    last_progress: dict,
    *,
    truncated_reason: str,
    elapsed_seconds: float,
    timed_out: bool,
    timeout_extensions_used: int,
    worker_exitcode: int | None = None,
    worker_error: str | None = None,
) -> dict:
    result = dict(last_progress)
    result["timed_out"] = bool(timed_out)
    result["truncated_reason"] = truncated_reason
    result["elapsed_seconds"] = float(elapsed_seconds)
    result["initial_pair_count"] = None
    result["max_pairs_processed_limit"] = None
    result["benchmark_pairs_processed_limit"] = None
    result["max_steps_since_basis_update_limit"] = None
    result["timeout_extensions_used"] = int(timeout_extensions_used)
    if worker_exitcode is not None:
        result["worker_exitcode"] = int(worker_exitcode)
    if worker_error is not None:
        result["worker_error"] = str(worker_error)
    return result


def _print_greedy_eval_progress_line(
    cfg,
    progress_prefix: str,
    seed: int,
    progress_state: dict,
    *,
    elapsed_seconds: float,
) -> None:
    _console_print(
        cfg,
        f"{progress_prefix}: seed={seed}, length={int(progress_state.get('length', 0))}, "
        f"n_pairs={progress_state.get('n_pairs')}, n_basis={progress_state.get('n_basis')}, "
        f"processed={int(progress_state.get('n_pairs_processed', 0))}, "
        f"stagnation={int(progress_state.get('steps_since_basis_update', 0))}, "
        f"elapsed={elapsed_seconds:.1f}s",
        flush=True,
        level="verbose",
    )


def _run_single_greedy_eval_episode(
    actor: nn.Module,
    eval_env: gym.Env,
    seed: int,
    device: torch.device,
    obs_sf_normalizer: RunningMeanStd | dict[str, torch.Tensor] | None,
    cfg,
    benchmark_episode_result: dict | None = None,
    *,
    progress_callback=None,
    progress_interval_seconds: float | None = None,
    progress_prefix: str = "Greedy eval progress",
) -> dict:
    obs, _ = eval_env.reset(seed=seed)
    state_features = np.asarray(obs["state_features"], dtype=np.float32).reshape(-1)
    initial_pair_count = max(0, int(round(float(state_features[STATE_FEATURE_P_CARDINALITY_IDX]))))

    done = False
    truncated = False
    ep_return = 0.0
    ep_length = 0
    hit_zero_action_state = False
    truncated_reason = None
    episode_start_time = time.monotonic()
    next_progress_time = (
        episode_start_time + progress_interval_seconds
        if progress_interval_seconds is not None and progress_interval_seconds > 0
        else None
    )

    while not (done or truncated):
        model_input = build_model_inputs_from_obs(obs, device)
        _normalize_eval_state_features(model_input, obs_sf_normalizer, cfg)

        valid_action_count = int(model_input.P_mask.sum().item())
        if valid_action_count == 0:
            done = True
            hit_zero_action_state = True
            break

        with torch.no_grad():
            logits = actor(model_input).squeeze(-1)
            masked_logits = torch.where(
                model_input.P_mask,
                logits,
                logits.new_full((), -1e8),
            )
            action = int(torch.argmax(masked_logits, dim=-1).item())

        obs, reward, done, truncated, _ = eval_env.step(action)
        ep_return += float(reward)
        ep_length += 1

        progress_state = _greedy_eval_episode_progress_state(
            obs,
            ep_return,
            ep_length,
            truncated_reason,
            hit_zero_action_state,
        )
        if progress_callback is not None:
            progress_callback(progress_state)

        if next_progress_time is not None:
            now = time.monotonic()
            if now >= next_progress_time:
                _console_print(
                    cfg,
                    f"{progress_prefix}: seed={seed}, length={ep_length}, "
                    f"n_pairs={progress_state['n_pairs']}, n_basis={progress_state['n_basis']}, "
                    f"processed={progress_state['n_pairs_processed']}, "
                    f"stagnation={progress_state['steps_since_basis_update']}, "
                    f"elapsed={now - episode_start_time:.1f}s",
                    flush=True,
                    level="verbose",
                )
                next_progress_time = now + progress_interval_seconds

    result = _greedy_eval_episode_progress_state(
        obs,
        ep_return,
        ep_length,
        truncated_reason,
        hit_zero_action_state,
    )
    result["elapsed_seconds"] = float(time.monotonic() - episode_start_time)
    result["timed_out"] = False
    result["timeout_extensions_used"] = 0
    result["initial_pair_count"] = int(initial_pair_count)
    result["max_pairs_processed_limit"] = None
    result["benchmark_pairs_processed_limit"] = None
    result["max_steps_since_basis_update_limit"] = None
    return result


def _make_obs_sf_normalizer_snapshot(
    obs_sf_normalizer: RunningMeanStd | None,
) -> dict[str, torch.Tensor] | None:
    if obs_sf_normalizer is None:
        return None
    return {
        "mean": obs_sf_normalizer.mean.detach().cpu(),
        "var": obs_sf_normalizer.var.detach().cpu(),
    }


def _run_greedy_eval_episode_worker(
    conn,
    actor_state_dict: dict[str, torch.Tensor],
    actor_cfg,
    eval_gym_id: str,
    env_kwargs: dict,
    seed: int,
    obs_sf_normalizer_state: dict[str, torch.Tensor] | None,
    cfg,
    benchmark_episode_result: dict | None,
    progress_interval_seconds: float | None,
    progress_prefix: str,
):
    env = None
    try:
        env = gym.make(eval_gym_id, **env_kwargs)
        actor = CombinedPolicyNetwork(actor_cfg).to(device="cpu")
        actor.load_state_dict(actor_state_dict)
        actor.eval()

        def _send_progress(progress_state: dict):
            conn.send({"kind": "progress", "state": progress_state})

        result = _run_single_greedy_eval_episode(
            actor=actor,
            eval_env=env,
            seed=seed,
            device=torch.device("cpu"),
            obs_sf_normalizer=obs_sf_normalizer_state,
            cfg=cfg,
            benchmark_episode_result=benchmark_episode_result,
            progress_callback=_send_progress,
            progress_interval_seconds=progress_interval_seconds,
            progress_prefix=progress_prefix,
        )
        conn.send({"kind": "final", "result": result})
    except BaseException as exc:
        conn.send({"kind": "error", "error": repr(exc)})
    finally:
        if env is not None:
            env.close()
        conn.close()


def _run_single_greedy_eval_episode_with_timeout(
    actor: nn.Module,
    eval_gym_id: str,
    env_kwargs: dict,
    seed: int,
    obs_sf_normalizer: RunningMeanStd | None,
    cfg,
    timeout_seconds: float,
    benchmark_episode_result: dict | None = None,
    progress_interval_seconds: float | None = None,
    progress_prefix: str = "Greedy eval progress",
) -> dict:
    ctx = mp.get_context("spawn")
    recv_conn, send_conn = ctx.Pipe(duplex=False)
    actor_state_dict = {
        key: value.detach().cpu()
        for key, value in actor.state_dict().items()
    }
    process = ctx.Process(
        target=_run_greedy_eval_episode_worker,
        args=(
            send_conn,
            actor_state_dict,
            cfg.actor,
            eval_gym_id,
            env_kwargs,
            seed,
            _make_obs_sf_normalizer_snapshot(obs_sf_normalizer),
            cfg,
            benchmark_episode_result,
            None,
            progress_prefix,
        ),
    )
    process.start()
    send_conn.close()

    watchdog_start_time = time.monotonic()
    deadline = time.monotonic() + timeout_seconds
    last_progress = {
        "return": 0.0,
        "length": 0,
        "truncated_reason": None,
        "hit_zero_action_state": False,
        "pair_cardinality": None,
        "n_pairs_processed": 0,
        "steps_since_basis_update": 0,
        "n_pairs": None,
        "n_basis": None,
    }
    progress_history: deque[dict] = deque(maxlen=EVAL_TIMEOUT_PROGRESS_HISTORY_MAXLEN)
    timeout_extensions_used = 0
    next_progress_print_time = (
        watchdog_start_time + progress_interval_seconds
        if progress_interval_seconds is not None and progress_interval_seconds > 0
        else None
    )
    try:
        while True:
            now = time.monotonic()
            remaining = deadline - now
            if remaining <= 0:
                if (
                    timeout_extensions_used < EVAL_TIMEOUT_MAX_EXTENSIONS
                    and _should_extend_greedy_eval_timeout(progress_history, now=now)
                ):
                    timeout_extensions_used += 1
                    deadline = now + EVAL_TIMEOUT_GRACE_SECONDS
                    _console_print(
                        cfg,
                        f"{progress_prefix}: seed={seed}, extending timeout by "
                        f"{EVAL_TIMEOUT_GRACE_SECONDS:.1f}s because pair_cardinality "
                        f"continues to shrink; current_pair_cardinality="
                        f"{last_progress.get('pair_cardinality')}, "
                        f"extensions_used={timeout_extensions_used}/{EVAL_TIMEOUT_MAX_EXTENSIONS}",
                        flush=True,
                        level="verbose",
                    )
                    continue

                process.terminate()
                process.join(timeout=1.0)
                return _build_greedy_eval_watchdog_result(
                    last_progress,
                    truncated_reason="episode_seconds",
                    elapsed_seconds=float(now - watchdog_start_time),
                    timed_out=True,
                    timeout_extensions_used=timeout_extensions_used,
                    worker_exitcode=process.exitcode,
                )

            poll_timeout = remaining
            if next_progress_print_time is not None:
                poll_timeout = min(poll_timeout, max(0.0, next_progress_print_time - now))

            if recv_conn.poll(timeout=poll_timeout):
                try:
                    message = recv_conn.recv()
                except (EOFError, OSError) as exc:
                    process.join(timeout=1.0)
                    return _build_greedy_eval_watchdog_result(
                        last_progress,
                        truncated_reason="worker_exit",
                        elapsed_seconds=float(time.monotonic() - watchdog_start_time),
                        timed_out=False,
                        timeout_extensions_used=timeout_extensions_used,
                        worker_exitcode=process.exitcode,
                        worker_error=f"{type(exc).__name__}: {exc}",
                    )
                kind = message.get("kind")
                if kind == "progress":
                    last_progress = dict(message["state"])
                    progress_history.append(
                        {
                            "received_at": time.monotonic(),
                            "pair_cardinality": last_progress.get("pair_cardinality"),
                        }
                    )
                    continue
                if kind == "final":
                    result = dict(message["result"])
                    result["timed_out"] = False
                    result["timeout_extensions_used"] = int(timeout_extensions_used)
                    process.join(timeout=1.0)
                    return result
                if kind == "error":
                    process.join(timeout=1.0)
                    raise RuntimeError(f"Greedy eval worker failed for seed {seed}: {message['error']}")
            else:
                now = time.monotonic()
                if (
                    next_progress_print_time is not None
                    and now >= next_progress_print_time
                ):
                    _print_greedy_eval_progress_line(
                        cfg,
                        progress_prefix,
                        seed,
                        last_progress,
                        elapsed_seconds=float(now - watchdog_start_time),
                    )
                    next_progress_print_time = now + progress_interval_seconds
                    continue
                if not process.is_alive():
                    break
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
        recv_conn.close()

    raise RuntimeError(f"Greedy eval worker exited without reporting a result for seed {seed}.")


def run_fixed_greedy_eval(
    actor: nn.Module,
    eval_gym_id: str,
    eval_corpus_rows: list[dict],
    device: torch.device,
    obs_sf_normalizer: RunningMeanStd | None,
    cfg,
    benchmark_episode_results_by_seed: dict[int, dict] | None = None,
) -> dict:
    """Runs greedy policy evaluation on a fixed ideal corpus and returns aggregate stats."""
    was_training = actor.training
    actor.eval()

    returns = []
    lengths = []
    completed_returns = []
    completed_lengths = []
    zero_action_episodes = 0
    timed_out_episodes = 0
    worker_exited_episodes = 0
    truncated_max_pairs_processed = 0
    truncated_steps_since_basis_update = 0
    truncated_episode_seconds = 0
    truncated_worker_exit = 0
    benchmark_truncated_episodes = 0
    episode_results = []
    eval_start_time = time.monotonic()
    progress_interval = float(getattr(cfg.training, "hparam_eval_progress_interval_seconds", 0.0) or 0.0)
    timeout_seconds = getattr(cfg.training, "hparam_eval_max_episode_seconds", None)
    if timeout_seconds is not None:
        timeout_seconds = float(timeout_seconds)
        if timeout_seconds <= 0:
            timeout_seconds = None

    for row_idx, row in enumerate(eval_corpus_rows, start=1):
        episode_seed = int(row["episode_seed"])
        env_kwargs = build_fixed_ideal_env_kwargs_for_gym_id(cfg, eval_gym_id, row)
        benchmark_episode_result = None
        if benchmark_episode_results_by_seed is not None:
            benchmark_episode_result = benchmark_episode_results_by_seed.get(episode_seed)

        if timeout_seconds is not None:
            result = _run_single_greedy_eval_episode_with_timeout(
                actor=actor,
                eval_gym_id=eval_gym_id,
                env_kwargs=env_kwargs,
                seed=episode_seed,
                obs_sf_normalizer=obs_sf_normalizer,
                cfg=cfg,
                timeout_seconds=timeout_seconds,
                benchmark_episode_result=benchmark_episode_result,
                progress_interval_seconds=progress_interval,
                progress_prefix=f"Greedy eval progress: example={row_idx}/{len(eval_corpus_rows)}",
            )
        else:
            eval_env = gym.make(eval_gym_id, **env_kwargs)
            try:
                result = _run_single_greedy_eval_episode(
                    actor=actor,
                    eval_env=eval_env,
                    seed=episode_seed,
                    device=device,
                    obs_sf_normalizer=obs_sf_normalizer,
                    cfg=cfg,
                    benchmark_episode_result=benchmark_episode_result,
                    progress_interval_seconds=progress_interval,
                    progress_prefix=f"Greedy eval progress: example={row_idx}/{len(eval_corpus_rows)}",
                )
            finally:
                eval_env.close()

        if result["hit_zero_action_state"]:
            zero_action_episodes += 1
        if result.get("timed_out", False):
            timed_out_episodes += 1

        truncated_reason = result.get("truncated_reason")
        if truncated_reason == "max_pairs_processed":
            truncated_max_pairs_processed += 1
        elif truncated_reason == "steps_since_basis_update":
            truncated_steps_since_basis_update += 1
        elif truncated_reason == "episode_seconds":
            truncated_episode_seconds += 1
        elif truncated_reason == "worker_exit":
            truncated_worker_exit += 1
            worker_exited_episodes += 1

        if (
            result.get("benchmark_pairs_processed_limit") is not None
            and truncated_reason == "max_pairs_processed"
        ):
            benchmark_truncated_episodes += 1

        returns.append(float(result["return"]))
        lengths.append(int(result["length"]))
        if truncated_reason is None:
            completed_returns.append(float(result["return"]))
            completed_lengths.append(int(result["length"]))
        episode_results.append(
            {
                "ideal_id": row.get("ideal_id"),
                "episode_seed": episode_seed,
                **result,
            }
        )

    if was_training:
        actor.train()

    returns_summary = _summarize_greedy_eval_scalar_array(returns)
    lengths_summary = _summarize_greedy_eval_scalar_array(lengths)
    completed_returns_summary = _summarize_greedy_eval_scalar_array(completed_returns)
    completed_lengths_summary = _summarize_greedy_eval_scalar_array(completed_lengths)
    num_examples = int(len(eval_corpus_rows))
    num_completed = int(len(completed_returns))
    truncated_total = int(sum(1 for episode_result in episode_results if episode_result.get("truncated_reason") is not None))
    return {
        "mean_return": returns_summary["mean"],
        "median_return": returns_summary["median"],
        "min_return": returns_summary["min"],
        "max_return": returns_summary["max"],
        "mean_length": lengths_summary["mean"],
        "max_length": 0 if lengths_summary["max"] is None else int(lengths_summary["max"]),
        "mean_return_completed_only": completed_returns_summary["mean"],
        "median_return_completed_only": completed_returns_summary["median"],
        "min_return_completed_only": completed_returns_summary["min"],
        "max_return_completed_only": completed_returns_summary["max"],
        "mean_length_completed_only": completed_lengths_summary["mean"],
        "max_length_completed_only": (
            0 if completed_lengths_summary["max"] is None else int(completed_lengths_summary["max"])
        ),
        "num_examples": num_examples,
        "num_completed": num_completed,
        "completion_rate": (
            float(num_completed / num_examples)
            if num_examples > 0
            else None
        ),
        "num_zero_action": int(zero_action_episodes),
        "num_timed_out": int(timed_out_episodes),
        "num_worker_exited": int(worker_exited_episodes),
        "num_truncated_max_pairs_processed": int(truncated_max_pairs_processed),
        "num_truncated_steps_since_basis_update": int(truncated_steps_since_basis_update),
        "num_truncated_episode_seconds": int(truncated_episode_seconds),
        "num_truncated_worker_exit": int(truncated_worker_exit),
        "num_benchmark_truncated": int(benchmark_truncated_episodes),
        "num_truncated_total": truncated_total,
        "elapsed_seconds": float(time.monotonic() - eval_start_time),
        "episode_results": episode_results,
    }


def benchmark_supports_n(ideal_class: str, n: int) -> bool:
    if ideal_class == "katsura":
        return n >= 1
    return n >= 2


def enumerate_benchmark_distributions(ideal_dist: str) -> list[str] | None:
    dist_args = _split_top_level_dash(ideal_dist.strip())
    if len(dist_args) != 2 or not dist_args[0]:
        return None

    head = dist_args[0]
    if head == "geom":
        return None
    if not (
        head[0].isalpha()
        or (head[0] == "(" and len(head) > 1 and head[1].isalpha())
    ):
        return None

    classes = _parse_class_list_token(head)
    n_lo, n_hi = _parse_int_or_range_token(dist_args[1], "n")

    members = [
        f"{ideal_class}-{n}"
        for ideal_class in classes
        for n in range(n_lo, n_hi + 1)
        if benchmark_supports_n(ideal_class, n)
    ]
    if not members:
        raise ValueError(f"No valid benchmark ideals for ideal_dist: {ideal_dist}")
    return members


def count_laman_graphs(n: int) -> int:
    data_path = pathlib.Path(__file__).resolve().parent / "data" / "LamanGraphs3-10" / f"LamanGraphs{n}.txt"
    text = data_path.read_text(encoding="utf-8")
    return len(re.findall(r"-?\d+", text))


def geom_support_size(ideal_dist: str) -> int | None:
    dist_args = _split_top_level_dash(ideal_dist.strip())
    if len(dist_args) != 2 or dist_args[0] != "geom":
        return None

    n_lo, n_hi = _parse_int_or_range_token(dist_args[1], "n")
    if n_lo < 3 or n_hi > 10:
        raise ValueError(f"Malformed geom ideal_dist: {ideal_dist}. Expected 3 <= n <= 10")

    return sum(count_laman_graphs(n) for n in range(n_lo, n_hi + 1))


def finite_ideal_support_size(ideal_dist: str | None) -> int | None:
    if ideal_dist is None:
        return 1

    benchmark_members = enumerate_benchmark_distributions(ideal_dist)
    if benchmark_members is not None:
        return len(benchmark_members)

    return geom_support_size(ideal_dist)


def collect_unique_ideal_eval_seeds(cfg, target_count: int, *, base_seed: int | None = None) -> list[int]:
    """Finds seeds that produce unique ideals for finite-support distributions."""
    if target_count <= 0:
        return []

    env = CGLeadMonomialsEnv(
        ideal_dist=cfg.env.ideal_dist,
        elimination=cfg.env.elimination,
        rewards=cfg.env.rewards,
        pq_strategy=cfg.env.priority_queue_strategy,
        sort_input=cfg.env.sort_input,
        sort_reducers=cfg.env.sort_reducers,
        fix_ideal=cfg.env.fix_ideal,
        N=cfg.env.N,
        k=cfg.env.k,
        N_P=cfg.env.N_P,
        N_G=cfg.env.N_G,
    )

    seen: set[tuple[str, ...]] = set()
    seeds: list[int] = []
    max_attempts = max(target_count * 20, target_count + 1_000)
    if base_seed is None:
        base_seed = int(cfg.training.hparam_eval_seed)

    for attempt in range(max_attempts):
        seed = int(base_seed) + attempt
        env.seed(seed)
        _, info = env.reset()
        basis_key = tuple(info.get("basis_elements", []))
        if basis_key in seen:
            continue
        seen.add(basis_key)
        seeds.append(seed)
        if len(seeds) >= target_count:
            break

    if len(seeds) < target_count:
        _console_print(
            cfg,
            "Greedy eval could only find "
            f"{len(seeds)} unique ideals out of requested finite support {target_count} "
            f"for ideal_dist={cfg.env.ideal_dist!r}.",
            flush=True,
        )

    return seeds


def build_greedy_eval_seeds(cfg, greedy_eval_enabled: bool) -> list[int]:
    """Returns eval seeds, avoiding duplicate work for finite-support ideal sets."""
    if not greedy_eval_enabled:
        return []

    num_examples = int(cfg.training.hparam_eval_num_examples)
    if num_examples <= 0:
        return []

    fixed_ideal_eval = (
        bool(cfg.env.fix_ideal)
        or cfg.env.ideal_dist is None
        or cfg.env.basis_elements is not None
    )
    if fixed_ideal_eval:
        if num_examples > 1:
            _console_print(
                cfg,
                "Greedy eval is using a fixed ideal; evaluating one unique seed "
                f"instead of repeating {num_examples} identical episodes.",
                flush=True,
            )
        num_examples = 1

    finite_support = finite_ideal_support_size(cfg.env.ideal_dist)
    if finite_support is not None and not fixed_ideal_eval:
        target_count = min(num_examples, finite_support)
        if target_count < num_examples:
            _console_print(
                cfg,
                "Greedy eval is using finite ideal support "
                f"({finite_support} ideals for {cfg.env.ideal_dist!r}); "
                f"evaluating {target_count} unique ideals instead of {num_examples}.",
                flush=True,
            )
        return collect_unique_ideal_eval_seeds(
            cfg,
            target_count,
            base_seed=int(cfg.training.hparam_eval_seed),
        )

    return [cfg.training.hparam_eval_seed + i for i in range(num_examples)]


def _split_symbols(symbols: str) -> list[str]:
    return [tok.strip() for tok in symbols.split(",") if tok.strip()]


def _resolve_corpus_symbols(corpus_rows: list[dict]) -> str:
    if not corpus_rows:
        raise ValueError("Empty greedy eval corpus.")

    parsed_rows: list[list[str]] = []
    widest_symbols: list[str] = []
    for row_idx, row in enumerate(corpus_rows):
        symbols = row.get("symbols")
        if not isinstance(symbols, str) or not symbols.strip():
            raise ValueError(f"Greedy eval corpus row {row_idx} is missing a non-empty 'symbols' field.")
        row_symbols = _split_symbols(symbols)
        parsed_rows.append(row_symbols)
        if len(row_symbols) > len(widest_symbols):
            widest_symbols = row_symbols.copy()

    resolved = widest_symbols.copy()
    seen = set(resolved)
    for row_symbols in parsed_rows:
        for symbol in row_symbols:
            if symbol not in seen:
                seen.add(symbol)
                resolved.append(symbol)
    return ",".join(resolved)


def _infer_symbols_from_basis_elements(basis_elements: list[str]) -> str:
    seen: set[str] = set()
    ordered_symbols: list[str] = []
    for basis_element in basis_elements:
        for match in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*(?:\[\d+\])?", basis_element):
            symbol = match.group(0)
            if symbol not in seen:
                seen.add(symbol)
                ordered_symbols.append(symbol)
    if not ordered_symbols:
        raise ValueError("Unable to infer any symbols from greedy eval basis elements.")
    return ",".join(ordered_symbols)


def _resolve_symbols_for_eval_corpus(cfg, basis_elements: list[str]) -> str:
    if cfg.env.symbols is not None:
        return cfg.env.symbols
    if cfg.env.ideal_dist is None:
        return _infer_symbols_from_basis_elements(basis_elements)

    dist_args = _split_top_level_dash(cfg.env.ideal_dist.strip())
    if len(dist_args) == 2 and dist_args[0] == "geom":
        _, hi = _parse_int_or_range_token(dist_args[1], "n")
        symbols = [f"x{i}" for i in range(2, hi + 1)]
        symbols.extend(f"y{i}" for i in range(3, hi + 1))
        return ",".join(symbols)

    n_vars = _parse_nvars_from_ideal_dist(cfg.env.ideal_dist)
    return ",".join(f"x[{i}]" for i in range(n_vars))


def build_greedy_eval_corpus(cfg, eval_seeds: list[int]) -> list[dict]:
    """Materializes a fixed ideal corpus so all eval targets share the same examples."""
    if not eval_seeds:
        return []

    env = CGLeadMonomialsEnv(
        ideal_dist=cfg.env.ideal_dist,
        elimination=cfg.env.elimination,
        rewards=cfg.env.rewards,
        pq_strategy=cfg.env.priority_queue_strategy,
        sort_input=cfg.env.sort_input,
        sort_reducers=cfg.env.sort_reducers,
        fix_ideal=cfg.env.fix_ideal,
        N=cfg.env.N,
        k=cfg.env.k,
        N_P=cfg.env.N_P,
        N_G=cfg.env.N_G,
        coeff_ring=cfg.env.coeff_ring,
        symbols=cfg.env.symbols,
        basis=cfg.env.basis_elements,
    )

    corpus_rows = []
    default_coeff_ring = cfg.env.coeff_ring if cfg.env.coeff_ring is not None else "QQ"
    for ideal_id, seed in enumerate(eval_seeds):
        env.seed(seed)
        _, info = env.reset()
        basis_elements = list(info.get("basis_elements", []))
        if len(basis_elements) == 0:
            raise ValueError(f"Greedy eval corpus row {ideal_id} had no basis_elements for seed {seed}.")
        symbols = _resolve_symbols_for_eval_corpus(cfg, basis_elements)
        corpus_rows.append(
            {
                "ideal_id": ideal_id,
                "episode_seed": int(seed),
                "distribution": cfg.env.ideal_dist,
                "coeff_ring": default_coeff_ring,
                "symbols": symbols,
                "basis_elements": basis_elements,
            }
        )

    corpus_symbols = _resolve_corpus_symbols(corpus_rows)
    for row in corpus_rows:
        row["symbols"] = corpus_symbols

    return corpus_rows


def _greedy_eval_corpus_cache_root(cache_root: str | pathlib.Path | None = None) -> pathlib.Path:
    if cache_root is not None:
        return pathlib.Path(cache_root)
    return pathlib.Path(".") / "eval_corpora" / "greedy_eval"


def _greedy_eval_corpus_cache_key(cfg, eval_seeds: list[int]) -> str:
    descriptor = {
        "ideal_dist": cfg.env.ideal_dist,
        "base_seed": int(cfg.training.hparam_eval_seed),
        "eval_seeds": [int(seed) for seed in eval_seeds],
        "fix_ideal": bool(cfg.env.fix_ideal),
        "basis_elements": cfg.env.basis_elements,
    }
    descriptor_json = json.dumps(descriptor, sort_keys=True)
    digest = hashlib.sha1(descriptor_json.encode("utf-8")).hexdigest()[:12]
    ideal_label = cfg.env.ideal_dist if cfg.env.ideal_dist is not None else "fixed_ideal"
    ideal_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", ideal_label).strip("_") or "ideal"
    ideal_label = ideal_label[:80]
    return (
        f"{ideal_label}__seed_{int(cfg.training.hparam_eval_seed)}"
        f"__n_{len(eval_seeds)}__{digest}"
    )


def greedy_eval_corpus_cache_path(
    cfg,
    eval_seeds: list[int],
    *,
    cache_root: str | pathlib.Path | None = None,
) -> pathlib.Path | None:
    if not eval_seeds:
        return None
    root = _greedy_eval_corpus_cache_root(cache_root)
    return root / f"{_greedy_eval_corpus_cache_key(cfg, eval_seeds)}.jsonl"


def load_greedy_eval_corpus(corpus_path: str | pathlib.Path) -> list[dict]:
    rows = []
    with pathlib.Path(corpus_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_greedy_eval_corpus(
    corpus_rows: list[dict],
    corpus_path: str | pathlib.Path,
) -> str | None:
    """Persists the materialized greedy-eval corpus as JSONL for exact replay."""
    if not corpus_rows:
        return None

    path = pathlib.Path(corpus_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in corpus_rows:
            f.write(json.dumps(row))
            f.write("\n")
    return str(path)


def resolve_greedy_eval_corpus(
    cfg,
    eval_seeds: list[int],
    *,
    cache_root: str | pathlib.Path | None = None,
) -> tuple[list[dict], str | None, bool]:
    """Loads a cached greedy-eval corpus when available, otherwise builds and saves it."""
    corpus_path = greedy_eval_corpus_cache_path(cfg, eval_seeds, cache_root=cache_root)
    if corpus_path is None:
        return [], None, False

    expected_seeds = [int(seed) for seed in eval_seeds]
    if corpus_path.exists():
        corpus_rows = load_greedy_eval_corpus(corpus_path)
        loaded_seeds = [int(row.get("episode_seed")) for row in corpus_rows]
        if loaded_seeds != expected_seeds:
            raise ValueError(
                "Cached greedy eval corpus seeds do not match requested eval seeds: "
                f"path={corpus_path}, loaded={loaded_seeds}, expected={expected_seeds}"
            )
        return corpus_rows, str(corpus_path), False

    corpus_rows = build_greedy_eval_corpus(cfg, eval_seeds)
    save_greedy_eval_corpus(corpus_rows, corpus_path)
    return corpus_rows, str(corpus_path), True


def resolve_greedy_eval_targets(cfg) -> dict:
    """Resolves which greedy-eval envs to run from the explicit config toggles."""
    eval_enabled = (
        cfg.training.hparam_eval_interval_steps > 0
        and cfg.training.hparam_eval_num_examples > 0
    )
    if not eval_enabled:
        return {
            "enabled": False,
            "run_train_env": False,
            "run_nopq_env": False,
            "primary_target": None,
            "nopq_gym_id": cfg.training.hparam_eval_nopq_gym_id,
        }

    run_train_env = bool(cfg.training.hparam_eval_on_train_env)
    requested_run_nopq_env = bool(cfg.training.hparam_eval_on_nopq_env)
    nopq_gym_id = cfg.training.hparam_eval_nopq_gym_id

    nopq_matches_train_env = (nopq_gym_id.lower() == cfg.env.gym_id.lower())
    run_nopq_env = requested_run_nopq_env and not nopq_matches_train_env
    if run_train_env and run_nopq_env:
        primary_target = "nopq"
    else:
        primary_target = "train" if run_train_env else ("nopq" if run_nopq_env else None)

    return {
        "enabled": run_train_env or run_nopq_env,
        "run_train_env": run_train_env,
        "run_nopq_env": run_nopq_env,
        "primary_target": primary_target,
        "nopq_gym_id": nopq_gym_id,
    }


def compute_greedy_eval_selection_metric(
    train_stats: dict | None,
    nopq_stats: dict | None,
    *,
    run_train_env: bool,
    run_nopq_env: bool,
) -> tuple[float | None, str | None]:
    """Returns the checkpoint-selection metric and a label describing it."""
    def _metric_from_stats(
        stats: dict | None,
        label: str,
    ) -> tuple[float | None, str]:
        if stats is None:
            return None, label
        if int(stats.get("num_truncated_total", 0)) > 0:
            return None, f"{label}_requires_untruncated_eval"
        metric = stats.get("mean_return_completed_only", stats.get("mean_return"))
        if metric is None:
            return None, f"{label}_requires_completed_episodes"
        return float(metric), label

    if run_nopq_env:
        return _metric_from_stats(nopq_stats, "nopq_mean_return")
    if run_train_env:
        return _metric_from_stats(train_stats, "train_mean_return")
    return None, None


def build_env_kwargs_for_gym_id(cfg, gym_id: str) -> dict:
    """Builds env kwargs and drops unsupported options for NoPQ environments."""
    env_kwargs = {
        'ideal_dist': cfg.env.ideal_dist,
        'elimination': cfg.env.elimination,
        'rewards': cfg.env.rewards,
        'pq_strategy': cfg.env.priority_queue_strategy,
        'sort_input': cfg.env.sort_input,
        'sort_reducers': cfg.env.sort_reducers,
        'fix_ideal': cfg.env.fix_ideal,
        'N': cfg.env.N,
        'N_P': cfg.env.N_P,
        'N_G': cfg.env.N_G,
        'k': cfg.env.k,
        'coeff_ring': cfg.env.coeff_ring,
        'symbols': cfg.env.symbols,
        'basis_elements': cfg.env.basis_elements,
    }
    if "nopq" in gym_id.lower():
        env_kwargs.pop('pq_strategy', None)
        env_kwargs.pop('N', None)
        env_kwargs.pop('N_P', None)
        env_kwargs.pop('N_G', None)
    return env_kwargs


def build_fixed_ideal_env_kwargs_for_gym_id(cfg, gym_id: str, corpus_row: dict) -> dict:
    """Builds env kwargs for evaluating a specific fixed ideal corpus row."""
    env_kwargs = {
        "ideal_dist": None,
        "elimination": cfg.env.elimination,
        "rewards": cfg.env.rewards,
        "sort_input": cfg.env.sort_input,
        "sort_reducers": cfg.env.sort_reducers,
        "fix_ideal": True,
        "k": cfg.env.k,
        "coeff_ring": corpus_row["coeff_ring"],
        "symbols": corpus_row["symbols"],
        "basis_elements": corpus_row["basis_elements"],
    }
    if "nopq" not in gym_id.lower():
        env_kwargs["pq_strategy"] = cfg.env.priority_queue_strategy
        env_kwargs["N"] = cfg.env.N
        env_kwargs["N_P"] = cfg.env.N_P
        env_kwargs["N_G"] = cfg.env.N_G
    return env_kwargs
    
# -------------------------------
# ------ Policy Config ----------
# -------------------------------

class Agent(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.shared_actor_critic_trunk = None

        if cfg.training.share_actor_critic_trunk:
            if cfg.critic.critic_strategy == "heuristic-only":
                raise ValueError(
                    "cfg.training.share_actor_critic_trunk requires a learned critic."
                )
            self.shared_actor_critic_trunk = build_shared_actor_critic_state_trunk(
                cfg.actor,
                cfg.critic,
            )

        shared_state_kwargs = {}
        if self.shared_actor_critic_trunk is not None:
            shared_state_kwargs = {
                "state_embedding_head": self.shared_actor_critic_trunk.state_embedding_head,
                "state_features_embedding_head": self.shared_actor_critic_trunk.state_features_embedding_head,
                "state_feature_combiner": self.shared_actor_critic_trunk.state_feature_combiner,
            }

        if cfg.critic.critic_strategy != "heuristic-only":
            self.critic = CombinedValueNetwork(cfg.critic, **shared_state_kwargs)
            self.critic_train = self.critic
        else:
            self.critic_train = None
        self.actor = CombinedPolicyNetwork(cfg.actor, **shared_state_kwargs)
        self.actor_train = self.actor

        if self.shared_actor_critic_trunk is not None:
            shared_param_ids = {id(param) for param in self.shared_actor_critic_trunk.parameters()}
            self.shared_trunk_parameters = list(self.shared_actor_critic_trunk.parameters())
            self.actor_head_parameters = [
                param for param in self.actor.parameters()
                if id(param) not in shared_param_ids
            ]
            self.critic_head_parameters = (
                [
                    param for param in self.critic.parameters()
                    if id(param) not in shared_param_ids
                ]
                if cfg.critic.critic_strategy != "heuristic-only"
                else []
            )
        else:
            self.shared_trunk_parameters = []
            self.actor_head_parameters = list(self.actor.parameters())
            self.critic_head_parameters = (
                list(self.critic.parameters())
                if cfg.critic.critic_strategy != "heuristic-only"
                else []
            )

    def configure_training_modules(self, actor_train: nn.Module, critic_train: nn.Module | None = None):
        """Sets modules used on the hot training path; keep eager modules for diagnostics."""
        self.actor_train = actor_train
        if self.critic_train is not None and critic_train is not None:
            self.critic_train = critic_train

    def get_value(self, inputs):

        if cfg.critic.critic_strategy == "learned-only":
            # If the value is coming only from the learned critic, then we can just pass the inputs through the critic as normal without worrying about computing a heuristic value or passing it in as an input
            values, _, _ = self.critic_train(inputs, h_value=None)
            return values, None
        else:
            # If we are using the heuristic only strategy or the heuristic plus learned residual strategy, then we need to compute the heuristic value and pass it in as an input to the critic (in the case of heuristic plus learned residual) or use it as the value directly (in the case of heuristic only)
            if inputs.h_value is None:
                if isinstance(envs, gym.vector.SyncVectorEnv):
                    h_value = torch.zeros(inputs.G.shape[0]).to(device)
                    for i, env in enumerate(envs.envs):
                        h_value[i] = torch.tensor(env.unwrapped.value(strategy=cfg.critic.value_function_heuristic, gamma=cfg.training.gamma)).to(device)
                elif isinstance(envs, gym.vector.AsyncVectorEnv):
                    h_value = torch.tensor(envs.call('value', strategy=cfg.critic.value_function_heuristic, gamma=cfg.training.gamma)).to(device)
                else:
                    h_value = torch.tensor(
                        [envs.value(strategy=cfg.critic.value_function_heuristic, gamma=cfg.training.gamma)],
                        device=device,
                    )
                h_value = h_value.unsqueeze(-1)
            else:
                h_value = inputs.h_value
            if cfg.critic.critic_strategy == "heuristic-only":
                return h_value, h_value
            elif cfg.critic.critic_strategy == "heuristic-plus-learned-residual":
                values, _, _ = self.critic_train(inputs, h_value=h_value)
                return values, h_value

    def get_action_and_value(self, inputs: ModelInputs, action=None):
        logits = self.actor_train(inputs)
        logits = logits.squeeze(-1)
        probs = CategoricalMasked(logits=logits, masks=inputs.P_mask)
        if action is None:
            action = probs.sample()   
        return action, probs.log_prob(action), probs.entropy(), *self.get_value(inputs)

if __name__ == "__main__":

    # ------------------------------
    # ---- Command Line Parsing ----
    # ------------------------------

    cfg = tyro.cli(Config)

    # Set batch_size and minibatch_size based on cfg
    cfg.training.batch_size = int(cfg.training.num_envs * cfg.training.num_steps_per_update)
    cfg.training.minibatch_size = int(cfg.training.batch_size // cfg.training.num_minibatches)

    if _console_allows(cfg, "verbose"):
        pp.pprint(vars(cfg))
        print("\n\n\n")

    wandb_id = _generate_run_id()

    if (cfg.env.coeff_ring is not None) and (cfg.env.symbols is not None) and (cfg.env.basis_elements is not None):
        cfg.env.basis_elements = cfg.env.basis_elements.split(",")
        cfg.env.fix_ideal = True
        cfg.env.ideal_dist = None
        run_name = f"{cfg.env.gym_id}__fixed-ideal__{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}__{wandb_id}"
        _console_print(
            cfg,
            "Using fixed ideal with coefficient ring:",
            cfg.env.coeff_ring,
            "and symbols:",
            cfg.env.symbols,
            "and basis elements:",
            cfg.env.basis_elements,
        )
    else:
        run_name = f"{cfg.env.gym_id}__{cfg.env.ideal_dist}__{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}__{wandb_id}"
    if cfg.training.track:
        init_wandb_run(cfg, run_name, wandb_id)
    mirror_to_wandb = cfg.training.track
    if cfg.training.wandb_group is not None:
        if cfg.training.output_directory is not None:
            writer = make_summary_writer(
                f"{cfg.training.output_directory}/runs/{cfg.training.wandb_group}/{run_name}",
                mirror_to_wandb=mirror_to_wandb,
                wandb_module=wandb,
            )
        else:
            writer = make_summary_writer(
                f"runs/{cfg.training.wandb_group}/{run_name}",
                mirror_to_wandb=mirror_to_wandb,
                wandb_module=wandb,
            )
    else:
        if cfg.training.output_directory is not None:
            writer = make_summary_writer(
                f"{cfg.training.output_directory}/runs/{run_name}",
                mirror_to_wandb=mirror_to_wandb,
                wandb_module=wandb,
            )
        else:
            writer = make_summary_writer(
                f"runs/{run_name}",
                mirror_to_wandb=mirror_to_wandb,
                wandb_module=wandb,
            )
    print(f"Local run logs: {writer.log_dir}")
    print(f"Scalar logs: {writer.scalar_path} and {writer.scalar_csv_path}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(cfg).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    if cfg.training.pytorch_num_interop_threads is not None:
        if cfg.training.pytorch_num_interop_threads < 1:
            raise ValueError("cfg.training.pytorch_num_interop_threads must be >= 1")
        torch.set_num_interop_threads(cfg.training.pytorch_num_interop_threads)

    if cfg.training.pytorch_num_threads is not None:
        if cfg.training.pytorch_num_threads < 1:
            raise ValueError("cfg.training.pytorch_num_threads must be >= 1")
        torch.set_num_threads(cfg.training.pytorch_num_threads)

    random.seed(cfg.training.training_seed)
    np.random.seed(cfg.training.training_seed)
    torch.manual_seed(cfg.training.training_seed)
    torch.backends.cudnn.deterministic = cfg.training.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.training.cuda else "cpu")
    system_metrics_recorder = SystemMetricsRecorder(
        os.path.join(writer.log_dir, "system_metrics.jsonl"),
        device=device,
        enabled=cfg.training.system_metrics_enabled,
        sample_interval_seconds=cfg.training.system_metrics_interval_seconds,
    )

    env_kwargs = build_env_kwargs_for_gym_id(cfg, cfg.env.gym_id)
    ragged_nopq_mode = "nopq" in cfg.env.gym_id.lower()
    training_pq_budget = None
    if "nopq" not in cfg.env.gym_id.lower():
        training_pq_budget = int(cfg.env.N if cfg.env.N_P is None else cfg.env.N_P)

    if cfg.training.num_envs == 1:
        base_env = gym.make(cfg.env.gym_id, **env_kwargs)
        base_env = gym.wrappers.RecordEpisodeStatistics(base_env)
        envs = SingleEnvAdapter(base_env)
        _console_print(cfg, "Using non-vectorized environment path for num_envs=1")
    else:
        vector_kwargs = None
        if ragged_nopq_mode and cfg.training.vectorization_mode == "async":
            # AsyncVectorEnv cannot use shared-memory buffers for dynamic Sequence spaces.
            vector_kwargs = {"shared_memory": False}
        envs = gym.make_vec(
            cfg.env.gym_id,
            num_envs=cfg.training.num_envs,
            vectorization_mode=cfg.training.vectorization_mode,
            vector_kwargs=vector_kwargs,
            wrappers=[gym.wrappers.RecordEpisodeStatistics],
            **env_kwargs,
        )
    print_runtime_cpu_thread_info(envs, cfg)

    greedy_eval_plan = resolve_greedy_eval_targets(cfg)
    greedy_eval_enabled = greedy_eval_plan["enabled"]
    greedy_eval_run_train_env = greedy_eval_plan["run_train_env"]
    greedy_eval_run_nopq_env = greedy_eval_plan["run_nopq_env"]
    greedy_eval_primary_target = greedy_eval_plan["primary_target"]
    greedy_eval_nopq_gym_id = greedy_eval_plan["nopq_gym_id"]
    greedy_eval_gym_id = (
        cfg.env.gym_id
        if greedy_eval_primary_target == "train"
        else greedy_eval_nopq_gym_id
        if greedy_eval_primary_target == "nopq"
        else None
    )
    greedy_eval_targets = []
    if greedy_eval_run_train_env:
        greedy_eval_targets.append(
            {
                "key": "train",
                "label": "",
                "gym_id": cfg.env.gym_id,
                "is_primary": greedy_eval_primary_target == "train",
            }
        )
    if greedy_eval_run_nopq_env:
        greedy_eval_targets.append(
            {
                "key": "nopq",
                "label": " NoPQ",
                "gym_id": greedy_eval_nopq_gym_id,
                "is_primary": greedy_eval_primary_target == "nopq",
            }
        )
    greedy_eval_seeds = build_greedy_eval_seeds(cfg, greedy_eval_enabled)
    if greedy_eval_enabled:
        greedy_eval_corpus_rows, greedy_eval_corpus_path, greedy_eval_corpus_generated = resolve_greedy_eval_corpus(
            cfg,
            greedy_eval_seeds,
        )
    else:
        greedy_eval_corpus_rows = []
        greedy_eval_corpus_path = None
        greedy_eval_corpus_generated = False
    if greedy_eval_corpus_path is not None:
        if greedy_eval_corpus_generated:
            _console_print(cfg, f"Generated greedy eval corpus at {greedy_eval_corpus_path}")
        else:
            _console_print(cfg, f"Loaded greedy eval corpus from {greedy_eval_corpus_path}")
    next_greedy_eval_step = cfg.training.hparam_eval_interval_steps if greedy_eval_enabled else None
    last_greedy_eval = None
    last_greedy_eval_train = None
    last_greedy_eval_nopq = None
    last_greedy_eval_selection_metric = None
    best_greedy_eval_selection_metric = float("-inf")
    best_greedy_eval_step = None
    best_greedy_eval_nopq_mean = float("-inf")
    best_greedy_eval_nopq_step = None
    best_greedy_eval_checkpoint_paths = {}

    obs_sf_normalizer = None
    reward_normalizer = None
    value_normalizer = None
    if cfg.training.normalize_observation_state_features:
        obs_sf_normalizer = RunningMeanStd(
            shape=envs.single_observation_space["state_features"].shape,
            epsilon=1e-4,
            device=device,
        )
    if cfg.training.normalize_reward:
        reward_normalizer = RewardNormalizer(
            num_envs=cfg.training.num_envs,
            gamma=cfg.training.gamma,
            epsilon=cfg.training.reward_normalization_epsilon,
            clip_range=cfg.training.reward_normalization_clip,
            device=device,
        )
    # Once the environment has been created, the model config options which are determined by the environment can be set
    cfg.actor.state_embedding_head_in_dim = get_space_last_dim(envs.single_observation_space, "G")
    cfg.actor.state_features_embedding_head_in_dim = get_space_last_dim(envs.single_observation_space, "state_features")
    cfg.actor.pair_embedding_head_in_dim = get_space_last_dim(envs.single_observation_space, "P")

    cfg.critic.state_embedding_head_in_dim = get_space_last_dim(envs.single_observation_space, "G")
    cfg.critic.state_features_embedding_head_in_dim = get_space_last_dim(envs.single_observation_space, "state_features")

    agent = Agent(cfg).to(device)
    value_normalizer = build_value_normalizer(cfg, agent, device=device)
    if cfg.training.normalize_value_function and cfg.critic.critic_strategy == "heuristic-only":
        _console_print(
            cfg,
            "Skipping value normalization because "
            "cfg.critic.critic_strategy='heuristic-only' has no learned critic."
        )

    if cfg.training.share_actor_critic_trunk:
        _console_print(cfg, "Using shared actor/critic state trunk.")

    # For the policy network print out its architecture and number of parameters
    _console_print(cfg, "Actor Network:", level="verbose")
    if _console_allows(cfg, "verbose"):
        print(agent.actor)
    nb_param = 0
    for param in agent.actor.parameters():
        nb_param += np.prod(list(param.data.size()))
    _console_print(cfg, f"Actor parameters: {nb_param}")
    _console_print(cfg, "\n\n\n", level="verbose")

    # If there is a critic network, print out its architecture and number of parameters as well
    if cfg.critic.critic_strategy != "heuristic-only":
        _console_print(cfg, "Critic Network:", level="verbose")
        if _console_allows(cfg, "verbose"):
            print(agent.critic)
        nb_param = 0
        for param in agent.critic.parameters():
            nb_param += np.prod(list(param.data.size()))
        _console_print(cfg, f"Critic parameters: {nb_param}")
        _console_print(cfg, "\n\n\n", level="verbose")

    if cfg.training.enable_torch_compile:
        if ragged_nopq_mode:
            _console_print(cfg, "Skipping torch.compile because NoPQ observation mode has dynamic shapes.")
        elif not hasattr(torch, "compile"):
            _console_print(cfg, "Skipping torch.compile because this PyTorch build does not expose torch.compile.")
        else:
            compile_kwargs = {
                "backend": cfg.training.torch_compile_backend,
                "fullgraph": cfg.training.torch_compile_fullgraph,
                "dynamic": cfg.training.torch_compile_dynamic,
            }
            if cfg.training.torch_compile_mode != "default":
                compile_kwargs["mode"] = cfg.training.torch_compile_mode
            try:
                compiled_actor = torch.compile(agent.actor, **compile_kwargs)
                compiled_critic = None
                if cfg.critic.critic_strategy != "heuristic-only":
                    compiled_critic = torch.compile(agent.critic, **compile_kwargs)
                agent.configure_training_modules(compiled_actor, compiled_critic)
                _console_print(cfg, f"Enabled torch.compile for training forwards with kwargs={compile_kwargs}")
            except Exception as exc:
                _console_print(cfg, f"torch.compile setup failed ({exc}); falling back to eager mode.")

    optimizer = optim.Adam(agent.parameters(), lr=cfg.training.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs_space = envs.single_observation_space
    has_pair_idx_raw = hasattr(obs_space, "spaces") and ("P_idx_raw" in obs_space.spaces)
    has_pair_sugar = hasattr(obs_space, "spaces") and ("P_sugar" in obs_space.spaces)
    has_pair_normal_rank = hasattr(obs_space, "spaces") and ("P_normal_rank" in obs_space.spaces)
    if ragged_nopq_mode:
        obs_G = []
        obs_P = []
        obs_sf = []
        P_idxs = []
        P_idxs_raw = [] if has_pair_idx_raw else None
        G_poly_terms = []
        G_poly_term_masks = []
        P_sugars = [] if has_pair_sugar else None
        P_normal_ranks = [] if has_pair_normal_rank else None
        G_masks = []
        P_masks = []
    else:
        obs_G = make_buffer(obs_space["G"].shape)
        obs_P = make_buffer(obs_space["P"].shape)
        obs_sf = make_buffer(obs_space["state_features"].shape)
        P_idxs = make_buffer(obs_space["P_idx"].shape, dtype=torch.long)
        P_idxs_raw = make_buffer(obs_space["P_idx_raw"].shape, dtype=torch.long) if has_pair_idx_raw else None
        P_sugars = make_buffer(obs_space["P_sugar"].shape, dtype=torch.float32) if has_pair_sugar else None
        P_normal_ranks = make_buffer(obs_space["P_normal_rank"].shape, dtype=torch.float32) if has_pair_normal_rank else None
        G_masks = make_buffer(obs_space["G_mask"].shape, dtype=torch.bool)
        P_masks = make_buffer(obs_space["P_mask"].shape, dtype=torch.bool)
    actions = make_buffer(envs.single_action_space.shape)
    logprobs = make_buffer(())
    rewards = make_buffer(())
    dones = make_buffer(())
    values = make_buffer(())
    if cfg.critic.critic_strategy == "heuristic-plus-learned-residual":
        h_values = make_buffer(())

    # TRY NOT TO MODIFY: start the environment 
    global_step = 0
    episode_counter = 0
    max_episodic_return = float("-inf")
    min_episodic_return = float("inf")
    recent_returns = deque(maxlen=100)
    last_k_avg_return = float("nan")
    timed_out = False
    policy_updates = 0 # This counts each set of arg.num_epochs_per_update as one update
    start_time = time.time()
    if cfg.env.fix_ideal:
        # If we want to run training using a fixed ideal, we need to make sure that every parallel seed starts with the same seed
        next_obs, infos = envs.reset(seed=[cfg.env.env_seed] * cfg.training.num_envs)
    else:
        # If not then having parallel environments start with different seeds is beneficial
        next_obs, infos = envs.reset(seed=cfg.env.env_seed)
    use_zero_copy_obs_views = (device.type == "cpu")

    if ragged_nopq_mode:
        input = build_model_inputs_from_batched_obs(next_obs, device)
    elif use_zero_copy_obs_views:
        input = bind_obs_dict_as_model_inputs_views(next_obs)
    else:
        input = ModelInputs(
            P=torch.empty(size=(cfg.training.num_envs,) + obs_space["P"].shape, device=device, dtype=torch.float32),
            P_idx=torch.empty(size=(cfg.training.num_envs,) + obs_space["P_idx"].shape, device=device, dtype=torch.long),
            P_idx_raw=(
                torch.empty(size=(cfg.training.num_envs,) + obs_space["P_idx_raw"].shape, device=device, dtype=torch.long)
                if has_pair_idx_raw else None
            ),
            G=torch.empty(size=(cfg.training.num_envs,) + obs_space["G"].shape, device=device, dtype=torch.float32),
            G_mask=torch.empty(size=(cfg.training.num_envs,) + obs_space["G_mask"].shape, dtype=torch.bool, device=device),
            P_mask=torch.empty(size=(cfg.training.num_envs,) + obs_space["P_mask"].shape, dtype=torch.bool, device=device),
            P_sugar=(
                torch.empty(size=(cfg.training.num_envs,) + obs_space["P_sugar"].shape, device=device, dtype=torch.float32)
                if has_pair_sugar else None
            ),
            P_normal_rank=(
                torch.empty(size=(cfg.training.num_envs,) + obs_space["P_normal_rank"].shape, device=device, dtype=torch.float32)
                if has_pair_normal_rank else None
            ),
            state_features=torch.empty(size=(cfg.training.num_envs,) + obs_space["state_features"].shape, device=device, dtype=torch.float32),
        )
        copy_obs_dict_to_model_inputs(next_obs, input)
    if obs_sf_normalizer is not None:
        obs_sf_normalizer.update(input.state_features)
        input.state_features.copy_(
            obs_sf_normalizer.normalize(
                input.state_features,
                epsilon=cfg.training.observation_normalization_epsilon,
                clip_range=cfg.training.observation_normalization_clip,
            )
        )
    initial_system_metrics = system_metrics_recorder.maybe_record(
        global_step=global_step,
        elapsed_seconds=0.0,
        policy_updates=policy_updates,
        extra_metrics=summarize_state_feature_rollout(
            input.state_features,
            pq_budget=training_pq_budget,
        ),
        force=True,
    )
    if initial_system_metrics is not None:
        _log_system_metrics(writer, initial_system_metrics, global_step)
    next_done = torch.zeros(cfg.training.num_envs).to(device, dtype=torch.float32)
    num_updates = cfg.training.total_timesteps // cfg.training.batch_size
    
    # Create reusable ModelInputs objects to avoid repeated object creation
    if ragged_nopq_mode:
        minibatch_input = None
        diagnostics_input = None
    else:
        minibatch_input = ModelInputs(
            P=torch.empty(size=(cfg.training.minibatch_size,) + obs_space["P"].shape, device=device),
            P_idx=torch.empty(size=(cfg.training.minibatch_size,) + obs_space["P_idx"].shape, device=device),
            G=torch.empty(size=(cfg.training.minibatch_size,) + obs_space["G"].shape, device=device),
            G_mask=torch.empty(size=(cfg.training.minibatch_size,) + obs_space["G_mask"].shape, dtype=torch.bool, device=device), 
            P_mask=torch.empty(size=(cfg.training.minibatch_size,) + obs_space["P_mask"].shape, dtype=torch.bool, device=device), state_features=torch.empty(size=(cfg.training.minibatch_size,) + obs_space["state_features"].shape, device=device),
            h_value=torch.empty(size=(cfg.training.minibatch_size, 1), device=device) if cfg.critic.critic_strategy in ["heuristic-only", "heuristic-plus-learned-residual"] else None
        )
        diagnostics_input = ModelInputs(
            P=torch.empty(size=(cfg.training.minibatch_size,) + obs_space["P"].shape, device=device),
            P_idx=torch.empty(size=(cfg.training.minibatch_size,) + obs_space["P_idx"].shape, device=device),
            P_idx_raw=(
                torch.empty(size=(cfg.training.minibatch_size,) + obs_space["P_idx_raw"].shape, device=device, dtype=torch.long)
                if has_pair_idx_raw else None
            ),
            G=torch.empty(size=(cfg.training.minibatch_size,) + obs_space["G"].shape, device=device),
            G_mask=torch.empty(size=(cfg.training.minibatch_size,) + obs_space["G_mask"].shape, dtype=torch.bool, device=device), 
            P_mask=torch.empty(size=(cfg.training.minibatch_size,) + obs_space["P_mask"].shape, dtype=torch.bool, device=device), 
            P_sugar=torch.empty(size=(cfg.training.minibatch_size,) + obs_space["P_sugar"].shape, device=device) if has_pair_sugar else None,
            P_normal_rank=torch.empty(size=(cfg.training.minibatch_size,) + obs_space["P_normal_rank"].shape, device=device) if has_pair_normal_rank else None,
            state_features=torch.empty(size=(cfg.training.minibatch_size,) + obs_space["state_features"].shape, device=device)
        )

    for update in range(1, num_updates + 1):
        if cfg.training.max_runtime_seconds is not None:
            elapsed = time.time() - start_time
            if elapsed >= cfg.training.max_runtime_seconds:
                timed_out = True
                print(
                    f"Stopping early due to max runtime budget: elapsed={elapsed:.1f}s, limit={cfg.training.max_runtime_seconds}s"
                )
                break

        if ragged_nopq_mode:
            # Ragged rollout containers are append-only; clear them every PPO update.
            obs_G.clear()
            obs_P.clear()
            obs_sf.clear()
            P_idxs.clear()
            if P_idxs_raw is not None:
                P_idxs_raw.clear()
            G_poly_terms.clear()
            G_poly_term_masks.clear()
            G_masks.clear()
            P_masks.clear()
            if P_sugars is not None:
                P_sugars.clear()
            if P_normal_ranks is not None:
                P_normal_ranks.clear()

        # Annealing the rate if instructed to do so.
        if cfg.training.anneal_learning_rate:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * cfg.training.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, cfg.training.num_steps_per_update):
            global_step += 1 * cfg.training.num_envs

            if ragged_nopq_mode:
                # Keep flattened ordering consistent with tensor reshape(T, N) -> (T*N,).
                for env_i in range(cfg.training.num_envs):
                    obs_G.append(input.G[env_i].detach().clone())
                    obs_P.append(input.P[env_i].detach().clone())
                    obs_sf.append(input.state_features[env_i].detach().clone())
                    P_idxs.append(input.P_idx[env_i].detach().clone())
                    if P_idxs_raw is not None and input.P_idx_raw is not None:
                        P_idxs_raw.append(input.P_idx_raw[env_i].detach().clone())
                    if input.G_poly_terms is not None and input.G_poly_term_mask is not None:
                        G_poly_terms.append(input.G_poly_terms[env_i].detach().clone())
                        G_poly_term_masks.append(input.G_poly_term_mask[env_i].detach().clone())
                    if P_sugars is not None and input.P_sugar is not None:
                        P_sugars.append(input.P_sugar[env_i].detach().clone())
                    if P_normal_ranks is not None and input.P_normal_rank is not None:
                        P_normal_ranks.append(input.P_normal_rank[env_i].detach().clone())
                    G_masks.append(input.G_mask[env_i].detach().clone())
                    P_masks.append(input.P_mask[env_i].detach().clone())
            else:
                obs_G[step] = input.G
                obs_P[step] = input.P
                obs_sf[step] = input.state_features
                P_idxs[step] = input.P_idx
                if P_idxs_raw is not None and input.P_idx_raw is not None:
                    P_idxs_raw[step] = input.P_idx_raw
                if P_sugars is not None and input.P_sugar is not None:
                    P_sugars[step] = input.P_sugar
                if P_normal_ranks is not None and input.P_normal_rank is not None:
                    P_normal_ranks[step] = input.P_normal_rank
                G_masks[step] = input.G_mask
                P_masks[step] = input.P_mask
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, h_value = agent.get_action_and_value(input)
                values[step] = value.flatten()
                if cfg.critic.critic_strategy == "heuristic-plus-learned-residual":
                    h_values[step] = h_value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            term_tensor = torch.from_numpy(terminations).to(device=device, dtype=torch.bool)
            trunc_tensor = torch.from_numpy(truncations).to(device=device, dtype=torch.bool)
            done_tensor = torch.logical_or(term_tensor, trunc_tensor)
            done_float = done_tensor.to(dtype=torch.float32)
            next_done.copy_(done_float)

            reward_tensor = torch.from_numpy(reward).to(device=device, dtype=torch.float32)
            if reward_normalizer is not None:
                reward_tensor = reward_normalizer.normalize(reward_tensor, done_float)
            rewards[step].copy_(reward_tensor)

            if ragged_nopq_mode:
                input = build_model_inputs_from_batched_obs(next_obs, device)
            elif use_zero_copy_obs_views:
                input = bind_obs_dict_as_model_inputs_views(next_obs)
            else:
                copy_obs_dict_to_model_inputs(next_obs, input)

            state_features_tensor = input.state_features
            if obs_sf_normalizer is not None:
                obs_sf_normalizer.update(state_features_tensor)
                state_features_tensor = obs_sf_normalizer.normalize(
                    state_features_tensor,
                    epsilon=cfg.training.observation_normalization_epsilon,
                    clip_range=cfg.training.observation_normalization_clip,
                )
                input.state_features.copy_(state_features_tensor)

            if "_episode" in infos:
                for i, entry in enumerate(infos["_episode"]):
                    if entry:
                        episode_counter += 1
                        episodic_return = infos["episode"]["r"][i].item()
                        recent_returns.append(episodic_return)
                        last_k_avg_return = float(np.mean(recent_returns))
                        max_episodic_return = max(max_episodic_return, episodic_return)
                        min_episodic_return = min(min_episodic_return, episodic_return)
                        writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                        writer.add_scalar("charts/max_episodic_return", max_episodic_return, global_step)
                        writer.add_scalar("charts/min_episodic_return", min_episodic_return, global_step)
                        writer.add_scalar("charts/episodic_return_last_100_avg", last_k_avg_return, global_step)
                        writer.add_scalar("charts/episodic_length", infos["episode"]["l"][i].item(), global_step)
                        writer.add_scalar("charts/episodes_completed", episode_counter, global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value, _ = agent.get_value(input)
            next_value = next_value.reshape(1, -1)
            if cfg.training.use_gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(cfg.training.num_steps_per_update)):
                    if t == cfg.training.num_steps_per_update - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + cfg.training.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + cfg.training.gamma * cfg.training.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(cfg.training.num_steps_per_update)):
                    if t == cfg.training.num_steps_per_update - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + cfg.training.gamma * nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        if ragged_nopq_mode:
            b_obs_G = obs_G
            b_obs_P = obs_P
            b_P_idxs = P_idxs
            b_P_idxs_raw = P_idxs_raw
            b_obs_sf = obs_sf
            b_G_poly_terms = G_poly_terms if len(G_poly_terms) > 0 else None
            b_G_poly_term_masks = G_poly_term_masks if len(G_poly_term_masks) > 0 else None
            b_P_sugars = P_sugars
            b_P_normal_ranks = P_normal_ranks
            b_G_masks = G_masks
            b_P_masks = P_masks
        else:
            b_obs_G = obs_G.reshape((-1,) + envs.single_observation_space["G"].shape)
            b_obs_P = obs_P.reshape((-1,) + envs.single_observation_space["P"].shape)
            b_P_idxs = P_idxs.reshape((-1,) + envs.single_observation_space["P_idx"].shape)
            b_P_idxs_raw = P_idxs_raw.reshape((-1,) + envs.single_observation_space["P_idx_raw"].shape) if P_idxs_raw is not None else None
            b_obs_sf = obs_sf.reshape((-1,) + envs.single_observation_space["state_features"].shape)
            b_P_sugars = P_sugars.reshape((-1,) + envs.single_observation_space["P_sugar"].shape) if P_sugars is not None else None
            b_P_normal_ranks = P_normal_ranks.reshape((-1,) + envs.single_observation_space["P_normal_rank"].shape) if P_normal_ranks is not None else None
            b_G_poly_terms = None
            b_G_poly_term_masks = None
            b_G_masks = G_masks.reshape((-1, G_masks.shape[-1]))
            b_P_masks = P_masks.reshape((-1, P_masks.shape[-1]))
        rollout_system_metrics = summarize_state_feature_rollout(
            b_obs_sf,
            pq_budget=training_pq_budget,
        )
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        if cfg.critic.critic_strategy == "heuristic-plus-learned-residual":
            b_h_values = h_values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(cfg.training.batch_size)
        clipfracs = []
        v_clipfracs = []
        entropy_raw_mean = torch.tensor(0.0, device=device)
        entropy_normalized_mean = torch.tensor(0.0, device=device)
        entropy_bonus_mean = torch.tensor(0.0, device=device)
        advantages_means = []
        advantages_stds = []
        for epoch in range(cfg.training.num_epochs_per_update):
            np.random.shuffle(b_inds)
            for start in range(0, cfg.training.batch_size, cfg.training.minibatch_size):
                end = start + cfg.training.minibatch_size
                mb_inds = b_inds[start:end]

                if ragged_nopq_mode:
                    minibatch_input = collate_ragged_model_inputs(
                        indices=mb_inds,
                        obs_G_list=b_obs_G,
                        obs_P_list=b_obs_P,
                        p_idx_list=b_P_idxs,
                        p_idx_raw_list=b_P_idxs_raw,
                        g_mask_list=b_G_masks,
                        p_mask_list=b_P_masks,
                        obs_sf_list=b_obs_sf,
                        device=device,
                        p_sugar_list=b_P_sugars,
                        p_normal_rank_list=b_P_normal_ranks,
                        g_poly_terms_list=b_G_poly_terms,
                        g_poly_term_mask_list=b_G_poly_term_masks,
                    )
                    if cfg.critic.critic_strategy == "heuristic-plus-learned-residual":
                        minibatch_input.h_value = b_h_values[mb_inds].unsqueeze(-1)
                else:
                    # Update reusable ModelInputs object instead of creating new one
                    minibatch_input.G.copy_(b_obs_G[mb_inds])
                    minibatch_input.P.copy_(b_obs_P[mb_inds])
                    minibatch_input.P_idx.copy_(b_P_idxs[mb_inds])
                    minibatch_input.G_mask.copy_(b_G_masks[mb_inds])
                    minibatch_input.P_mask.copy_(b_P_masks[mb_inds])
                    minibatch_input.state_features.copy_(b_obs_sf[mb_inds])
                    if cfg.critic.critic_strategy == "heuristic-plus-learned-residual":
                        minibatch_input.h_value.copy_(b_h_values[mb_inds].unsqueeze(-1))

                mb_value_targets = None
                if value_normalizer is not None and cfg.critic.critic_strategy != "heuristic-only":
                    if cfg.critic.critic_strategy == "heuristic-plus-learned-residual":
                        mb_value_targets = b_returns[mb_inds] - b_h_values[mb_inds]
                    else:
                        mb_value_targets = b_returns[mb_inds]
                    with torch.no_grad():
                        value_normalizer.update(mb_value_targets.detach())

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    minibatch_input,
                    action=b_actions[mb_inds].to(dtype=torch.long)
                )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > cfg.training.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if cfg.training.normalize_advantage:
                    mb_advantages_mean = mb_advantages.mean()
                    advantages_means.append(float(mb_advantages_mean.detach().item()))
                    mb_advantages_std = mb_advantages.std()
                    advantages_stds.append(float(mb_advantages_std.detach().item()))
                    mb_advantages = (mb_advantages - mb_advantages_mean) / (mb_advantages_std + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.training.clip_coef, 1 + cfg.training.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                if cfg.critic.critic_strategy != "heuristic-only":
                    newvalue = newvalue.view(-1)
                    if value_normalizer is not None:
                        if cfg.critic.critic_strategy == "heuristic-plus-learned-residual":
                            learned_target = b_returns[mb_inds] - b_h_values[mb_inds]
                            learned_newvalue = newvalue - b_h_values[mb_inds]
                            learned_oldvalue = b_values[mb_inds] - b_h_values[mb_inds]
                        else:
                            learned_target = b_returns[mb_inds]
                            learned_newvalue = newvalue
                            learned_oldvalue = b_values[mb_inds]
                        normalized_newvalue = value_normalizer.normalize(
                            learned_newvalue,
                            clip_range=cfg.training.value_normalization_clip,
                        )
                        normalized_returns = value_normalizer.normalize(
                            learned_target,
                            clip_range=cfg.training.value_normalization_clip,
                        )
                    else:
                        normalized_newvalue = newvalue
                        normalized_returns = b_returns[mb_inds]

                    if cfg.training.clip_critic_loss:
                        if value_normalizer is not None:
                            normalized_oldvalue = value_normalizer.normalize(
                                learned_oldvalue,
                                clip_range=cfg.training.value_normalization_clip,
                            )
                            v_delta = normalized_newvalue - normalized_oldvalue
                            abs_delta = v_delta.abs()
                            v_clipfrac = (abs_delta > cfg.training.critic_clip_coef).float().mean().item()
                            v_clipfracs += [v_clipfrac]
                            v_loss_unclipped = (normalized_newvalue - normalized_returns) ** 2
                            v_clipped = normalized_oldvalue + torch.clamp(
                                v_delta,
                                -cfg.training.critic_clip_coef,
                                cfg.training.critic_clip_coef,
                            )
                            v_loss_clipped = (v_clipped - normalized_returns) ** 2
                        else:
                            v_delta = newvalue - b_values[mb_inds]
                            abs_delta = v_delta.abs()
                            v_clipfrac = (abs_delta > cfg.training.critic_clip_coef).float().mean().item()
                            v_clipfracs += [v_clipfrac]
                            v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                            v_clipped = b_values[mb_inds] + torch.clamp(
                                v_delta,
                                -cfg.training.critic_clip_coef,
                                cfg.training.critic_clip_coef,
                            )
                            v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((normalized_newvalue - normalized_returns) ** 2).mean()
                else:
                    v_loss = torch.tensor(0.0).to(device)

                entropy_raw_mean = entropy.mean()
                entropy_normalized = normalized_masked_entropy(entropy, minibatch_input.P_mask)
                entropy_normalized_mean = entropy_normalized.mean()
                entropy_bonus_mean = (
                    entropy_normalized_mean
                    if cfg.training.normalize_masked_action_entropy
                    else entropy_raw_mean
                )

                loss = (
                    pg_loss
                    - cfg.training.entropy_coefficient * entropy_bonus_mean
                    + v_loss * cfg.training.critic_loss_coefficient
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if agent.shared_actor_critic_trunk is not None:
                    actor_grad_norm = global_grad_norm(agent.actor.parameters())
                    if cfg.critic.critic_strategy != "heuristic-only":
                        critic_grad_norm = global_grad_norm(agent.critic.parameters())

                    if cfg.training.clip_grad_norm_actor and len(agent.actor_head_parameters) > 0:
                        nn.utils.clip_grad_norm_(
                            agent.actor_head_parameters,
                            cfg.training.max_grad_norm_actor,
                        )

                    if cfg.critic.critic_strategy != "heuristic-only":
                        if cfg.training.clip_grad_norm_critic and len(agent.critic_head_parameters) > 0:
                            nn.utils.clip_grad_norm_(
                                agent.critic_head_parameters,
                                cfg.training.max_grad_norm_critic,
                            )

                        shared_clip_limits = []
                        if cfg.training.clip_grad_norm_actor:
                            shared_clip_limits.append(cfg.training.max_grad_norm_actor)
                        if cfg.training.clip_grad_norm_critic:
                            shared_clip_limits.append(cfg.training.max_grad_norm_critic)
                        if shared_clip_limits and len(agent.shared_trunk_parameters) > 0:
                            nn.utils.clip_grad_norm_(
                                agent.shared_trunk_parameters,
                                min(shared_clip_limits),
                            )
                else:
                    if cfg.training.clip_grad_norm_actor:
                        actor_grad_norm = nn.utils.clip_grad_norm_(agent.actor.parameters(), cfg.training.max_grad_norm_actor)
                    else:
                        actor_grad_norm = global_grad_norm(agent.actor.parameters())
                    if cfg.critic.critic_strategy != "heuristic-only":
                        if cfg.training.clip_grad_norm_critic:
                            critic_grad_norm = nn.utils.clip_grad_norm_(agent.critic.parameters(), cfg.training.max_grad_norm_critic)
                        else:
                            critic_grad_norm = global_grad_norm(agent.critic.parameters())
                optimizer.step()

            if cfg.training.target_kl is not None:
                if approx_kl > cfg.training.target_kl:
                    break

        pg_loss_scalar = _scalarize_metric(pg_loss)
        entropy_raw_mean_scalar = _scalarize_metric(entropy_raw_mean)
        entropy_normalized_mean_scalar = _scalarize_metric(entropy_normalized_mean)
        entropy_bonus_mean_scalar = _scalarize_metric(entropy_bonus_mean)
        old_approx_kl_scalar = _scalarize_metric(old_approx_kl)
        approx_kl_scalar = _scalarize_metric(approx_kl)
        actor_grad_norm_scalar = _scalarize_metric(actor_grad_norm)
        critic_grad_norm_scalar = (
            _scalarize_metric(critic_grad_norm)
            if cfg.critic.critic_strategy != "heuristic-only"
            else None
        )
        v_loss_scalar = (
            _scalarize_metric(v_loss)
            if cfg.critic.critic_strategy != "heuristic-only"
            else 0.0
        )

        # Drop references to the final minibatch autograd graph before diagnostics.
        del loss, pg_loss, entropy_raw_mean, entropy_normalized_mean, entropy_bonus_mean
        del newlogprob, entropy, newvalue, logratio, ratio, mb_advantages
        if cfg.critic.critic_strategy != "heuristic-only":
            del v_loss

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Section for calculating counterfactual and gradient based diagnostics

        if ragged_nopq_mode:
            diagnostics_input = collate_ragged_model_inputs(
                indices=mb_inds,
                obs_G_list=b_obs_G,
                obs_P_list=b_obs_P,
                p_idx_list=b_P_idxs,
                p_idx_raw_list=b_P_idxs_raw,
                g_mask_list=b_G_masks,
                p_mask_list=b_P_masks,
                obs_sf_list=b_obs_sf,
                device=device,
                p_sugar_list=b_P_sugars,
                p_normal_rank_list=b_P_normal_ranks,
                g_poly_terms_list=b_G_poly_terms,
                g_poly_term_mask_list=b_G_poly_term_masks,
            )
        else:
            # Update reusable ModelInputs object for diagnostics
            diagnostics_input.G.copy_(b_obs_G[mb_inds])
            diagnostics_input.P.copy_(b_obs_P[mb_inds])
            diagnostics_input.P_idx.copy_(b_P_idxs[mb_inds])
            if diagnostics_input.P_idx_raw is not None and b_P_idxs_raw is not None:
                diagnostics_input.P_idx_raw.copy_(b_P_idxs_raw[mb_inds])
            diagnostics_input.G_mask.copy_(b_G_masks[mb_inds])
            diagnostics_input.P_mask.copy_(b_P_masks[mb_inds])
            if diagnostics_input.P_sugar is not None and b_P_sugars is not None:
                diagnostics_input.P_sugar.copy_(b_P_sugars[mb_inds])
            if diagnostics_input.P_normal_rank is not None and b_P_normal_ranks is not None:
                diagnostics_input.P_normal_rank.copy_(b_P_normal_ranks[mb_inds])
            diagnostics_input.state_features.copy_(b_obs_sf[mb_inds])

        if cfg.actor.pair_embedding_head == "equivariant-set-transformer" or cfg.actor.state_embedding_head == "invariant-set-transformer" or cfg.actor.state_embedding_head == "gnn":
            return_attention = True
        else:
            return_attention = False

        if cfg.actor.state_embedding_head is None and cfg.actor.state_features_embedding_head is None:
            pass
        else:
            run_actor_global_diagnostics(
                actor=agent.actor,
                input=diagnostics_input,
                writer=writer,
                update_idx=policy_updates,
                global_step=global_step,
                diag_every=cfg.training.counterfactual_diagnostic_frequency,
                grad_every=cfg.training.gradient_sensitivity_diagnostic_frequency,
                return_attention=return_attention
            )

            agent.actor.zero_grad(set_to_none=True)

        run_actor_ranking_diagnostics(
            actor=agent.actor,
            input=diagnostics_input,
            writer=writer,
            update_idx=policy_updates,
            global_step=global_step,
            ranking_every=cfg.training.ranking_diagnostic_frequency,
        )

           
        # Periodic GPU memory cleanup to prevent fragmentation
        if update % 50 == 0 and torch.cuda.is_available():  # Every 50 updates
            torch.cuda.empty_cache()


        # TRY NOT TO MODIFY: record rewards for plotting purposes
        policy_updates += 1
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/policy_updates", policy_updates, global_step)
        if cfg.critic.critic_strategy != "heuristic-only":
            writer.add_scalar("losses/critic_loss", v_loss_scalar, global_step)
            writer.add_scalar("losses/c1*critic_loss", cfg.training.critic_loss_coefficient * v_loss_scalar, global_step)
            writer.add_scalar("losses/critic_clipfrac", np.mean(v_clipfracs), global_step)
            writer.add_scalar("losses/pre_clip_critic_grad_norm", critic_grad_norm_scalar, global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/policy_loss", pg_loss_scalar, global_step)
        writer.add_scalar("losses/entropy", entropy_bonus_mean_scalar, global_step)
        writer.add_scalar("losses/c2*entropy", cfg.training.entropy_coefficient * entropy_bonus_mean_scalar, global_step)
        writer.add_scalar("losses/entropy_raw", entropy_raw_mean_scalar, global_step)
        writer.add_scalar("losses/entropy_normalized", entropy_normalized_mean_scalar, global_step)
        writer.add_scalar("losses/entropy_bonus_used", entropy_bonus_mean_scalar, global_step)
        writer.add_scalar("losses/c2*entropy_bonus_used", cfg.training.entropy_coefficient * entropy_bonus_mean_scalar, global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl_scalar, global_step)
        writer.add_scalar("losses/approx_kl", approx_kl_scalar, global_step)
        advantage_mean_scalar = (
            sum(advantages_means) / len(advantages_means)
            if advantages_means
            else float("nan")
        )
        advantage_std_scalar = (
            (sum(s**2 for s in advantages_stds) / len(advantages_stds))**0.5
            if advantages_stds
            else float("nan")
        )
        writer.add_scalar("losses/advantage_mean", advantage_mean_scalar, global_step)
        writer.add_scalar("losses/advantage_std", advantage_std_scalar, global_step)
        writer.add_scalar("losses/pre_clip_actor_grad_norm", actor_grad_norm_scalar, global_step)
        writer.add_scalar("losses/policy_clipfrac", np.mean(clipfracs), global_step)

        # Normalization diagnostics
        if obs_sf_normalizer is not None:
            obs_sf_std = torch.sqrt(obs_sf_normalizer.var + cfg.training.observation_normalization_epsilon)
            writer.add_scalar("normalization/state_features/running_mean", _tensor_stat_to_float(obs_sf_normalizer.mean), global_step)
            writer.add_scalar("normalization/state_features/running_std", _tensor_stat_to_float(obs_sf_std), global_step)
            writer.add_scalar("normalization/state_features/running_std_max", obs_sf_std.detach().max().item(), global_step)
            writer.add_scalar("normalization/state_features/running_std_min", obs_sf_std.detach().min().item(), global_step)

        if reward_normalizer is not None:
            reward_std = torch.sqrt(reward_normalizer.rms.var + cfg.training.reward_normalization_epsilon)
            writer.add_scalar("normalization/reward/running_return_mean", _tensor_stat_to_float(reward_normalizer.rms.mean), global_step)
            writer.add_scalar("normalization/reward/running_return_std", _tensor_stat_to_float(reward_std), global_step)

        if value_normalizer is not None:
            value_std = torch.sqrt(value_normalizer.var + cfg.training.value_normalization_epsilon)
            writer.add_scalar("normalization/value/running_target_mean", _tensor_stat_to_float(value_normalizer.mean), global_step)
            writer.add_scalar("normalization/value/running_target_std", _tensor_stat_to_float(value_std), global_step)

        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)

        system_metrics_record = system_metrics_recorder.maybe_record(
            global_step=global_step,
            elapsed_seconds=time.time() - start_time,
            policy_updates=policy_updates,
            extra_metrics=rollout_system_metrics,
        )
        if system_metrics_record is not None:
            _log_system_metrics(writer, system_metrics_record, global_step)
        _print_update_summary(
            cfg=cfg,
            update_idx=update,
            num_updates=num_updates,
            global_step=global_step,
            sps=sps,
            recent_returns=recent_returns,
            last_k_avg_return=last_k_avg_return,
            policy_loss=pg_loss_scalar,
            entropy=entropy_bonus_mean_scalar,
            approx_kl=approx_kl_scalar,
            critic_loss=v_loss_scalar if cfg.critic.critic_strategy != "heuristic-only" else None,
        )

        if greedy_eval_enabled and next_greedy_eval_step is not None and global_step >= next_greedy_eval_step:
            greedy_eval_round_stats = {}
            for greedy_eval_target in greedy_eval_targets:
                benchmark_episode_results_by_seed = None
                if (
                    greedy_eval_target["key"] == "nopq"
                    and greedy_eval_run_train_env
                    and "train" in greedy_eval_round_stats
                ):
                    benchmark_episode_results_by_seed = {
                        int(result["episode_seed"]): result
                        for result in greedy_eval_round_stats["train"].get("episode_results", [])
                    }
                _console_print(
                    cfg,
                    f"Starting greedy eval{greedy_eval_target['label']} @ step {global_step}: "
                    f"examples={len(greedy_eval_corpus_rows)}",
                    flush=True,
                )
                greedy_eval_stats = run_fixed_greedy_eval(
                    actor=agent.actor,
                    eval_gym_id=greedy_eval_target["gym_id"],
                    eval_corpus_rows=greedy_eval_corpus_rows,
                    device=device,
                    obs_sf_normalizer=obs_sf_normalizer,
                    cfg=cfg,
                    benchmark_episode_results_by_seed=benchmark_episode_results_by_seed,
                )
                greedy_eval_round_stats[greedy_eval_target["key"]] = greedy_eval_stats

                if greedy_eval_target["key"] == "train":
                    last_greedy_eval_train = greedy_eval_stats
                elif greedy_eval_target["key"] == "nopq":
                    last_greedy_eval_nopq = greedy_eval_stats
                    if (
                        greedy_eval_stats["num_truncated_total"] == 0
                        and greedy_eval_stats["mean_return_completed_only"] is not None
                        and greedy_eval_stats["mean_return_completed_only"] > best_greedy_eval_nopq_mean
                    ):
                        best_greedy_eval_nopq_mean = greedy_eval_stats["mean_return_completed_only"]
                        best_greedy_eval_nopq_step = global_step

                    writer.add_scalar("charts/greedy_eval_nopq_mean_return", greedy_eval_stats["mean_return"], global_step)
                    writer.add_scalar("charts/greedy_eval_nopq_median_return", greedy_eval_stats["median_return"], global_step)
                    writer.add_scalar("charts/greedy_eval_nopq_min_return", greedy_eval_stats["min_return"], global_step)
                    writer.add_scalar("charts/greedy_eval_nopq_max_return", greedy_eval_stats["max_return"], global_step)
                    writer.add_scalar("charts/greedy_eval_nopq_mean_length", greedy_eval_stats["mean_length"], global_step)
                    writer.add_scalar("charts/greedy_eval_nopq_max_length", greedy_eval_stats["max_length"], global_step)
                    writer.add_scalar("charts/greedy_eval_nopq_num_completed", greedy_eval_stats["num_completed"], global_step)
                    writer.add_scalar("charts/greedy_eval_nopq_completion_rate", greedy_eval_stats["completion_rate"], global_step)
                    writer.add_scalar("charts/greedy_eval_nopq_num_truncated", greedy_eval_stats["num_truncated_total"], global_step)
                    writer.add_scalar("charts/greedy_eval_nopq_num_benchmark_truncated", greedy_eval_stats["num_benchmark_truncated"], global_step)
                    writer.add_scalar("charts/greedy_eval_nopq_num_timed_out", greedy_eval_stats["num_timed_out"], global_step)
                    writer.add_scalar("charts/greedy_eval_nopq_num_worker_exited", greedy_eval_stats["num_worker_exited"], global_step)
                    if greedy_eval_stats["mean_return_completed_only"] is not None:
                        writer.add_scalar(
                            "charts/greedy_eval_nopq_mean_return_completed_only",
                            greedy_eval_stats["mean_return_completed_only"],
                            global_step,
                        )

                if greedy_eval_target["is_primary"]:
                    last_greedy_eval = greedy_eval_stats
                    writer.add_scalar("charts/greedy_eval_mean_return", greedy_eval_stats["mean_return"], global_step)
                    writer.add_scalar("charts/greedy_eval_median_return", greedy_eval_stats["median_return"], global_step)
                    writer.add_scalar("charts/greedy_eval_min_return", greedy_eval_stats["min_return"], global_step)
                    writer.add_scalar("charts/greedy_eval_max_return", greedy_eval_stats["max_return"], global_step)
                    writer.add_scalar("charts/greedy_eval_mean_length", greedy_eval_stats["mean_length"], global_step)
                    writer.add_scalar("charts/greedy_eval_max_length", greedy_eval_stats["max_length"], global_step)
                    writer.add_scalar("charts/greedy_eval_num_completed", greedy_eval_stats["num_completed"], global_step)
                    writer.add_scalar("charts/greedy_eval_completion_rate", greedy_eval_stats["completion_rate"], global_step)
                    writer.add_scalar("charts/greedy_eval_num_truncated", greedy_eval_stats["num_truncated_total"], global_step)
                    writer.add_scalar("charts/greedy_eval_num_timed_out", greedy_eval_stats["num_timed_out"], global_step)
                    writer.add_scalar("charts/greedy_eval_num_worker_exited", greedy_eval_stats["num_worker_exited"], global_step)
                    if greedy_eval_stats["mean_return_completed_only"] is not None:
                        writer.add_scalar(
                            "charts/greedy_eval_mean_return_completed_only",
                            greedy_eval_stats["mean_return_completed_only"],
                            global_step,
                        )

                _console_print(
                    cfg,
                    f"Greedy eval{greedy_eval_target['label']} @ step {global_step}: "
                    f"mean={greedy_eval_stats['mean_return']:.4f}, "
                    f"median={greedy_eval_stats['median_return']:.4f}, "
                    f"examples={greedy_eval_stats['num_examples']}, "
                    f"completed={greedy_eval_stats['num_completed']}, "
                    f"truncated={greedy_eval_stats['num_truncated_total']}, "
                    f"timed_out={greedy_eval_stats['num_timed_out']}, "
                    f"worker_exited={greedy_eval_stats['num_worker_exited']}, "
                    f"elapsed={greedy_eval_stats['elapsed_seconds']:.1f}s"
                )

            last_greedy_eval = (
                greedy_eval_round_stats.get(greedy_eval_primary_target)
                if greedy_eval_primary_target is not None
                else None
            )
            selection_metric, selection_metric_kind = compute_greedy_eval_selection_metric(
                greedy_eval_round_stats.get("train"),
                greedy_eval_round_stats.get("nopq"),
                run_train_env=greedy_eval_run_train_env,
                run_nopq_env=greedy_eval_run_nopq_env,
            )
            last_greedy_eval_selection_metric = selection_metric
            if selection_metric is not None:
                writer.add_scalar("charts/greedy_eval_selection_metric", selection_metric, global_step)
            if greedy_eval_run_train_env and greedy_eval_run_nopq_env:
                train_stats = greedy_eval_round_stats.get("train")
                nopq_stats = greedy_eval_round_stats.get("nopq")
                if train_stats is not None and nopq_stats is not None:
                    writer.add_scalar(
                        "charts/greedy_eval_nopq_minus_train_mean_return",
                        float(nopq_stats["mean_return"] - train_stats["mean_return"]),
                        global_step,
                    )
                    writer.add_scalar(
                        "charts/greedy_eval_nopq_num_benchmark_truncated",
                        int(nopq_stats["num_benchmark_truncated"]),
                        global_step,
                    )

            if selection_metric is not None and selection_metric > best_greedy_eval_selection_metric:
                best_greedy_eval_selection_metric = selection_metric
                best_greedy_eval_step = global_step
                best_greedy_eval_checkpoint_paths = save_model_checkpoints(
                    cfg,
                    agent,
                    run_name,
                    checkpoint_suffix="best_greedy_eval",
                )
                for checkpoint_kind, checkpoint_path in sorted(best_greedy_eval_checkpoint_paths.items()):
                    _console_print(
                        cfg,
                        f"Saved best greedy eval {checkpoint_kind} checkpoint "
                        f"@ step {global_step} using {selection_metric_kind}={selection_metric:.4f} "
                        f"to {checkpoint_path}"
                    )

            if selection_metric is not None:
                _console_print(
                    cfg,
                    f"Greedy eval selection metric @ step {global_step}: "
                    f"{selection_metric_kind}={selection_metric:.4f}",
                    flush=True,
                )
            elif selection_metric_kind is not None:
                _console_print(
                    cfg,
                    f"Greedy eval selection metric @ step {global_step} skipped: "
                    f"{selection_metric_kind}",
                    flush=True,
                )

            while next_greedy_eval_step is not None and next_greedy_eval_step <= global_step:
                next_greedy_eval_step += cfg.training.hparam_eval_interval_steps

    # Saving the final model weights to file
    final_checkpoint_paths = save_model_checkpoints(cfg, agent, run_name)
    for checkpoint_kind, checkpoint_path in sorted(final_checkpoint_paths.items()):
        print(f"Final {checkpoint_kind} checkpoint saved to {checkpoint_path}")

    if cfg.training.hparam_metrics_out is not None:
        os.makedirs(os.path.dirname(cfg.training.hparam_metrics_out) or ".", exist_ok=True)
        greedy_metric = None if last_greedy_eval is None else float(last_greedy_eval["mean_return"])
        greedy_metric_completed_only = (
            None
            if last_greedy_eval is None or last_greedy_eval["mean_return_completed_only"] is None
            else float(last_greedy_eval["mean_return_completed_only"])
        )
        selection_metric_kind = compute_greedy_eval_selection_metric(
            last_greedy_eval_train,
            last_greedy_eval_nopq,
            run_train_env=greedy_eval_run_train_env,
            run_nopq_env=greedy_eval_run_nopq_env,
        )[1]
        summary = {
            "run_name": run_name,
            "global_step": int(global_step),
            "policy_updates": int(policy_updates),
            "episodes_completed": int(episode_counter),
            "max_episodic_return": None if episode_counter == 0 else float(max_episodic_return),
            "last_100_avg_return": None if len(recent_returns) == 0 else float(last_k_avg_return),
            "greedy_eval_last_mean_return": greedy_metric,
            "greedy_eval_last_mean_return_completed_only": greedy_metric_completed_only,
            "greedy_eval_last_median_return": None if last_greedy_eval is None else float(last_greedy_eval["median_return"]),
            "greedy_eval_last_min_return": None if last_greedy_eval is None else float(last_greedy_eval["min_return"]),
            "greedy_eval_last_max_return": None if last_greedy_eval is None else float(last_greedy_eval["max_return"]),
            "greedy_eval_last_mean_length": None if last_greedy_eval is None else float(last_greedy_eval["mean_length"]),
            "greedy_eval_last_max_length": None if last_greedy_eval is None else int(last_greedy_eval["max_length"]),
            "greedy_eval_last_num_completed": None if last_greedy_eval is None else int(last_greedy_eval["num_completed"]),
            "greedy_eval_last_completion_rate": None if last_greedy_eval is None else float(last_greedy_eval["completion_rate"]),
            "greedy_eval_last_num_truncated": None if last_greedy_eval is None else int(last_greedy_eval["num_truncated_total"]),
            "greedy_eval_last_num_timed_out": None if last_greedy_eval is None else int(last_greedy_eval["num_timed_out"]),
            "greedy_eval_last_num_worker_exited": None if last_greedy_eval is None else int(last_greedy_eval["num_worker_exited"]),
            "greedy_eval_num_examples": int(len(greedy_eval_corpus_rows)) if greedy_eval_enabled else 0,
            "greedy_eval_interval_steps": int(cfg.training.hparam_eval_interval_steps) if greedy_eval_enabled else 0,
            "greedy_eval_seed": int(cfg.training.hparam_eval_seed) if greedy_eval_enabled else None,
            "greedy_eval_corpus_path": greedy_eval_corpus_path,
            "greedy_eval_last_selection_metric": None if last_greedy_eval_selection_metric is None else float(last_greedy_eval_selection_metric),
            "greedy_eval_selection_metric_kind": selection_metric_kind,
            "greedy_eval_best_selection_metric": None if best_greedy_eval_step is None else float(best_greedy_eval_selection_metric),
            "greedy_eval_best_mean_return": (
                None
                if best_greedy_eval_step is None
                else float(best_greedy_eval_nopq_mean)
                if greedy_eval_primary_target == "nopq" and best_greedy_eval_nopq_step is not None
                else float(best_greedy_eval_selection_metric)
            ),
            "greedy_eval_best_step": None if best_greedy_eval_step is None else int(best_greedy_eval_step),
            "greedy_eval_best_actor_path": best_greedy_eval_checkpoint_paths.get("actor"),
            "greedy_eval_best_critic_path": best_greedy_eval_checkpoint_paths.get("critic"),
            "greedy_eval_gym_id": greedy_eval_gym_id if greedy_eval_enabled else None,
            "greedy_eval_compare_nopq": bool(greedy_eval_run_nopq_env),
            "greedy_eval_train_is_benchmark": bool(greedy_eval_run_train_env and greedy_eval_run_nopq_env),
            "greedy_eval_nopq_uses_train_benchmark": bool(greedy_eval_run_train_env and greedy_eval_run_nopq_env),
            "greedy_eval_nopq_gym_id": greedy_eval_nopq_gym_id if greedy_eval_run_nopq_env else None,
            "greedy_eval_nopq_last_mean_return": None if last_greedy_eval_nopq is None else float(last_greedy_eval_nopq["mean_return"]),
            "greedy_eval_nopq_last_mean_return_completed_only": (
                None
                if last_greedy_eval_nopq is None or last_greedy_eval_nopq["mean_return_completed_only"] is None
                else float(last_greedy_eval_nopq["mean_return_completed_only"])
            ),
            "greedy_eval_nopq_last_median_return": None if last_greedy_eval_nopq is None else float(last_greedy_eval_nopq["median_return"]),
            "greedy_eval_nopq_last_min_return": None if last_greedy_eval_nopq is None else float(last_greedy_eval_nopq["min_return"]),
            "greedy_eval_nopq_last_max_return": None if last_greedy_eval_nopq is None else float(last_greedy_eval_nopq["max_return"]),
            "greedy_eval_nopq_last_mean_length": None if last_greedy_eval_nopq is None else float(last_greedy_eval_nopq["mean_length"]),
            "greedy_eval_nopq_last_max_length": None if last_greedy_eval_nopq is None else int(last_greedy_eval_nopq["max_length"]),
            "greedy_eval_nopq_last_num_completed": None if last_greedy_eval_nopq is None else int(last_greedy_eval_nopq["num_completed"]),
            "greedy_eval_nopq_last_completion_rate": None if last_greedy_eval_nopq is None else float(last_greedy_eval_nopq["completion_rate"]),
            "greedy_eval_nopq_last_num_truncated": None if last_greedy_eval_nopq is None else int(last_greedy_eval_nopq["num_truncated_total"]),
            "greedy_eval_nopq_last_num_benchmark_truncated": None if last_greedy_eval_nopq is None else int(last_greedy_eval_nopq["num_benchmark_truncated"]),
            "greedy_eval_nopq_last_num_timed_out": None if last_greedy_eval_nopq is None else int(last_greedy_eval_nopq["num_timed_out"]),
            "greedy_eval_nopq_last_num_worker_exited": None if last_greedy_eval_nopq is None else int(last_greedy_eval_nopq["num_worker_exited"]),
            "greedy_eval_nopq_best_mean_return": None if best_greedy_eval_nopq_step is None else float(best_greedy_eval_nopq_mean),
            "greedy_eval_nopq_best_step": None if best_greedy_eval_nopq_step is None else int(best_greedy_eval_nopq_step),
            "greedy_eval_nopq_minus_base_last_mean_return": (
                None
                if (
                    last_greedy_eval_train is None
                    or last_greedy_eval_nopq is None
                )
                else float(last_greedy_eval_nopq["mean_return"] - last_greedy_eval_train["mean_return"])
            ),
            "selection_metric": (
                float(last_greedy_eval_selection_metric)
                if last_greedy_eval_selection_metric is not None
                else (None if len(recent_returns) == 0 else float(last_k_avg_return))
            ),
            "timed_out": bool(timed_out),
            "elapsed_seconds": float(time.time() - start_time),
        }
        with open(cfg.training.hparam_metrics_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote hparam metrics summary to {cfg.training.hparam_metrics_out}")

    final_system_metrics = system_metrics_recorder.maybe_record(
        global_step=global_step,
        elapsed_seconds=time.time() - start_time,
        policy_updates=policy_updates,
        extra_metrics=summarize_state_feature_rollout(
            input.state_features,
            pq_budget=training_pq_budget,
        ),
        force=True,
    )
    if final_system_metrics is not None:
        _log_system_metrics(writer, final_system_metrics, global_step)

    envs.close()
    system_metrics_recorder.close()
    writer.close()
