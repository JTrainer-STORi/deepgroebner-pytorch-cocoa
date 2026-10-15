"""Functions for verifying training configuration, logging training runs either locally or using wandb tracking, Saving model checkpoints"""

from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy
import datetime
import json
import os
from pathlib import Path
import random
import uuid
import numpy as np
import torch

try:
    import wandb
except ImportError:
    wandb = None
from .config_dataclasses import Config
from .local_logging import make_summary_writer

_VERBOSITY_LEVELS = {"quiet": 0, "normal": 1, "verbose": 2}

_SYSTEM_METRIC_TAGS = {
    "elapsed_seconds": "performance/elapsed_seconds",
    "process_cpu_user_seconds": "system_cpu/process_user_seconds",
    "process_cpu_system_seconds": "system_cpu/process_system_seconds",
    "process_cpu_total_seconds": "system_cpu/process_total_seconds",
    "process_cpu_percent": "system_cpu/process_utilization_percent",
    "rss_gib": "system_memory/process_rss_gib",
    "rss_hwm_gib": "system_memory/process_peak_rss_gib",
    "vms_gib": "system_memory/process_virtual_gib",
    "swap_gib": "system_memory/process_swap_gib",
    "cuda_enabled": "system_gpu/cuda_enabled",
    "cuda_memory_allocated_gib": "system_gpu/memory_allocated_gib",
    "cuda_memory_reserved_gib": "system_gpu/memory_reserved_gib",
    "cuda_max_memory_allocated_gib": "system_gpu/peak_memory_allocated_gib",
    "cuda_max_memory_reserved_gib": "system_gpu/peak_memory_reserved_gib",
}


def resolve_training_config(cfg: Config) -> Config:
    """Function for versifying that all of the arguments to the config are acceptable and setting config arguments like batch_size and minibatch_size that depend on other config options"""

    cfg = deepcopy(cfg)

    for name in (
        "num_envs",
        "num_steps_per_update",
        "num_minibatches",
        "num_epochs_per_update",
    ):
        if getattr(cfg.training, name) < 1:
            raise ValueError(f"training.{name} must be positive.")
        
    cfg.training.batch_size = cfg.training.num_envs * cfg.training.num_steps_per_update

    cfg.training.minibatch_size = (
        cfg.training.batch_size // cfg.training.num_minibatches
    )

    if cfg.training.minibatch_size < 1:
        raise ValueError(
            "training.num_minibatches cannot exceed the rollout batch size."
        )
    
    if all(value is not None for value in (cfg.env.coeff_ring, cfg.env.symbols, cfg.env.basis_elements)):
        if isinstance(cfg.env.basis_elements, str):
            cfg.env.basis_elements = cfg.env.basis_elements.split(",")
        cfg.env.fix_ideal = True
        cfg.env.ideal_dist = None
    return cfg


def get_training_output_directory(cfg: Config) -> Path:
    """Resolve the shared root for managed run outputs without changing the config."""
    output_directory = Path(cfg.training.output_directory or "")
    if cfg.training.group is not None:
        output_directory /= cfg.training.group
    return output_directory


def create_run_writer(cfg: Config):
    """Generates the run name and directory for storing run data. Creates a  writer object for tracking run data"""

    # generate a unique ID for the run
    wandb_id = _generate_run_id()

    # Check if the user has specified a fixed ideal or will be using a distribution
    fixed_ideal = all(value is not None for value in (cfg.env.coeff_ring, cfg.env.symbols, cfg.env.basis_elements))

    distribution = "fixed-ideal" if fixed_ideal else cfg.env.ideal_dist

    run_name = f"{cfg.env.gym_id}__{distribution}__{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}__{wandb_id}"

    if fixed_ideal:
        console_print(
            cfg,
            "Using fixed ideal with coefficient ring:",
            cfg.env.coeff_ring,
            "and symbols:",
            cfg.env.symbols,
            "and basis elements:",
            cfg.env.basis_elements,
        )


    if cfg.training.track:
        init_wandb_run(cfg, run_name, wandb_id)

    log_dir = get_training_output_directory(cfg) / "runs"

    writer = make_summary_writer(
        str(log_dir / run_name), mirror_to_wandb=cfg.training.track, wandb_module=wandb
    )

    print(f"Local run logs: {writer.log_dir}")
    print(f"Scalar logs: {writer.scalar_path} and {writer.scalar_csv_path}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join(f"|{key}|{value}|" for key, value in vars(cfg).items())),
    )
    return run_name, writer


def configure_training_runtime(cfg: Config) -> torch.device:
    """Set CPU threads and random seeds, then select the training device."""
    for setting, setter in (
        ("pytorch_num_interop_threads", torch.set_num_interop_threads),
        ("pytorch_num_threads", torch.set_num_threads),
    ):
        value = getattr(cfg.training, setting)
        if value is not None:
            if value < 1:
                raise ValueError(f"training.{setting} must be >= 1")
            if (
                setting != "pytorch_num_interop_threads"
                or value != torch.get_num_interop_threads()
            ):
                setter(value)

    random.seed(cfg.training.training_seed)
    np.random.seed(cfg.training.training_seed)
    torch.manual_seed(cfg.training.training_seed)
    torch.backends.cudnn.deterministic = cfg.training.torch_deterministic

    return torch.device("cuda" if torch.cuda.is_available() and cfg.training.cuda else "cpu")

def _wandb_run_active() -> bool:
    return wandb is not None and wandb.run is not None


def _generate_run_id() -> str:
    """Create an ID for local and tracked runs without relying on W&B SDK helpers."""
    return uuid.uuid4().hex[:8]


def scalarize_metric(value):
    if torch.is_tensor(value):
        return float(value.detach().item())
    return float(value)


def console_allows(cfg, level: str = "normal") -> bool:
    configured_level = getattr(cfg.training, "console_verbosity", "normal")
    return _VERBOSITY_LEVELS.get(configured_level, 1) >= _VERBOSITY_LEVELS[level]


def console_print(cfg, *args, level: str = "normal", **kwargs) -> None:
    if console_allows(cfg, level):
        print(*args, **kwargs)


def format_optional_metric(value, *, precision: int = 4) -> str:
    if value is None:
        return "n/a"
    numeric_value = float(value)
    if not np.isfinite(numeric_value):
        return "n/a"
    return f"{numeric_value:.{precision}f}"


def print_update_summary(
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
        f"avg100={format_optional_metric(last_k_avg_return)}",
        f"last_ep={format_optional_metric(last_episode_return)}",
        f"policy={format_optional_metric(policy_loss)}",
        f"entropy={format_optional_metric(entropy)}",
        f"approx_kl={format_optional_metric(approx_kl)}",
    ]
    if critic_loss is not None:
        parts.append(f"critic={format_optional_metric(critic_loss)}")
    console_print(cfg, " ".join(parts), flush=True)


def log_system_metrics(writer, record: dict[str, object], global_step: int):
    """Route hardware, throughput, and environment samples to their sections.

    Raw timestamps and byte counts stay in system_metrics.jsonl; scalar plots
    use elapsed time and GiB to avoid duplicate curves for the same quantity.
    """
    for key, value in record.items():
        tag = (
            f"evaluation_training/{key}"
            if key.startswith("rollout_")
            else _SYSTEM_METRIC_TAGS.get(key)
        )
        if tag is None:
            continue
        if isinstance(value, bool):
            writer.add_scalar(tag, 1.0 if value else 0.0, global_step)
        elif isinstance(value, (int, float)) and np.isfinite(value):
            writer.add_scalar(tag, value, global_step)


def tensor_stat_to_float(x: torch.Tensor):
    """Converts scalar/vector tensors to a stable float for scalar logging."""
    return x.detach().mean().item()

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


def save_model_checkpoints(
    cfg, agent, run_name: str, checkpoint_suffix: str | None = None
) -> dict[str, str]:
    """Saves configured actor/critic checkpoints and companion config snapshots."""
    output_dir = get_training_output_directory(cfg)
    checkpoint_stem = (
        run_name if checkpoint_suffix is None else f"{run_name}__{checkpoint_suffix}"
    )
    saved_paths = {}

    if cfg.training.save_actor:
        actor_dir = output_dir / "models" / "actors"
        actor_dir.mkdir(parents=True, exist_ok=True)
        actor_model_path = str(actor_dir / f"{checkpoint_stem}.pt")
        torch.save(agent.actor.state_dict(), actor_model_path)
        save_config_snapshot(
            cfg, str(actor_dir / f"{checkpoint_stem}__config.json")
        )
        saved_paths["actor"] = actor_model_path

    if cfg.critic.critic_strategy != "heuristic-only" and cfg.training.save_critic:
        critic_dir = output_dir / "models" / "critics"
        critic_dir.mkdir(parents=True, exist_ok=True)
        critic_model_path = str(critic_dir / f"{checkpoint_stem}.pt")
        torch.save(agent.critic.state_dict(), critic_model_path)
        save_config_snapshot(
            cfg, str(critic_dir / f"{checkpoint_stem}__config.json")
        )
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
    if (
        cfg.training.output_directory is not None
        or cfg.training.group is not None
    ):
        output_dir = get_training_output_directory(cfg)
        output_dir.mkdir(parents=True, exist_ok=True)
        init_kwargs["dir"] = str(output_dir)
    if cfg.training.group is not None:
        init_kwargs["group"] = cfg.training.group
    # In sweep runs, W&B injects parameter config already; avoid duplicate keys.
    if not is_sweep_run:
        init_kwargs["config"] = _jsonify_config_value(asdict(cfg))

    # Try except catches any errors that can occur due to a timeout when connecting to wandb server
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

    console_print(cfg, "\n=== Runtime CPU Concurrency Info ===", level="verbose")
    console_print(
        cfg,
        "PyTorch threads: "
        f"intra_op={torch_intra_threads}, inter_op={torch_interop_threads}, "
        f"OMP_NUM_THREADS={omp_threads}, MKL_NUM_THREADS={mkl_threads}",
        level="verbose",
    )
    console_print(
        cfg,
        "Gymnasium vectorization: "
        f"mode={gym_vector_mode}, num_envs={gym_num_envs}, worker_units={gym_worker_units}",
        level="verbose",
    )
    console_print(cfg, "====================================\n", level="verbose")
