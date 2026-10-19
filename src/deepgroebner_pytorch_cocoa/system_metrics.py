"""Functions for monitoring and logging system metrics during training"""

import hashlib
import json
import logging
import math
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import torch


_BYTES_PER_GIB = float(1024 ** 3)
_LOGGER = logging.getLogger(__name__)
_STATUS_FIELDS_KIB = {
    "VmRSS": "rss_bytes",
    "VmHWM": "rss_hwm_bytes",
    "VmSize": "vms_bytes",
    "VmSwap": "swap_bytes",
}

STATE_FEATURE_G_CARDINALITY_IDX = 0
STATE_FEATURE_P_CARDINALITY_IDX = 1
STATE_FEATURE_N_PAIRS_PROCESSED_IDX = 12
STATE_FEATURE_STEPS_SINCE_BASIS_UPDATE_IDX = 13


def _wandb_system_metric_tag(raw_name: str) -> str:
    """Give SDK metrics one history section while retaining their device/path names."""
    name = raw_name.removeprefix("system.").removeprefix("system/")
    components = re.split(r"[./_]", name.lower())
    if components[0] == "gpu":
        section = "system_gpu"
    elif components[0] == "cpu" or components[:2] == ["proc", "cpu"]:
        section = "system_cpu"
    elif components[0] in {"memory", "swap"} or components[:2] == ["proc", "memory"]:
        section = "system_memory"
    else:
        section = "system_general"
    leaf = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_") or "metric"
    return f"{section}/{leaf}"


class WandbSystemMetricsAdapter:
    """Mirror fresh SDK system samples into ordinary namespaced history metrics.

    Enable ``wandb.Settings(x_stats_buffer_size=...)`` before creating the run.
    W&B exposes these samples through ``run._system_metrics``; that SDK hook is
    internal, so access failures leave training and native telemetry running.
    Native System charts remain available because their stream cannot be renamed.
    Call ``collect`` while the run is active, including once before closing it.
    """

    def __init__(self, run):
        self.run = run
        self._last_timestamps: dict[str, float] = {}
        self._metric_tags: dict[str, str] = {}
        self._tag_owners: dict[str, str] = {}
        self._warned_failure = False

    def _metric_tag(self, raw_name: str) -> str:
        if raw_name in self._metric_tags:
            return self._metric_tags[raw_name]
        base_tag = _wandb_system_metric_tag(raw_name)
        tag = base_tag
        if tag in self._tag_owners:
            suffix = hashlib.sha256(raw_name.encode("utf-8")).hexdigest()[:8]
            tag = f"{base_tag}_{suffix}"
            counter = 1
            while tag in self._tag_owners:
                tag = f"{base_tag}_{suffix}_{counter}"
                counter += 1
        self._metric_tags[raw_name] = tag
        self._tag_owners[tag] = raw_name
        return tag

    def collect(self) -> dict[str, float]:
        """Return each metric's latest finite sample, once per SDK timestamp."""
        try:
            buffered_metrics = self.run._system_metrics
        except Exception:
            if not self._warned_failure:
                _LOGGER.warning(
                    "Cannot read W&B's system metric buffer; namespaced copies are "
                    "unavailable, while W&B's native system collection remains enabled."
                )
                self._warned_failure = True
            return {}
        if not isinstance(buffered_metrics, Mapping):
            return {}

        metrics = {}
        raw_names = sorted(
            name for name in buffered_metrics if isinstance(name, str) and name
        )
        for raw_name in raw_names:
            samples = buffered_metrics[raw_name]
            if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
                continue
            newest_timestamp = self._last_timestamps.get(raw_name, -math.inf)
            newest_value = None
            for sample in samples:
                if not isinstance(sample, (tuple, list)) or len(sample) != 2:
                    continue
                timestamp, value = sample
                if isinstance(timestamp, datetime):
                    timestamp = timestamp.timestamp()
                if not isinstance(timestamp, Real) or not isinstance(value, Real):
                    continue
                timestamp, value = float(timestamp), float(value)
                if (
                    math.isfinite(timestamp)
                    and math.isfinite(value)
                    and timestamp > newest_timestamp
                ):
                    newest_timestamp, newest_value = timestamp, value
            if newest_value is not None:
                metrics[self._metric_tag(raw_name)] = newest_value
                self._last_timestamps[raw_name] = newest_timestamp
        return metrics


def _bytes_to_gib(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / _BYTES_PER_GIB


def _read_proc_status_bytes() -> dict[str, int | None]:
    values = {name: None for name in _STATUS_FIELDS_KIB.values()}
    status_path = Path("/proc/self/status")
    if not status_path.exists():
        return values

    try:
        with status_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                key, _, raw_value = line.partition(":")
                field_name = _STATUS_FIELDS_KIB.get(key)
                if field_name is None:
                    continue
                parts = raw_value.strip().split()
                if len(parts) < 2 or parts[1].lower() != "kb":
                    continue
                values[field_name] = int(parts[0]) * 1024
    except OSError:
        return values

    return values


def collect_system_metrics_snapshot(device: torch.device | None = None) -> dict[str, Any]:
    """Collect main-process CPU times and best-effort process/device memory."""
    record = _read_proc_status_bytes()
    for byte_key in list(record.keys()):
        gib_key = byte_key.removesuffix("_bytes") + "_gib"
        record[gib_key] = _bytes_to_gib(record[byte_key])

    cpu_times = os.times()
    record.update(
        process_cpu_user_seconds=cpu_times.user,
        process_cpu_system_seconds=cpu_times.system,
        process_cpu_total_seconds=cpu_times.user + cpu_times.system,
    )

    cuda_enabled = (
        device is not None
        and getattr(device, "type", None) == "cuda"
        and torch.cuda.is_available()
    )
    record["cuda_enabled"] = bool(cuda_enabled)

    if cuda_enabled:
        device_idx = device.index if device.index is not None else torch.cuda.current_device()
        allocated = int(torch.cuda.memory_allocated(device_idx))
        reserved = int(torch.cuda.memory_reserved(device_idx))
        max_allocated = int(torch.cuda.max_memory_allocated(device_idx))
        max_reserved = int(torch.cuda.max_memory_reserved(device_idx))
        record.update(
            {
                "cuda_memory_allocated_bytes": allocated,
                "cuda_memory_allocated_gib": _bytes_to_gib(allocated),
                "cuda_memory_reserved_bytes": reserved,
                "cuda_memory_reserved_gib": _bytes_to_gib(reserved),
                "cuda_max_memory_allocated_bytes": max_allocated,
                "cuda_max_memory_allocated_gib": _bytes_to_gib(max_allocated),
                "cuda_max_memory_reserved_bytes": max_reserved,
                "cuda_max_memory_reserved_gib": _bytes_to_gib(max_reserved),
            }
        )

    return record


def _coerce_state_feature_tensor(state_features: torch.Tensor | np.ndarray | Sequence[Any]) -> torch.Tensor | None:
    if torch.is_tensor(state_features):
        tensor = state_features.detach().to(device="cpu", dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    if isinstance(state_features, np.ndarray):
        tensor = torch.as_tensor(state_features, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    if not isinstance(state_features, Sequence):
        return None

    rows: list[torch.Tensor] = []
    for row in state_features:
        if torch.is_tensor(row):
            rows.append(row.detach().to(device="cpu", dtype=torch.float32).reshape(-1))
        else:
            rows.append(torch.as_tensor(row, dtype=torch.float32).reshape(-1))

    if not rows:
        return None

    return torch.stack(rows, dim=0)


def summarize_state_feature_rollout(
    state_features: torch.Tensor | np.ndarray | Sequence[Any],
    *,
    pq_budget: int | None = None,
) -> dict[str, float | bool]:
    """Summarizes rollout-scale environment growth statistics from state_features."""
    tensor = _coerce_state_feature_tensor(state_features)
    if tensor is None or tensor.numel() == 0 or tensor.shape[-1] <= STATE_FEATURE_STEPS_SINCE_BASIS_UPDATE_IDX:
        return {}

    metrics: dict[str, float | bool] = {"rollout_num_states": float(tensor.shape[0])}
    feature_specs = (
        ("basis_count", STATE_FEATURE_G_CARDINALITY_IDX),
        ("pair_count", STATE_FEATURE_P_CARDINALITY_IDX),
        ("pairs_processed", STATE_FEATURE_N_PAIRS_PROCESSED_IDX),
        ("steps_since_basis_update", STATE_FEATURE_STEPS_SINCE_BASIS_UPDATE_IDX),
    )
    for name, idx in feature_specs:
        values = tensor[:, idx]
        metrics[f"rollout_{name}_mean"] = float(values.mean().item())
        metrics[f"rollout_{name}_max"] = float(values.max().item())

    if pq_budget is not None and pq_budget > 0:
        pair_counts = tensor[:, STATE_FEATURE_P_CARDINALITY_IDX]
        pair_overflow = torch.clamp(pair_counts - float(pq_budget), min=0.0)
        binding_mask = pair_counts > float(pq_budget)
        metrics["rollout_pq_budget"] = float(pq_budget)
        metrics["rollout_pq_binding_num_states"] = float(binding_mask.sum().item())
        metrics["rollout_pq_binding_step_fraction"] = float(binding_mask.float().mean().item())
        metrics["rollout_pq_binding_ever"] = bool(binding_mask.any().item())
        metrics["rollout_pair_overflow_mean"] = float(pair_overflow.mean().item())
        metrics["rollout_pair_overflow_max"] = float(pair_overflow.max().item())

    return metrics


class SystemMetricsRecorder:
    """Writes time-stamped system metrics to a JSONL file."""

    def __init__(
        self,
        output_path: str | os.PathLike[str],
        *,
        device: torch.device | None = None,
        enabled: bool = True,
        sample_interval_seconds: float = 10.0,
        snapshot_fn: Callable[[torch.device | None], dict[str, Any]] = collect_system_metrics_snapshot,
        clock_fn: Callable[[], float] = time.time,
    ):
        self.output_path = Path(output_path)
        self.device = device
        self.enabled = bool(enabled)
        self.sample_interval_seconds = float(sample_interval_seconds)
        self.snapshot_fn = snapshot_fn
        self.clock_fn = clock_fn
        self.last_sample_time: float | None = None
        self._last_cpu_total_seconds: float | None = None
        self._handle = None

        if self.enabled:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.output_path.open("a", encoding="utf-8")

    def record(
        self,
        *,
        global_step: int,
        elapsed_seconds: float,
        policy_updates: int | None = None,
        extra_metrics: dict[str, Any] | None = None,
        sample_time: float | None = None,
    ) -> dict[str, Any]:
        """Write one snapshot; the caller decides whether and when to record."""
        if self._handle is None:
            raise RuntimeError(
                "System metrics recorder must be enabled and open to record."
            )

        now = self.clock_fn() if sample_time is None else sample_time

        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "timestamp_unix_seconds": float(now),
            "elapsed_seconds": float(elapsed_seconds),
            "global_step": int(global_step),
        }
        if policy_updates is not None:
            record["policy_updates"] = int(policy_updates)
        record.update(self.snapshot_fn(self.device))
        if extra_metrics:
            record.update(extra_metrics)

        cpu_total = record.get("process_cpu_total_seconds")
        if isinstance(cpu_total, (int, float)) and math.isfinite(cpu_total):
            if self.last_sample_time is not None and self._last_cpu_total_seconds is not None:
                wall_delta = now - self.last_sample_time
                cpu_delta = cpu_total - self._last_cpu_total_seconds
                if wall_delta > 0 and cpu_delta >= 0:
                    # 100% means one CPU core; multiple busy threads can exceed it.
                    record["process_cpu_percent"] = 100.0 * cpu_delta / wall_delta
            self._last_cpu_total_seconds = cpu_total
        else:
            self._last_cpu_total_seconds = None

        self._handle.write(json.dumps(record) + "\n")
        self._handle.flush()
        self.last_sample_time = now
        return record

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
