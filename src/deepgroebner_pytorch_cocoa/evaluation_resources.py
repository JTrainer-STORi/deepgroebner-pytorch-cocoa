"""Runtime and process/device memory measurements for standalone evaluations.

RSS includes native CoCoA allocations, unlike Python allocation trackers. Memory
peaks are cumulative high-water marks; they are not isolated episode peaks.
"""

from __future__ import annotations

import math
import statistics
import sys
import time
from collections.abc import Iterable, Mapping
from numbers import Real
from typing import Any

import torch

from .system_metrics import collect_system_metrics_snapshot

try:
    import resource as _resource
except ImportError:  # Windows does not provide the resource module.
    _resource = None


RESOURCE_FIELDS = (
    "elapsed_seconds", "cpu_seconds", "rss_start_bytes", "rss_end_bytes",
    "rss_delta_bytes", "process_peak_rss_bytes", "cuda_allocated_start_bytes",
    "cuda_allocated_end_bytes", "cuda_allocated_delta_bytes",
    "cuda_reserved_start_bytes", "cuda_reserved_end_bytes",
    "cuda_reserved_delta_bytes", "cuda_peak_allocated_bytes",
    "cuda_peak_reserved_bytes",
)
RESOURCE_STATISTIC_FIELDS = tuple(
    f"{key}_{statistic}"
    for key in ("elapsed_seconds", "cpu_seconds")
    for statistic in ("sum", "mean", "std", "min", "median", "max")
) + (
    "rss_end_bytes_mean", "rss_end_bytes_min", "rss_end_bytes_max",
    "process_peak_rss_bytes_max", "cuda_peak_allocated_bytes_max",
    "cuda_peak_reserved_bytes_max",
)


def _byte_count(value: Any) -> int | None:
    if (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    ):
        return int(value)
    return None


def _process_peak_rss_bytes(snapshot: Mapping[str, Any]) -> int | None:
    """Use the OS RSS high-water mark, normalizing getrusage's platform units."""
    peak = _byte_count(snapshot.get("rss_hwm_bytes"))
    if peak is not None or _resource is None:
        return peak
    try:
        peak = _byte_count(_resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss)
    except (AttributeError, OSError, ValueError):
        return None
    if peak is None:
        return None
    # Linux and other Unix systems report KiB; macOS reports bytes.
    return peak if sys.platform == "darwin" else peak * 1024


def _difference(end: int | None, start: int | None) -> int | None:
    return None if end is None or start is None else end - start


def _maximum(*values: int | None) -> int | None:
    available = [value for value in values if value is not None]
    return max(available) if available else None


class ResourceMeasurement:
    """Measure one caller-selected scope, starting at construction.

    CUDA work on the selected device is synchronized before each timing boundary.
    Allocator peak counters are read without resetting shared global state.
    Calling ``finish`` again returns the original measurement.
    """

    def __init__(self, device: torch.device | None = None):
        self.device = torch.device(device) if device is not None else None
        self._cuda_enabled = bool(
            self.device is not None
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        )
        if self._cuda_enabled:
            torch.cuda.synchronize(self.device)
        self._start_snapshot = collect_system_metrics_snapshot(self.device)
        self._start_peak = _process_peak_rss_bytes(self._start_snapshot)
        self._cpu_start = time.process_time()
        self._wall_start = time.perf_counter()
        self._result: dict[str, float | int | None] | None = None

    def finish(self) -> dict[str, float | int | None]:
        """Return seconds and bytes, with ``None`` for unavailable measurements."""
        if self._result is not None:
            return dict(self._result)
        if self._cuda_enabled:
            torch.cuda.synchronize(self.device)
        wall_end = time.perf_counter()
        cpu_end = time.process_time()
        snapshot = collect_system_metrics_snapshot(self.device)
        rss_start = _byte_count(self._start_snapshot.get("rss_bytes"))
        rss_end = _byte_count(snapshot.get("rss_bytes"))
        result: dict[str, float | int | None] = {
            "elapsed_seconds": wall_end - self._wall_start,
            "cpu_seconds": cpu_end - self._cpu_start,
            "rss_start_bytes": rss_start,
            "rss_end_bytes": rss_end,
            "rss_delta_bytes": _difference(rss_end, rss_start),
            "process_peak_rss_bytes": _maximum(
                self._start_peak, _process_peak_rss_bytes(snapshot)
            ),
        }
        for label in ("allocated", "reserved"):
            source_key = f"cuda_memory_{label}_bytes"
            start = (
                _byte_count(self._start_snapshot.get(source_key))
                if self._cuda_enabled else None
            )
            end = _byte_count(snapshot.get(source_key)) if self._cuda_enabled else None
            peak_key = f"cuda_max_memory_{label}_bytes"
            peak = (
                _maximum(
                    _byte_count(self._start_snapshot.get(peak_key)),
                    _byte_count(snapshot.get(peak_key)),
                )
                if self._cuda_enabled else None
            )
            result.update({
                f"cuda_{label}_start_bytes": start,
                f"cuda_{label}_end_bytes": end,
                f"cuda_{label}_delta_bytes": _difference(end, start),
                f"cuda_peak_{label}_bytes": peak,
            })
        self._result = result
        return dict(result)


def _values(rows: Iterable[Mapping[str, Any]], key: str) -> list[int | float]:
    return [
        value
        for row in rows
        if isinstance(value := row.get(key), Real)
        and not isinstance(value, bool)
        and math.isfinite(value)
    ]


def resource_statistics(rows: Iterable[Mapping[str, Any]]) -> dict[str, float | int | None]:
    """Aggregate episode timings and memory, using population standard deviation.

    Cumulative high-water marks are aggregated only by their maximum. Empty or
    unavailable measurements produce ``None`` (timing sums are zero for no rows).
    """
    rows = tuple(rows)
    result: dict[str, float | int | None] = {}
    for key in ("elapsed_seconds", "cpu_seconds"):
        values = _values(rows, key)
        result.update({
            f"{key}_sum": sum(values) if values else (0.0 if not rows else None),
            f"{key}_mean": statistics.fmean(values) if values else None,
            f"{key}_std": statistics.pstdev(values) if values else None,
            f"{key}_min": min(values) if values else None,
            f"{key}_median": statistics.median(values) if values else None,
            f"{key}_max": max(values) if values else None,
        })
    rss = _values(rows, "rss_end_bytes")
    result.update({
        "rss_end_bytes_mean": statistics.fmean(rss) if rss else None,
        "rss_end_bytes_min": min(rss) if rss else None,
        "rss_end_bytes_max": max(rss) if rss else None,
    })
    for key in (
        "process_peak_rss_bytes",
        "cuda_peak_allocated_bytes",
        "cuda_peak_reserved_bytes",
    ):
        values = _values(rows, key)
        result[f"{key}_max"] = max(values) if values else None
    return result


def format_resource_summary(resources: Mapping[str, Any]) -> str:
    """Format either one measurement or aggregate statistics for the terminal."""
    def scalar(key: str, aggregate_suffix: str, scale: float, digits: int) -> str:
        value = resources.get(key, resources.get(f"{key}_{aggregate_suffix}"))
        if not isinstance(value, Real) or not math.isfinite(value):
            return "n/a"
        return f"{value / scale:.{digits}f}"

    parts = [
        f"wall {scalar('elapsed_seconds', 'sum', 1, 3)} s",
        f"CPU {scalar('cpu_seconds', 'sum', 1, 3)} s",
        f"peak RSS {scalar('process_peak_rss_bytes', 'max', 1024 ** 2, 1)} MiB",
    ]
    if resources.get(
        "cuda_peak_allocated_bytes", resources.get("cuda_peak_allocated_bytes_max")
    ) is not None:
        parts.append(
            "CUDA peak allocated "
            f"{scalar('cuda_peak_allocated_bytes', 'max', 1024 ** 2, 1)} MiB"
        )
        parts.append(
            "reserved "
            f"{scalar('cuda_peak_reserved_bytes', 'max', 1024 ** 2, 1)} MiB"
        )
    return ", ".join(parts)


def resource_metadata() -> dict[str, str]:
    """Describe the scope and limitations accompanying resource statistics."""
    return {
        "time_unit": "seconds",
        "memory_unit": "bytes",
        "elapsed_seconds": (
            "Wall time measured with perf_counter; CUDA work is synchronized at "
            "timing boundaries. Report serialization is excluded."
        ),
        "cpu_seconds": (
            "Process CPU time across all threads, including Python, PyTorch and "
            "native CoCoA work; excludes child processes."
        ),
        "rss": (
            "Process-wide resident memory at timing boundaries, including Python, "
            "PyTorch and native CoCoA allocations; deltas can be negative."
        ),
        "process_peak_rss": (
            "Cumulative process RSS high-water mark since process start, including "
            "setup and earlier evaluations; not an isolated episode peak."
        ),
        "cuda_memory": "PyTorch allocator memory on the selected CUDA device only.",
        "cuda_peak_memory": (
            "Cumulative PyTorch allocator peaks since its last external reset, "
            "including setup and earlier evaluations; counters are not reset here."
        ),
        "missing_measurements": "Unavailable or inapplicable memory values are null.",
    }
