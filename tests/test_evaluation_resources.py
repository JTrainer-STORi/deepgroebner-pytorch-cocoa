from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from deepgroebner_pytorch_cocoa import evaluation_resources as resources


def test_cpu_measurement_has_stable_fields_and_finishes_once(monkeypatch):
    snapshots = Mock(side_effect=[
        {"rss_bytes": 4096, "rss_hwm_bytes": 8192},
        {"rss_bytes": 2048, "rss_hwm_bytes": 16384},
    ])
    wall_clock = Mock(side_effect=[10.0, 12.5])
    cpu_clock = Mock(side_effect=[20.0, 21.5])
    synchronize = Mock(side_effect=AssertionError("CPU must not synchronize CUDA"))
    monkeypatch.setattr(resources, "collect_system_metrics_snapshot", snapshots)
    monkeypatch.setattr(resources.time, "perf_counter", wall_clock)
    monkeypatch.setattr(resources.time, "process_time", cpu_clock)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)

    measurement = resources.ResourceMeasurement(torch.device("cpu"))
    result = measurement.finish()

    assert set(result) == set(resources.RESOURCE_FIELDS)
    assert result["elapsed_seconds"] == 2.5
    assert result["cpu_seconds"] == 1.5
    assert result["rss_start_bytes"] == 4096
    assert result["rss_end_bytes"] == 2048
    assert result["rss_delta_bytes"] == -2048
    assert result["process_peak_rss_bytes"] == 16384
    assert all(value is None for key, value in result.items() if key.startswith("cuda_"))
    assert measurement.finish() == result
    result["elapsed_seconds"] = -1
    assert measurement.finish()["elapsed_seconds"] == 2.5
    assert wall_clock.call_count == cpu_clock.call_count == snapshots.call_count == 2
    synchronize.assert_not_called()


@pytest.mark.parametrize("platform,multiplier", [("linux", 1024), ("darwin", 1)])
def test_process_peak_fallback_normalizes_platform_units(monkeypatch, platform, multiplier):
    usage = Mock(side_effect=[
        SimpleNamespace(ru_maxrss=1024), SimpleNamespace(ru_maxrss=2048),
    ])
    monkeypatch.setattr(resources, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(
        resources, "_resource", SimpleNamespace(RUSAGE_SELF=0, getrusage=usage)
    )
    monkeypatch.setattr(resources, "collect_system_metrics_snapshot", lambda device: {})

    result = resources.ResourceMeasurement().finish()

    assert result["process_peak_rss_bytes"] == 2048 * multiplier
    assert result["rss_start_bytes"] is None
    assert result["rss_end_bytes"] is None
    assert result["rss_delta_bytes"] is None
    assert usage.call_count == 2
    usage.assert_called_with(0)


def test_native_process_peak_takes_precedence_over_fallback(monkeypatch):
    usage = Mock(side_effect=AssertionError("Native high-water mark is available"))
    monkeypatch.setattr(
        resources, "_resource", SimpleNamespace(RUSAGE_SELF=0, getrusage=usage)
    )
    monkeypatch.setattr(
        resources, "collect_system_metrics_snapshot",
        lambda device: {"rss_bytes": 4096, "rss_hwm_bytes": 8192},
    )

    assert resources.ResourceMeasurement().finish()["process_peak_rss_bytes"] == 8192
    usage.assert_not_called()


@pytest.mark.parametrize("resource_available", [False, True])
def test_unavailable_memory_is_null_without_losing_timing(monkeypatch, resource_available):
    fallback = (
        SimpleNamespace(RUSAGE_SELF=0, getrusage=Mock(side_effect=OSError("unavailable")))
        if resource_available else None
    )
    monkeypatch.setattr(resources, "_resource", fallback)
    monkeypatch.setattr(resources, "collect_system_metrics_snapshot", lambda device: {})
    monkeypatch.setattr(resources.time, "perf_counter", Mock(side_effect=[1.0, 4.0]))
    monkeypatch.setattr(resources.time, "process_time", Mock(side_effect=[2.0, 3.0]))

    result = resources.ResourceMeasurement().finish()

    assert result["elapsed_seconds"] == 3.0
    assert result["cpu_seconds"] == 1.0
    assert all(value is None for key, value in result.items() if key.endswith("_bytes"))


def test_cuda_synchronizes_timing_boundaries_without_resetting_global_peaks(monkeypatch):
    events = []
    snapshots = iter([
        {
            "rss_bytes": 4096, "rss_hwm_bytes": 8192,
            "cuda_memory_allocated_bytes": 100,
            "cuda_memory_reserved_bytes": 200,
            "cuda_max_memory_allocated_bytes": 300,
            "cuda_max_memory_reserved_bytes": 400,
        },
        {
            "rss_bytes": 8192, "rss_hwm_bytes": 16384,
            "cuda_memory_allocated_bytes": 50,
            "cuda_memory_reserved_bytes": 250,
            "cuda_max_memory_allocated_bytes": 350,
            "cuda_max_memory_reserved_bytes": 500,
        },
    ])
    wall_times = iter([10.0, 15.0])
    cpu_times = iter([2.0, 3.0])

    def snapshot(device):
        assert device == torch.device("cuda:2")
        events.append("snapshot")
        return next(snapshots)

    def wall_clock():
        events.append("wall")
        return next(wall_times)

    def cpu_clock():
        events.append("cpu")
        return next(cpu_times)

    def synchronize(device):
        assert device == torch.device("cuda:2")
        events.append("synchronize")

    reset_peaks = Mock(side_effect=AssertionError("Global peaks must not be reset"))
    monkeypatch.setattr(resources, "collect_system_metrics_snapshot", snapshot)
    monkeypatch.setattr(resources.time, "perf_counter", wall_clock)
    monkeypatch.setattr(resources.time, "process_time", cpu_clock)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset_peaks)

    measurement = resources.ResourceMeasurement(torch.device("cuda:2"))
    events.append("work")
    result = measurement.finish()

    assert events == [
        "synchronize", "snapshot", "cpu", "wall", "work", "synchronize",
        "wall", "cpu", "snapshot",
    ]
    assert result["elapsed_seconds"] == 5.0
    assert result["cpu_seconds"] == 1.0
    assert result["cuda_allocated_start_bytes"] == 100
    assert result["cuda_allocated_end_bytes"] == 50
    assert result["cuda_allocated_delta_bytes"] == -50
    assert result["cuda_reserved_start_bytes"] == 200
    assert result["cuda_reserved_end_bytes"] == 250
    assert result["cuda_reserved_delta_bytes"] == 50
    assert result["cuda_peak_allocated_bytes"] == 350
    assert result["cuda_peak_reserved_bytes"] == 500
    reset_peaks.assert_not_called()


def test_unavailable_cuda_device_keeps_cuda_fields_null(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    synchronize = Mock(side_effect=AssertionError("CUDA is unavailable"))
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(
        resources, "collect_system_metrics_snapshot",
        lambda device: {"rss_hwm_bytes": 8192},
    )

    result = resources.ResourceMeasurement(torch.device("cuda:0")).finish()

    assert all(value is None for key, value in result.items() if key.startswith("cuda_"))
    synchronize.assert_not_called()


def test_resource_statistics_aggregates_timings_and_preserves_peak_meaning():
    rows = [
        {
            "elapsed_seconds": wall, "cpu_seconds": wall + 1,
            "rss_end_bytes": wall * 10, "process_peak_rss_bytes": wall * 100,
            "cuda_peak_allocated_bytes": wall * 20,
            "cuda_peak_reserved_bytes": wall * 30,
        }
        for wall in (1.0, 3.0, 5.0)
    ]

    result = resources.resource_statistics(iter(rows))

    assert set(result) == set(resources.RESOURCE_STATISTIC_FIELDS)
    assert result["elapsed_seconds_sum"] == 9.0
    assert result["elapsed_seconds_mean"] == 3.0
    assert result["elapsed_seconds_std"] == pytest.approx((8 / 3) ** 0.5)
    assert result["elapsed_seconds_min"] == 1.0
    assert result["elapsed_seconds_median"] == 3.0
    assert result["elapsed_seconds_max"] == 5.0
    assert result["cpu_seconds_sum"] == 12.0
    assert result["cpu_seconds_mean"] == 4.0
    assert result["cpu_seconds_std"] == pytest.approx((8 / 3) ** 0.5)
    assert result["cpu_seconds_min"] == 2.0
    assert result["cpu_seconds_median"] == 4.0
    assert result["cpu_seconds_max"] == 6.0
    assert result["rss_end_bytes_mean"] == 30.0
    assert result["rss_end_bytes_min"] == 10.0
    assert result["rss_end_bytes_max"] == 50.0
    assert result["process_peak_rss_bytes_max"] == 500.0
    assert result["cuda_peak_allocated_bytes_max"] == 100.0
    assert result["cuda_peak_reserved_bytes_max"] == 150.0


def test_statistics_filters_unavailable_and_nonfinite_values():
    rows = [
        {"elapsed_seconds": 2.0, "rss_end_bytes": 4096},
        {"elapsed_seconds": float("inf"), "rss_end_bytes": float("nan")},
        {"elapsed_seconds": None, "rss_end_bytes": None},
    ]

    result = resources.resource_statistics(rows)

    assert result["elapsed_seconds_sum"] == 2.0
    assert result["elapsed_seconds_mean"] == 2.0
    assert result["elapsed_seconds_std"] == 0.0
    assert result["cpu_seconds_sum"] is None
    assert result["rss_end_bytes_mean"] == 4096
    assert result["process_peak_rss_bytes_max"] is None


def test_empty_statistics_preserves_field_schema():
    result = resources.resource_statistics([])

    assert set(result) == set(resources.RESOURCE_STATISTIC_FIELDS)
    assert result["elapsed_seconds_sum"] == 0.0
    assert result["cpu_seconds_sum"] == 0.0
    assert all(value is None for key, value in result.items() if not key.endswith("_sum"))


@pytest.mark.parametrize("aggregate", [False, True])
def test_terminal_summary_formats_measurements_and_aggregates(aggregate):
    measurement = {
        "elapsed_seconds": 1.2345, "cpu_seconds": 0.25,
        "process_peak_rss_bytes": 32 * 1024 ** 2,
        "cuda_peak_allocated_bytes": 2 * 1024 ** 2,
        "cuda_peak_reserved_bytes": 4 * 1024 ** 2,
    }
    result = resources.resource_statistics([measurement]) if aggregate else measurement

    assert resources.format_resource_summary(result) == (
        "wall 1.234 s, CPU 0.250 s, peak RSS 32.0 MiB, "
        "CUDA peak allocated 2.0 MiB, reserved 4.0 MiB"
    )


def test_terminal_summary_handles_unavailable_memory_and_metadata_describes_peaks():
    text = resources.format_resource_summary({"elapsed_seconds": 1.0, "cpu_seconds": 0.5})
    metadata = resources.resource_metadata()

    assert text == "wall 1.000 s, CPU 0.500 s, peak RSS n/a MiB"
    assert metadata["time_unit"] == "seconds"
    assert metadata["memory_unit"] == "bytes"
    assert "CoCoA" in metadata["rss"]
    assert "process start" in metadata["process_peak_rss"]
    assert "last external reset" in metadata["cuda_peak_memory"]
