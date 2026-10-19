from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from deepgroebner_pytorch_cocoa.system_metrics import WandbSystemMetricsAdapter


@pytest.mark.parametrize(
    "raw_name,expected_tag",
    [
        ("cpu", "system_cpu/cpu"),
        ("system.cpu.0.cpu_percent", "system_cpu/cpu_0_cpu_percent"),
        ("proc.cpu.threads", "system_cpu/proc_cpu_threads"),
        ("system/gpu/0/gpu", "system_gpu/gpu_0_gpu"),
        ("gpu.0.memoryAllocatedBytes", "system_gpu/gpu_0_memoryAllocatedBytes"),
        ("gpu.process.1.memory", "system_gpu/gpu_process_1_memory"),
        ("gpu.0.temp", "system_gpu/gpu_0_temp"),
        ("gpu.0.powerWatts", "system_gpu/gpu_0_powerWatts"),
        ("gpu.0.powerPercent", "system_gpu/gpu_0_powerPercent"),
        ("memory", "system_memory/memory"),
        ("memory_percent", "system_memory/memory_percent"),
        ("proc.memory.rssMB", "system_memory/proc_memory_rssMB"),
        ("proc.memory.availableMB", "system_memory/proc_memory_availableMB"),
        ("swap.percent", "system_memory/swap_percent"),
        ("disk./.usagePercent", "system_general/disk_usagePercent"),
        ("disk./home.usageGB", "system_general/disk_home_usageGB"),
        ("disk.in", "system_general/disk_in"),
        ("network.sent", "system_general/network_sent"),
        ("network.recv", "system_general/network_recv"),
        ("openmetrics.cluster.metric", "system_general/openmetrics_cluster_metric"),
    ],
)
def test_sdk_metrics_are_routed_without_losing_host_or_gpu_telemetry(raw_name, expected_tag):
    #arrange
    timestamp = datetime(2026, 10, 15, tzinfo=timezone.utc)
    run = SimpleNamespace(_system_metrics={raw_name: [(timestamp, 1234.5)]})
    adapter = WandbSystemMetricsAdapter(run)

    #act
    metrics = adapter.collect()

    #assert
    assert metrics == {expected_tag: 1234.5}
    assert expected_tag.count("/") == 1


def test_adapter_uses_latest_sample_and_does_not_repeat_buffered_measurements():
    #arrange
    start = datetime(2026, 10, 15, tzinfo=timezone.utc)
    samples = [(start, 10.0), (start + timedelta(seconds=2), 30.0),
               (start + timedelta(seconds=1), 20.0)]
    run = SimpleNamespace(_system_metrics={"cpu": samples})
    adapter = WandbSystemMetricsAdapter(run)

    #act
    first = adapter.collect()
    repeated = adapter.collect()

    #assert
    assert first == {"system_cpu/cpu": 30.0}
    assert repeated == {}

    #arrange
    samples.append((start + timedelta(seconds=3), 40.0))

    #act
    next_sample = adapter.collect()

    #assert
    assert next_sample == {"system_cpu/cpu": 40.0}


def test_adapter_tracks_freshness_per_metric_with_asynchronous_sdk_sampling():
    #arrange
    run = SimpleNamespace(_system_metrics={"cpu": [(10.0, 25.0)]})
    adapter = WandbSystemMetricsAdapter(run)

    #act
    first = adapter.collect()

    #assert
    assert first == {"system_cpu/cpu": 25.0}

    #arrange
    run._system_metrics["gpu.0.gpu"] = [(9.0, 75.0)]
    run._system_metrics["cpu"] = [(8.0, 10.0), (10.0, 25.0)]

    #act
    second = adapter.collect()

    #assert
    assert second == {"system_gpu/gpu_0_gpu": 75.0}


def test_flattened_metric_names_preserve_colliding_paths_and_stay_stable():
    #arrange
    dashed_path = "disk./mnt/data-set.usagePercent"
    underscored_path = "disk./mnt/data_set.usagePercent"
    run = SimpleNamespace(
        _system_metrics={
            dashed_path: [(1.0, 25.0)],
            underscored_path: [(1.0, 75.0)],
        }
    )
    adapter = WandbSystemMetricsAdapter(run)

    #act
    first = adapter.collect()

    #assert
    assert len(first) == 2
    assert sorted(first.values()) == [25.0, 75.0]
    assert "system_general/disk_mnt_data_set_usagePercent" in first
    assert all(tag.count("/") == 1 for tag in first)

    #arrange
    run._system_metrics = {
        underscored_path: [(2.0, 85.0)],
        dashed_path: [(2.0, 35.0)],
    }
    reversed_adapter = WandbSystemMetricsAdapter(run)

    #act
    reversed_buffer = adapter.collect()
    reversed_initial_buffer = reversed_adapter.collect()

    #assert
    assert reversed_buffer == {tag: value + 10.0 for tag, value in first.items()}
    assert reversed_initial_buffer == reversed_buffer


def test_adapter_skips_malformed_and_nonfinite_samples_without_hiding_valid_metrics():
    #arrange
    run = SimpleNamespace(
        _system_metrics={
            "cpu": [(1.0, 25.0), (2.0, float("nan")), (3.0, float("inf"))],
            "memory": [(float("nan"), 50.0), (float("inf"), 60.0)],
            "gpu.0.temp": [(1.0, "warm"), (2.0,), None, (3.0, 65.0)],
            "network.recv": "not a sample sequence",
            "network.sent": 123.0,
            "disk.in": [("no timestamp", 1.0)],
            "": [(1.0, 2.0)],
            None: [(1.0, 2.0)],
        }
    )
    adapter = WandbSystemMetricsAdapter(run)

    #act
    metrics = adapter.collect()
    repeated = adapter.collect()

    #assert
    assert metrics == {"system_cpu/cpu": 25.0, "system_gpu/gpu_0_temp": 65.0}
    assert repeated == {}


@pytest.mark.parametrize("buffer", [{}, None, []])
def test_adapter_handles_missing_or_empty_sdk_samples(buffer):
    #arrange
    adapter = WandbSystemMetricsAdapter(SimpleNamespace(_system_metrics=buffer))

    #act
    metrics = adapter.collect()

    #assert
    assert metrics == {}


def test_adapter_failure_is_reported_once_and_can_recover(caplog):
    #arrange
    class Run:
        available = False

        @property
        def _system_metrics(self):
            if not self.available:
                raise RuntimeError("SDK service is unavailable")
            return {"cpu": [(1.0, 50.0)]}

    run = Run()
    adapter = WandbSystemMetricsAdapter(run)

    #act
    first = adapter.collect()
    second = adapter.collect()

    #assert
    assert first == second == {}
    assert len(caplog.records) == 1
    assert "native system collection remains enabled" in caplog.text

    #arrange
    run.available = True

    #act
    recovered = adapter.collect()

    #assert
    assert recovered == {"system_cpu/cpu": 50.0}


def test_adapter_gracefully_handles_an_sdk_without_the_buffer_api(caplog):
    #arrange
    adapter = WandbSystemMetricsAdapter(SimpleNamespace())

    #act
    metrics = adapter.collect()

    #assert
    assert metrics == {}
    assert "Cannot read W&B's system metric buffer" in caplog.text
