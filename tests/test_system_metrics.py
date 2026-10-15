from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from deepgroebner_pytorch_cocoa import ppo_training, system_metrics
from deepgroebner_pytorch_cocoa.local_logging import SimpleRunWriter
from deepgroebner_pytorch_cocoa.system_metrics import SystemMetricsRecorder
from deepgroebner_pytorch_cocoa.training_utils import log_system_metrics


@pytest.fixture
def make_trainer(tmp_path):
    with ExitStack() as resources:

        def make(*, times, interval=10.0, enabled=True):
            clock = Mock(side_effect=times)
            snapshot = Mock(return_value={"rss_bytes": 4096})
            recorder = SystemMetricsRecorder(
                tmp_path / "system_metrics.jsonl",
                enabled=enabled,
                sample_interval_seconds=interval,
                clock_fn=clock,
                snapshot_fn=snapshot,
            )
            resources.callback(recorder.close)
            writer = SimpleRunWriter(str(tmp_path / "run"))
            resources.callback(writer.close)
            trainer = ppo_training.PPOTrainer.__new__(ppo_training.PPOTrainer)
            trainer.system_metrics_recorder = recorder
            trainer.writer = writer
            trainer.global_step = 0
            trainer.policy_updates = 0
            trainer.start_time = 0.0
            trainer.pq_budget = None
            features = torch.zeros((1, 14))
            features[0, :2] = torch.tensor([2.0, 3.0])
            trainer.inputs = SimpleNamespace(state_features=features)
            return trainer, clock, snapshot

        yield make


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def _scalar_elapsed_times(trainer):
    return [
        (record["step"], record["value"])
        for record in _read_jsonl(trainer.writer.scalar_path)
        if record["tag"] == "performance/elapsed_seconds"
    ]


def test_trainer_records_at_interval_boundary_and_forces_final_sample(make_trainer):
    #arrange
    trainer, clock, snapshot = make_trainer(times=[100.0, 109.999, 110.0, 110.001])
    rollout_features = torch.zeros((2, 14))
    rollout_features[:, :2] = torch.tensor([[7.0, 11.0], [9.0, 13.0]])

    #act
    trainer._record_system_metrics(force=True, elapsed_seconds=0.0)
    for step in (1, 2):
        trainer.global_step = step
        trainer.policy_updates = step
        trainer._record_system_metrics(
            elapsed_seconds=float(step), state_features=rollout_features
        )
    trainer.global_step = 3
    trainer._record_system_metrics(force=True, elapsed_seconds=3.0)

    records = _read_jsonl(trainer.system_metrics_recorder.output_path)
    elapsed_times = _scalar_elapsed_times(trainer)

    #assert
    assert [record["global_step"] for record in records] == [0, 2, 3]
    assert [record["timestamp_unix_seconds"] for record in records] == [
        100.0,
        110.0,
        110.001,
    ]
    assert [record["rollout_num_states"] for record in records] == [1.0, 2.0, 1.0]
    assert [record["rollout_basis_count_mean"] for record in records] == [2.0, 8.0, 2.0]
    assert [record["rollout_pair_count_mean"] for record in records] == [3.0, 12.0, 3.0]
    assert elapsed_times == [(0, 0.0), (2, 2.0), (3, 3.0)]
    assert clock.call_count == 4
    assert snapshot.call_count == 3


@pytest.mark.parametrize("interval", [0.0, -1.0])
def test_nonpositive_interval_records_every_update(make_trainer, interval):
    #arrange
    trainer, clock, snapshot = make_trainer(times=[100.0] * 3, interval=interval)

    #act
    for step in range(3):
        trainer.global_step = step
        trainer._record_system_metrics(elapsed_seconds=float(step))

    records = _read_jsonl(trainer.system_metrics_recorder.output_path)
    elapsed_times = _scalar_elapsed_times(trainer)

    #assert
    assert [record["global_step"] for record in records] == [0, 1, 2]
    assert elapsed_times == [(0, 0.0), (1, 1.0), (2, 2.0)]
    assert clock.call_count == snapshot.call_count == 3


@pytest.mark.parametrize("force", [False, True])
def test_disabled_metrics_do_not_collect_or_log(make_trainer, monkeypatch, force):
    #arrange
    trainer, clock, snapshot = make_trainer(times=[], enabled=False)
    summarize = Mock(side_effect=AssertionError("disabled metrics must not summarize"))
    monkeypatch.setattr(ppo_training, "summarize_state_feature_rollout", summarize)

    #act
    trainer._record_system_metrics(force=force, elapsed_seconds=0.0)
    scalar_records = _read_jsonl(trainer.writer.scalar_path)

    #assert
    assert not trainer.system_metrics_recorder.output_path.exists()
    assert scalar_records == []
    clock.assert_not_called()
    snapshot.assert_not_called()
    summarize.assert_not_called()


def test_direct_record_writes_every_call_and_preserves_supplied_timestamp(make_trainer):
    #arrange
    trainer, clock, snapshot = make_trainer(times=[200.0, 200.1], interval=10.0)
    recorder = trainer.system_metrics_recorder

    #act
    recorder.record(global_step=0, elapsed_seconds=0.0)
    recorder.record(global_step=1, elapsed_seconds=0.1)
    recorder.record(global_step=2, elapsed_seconds=0.2, sample_time=200.2)

    records = _read_jsonl(recorder.output_path)

    #assert
    assert [record["global_step"] for record in records] == [0, 1, 2]
    assert [record["timestamp_unix_seconds"] for record in records] == [
        200.0,
        200.1,
        200.2,
    ]
    assert recorder.last_sample_time == 200.2
    assert clock.call_count == 2
    assert snapshot.call_count == 3


def test_snapshot_measures_main_process_cpu_and_preserves_memory_units(monkeypatch):
    #arrange
    monkeypatch.setattr(
        system_metrics, "_read_proc_status_bytes", lambda: {"rss_bytes": 2 ** 30}
    )
    monkeypatch.setattr(
        system_metrics.os,
        "times",
        lambda: SimpleNamespace(user=2.5, system=0.5, children_user=50.0, children_system=10.0),
    )

    #act
    snapshot = system_metrics.collect_system_metrics_snapshot(torch.device("cpu"))

    #assert
    assert snapshot["rss_bytes"] == 2 ** 30
    assert snapshot["rss_gib"] == 1.0
    assert snapshot["process_cpu_user_seconds"] == 2.5
    assert snapshot["process_cpu_system_seconds"] == 0.5
    assert snapshot["process_cpu_total_seconds"] == 3.0
    assert snapshot["cuda_enabled"] is False
    assert "process_cpu_percent" not in snapshot  # Requires two timed samples.


def test_cuda_snapshot_without_hardware_preserves_allocated_and_reserved_memory(monkeypatch):
    #arrange
    monkeypatch.setattr(system_metrics, "_read_proc_status_bytes", lambda: {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for name, gib in (
        ("memory_allocated", 1),
        ("memory_reserved", 2),
        ("max_memory_allocated", 3),
        ("max_memory_reserved", 4),
    ):
        monkeypatch.setattr(torch.cuda, name, lambda index, amount=gib: amount * 2 ** 30)

    #act
    snapshot = system_metrics.collect_system_metrics_snapshot(torch.device("cuda:0"))

    #assert
    assert snapshot["cuda_enabled"] is True
    assert snapshot["cuda_memory_allocated_gib"] == 1.0
    assert snapshot["cuda_memory_reserved_gib"] == 2.0
    assert snapshot["cuda_max_memory_allocated_gib"] == 3.0
    assert snapshot["cuda_max_memory_reserved_gib"] == 4.0
    assert snapshot["cuda_max_memory_reserved_bytes"] == 4 * 2 ** 30


@pytest.mark.parametrize(
    "wall_times,cpu_times,expected_percent",
    [
        ([100.0, 110.0], [2.0, 5.0], 30.0),
        ([100.0, 110.0], [2.0, 22.0], 200.0),
        ([100.0, 100.0], [2.0, 5.0], None),
        ([110.0, 100.0], [2.0, 5.0], None),
        ([100.0, 110.0], [5.0, 2.0], None),
    ],
)
def test_cpu_utilization_uses_sample_deltas(tmp_path, wall_times, cpu_times, expected_percent):
    #arrange
    recorder = SystemMetricsRecorder(
        tmp_path / "system_metrics.jsonl",
        clock_fn=Mock(side_effect=wall_times),
        snapshot_fn=Mock(side_effect=[{"process_cpu_total_seconds": t} for t in cpu_times]),
    )
    #act
    try:
        first = recorder.record(global_step=0, elapsed_seconds=0.0)
        second = recorder.record(global_step=4, elapsed_seconds=10.0)
    finally:
        recorder.close()
    records = _read_jsonl(recorder.output_path)

    #assert
    assert "process_cpu_percent" not in first
    if expected_percent is None:
        assert "process_cpu_percent" not in second
    else:
        assert second["process_cpu_percent"] == expected_percent
    assert records == [first, second]


def test_system_scalars_route_to_hardware_performance_and_environment_sections(tmp_path):
    #arrange
    sample = {
        "timestamp_utc": "2026-10-15T12:00:00Z",
        "timestamp_unix_seconds": 100.0,
        "global_step": 4,
        "policy_updates": 1,
        "elapsed_seconds": 10.0,
        "process_cpu_total_seconds": 3.0,
        "process_cpu_percent": 30.0,
        "rss_bytes": 2 ** 30,
        "rss_gib": 1.0,
        "rss_hwm_gib": 2.0,
        "vms_gib": None,
        "swap_gib": float("nan"),
        "cuda_enabled": True,
        "cuda_memory_allocated_bytes": 2 ** 30,
        "cuda_memory_allocated_gib": 1.0,
        "cuda_max_memory_reserved_gib": 4.0,
        "rollout_pair_count_mean": 12.0,
        "rollout_pq_binding_ever": False,
    }
    writer = SimpleRunWriter(str(tmp_path / "run"))

    #act
    try:
        log_system_metrics(writer, sample, global_step=4)
    finally:
        writer.close()

    records = _read_jsonl(writer.scalar_path)

    #assert
    assert {record["step"] for record in records} == {4}
    assert {record["tag"]: record["value"] for record in records} == {
        "performance/elapsed_seconds": 10.0,
        "system_cpu/process_total_seconds": 3.0,
        "system_cpu/process_utilization_percent": 30.0,
        "system_memory/process_rss_gib": 1.0,
        "system_memory/process_peak_rss_gib": 2.0,
        "system_gpu/cuda_enabled": 1.0,
        "system_gpu/memory_allocated_gib": 1.0,
        "system_gpu/peak_memory_reserved_gib": 4.0,
        "evaluation_training/rollout_pair_count_mean": 12.0,
        "evaluation_training/rollout_pq_binding_ever": 0.0,
    }
