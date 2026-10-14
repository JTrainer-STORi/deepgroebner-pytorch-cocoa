from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from deepgroebner_pytorch_cocoa import ppo_training
from deepgroebner_pytorch_cocoa.local_logging import SimpleRunWriter
from deepgroebner_pytorch_cocoa.system_metrics import SystemMetricsRecorder


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


def _scalar_sample_times(trainer):
    return [
        (record["step"], record["value"])
        for record in _read_jsonl(trainer.writer.scalar_path)
        if record["tag"] == "system/timestamp_unix_seconds"
    ]


def test_trainer_records_at_interval_boundary_and_forces_final_sample(make_trainer):
    trainer, clock, snapshot = make_trainer(times=[100.0, 109.999, 110.0, 110.001])
    rollout_features = torch.zeros((2, 14))
    rollout_features[:, :2] = torch.tensor([[7.0, 11.0], [9.0, 13.0]])

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
    assert [record["global_step"] for record in records] == [0, 2, 3]
    assert [record["timestamp_unix_seconds"] for record in records] == [
        100.0,
        110.0,
        110.001,
    ]
    assert [record["rollout_num_states"] for record in records] == [1.0, 2.0, 1.0]
    assert [record["rollout_basis_count_mean"] for record in records] == [2.0, 8.0, 2.0]
    assert [record["rollout_pair_count_mean"] for record in records] == [3.0, 12.0, 3.0]
    assert _scalar_sample_times(trainer) == [(0, 100.0), (2, 110.0), (3, 110.001)]
    assert clock.call_count == 4
    assert snapshot.call_count == 3


@pytest.mark.parametrize("interval", [0.0, -1.0])
def test_nonpositive_interval_records_every_update(make_trainer, interval):
    trainer, clock, snapshot = make_trainer(times=[100.0] * 3, interval=interval)
    for step in range(3):
        trainer.global_step = step
        trainer._record_system_metrics(elapsed_seconds=float(step))

    records = _read_jsonl(trainer.system_metrics_recorder.output_path)
    assert [record["global_step"] for record in records] == [0, 1, 2]
    assert _scalar_sample_times(trainer) == [(0, 100.0), (1, 100.0), (2, 100.0)]
    assert clock.call_count == snapshot.call_count == 3


@pytest.mark.parametrize("force", [False, True])
def test_disabled_metrics_do_not_collect_or_log(make_trainer, monkeypatch, force):
    trainer, clock, snapshot = make_trainer(times=[], enabled=False)
    summarize = Mock(side_effect=AssertionError("disabled metrics must not summarize"))
    monkeypatch.setattr(ppo_training, "summarize_state_feature_rollout", summarize)

    trainer._record_system_metrics(force=force, elapsed_seconds=0.0)

    assert not trainer.system_metrics_recorder.output_path.exists()
    assert _read_jsonl(trainer.writer.scalar_path) == []
    clock.assert_not_called()
    snapshot.assert_not_called()
    summarize.assert_not_called()


def test_direct_record_writes_every_call_and_preserves_supplied_timestamp(make_trainer):
    trainer, clock, snapshot = make_trainer(times=[200.0, 200.1], interval=10.0)
    recorder = trainer.system_metrics_recorder
    recorder.record(global_step=0, elapsed_seconds=0.0)
    recorder.record(global_step=1, elapsed_seconds=0.1)
    recorder.record(global_step=2, elapsed_seconds=0.2, sample_time=200.2)

    records = _read_jsonl(recorder.output_path)
    assert [record["global_step"] for record in records] == [0, 1, 2]
    assert [record["timestamp_unix_seconds"] for record in records] == [
        200.0,
        200.1,
        200.2,
    ]
    assert recorder.last_sample_time == 200.2
    assert clock.call_count == 2
    assert snapshot.call_count == 3
