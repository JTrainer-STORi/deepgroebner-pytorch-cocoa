from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from deepgroebner_pytorch_cocoa.config_dataclasses import Config
from deepgroebner_pytorch_cocoa.local_logging import SimpleRunWriter
from deepgroebner_pytorch_cocoa import local_logging
from deepgroebner_pytorch_cocoa import training_utils


def test_simple_run_writer_writes_jsonl_csv_and_text(tmp_path):
    #arrange
    writer = SimpleRunWriter(str(tmp_path / "run"))
    scalar_jsonl_path = tmp_path / "run" / "scalars.jsonl"
    scalar_csv_path = tmp_path / "run" / "scalars.csv"
    hyperparameters_path = tmp_path / "run" / "text" / "hyperparameters.md"

    #act
    writer.add_scalar("performance/steps_per_second", 123, 10)
    writer.add_text("hyperparameters", "learning_rate: 0.001")
    writer.close()
    records = [json.loads(line) for line in scalar_jsonl_path.read_text(encoding="utf-8").splitlines()]
    with scalar_csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    hyperparameters = hyperparameters_path.read_text(encoding="utf-8")

    #assert
    assert records == [
        {
            "type": "scalar",
            "tag": "performance/steps_per_second",
            "value": 123.0,
            "step": 10,
        }
    ]

    assert rows == [
        {
            "type": "scalar",
            "tag": "performance/steps_per_second",
            "value": "123.0",
            "step": "10",
        }
    ]

    assert hyperparameters == "learning_rate: 0.001"


@pytest.mark.parametrize(
    "wandb_module",
    [None, SimpleNamespace(util=SimpleNamespace())],
    ids=["wandb-not-installed", "wandb-without-id-helper"],
)
def test_create_local_run_writer_without_wandb_id_helper(
    wandb_module, monkeypatch, tmp_path
):
    #arrange
    monkeypatch.setattr(training_utils, "wandb", wandb_module)
    cfg = Config()
    cfg.training.track = False
    cfg.training.output_directory = str(tmp_path)

    #act
    run_name, writer = training_utils.create_run_writer(cfg)
    try:
        writer.add_scalar("performance/steps_per_second", 42, 1)
    finally:
        writer.close()

    run_directory = tmp_path / "runs" / run_name
    scalar_record = json.loads((run_directory / "scalars.jsonl").read_text())

    #assert
    assert Path(writer.log_dir) == tmp_path / "runs" / run_name
    assert run_name.rsplit("__", 1)[1]
    assert (run_directory / "scalars.jsonl").is_file()
    assert (run_directory / "scalars.csv").is_file()
    assert (run_directory / "text" / "hyperparameters.md").is_file()
    assert scalar_record["value"] == 42


@pytest.mark.parametrize("configured_output", [False, True])
@pytest.mark.parametrize("group", [None, "sweep-001"])
def test_tracked_run_uses_same_id_and_output_root_for_wandb_and_local_logs(
    configured_output, group, monkeypatch, tmp_path
):
    #arrange
    monkeypatch.chdir(tmp_path)
    init_calls = []
    log_calls = []
    wandb_module = SimpleNamespace(
        util=SimpleNamespace(),
        run=None,
        summary={},
        Settings=lambda **kwargs: SimpleNamespace(**kwargs),
        log=lambda values, step: log_calls.append((values, step)),
    )

    def init(**kwargs):
        init_calls.append(kwargs)
        wandb_module.run = SimpleNamespace(id=kwargs["id"])
        return wandb_module.run

    wandb_module.init = init
    monkeypatch.setattr(training_utils, "wandb", wandb_module)
    cfg = Config()
    cfg.training.track = True
    cfg.training.output_directory = (
        str(tmp_path / "configured-output") if configured_output else None
    )
    cfg.training.group = group
    original_config = asdict(cfg)
    output_root = tmp_path / "configured-output" if configured_output else tmp_path
    if group is not None:
        output_root /= group
    expected_metrics = {
        "performance/steps_per_second": 42.0,
        "policy_function/loss": 1.25,
        "evaluation_greedy/selection_metric": -3.0,
    }

    #act
    run_name, writer = training_utils.create_run_writer(cfg)
    try:
        for tag, value in expected_metrics.items():
            writer.add_scalar(tag, value, 1)
    finally:
        writer.close()

    (init_kwargs,) = init_calls
    run_id = run_name.rsplit("__", 1)[1]
    records = [json.loads(line) for line in Path(writer.scalar_path).read_text().splitlines()]

    #assert
    assert run_id
    assert init_kwargs["id"] == run_id
    assert init_kwargs["name"] == run_name
    assert init_kwargs["settings"].init_timeout == cfg.training.wandb_init_timeout
    assert init_kwargs["settings"]._stats_buffer_size == 1
    assert Path(writer.log_dir).resolve() == output_root / "runs" / run_name
    if configured_output or group is not None:
        assert Path(init_kwargs["dir"]).resolve() == output_root
        assert output_root.is_dir()
    else:
        assert "dir" not in init_kwargs
    if group is not None:
        assert init_kwargs["group"] == group
    else:
        assert "group" not in init_kwargs
    assert log_calls == [({tag: value}, 1) for tag, value in expected_metrics.items()]
    assert Path(wandb_module.summary["hyperparameters_path"]).is_file()
    assert {record["tag"]: record["value"] for record in records} == expected_metrics
    assert Path(writer.scalar_csv_path).is_file()
    assert asdict(cfg) == original_config


def test_native_system_metrics_are_sampled_once_per_step_and_interval_then_flushed(
    monkeypatch, tmp_path
):
    #arrange
    adapter = Mock()
    adapter.collect.side_effect = [
        {"system_cpu/cpu": 20.0},
        {"system_gpu/gpu_0_gpu": 75.0},
        {"system_general/network_sent": 128.0},
    ]
    log_calls = []
    wandb_module = SimpleNamespace(
        run=object(), log=lambda values, step: log_calls.append((values, step))
    )
    clock = Mock(side_effect=[0.0, 0.0, 5.0, 10.0, 10.0, 10.0])
    monkeypatch.setattr(local_logging.time, "monotonic", clock)
    writer = SimpleRunWriter(
        str(tmp_path / "run"),
        mirror_to_wandb=True,
        wandb_module=wandb_module,
        system_metrics_adapter=adapter,
        system_metrics_interval_seconds=10,
    )

    #act
    writer.add_scalar("policy_function/loss", 1, 4)
    writer.add_scalar("value_function/loss", 2, 4)
    writer.add_scalar("policy_function/loss", 3, 8)
    writer.add_scalar("policy_function/loss", 4, 12)
    writer.add_scalar("value_function/loss", 5, 12)
    writer.close()
    writer.close()
    records = [json.loads(line) for line in Path(writer.scalar_path).read_text().splitlines()]
    with Path(writer.scalar_csv_path).open(newline="") as handle:
        csv_records = list(csv.DictReader(handle))
    system_records = [record for record in records if record["tag"].startswith("system_")]

    #assert
    assert adapter.collect.call_count == 3
    assert [(record["tag"], record["value"], record["step"]) for record in system_records] == [
        ("system_cpu/cpu", 20.0, 4),
        ("system_gpu/gpu_0_gpu", 75.0, 12),
        ("system_general/network_sent", 128.0, 12),
    ]
    assert log_calls == [({record["tag"]: record["value"]}, record["step"]) for record in records]
    assert len(csv_records) == len(records)
    assert writer._scalar_handle.closed
    assert writer._scalar_csv_handle.closed


@pytest.mark.parametrize("mirror,active", [(False, True), (True, False)])
def test_system_metric_adapter_is_not_polled_without_an_active_tracked_run(
    mirror, active, tmp_path
):
    #arrange
    adapter = Mock()
    writer = SimpleRunWriter(
        str(tmp_path / "run"),
        mirror_to_wandb=mirror,
        wandb_module=SimpleNamespace(run=object() if active else None),
        system_metrics_adapter=adapter,
    )

    #act
    writer.add_scalar("policy_function/loss", 1, 4)
    writer.close()

    #assert
    adapter.collect.assert_not_called()
