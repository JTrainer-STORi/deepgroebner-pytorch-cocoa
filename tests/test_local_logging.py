from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from deepgroebner_pytorch_cocoa.config_dataclasses import Config
from deepgroebner_pytorch_cocoa.local_logging import SimpleRunWriter
from deepgroebner_pytorch_cocoa import training_utils


def test_simple_run_writer_writes_jsonl_csv_and_text(tmp_path):
    writer = SimpleRunWriter(str(tmp_path / "run"))
    writer.add_scalar("charts/SPS", 123, 10)
    writer.add_text("hyperparameters", "learning_rate: 0.001")
    writer.close()

    scalar_jsonl_path = tmp_path / "run" / "scalars.jsonl"
    scalar_csv_path = tmp_path / "run" / "scalars.csv"
    hyperparameters_path = tmp_path / "run" / "text" / "hyperparameters.md"

    records = [json.loads(line) for line in scalar_jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert records == [
        {
            "type": "scalar",
            "tag": "charts/SPS",
            "value": 123.0,
            "step": 10,
        }
    ]

    with scalar_csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "type": "scalar",
            "tag": "charts/SPS",
            "value": "123.0",
            "step": "10",
        }
    ]

    assert hyperparameters_path.read_text(encoding="utf-8") == "learning_rate: 0.001"


@pytest.mark.parametrize(
    "wandb_module",
    [None, SimpleNamespace(util=SimpleNamespace())],
    ids=["wandb-not-installed", "wandb-without-id-helper"],
)
def test_create_local_run_writer_without_wandb_id_helper(
    wandb_module, monkeypatch, tmp_path
):
    monkeypatch.setattr(training_utils, "wandb", wandb_module)
    cfg = Config()
    cfg.training.track = False
    cfg.training.output_directory = str(tmp_path)

    run_name, writer = training_utils.create_run_writer(cfg)
    try:
        assert Path(writer.log_dir) == tmp_path / "runs" / run_name
        assert run_name.rsplit("__", 1)[1]
        writer.add_scalar("charts/SPS", 42, 1)
    finally:
        writer.close()

    run_directory = tmp_path / "runs" / run_name
    assert (run_directory / "scalars.jsonl").is_file()
    assert (run_directory / "scalars.csv").is_file()
    assert (run_directory / "text" / "hyperparameters.md").is_file()
    assert json.loads((run_directory / "scalars.jsonl").read_text())["value"] == 42


def test_tracked_run_uses_same_id_for_wandb_and_local_logs(monkeypatch, tmp_path):
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
    cfg.training.output_directory = str(tmp_path)

    run_name, writer = training_utils.create_run_writer(cfg)
    try:
        (init_kwargs,) = init_calls
        run_id = run_name.rsplit("__", 1)[1]
        assert run_id
        assert init_kwargs["id"] == run_id
        assert init_kwargs["name"] == run_name
        assert init_kwargs["settings"].init_timeout == cfg.training.wandb_init_timeout
        assert Path(writer.log_dir) == tmp_path / "runs" / run_name
        writer.add_scalar("charts/SPS", 42, 1)
    finally:
        writer.close()

    assert log_calls == [({"charts/SPS": 42.0}, 1)]
    assert Path(wandb_module.summary["hyperparameters_path"]).is_file()
    assert json.loads(Path(writer.scalar_path).read_text())["value"] == 42
