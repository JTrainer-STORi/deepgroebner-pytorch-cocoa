from __future__ import annotations

import csv
import json

from deepgroebner_pytorch_cocoa.local_logging import SimpleRunWriter


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
