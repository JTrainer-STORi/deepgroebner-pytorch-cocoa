"""A simple object for writing run metrics, diagnostics etc."""

from __future__ import annotations

import csv
import json
import os
import re
import time


class SimpleRunWriter:
    """Lightweight local metric writer that avoids TensorBoard/TensorFlow imports."""

    def __init__(
        self,
        log_dir: str,
        *,
        mirror_to_wandb: bool = False,
        wandb_module=None,
        system_metrics_adapter=None,
        system_metrics_interval_seconds: float = 10.0,
    ):
        self.log_dir = log_dir
        self.mirror_to_wandb = mirror_to_wandb
        self.wandb_module = wandb_module
        self.system_metrics_adapter = system_metrics_adapter
        self.system_metrics_interval_seconds = max(
            0.0, float(system_metrics_interval_seconds)
        )
        self._last_system_metrics_time = None
        self._last_system_metrics_step = None
        self._last_global_step = None
        self._closed = False
        os.makedirs(self.log_dir, exist_ok=True)

        self.scalar_path = os.path.join(self.log_dir, "scalars.jsonl")
        self.scalar_csv_path = os.path.join(self.log_dir, "scalars.csv")
        self.text_dir = os.path.join(self.log_dir, "text")
        os.makedirs(self.text_dir, exist_ok=True)

        self._scalar_handle = open(self.scalar_path, "a", encoding="utf-8")
        self._scalar_csv_handle = open(self.scalar_csv_path, "a", encoding="utf-8", newline="")
        self._scalar_csv_writer = csv.DictWriter(
            self._scalar_csv_handle,
            fieldnames=["type", "tag", "value", "step"],
        )
        if self._scalar_csv_handle.tell() == 0:
            self._scalar_csv_writer.writeheader()
            self._scalar_csv_handle.flush()

    def _wandb_run_active(self) -> bool:
        return self.wandb_module is not None and self.wandb_module.run is not None

    def add_scalar(self, tag: str, scalar_value, global_step: int):
        self._last_global_step = int(global_step)
        self._write_scalar(tag, scalar_value, global_step)
        self._collect_wandb_system_metrics(global_step)

    def _write_scalar(self, tag: str, scalar_value, global_step: int):
        scalar_float = float(scalar_value)
        record = {
            "type": "scalar",
            "tag": tag,
            "value": scalar_float,
            "step": int(global_step),
        }
        self._scalar_handle.write(json.dumps(record) + "\n")
        self._scalar_handle.flush()

        self._scalar_csv_writer.writerow(record)
        self._scalar_csv_handle.flush()

        if self.mirror_to_wandb and self._wandb_run_active():
            self.wandb_module.log({tag: scalar_float}, step=int(global_step))

    def _collect_wandb_system_metrics(self, global_step: int, *, force=False):
        if (
            self.system_metrics_adapter is None
            or not self.mirror_to_wandb
            or not self._wandb_run_active()
        ):
            return
        now = time.monotonic()
        if not force and (
            self._last_system_metrics_step == global_step
            or (
                self._last_system_metrics_time is not None
                and now - self._last_system_metrics_time
                < self.system_metrics_interval_seconds
            )
        ):
            return
        self._last_system_metrics_time = now
        self._last_system_metrics_step = int(global_step)
        for tag, value in self.system_metrics_adapter.collect().items():
            self._write_scalar(tag, value, global_step)

    def add_text(self, tag: str, text_string: str, global_step: int | None = None):
        safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag).strip("_") or "text"
        suffix = f"__step_{int(global_step)}" if global_step is not None else ""
        out_path = os.path.join(self.text_dir, f"{safe_tag}{suffix}.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text_string)

        if self.mirror_to_wandb and self._wandb_run_active():
            self.wandb_module.summary[f"{tag}_path"] = out_path

    def close(self):
        if self._closed:
            return
        try:
            if self._last_global_step is not None:
                self._collect_wandb_system_metrics(self._last_global_step, force=True)
        finally:
            self._scalar_handle.close()
            self._scalar_csv_handle.close()
            self._closed = True


def make_summary_writer(
    log_dir: str,
    *,
    mirror_to_wandb: bool = False,
    wandb_module=None,
    system_metrics_adapter=None,
    system_metrics_interval_seconds: float = 10.0,
):
    return SimpleRunWriter(
        log_dir,
        mirror_to_wandb=mirror_to_wandb,
        wandb_module=wandb_module,
        system_metrics_adapter=system_metrics_adapter,
        system_metrics_interval_seconds=system_metrics_interval_seconds,
    )
