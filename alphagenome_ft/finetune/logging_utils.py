"""Lightweight per-head loss logging for JAX finetuning runs.

Mirrors the behavior of alphagenome-pytorch's
``extensions.finetuning.logging.TrainingLogger``: always writes append-only
CSVs (``training_log.csv`` for per-step metrics, ``epoch_log.csv`` for
per-epoch metrics) to the run's checkpoint directory, and optionally mirrors
the same metrics to Weights & Biases.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import jax


class TrainingLogger:
    """Writes per-step and per-epoch metrics to CSV, and optionally W&B."""

    def __init__(
        self,
        log_dir: Path | None,
        *,
        use_wandb: bool = False,
        wandb_project: str | None = None,
        wandb_entity: str | None = None,
        run_name: str | None = None,
        config: dict | None = None,
    ) -> None:
        self._is_main_process = jax.process_index() == 0
        self.wandb = None
        self._step_fieldnames: list[str] | None = None
        self._epoch_fieldnames: list[str] | None = None
        self._step_csv_path: Path | None = None
        self._epoch_csv_path: Path | None = None

        if not self._is_main_process:
            return

        if log_dir is not None:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            self._step_csv_path = log_dir / "training_log.csv"
            self._epoch_csv_path = log_dir / "epoch_log.csv"

        if use_wandb:
            import wandb

            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=run_name,
                config=config,
            )
            self.wandb = wandb

    def log_step(self, metrics: dict) -> None:
        """Append one row of per-step metrics (e.g. per-head training loss)."""
        if not self._is_main_process:
            return

        metrics = {**metrics, "timestamp": datetime.now().isoformat()}

        if self._step_csv_path is not None:
            if self._step_fieldnames is None:
                other_keys = sorted(k for k in metrics if k not in ("step", "epoch", "timestamp"))
                self._step_fieldnames = ["step", "epoch", "timestamp", *other_keys]
            self._write_row(self._step_csv_path, self._step_fieldnames, metrics, tolerant=True)

        if self.wandb is not None:
            self.wandb.log(metrics, step=metrics.get("step"))

    def log_epoch(self, epoch: int, metrics: dict) -> None:
        """Append one row of per-epoch metrics (train/val loss, per head)."""
        if not self._is_main_process:
            return

        metrics = {"epoch": epoch, **metrics, "timestamp": datetime.now().isoformat()}

        if self._epoch_csv_path is not None:
            if self._epoch_fieldnames is None:
                self._epoch_fieldnames = list(metrics.keys())
            else:
                for key in metrics:
                    if key not in self._epoch_fieldnames:
                        self._epoch_fieldnames.append(key)
            self._write_row(
                self._epoch_csv_path, self._epoch_fieldnames, metrics, tolerant=True
            )

        if self.wandb is not None:
            self.wandb.log({f"epoch/{k}": v for k, v in metrics.items()})

    def _write_row(
        self, path: Path, fieldnames: list[str], row: dict, tolerant: bool = False
    ) -> None:
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                restval="" if tolerant else None,
                extrasaction="ignore" if tolerant else "raise",
            )
            if write_header:
                writer.writeheader()
            writer.writerow(row)
            f.flush()

    def finish(self) -> None:
        if self.wandb is not None:
            self.wandb.finish()


__all__ = ["TrainingLogger"]
