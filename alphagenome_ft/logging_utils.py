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

    def log_epoch(
        self,
        epoch: int,
        metrics: dict,
        per_sample: list[dict] | None = None,
        histograms: dict[str, list[float]] | None = None,
        learning_rate: float | None = None,
        is_best: bool | None = None,
    ) -> None:
        """Append per-epoch rows, laid out like alphagenome-pytorch's
        ``TrainingLogger.log_epoch``: an aggregate row (``sample`` = NaN)
        followed by one row per biological sample (``--metrics-per-sample``).
        ``histograms`` go to W&B only, as in PyTorch."""
        if not self._is_main_process:
            return

        # Same leading columns as PyTorch: epoch, train_loss, val_loss,
        # learning_rate, is_best, timestamp, then the remaining metrics.
        rest = dict(metrics)
        agg = {"sample": float("nan"), "epoch": epoch}
        for key in ("train_loss", "val_loss"):
            if key in rest:
                agg[key] = rest.pop(key)
        if learning_rate is not None:
            agg["learning_rate"] = learning_rate
        if is_best is not None:
            agg["is_best"] = is_best
        agg["timestamp"] = datetime.now().isoformat()
        agg.update(rest)
        rows = [agg] + [{"sample": s, **row} for s, row in enumerate(per_sample or [])]

        if self._epoch_csv_path is not None:
            path = self._epoch_csv_path
            if self._epoch_fieldnames is None:
                # Resumed run: continue the existing file's columns.
                self._epoch_fieldnames = _read_csv_header(path)
            new_keys = [
                key for row in rows for key in row if key not in self._epoch_fieldnames
            ]
            new_keys = list(dict.fromkeys(new_keys))
            if new_keys:
                self._epoch_fieldnames.extend(new_keys)
                if path.exists() and path.stat().st_size > 0:
                    # Widen the header in place so earlier rows stay aligned.
                    _rewrite_csv_with_fieldnames(path, self._epoch_fieldnames)
            for row in rows:
                self._write_row(
                    self._epoch_csv_path, self._epoch_fieldnames, row, tolerant=True
                )

        if self.wandb is not None:
            payload = {f"epoch/{k}": v for k, v in metrics.items()}
            if learning_rate is not None:
                payload["epoch/learning_rate"] = learning_rate
            payload["epoch"] = epoch
            for k, values in (histograms or {}).items():
                payload[f"epoch/{k}"] = self.wandb.Histogram(values)
            self.wandb.log(payload)

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


def _read_csv_header(path: Path) -> list[str]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as f:
        return next(csv.reader(f), [])


def _rewrite_csv_with_fieldnames(path: Path, fieldnames: list[str]) -> None:
    with path.open(newline="") as f:
        existing = list(csv.DictReader(f))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing)
    tmp.replace(path)


__all__ = ["TrainingLogger"]
