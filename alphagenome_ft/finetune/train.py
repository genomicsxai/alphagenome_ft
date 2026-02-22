"""Training utilities for fine-tuning AlphaGenome models."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import jax
import numpy as np
import optax
from alphagenome.models import dna_model as ag_dna_model
from alphagenome_research.model import dna_model as research_dna_model

from alphagenome_ft import parameter_utils
from alphagenome_ft.custom_model import CustomAlphaGenomeModel
from alphagenome_ft.finetune.config import HeadSpec
from alphagenome_ft.finetune.data import BigWigDataModule, prepare_batch


def register_predefined_heads(head_specs: Sequence[HeadSpec]) -> None:
    """Register predefined heads from parsed head specs."""
    from alphagenome_ft import register_predefined_head

    for spec in head_specs:
        if spec.source != "predefined":
            continue
        if spec.config is None:
            raise ValueError(
                f'Predefined head "{spec.head_id}" missing config.'
            )
        register_predefined_head(
            spec.head_id,
            spec.config,
            metadata=spec.metadata,
        )


def _keypath_to_str(path_tuple: tuple) -> str:
    """Convert a JAX parameter key-path tuple to a slash-delimited string."""
    parts = []
    for key in path_tuple:
        if isinstance(key, parameter_utils.DictKey):
            parts.append(str(key.key))
        elif isinstance(key, parameter_utils.GetAttrKey):
            parts.append(str(key.name))
        elif isinstance(key, parameter_utils.SequenceKey):
            parts.append(str(key.idx))
        else:
            parts.append(str(key))
    return "/".join(parts)


def _is_trainable_head_path(path_str: str, trainable_heads: set[str]) -> bool:
    """Return True if a parameter path belongs to any requested trainable head."""
    for head_name in trainable_heads:
        if f"/head/{head_name}/" in path_str or path_str.startswith(f"head/{head_name}/"):
            return True
    return False


def _label_params_for_heads(params, trainable_heads: Sequence[str]):
    """Label model parameters as trainable head params vs frozen params."""
    head_set = {str(name) for name in trainable_heads}

    def label_fn(path, _value):
        path_str = _keypath_to_str(path)
        return "head" if _is_trainable_head_path(path_str, head_set) else "frozen"

    return jax.tree_util.tree_map_with_path(label_fn, params)


def create_optimizer(
    params,
    trainable_head_names: Sequence[str],
    learning_rate: float,
    weight_decay: float,
    heads_only: bool,
):
    """Create optimizer for full finetuning or heads-only finetuning."""
    if heads_only:
        head_set = {str(name) for name in trainable_head_names}
        head_paths = parameter_utils.get_head_parameter_paths(params)
        matched_paths = [path for path in head_paths if _is_trainable_head_path(path, head_set)]
        if not matched_paths:
            sample_paths = ", ".join(head_paths[:5]) if head_paths else "<none>"
            raise ValueError(
                "No trainable head parameters matched --heads-only filter. "
                f"Names tried: {sorted(head_set)}. "
                f"Head parameter sample: {sample_paths}"
            )
        param_labels = _label_params_for_heads(params, trainable_head_names)
        return optax.multi_transform(
            {
                "head": optax.adamw(learning_rate, weight_decay=weight_decay),
                "frozen": optax.set_to_zero(),
            },
            param_labels,
        )
    return optax.adamw(learning_rate, weight_decay=weight_decay)


def train(
    model: CustomAlphaGenomeModel,
    data_module: BigWigDataModule,
    head_specs: Sequence[HeadSpec],
    *,
    learning_rate: float,
    weight_decay: float,
    num_epochs: int,
    seed: int = 42,
    max_train_steps: int | None = None,
    heads_only: bool = False,
    checkpoint_dir: Path | None = None,
    organism: str = "HOMO_SAPIENS",
    best_metric: str = "valid_loss",
    best_metric_mode: str = "min",
    early_stopping_patience: int = 0,
    early_stopping_min_delta: float = 0.0,
    verbose: bool = False,
    use_wandb: bool = False,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_run_name: str | None = None,
    wandb_config: dict | None = None,
) -> None:
    """Run fine-tuning with JIT-compiled train/eval steps.

    Args:
        model: Initialized AlphaGenome model wrapper to fine-tune.
        data_module: Batch provider with train/valid intervals and BigWig targets.
        head_specs: Head definitions used to build losses and optimizer filters.
        learning_rate: Base AdamW learning rate.
        weight_decay: AdamW weight decay.
        num_epochs: Maximum number of epochs to run.
        seed: Base RNG seed used for per-epoch training shuffles.
        max_train_steps: Optional global cap on optimizer updates across all epochs.
        heads_only: If True, freeze backbone and optimize selected heads only.
        checkpoint_dir: Optional output directory for ``best``/``last`` checkpoints.
        organism: Organism enum name used for model organism indexing.
        best_metric: Metric name used for best-checkpoint and early-stopping tracking.
        best_metric_mode: Improvement direction for ``best_metric`` (``min`` or ``max``).
        early_stopping_patience: Stop after this many non-improving epochs (0 disables).
        early_stopping_min_delta: Minimum metric change required to count as improvement.
        verbose: If True, print per-step progress and extra diagnostics.
        use_wandb: If True, log metrics to Weights & Biases.
        wandb_project: Optional W&B project name override.
        wandb_entity: Optional W&B entity/team override.
        wandb_run_name: Optional W&B run-name override.
        wandb_config: Optional extra config keys to merge into W&B config.

    Notes:
        Total planned steps are computed before training from train-set size and
        batch settings as ``steps_per_epoch * num_epochs`` (or capped by
        ``max_train_steps`` when provided). Progress is reported with a global
        counter in ``current/total`` format.
    """
    train_intervals = list(data_module._intervals.get("train", ()))
    num_train_examples = len(train_intervals)
    if num_train_examples == 0:
        raise ValueError("No train intervals available for training.")

    if data_module._drop_last:
        steps_per_epoch = num_train_examples // data_module._batch_size
    else:
        steps_per_epoch = math.ceil(num_train_examples / data_module._batch_size)
    if steps_per_epoch == 0:
        raise ValueError(
            "Computed zero training steps per epoch. Check batch size, drop_last, and train intervals."
        )

    planned_steps = steps_per_epoch * num_epochs
    total_train_steps = (
        min(planned_steps, max_train_steps) if max_train_steps is not None else planned_steps
    )
    step_width = len(str(steps_per_epoch))

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if use_wandb:
        import wandb

        wb_config = {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "num_epochs": num_epochs,
            "batch_size": data_module._batch_size,
            "steps_per_epoch": steps_per_epoch,
            "total_train_steps": total_train_steps,
            "heads_only": heads_only,
            "organism": organism,
            "best_metric": best_metric,
            "best_metric_mode": best_metric_mode,
            "early_stopping_patience": early_stopping_patience,
            "seed": seed,
            **(wandb_config or {}),
        }
        wandb.init(
            project=wandb_project or "alphagenome-ft",
            entity=wandb_entity,
            name=wandb_run_name,
            config=wb_config,
        )

    head_names = [spec.head_id for spec in head_specs]
    if heads_only:
        model.freeze_backbone()

    optimizer = create_optimizer(
        model._params,
        head_names,
        learning_rate,
        weight_decay,
        heads_only,
    )
    opt_state = optimizer.init(model._params)

    organism_enum = getattr(ag_dna_model.Organism, organism)
    organism_index_value = research_dna_model.convert_to_organism_index(organism_enum)
    strand_reindexing = jax.device_put(
        model._metadata[organism_enum].strand_reindexing,
        model._device_context._device,
    )

    loss_fns = {name: model.create_loss_fn_for_head(name) for name in head_names}

    @jax.jit
    def train_step(params, state, current_opt_state, batch):
        def loss_fn(current_params):
            predictions = model._predict(
                current_params,
                state,
                batch["sequences"],
                batch["organism_index"],
                negative_strand_mask=batch["negative_strand_mask"],
                strand_reindexing=strand_reindexing,
            )
            total_loss = 0.0
            for head_name in head_names:
                head_loss_dict = loss_fns[head_name](
                    predictions[head_name],
                    {
                        "targets": batch[f"targets_{head_name}"],
                        "organism_index": batch["organism_index"],
                    },
                )
                total_loss = total_loss + head_loss_dict["loss"]
            return total_loss

        loss_value, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, current_opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss_value

    @jax.jit
    def eval_step(params, state, batch):
        predictions = model._predict(
            params,
            state,
            batch["sequences"],
            batch["organism_index"],
            negative_strand_mask=batch["negative_strand_mask"],
            strand_reindexing=strand_reindexing,
        )
        head_losses = {}
        for head_name in head_names:
            loss_dict = loss_fns[head_name](
                predictions[head_name],
                {
                    "targets": batch[f"targets_{head_name}"],
                    "organism_index": batch["organism_index"],
                },
            )
            head_losses[head_name] = loss_dict["loss"]
        return head_losses

    if verbose:
        print("JIT-compiling step functions (first call will be slow)...")

    def aggregate_valid_loss(metrics: Mapping[str, float]) -> float | None:
        return float(sum(metrics.values())) if metrics else None

    def resolve_metric(
        metric_name: str,
        train_loss: float | None,
        valid_metrics: Mapping[str, float] | None,
    ):
        if metric_name in {"train", "train_loss"}:
            return "train_loss", train_loss
        if metric_name in {"valid", "val", "valid_loss", "val_loss"}:
            return "valid_loss", aggregate_valid_loss(valid_metrics or {})
        if metric_name.startswith("valid:") or metric_name.startswith("valid/"):
            head = metric_name.split(":", 1)[-1].split("/", 1)[-1]
            return f"valid/{head}", (valid_metrics or {}).get(head)
        if valid_metrics and metric_name in valid_metrics:
            return f"valid/{metric_name}", valid_metrics[metric_name]
        return metric_name, None

    def is_improved(current: float, best: float | None) -> bool:
        if best is None:
            return True
        if best_metric_mode == "max":
            return current > best + early_stopping_min_delta
        return current < best - early_stopping_min_delta

    best_value: float | None = None
    epochs_since_improvement = 0
    global_step = 0

    print(
        "Train plan: "
        f"{num_train_examples} examples | "
        f"{steps_per_epoch} step(s)/epoch | "
        f"{num_epochs} epoch(s) | "
        f"total step(s) {total_train_steps}"
    )

    with model._device_context:
        stop_training = False
        for epoch in range(1, num_epochs + 1):
            if verbose:
                print(f'\n{"=" * 60}')
                print(f"Epoch {epoch}/{num_epochs}")
                print(f'{"=" * 60}')
            else:
                print(f"Epoch {epoch}/{num_epochs}")

            epoch_step = 0
            train_losses: list[float] = []
            for batch_np in data_module.iter_batches("train", seed=seed + epoch):
                batch = prepare_batch(batch_np, organism_index_value, head_names)
                model._params, opt_state, loss_value = train_step(
                    model._params,
                    model._state,
                    opt_state,
                    batch,
                )
                loss_scalar = float(loss_value)
                train_losses.append(loss_scalar)
                epoch_step += 1
                global_step += 1

                if verbose:
                    print(
                        # f"  step {global_step:0{step_width}d}/{total_train_steps:0{step_width}d}"
                        # f" | epoch_step {epoch_step:04d} | loss {loss_scalar:.4f}",
                        f"  step {epoch_step:0{step_width}d}/{steps_per_epoch:0{step_width}d}"
                        f" | loss {loss_scalar:.4f}",
                        end="\r",
                        flush=True,
                    )

                if use_wandb:
                    wandb.log(
                        {
                            "train/step_loss": loss_scalar,
                            "epoch": epoch,
                            "step": global_step,
                            # "epoch_step": epoch_step,
                        }
                    )

                if global_step >= total_train_steps:
                    stop_training = True
                    break

            train_loss_avg = float(np.mean(train_losses)) if train_losses else None
            if verbose:
                print()
            if train_loss_avg is not None:
                print(f"  Train loss: {train_loss_avg:.4f}")
                if use_wandb:
                    wandb.log({"train/epoch_loss": train_loss_avg, "epoch": epoch})

            valid_metrics: Mapping[str, float] | None = None
            if "valid" in data_module._intervals and len(data_module._intervals["valid"]) > 0:
                losses = {head: [] for head in head_names}
                for batch_np in data_module.iter_batches("valid"):
                    batch = prepare_batch(batch_np, organism_index_value, head_names)
                    head_losses = eval_step(model._params, model._state, batch)
                    for head_name in head_names:
                        losses[head_name].append(float(head_losses[head_name]))

                valid_metrics = {head: float(np.mean(values)) for head, values in losses.items() if values}
                print("  Validation metrics:", ", ".join(f"{k}={v:.4f}" for k, v in valid_metrics.items()))
                if use_wandb:
                    valid_log = {f"valid/{head}": v for head, v in valid_metrics.items()}
                    valid_log["valid/loss"] = float(sum(valid_metrics.values()))
                    valid_log["epoch"] = epoch
                    wandb.log(valid_log)

            metric_label, metric_value = resolve_metric(best_metric, train_loss_avg, valid_metrics)
            if metric_value is not None and math.isfinite(metric_value):
                if is_improved(metric_value, best_value):
                    best_value = metric_value
                    epochs_since_improvement = 0
                    if use_wandb:
                        wandb.log({"best/" + metric_label: metric_value, "epoch": epoch})
                    if checkpoint_dir:
                        print(
                            f"  Metric improved ({metric_label} = {metric_value:.4f}) "
                            " -- saving best checkpoint"
                        )
                        model.save_checkpoint(checkpoint_dir / "best", save_full_model=False)
                else:
                    epochs_since_improvement += 1
            else:
                print(f"  Best metric ({metric_label}): unavailable")

            if checkpoint_dir:
                model.save_checkpoint(checkpoint_dir / "last", save_full_model=False)

            if early_stopping_patience > 0 and epochs_since_improvement >= early_stopping_patience:
                print(f"\n  Early stopping: no improvement for {epochs_since_improvement} epoch(s)")
                break
            if stop_training:
                print(f"  Reached requested training steps: {global_step}/{total_train_steps}")
                break

    if checkpoint_dir and not (checkpoint_dir / "last").exists():
        model.save_checkpoint(checkpoint_dir / "last", save_full_model=False)

    print(f'\n{"=" * 60}')
    print("Training complete!")
    print(f'{"=" * 60}')

    if use_wandb:
        wandb.finish()


__all__ = [
    "register_predefined_heads",
    "create_optimizer",
    "train",
]
