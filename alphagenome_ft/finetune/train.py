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
) -> None:
    """Main training loop with JIT-compiled train/eval step functions."""
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
        print("  JIT-compiling step functions (first call will be slow)...")

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

    with model._device_context:
        for epoch in range(1, num_epochs + 1):
            if verbose:
                print(f'\n{"=" * 60}')
                print(f"Epoch {epoch}/{num_epochs}")
                print(f'{"=" * 60}')
            else:
                print(f"Epoch {epoch}/{num_epochs}")

            step = 0
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
                step += 1

                if max_train_steps and step >= max_train_steps:
                    break

                if verbose:
                    print(
                        f"  step {step:04d} | loss {loss_scalar:.4f}",
                        end="\r",
                        flush=True,
                    )

            train_loss_avg = float(np.mean(train_losses)) if train_losses else None
            if verbose:
                print()
            if train_loss_avg is not None:
                print(f"  Train loss: {train_loss_avg:.4f}")

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

            metric_label, metric_value = resolve_metric(best_metric, train_loss_avg, valid_metrics)
            if metric_value is not None and math.isfinite(metric_value):
                if is_improved(metric_value, best_value):
                    best_value = metric_value
                    epochs_since_improvement = 0
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

    if checkpoint_dir and not (checkpoint_dir / "last").exists():
        model.save_checkpoint(checkpoint_dir / "last", save_full_model=False)

    print(f'\n{"=" * 60}')
    print("Training complete!")
    print(f'{"=" * 60}')


__all__ = [
    "register_predefined_heads",
    "create_optimizer",
    "train",
]
