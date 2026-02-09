#!/usr/bin/env python3
"""Fine-tune AlphaGenome on custom BED/BigWig datasets.

This script fine-tunes AlphaGenome models on genomic data:
1. Loads genomic intervals from a BED file (with train/valid/test splits)
2. Reads signal targets from BigWig tracks defined in the targets config
3. Fine-tunes prediction heads (custom or predefined) while optionally freezing the backbone

Example usage:
    # Fine-tune with frozen backbone (recommended)
    python finetune.py \\
        --fasta hg38.fa \\
        --bed regions.bed \\
        --targets-config targets.yaml \\
        --heads-only \\
        --num-epochs 10 \\
        --checkpoint-dir ./checkpoints

    # Resume from checkpoint
    python finetune.py \\
        --fasta hg38.fa \\
        --bed regions.bed \\
        --targets-config targets.yaml \\
        --resume-from ./checkpoints/last

Requirements: alphagenome_research, alphagenome_ft, pyBigWig, optax
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence

import jax
import numpy as np
import optax
from alphagenome.models import dna_model as ag_dna_model
from alphagenome_research.model import dna_model as research_dna_model

from alphagenome_ft import (
    register_predefined_head,
    create_model_with_heads,
    load_checkpoint,
    parameter_utils,
)
from alphagenome_ft.custom_model import CustomAlphaGenomeModel
from alphagenome_ft.finetune.config import HeadSpec, build_head_specs, validate_heads
from alphagenome_ft.finetune.data import BigWigDataModule, load_intervals, prepare_batch


# ============================================================================
# Argument Parsing
# ============================================================================

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # === Data ===
    data = parser.add_argument_group('Data')
    data.add_argument(
        '--fasta', type=Path, required=True,
        help='Reference FASTA file for sequence extraction'
    )
    data.add_argument(
        '--bed', type=Path, required=True,
        help='BED file with 4 columns: chrom, start, end, split (train/valid/test)'
    )
    data.add_argument(
        '--targets-config', type=Path, required=True,
        help=(
            'YAML/JSON file defining prediction heads and target tracks '
            '(use "targets" with optional metadata, or legacy "bigwigs").'
        )
    )
    data.add_argument(
        '--window-size', type=int,
        help='Override BED interval size; centers each region to this length (bp)'
    )
    data.add_argument(
        '--limit-train', type=int,
        help='Limit training windows (for quick testing)'
    )
    data.add_argument(
        '--limit-valid', type=int,
        help='Limit validation windows'
    )
    data.add_argument(
        '--limit-test', type=int,
        help='Limit test windows'
    )

    # === Training ===
    train = parser.add_argument_group('Training')
    train.add_argument(
        '--batch-size', type=int, default=1,
        help='Batch size (AlphaGenome typically uses 1 for 1Mbp windows)'
    )
    train.add_argument(
        '--num-epochs', type=int, default=1,
        help='Number of training epochs'
    )
    train.add_argument(
        '--learning-rate', type=float, default=3e-4,
        help='Optimizer learning rate'
    )
    train.add_argument(
        '--weight-decay', type=float, default=1e-2,
        help='AdamW weight decay coefficient'
    )
    train.add_argument(
        '--shuffle', action='store_true',
        help='Shuffle training data each epoch'
    )
    train.add_argument(
        '--drop-last', action='store_true',
        help='Drop incomplete final batch'
    )
    train.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    train.add_argument(
        '--max-train-steps', type=int,
        help='Cap training steps per epoch (for debugging)'
    )
    train.add_argument(
        '--heads-only', action='store_true',
        help='Freeze AlphaGenome backbone, only train heads (recommended)'
    )

    # === Model ===
    model = parser.add_argument_group('Model')
    model.add_argument(
        '--organism', default='HOMO_SAPIENS',
        help='Organism enum name (e.g., HOMO_SAPIENS, MUS_MUSCULUS)'
    )
    model.add_argument(
        '--model-version', default='all_folds',
        help='AlphaGenome checkpoint identifier'
    )
    model.add_argument(
        '--checkpoint-path', type=Path,
        help='Local AlphaGenome checkpoint directory to initialize weights from'
    )

    # === Checkpointing & Early Stopping ===
    ckpt = parser.add_argument_group('Checkpointing')
    ckpt.add_argument(
        '--checkpoint-dir', type=Path,
        help='Directory to save checkpoints (creates best/ and last/ subdirs)'
    )
    ckpt.add_argument(
        '--resume-from', type=Path,
        help='Resume training from checkpoint directory'
    )
    ckpt.add_argument(
        '--best-metric', default='valid_loss',
        help='Metric to track for best checkpoint (valid_loss, train_loss, or head name)'
    )
    ckpt.add_argument(
        '--best-metric-mode', choices=['min', 'max'], default='min',
        help='Whether lower (min) or higher (max) metric values are better'
    )
    ckpt.add_argument(
        '--early-stopping-patience', type=int, default=0,
        help='Stop after N epochs without improvement (0 = disabled)'
    )
    ckpt.add_argument(
        '--early-stopping-min-delta', type=float, default=0.0,
        help='Minimum change to qualify as improvement'
    )

    return parser.parse_args(argv)


# ============================================================================
# Optimizer Setup
# ============================================================================

def _keypath_to_str(path_tuple: tuple) -> str:
    """Convert JAX parameter path to readable string."""
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
    return '/'.join(parts)


def _is_trainable_head_path(path_str: str, trainable_heads: set[str]) -> bool:
    """Return True when a parameter path belongs to one of the trainable heads."""
    for head_name in trainable_heads:
        if f'/head/{head_name}/' in path_str or path_str.startswith(f'head/{head_name}/'):
            return True
    return False


def _label_params_for_heads(params, trainable_heads: Sequence[str]):
    """Label parameters as 'head' or 'frozen' for selective optimization."""
    head_set = {str(name) for name in trainable_heads}

    def label_fn(path, _value):
        path_str = _keypath_to_str(path)
        return 'head' if _is_trainable_head_path(path_str, head_set) else 'frozen'

    return jax.tree_util.tree_map_with_path(label_fn, params)


def create_optimizer(
    params,
    trainable_head_names: Sequence[str],
    learning_rate: float,
    weight_decay: float,
    heads_only: bool,
):
    """Create optimizer that optionally freezes the backbone."""
    if heads_only:
        head_set = {str(name) for name in trainable_head_names}
        head_paths = parameter_utils.get_head_parameter_paths(params)
        matched_paths = [path for path in head_paths if _is_trainable_head_path(path, head_set)]
        if not matched_paths:
            sample_paths = ', '.join(head_paths[:5]) if head_paths else '<none>'
            raise ValueError(
                'No trainable head parameters matched --heads-only filter. '
                f'Names tried: {sorted(head_set)}. '
                f'Head parameter sample: {sample_paths}'
            )
        param_labels = _label_params_for_heads(params, trainable_head_names)
        return optax.multi_transform(
            {
                'head': optax.adamw(learning_rate, weight_decay=weight_decay),
                'frozen': optax.set_to_zero(),
            },
            param_labels,
        )
    else:
        return optax.adamw(learning_rate, weight_decay=weight_decay)


# ============================================================================
# Training Loop
# ============================================================================

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
) -> None:
    """Main training loop with JIT-compiled step functions."""

    # Setup
    head_names = [spec.head_name for spec in head_specs]
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

    # Organism-specific metadata
    organism_enum = getattr(ag_dna_model.Organism, organism)
    organism_index_value = research_dna_model.convert_to_organism_index(organism_enum)
    strand_reindexing = jax.device_put(
        model._metadata[organism_enum].strand_reindexing,
        model._device_context._device,
    )

    # Loss functions
    loss_fns = {name: model.create_loss_fn_for_head(name) for name in head_names}

    # Define JIT-compiled training step
    @jax.jit
    def train_step(params, state, opt_state, batch):
        """Compute loss, gradients, and update parameters."""
        def loss_fn(params):
            predictions = model._predict(
                params,
                state,
                batch['sequences'],
                batch['organism_index'],
                negative_strand_mask=batch['negative_strand_mask'],
                strand_reindexing=strand_reindexing,
            )
            total_loss = 0.0
            for head_name in head_names:
                head_loss_dict = loss_fns[head_name](
                    predictions[head_name],
                    {
                        'targets': batch[f'targets_{head_name}'],
                        'organism_index': batch['organism_index'],
                    },
                )
                total_loss = total_loss + head_loss_dict['loss']
            return total_loss

        loss_value, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss_value

    # Define JIT-compiled evaluation step
    @jax.jit
    def eval_step(params, state, batch):
        """Compute predictions and losses without gradients."""
        predictions = model._predict(
            params,
            state,
            batch['sequences'],
            batch['organism_index'],
            negative_strand_mask=batch['negative_strand_mask'],
            strand_reindexing=strand_reindexing,
        )

        head_losses = {}
        for head_name in head_names:
            loss_dict = loss_fns[head_name](
                predictions[head_name],
                {
                    'targets': batch[f'targets_{head_name}'],
                    'organism_index': batch['organism_index'],
                },
            )
            head_losses[head_name] = loss_dict['loss']
        return head_losses

    print('  JIT-compiling step functions (first call will be slow)...')

    # Metric tracking helpers
    def aggregate_valid_loss(metrics: Mapping[str, float]) -> float | None:
        return float(sum(metrics.values())) if metrics else None

    def resolve_metric(metric_name: str, train_loss: float | None, valid_metrics: Mapping[str, float] | None):
        """
        Resolve metric name to value for checkpointing.
        Returns (resolved_metric_name, metric_value)
        """
        if metric_name in {'train', 'train_loss'}:
            return 'train_loss', train_loss
        if metric_name in {'valid', 'val', 'valid_loss', 'val_loss'}:
            return 'valid_loss', aggregate_valid_loss(valid_metrics or {})
        if metric_name.startswith('valid:') or metric_name.startswith('valid/'):
            head = metric_name.split(':', 1)[-1].split('/', 1)[-1]
            return f'valid/{head}', (valid_metrics or {}).get(head)
        if valid_metrics and metric_name in valid_metrics:
            return f'valid/{metric_name}', valid_metrics[metric_name]
        return metric_name, None

    def is_improved(current: float, best: float | None) -> bool:
        if best is None:
            return True
        if best_metric_mode == 'max':
            return current > best + early_stopping_min_delta
        return current < best - early_stopping_min_delta

    # Early stopping state
    best_value: float | None = None
    epochs_since_improvement = 0

    # Training loop
    with model._device_context:
        for epoch in range(1, num_epochs + 1):
            print(f'\n{"="*60}')
            print(f'Epoch {epoch}/{num_epochs}')
            print(f'{"="*60}')

            # === Training ===
            step = 0
            train_losses: list[float] = []

            for batch_np in data_module.iter_batches('train', seed=seed + epoch):
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

                print(f'  step {step:04d} | loss {loss_scalar:.4f}', end='\r', flush=True)

            # Training summary
            train_loss_avg = float(np.mean(train_losses)) if train_losses else None
            print()
            if train_loss_avg is not None:
                print(f'  Train loss: {train_loss_avg:.4f}')

            # === Validation ===
            valid_metrics: Mapping[str, float] | None = None
            if 'valid' in data_module._intervals and len(data_module._intervals['valid']) > 0:
                losses = {head: [] for head in head_names}
                for batch_np in data_module.iter_batches('valid'):
                    batch = prepare_batch(batch_np, organism_index_value, head_names)
                    head_losses = eval_step(model._params, model._state, batch)
                    for head_name in head_names:
                        losses[head_name].append(float(head_losses[head_name]))

                valid_metrics = {head: float(np.mean(values)) for head, values in losses.items() if values}
                print('  Validation metrics:', ', '.join(f'{k}={v:.4f}' for k, v in valid_metrics.items()))

            # === Checkpointing ===
            metric_label, metric_value = resolve_metric(best_metric, train_loss_avg, valid_metrics)

            # Track best model
            if metric_value is not None and math.isfinite(metric_value):
                if is_improved(metric_value, best_value):
                    best_value = metric_value
                    epochs_since_improvement = 0
                    if checkpoint_dir:
                        print(f'  Metric improved ({metric_label} = {metric_value:.4f}) -- saving best checkpoint')
                        model.save_checkpoint(checkpoint_dir / 'best', save_full_model=False)
                else:
                    epochs_since_improvement += 1
            else:
                print(f'  Best metric ({metric_label}): unavailable')

            # Save latest checkpoint
            if checkpoint_dir:
                model.save_checkpoint(checkpoint_dir / 'last', save_full_model=False)

            # === Early Stopping ===
            if early_stopping_patience > 0 and epochs_since_improvement >= early_stopping_patience:
                print(f'\n  Early stopping: no improvement for {epochs_since_improvement} epoch(s)')
                break

    # Final checkpoint
    if checkpoint_dir and not (checkpoint_dir / 'last').exists():
        model.save_checkpoint(checkpoint_dir / 'last', save_full_model=False)

    print(f'\n{"="*60}')
    print('Training complete!')
    print(f'{"="*60}')

# ============================================================================
# Main
# ============================================================================

def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)

    # Resolve paths
    if args.checkpoint_dir:
        args.checkpoint_dir = args.checkpoint_dir.resolve()
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.resume_from:
        args.resume_from = args.resume_from.resolve()
    if args.checkpoint_path:
        args.checkpoint_path = args.checkpoint_path.resolve()

    # Load configuration
    print('[1/4] Loading head specifications...')
    head_specs = build_head_specs(args.targets_config, organism=args.organism)
    validate_heads(head_specs)
    for spec in head_specs:
        if spec.source == 'predefined':
            if spec.config is None:
                raise ValueError(
                    f'Predefined head "{spec.head_name}" missing config.'
                )
            register_predefined_head(
                spec.head_name,
                spec.config,
                metadata=spec.metadata,
            )

    print('[2/4] Loading genomic intervals...')
    intervals = load_intervals(
        bed=args.bed,
        window_size=args.window_size,
        limit_train=args.limit_train,
        limit_valid=args.limit_valid,
        limit_test=args.limit_test,
    )

    # Initialize model
    print('[3/4] Initializing model...')
    if args.resume_from:
        print(f'  Resuming from {args.resume_from}')
        model = load_checkpoint(args.resume_from, base_model_version=args.model_version)
        if args.heads_only:
            model.freeze_backbone()
    else:
        print(f'  Creating new model (heads_only={args.heads_only})')
        model = create_model_with_heads(
            args.model_version,
            heads=[spec.head_name for spec in head_specs],
            detach_backbone=args.heads_only,
            init_seq_len=2**14,  # heads are length-agnostic
            checkpoint_path=args.checkpoint_path,
        )

    # Prepare data
    print('[4/4] Preparing data module...')
    data_module = BigWigDataModule(
        intervals=intervals,
        fasta_path=args.fasta,
        head_specs=head_specs,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        window_size=args.window_size,
        drop_last=args.drop_last,
    )

    # Train
    print('\nStarting training...\n')
    train(
        model,
        data_module,
        head_specs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        seed=args.seed,
        max_train_steps=args.max_train_steps,
        heads_only=args.heads_only,
        organism=args.organism,
        checkpoint_dir=args.checkpoint_dir,
        best_metric=args.best_metric,
        best_metric_mode=args.best_metric_mode,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )

    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('\n\n[Interrupted by user]')
        sys.exit(130)
    except Exception as exc:
        print(f'\n[ERROR] {exc}', file=sys.stderr)
        raise
