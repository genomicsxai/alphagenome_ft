"""Training utilities for fine-tuning AlphaGenome models."""

from __future__ import annotations

import functools
import json
import math
import queue
import threading
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import optax
import orbax.checkpoint as ocp
from alphagenome.models import dna_model as ag_dna_model
from alphagenome_research.model import dna_model as research_dna_model

from alphagenome_ft.custom_model import CustomAlphaGenomeModel
from alphagenome_ft.finetune.config import HeadSpec
from alphagenome_ft.finetune.data import BigWigDataModule, prepare_batch
from alphagenome_ft.finetune.logging_utils import TrainingLogger
from alphagenome_ft.finetune.splice_data import SpliceDataModule
from alphagenome_ft.optimizer_utils import create_optimizer, label_params_for_trainable_heads


def _prefetch(iterator: Iterator, buffer_size: int = 2) -> Iterator:
    """Run ``iterator`` on a background thread, buffering up to
    ``buffer_size`` items ahead of the consumer.

    The training loop's per-microbatch data loading (FASTA decode + bigwig/
    parquet lookups, all CPU-bound) previously ran fully serialized with the
    GPU compute for the *previous* microbatch: the generator only produces
    the next batch once the consumer asks for it, and JAX's async dispatch
    means the GPU sits idle while that happens. Loading the next batch on a
    separate thread while the main thread's `grad_step` call keeps the GPU
    busy overlaps the two instead.
    """
    q: queue.Queue = queue.Queue(maxsize=buffer_size)
    _SENTINEL = object()

    def _worker() -> None:
        try:
            for item in iterator:
                q.put(item)
        except BaseException as exc:  # noqa: BLE001 - re-raised on consumer side
            q.put(exc)
            return
        q.put(_SENTINEL)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    while True:
        item = q.get()
        if item is _SENTINEL:
            break
        if isinstance(item, BaseException):
            raise item
        yield item


def register_predefined_heads(head_specs: Sequence[HeadSpec]) -> None:
    """Register predefined heads from parsed head specs."""
    from alphagenome_ft import register_predefined_head
    from alphagenome_ft.custom_heads import register_junction_position_source

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
        if (
            spec.splice_source is not None
            and spec.splice_source.junction_position_source == "predicted"
        ):
            register_junction_position_source(
                spec.head_id,
                top_k=spec.splice_source.junction_top_k,
                classification_head_id=spec.splice_source.classification_head_id,
            )


## create_optimizer used to be redefined locally here (heads_only masking via
## optax.multi_transform, no gradient clipping) - a duplicate of, and shadowed
## the import of, alphagenome_ft.optimizer_utils.create_optimizer below, which
## already does the same heads_only masking AND supports
## gradient_clip_global_norm. Removed the local duplicate so the import is
## what actually runs, and threaded gradient_clip_global_norm through train().


def _replicate_tree(tree, devices):
    """Replicate a pytree across local devices for pmap."""
    mesh = Mesh(np.array(devices), ("data",))
    sharding = NamedSharding(mesh, P("data"))
    return jax.tree_util.tree_map(
        lambda value: jax.device_put(jnp.stack([value] * len(devices)), sharding),
        tree,
    )


def _unreplicate_tree(tree):
    """Extract the first local replica from a replicated pytree."""
    return jax.tree_util.tree_map(
        lambda value: np.asarray(value.addressable_shards[0].data).squeeze(0)
        if hasattr(value, "addressable_shards")
        else value[0],
        tree,
    )


def _shard_batch(batch: Mapping[str, jax.Array], num_devices: int):
    """Reshape a batch from [global_batch, ...] to [num_devices, per_device, ...]."""

    def shard_array(value: jax.Array) -> jax.Array:
        if value.shape[0] % num_devices != 0:
            raise ValueError(
                f"Batch size of {value.shape[0]} is not divisible by num_devices={num_devices}."
                f"Use a global batch size divisible by the number of local devices."
            )
        per_device_batch = value.shape[0] // num_devices
        return value.reshape((num_devices, per_device_batch, *value.shape[1:]))

    return {name: shard_array(value) for name, value in batch.items()}


def train(
    model: CustomAlphaGenomeModel,
    data_module: BigWigDataModule | SpliceDataModule,
    head_specs: Sequence[HeadSpec],
    *,
    learning_rate: float,
    weight_decay: float,
    num_epochs: int,
    seed: int = 42,
    max_train_steps: int | None = None,
    heads_only: bool = False,
    lora_enabled: bool = False,
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
    num_devices: int = 1,
    gradient_accumulation_steps: int = 1,
    resume_from: Path | None = None,
    save_every_steps: int | None = None,
    gradient_clip_global_norm: float | None = None,
    log_every: int = 50,
) -> None:
    """Run fine-tuning with pmapped train/eval steps.

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
        lora_enabled: If True (requires ``heads_only=True``), also keep
            backbone LoRA adapter params trainable alongside the heads —
            matches alphagenome-pytorch's ``--mode lora``. The caller is
            responsible for having installed the LoRA-patched backbone (see
            ``alphagenome_ft.lora.install_mha_backbone_lora``) and for NOT
            passing ``detach_backbone=True`` to model construction (LoRA
            needs gradients to reach the adapters through the backbone).
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
        num_devices: Number of local devices to use. Defaults to single-device.
        gradient_accumulation_steps: Number of data-module batches to average
            gradients over before applying one optimizer update. Effective
            batch size per optimizer step is
            ``data_module._batch_size * gradient_accumulation_steps``. Defaults
            to 1 (one optimizer step per batch, matching the original
            behavior). Trailing batches in an epoch that don't complete a full
            accumulation window are dropped, same as ``drop_last`` for
            multi-device sharding.
        resume_from: Optional checkpoint directory (as saved by this function,
            e.g. ``checkpoint_dir / "last"``) to resume epoch/step bookkeeping,
            and optimizer (Adam) state if an ``opt_state`` sidecar is present
            there, from. Model weights/state to resume from must still be
            loaded into ``model`` by the caller *before* calling ``train()``
            (e.g. via ``alphagenome_ft.load_checkpoint``), since ``train()``
            always starts from whatever ``model._params``/``model._state``
            currently are — only the optimizer state and epoch/step
            bookkeeping are handled here. If ``train_state.json`` exists but
            ``opt_state`` doesn't (e.g. a checkpoint saved before this
            parameter existed), optimizer state reinitializes from zero and a
            warning is printed rather than failing.
        save_every_steps: If set, also save a "last" checkpoint every this
            many optimizer steps, in addition to the always-on end-of-epoch
            "last" save — mirrors alphagenome-pytorch's ``--save-every-steps``
            (used there for exactly the same reason: without it, a run killed
            mid-epoch on a wall-time-limited partition loses the *entire*
            epoch's progress, not just progress since the last checkpoint,
            since the previous checkpoint is the *prior* epoch's end).
            ``resume_from`` correctly resumes mid-epoch from one of these:
            the train_state.json sidecar records not just the last completed
            epoch but how many optimizer steps into the *current* epoch were
            done, and the data-batch iterator is fast-forwarded (batches
            skipped without computing gradients, not replayed) to the same
            position before real training resumes — safe because
            ``iter_batches``' shuffle is a deterministic function of
            ``(len(windows), seed)`` (see both data modules' ``iter_batches``
            directly), so the same ``seed + epoch`` always reproduces the
            same batch order. Does not affect ``best`` (which is inherently
            tied to a completed epoch's validation pass) or the epoch's
            reported average train loss (which only reflects steps *after*
            the resume point when resuming mid-epoch — a minor, accepted
            approximation).
        gradient_clip_global_norm: If set, clip gradients to this global norm
            before the optimizer update (``optax.clip_by_global_norm``, via
            ``alphagenome_ft.optimizer_utils.create_optimizer`` — see that
            function's docstring). Matches alphagenome-pytorch's
            ``--max-grad-norm``. ``None`` (default) disables clipping.
        log_every: Log per-head training loss (to ``checkpoint_dir``'s
            ``training_log.csv``, and to W&B if enabled) every this many
            optimizer steps. Matches alphagenome-pytorch's ``--log-every``.

    Notes:
        Total planned steps are computed before training from train-set size and
        batch settings as ``steps_per_epoch * num_epochs`` (or capped by
        ``max_train_steps`` when provided). Progress is reported with a global
        counter in ``current/total`` format.

        Multi-GPU training on a single node is supported by passing a non-zero ``num_devices``
        with a global ``batch_size`` divisible by ``num_devices``. For multi-GPU training,
        the code requires ``drop_last=True`` to ensure all batches are evenly divisible
        across devices.

        The multi-GPU implementation is a distributed data-parallel (DDP) style approach
        using JAX's ``pmap``. Model parameters and optimizer state are replicated across
        devices, and each device processes a shard of each batch. Gradients and metrics
        are averaged across devices with ``lax.pmean`` to keep them in sync.
    """
    train_intervals = list(data_module._intervals.get("train", ()))
    num_train_examples = len(train_intervals)
    if num_train_examples == 0:
        raise ValueError("No train intervals available for training.")

    if gradient_accumulation_steps < 1:
        raise ValueError(
            f"gradient_accumulation_steps must be at least 1, got {gradient_accumulation_steps}."
        )

    if data_module._drop_last:
        micro_steps_per_epoch = num_train_examples // data_module._batch_size
    else:
        micro_steps_per_epoch = math.ceil(num_train_examples / data_module._batch_size)
    if micro_steps_per_epoch == 0:
        raise ValueError(
            "Computed zero training steps per epoch. Check batch size, drop_last, and train intervals."
        )
    # Optimizer steps per epoch, after accumulating gradient_accumulation_steps
    # batches per update. Trailing micro-batches that don't fill a full
    # accumulation window are dropped (see docstring).
    steps_per_epoch = micro_steps_per_epoch // gradient_accumulation_steps
    if steps_per_epoch == 0:
        raise ValueError(
            f"gradient_accumulation_steps={gradient_accumulation_steps} exceeds "
            f"micro_steps_per_epoch={micro_steps_per_epoch}; no optimizer step would "
            "ever run. Reduce gradient_accumulation_steps or increase train data."
        )

    planned_steps = steps_per_epoch * num_epochs
    total_train_steps = (
        min(planned_steps, max_train_steps) if max_train_steps is not None else planned_steps
    )
    step_width = len(str(steps_per_epoch))

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if num_devices < 1:
        raise ValueError(f"num_devices must be at least 1, got {num_devices}.")

    available_devices = jax.local_devices()
    if num_devices > len(available_devices):
        raise ValueError(
            f"Requested num_devices={num_devices}, but only {len(available_devices)} local "
            f"device(s) are available."
        )
    devices = available_devices[:num_devices]

    wb_config = {
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "num_epochs": num_epochs,
        "batch_size": data_module._batch_size,
        "steps_per_epoch": steps_per_epoch,
        "total_train_steps": total_train_steps,
        "heads_only": heads_only,
        "organism": organism,
        "num_devices": num_devices,
        "best_metric": best_metric,
        "best_metric_mode": best_metric_mode,
        "early_stopping_patience": early_stopping_patience,
        "seed": seed,
        **(wandb_config or {}),
    }
    logger = TrainingLogger(
        log_dir=checkpoint_dir,
        use_wandb=use_wandb,
        wandb_project=wandb_project or "alphagenome-ft",
        wandb_entity=wandb_entity,
        run_name=wandb_run_name,
        config=wb_config,
    )

    head_names = [spec.head_id for spec in head_specs]
    if num_devices > 1 and not data_module._drop_last:
        raise ValueError(
            "Single-host multi-GPU training currently requires drop_last=True so every "
            "batch can be sharded evenly across devices."
        )
    if lora_enabled and not heads_only:
        raise ValueError("lora_enabled requires heads_only=True.")
    if heads_only:
        model.freeze_backbone()

    optimizer = create_optimizer(
        model._params,
        trainable_head_names=head_names,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        heads_only=heads_only,
        lora_enabled=lora_enabled,
        gradient_clip_global_norm=gradient_clip_global_norm,
    )
    opt_state = optimizer.init(model._params)

    organism_enum = getattr(ag_dna_model.Organism, organism)
    organism_index_value = research_dna_model.convert_to_organism_index(organism_enum)
    strand_reindexing = model._metadata[organism_enum].strand_reindexing

    loss_fns = {name: model.create_loss_fn_for_head(name) for name in head_names}

    def _predict_extra_kwargs(batch):
        # Only forward splice_site_positions when the batch actually carries it
        # (splice-junction training runs); other model._predict implementations
        # (e.g. the raw base model with no custom heads) don't accept this kwarg.
        if "splice_site_positions" in batch:
            return {"splice_site_positions": batch["splice_site_positions"]}
        return {}

    _RAW_JUNCTION_EVENT_KEYS = (
        "junction_d_rel", "junction_a_rel", "junction_is_pos_strand", "junction_counts",
    )

    def _head_batch(batch, head_name):
        head_batch = {
            "targets": batch[f"targets_{head_name}"],
            "organism_index": batch["organism_index"],
        }
        if "splice_site_positions" in batch:
            head_batch["splice_site_positions"] = batch["splice_site_positions"]
        for key in _RAW_JUNCTION_EVENT_KEYS:
            if key in batch:
                head_batch[key] = batch[key]
        return head_batch

    # Which leaves jax.grad should actually bother computing a real gradient
    # for. Without this, jax.value_and_grad(loss_fn)(params) below
    # differentiates w.r.t. the WHOLE ~450M-parameter tree every micro-batch
    # -- including the frozen backbone -- and create_optimizer's heads_only
    # masking only discards those gradients *after* the (expensive) backward
    # pass already computed them. --mode linear-probe avoids this for free
    # via detach_backbone (a single stop_gradient on the backbone's output
    # embeddings cuts every path back to every backbone weight at once), but
    # --mode lora can't use that trick (gradients must reach the LoRA
    # adapters through the backbone) -- so apply stop_gradient per-weight
    # instead: frozen leaves get zero gradient AND skip the real backward
    # computation for that leaf (mirrors PyTorch's requires_grad=False, which
    # has the same compute-skipping effect, not just a training-update
    # mask). Confirmed necessary empirically: --mode lora still OOM'd after
    # switching to per-layer remat alone.
    _trainable_labels = None
    if heads_only:
        _trainable_labels = label_params_for_trainable_heads(
            model._params, head_names, lora_enabled=lora_enabled,
        )

    @functools.partial(jax.pmap, axis_name="data")
    def grad_step(params, state, batch):
        """Compute (loss, grads) for one micro-batch, without applying them.

        Split out from the optimizer update so gradient_accumulation_steps > 1
        can average grads over several micro-batches before one
        optimizer.update call — alphagenome_ft's train() previously took one
        full optimizer step per data_module batch with no accumulation.
        """

        def loss_fn(current_params):
            if _trainable_labels is not None:
                current_params = jax.tree_util.tree_map(
                    lambda label, x: x if label == "head" else jax.lax.stop_gradient(x),
                    _trainable_labels, current_params,
                )
            predictions = model._predict(
                current_params,
                state,
                batch["sequences"],
                batch["organism_index"],
                negative_strand_mask=batch["negative_strand_mask"],
                strand_reindexing=batch["strand_reindexing"],
                **_predict_extra_kwargs(batch),
            )
            head_losses = {}
            for head_name in head_names:
                head_loss_dict = loss_fns[head_name](
                    predictions[head_name], _head_batch(batch, head_name)
                )
                head_losses[head_name] = head_loss_dict["loss"]
            total_loss = sum(head_losses.values())
            return total_loss, head_losses

        (loss_value, head_losses), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)
        loss_value = jax.lax.pmean(loss_value, axis_name="data")
        grads = jax.lax.pmean(grads, axis_name="data")
        head_losses = jax.tree_util.tree_map(
            lambda loss_value: jax.lax.pmean(loss_value, axis_name="data"),
            head_losses,
        )
        return loss_value, grads, head_losses

    @functools.partial(jax.pmap, axis_name="data")
    def apply_grads(params, current_opt_state, grads):
        """Apply one optimizer update from (possibly accumulated) grads."""
        updates, new_opt_state = optimizer.update(grads, current_opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    @functools.partial(jax.pmap, axis_name="data")
    def eval_step(params, state, batch):
        predictions = model._predict(
            params,
            state,
            batch["sequences"],
            batch["organism_index"],
            negative_strand_mask=batch["negative_strand_mask"],
            strand_reindexing=batch["strand_reindexing"],
            **_predict_extra_kwargs(batch),
        )
        head_losses = {}
        for head_name in head_names:
            loss_dict = loss_fns[head_name](
                predictions[head_name], _head_batch(batch, head_name)
            )
            head_losses[head_name] = loss_dict["loss"]
        head_losses = jax.tree_util.tree_map(
            lambda loss_value: jax.lax.pmean(loss_value, axis_name="data"),
            head_losses,
        )
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
    start_epoch = 1
    # How many optimizer steps of start_epoch are already done (mid-epoch
    # resume via save_every_steps). 0 for a fresh start or an epoch-boundary
    # resume; only ever nonzero for the FIRST epoch iterated after resuming.
    start_epoch_step = 0

    if resume_from is not None:
        state_path = Path(resume_from) / "train_state.json"
        if state_path.exists():
            saved = json.loads(state_path.read_text())
            start_epoch = saved["resume_epoch"]
            start_epoch_step = saved.get("resume_epoch_step", 0)
            global_step = saved["global_step"]
            best_value = saved.get("best_value")
            epochs_since_improvement = saved.get("epochs_since_improvement", 0)
            opt_state_path = Path(resume_from) / "opt_state"
            if opt_state_path.exists():
                # orbax's tensorstore backend requires an absolute path.
                opt_state = ocp.StandardCheckpointer().restore(
                    str(opt_state_path.resolve()), target=opt_state,
                )
                opt_state_msg = f"optimizer state restored from {opt_state_path}"
            else:
                opt_state_msg = (
                    f"no opt_state found at {opt_state_path}; optimizer state "
                    f"reinitializes from zero"
                )
            print(
                f"Resuming from {state_path}: model weights are assumed already "
                f"loaded by the caller; continuing bookkeeping at epoch "
                f"{start_epoch} step {start_epoch_step}, global_step "
                f"{global_step} ({opt_state_msg})."
            )
        else:
            print(
                f"--resume_from={resume_from} given but no train_state.json found "
                f"there; starting epoch/step bookkeeping from scratch (any model "
                f"weights already loaded by the caller are still used)."
            )

    print(
        "Train plan: "
        f"{num_train_examples} examples | "
        f"{steps_per_epoch} step(s)/epoch | "
        f"{num_epochs} epoch(s) | "
        f"total step(s) {total_train_steps} | "
        f"starting at epoch {start_epoch} step {start_epoch_step}, "
        f"global_step {global_step}"
    )

    def _write_train_state(target_dir: Path, resume_epoch: int, resume_epoch_step: int = 0) -> None:
        (target_dir / "train_state.json").write_text(json.dumps({
            "resume_epoch": resume_epoch,
            "resume_epoch_step": resume_epoch_step,
            "global_step": global_step,
            "best_value": best_value,
            "epochs_since_improvement": epochs_since_improvement,
        }))

    def _save_opt_state(target_dir: Path) -> None:
        checkpoint_path = target_dir / "opt_state"
        if checkpoint_path.exists():
            import shutil
            shutil.rmtree(checkpoint_path)
        checkpointer = ocp.StandardCheckpointer()
        # Save with the same absolute-path convention as the restore above
        # (orbax's tensorstore backend requires it there; matching it here
        # too, rather than relying on save's laxer/inconsistent handling of
        # relative paths, keeps both sides unambiguous).
        checkpointer.save(str(checkpoint_path.resolve()), _unreplicate_tree(opt_state))
        checkpointer.wait_until_finished()

    if start_epoch > num_epochs:
        print(f"Already completed {num_epochs}/{num_epochs} requested epochs; nothing to do.")

    with model._device_context:
        replicated_params = _replicate_tree(model._params, devices)
        replicated_state = _replicate_tree(model._state, devices)
        opt_state = _replicate_tree(opt_state, devices)
        strand_reindexing_replicated = _replicate_tree(strand_reindexing, devices)
        stop_training = False
        for epoch in range(start_epoch, num_epochs + 1):
            if verbose:
                print(f"\n{'=' * 60}")
                print(f"Epoch {epoch}/{num_epochs}")
                print(f"{'=' * 60}")
            else:
                print(f"Epoch {epoch}/{num_epochs}")

            epoch_step = start_epoch_step if epoch == start_epoch else 0
            # Fast-forward past already-completed micro-batches when resuming
            # mid-epoch (save_every_steps) — see train()'s docstring for why
            # this is safe: iter_batches' shuffle depends only on
            # (len(windows), seed), so seed + epoch reproduces the identical
            # batch order every time, and skip_batches is always an exact
            # multiple of gradient_accumulation_steps (an accumulation-window
            # boundary), so no special-casing of accum_grads is needed below.
            # Passed into iter_batches itself (rather than iterated-and-
            # discarded here) so it skips via a slice of the deterministic
            # shuffle order instead of actually extracting (decoding FASTA +
            # bigwig lookups for) every already-completed batch — the latter
            # made resuming late in a long epoch cost almost as much wall
            # time as redoing the training itself.
            skip_batches = epoch_step * gradient_accumulation_steps if epoch == start_epoch else 0
            train_losses: list[float] = []
            train_head_losses: dict[str, list[float]] = {head: [] for head in head_names}
            accum_grads = None
            accum_loss_sum = 0.0
            accum_head_loss_sum = {head: 0.0 for head in head_names}
            accum_count = 0
            for batch_np in _prefetch(data_module.iter_batches(
                "train", seed=seed + epoch, skip_batches=skip_batches,
            )):
                batch = prepare_batch(batch_np, organism_index_value, head_names)
                batch = _shard_batch(batch, num_devices)
                batch["strand_reindexing"] = strand_reindexing_replicated
                micro_loss_value, micro_grads, micro_head_losses = grad_step(
                    replicated_params, replicated_state, batch,
                )
                accum_grads = (
                    micro_grads
                    if accum_grads is None
                    else jax.tree_util.tree_map(jnp.add, accum_grads, micro_grads)
                )
                accum_loss_sum += float(np.asarray(micro_loss_value)[0])
                for head_name in head_names:
                    accum_head_loss_sum[head_name] += float(
                        np.asarray(micro_head_losses[head_name])[0]
                    )
                accum_count += 1

                if accum_count < gradient_accumulation_steps:
                    continue

                # Full accumulation window reached: average grads over the
                # window and take exactly one optimizer step.
                averaged_grads = jax.tree_util.tree_map(
                    lambda g: g / gradient_accumulation_steps, accum_grads,
                )
                replicated_params, opt_state = apply_grads(
                    replicated_params, opt_state, averaged_grads,
                )
                loss_scalar = accum_loss_sum / accum_count
                head_loss_scalars = {
                    head: accum_head_loss_sum[head] / accum_count for head in head_names
                }
                accum_grads = None
                accum_loss_sum = 0.0
                accum_head_loss_sum = {head: 0.0 for head in head_names}
                accum_count = 0

                train_losses.append(loss_scalar)
                for head_name in head_names:
                    train_head_losses[head_name].append(head_loss_scalars[head_name])
                epoch_step += 1
                global_step += 1

                if verbose:
                    print(
                        f"  step {epoch_step:0{step_width}d}/{steps_per_epoch:0{step_width}d}"
                        f" | loss {loss_scalar:.4f}",
                        end="\r",
                        flush=True,
                    )

                if global_step % log_every == 0:
                    logger.log_step(
                        {
                            "step": global_step,
                            "epoch": epoch,
                            "loss": loss_scalar,
                            **{
                                f"{head}_loss": v
                                for head, v in head_loss_scalars.items()
                            },
                        }
                    )

                if checkpoint_dir and save_every_steps and global_step % save_every_steps == 0:
                    model._params = _unreplicate_tree(replicated_params)
                    model._state = _unreplicate_tree(replicated_state)
                    model.save_checkpoint(checkpoint_dir / "last", save_full_model=False)
                    _write_train_state(checkpoint_dir / "last", resume_epoch=epoch, resume_epoch_step=epoch_step)
                    _save_opt_state(checkpoint_dir / "last")
                    if verbose:
                        print(f"\n  Mid-epoch checkpoint saved (step {epoch_step}/{steps_per_epoch})")

                if global_step >= total_train_steps:
                    stop_training = True
                    break

            train_loss_avg = float(np.mean(train_losses)) if train_losses else None
            train_head_loss_avgs = {
                head: float(np.mean(values))
                for head, values in train_head_losses.items()
                if values
            }
            if verbose:
                print()
            if train_loss_avg is not None:
                print(f"  Train loss: {train_loss_avg:.4f}")

            valid_metrics: Mapping[str, float] | None = None
            if "valid" in data_module._intervals and len(data_module._intervals["valid"]) > 0:
                losses = {head: [] for head in head_names}
                # Explicit seed, not the default (None): CombinedDataModule
                # passes this same seed to both its underlying modules, so an
                # explicit shared value is what keeps their independent
                # shuffles in lock-step (see the "train" call above). Passing
                # none here would let each module draw its own OS-entropy
                # seed via np.random.default_rng(None), desynchronizing them
                # even though both received the same "None" argument.
                for batch_np in _prefetch(data_module.iter_batches("valid", seed=seed)):
                    batch = prepare_batch(batch_np, organism_index_value, head_names)
                    batch = _shard_batch(batch, num_devices)
                    batch["strand_reindexing"] = strand_reindexing_replicated
                    head_losses = eval_step(replicated_params, replicated_state, batch)
                    for head_name in head_names:
                        losses[head_name].append(float(np.asarray(head_losses[head_name])[0]))

                valid_metrics = {
                    head: float(np.mean(values)) for head, values in losses.items() if values
                }
                print(
                    "  Validation metrics:",
                    ", ".join(f"{k}={v:.4f}" for k, v in valid_metrics.items()),
                )

            if train_loss_avg is not None:
                logger.log_epoch(
                    epoch,
                    {
                        "train_loss": train_loss_avg,
                        **{
                            f"train_{head}_loss": v
                            for head, v in train_head_loss_avgs.items()
                        },
                        **(
                            {
                                "val_loss": float(sum(valid_metrics.values())),
                                **{
                                    f"val_{head}_loss": v
                                    for head, v in valid_metrics.items()
                                },
                            }
                            if valid_metrics is not None
                            else {}
                        ),
                    },
                )

            metric_label, metric_value = resolve_metric(best_metric, train_loss_avg, valid_metrics)
            if metric_value is not None and math.isfinite(metric_value):
                if is_improved(metric_value, best_value):
                    best_value = metric_value
                    epochs_since_improvement = 0
                    if logger.wandb is not None:
                        logger.wandb.log({"best/" + metric_label: metric_value, "epoch": epoch})
                    if checkpoint_dir:
                        model._params = _unreplicate_tree(replicated_params)
                        model._state = _unreplicate_tree(replicated_state)
                        print(
                            f"  Metric improved ({metric_label} = {metric_value:.4f}) "
                            " -- saving best checkpoint"
                        )
                        model.save_checkpoint(checkpoint_dir / "best", save_full_model=False)
                        _write_train_state(checkpoint_dir / "best", resume_epoch=epoch + 1, resume_epoch_step=0)
                        _save_opt_state(checkpoint_dir / "best")
                else:
                    epochs_since_improvement += 1
            else:
                print(f"  Best metric ({metric_label}): unavailable")

            if checkpoint_dir:
                model._params = _unreplicate_tree(replicated_params)
                model._state = _unreplicate_tree(replicated_state)
                model.save_checkpoint(checkpoint_dir / "last", save_full_model=False)
                _write_train_state(checkpoint_dir / "last", resume_epoch=epoch + 1, resume_epoch_step=0)
                _save_opt_state(checkpoint_dir / "last")

            if early_stopping_patience > 0 and epochs_since_improvement >= early_stopping_patience:
                print(f"\n  Early stopping: no improvement for {epochs_since_improvement} epoch(s)")
                break
            if stop_training:
                print(f"  Reached requested training steps: {global_step}/{total_train_steps}")
                break

        model._params = _unreplicate_tree(replicated_params)
        model._state = _unreplicate_tree(replicated_state)

    if checkpoint_dir and not (checkpoint_dir / "last").exists():
        model.save_checkpoint(checkpoint_dir / "last", save_full_model=False)
        _write_train_state(
            checkpoint_dir / "last", resume_epoch=start_epoch, resume_epoch_step=start_epoch_step,
        )
        _save_opt_state(checkpoint_dir / "last")

    print(f"\n{'=' * 60}")
    print("Training complete!")
    print(f"{'=' * 60}")

    logger.finish()


__all__ = [
    "register_predefined_heads",
    "create_optimizer",
    "train",
]
