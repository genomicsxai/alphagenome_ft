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
from alphagenome_ft.optimizer_utils import (
    build_lr_schedule,
    create_optimizer,
    label_params_for_trainable_heads,
)


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
                splice_site_threshold=spec.splice_source.splice_site_threshold,
                classification_head_id=spec.splice_source.classification_head_id,
            )


## create_optimizer used to be redefined locally here (heads_only masking via
## optax.multi_transform, no gradient clipping) - a duplicate of, and shadowed
## the import of, alphagenome_ft.optimizer_utils.create_optimizer below, which
## already does the same heads_only masking AND supports
## gradient clipping (``max_grad_norm`` here). Removed the local duplicate so the
## import is what actually runs.


# --- Validation metrics, aligned with alphagenome-pytorch -------------------
#
# Every definition below mirrors alphagenome-pytorch's
# ``extensions/finetuning/training.py`` (``validate_multihead``,
# ``_extract_splice_pearson_pairs``, ``_extract_junction_pearson_per_sample``,
# ``_extract_usage_pearson_per_sample``) and ``alphagenome_pytorch.metrics``,
# including metric key names, so epoch logs from the two repos are directly
# comparable. Pooled Pearson correlations are accumulated as float64
# sufficient statistics instead of concatenating every value (PyTorch's
# approach); this is the same quantity, computed in bounded memory.

_PEARSON_EPS = 1e-8  # alphagenome_pytorch.metrics.pearson_r default eps
_SPLICE_CLS_NAMES = ("donor_pos", "acceptor_pos", "donor_neg", "acceptor_neg", "no_site")
_JUNCTION_COUNT_VARIANTS = ("full", "nonzero")  # log1p before Pearson
_JUNCTION_VARIANTS = ("full", "nonzero", "psi5", "psi3")
_PSI_EPS = 1e-8


def _pearson_r_1d(pred, true) -> float:
    """``alphagenome_pytorch.metrics.pearson_r`` on two flat arrays."""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    true = np.asarray(true, dtype=np.float64).ravel()
    pc = pred - pred.mean()
    tc = true - true.mean()
    denom = np.sqrt((pc * pc).sum()) * np.sqrt((tc * tc).sum())
    return float((pc * tc).sum() / (denom + _PEARSON_EPS))


class _PearsonSums:
    """Pooled Pearson r from streamed sufficient statistics (float64).

    Equal to ``pearson_r(cat(pred), cat(true))`` over everything added;
    ``shape`` gives independent accumulators (e.g. one per channel).
    """

    _KEYS = ("n", "sx", "sy", "sxy", "sx2", "sy2")

    def __init__(self, shape=()):
        for k in self._KEYS:
            setattr(self, k, np.zeros(shape, dtype=np.float64))

    def add(self, pred, true, weight=None, axis=None) -> None:
        pred = np.asarray(pred, dtype=np.float64)
        true = np.asarray(true, dtype=np.float64)
        w = np.ones_like(pred) if weight is None else np.asarray(weight, dtype=np.float64)
        self.n = self.n + np.sum(w, axis=axis)
        self.sx = self.sx + np.sum(w * pred, axis=axis)
        self.sy = self.sy + np.sum(w * true, axis=axis)
        self.sxy = self.sxy + np.sum(w * pred * true, axis=axis)
        self.sx2 = self.sx2 + np.sum(w * pred * pred, axis=axis)
        self.sy2 = self.sy2 + np.sum(w * true * true, axis=axis)

    def add_stats(self, stats: Mapping[str, np.ndarray]) -> None:
        for k in self._KEYS:
            setattr(self, k, getattr(self, k) + np.asarray(stats[k], dtype=np.float64))

    def r(self):
        """Pearson r; NaN where fewer than 2 values (PyTorch skips those)."""
        n = self.n
        with np.errstate(divide="ignore", invalid="ignore"):
            num = self.sxy - self.sx * self.sy / n
            vx = np.clip(self.sx2 - self.sx * self.sx / n, 0, None)
            vy = np.clip(self.sy2 - self.sy * self.sy / n, 0, None)
            r = num / (np.sqrt(vx) * np.sqrt(vy) + _PEARSON_EPS)
        return np.where(n > 1, r, np.nan)


def _device_pearson_stats(pred, true, weight, axis):
    """Per-batch sufficient statistics on device (summed over ``axis``)."""
    pred = pred.astype(jnp.float32)
    true = true.astype(jnp.float32)
    w = weight.astype(jnp.float32)
    return {
        "n": jnp.sum(w, axis=axis),
        "sx": jnp.sum(w * pred, axis=axis),
        "sy": jnp.sum(w * true, axis=axis),
        "sxy": jnp.sum(w * pred * true, axis=axis),
        "sx2": jnp.sum(w * pred * pred, axis=axis),
        "sy2": jnp.sum(w * true * true, axis=axis),
    }


def _profile_pearson_r(pred, true):
    """``alphagenome_pytorch.metrics.profile_pearson_r``: Pearson over positions
    (axis 1) per (region, track). ``pred``/``true``: (B, S, T) -> (B, T)."""
    pred = pred.astype(jnp.float32)
    true = true.astype(jnp.float32)
    pc = pred - pred.mean(axis=1, keepdims=True)
    tc = true - true.mean(axis=1, keepdims=True)
    num = (pc * tc).sum(axis=1)
    denom = jnp.sqrt((pc * pc).sum(axis=1)) * jnp.sqrt((tc * tc).sum(axis=1))
    return num / (denom + _PEARSON_EPS)


def _usage_device_stats(pred, target):
    """splice_usage stats, as ``_extract_splice_pearson_pairs`` (usage) and
    ``_extract_usage_pearson_per_sample``: predictions are sigmoid
    probabilities; "full" pools every channel at positions where any channel's
    target > 0; per-sample pools channel ``c`` where its own target > 0."""
    full_w = jnp.broadcast_to(jnp.any(target > 0, axis=-1, keepdims=True), target.shape)
    per_w = target > 0
    return {
        "full": _device_pearson_stats(pred, target, full_w, axis=None),
        "per_channel": _device_pearson_stats(pred, target, per_w, axis=(0, 1)),
    }


def _junction_variants(pred, target, positions):
    """Host-side mirror of ``_extract_splice_pearson_pairs`` (junction head).

    ``pred``/``target``: (B, D, A, 2T) with positive-strand tissues first;
    ``positions``: (B, 4, K) [pos_donor, pos_acceptor, neg_donor, neg_acceptor].
    Returns {variant: (pred_flat, true_flat)} for full/nonzero/psi5/psi3 and
    binary_cls (pred scores vs target > 0), pooled over both strands.
    """
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    positions = np.asarray(positions)
    n_t = pred.shape[-1] // 2
    out = {k: ([], []) for k in (*_JUNCTION_VARIANTS, "binary_cls")}
    for sl, d_row, a_row in ((slice(None, n_t), 0, 1), (slice(n_t, None), 2, 3)):
        p = pred[..., sl]
        t = target[..., sl]
        pairs = (positions[:, d_row, :] >= 0)[:, :, None] & (positions[:, a_row, :] >= 0)[:, None, :]
        m4 = np.broadcast_to(pairs[..., None], p.shape)
        if m4.any():
            out["full"][0].append(p[m4]); out["full"][1].append(t[m4])
            out["binary_cls"][0].append(p[m4]); out["binary_cls"][1].append((t[m4] > 0).astype(np.float32))
        nz = m4 & (t > 0)
        if nz.any():
            out["nonzero"][0].append(p[nz]); out["nonzero"][1].append(t[nz])
        t_d = t.sum(axis=2, keepdims=True)
        p_d = p.sum(axis=2, keepdims=True)
        v5 = m4 & np.broadcast_to(t_d > 0, m4.shape)
        if v5.any():
            out["psi5"][0].append((p / (p_d + _PSI_EPS))[v5]); out["psi5"][1].append((t / (t_d + _PSI_EPS))[v5])
        t_a = t.sum(axis=1, keepdims=True)
        p_a = p.sum(axis=1, keepdims=True)
        v3 = m4 & np.broadcast_to(t_a > 0, m4.shape)
        if v3.any():
            out["psi3"][0].append((p / (p_a + _PSI_EPS))[v3]); out["psi3"][1].append((t / (t_a + _PSI_EPS))[v3])
    return {
        k: (np.concatenate(ps), np.concatenate(ts)) for k, (ps, ts) in out.items() if ps
    }


def _junction_per_sample(pred, target, positions):
    """Host-side mirror of ``_extract_junction_pearson_per_sample``: per
    biological sample ``s``, both strands' valid pairs pooled; returns a list
    of {"full": (p, t), "nonzero": (p, t) | None}."""
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    positions = np.asarray(positions)
    n_t = pred.shape[-1] // 2
    per = [([], []) for _ in range(n_t)]
    for off, d_row, a_row in ((0, 0, 1), (n_t, 2, 3)):
        pairs = (positions[:, d_row, :] >= 0)[:, :, None] & (positions[:, a_row, :] >= 0)[:, None, :]
        for s_idx in range(n_t):
            per[s_idx][0].append(pred[..., off + s_idx][pairs])
            per[s_idx][1].append(target[..., off + s_idx][pairs])
    result = []
    for ps, ts in per:
        p = np.concatenate(ps); t = np.concatenate(ts)
        nz = t > 0
        result.append({"full": (p, t), "nonzero": (p[nz], t[nz]) if nz.any() else None})
    return result


def _cls_active_and_background(logits, target, rng):
    """Mirror of PyTorch's classification auPRC sampling: every active
    (any target > 0) position plus an equal-sized random background subsample."""
    logits = np.asarray(logits, dtype=np.float32).reshape(-1, 5)
    target = np.asarray(target, dtype=np.float32).reshape(-1, 5)
    active = target[:, :4].any(axis=-1)  # splice classes only; column 4 is the "no site" class
    n_active = int(active.sum())
    if n_active == 0:
        return None
    bg = np.flatnonzero(~active)
    parts_l, parts_t = [logits[active]], [target[active]]
    if bg.size:
        pick = bg[rng.permutation(bg.size)[: min(n_active, bg.size)]]
        parts_l.append(logits[pick]); parts_t.append(target[pick])
    return np.concatenate(parts_l), np.concatenate(parts_t)


class _ValidationMetrics:
    """Accumulates and finalizes alphagenome-pytorch's validation metrics."""

    def __init__(self, per_sample: bool, seed: int = 0):
        self.per_sample = per_sample
        self.rng = np.random.default_rng(seed)
        self.profile_r: dict[str, list] = {}
        self.pred_counts: dict[str, list] = {}
        self.true_counts: dict[str, list] = {}
        self.usage_full: dict[str, _PearsonSums] = {}
        self.usage_per_channel: dict[str, _PearsonSums] = {}
        self.junction: dict[str, dict[str, _PearsonSums]] = {}
        self.junction_nonzero: dict[str, list] = {}  # [n_nonzero, n_total]
        self.junction_bincls: dict[str, tuple[list, list]] = {}
        self.junction_per_sample: dict[str, list] = {}
        self.cls: dict[str, tuple[list, list]] = {}

    # Genome tracks (rna_seq etc.): profile r per (region, track) + counts.
    def add_genome_tracks(self, name, profile_r, pred_counts, true_counts):
        self.profile_r.setdefault(name, []).append(np.asarray(profile_r, np.float64))
        self.pred_counts.setdefault(name, []).append(np.asarray(pred_counts, np.float64))
        self.true_counts.setdefault(name, []).append(np.asarray(true_counts, np.float64))

    def add_usage(self, name, stats):
        full = self.usage_full.setdefault(name, _PearsonSums())
        full.add_stats(stats["full"])
        n_ch = np.asarray(stats["per_channel"]["n"]).shape
        pc = self.usage_per_channel.setdefault(name, _PearsonSums(n_ch))
        pc.add_stats(stats["per_channel"])

    def add_junction(self, name, pred, target, positions):
        variants = _junction_variants(pred, target, positions)
        if "full" not in variants:
            return
        sums = self.junction.setdefault(name, {v: _PearsonSums() for v in _JUNCTION_VARIANTS})
        for v in _JUNCTION_VARIANTS:
            if v in variants:
                p, t = variants[v]
                if v in _JUNCTION_COUNT_VARIANTS:
                    p, t = np.log1p(p.astype(np.float64)), np.log1p(t.astype(np.float64))
                sums[v].add(p, t)
        full_t = variants["full"][1]
        cnt = self.junction_nonzero.setdefault(name, [0, 0])
        cnt[0] += int((full_t > 0).sum()); cnt[1] += int(full_t.size)
        bp, bt = self.junction_bincls.setdefault(name, ([], []))
        bp.append(variants["binary_cls"][0]); bt.append(variants["binary_cls"][1])
        if self.per_sample:
            ps = _junction_per_sample(pred, target, positions)
            acc = self.junction_per_sample.setdefault(
                name, [{"full": _PearsonSums(), "nonzero": _PearsonSums()} for _ in ps]
            )
            for s_idx, data in enumerate(ps):
                for v in ("full", "nonzero"):
                    if data[v] is not None:
                        p, t = data[v]
                        acc[s_idx][v].add(np.log1p(p.astype(np.float64)), np.log1p(t.astype(np.float64)))

    def add_classification(self, name, logits, target):
        picked = _cls_active_and_background(logits, target, self.rng)
        if picked is None:
            return
        lg, tg = self.cls.setdefault(name, ([], []))
        lg.append(picked[0]); tg.append(picked[1])

    def finalize(self) -> tuple[dict, list[dict], dict]:
        """Returns (metrics, per_sample_rows, histograms) with PyTorch's keys."""
        metrics: dict = {}
        histograms: dict = {}
        for name, chunks in self.profile_r.items():
            r = np.concatenate(chunks, axis=0)
            metrics[f"{name}_1bp_profile_pearson_r_mean"] = float(r.mean())
            metrics[f"{name}_1bp_profile_pearson_r_std"] = float(r.std(ddof=1)) if r.size > 1 else float("nan")
            histograms[f"{name}_1bp_profile_pearson_r_values"] = r.ravel().tolist()
            pc = np.concatenate(self.pred_counts[name], axis=0)
            tc = np.concatenate(self.true_counts[name], axis=0)
            if pc.shape[0] > 1:
                per_track = [_pearson_r_1d(pc[:, j], tc[:, j]) for j in range(pc.shape[1])]
                metrics[f"{name}_1bp_count_pearson_r"] = float(np.mean(per_track))
            else:
                metrics[f"{name}_1bp_count_pearson_r"] = float("nan")
        for name, sums in self.usage_full.items():
            r = float(sums.r())
            if math.isfinite(r):
                metrics[f"{name}_pearson_r"] = r
        for name, by_variant in self.junction.items():
            for v, sums in by_variant.items():
                r = float(sums.r())
                if math.isfinite(r):
                    metrics[f"{name}_pearson_r" + ("" if v == "full" else f"_{v}")] = r
            nz, total = self.junction_nonzero[name]
            if total:
                metrics[f"{name}_target_nonzero_frac"] = nz / total
        if self.junction_bincls or self.cls:
            from sklearn.metrics import average_precision_score
        for name, (bp, bt) in self.junction_bincls.items():
            p = np.concatenate(bp); t = np.concatenate(bt)
            if t.sum() > 0 and p.shape[0] > 1:
                metrics[f"{name}_auprc_junction"] = float(average_precision_score(t, p))
        for name, (lg, tg) in self.cls.items():
            logits = np.concatenate(lg).astype(np.float64)
            true = np.concatenate(tg)
            if true.shape[0] == 0:
                continue
            e = np.exp(logits - logits.max(axis=-1, keepdims=True))
            probs = e / e.sum(axis=-1, keepdims=True)
            for i, cls_name in enumerate(_SPLICE_CLS_NAMES):
                if true[:, i].sum() > 0:
                    metrics[f"{name}_auprc_{cls_name}"] = float(
                        average_precision_score(true[:, i], probs[:, i])
                    )
            cols = [i for i, c in enumerate(_SPLICE_CLS_NAMES) if c != "no_site" and true[:, i].sum() > 0]
            if cols:
                aps = [metrics[f"{name}_auprc_{_SPLICE_CLS_NAMES[i]}"] for i in cols]
                metrics[f"{name}_auprc_macro"] = sum(aps) / len(aps)
                w = [float(true[:, i].sum()) for i in cols]
                metrics[f"{name}_auprc_weighted"] = sum(a * wi / sum(w) for a, wi in zip(aps, w))

        rows: list[dict] = []
        if self.per_sample:
            # Row count follows the junction head's sample count, falling back
            # to splice_usage's channel count (as in PyTorch).
            n_s = None
            for acc in self.junction_per_sample.values():
                n_s = len(acc); break
            if n_s is None:
                for sums in self.usage_per_channel.values():
                    n_s = int(np.asarray(sums.n).shape[0]); break
            if n_s is not None:
                rows = [{} for _ in range(n_s)]
                for name, acc in self.junction_per_sample.items():
                    for s_idx in range(min(n_s, len(acc))):
                        for v, key in (("full", f"{name}_pearson_r"), ("nonzero", f"{name}_pearson_r_nonzero")):
                            r = float(acc[s_idx][v].r())
                            if math.isfinite(r):
                                rows[s_idx][key] = r
                for name, sums in self.usage_per_channel.items():
                    r = np.atleast_1d(sums.r())
                    for s_idx in range(min(n_s, r.shape[0])):
                        if math.isfinite(float(r[s_idx])):
                            rows[s_idx][f"{name}_pearson_r"] = float(r[s_idx])
        return metrics, rows, histograms


# Kept under its historical name: upstream's tests (and some callers) import it from here.
_label_params_for_heads = label_params_for_trainable_heads


def stop_frozen_parameter_gradients(params, labels):
    """Freeze weight derivatives while retaining derivatives through activations."""
    return jax.tree_util.tree_map(
        lambda parameter, label: parameter if label == "head" else jax.lax.stop_gradient(parameter),
        params, labels,
    )


def _resolve_lora_training_mode(
    model: CustomAlphaGenomeModel,
    *,
    heads_only: bool | None,
    train_lora: bool | None,
) -> tuple[bool, bool]:
    """Infer safe trainer defaults and reject accidental full-backbone LoRA runs."""
    has_backbone_lora = getattr(model, "_backbone_lora_config", None) is not None
    resolved_train_lora = has_backbone_lora if train_lora is None else train_lora
    if heads_only is None:
        resolved_heads_only = has_backbone_lora or resolved_train_lora
    else:
        resolved_heads_only = heads_only
    if resolved_train_lora and not resolved_heads_only:
        raise ValueError(
            "train_lora=True requires heads_only=True; otherwise the pretrained "
            "backbone would also be updated. Pass heads_only=True, or set "
            "train_lora=False for full-model finetuning."
        )
    return bool(resolved_heads_only), bool(resolved_train_lora)


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
    max_grad_norm: float | None = 1.0,
    heads_only: bool | None = None,
    checkpoint_dir: Path | None = None,
    organism: str = "HOMO_SAPIENS",
    best_metric: str = "valid_loss",
    best_metric_mode: str = "min",
    early_stopping_patience: int = 0,
    early_stopping_min_delta: float = 0.0,
    early_stopping_min_epochs: int = 0,
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
    log_every: int = 50,
    modality_weights: dict[str, float] | None = None,
    warmup_steps: int = 0,
    lr_schedule: str = "constant",
    compute_pearson: bool = True,
    metrics_per_sample: bool = False,
    train_lora: bool | None = None,
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
        max_grad_norm: Clip gradients to this global norm before the AdamW
            update. Defaults to 1.0 to match the PyTorch pipeline's
            ``--max-grad-norm`` default. Pass ``None`` (or 0) to disable
            clipping, which reproduces the pre-parity JAX behaviour.
        heads_only: If True, freeze the backbone and optimize selected heads
            only. ``None`` (the default) enables this automatically for a model
            created with ``backbone_lora_config`` or when ``train_lora=True``.
        checkpoint_dir: Optional output directory for ``best``/``last`` checkpoints.
        organism: Organism enum name used for model organism indexing.
        best_metric: Metric name used for best-checkpoint and early-stopping tracking.
        best_metric_mode: Improvement direction for ``best_metric`` (``min`` or ``max``).
        early_stopping_patience: Stop after this many non-improving epochs (0 disables).
        early_stopping_min_delta: Minimum metric change required to count as improvement.
        early_stopping_min_epochs: Do not stop before this many epochs have completed.
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
        log_every: Log per-head training loss (to ``checkpoint_dir``'s
            ``training_log.csv``, and to W&B if enabled) every this many
            optimizer steps. Matches alphagenome-pytorch's ``--log-every``.
        modality_weights: Optional per-head loss weights (``{head_id: weight}``,
            default 1.0 for heads not listed). Applied to the total loss
            optimized against; per-head CSV/W&B logging stays unweighted.
            Matches alphagenome-pytorch's ``--modality-weights``.
        warmup_steps: Linear LR warmup over this many optimizer steps before
            ``lr_schedule`` takes over. Matches alphagenome-pytorch's
            ``--warmup-steps``. 0 (default) disables warmup.
        lr_schedule: ``"constant"`` (default) or ``"cosine"`` (decays to 0 by
            the end of training). Matches alphagenome-pytorch's
            ``--lr-schedule``. See ``optimizer_utils.build_lr_schedule``.
        compute_pearson: If True (default), compute per-head validation
            Pearson correlation for rna_seq/splice_usage/splice_junctions
            (not splice_site, which is classification), pooled exactly
            across the whole validation set via streaming sufficient
            statistics. Matches alphagenome-pytorch's ``--no-val-pearson``
            (inverted: this flag defaults True like PyTorch's).
        metrics_per_sample: If True, additionally log per-track (= per
            biological sample, for splice_usage/splice_junctions) Pearson
            correlation. No effect if ``compute_pearson`` is False. Matches
            alphagenome-pytorch's ``--metrics-per-sample``. Default False.
        train_lora: Train LoRA adapter leaves in addition to the selected heads.
            ``None`` (the default) enables this automatically when the model was
            created with ``backbone_lora_config``. ``train_lora=True`` with
            ``heads_only=False`` is rejected to prevent accidental full-backbone
            updates.

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

    heads_only, train_lora = _resolve_lora_training_mode(
        model,
        heads_only=heads_only,
        train_lora=train_lora,
    )

    wb_config = {
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "max_grad_norm": max_grad_norm,
        "num_epochs": num_epochs,
        "batch_size": data_module._batch_size,
        "steps_per_epoch": steps_per_epoch,
        "total_train_steps": total_train_steps,
        "heads_only": heads_only,
        "train_lora": train_lora,
        "organism": organism,
        "num_devices": num_devices,
        "best_metric": best_metric,
        "best_metric_mode": best_metric_mode,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_epochs": early_stopping_min_epochs,
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
    head_specs_by_id = {spec.head_id: spec for spec in head_specs}
    if num_devices > 1 and not data_module._drop_last:
        raise ValueError(
            "Single-host multi-GPU training currently requires drop_last=True so every "
            "batch can be sharded evenly across devices."
        )
    if heads_only:
        model.freeze_backbone()

    lr_or_schedule = build_lr_schedule(learning_rate, warmup_steps, total_train_steps, lr_schedule)
    optimizer = create_optimizer(
        model._params,
        trainable_head_names=head_names,
        learning_rate=lr_or_schedule,
        weight_decay=weight_decay,
        heads_only=heads_only,
        train_lora=train_lora,
        gradient_clip_global_norm=max_grad_norm,
    )
    opt_state = optimizer.init(model._params)
    modality_weights = modality_weights or {}

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
            model._params, head_names, include_lora=train_lora,
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
                current_params = stop_frozen_parameter_gradients(current_params, _trainable_labels)
            predictions = model._predict(
                current_params,
                state,
                batch["sequences"],
                batch["organism_index"],
                negative_strand_mask=batch["negative_strand_mask"],
                strand_reindexing=batch["strand_reindexing"],
                **_predict_extra_kwargs(batch),
                requested_outputs=frozenset(),
            )
            head_losses = {}
            for head_name in head_names:
                head_loss_dict = loss_fns[head_name](
                    predictions[head_name], _head_batch(batch, head_name)
                )
                head_losses[head_name] = head_loss_dict["loss"]
            # head_losses (returned as aux for per-head CSV/W&B logging) stays
            # unweighted; only the optimized total applies modality_weights,
            # matching alphagenome-pytorch's loss_components vs. loss split.
            total_loss = sum(
                modality_weights.get(h, 1.0) * head_losses[h] for h in head_names
            )
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

    # Validation metric kinds per head, matching alphagenome-pytorch's
    # validate_multihead: genome tracks (profile/count Pearson), splice_usage,
    # splice_junctions (log1p/psi Pearson + auPRC) and splice_site
    # classification (per-class auPRC). Dispatch is by predefined head kind.
    _SPLICE_METRIC_KINDS = {
        "splice_sites_classification": "classification",
        "splice_sites_usage": "usage",
        "splice_sites_junction": "junction",
    }

    def _metric_kind(head_name):
        spec = head_specs_by_id.get(head_name)
        kind = getattr(spec, "kind", None)
        if kind in _SPLICE_METRIC_KINDS:
            return _SPLICE_METRIC_KINDS[kind]
        return "genome_tracks"

    from alphagenome_ft.custom_heads import get_junction_position_source
    from alphagenome_ft.finetune.splice_positions import build_junction_matrix

    def _junction_eval_target(head_name, batch, head_pred):
        """Target aligned to the positions the head scored: rebuilt from raw
        events in predicted mode (PyTorch's _get_junction_targets)."""
        if get_junction_position_source(head_name) is None:
            return batch[f"targets_{head_name}"]
        positions = head_pred["splice_site_positions"]
        return build_junction_matrix(
            positions,
            batch["junction_d_rel"],
            batch["junction_a_rel"],
            batch["junction_is_pos_strand"],
            batch["junction_counts"],
            max_splice_sites=positions.shape[-1],
        )

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
            requested_outputs=frozenset(),
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

        # Per-device metric inputs; the host concatenates devices and batches.
        # Large per-position tensors are reduced on device (genome tracks,
        # usage); sparse splice tensors come back whole for host-side masking.
        metric_inputs: dict = {}
        if compute_pearson:
            for head_name in head_names:
                head_pred = predictions[head_name]
                target = batch[f"targets_{head_name}"]
                kind = _metric_kind(head_name)
                if kind == "genome_tracks":
                    if "predictions_1bp" not in head_pred:
                        continue
                    pred = head_pred["predictions_1bp"].astype(jnp.float32)
                    true = target.astype(jnp.float32)
                    metric_inputs[head_name] = {
                        "profile_r": _profile_pearson_r(pred, true),
                        "pred_counts": pred.sum(axis=1),
                        "true_counts": true.sum(axis=1),
                    }
                elif kind == "usage":
                    stats = _usage_device_stats(
                        head_pred["predictions"].astype(jnp.float32), target.astype(jnp.float32),
                    )
                    metric_inputs[head_name] = jax.tree_util.tree_map(
                        lambda x: jax.lax.psum(x, axis_name="data"), stats,
                    )
                elif kind == "junction":
                    metric_inputs[head_name] = {
                        "pred": head_pred["predictions"].astype(jnp.float32),
                        "target": _junction_eval_target(head_name, batch, head_pred).astype(jnp.float32),
                        "positions": head_pred["splice_site_positions"],
                    }
                elif kind == "classification":
                    metric_inputs[head_name] = {
                        "logits": head_pred["logits"].astype(jnp.float32),
                        "target": target.astype(jnp.float32),
                    }
        return head_losses, metric_inputs

    if verbose:
        print("JIT-compiling step functions (first call will be slow)...")

    def aggregate_valid_loss(metrics: Mapping[str, float]) -> float | None:
        # Weighted the same way as the training loss (grad_step's loss_fn
        # above), so the metric driving best-checkpoint/early-stopping stays
        # consistent with what's actually being optimized; per-head
        # val_<head>_loss CSV columns stay unweighted.
        if not metrics:
            return None
        return float(sum(modality_weights.get(k, 1.0) * v for k, v in metrics.items()))

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
            if global_step >= total_train_steps:
                # Resumed a run that already reached its step budget: no extra
                # step, no re-validation, no checkpoint overwrite.
                print(f"  Already at requested training steps: {global_step}/{total_train_steps}")
                break
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

            # Where a resume continues from: the next epoch, unless the step budget
            # stopped this one part-way (then the rest of it is still to run).
            if stop_training and epoch_step < steps_per_epoch:
                resume_epoch_after, resume_step_after = epoch, epoch_step
            else:
                resume_epoch_after, resume_step_after = epoch + 1, 0

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
            pearson_metrics: dict[str, float] = {}
            per_sample_rows: list[dict] = []
            metric_histograms: dict[str, list] = {}
            if "valid" in data_module._intervals and len(data_module._intervals["valid"]) > 0:
                losses = {head: [] for head in head_names}
                val_metrics_acc = _ValidationMetrics(per_sample=metrics_per_sample, seed=seed)
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
                    head_losses, batch_metric_inputs = eval_step(
                        replicated_params, replicated_state, batch,
                    )
                    for head_name in head_names:
                        losses[head_name].append(float(np.asarray(head_losses[head_name])[0]))

                    def _host_all_devices(x):
                        x = np.asarray(x)
                        return x.reshape(-1, *x.shape[2:])

                    for head_name, inputs in batch_metric_inputs.items():
                        kind = _metric_kind(head_name)
                        if kind == "genome_tracks":
                            val_metrics_acc.add_genome_tracks(
                                head_name,
                                _host_all_devices(inputs["profile_r"]),
                                _host_all_devices(inputs["pred_counts"]),
                                _host_all_devices(inputs["true_counts"]),
                            )
                        elif kind == "usage":
                            # Already psum'd across devices: take device 0.
                            val_metrics_acc.add_usage(
                                head_name,
                                jax.tree_util.tree_map(lambda x: np.asarray(x)[0], inputs),
                            )
                        elif kind == "junction":
                            val_metrics_acc.add_junction(
                                head_name,
                                _host_all_devices(inputs["pred"]),
                                _host_all_devices(inputs["target"]),
                                _host_all_devices(inputs["positions"]),
                            )
                        elif kind == "classification":
                            val_metrics_acc.add_classification(
                                head_name,
                                _host_all_devices(inputs["logits"]),
                                _host_all_devices(inputs["target"]),
                            )

                valid_metrics = {
                    head: float(np.mean(values)) for head, values in losses.items() if values
                }
                print(
                    "  Validation metrics:",
                    ", ".join(f"{k}={v:.4f}" for k, v in valid_metrics.items()),
                )

                if compute_pearson:
                    pearson_metrics, per_sample_rows, metric_histograms = val_metrics_acc.finalize()
                    if pearson_metrics:
                        print(
                            "  Validation metrics (alphagenome-pytorch definitions):",
                            ", ".join(
                                f"{k}={v:.4f}" for k, v in pearson_metrics.items()
                                if not k.endswith("_std")
                            ),
                        )

            # Resolved before logging so the epoch row can carry is_best, as
            # alphagenome-pytorch's epoch_log.csv does.
            metric_label, metric_value = resolve_metric(best_metric, train_loss_avg, valid_metrics)
            epoch_is_best = bool(
                metric_value is not None and math.isfinite(metric_value)
                and is_improved(metric_value, best_value)
            )
            epoch_lr = float(
                lr_or_schedule(global_step) if callable(lr_or_schedule) else lr_or_schedule
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
                                "val_loss": aggregate_valid_loss(valid_metrics),
                                **{
                                    f"val_{head}_loss": v
                                    for head, v in valid_metrics.items()
                                },
                                **pearson_metrics,
                            }
                            if valid_metrics is not None
                            else {}
                        ),
                    },
                    per_sample=per_sample_rows,
                    histograms=metric_histograms,
                    learning_rate=epoch_lr,
                    is_best=epoch_is_best,
                )

            if metric_value is not None and math.isfinite(metric_value):
                if epoch_is_best:
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
                        _write_train_state(
                            checkpoint_dir / "best",
                            resume_epoch=resume_epoch_after, resume_epoch_step=resume_step_after,
                        )
                        _save_opt_state(checkpoint_dir / "best")
                else:
                    epochs_since_improvement += 1
            else:
                print(f"  Best metric ({metric_label}): unavailable")

            if checkpoint_dir:
                model._params = _unreplicate_tree(replicated_params)
                model._state = _unreplicate_tree(replicated_state)
                model.save_checkpoint(checkpoint_dir / "last", save_full_model=False)
                _write_train_state(
                    checkpoint_dir / "last",
                    resume_epoch=resume_epoch_after, resume_epoch_step=resume_step_after,
                )
                _save_opt_state(checkpoint_dir / "last")

            if (
                early_stopping_patience > 0
                and epoch >= early_stopping_min_epochs
                and epochs_since_improvement >= early_stopping_patience
            ):
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
