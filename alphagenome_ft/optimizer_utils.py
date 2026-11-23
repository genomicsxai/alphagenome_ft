"""Optax helpers for fine-tuning: mask updates so only selected heads (or full model) train.

``parameter_utils.freeze_*`` applies ``jax.lax.stop_gradient`` to parameter *values* at call time.
That does **not** stop gradients on later ``jax.grad`` / ``optax`` steps. For true freezing during
training, use :func:`create_optimizer` with ``heads_only=True`` (or pass explicit masks).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import optax
from jaxtyping import PyTree

from alphagenome_ft import parameter_utils


def parameter_path_to_str(path_tuple: tuple) -> str:
    """Public alias for converting a JAX keypath tuple to a slash-separated string."""
    return parameter_utils._keypath_to_str(path_tuple)


def is_trainable_head_path(
    path_str: str,
    trainable_heads: set[str],
    top_level_heads: frozenset[str] | set[str] = frozenset(),
) -> bool:
    """Return True if ``path_str`` belongs to one of the named heads.

    A head may live at top level (``head/<name>/...``, custom heads) or inside
    the model scope (``.../head/<name>/...``). For names in ``top_level_heads``
    only the top-level copy matches, so a pretrained standard head with the same
    name (e.g. ``alphagenome/head/rna_seq``, unused once a custom ``rna_seq``
    head exists) stays frozen.
    """
    for head_name in trainable_heads:
        if path_str.startswith(f"head/{head_name}/"):
            return True
        if head_name not in top_level_heads and f"/head/{head_name}/" in path_str:
            return True
    return False


def _top_level_head_names(path_strs, trainable_heads: set[str]) -> frozenset[str]:
    return frozenset(
        n for n in trainable_heads if any(p.startswith(f"head/{n}/") for p in path_strs)
    )


def is_lora_parameter_path(path_str: str) -> bool:
    """Return whether a path points to a LoRA A/B matrix."""
    return path_str.rsplit("/", 1)[-1] in {"lora_a", "lora_b"}


def label_params_for_trainable_heads(
    params: PyTree,
    trainable_head_names: Sequence[str],
    *,
    include_lora: bool = False,
) -> PyTree:
    """Label trainable heads/adapters versus frozen parameter leaves."""
    head_set = {str(n) for n in trainable_head_names}
    all_paths = [parameter_path_to_str(p) for p, _ in jax.tree_util.tree_flatten_with_path(params)[0]]
    top_level = _top_level_head_names(all_paths, head_set)

    def label_fn(path, _value):
        ps = parameter_path_to_str(path)
        trainable = is_trainable_head_path(ps, head_set, top_level) or (
            include_lora and is_lora_parameter_path(ps)
        )
        return "head" if trainable else "frozen"

    return jax.tree_util.tree_map_with_path(label_fn, params)


def assert_trainable_head_params_exist(params: PyTree, trainable_head_names: Sequence[str]) -> None:
    """Raise if no head parameters match ``trainable_head_names``."""
    head_set = {str(n) for n in trainable_head_names}
    head_paths = parameter_utils.get_head_parameter_paths(params)
    top_level = _top_level_head_names(head_paths, head_set)
    matched = [p for p in head_paths if is_trainable_head_path(p, head_set, top_level)]
    if not matched:
        sample = ", ".join(head_paths[:5]) if head_paths else "<none>"
        raise ValueError(
            "No parameters matched trainable heads "
            f"{sorted(head_set)!r} for heads-only optimizer. "
            f"Sample head paths: {sample}"
        )


def _build_adam_or_adamw(
    learning_rate: Any,
    *,
    optimizer_type: str,
    weight_decay: float | None,
) -> optax.GradientTransformation:
    ot = optimizer_type.lower()
    if ot == "adamw":
        if weight_decay is not None:
            return optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
        return optax.adamw(learning_rate=learning_rate)
    if ot == "adam":
        inner = optax.adam(learning_rate)
        if weight_decay is not None:
            return optax.chain(optax.add_decayed_weights(weight_decay), inner)
        return inner
    raise ValueError(
        f"optimizer_type must be 'adam' or 'adamw', got {optimizer_type!r}"
    )


def create_optimizer(
    params: PyTree,
    *,
    trainable_head_names: Sequence[str],
    learning_rate: Any,
    weight_decay: float | None = None,
    heads_only: bool = False,
    optimizer_type: str = "adamw",
    gradient_clip_global_norm: float | None = None,
    train_lora: bool = False,
) -> optax.GradientTransformation:
    """Build an Optax optimizer, optionally applying zero updates outside trainable heads.

    When ``heads_only`` is True, only parameters under the given head name(s) receive
    Adam/AdamW updates (including weight decay on those leaves only). All other leaves use
    ``optax.set_to_zero()``, so backbone weights stay fixed even when the forward pass still
    depends on them.

    Args:
        params: Parameter PyTree (structure used for labels; typically ``model._params``).
        trainable_head_names: Head ids to keep trainable when ``heads_only`` is True.
        learning_rate: Scalar LR or Optax schedule.
        weight_decay: Optional L2 / AdamW decay. ``None`` uses Optax defaults (no extra decay
            for AdamW beyond its default).
        heads_only: If True, apply ``optax.multi_transform`` head vs frozen masking.
        optimizer_type: ``\"adamw\"`` or ``\"adam\"``.
        gradient_clip_global_norm: If set, clip the global norm of trainable
            leaves. Frozen gradients are excluded, matching
            ``torch.nn.utils.clip_grad_norm_(trainable_params, ...)``.
        train_lora: Include all ``lora_a``/``lora_b`` leaves in the trainable mask.

    Returns:
        An ``optax.GradientTransformation``.
    """
    inner = _build_adam_or_adamw(
        learning_rate,
        optimizer_type=optimizer_type,
        weight_decay=weight_decay,
    )

    def with_clipping(
        tx: optax.GradientTransformation,
    ) -> optax.GradientTransformation:
        if gradient_clip_global_norm is None or gradient_clip_global_norm <= 0:
            return tx
        return optax.chain(
            optax.clip_by_global_norm(gradient_clip_global_norm),
            tx,
        )

    if heads_only:
        assert_trainable_head_params_exist(params, trainable_head_names)
        labels = label_params_for_trainable_heads(
            params,
            trainable_head_names,
            include_lora=train_lora,
        )
        if train_lora and not any(
            is_lora_parameter_path(parameter_utils._keypath_to_str(path))
            for path, _value in jax.tree_util.tree_flatten_with_path(params)[0]
        ):
            raise ValueError("train_lora=True but no lora_a/lora_b parameters were found.")
        inner = optax.multi_transform(
            {"head": with_clipping(inner), "frozen": optax.set_to_zero()},
            labels,
        )
        return inner

    return with_clipping(inner)


def build_lr_schedule(
    base_lr: float, warmup_steps: int, total_steps: int, schedule: str,
) -> float | Any:
    """Build a learning-rate value/schedule for :func:`create_optimizer`.

    Mirrors alphagenome-pytorch's ``create_lr_scheduler`` (a ``LambdaLR`` multiplicative
    factor) formula exactly, rather than reaching for ``optax.warmup_cosine_decay_schedule``
    (whose endpoint semantics aren't guaranteed to match PyTorch's bit-for-bit): linear
    warmup from 0 to ``base_lr`` over ``warmup_steps``, then either held constant or cosine
    decayed to 0 by ``total_steps``.

    Returns the plain ``base_lr`` float unchanged when ``warmup_steps == 0`` and
    ``schedule == "constant"`` (today's default), so existing callers that never pass these
    flags see no behavior change and no schedule-tracing overhead.
    """
    if warmup_steps == 0 and schedule == "constant":
        return base_lr
    if schedule not in ("constant", "cosine"):
        raise ValueError(f"schedule must be 'constant' or 'cosine', got {schedule!r}.")

    def lr_fn(step):
        warmup_factor = jax.numpy.minimum(step / max(warmup_steps, 1), 1.0)
        if schedule == "constant":
            decay_factor = 1.0
        else:
            progress = jax.numpy.clip(
                (step - warmup_steps) / max(total_steps - warmup_steps, 1), 0.0, 1.0,
            )
            decay_factor = 0.5 * (1.0 + jax.numpy.cos(jax.numpy.pi * progress))
        factor = jax.numpy.where(step < warmup_steps, warmup_factor, decay_factor)
        return base_lr * factor

    return lr_fn
