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


def is_trainable_head_path(path_str: str, trainable_heads: set[str]) -> bool:
    """Return True if ``path_str`` belongs to one of the named heads."""
    for head_name in trainable_heads:
        if f"/head/{head_name}/" in path_str or path_str.startswith(f"head/{head_name}/"):
            return True
    return False


def label_params_for_trainable_heads(params: PyTree, trainable_head_names: Sequence[str]) -> PyTree:
    """Label each leaf ``\"head\"`` (train) vs ``\"frozen\"`` for :func:`optax.multi_transform`."""
    head_set = {str(n) for n in trainable_head_names}

    def label_fn(path, _value):
        ps = parameter_path_to_str(path)
        return "head" if is_trainable_head_path(ps, head_set) else "frozen"

    return jax.tree_util.tree_map_with_path(label_fn, params)


def assert_trainable_head_params_exist(params: PyTree, trainable_head_names: Sequence[str]) -> None:
    """Raise if no head parameters match ``trainable_head_names``."""
    head_set = {str(n) for n in trainable_head_names}
    head_paths = parameter_utils.get_head_parameter_paths(params)
    matched = [p for p in head_paths if is_trainable_head_path(p, head_set)]
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
        gradient_clip_global_norm: If set, prepend ``optax.clip_by_global_norm``.

    Returns:
        An ``optax.GradientTransformation``.
    """
    inner = _build_adam_or_adamw(
        learning_rate,
        optimizer_type=optimizer_type,
        weight_decay=weight_decay,
    )

    if heads_only:
        assert_trainable_head_params_exist(params, trainable_head_names)
        labels = label_params_for_trainable_heads(params, trainable_head_names)
        inner = optax.multi_transform(
            {"head": inner, "frozen": optax.set_to_zero()},
            labels,
        )

    if gradient_clip_global_norm is not None:
        return optax.chain(
            optax.clip_by_global_norm(gradient_clip_global_norm),
            inner,
        )
    return inner
