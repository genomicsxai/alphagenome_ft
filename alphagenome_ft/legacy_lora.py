"""Loading backbone-LoRA checkpoints written before the upstream LoRA refactor.

Until the merge of ``genomicsxai/alphagenome_ft`` PR #13, this fork (and the
0.1.7 build used for the JAX reference runs) added backbone LoRA adapters by
monkeypatching ``MHABlock.__call__`` and stored them in sibling scopes::

    alphagenome/transformer_tower/mha_block[_N]/q_lora/{lora_a,lora_b}
    alphagenome/transformer_tower/mha_block[_N]/v_lora/{lora_a,lora_b}

``alphagenome_ft.backbone_lora`` (upstream layout, the one every new run uses)
keeps the adapters next to the pretrained projection weight::

    alphagenome/transformer_tower/mha_block[_N]/q_layer/{w,lora_a,lora_b}
    alphagenome/transformer_tower/mha_block[_N]/v_layer/{w,lora_a,lora_b}

The adapter maths is identical (A ~ U(+-1/sqrt(fan_in)), B = 0, delta =
(x A) B alpha/rank), so a legacy checkpoint only needs its keys renamed. This
module is the single, explicit exception that does that; nothing else in the
package knows about the old layout.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

_LEGACY_TO_CURRENT = {"q_lora": "q_layer", "k_lora": "k_layer", "v_lora": "v_layer"}


def remap_legacy_lora_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Rename legacy ``*_lora`` module keys of a flat Haiku params dict.

    Args:
        params: ``{module_path: {param_name: array}}`` as in ``model._params``
            or as returned by an orbax ``target=None`` restore.

    Returns:
        A new dict where every ``.../<x>_lora`` module is merged into the
        matching ``.../<x>_layer`` module (so ``lora_a``/``lora_b`` sit next to
        the pretrained ``w``). Keys that are not legacy LoRA scopes are copied
        unchanged; an already-current checkpoint is returned as-is.
    """
    out: dict[str, Any] = {}
    for key, value in params.items():
        parent, _, leaf = str(key).rpartition("/")
        if leaf in _LEGACY_TO_CURRENT:
            new_key = f"{parent}/{_LEGACY_TO_CURRENT[leaf]}" if parent else _LEGACY_TO_CURRENT[leaf]
            merged = dict(out.get(new_key, {}))
            merged.update(value)
            out[new_key] = merged
        elif key in out and isinstance(out[key], dict) and isinstance(value, Mapping):
            merged = dict(out[key])
            merged.update(value)
            out[key] = merged
        else:
            out[key] = dict(value) if isinstance(value, Mapping) else value
    return out


def has_legacy_lora_params(params: Mapping[str, Any]) -> bool:
    """True if any module key ends in a legacy ``q_lora``/``k_lora``/``v_lora``."""
    return any(str(key).rpartition("/")[2] in _LEGACY_TO_CURRENT for key in params)


def restore_into_model(model: Any, checkpoint_dir: str | Path) -> dict[str, int]:
    """Restore a delta checkpoint (heads + LoRA leaves) into ``model``, any LoRA layout.

    ``model`` must have been built with ``create_model_with_heads(...,
    backbone_lora_config=...)``. The checkpoint's own structure is accepted as
    is (``target=None``) and every returned module is overlaid on the model's
    freshly initialised/pretrained params, so the pretrained backbone stays
    untouched. Legacy ``q_lora``/``v_lora`` keys are renamed first.

    Returns ``{"modules": n, "lora_leaves": n}`` for logging/assertions.
    """
    import orbax.checkpoint as ocp

    checkpoint_path = Path(checkpoint_dir).resolve() / "checkpoint"
    restored = ocp.StandardCheckpointer().restore(str(checkpoint_path), target=None)
    if isinstance(restored, (tuple, list)) and len(restored) == 2:
        loaded_params, loaded_state = restored
    elif isinstance(restored, (tuple, list)) and len(restored) == 1:
        loaded_params, loaded_state = restored[0], {}
    else:
        loaded_params, loaded_state = restored, {}

    loaded_params = remap_legacy_lora_params(loaded_params)
    n_lora = 0
    for module_key, leaves in loaded_params.items():
        target = model._params.get(module_key)
        if target is None:
            model._params[module_key] = dict(leaves)
        else:
            # Haiku params may be immutable mappings; rebuild the module dict.
            model._params[module_key] = {**target, **leaves}
        n_lora += sum(1 for name in leaves if name in ("lora_a", "lora_b"))
    if loaded_state:
        for module_key, leaves in loaded_state.items():
            target = model._state.get(module_key)
            model._state[module_key] = {**target, **leaves} if target is not None else dict(leaves)
    return {"modules": len(loaded_params), "lora_leaves": n_lora}
