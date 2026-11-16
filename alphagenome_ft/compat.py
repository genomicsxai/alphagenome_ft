"""Compatibility helpers for calling ``alphagenome_research`` Haiku modules.

``alphagenome_research`` builds from June 2026 onwards (0.3.0+) added a required,
keyword-only ``is_training`` argument to the trunk modules (``SequenceEncoder``,
``TransformerTower``, ``SequenceDecoder``, ``OutputEmbedder``, ``RMSBatchNorm`` and
friends). Earlier builds reject that argument. :func:`call_with_is_training` lets a
single call site work against both APIs.
"""

from __future__ import annotations

import inspect
from typing import Any


def accepts_is_training(module: Any) -> bool:
    """Return True if ``module.__call__`` takes an ``is_training`` parameter."""
    return "is_training" in inspect.signature(module.__call__).parameters


def call_with_is_training(
    module: Any, *args: Any, is_training: bool = False, **kwargs: Any
) -> Any:
    """Call ``module(*args, **kwargs)``, passing ``is_training`` only when it is accepted.

    Pass optional module arguments such as ``OutputEmbedder``'s ``skip_x`` as keywords:
    newer ``alphagenome_research`` releases make them keyword-only, and keywords are
    accepted by both the old and the new signatures.

    Args:
        module: A constructed Haiku module (or any object with a ``__call__``).
        *args: Positional arguments forwarded to the module.
        is_training: Forwarded as ``is_training`` when the module accepts it. ``False``
            matches ``AlphaGenome.__call__``'s default (inference-mode batch-norm
            statistics), which is the behaviour fine-tuning with a frozen trunk wants.
        **kwargs: Keyword arguments forwarded to the module.

    Returns:
        Whatever the module returns.
    """
    if accepts_is_training(module):
        kwargs["is_training"] = is_training
    return module(*args, **kwargs)


__all__ = ["accepts_is_training", "call_with_is_training"]
