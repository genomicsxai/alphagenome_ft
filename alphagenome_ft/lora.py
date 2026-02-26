"""
LoRA (Low-Rank Adaptation) utilities for AlphaGenome finetuning.

Provides building blocks for parameter-efficient finetuning by adding small
low-rank matrices to selected linear layers while keeping the backbone frozen.

Typical usage pattern:
1. Write custom Haiku modules/heads that use LoRALinear instead of hk.Linear.
2. Pass frozen backbone embeddings as input to those modules.
3. Freeze backbone parameters with parameter_utils.freeze_backbone_keep_lora.
4. Train only the LoRA parameters (lora_a, lora_b).
"""
from __future__ import annotations

from dataclasses import dataclass

import haiku as hk
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from alphagenome_ft.parameter_utils import _keypath_to_str


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapters.

    Attributes:
        rank: Rank of the low-rank decomposition. Lower values mean fewer
            trainable parameters. Common values: 4, 8, 16.
        alpha: Scaling factor applied to the LoRA output. The effective
            scale is ``alpha / rank``. Setting alpha == rank gives scale 1.
    """
    rank: int = 8
    alpha: float = 1.0


class LoRALinear(hk.Module):
    """Linear layer augmented with a trainable low-rank adapter.

    Forward pass computes x @ W + (x @ A) @ B * (alpha / rank)
    where ``W`` is a base weight (typically kept frozen by the caller via
    ``jax.lax.stop_gradient`` or ``parameter_utils.freeze_backbone_keep_lora``)
    and ``A``, ``B`` are small trainable matrices.

    The parameter ``W`` is stored under the Haiku key ``"w"``, the adapters
    under ``"lora_a"`` and ``"lora_b"``.  This mirrors the naming used by
    AlphaGenome's own linear layers so that LoRA modules placed inside a
    backbone-matching name scope will coexist cleanly with loaded checkpoints.

    Args:
        out_dim: Output feature dimension.
        config: LoRA hyperparameters (rank and alpha). Defaults to
            ``LoRAConfig()`` (rank=8, alpha=1.0).
        with_bias: Whether to add a bias term to the base linear.
        name: Optional Haiku module name.
    """
    def __init__(
        self,
        out_dim: int,
        config: LoRAConfig | None = None,
        with_bias: bool = False,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.out_dim = out_dim
        self.config = config or LoRAConfig()
        self.with_bias = with_bias

    def __call__(self, x: Float[Array, '... D_in']) -> Float[Array, '... D_out']:
        in_dim = x.shape[-1]
        rank = self.config.rank
        alpha = self.config.alpha

        w = hk.get_parameter(
            'w',
            shape=(in_dim, self.out_dim),
            dtype=x.dtype,
            init=hk.initializers.VarianceScaling(),
        )
        a = hk.get_parameter(
            'lora_a',
            shape=(in_dim, rank),
            dtype=x.dtype,
            init=hk.initializers.RandomNormal(stddev=0.01),
        )
        b = hk.get_parameter(
            'lora_b',
            shape=(rank, self.out_dim),
            dtype=x.dtype,
            init=hk.initializers.Constant(0.0),
        )

        base = x @ w
        if self.with_bias:
            bias = hk.get_parameter(
                'b',
                shape=(self.out_dim,),
                dtype=x.dtype,
                init=hk.initializers.Constant(0.0),
            )
            base = base + bias

        delta = (x @ a) @ b * (alpha / rank)
        return base + delta


def get_lora_parameter_paths(params: PyTree) -> list[str]:
    """Return all parameter paths that correspond to LoRA adapter matrices.

    A path is considered a LoRA path when its final segment is ``lora_a`` or
    ``lora_b``.

    Args:
        params: Haiku parameter tree (e.g. ``model._params``).

    Returns:
        List of slash-delimited path strings, e.g.
        ``['my_head/lora_linear/lora_a', 'my_head/lora_linear/lora_b']``.
    """
    paths: list[str] = []

    def collect(path_tuple, value):
        if not hasattr(value, 'shape'):
            return
        path_str = _keypath_to_str(path_tuple)
        leaf = path_str.split('/')[-1]
        if leaf in ('lora_a', 'lora_b'):
            paths.append(path_str)

    jax.tree_util.tree_map_with_path(collect, params)
    return paths


def count_lora_parameters(params: PyTree) -> int:
    """Count the total number of trainable LoRA adapter elements.

    Args:
        params: Haiku parameter tree (e.g. ``model._params``).

    Returns:
        Total element count across all ``lora_a`` and ``lora_b`` arrays.
    """
    lora_paths = set(get_lora_parameter_paths(params))
    total = 0

    def accumulate(path_tuple, value):
        nonlocal total
        if not hasattr(value, 'size'):
            return
        if _keypath_to_str(path_tuple) in lora_paths:
            total += value.size

    jax.tree_util.tree_map_with_path(accumulate, params)
    return total
