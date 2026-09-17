"""LoRA adapters for AlphaGenome transformer query/value projections.

The modules in this file intentionally retain AlphaGenome's Haiku module names
(``transformer_tower/mha_block*/q_layer`` and ``v_layer``).  Consequently the
pretrained ``w`` leaves merge at their original paths while the new
``lora_a``/``lora_b`` leaves remain independently trainable.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence

import haiku as hk
import jax
import jax.numpy as jnp

from alphagenome_research.model import attention
from alphagenome_research.model import convolutions
from alphagenome_research.model import embeddings as embeddings_module
from alphagenome_research.model import layers
from alphagenome_research.model import model as model_lib


_TARGET_ALIASES = {
    "q_proj": "q_layer",
    "q_layer": "q_layer",
    "v_proj": "v_layer",
    "v_layer": "v_layer",
}


@dataclasses.dataclass(frozen=True)
class BackboneLoRAConfig:
    """Configuration for LoRA in AlphaGenome sequence-attention projections."""

    rank: int = 8
    alpha: float = 16.0
    targets: tuple[str, ...] = ("q_proj", "v_proj")
    gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {self.rank}.")
        unknown = sorted(set(self.targets) - set(_TARGET_ALIASES))
        if unknown:
            raise ValueError(
                f"Unsupported backbone LoRA targets {unknown}; use q_proj and/or v_proj."
            )
        if not self.targets:
            raise ValueError("At least one backbone LoRA target is required.")

    @property
    def jax_targets(self) -> tuple[str, ...]:
        """Return deduplicated Haiku projection names in user-specified order."""
        return tuple(dict.fromkeys(_TARGET_ALIASES[target] for target in self.targets))


def _call_with_optional_training(module, *args, is_training: bool):
    """Call old and new alphagenome_research module signatures."""
    try:
        return module(*args, is_training=is_training)
    except TypeError as exc:
        if "is_training" not in str(exc):
            raise
        return module(*args)


def _call_output_embedder(
    module,
    x,
    organism_index,
    *,
    is_training: bool,
    skip_x=None,
):
    """Call OutputEmbedder across the pre/post-is_training APIs."""
    try:
        return module(
            x,
            organism_index,
            is_training=is_training,
            skip_x=skip_x,
        )
    except TypeError as exc:
        message = str(exc)
        if "is_training" not in message and "skip_x" not in message:
            raise
        if skip_x is None:
            return module(x, organism_index)
        return module(x, organism_index, skip_x)


class BackboneLoRALinear(hk.Module):
    """A pretrained Haiku linear projection plus a zero-initialized LoRA delta."""

    def __init__(
        self,
        output_size: int,
        *,
        rank: int,
        alpha: float,
        name: str,
    ) -> None:
        super().__init__(name=name)
        self._output_size = output_size
        self._rank = rank
        self._scale = alpha / rank
        if rank > output_size:
            raise ValueError(
                f"LoRA rank {rank} must be <= output features {output_size}."
            )

    def __call__(self, x):
        input_size = x.shape[-1]
        # Keep trainable/master parameters in fp32, like the pretrained model,
        # and cast only for the mixed-precision matmuls.
        w = hk.get_parameter(
            "w",
            (input_size, self._output_size),
            dtype=jnp.float32,
            init=hk.initializers.VarianceScaling(),
        ).astype(x.dtype)
        lora_a = hk.get_parameter(
            "lora_a",
            (input_size, self._rank),
            dtype=jnp.float32,
            # Equivalent to torch.nn.Linear(input_size, rank)'s default
            # kaiming_uniform_(a=sqrt(5)): U[-1/sqrt(fan_in), 1/sqrt(fan_in)].
            init=hk.initializers.RandomUniform(
                minval=-1.0 / math.sqrt(input_size),
                maxval=1.0 / math.sqrt(input_size),
            ),
        ).astype(x.dtype)
        lora_b = hk.get_parameter(
            "lora_b",
            (self._rank, self._output_size),
            dtype=jnp.float32,
            init=hk.initializers.Constant(0.0),
        ).astype(x.dtype)
        return x @ w + ((x @ lora_a) @ lora_b) * self._scale


class BackboneLoRAMHABlock(hk.Module):
    """AlphaGenome MHABlock with opt-in LoRA on q_layer and v_layer."""

    def __init__(self, config: BackboneLoRAConfig, name: str | None = None):
        super().__init__(name=name or "mha_block")
        self._config = config

    @hk.transparent
    def _projection(self, name: str, output_size: int, x):
        if name in self._config.jax_targets:
            return BackboneLoRALinear(
                output_size,
                rank=self._config.rank,
                alpha=self._config.alpha,
                name=name,
            )(x)
        return hk.Linear(output_size, with_bias=False, name=name)(x)

    def __call__(self, x, attention_bias, *, is_training: bool = False):
        batch_size, seq_len, _ = x.shape
        h = _call_with_optional_training(
            layers.RMSBatchNorm(), x, is_training=is_training
        )
        q = layers.LayerNorm(name="norm_q")(
            self._projection("q_layer", 8 * 128, h).reshape(
                batch_size, seq_len, 8, 128
            )
        )
        k = layers.LayerNorm(name="norm_k")(
            hk.Linear(128, with_bias=False, name="k_layer")(h).reshape(
                batch_size, seq_len, 1, 128
            )
        )
        v = layers.LayerNorm(name="norm_v")(
            self._projection("v_layer", 192, h).reshape(
                batch_size, seq_len, 1, 192
            )
        )
        q = attention.apply_rope(q, None, max_position=8192)
        k = attention.apply_rope(k, None, max_position=8192)

        logits = jnp.einsum(
            "bshc,bS1c->bhsS",
            q,
            k,
            precision=jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
            preferred_element_type=jnp.float32,
        ) / math.sqrt(128.0)
        logits = (logits + attention_bias).astype(jnp.float32)
        logits = jnp.tanh(logits / 5.0) * 5.0
        weights = jax.nn.softmax(logits, axis=-1)
        y = jnp.einsum(
            "bhsS,bS1c->bshc",
            weights,
            v,
            precision=jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
        ).astype(q.dtype)
        y = hk.Linear(
            x.shape[-1],
            name="linear_embedding",
            w_init=hk.initializers.TruncatedNormal(stddev=1e-6),
        )(y.reshape(batch_size, seq_len, -1))
        return _call_with_optional_training(
            layers.RMSBatchNorm(), y, is_training=is_training
        )


class BackboneLoRATransformerTower(hk.Module):
    """Nine-block AlphaGenome transformer with LoRA-aware sequence MHA."""

    def __init__(self, config: BackboneLoRAConfig, name: str | None = None):
        super().__init__(name=name or "transformer_tower")
        self._config = config

    def __call__(self, x, *, is_training: bool = False):
        pair_x = None
        for index in range(9):
            def block(x, pair_x):
                if index % 2 == 0:
                    pair_x = attention.PairUpdateBlock()(x, pair_x)
                mha_bias = _call_with_optional_training(
                    attention.AttentionBiasBlock(), pair_x, is_training=is_training
                )
                x = x + BackboneLoRAMHABlock(self._config, name="mha_block")(
                    x, mha_bias, is_training=is_training
                )
                x = x + _call_with_optional_training(
                    attention.MLPBlock(), x, is_training=is_training
                )
                return x, pair_x

            # Haiku's remat preserves RNG, state and parameter-name counters.
            # Recompute each block's attention intermediates during backward.
            x, pair_x = (hk.remat(block) if self._config.gradient_checkpointing else block)(x, pair_x)
        return x, pair_x


class BackboneLoRASequenceDecoder(hk.Module):
    """Native decoder with per-block activation checkpointing and original names."""

    def __init__(self, name="sequence_decoder"):
        super().__init__(name=name)

    def __call__(self, x, intermediates, *, is_training=False):
        for bin_size in [64, 32, 16, 8, 4, 2, 1]:
            block = convolutions.UpResBlock()
            x = hk.remat(lambda values, skip: _call_with_optional_training(
                block, values, skip, is_training=is_training
            ))(x, intermediates[f"bin_size_{bin_size}"])
        return x


def forward_embeddings_with_backbone_lora(
    dna_sequence,
    organism_index,
    *,
    num_organisms: int,
    config: BackboneLoRAConfig,
    is_training: bool = False,
):
    """Run the AlphaGenome trunk with Q/V LoRA and return standard embeddings."""
    with hk.name_scope("alphagenome"):
        trunk, intermediates = _call_with_optional_training(
            model_lib.SequenceEncoder(), dna_sequence, is_training=is_training
        )
        if num_organisms >= 1:
            if hasattr(embeddings_module, "create_default_embedding"):
                organism_embedder = embeddings_module.create_default_embedding(
                    num_organisms, trunk.shape[-1]
                )
            else:
                organism_embedder = hk.Embed(num_organisms, trunk.shape[-1])
            trunk += organism_embedder(organism_index)[:, None, :]

        trunk, pair_activations = BackboneLoRATransformerTower(config)(
            trunk, is_training=is_training
        )
        decoded = _call_with_optional_training(
            (BackboneLoRASequenceDecoder() if config.gradient_checkpointing
             else model_lib.SequenceDecoder()),
            trunk,
            intermediates,
            is_training=is_training,
        )
        embeddings_128bp = _call_output_embedder(
            embeddings_module.OutputEmbedder(num_organisms),
            trunk,
            organism_index,
            is_training=is_training,
        )
        embeddings_1bp = _call_output_embedder(
            embeddings_module.OutputEmbedder(num_organisms),
            decoded,
            organism_index,
            is_training=is_training,
            skip_x=embeddings_128bp,
        )
        embeddings_pair = embeddings_module.OutputPair(num_organisms)(
            pair_activations, organism_index
        )
    return embeddings_module.Embeddings(
        embeddings_1bp=embeddings_1bp,
        embeddings_128bp=embeddings_128bp,
        embeddings_pair=embeddings_pair,
    )


def expected_lora_parameter_count(
    *,
    input_size: int = 1536,
    rank: int = 8,
    targets: Sequence[str] = ("q_proj", "v_proj"),
    num_blocks: int = 9,
) -> int:
    """Return the expected number of Q/V adapter scalars."""
    normalized = tuple(dict.fromkeys(_TARGET_ALIASES[target] for target in targets))
    output_sizes = {"q_layer": 1024, "v_layer": 192}
    return num_blocks * sum(
        input_size * rank + rank * output_sizes[target] for target in normalized
    )


__all__ = [
    "BackboneLoRAConfig",
    "BackboneLoRALinear",
    "BackboneLoRAMHABlock",
    "BackboneLoRATransformerTower",
    "expected_lora_parameter_count",
    "forward_embeddings_with_backbone_lora",
]
