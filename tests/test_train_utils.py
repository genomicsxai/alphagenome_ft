"""Small regression tests for the fine-tuning trainer helpers."""

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from alphagenome_ft.finetune.train import _resolve_lora_training_mode, _shard_batch


def test_shard_batch_adds_device_and_per_device_axes():
    batch = {
        "sequences": jnp.zeros((4, 8, 4)),
        "negative_strand_mask": jnp.zeros((4,), dtype=bool),
    }

    sharded = _shard_batch(batch, num_devices=2)

    assert sharded["sequences"].shape == (2, 2, 8, 4)
    assert sharded["negative_strand_mask"].shape == (2, 2)


def test_shard_batch_rejects_nondivisible_global_batch():
    with pytest.raises(ValueError, match="not divisible"):
        _shard_batch({"sequences": jnp.zeros((3, 8, 4))}, num_devices=2)


def test_lora_training_mode_defaults_to_adapters_and_heads_only():
    model = SimpleNamespace(_backbone_lora_config=object())

    assert _resolve_lora_training_mode(
        model, heads_only=None, train_lora=None
    ) == (True, True)


def test_lora_training_mode_rejects_unfrozen_backbone():
    model = SimpleNamespace(_backbone_lora_config=object())

    with pytest.raises(ValueError, match="requires heads_only=True"):
        _resolve_lora_training_mode(
            model, heads_only=False, train_lora=True
        )


def test_non_lora_training_defaults_remain_full_model():
    model = SimpleNamespace(_backbone_lora_config=None)

    assert _resolve_lora_training_mode(
        model, heads_only=None, train_lora=None
    ) == (False, False)
