"""Small regression tests for the fine-tuning trainer helpers."""

import jax.numpy as jnp
import pytest

from alphagenome_ft.finetune.train import _shard_batch


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
