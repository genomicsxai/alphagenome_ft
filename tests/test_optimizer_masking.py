"""Tests that heads-only optimizers leave backbone parameters unchanged."""

import jax
import jax.numpy as jnp
import optax
import pytest

from alphagenome_ft.optimizer_utils import create_optimizer, label_params_for_trainable_heads
from tests.kaggle_util import kaggle_credentials_available


def test_heads_only_optimizer_fake_param_tree():
    """No model download: frozen leaves get zero updates from multi_transform."""
    params = {
        "alphagenome": {
            "sequence_encoder": {"w": jnp.ones((2, 3))},
            "head": {"my_task": {"b": jnp.array([0.5])}},
        }
    }
    opt = create_optimizer(
        params,
        trainable_head_names=("my_task",),
        learning_rate=1e-2,
        weight_decay=None,
        heads_only=True,
    )
    state = opt.init(params)
    labels = label_params_for_trainable_heads(params, ("my_task",))

    def loss_fn(p):
        return sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(p))

    grads = jax.grad(loss_fn)(params)
    updates, _ = opt.update(grads, state, params)

    def assert_frozen_zero(u, lab):
        if lab == "frozen":
            assert jnp.allclose(u, 0.0)

    jax.tree_util.tree_map(assert_frozen_zero, updates, labels)

    new_params = optax.apply_updates(params, updates)

    def assert_head_moved(p_old, p_new, lab):
        if lab == "head":
            assert not jnp.allclose(p_old, p_new)

    jax.tree_util.tree_map(assert_head_moved, params, new_params, labels)


@pytest.mark.skipif(
    not kaggle_credentials_available(),
    reason="Kaggle credentials required for model fixture (~/.kaggle/kaggle.json or env)",
)
def test_heads_only_optimizer_zeros_backbone_updates(wrapped_model_with_head):
    """Frozen leaves must get zero Optax updates even with non-zero gradients."""
    model = wrapped_model_with_head
    params = model._params
    head_id = "test_mpra_head"

    opt = create_optimizer(
        params,
        trainable_head_names=(head_id,),
        learning_rate=1.0,
        weight_decay=None,
        heads_only=True,
    )
    state = opt.init(params)
    labels = label_params_for_trainable_heads(params, (head_id,))

    def fake_loss(p):
        leaves = jax.tree_util.tree_leaves(p)
        return sum(jnp.sum(x**2) for x in leaves)

    grads = jax.grad(fake_loss)(params)
    updates, _ = opt.update(grads, state, params)

    def assert_frozen_update_zero(u, lab):
        if lab == "frozen":
            assert jnp.allclose(u, 0.0)

    jax.tree_util.tree_map(assert_frozen_update_zero, updates, labels)

    # Head subtree includes Haiku state leaves that can get zero Adam updates; require
    # non-zero updates somewhere under the head label (trainable weights).
    head_update_energy = [0.0]

    def sum_head_updates(u, lab):
        if lab == "head":
            head_update_energy[0] += float(
                jnp.sum(jnp.square(u.astype(jnp.float32)))
            )

    jax.tree_util.tree_map(sum_head_updates, updates, labels)
    assert head_update_energy[0] > 1e-6, (
        "Expected non-zero Adam updates for at least some head parameters"
    )

    _ = optax.apply_updates(params, updates)


@pytest.mark.skipif(
    not kaggle_credentials_available(),
    reason="Kaggle credentials required for model fixture (~/.kaggle/kaggle.json or env)",
)
def test_freeze_except_head_sets_hint(wrapped_model_with_head):
    model = wrapped_model_with_head
    assert model._heads_only_finetune_default is False
    model.freeze_except_head("test_mpra_head")
    assert model._heads_only_finetune_default is True
    assert model._heads_only_trainable_head_names == ("test_mpra_head",)
