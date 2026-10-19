"""Memory-saving transformations must preserve trainable gradients and names."""

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from alphagenome_ft.backbone_lora import (
    BackboneLoRAConfig, BackboneLoRATransformerTower, BackboneLoRASequenceDecoder,
    _call_with_optional_training,
    forward_embeddings_with_backbone_lora,
)
from alphagenome_ft.finetune.train import stop_frozen_parameter_gradients
from alphagenome_research.model import model as model_lib


def test_frozen_weights_have_zero_gradients_but_lora_receives_decoder_gradient():
    params = {"adapter": jnp.array([[0.2, 0.3], [0.1, -0.4]]),
              "decoder": jnp.array([[2.0], [3.0]])}
    labels = {"adapter": "head", "decoder": "frozen"}
    x = jnp.array([[1.0, 2.0]])

    def objective(p):
        return jnp.square(x @ p["adapter"] @ p["decoder"]).sum()

    baseline = jax.grad(objective)(params)
    frozen = jax.grad(lambda p: objective(stop_frozen_parameter_gradients(p, labels)))(params)
    np.testing.assert_allclose(frozen["adapter"], baseline["adapter"])
    assert np.any(np.asarray(frozen["adapter"]) != 0)
    np.testing.assert_array_equal(frozen["decoder"], 0)


def _assert_same_trees(a, b):
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        np.testing.assert_allclose(x, y, rtol=2e-5, atol=2e-5)


def test_transformer_remat_preserves_parameters_outputs_and_gradients():
    # Real sequence and pair attention, at a short sequence for CPU testing.
    def transform(checkpointing):
        return hk.transform_with_state(lambda x: BackboneLoRATransformerTower(
            BackboneLoRAConfig(rank=2, gradient_checkpointing=checkpointing)
        )(x))

    plain, remat = transform(False), transform(True)
    x = jnp.ones((1, 32, 32), dtype=jnp.float32)
    key = jax.random.PRNGKey(17)
    params, state = plain.init(key, x)
    remat_params, remat_state = remat.init(key, x)
    _assert_same_trees(params, remat_params)
    _assert_same_trees(state, remat_state)
    _assert_same_trees(plain.apply(params, state, None, x), remat.apply(params, state, None, x))
    # Nonzero B ensures the backward path actually uses LoRA A as well.
    params = {name: {k: (jnp.full_like(v, 0.01) if k == "lora_b" else v)
                     for k, v in leaves.items()} for name, leaves in params.items()}
    def grads(fn):
        return jax.grad(lambda p: jnp.square(fn.apply(p, state, None, x)[0][0]).mean())(params)
    _assert_same_trees(grads(plain), grads(remat))


def test_1mb_remat_backbone_preserves_all_parameter_and_state_shapes():
    def shapes(checkpointing):
        forward = hk.transform_with_state(lambda x, organism: forward_embeddings_with_backbone_lora(
            x, organism, num_organisms=2,
            config=BackboneLoRAConfig(gradient_checkpointing=checkpointing),
        ))
        return jax.eval_shape(
            forward.init, jax.random.PRNGKey(19),
            jax.ShapeDtypeStruct((2, 1_048_576, 4), jnp.bfloat16),
            jax.ShapeDtypeStruct((2,), jnp.int32),
        )
    plain, remat = shapes(False), shapes(True)
    assert jax.tree_util.tree_structure(plain) == jax.tree_util.tree_structure(remat)
    for a, b in zip(jax.tree_util.tree_leaves(plain), jax.tree_util.tree_leaves(remat)):
        assert (a.shape, a.dtype) == (b.shape, b.dtype)
    assert sum(value.size for leaves in remat[0].values() for name, value in leaves.items()
               if name in ("lora_a", "lora_b")) == 308736


def test_decoder_remat_preserves_checkpoint_paths_and_input_gradients(monkeypatch):
    from alphagenome_research.model import convolutions

    # Test the seven-block wrapper independently of large native channel widths.
    class SmallUpResBlock(hk.Module):
        def __init__(self):
            super().__init__(name="up_res_block")

        def __call__(self, x, skip, *, is_training=False):
            return jnp.tanh(hk.Linear(4)(x) + skip)

    monkeypatch.setattr(convolutions, "UpResBlock", SmallUpResBlock)
    def transform(cls):
        return hk.transform_with_state(lambda x, skips: _call_with_optional_training(
            cls(name="sequence_decoder"), x, skips, is_training=False
        ))
    plain, remat = transform(model_lib.SequenceDecoder), transform(BackboneLoRASequenceDecoder)
    x = jnp.ones((1, 4, 4))
    skips = {f"bin_size_{b}": jnp.ones_like(x) for b in (64, 32, 16, 8, 4, 2, 1)}
    key = jax.random.PRNGKey(18)
    params, state = plain.init(key, x, skips)
    _assert_same_trees((params, state), remat.init(key, x, skips))
    _assert_same_trees(plain.apply(params, state, None, x, skips), remat.apply(params, state, None, x, skips))
    def grads(fn):
        return jax.grad(lambda values: fn.apply(params, state, None, values, skips)[0].sum())(x)
    _assert_same_trees(grads(plain), grads(remat))
