"""Tests for alphagenome_ft.compat (no model download required).

Covers both alphagenome_research APIs with dummy modules, and traces the installed
alphagenome_research trunk modules through the helper so CI exercises whichever
signature is actually installed.
"""

from types import SimpleNamespace

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from alphagenome_research.model import embeddings as embeddings_module
from alphagenome_research.model import model as model_lib

from alphagenome_ft.compat import accepts_is_training, call_with_is_training
from alphagenome_ft.custom_forward import forward_with_encoder_output


class _LegacyModule(hk.Module):
    """Pre-June-2026 style: no ``is_training`` parameter."""

    def __call__(self, x, skip_x=None):
        return x if skip_x is None else x + skip_x


class _ModernModule(hk.Module):
    """June-2026+ style: required keyword-only ``is_training``, keyword-only ``skip_x``."""

    def __call__(self, x, *, is_training: bool, skip_x=None):
        out = x if skip_x is None else x + skip_x
        return out * (2.0 if is_training else 1.0)


def _apply(fn, *args):
    return hk.without_apply_rng(hk.transform(fn)).apply({}, *args)


def test_legacy_module_does_not_receive_is_training():
    def fn(x):
        # is_training must be dropped; skip_x must still be forwarded.
        return call_with_is_training(_LegacyModule(), x, skip_x=x, is_training=True)

    np.testing.assert_array_equal(_apply(fn, jnp.ones(3)), 2 * np.ones(3))


def test_modern_module_receives_is_training_with_false_default():
    def fn(x):
        return (
            call_with_is_training(_ModernModule(), x, skip_x=x, is_training=True),
            call_with_is_training(_ModernModule(), x, skip_x=x),
        )

    on, off = _apply(fn, jnp.ones(3))
    np.testing.assert_array_equal(on, 4 * np.ones(3))
    np.testing.assert_array_equal(off, 2 * np.ones(3))


def test_accepts_is_training_matches_installed_alphagenome_research():
    """Whichever alphagenome_research is installed, detection must agree with a direct call."""

    def fn(x):
        encoder = model_lib.SequenceEncoder()
        detected = accepts_is_training(encoder)
        try:
            encoder(x, is_training=False)
            direct = True
        except TypeError:
            direct = False
        return detected, direct

    detected, direct = hk.transform_with_state(fn).apply(
        *hk.transform_with_state(fn).init(jax.random.PRNGKey(0), jnp.zeros((1, 256, 4))),
        None,
        jnp.zeros((1, 256, 4)),
    )[0]
    assert detected == direct


def test_installed_trunk_modules_trace_through_helper():
    """SequenceEncoder and OutputEmbedder(skip_x=...) trace on the installed API (no weights)."""

    @hk.transform_with_state
    def fn(seq, org):
        with hk.name_scope("alphagenome"):
            trunk, _ = call_with_is_training(model_lib.SequenceEncoder(), seq, is_training=False)
            e128 = call_with_is_training(
                embeddings_module.OutputEmbedder(2), trunk, org, is_training=False
            )
            return call_with_is_training(
                embeddings_module.OutputEmbedder(2), trunk, org, skip_x=e128, is_training=False
            )

    seq = jax.ShapeDtypeStruct((1, 2048, 4), jnp.float32)
    org = jax.ShapeDtypeStruct((1,), jnp.int32)
    params, _ = jax.eval_shape(fn.init, jax.random.PRNGKey(0), seq, org)
    assert any(name.startswith("alphagenome/sequence_encoder") for name in params)


def test_forward_with_encoder_output_traces_on_installed_api():
    """The full custom forward (encoder, tower, decoder, embedders) traces without weights."""

    @hk.transform_with_state
    def fn(seq, org):
        with hk.name_scope("alphagenome"):
            return forward_with_encoder_output(SimpleNamespace(_num_organisms=2), seq, org)

    seq = jax.ShapeDtypeStruct((1, 2048, 4), jnp.float32)
    org = jax.ShapeDtypeStruct((1,), jnp.int32)
    params, _ = jax.eval_shape(fn.init, jax.random.PRNGKey(0), seq, org)
    assert "alphagenome/transformer_tower/mha_block/q_layer" in params
