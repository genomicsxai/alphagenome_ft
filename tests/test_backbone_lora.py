"""Tests for transformer query/value LoRA injection."""

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alphagenome_research.model import attention

from alphagenome_ft.backbone_lora import (
    BackboneLoRAConfig,
    BackboneLoRALinear,
    BackboneLoRAMHABlock,
    expected_lora_parameter_count,
    _call_with_optional_training,
)
from alphagenome_ft.custom_model import CustomAlphaGenomeModel
from alphagenome_ft.lora import count_lora_parameters, get_lora_parameter_paths


def _mutable(tree):
    return hk.data_structures.to_mutable_dict(tree)


def test_qv_parameter_count_matches_nine_alphagenome_blocks():
    assert expected_lora_parameter_count(rank=8) == 308_736


def test_backbone_lora_config_maps_pytorch_projection_names():
    config = BackboneLoRAConfig(targets=("q_proj", "v_proj", "q_layer"))
    assert config.jax_targets == ("q_layer", "v_layer")
    with pytest.raises(ValueError, match="Unsupported"):
        BackboneLoRAConfig(targets=("k_proj",))


def test_lora_mha_starts_as_exact_pretrained_noop():
    x = jnp.ones((1, 4, 32), dtype=jnp.bfloat16)
    bias = jnp.zeros((1, 8, 4, 4), dtype=jnp.float32)

    def original_forward(values, attention_bias):
        return _call_with_optional_training(
            attention.MHABlock(name="mha_block"), values, attention_bias,
            is_training=False,
        )

    def lora_forward(values, attention_bias):
        return BackboneLoRAMHABlock(
            BackboneLoRAConfig(rank=4, alpha=8), name="mha_block"
        )(values, attention_bias)

    original = hk.transform_with_state(original_forward)
    adapted = hk.transform_with_state(lora_forward)
    key = jax.random.PRNGKey(7)
    original_params, original_state = original.init(key, x, bias)
    adapted_params, adapted_state = adapted.init(key, x, bias)

    merged = _mutable(adapted_params)
    for module_name, module_params in _mutable(original_params).items():
        assert module_name in merged
        for parameter_name, value in module_params.items():
            assert parameter_name in merged[module_name]
            merged[module_name][parameter_name] = value

    adapter_paths = get_lora_parameter_paths(merged)
    assert len(adapter_paths) == 4
    assert any("q_layer/lora_a" in path for path in adapter_paths)
    assert any("v_layer/lora_b" in path for path in adapter_paths)
    assert count_lora_parameters(merged) == 32 * 4 + 4 * 1024 + 32 * 4 + 4 * 192

    expected, _ = original.apply(original_params, original_state, None, x, bias)
    observed, _ = adapted.apply(merged, adapted_state, None, x, bias)
    np.testing.assert_allclose(
        np.asarray(observed, dtype=np.float32),
        np.asarray(expected, dtype=np.float32),
        rtol=0,
        atol=0,
    )


def test_lora_a_matches_torch_linear_default_bounds():
    transformed = hk.without_apply_rng(
        hk.transform(
            lambda x: BackboneLoRALinear(
                7, rank=3, alpha=6, name="projection"
            )(x)
        )
    )
    params = transformed.init(jax.random.PRNGKey(3), jnp.ones((1, 11)))
    lora_a = next(
        values["lora_a"] for values in _mutable(params).values()
        if "lora_a" in values
    )
    bound = 1.0 / np.sqrt(11)
    assert np.max(np.abs(np.asarray(lora_a))) <= bound

    invalid = hk.without_apply_rng(
        hk.transform(
            lambda x: BackboneLoRALinear(
                2, rank=3, alpha=6, name="invalid"
            )(x)
        )
    )
    with pytest.raises(ValueError, match="output features"):
        invalid.init(jax.random.PRNGKey(4), jnp.ones((1, 11)))


def test_delta_checkpoint_slice_keeps_heads_and_lora_but_not_base_weights():
    model = object.__new__(CustomAlphaGenomeModel)
    model._params = {
        "alphagenome/transformer_tower/mha_block/q_layer": {
            "w": jnp.ones((2, 3)),
            "lora_a": jnp.ones((2, 1)),
            "lora_b": jnp.ones((1, 3)),
        },
        "head/task/projection": {"w": jnp.ones((2, 1))},
    }
    model._state = {}
    model._custom_heads = ["task"]
    model._backbone_lora_config = BackboneLoRAConfig(rank=1)

    params, state = model._checkpoint_slice_trees(False, False)

    assert state == {}
    assert "head/task/projection" in params
    q_params = params["alphagenome/transformer_tower/mha_block/q_layer"]
    assert set(q_params) == {"lora_a", "lora_b"}
