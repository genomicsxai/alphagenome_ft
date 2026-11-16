"""Pre-merge (monkeypatch-era) LoRA checkpoints must load into the upstream-layout model."""

import numpy as np
import orbax.checkpoint as ocp
import pytest

from alphagenome_ft import legacy_lora

TOWER = "alphagenome/transformer_tower/mha_block"


def _legacy_params():
    rng = np.random.default_rng(0)
    return {
        f"{TOWER}/q_lora": {"lora_a": rng.normal(size=(8, 2)).astype("f4"),
                            "lora_b": rng.normal(size=(2, 16)).astype("f4")},
        f"{TOWER}/v_lora": {"lora_a": rng.normal(size=(8, 2)).astype("f4"),
                            "lora_b": rng.normal(size=(2, 4)).astype("f4")},
        f"{TOWER}_1/q_lora": {"lora_a": rng.normal(size=(8, 2)).astype("f4"),
                              "lora_b": rng.normal(size=(2, 16)).astype("f4")},
        "head/rna_seq/multi_organism_linear": {"w": rng.normal(size=(1, 8, 4)).astype("f4")},
    }


def test_remap_moves_adapters_next_to_pretrained_weight():
    legacy = _legacy_params()
    assert legacy_lora.has_legacy_lora_params(legacy)
    new = legacy_lora.remap_legacy_lora_params(legacy)
    assert not legacy_lora.has_legacy_lora_params(new)
    assert set(new) == {f"{TOWER}/q_layer", f"{TOWER}/v_layer",
                        f"{TOWER}_1/q_layer", "head/rna_seq/multi_organism_linear"}
    np.testing.assert_array_equal(new[f"{TOWER}/q_layer"]["lora_a"], legacy[f"{TOWER}/q_lora"]["lora_a"])
    np.testing.assert_array_equal(new[f"{TOWER}_1/q_layer"]["lora_b"], legacy[f"{TOWER}_1/q_lora"]["lora_b"])


def test_remap_is_idempotent_and_merges_with_existing_weight():
    both = {f"{TOWER}/q_layer": {"w": np.ones((8, 16), "f4")}, **_legacy_params()}
    once = legacy_lora.remap_legacy_lora_params(both)
    assert set(once[f"{TOWER}/q_layer"]) == {"w", "lora_a", "lora_b"}
    twice = legacy_lora.remap_legacy_lora_params(once)
    assert set(twice) == set(once)


class _FakeModel:
    def __init__(self):
        self._params = {f"{TOWER}/q_layer": {"w": np.ones((8, 16), "f4"),
                                              "lora_a": np.zeros((8, 2), "f4"),
                                              "lora_b": np.zeros((2, 16), "f4")},
                        f"{TOWER}/v_layer": {"w": np.ones((8, 4), "f4"),
                                              "lora_a": np.zeros((8, 2), "f4"),
                                              "lora_b": np.zeros((2, 4), "f4")},
                        f"{TOWER}_1/q_layer": {"w": np.ones((8, 16), "f4"),
                                                "lora_a": np.zeros((8, 2), "f4"),
                                                "lora_b": np.zeros((2, 16), "f4")}}
        self._state = {}


def test_restore_into_model_reads_legacy_orbax_checkpoint(tmp_path):
    legacy = _legacy_params()
    saver = ocp.StandardCheckpointer()
    saver.save(str(tmp_path / "checkpoint"), (legacy, {}))
    saver.wait_until_finished()
    model = _FakeModel()
    info = legacy_lora.restore_into_model(model, tmp_path)
    assert info["lora_leaves"] == 6
    np.testing.assert_allclose(model._params[f"{TOWER}/q_layer"]["lora_a"], legacy[f"{TOWER}/q_lora"]["lora_a"])
    # pretrained weight untouched
    np.testing.assert_array_equal(model._params[f"{TOWER}/q_layer"]["w"], np.ones((8, 16), "f4"))
    assert "head/rna_seq/multi_organism_linear" in model._params
