"""Tests for PyTorch -> alphagenome_ft checkpoint conversion and self-describing checkpoints."""

import json

import numpy as np
import orbax.checkpoint as ocp
import pandas as pd
import pytest
from alphagenome.models import dna_output
from alphagenome_research.model import dna_model
from alphagenome_research.model import heads as research_heads
from alphagenome_research.model.metadata.metadata import AlphaGenomeOutputMetadata

from alphagenome_ft import custom_heads, custom_model
from alphagenome_ft.convert_pytorch import convert_pytorch_checkpoint
from tests.kaggle_util import kaggle_credentials_available

RANK = 2
HEAD_IDS = ("rna_seq", "splice_site", "splice_usage", "splice_junctions")
ROPE = ("pos_donor", "pos_acceptor", "neg_donor", "neg_acceptor")


@pytest.fixture(autouse=True)
def isolated_head_registry():
    """Run each test without (and restore afterwards) registrations of the converted head ids."""
    registries = (
        custom_heads._HEAD_REGISTRY,
        custom_heads._HEAD_CONFIG_REGISTRY,
        custom_heads._HEAD_METADATA_REGISTRY,
    )
    saved = [dict(registry) for registry in registries]
    for registry in registries:
        for head_id in HEAD_IDS:
            registry.pop(head_id, None)
    yield
    for registry, snapshot in zip(registries, saved):
        registry.clear()
        registry.update(snapshot)


def _pytorch_weights(num_slots=1, lora=True, seed=0):
    """Random tensors laid out as alphagenome-pytorch finetuning state-dict entries."""
    rng = np.random.default_rng(seed)

    def r(*shape):
        return rng.standard_normal(shape).astype(np.float32)

    p = num_slots
    weights = {
        "heads.rna_seq.track_means": rng.uniform(0.5, 2.0, (p, 3)).astype(np.float32),
        "heads.rna_seq.convs.1.weight": r(p, 3, 1536),
        "heads.rna_seq.convs.1.bias": r(p, 3),
        "heads.rna_seq.residual_scales.1": r(p, 3),
        "heads.rna_seq.convs.128.weight": r(p, 3, 3072),
        "heads.rna_seq.convs.128.bias": r(p, 3),
        "heads.rna_seq.residual_scales.128": r(p, 3),
        "splice_sites_classification_head.conv.weight": r(p, 5, 1536),
        "splice_sites_classification_head.conv.bias": r(p, 5),
        "splice_sites_usage_head.conv.weight": r(p, 4, 1536),
        "splice_sites_usage_head.conv.bias": r(p, 4),
        "splice_sites_junction_head.conv.weight": r(p, 768, 1536),
        "splice_sites_junction_head.conv.bias": r(p, 768),
        **{f"splice_sites_junction_head.rope_params.{name}": r(p, 2, 2, 768) for name in ROPE},
    }
    if lora:
        for block in range(9):
            for projection, out in (("q_proj", 1024), ("v_proj", 192)):
                prefix = f"tower.blocks.{block}.mha.{projection}"
                weights[f"{prefix}.lora_A.weight"] = r(RANK, 1536)
                weights[f"{prefix}.lora_B.weight"] = r(out, RANK)
    return weights


def _transfer_config(**overrides):
    config = {
        "mode": "lora",
        "lora_rank": RANK,
        "lora_alpha": 4,
        "lora_targets": ["q_proj", "v_proj"],
        "unfreeze_norm": False,
        "new_heads": {
            "rna_seq": {"modality": "rna_seq", "num_tracks": 3, "resolutions": [1, 128]},
            "splice_site": {"modality": "splice_site", "num_tracks": 5},
            "splice_usage": {"modality": "splice_usage", "num_tracks": 4},
            "splice_junctions": {"modality": "splice_junctions", "num_tracks": 2},
        },
    }
    config.update(overrides)
    return config


def _metadata(organism="human"):
    return {
        "track_names": {
            "rna_seq": ["s1_forward", "s1_reverse", "s2"],
            "splice_site": ["cls_donor_pos", "cls_acceptor_pos", "cls_donor_neg",
                            "cls_acceptor_neg", "cls_none"],
            "splice_usage": ["a_pos", "b_pos", "a_neg", "b_neg"],
            "splice_junctions": ["a_pos", "b_pos", "a_neg", "b_neg"],
        },
        "track_metadata": [
            {"output_name": "rna_seq", "track_index": index, "track_name": name,
             "organism": 0 if organism == "human" else 1, "strand": strand, "assay": "polyA"}
            for index, (name, strand) in enumerate(
                zip(["s1_forward", "s1_reverse", "s2"], ["+", "-", None])
            )
        ],
        "organism": organism,
    }


def _write_safetensors(path, weights, transfer_config, metadata):
    from safetensors.numpy import save_file

    header = {"transfer_config": transfer_config, **metadata}
    save_file(weights, str(path), metadata={k: json.dumps(v) for k, v in header.items()})
    return path


def _restore(checkpoint_dir):
    return ocp.StandardCheckpointer().restore(str(checkpoint_dir / "checkpoint"))


def _convert(tmp_path, weights=None, transfer_config=None, metadata=None, **kwargs):
    source = _write_safetensors(
        tmp_path / "model.safetensors",
        _pytorch_weights() if weights is None else weights,
        _transfer_config() if transfer_config is None else transfer_config,
        _metadata() if metadata is None else metadata,
    )
    return convert_pytorch_checkpoint(source, tmp_path / "converted", **kwargs)


def test_converts_head_and_lora_parameters(tmp_path):
    weights = _pytorch_weights()
    out = _convert(tmp_path, weights=weights)
    params, state = _restore(out)
    assert state == {}

    for resolution in (1, 128):
        linear = params[f"head/rna_seq/resolution_{resolution}/multi_organism_linear"]
        np.testing.assert_array_equal(
            linear["w"], weights[f"heads.rna_seq.convs.{resolution}.weight"].transpose(0, 2, 1))
        np.testing.assert_array_equal(linear["b"], weights[f"heads.rna_seq.convs.{resolution}.bias"])
        np.testing.assert_array_equal(
            params[f"head/rna_seq/resolution_{resolution}"]["learnt_scale"],
            weights[f"heads.rna_seq.residual_scales.{resolution}"])
    for head_id, prefix in (
        ("splice_site", "splice_sites_classification_head"),
        ("splice_usage", "splice_sites_usage_head"),
        ("splice_junctions", "splice_sites_junction_head"),
    ):
        linear = params[f"head/{head_id}/multi_organism_linear"]
        np.testing.assert_array_equal(linear["w"], weights[f"{prefix}.conv.weight"].transpose(0, 2, 1))
        np.testing.assert_array_equal(linear["b"], weights[f"{prefix}.conv.bias"])
    for name in ROPE:
        embeddings = params[f"head/splice_junctions/{name}_logits"]["embeddings"]
        assert embeddings.shape == (1, 2 * 2 * 768)
        np.testing.assert_array_equal(
            embeddings.reshape(1, 2, 2, 768),
            weights[f"splice_sites_junction_head.rope_params.{name}"])

    for block in range(9):
        suffix = "" if block == 0 else f"_{block}"
        for projection, layer in (("q_proj", "q_layer"), ("v_proj", "v_layer")):
            lora = params[f"alphagenome/transformer_tower/mha_block{suffix}/{layer}"]
            assert set(lora) == {"lora_a", "lora_b"}
            prefix = f"tower.blocks.{block}.mha.{projection}"
            np.testing.assert_array_equal(lora["lora_a"], weights[f"{prefix}.lora_A.weight"].T)
            np.testing.assert_array_equal(lora["lora_b"], weights[f"{prefix}.lora_B.weight"].T)

    config = json.loads((out / "config.json").read_text())
    assert config["custom_heads"] == list(HEAD_IDS)
    assert config["backbone_lora"] == {
        "rank": RANK, "alpha": 4.0, "targets": ["q_proj", "v_proj"],
        "gradient_checkpointing": False,
    }
    assert config["head_configs"]["rna_seq"]["resolutions"] == [1, 128]
    assert config["head_configs"]["splice_junctions"]["type"] == "SPLICE_SITES_JUNCTION"
    assert not config["save_full_model"] and not config["use_encoder_output"]


def test_head_metadata_from_pytorch_tracks(tmp_path):
    weights = _pytorch_weights()
    config = json.loads((_convert(tmp_path, weights=weights) / "config.json").read_text())
    metadata = config["head_metadata"]
    assert set(metadata) == set(HEAD_IDS)
    rows = metadata["rna_seq"]["HOMO_SAPIENS"]
    assert [row["name"] for row in rows] == ["s1_forward", "s1_reverse", "s2"]
    assert [row["strand"] for row in rows] == ["+", "-", "."]
    assert [row["assay"] for row in rows] == ["polyA"] * 3
    np.testing.assert_allclose(
        [row["nonzero_mean"] for row in rows], weights["heads.rna_seq.track_means"][0])
    assert [row["name"] for row in metadata["splice_usage"]["HOMO_SAPIENS"]] == [
        "a_pos", "b_pos", "a_neg", "b_neg"]
    # One junction metadata row per tissue, not per strand channel.
    assert [row["name"] for row in metadata["splice_junctions"]["HOMO_SAPIENS"]] == ["a", "b"]
    assert len(metadata["splice_site"]["HOMO_SAPIENS"]) == 5


def test_track_names_default_when_checkpoint_has_none(tmp_path):
    config = json.loads(
        (_convert(tmp_path, metadata={}) / "config.json").read_text())["head_metadata"]
    assert [row["name"] for row in config["rna_seq"]["HOMO_SAPIENS"]] == [
        "rna_seq_0", "rna_seq_1", "rna_seq_2"]
    assert [row["name"] for row in config["splice_site"]["HOMO_SAPIENS"]] == [
        "donor_pos", "acceptor_pos", "donor_neg", "acceptor_neg", "none"]


def test_heads_only_checkpoint_has_no_lora(tmp_path):
    out = _convert(
        tmp_path, weights=_pytorch_weights(lora=False),
        transfer_config=_transfer_config(mode="linear"),
    )
    params, _ = _restore(out)
    assert all(key.startswith("head/") for key in params)
    assert json.loads((out / "config.json").read_text())["backbone_lora"] is None


def test_organism_mouse_selects_slot(tmp_path):
    weights = _pytorch_weights(num_slots=2)
    out = _convert(tmp_path, weights=weights, metadata=_metadata(organism="mouse"))
    params, _ = _restore(out)
    linear = params["head/splice_site/multi_organism_linear"]
    np.testing.assert_array_equal(
        linear["w"], weights["splice_sites_classification_head.conv.weight"][1:].transpose(0, 2, 1))
    metadata = json.loads((out / "config.json").read_text())["head_metadata"]["rna_seq"]
    assert list(metadata) == ["MUS_MUSCULUS"]
    np.testing.assert_allclose(
        [row["nonzero_mean"] for row in metadata["MUS_MUSCULUS"]],
        weights["heads.rna_seq.track_means"][1])


def test_organism_argument_overrides_checkpoint(tmp_path):
    out = _convert(tmp_path, organism="mouse")
    metadata = json.loads((out / "config.json").read_text())["head_metadata"]
    assert list(metadata["splice_usage"]) == ["MUS_MUSCULUS"]
    # Rows recorded for human are not attributed to mouse.
    assert "assay" not in metadata["rna_seq"]["MUS_MUSCULUS"][0]


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"mode": "full"}, "full"),
        ({"mode": "encoder-only"}, "encoder-only"),
        ({"mode": "locon"}, "locon"),
        ({"mode": ["lora", "locon"]}, "locon"),
        ({"mode": "ia3"}, "ia3"),
        ({"mode": "houlsby"}, "houlsby"),
        ({"lora_targets": ["q_proj", "k_proj"]}, "k_proj"),
        ({"unfreeze_norm": True}, "unfreeze_norm"),
    ],
)
def test_unsupported_transfer_configs(tmp_path, overrides, message):
    with pytest.raises(ValueError, match=message):
        _convert(tmp_path, transfer_config=_transfer_config(**overrides))


def test_unsupported_adapter_weights(tmp_path):
    weights = _pytorch_weights()
    weights["encoder.down_blocks.5.conv.locon_down.weight"] = np.zeros((4, 1536, 1), np.float32)
    with pytest.raises(ValueError, match="locon_down"):
        _convert(tmp_path, weights=weights)


def test_unsupported_head(tmp_path):
    config = _transfer_config()
    config["new_heads"]["contact_maps"] = {"modality": "contact_maps", "num_tracks": 28}
    with pytest.raises(ValueError, match="contact_maps"):
        _convert(tmp_path, transfer_config=config)


def test_pth_requires_torch(tmp_path, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "torch", None)
    (tmp_path / "model.delta.pth").write_bytes(b"")
    with pytest.raises(ImportError, match="requires PyTorch"):
        convert_pytorch_checkpoint(tmp_path / "model.delta.pth", tmp_path / "out")


def test_pth_formats_match_safetensors(tmp_path):
    torch = pytest.importorskip("torch")
    weights = _pytorch_weights()
    transfer_config = _transfer_config()
    metadata = _metadata()
    tensors = {key: torch.from_numpy(value) for key, value in weights.items()}
    heads = {key: value for key, value in tensors.items() if "lora_" not in key}
    adapters = {key: value for key, value in tensors.items() if "lora_" in key}
    torch.save({
        "delta_checkpoint_version": 1,
        "transfer_config": transfer_config,
        "adapter_state_dict": adapters,
        "head_state_dict": heads,
        "norm_state_dict": {},
        "metadata": metadata,
    }, tmp_path / "best_model.delta.pth")
    full_state = {**heads, **adapters,
                  "tower.blocks.0.mha.q_proj.original_layer.weight": torch.zeros(1024, 1536),
                  "decoder.up_blocks.0.conv_in.conv.scale": torch.ones(3)}
    torch.save({"epoch": 1, "model_state_dict": full_state, "transfer_config": transfer_config,
                **metadata}, tmp_path / "best_model.pth")

    expected = _restore(_convert(tmp_path, weights=weights, metadata=metadata))
    for name in ("best_model.delta.pth", "best_model.pth"):
        out = convert_pytorch_checkpoint(tmp_path / name, tmp_path / f"out_{name}")
        restored = _restore(out)
        assert restored[0].keys() == expected[0].keys()
        for key in expected[0]:
            for leaf in expected[0][key]:
                np.testing.assert_array_equal(restored[0][key][leaf], expected[0][key][leaf])
        assert (out / "config.json").read_text() == (tmp_path / "converted/config.json").read_text()


def test_delta_with_trained_norms_is_rejected(tmp_path):
    torch = pytest.importorskip("torch")
    torch.save({
        "delta_checkpoint_version": 1, "transfer_config": _transfer_config(),
        "adapter_state_dict": {}, "head_state_dict": {},
        "norm_state_dict": {"tower.blocks.0.mha.norm.weight": torch.ones(1536)},
    }, tmp_path / "m.delta.pth")
    with pytest.raises(ValueError, match="unfreeze_norm"):
        convert_pytorch_checkpoint(tmp_path / "m.delta.pth", tmp_path / "out")


def test_head_metadata_serialization_round_trip():
    table = pd.DataFrame({
        "name": ["a", "b"], "strand": ["+", "-"],
        "nonzero_mean": np.array([0.25, np.nan], dtype=np.float32),
        "count": np.array([1, 2], dtype=np.int64),
    })
    metadata = {dna_model.Organism.HOMO_SAPIENS: AlphaGenomeOutputMetadata(rna_seq=table)}
    serialized = custom_heads.serialize_head_metadata(metadata, dna_output.OutputType.RNA_SEQ)
    assert serialized == {"HOMO_SAPIENS": [
        {"name": "a", "strand": "+", "nonzero_mean": 0.25, "count": 1},
        {"name": "b", "strand": "-", "nonzero_mean": None, "count": 2},
    ]}
    json.dumps(serialized, allow_nan=False)
    restored = custom_heads.deserialize_head_metadata(serialized, dna_output.OutputType.RNA_SEQ)
    restored_table = restored[dna_model.Organism.HOMO_SAPIENS].get(dna_output.OutputType.RNA_SEQ)
    pd.testing.assert_frame_equal(
        restored_table, table.astype({"nonzero_mean": np.float64}), check_dtype=True)


@pytest.mark.parametrize("kind", ["rna_seq", "atac", "splice_sites_classification",
                                  "splice_sites_usage", "splice_sites_junction"])
def test_predefined_head_config_round_trip(kind):
    config = custom_heads.get_predefined_head_config(kind, num_tracks=1)
    serialized = {"source": "predefined", **custom_model._serialize_config(config)}
    json.dumps(serialized)
    assert custom_heads.deserialize_predefined_head_config(serialized) == config


class _StopAfterRegistration(Exception):
    pass


def _stop_before_model_build(*args, **kwargs):
    raise _StopAfterRegistration


def test_load_checkpoint_registers_saved_heads(tmp_path, monkeypatch):
    out = _convert(tmp_path)
    monkeypatch.setattr(custom_model, "create_model_with_heads", _stop_before_model_build)
    with pytest.raises(_StopAfterRegistration):
        custom_model.load_checkpoint(out)
    for head_id in HEAD_IDS:
        assert custom_heads.is_head_registered(head_id)
    config = custom_heads.get_registered_head_config("rna_seq")
    assert isinstance(config, research_heads.GenomeTracksHeadConfig)
    assert config.name == "rna_seq" and list(config.resolutions) == [1, 128]
    table = custom_heads.get_registered_head_metadata("rna_seq")[
        dna_model.Organism.HOMO_SAPIENS].get(dna_output.OutputType.RNA_SEQ)
    assert list(table["name"]) == ["s1_forward", "s1_reverse", "s2"]
    junctions = custom_heads.get_registered_head_metadata("splice_junctions")
    assert len(junctions[dna_model.Organism.HOMO_SAPIENS].splice_junctions) == 2


def test_load_checkpoint_keeps_existing_registration(tmp_path, monkeypatch):
    out = _convert(tmp_path)
    config = custom_heads.get_predefined_head_config("rna_seq", num_tracks=3, resolutions=[1, 128])
    existing = {dna_model.Organism.HOMO_SAPIENS: AlphaGenomeOutputMetadata(
        rna_seq=pd.DataFrame({"name": ["x", "y", "z"], "strand": ["."] * 3}))}
    custom_heads.register_predefined_head("rna_seq", config, metadata=existing)
    monkeypatch.setattr(custom_model, "create_model_with_heads", _stop_before_model_build)
    with pytest.raises(_StopAfterRegistration):
        custom_model.load_checkpoint(out)
    assert custom_heads.get_registered_head_metadata("rna_seq") is existing


@pytest.mark.skipif(
    not kaggle_credentials_available(),
    reason="Kaggle credentials required for model fixture (~/.kaggle/kaggle.json or env)",
)
def test_converted_checkpoint_loads_and_saves_head_metadata(tmp_path, device):
    """A converted checkpoint loads without pre-registration and saves the same metadata."""
    out = _convert(tmp_path)
    expected, _ = _restore(out)
    model = custom_model.load_checkpoint(out, base_model_version="all_folds", device=device)
    for key, leaves in expected.items():
        for leaf, value in leaves.items():
            np.testing.assert_array_equal(np.asarray(model._params[key][leaf]), value)

    model.save_checkpoint(tmp_path / "resaved")
    resaved = json.loads((tmp_path / "resaved/config.json").read_text())
    original = json.loads((out / "config.json").read_text())
    assert resaved == original
    params, _ = _restore(tmp_path / "resaved")
    assert params.keys() == expected.keys()
