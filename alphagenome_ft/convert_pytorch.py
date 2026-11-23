"""Convert alphagenome-pytorch finetuned checkpoints to alphagenome_ft checkpoints.

Reads a checkpoint written by alphagenome-pytorch's finetuning code (a full
``.pth`` training checkpoint, a ``.delta.pth`` delta checkpoint, or an
``export_delta_weights`` ``.safetensors``/``.pth`` file) and writes a heads-only
alphagenome_ft checkpoint directory (``config.json`` + Orbax ``checkpoint/``)
that :func:`alphagenome_ft.load_checkpoint` loads on top of the pretrained JAX
base checkpoint, registering its heads from the saved ``head_metadata``.

Supported: linear-probe (heads-only) and ``lora`` finetunes with LoRA on the
transformer ``q_proj``/``v_proj`` projections; genome-track heads (rna_seq,
atac, dnase, cage, procap, chip_tf, chip_histone) and the splice_site,
splice_usage and splice_junctions heads.

Usage::

    python -m alphagenome_ft.convert_pytorch best_model.delta.pth out_dir

``torch`` is only imported to read ``.pth`` files; ``.safetensors`` exports are
read with safetensors' numpy API.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import dataclasses
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from alphagenome.models import dna_output
from alphagenome_research.model import dna_model
from alphagenome_research.model.metadata.metadata import AlphaGenomeOutputMetadata

from alphagenome_ft import custom_heads as custom_heads_module
from alphagenome_ft.backbone_lora import BackboneLoRAConfig
from alphagenome_ft.custom_model import _HeadConfigEntry, _checkpoint_config, _write_checkpoint

_TRACK_KINDS = ("rna_seq", "atac", "dnase", "cage", "procap", "chip_tf", "chip_histone")
# PyTorch splice head name -> (state-dict prefix, alphagenome_research head kind).
_SPLICE_HEADS = {
    "splice_site": ("splice_sites_classification_head.", "splice_sites_classification"),
    "splice_usage": ("splice_sites_usage_head.", "splice_sites_usage"),
    "splice_junctions": ("splice_sites_junction_head.", "splice_sites_junction"),
}
_SPLICE_CLASSIFICATION_LABELS = (
    "donor_pos", "acceptor_pos", "donor_neg", "acceptor_neg", "none",
)
_ROPE_PARAMS = ("pos_donor", "pos_acceptor", "neg_donor", "neg_acceptor")
_LORA_KEY = re.compile(r"tower\.blocks\.(\d+)\.mha\.([qv]_proj)\.lora_([AB])\.weight")
_ADAPTER_KEY = re.compile(r"\.(lora_[AB]|locon_(down|up)|adapter)\.")
_ORGANISMS = {
    "human": dna_model.Organism.HOMO_SAPIENS,
    "mouse": dna_model.Organism.MUS_MUSCULUS,
}
_ORGANISM_INDEX = {dna_model.Organism.HOMO_SAPIENS: 0, dna_model.Organism.MUS_MUSCULUS: 1}


def _to_numpy(tensor) -> np.ndarray:
    if isinstance(tensor, np.ndarray):
        return tensor
    tensor = tensor.detach().cpu()
    if tensor.is_floating_point():
        tensor = tensor.float()
    return tensor.numpy()


def read_pytorch_checkpoint(path: str | Path) -> tuple[dict, dict[str, np.ndarray], dict]:
    """Read a PyTorch finetuned checkpoint.

    Returns ``(transfer_config, weights, metadata)``: the serialized
    ``TransferConfig`` dict, the finetuned tensors (heads, adapters and, for a
    full checkpoint, the rest of the model) as numpy arrays, and the embedded
    metadata (``track_names``, ``track_metadata``, ``organism``, ...).
    """
    path = Path(path)
    if path.suffix == ".safetensors":
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise ImportError(
                f"Reading {path.name} requires safetensors (pip install safetensors)."
            ) from exc

        with safe_open(path, framework="np") as f:
            header = f.metadata() or {}
            weights = {key: f.get_tensor(key) for key in f.keys()}
        metadata = {key: json.loads(value) for key, value in header.items()}
    else:
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                f"Reading {path.name} requires PyTorch (pip install torch); "
                "alternatively convert an exported .safetensors file."
            ) from exc
        data = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(data, Mapping):
            raise ValueError(f"{path} is not an alphagenome-pytorch finetuning checkpoint.")
        if "delta_checkpoint_version" in data:
            if data.get("norm_state_dict"):
                raise ValueError(
                    "Unsupported: the checkpoint trains normalization layers "
                    "(unfreeze_norm); alphagenome_ft only finetunes heads and q/v LoRA."
                )
            weights = {**data["adapter_state_dict"], **data["head_state_dict"]}
            metadata = {**data.get("metadata", {}), "transfer_config": data["transfer_config"]}
        elif "weights" in data:  # export_delta_weights(..., format="pth")
            weights, metadata = data["weights"], dict(data)
        elif "model_state_dict" in data:
            weights, metadata = data["model_state_dict"], dict(data)
        else:
            raise ValueError(f"{path} is not an alphagenome-pytorch finetuning checkpoint.")
        weights = {key: _to_numpy(value) for key, value in weights.items()}
    transfer_config = metadata.get("transfer_config")
    if transfer_config is None:
        raise ValueError(
            f"{path} has no transfer_config, so its heads and adapters cannot be identified."
        )
    return transfer_config, weights, metadata


def _resolve_organism(organism: str | None, metadata: Mapping) -> dna_model.Organism:
    if organism is None:
        organism = metadata.get("organism")
    if organism is None and metadata.get("organism_indices"):
        indices = list(metadata["organism_indices"])
        if len(indices) != 1:
            raise ValueError(
                f"Checkpoint was trained for organisms {indices}; pass organism= to pick one."
            )
        organism = "mouse" if int(indices[0]) == 1 else "human"
    if organism is None:
        return dna_model.Organism.HOMO_SAPIENS
    if isinstance(organism, dna_model.Organism):
        return organism
    name = str(organism)
    if name.lower() in _ORGANISMS:
        return _ORGANISMS[name.lower()]
    try:
        return dna_model.Organism[name.upper()]
    except KeyError:
        raise ValueError(f"Unknown organism {organism!r}; use human or mouse.") from None


def _organism_slot(array: np.ndarray, slot: int, name: str) -> np.ndarray:
    """Select one organism slot of a ``(P, ...)`` head tensor as ``(1, ...)``."""
    if array.shape[0] == 1:
        return array
    if slot >= array.shape[0]:
        raise ValueError(f"{name} has {array.shape[0]} organism slots; slot {slot} is missing.")
    return array[slot : slot + 1]


def _row_organism_index(value: Any) -> int:
    """A track_metadata row's organism, recorded as an index or as "human"/"mouse"."""
    if isinstance(value, str) and not value.strip().isdigit():
        return _ORGANISM_INDEX[_resolve_organism(value, {})]
    return int(value)


def _track_rows(
    head_id: str,
    num_rows: int,
    metadata: Mapping,
    organism: dna_model.Organism,
    single_head: bool,
) -> list[dict[str, Any]]:
    """Per-track metadata rows (``name`` + extras) from the checkpoint, in track order."""
    organism_index = _ORGANISM_INDEX[organism]
    rows = [
        row for row in metadata.get("track_metadata") or []
        if str(row.get("output_name") or row.get("output_type") or "").lower() == head_id
        and _row_organism_index(row.get("organism", organism_index)) == organism_index
    ]
    if rows:
        rows = sorted(rows, key=lambda row: int(row.get("track_index", 0)))
        rows = [
            {"name": row.get("track_name"), **{
                key: value for key, value in row.items()
                if key not in {"track_name", "track_index", "output_name", "output_type", "organism"}
            }}
            for row in rows
        ]
    else:
        track_names = metadata.get("track_names")
        if isinstance(track_names, Mapping):
            track_names = track_names.get(head_id)
        elif not single_head:
            track_names = None
        rows = [{"name": name} for name in track_names or []]
    if head_id == "splice_junctions" and len(rows) == 2 * num_rows:
        # PyTorch names both strands (<sample>_pos..., <sample>_neg...); the JAX
        # junction metadata has one row per tissue.
        rows = [
            {**row, "name": re.sub(r"_pos$", "", str(row["name"]))} for row in rows[:num_rows]
        ]
    if not rows:
        if head_id == "splice_site":
            rows = [{"name": label} for label in _SPLICE_CLASSIFICATION_LABELS]
        else:
            rows = [{"name": f"{head_id}_{index}"} for index in range(num_rows)]
    if len(rows) != num_rows:
        raise ValueError(
            f"Head '{head_id}' has {num_rows} tracks but the checkpoint metadata names {len(rows)}."
        )
    return rows


def _check_transfer_config(transfer_config: Mapping) -> BackboneLoRAConfig | None:
    modes = transfer_config.get("mode", "linear")
    modes = [modes] if isinstance(modes, str) else list(modes)
    unsupported = [mode for mode in modes if mode not in {"linear", "lora"}]
    if unsupported:
        raise ValueError(
            f"Unsupported finetuning mode(s) {unsupported}: alphagenome_ft converts "
            "linear-probe (heads-only) and q/v LoRA checkpoints only "
            "(not full, encoder-only, LoCon, IA3 or Houlsby)."
        )
    if transfer_config.get("unfreeze_norm") and "lora" in modes:
        raise ValueError(
            "Unsupported: the checkpoint trains normalization layers (unfreeze_norm)."
        )
    for head_id, head in transfer_config.get("new_heads", {}).items():
        if head.get("encoder_only"):
            raise ValueError(f"Unsupported: head '{head_id}' is an encoder-only head.")
    if "lora" not in modes or int(transfer_config.get("lora_rank", 0)) <= 0:
        return None
    targets = list(transfer_config.get("lora_targets", []))
    unsupported = [target for target in targets if target not in {"q_proj", "v_proj"}]
    if unsupported or not targets:
        raise ValueError(
            f"Unsupported LoRA targets {unsupported or targets}: alphagenome_ft supports "
            "backbone LoRA on q_proj and v_proj only."
        )
    return BackboneLoRAConfig(
        rank=int(transfer_config["lora_rank"]),
        alpha=float(transfer_config["lora_alpha"]),
        targets=tuple(targets),
    )


def _convert_lora(
    weights: Mapping[str, np.ndarray], lora_config: BackboneLoRAConfig | None
) -> dict[str, dict[str, np.ndarray]]:
    adapter_keys = [key for key in weights if _ADAPTER_KEY.search(key)]
    unsupported = [key for key in adapter_keys if not _LORA_KEY.fullmatch(key)]
    if unsupported:
        raise ValueError(
            "Unsupported adapter parameters (alphagenome_ft supports q/v LoRA only): "
            f"{unsupported[:5]}"
        )
    if lora_config is None:
        if adapter_keys:
            raise ValueError("Checkpoint has LoRA weights but its transfer_config is not lora.")
        return {}
    params: dict[str, dict[str, np.ndarray]] = {}
    for key in adapter_keys:
        block, projection, matrix = _LORA_KEY.fullmatch(key).groups()
        if projection not in lora_config.targets:
            raise ValueError(f"LoRA weight {key} is not in lora_targets {lora_config.targets}.")
        suffix = "" if int(block) == 0 else f"_{block}"
        layer = "q_layer" if projection == "q_proj" else "v_layer"
        module = f"alphagenome/transformer_tower/mha_block{suffix}/{layer}"
        # PyTorch Linear stores (out, in); Haiku stores (in, out).
        params.setdefault(module, {})[f"lora_{matrix.lower()}"] = weights[key].T.astype(np.float32)
    incomplete = [module for module, leaves in params.items() if set(leaves) != {"lora_a", "lora_b"}]
    if incomplete:
        raise ValueError(f"LoRA weights incomplete for {incomplete}.")
    return params


def _head_weights(weights: Mapping[str, np.ndarray], prefix: str) -> dict[str, np.ndarray]:
    return {key[len(prefix):]: value for key, value in weights.items() if key.startswith(prefix)}


def _linear(weight: np.ndarray, bias: np.ndarray, slot: int, name: str) -> dict[str, np.ndarray]:
    # (P, out, in) conv weight -> (1, in, out) multi_organism_linear w.
    return {
        "w": np.swapaxes(_organism_slot(weight, slot, name), 1, 2).astype(np.float32),
        "b": _organism_slot(bias, slot, name).astype(np.float32),
    }


def convert_pytorch_checkpoint(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    organism: str | dna_model.Organism | None = None,
) -> Path:
    """Convert an alphagenome-pytorch finetuned checkpoint to an alphagenome_ft checkpoint.

    Args:
        checkpoint_path: Full ``.pth``, delta ``.delta.pth`` or exported
            ``.safetensors``/``.pth`` checkpoint from alphagenome-pytorch finetuning.
        output_dir: Directory to write (``config.json`` + ``checkpoint/``).
        organism: Organism the heads were trained for (``human``/``mouse`` or an
            ``Organism``). Defaults to the checkpoint's recorded organism, else human.

    Returns:
        The output directory; load it with ``load_checkpoint(output_dir,
        base_checkpoint_path=<pretrained JAX checkpoint>)``.

    Raises:
        ValueError: For unsupported modes, adapters, LoRA targets or heads.
    """
    transfer_config, weights, metadata = read_pytorch_checkpoint(checkpoint_path)
    lora_config = _check_transfer_config(transfer_config)
    organism = _resolve_organism(organism, metadata)
    slot = _ORGANISM_INDEX[organism]

    params = _convert_lora(weights, lora_config)
    head_configs: dict[str, _HeadConfigEntry] = {}
    head_metadata: dict[str, dict] = {}
    new_heads = transfer_config.get("new_heads", {})
    for head_id, head_spec in new_heads.items():
        modality = head_spec.get("modality", head_id)
        if modality in _SPLICE_HEADS:
            prefix, kind = _SPLICE_HEADS[modality]
            if head_id != modality:
                raise ValueError(f"Splice head '{head_id}' must be named '{modality}'.")
        elif modality in _TRACK_KINDS:
            prefix, kind = f"heads.{head_id}.", modality
        else:
            raise ValueError(f"Unsupported head '{head_id}' with modality {modality!r}.")
        head = _head_weights(weights, prefix)
        if not head:
            raise ValueError(f"Head '{head_id}' has no weights in the checkpoint.")
        name = f"head/{head_id}"
        extra_columns: dict[str, list] = {}

        if modality in _TRACK_KINDS:
            resolutions = sorted(
                int(match.group(1))
                for key in head if (match := re.fullmatch(r"convs\.(\d+)\.weight", key))
            )
            for resolution in resolutions:
                params[f"{name}/resolution_{resolution}/multi_organism_linear"] = _linear(
                    head[f"convs.{resolution}.weight"], head[f"convs.{resolution}.bias"], slot,
                    f"{prefix}convs.{resolution}",
                )
                params[f"{name}/resolution_{resolution}"] = {
                    "learnt_scale": _organism_slot(
                        head[f"residual_scales.{resolution}"], slot,
                        f"{prefix}residual_scales.{resolution}",
                    ).astype(np.float32),
                }
            num_rows = head[f"convs.{resolutions[0]}.bias"].shape[-1]
            track_means = head.get("track_means")
            if track_means is None:
                track_means = np.ones((1, num_rows), dtype=np.float32)
            track_means = np.atleast_2d(track_means)
            extra_columns["nonzero_mean"] = [
                float(value) for value in _organism_slot(track_means, slot, f"{prefix}track_means")[0]
            ]
            config = custom_heads_module.get_predefined_head_config(
                kind, num_tracks=num_rows, resolutions=resolutions,
            )
        else:
            params[f"{name}/multi_organism_linear"] = _linear(
                head["conv.weight"], head["conv.bias"], slot, f"{prefix}conv",
            )
            num_rows = head["conv.bias"].shape[-1]
            if modality == "splice_junctions":
                for rope in _ROPE_PARAMS:
                    embeddings = _organism_slot(head[f"rope_params.{rope}"], slot, f"{prefix}{rope}")
                    params[f"{name}/{rope}_logits"] = {
                        "embeddings": embeddings.reshape(1, -1).astype(np.float32),
                    }
                num_rows = head["rope_params.pos_donor"].shape[2]
            config = custom_heads_module.get_predefined_head_config(kind, num_tracks=num_rows)

        rows = _track_rows(head_id, num_rows, metadata, organism, len(new_heads) == 1)
        table = pd.DataFrame(rows)
        if "strand" not in table:
            table["strand"] = "." if modality in _TRACK_KINDS else "+"
        table["strand"] = table["strand"].fillna(".")
        for column, values in extra_columns.items():
            table[column] = values
        head_configs[head_id] = _HeadConfigEntry(
            source="predefined", config=dataclasses.replace(config, name=head_id),
        )
        head_metadata[head_id] = {
            organism: AlphaGenomeOutputMetadata(**{config.output_type.name.lower(): table})
        }

    if not head_configs:
        raise ValueError("Checkpoint transfer_config defines no new heads to convert.")
    config = _checkpoint_config(
        custom_heads=list(head_configs),
        head_configs=head_configs,
        head_metadata=head_metadata,
        save_full_model=False,
        save_minimal_model=False,
        use_encoder_output=False,
        backbone_lora_config=lora_config,
    )
    return _write_checkpoint(output_dir, config, params, {})


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m alphagenome_ft.convert_pytorch",
        description="Convert an alphagenome-pytorch finetuned checkpoint to an alphagenome_ft checkpoint.",
    )
    parser.add_argument("checkpoint", help="PyTorch .pth, .delta.pth or .safetensors checkpoint.")
    parser.add_argument("output_dir", help="alphagenome_ft checkpoint directory to write.")
    parser.add_argument(
        "--organism", choices=sorted(_ORGANISMS),
        help="Organism the heads were trained for (default: the checkpoint's, else human).",
    )
    args = parser.parse_args(argv)
    output_dir = convert_pytorch_checkpoint(args.checkpoint, args.output_dir, organism=args.organism)
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
