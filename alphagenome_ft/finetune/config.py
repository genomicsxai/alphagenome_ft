"""Configuration and head setup for fine-tuning.

Expected YAML/JSON format:

heads:
  - id: my_custom_head
    source: custom
    targets:
      - /data/tracks/track1.bw
      - /data/tracks/track2.bw
    # Custom heads must be registered in code before loading this config.

  - id: my_rna_instance  # instance alias used for training/checkpoints
    source: predefined
    kind: rna_seq         # predefined head kind
    # Optional predefined-head overrides:
    # resolutions: [1, 128]
    # apply_squashing: true # true for RNA-seq, false for the others
    # Use `targets` for all track definitions.
    targets:
      - path: /data/rnaseq/track_001.bw
        label: track_001  # optional; defaults to filename stem
        nonzero_mean: 0.8  # optional; defaults to 1.0 if any are provided
      - path: /data/rnaseq/track_002.bw
      # Must include one entry per track.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import yaml
except ImportError:  # pragma: no cover - handled at runtime
    yaml = None

import pandas as pd

from alphagenome.models import dna_client
from alphagenome.models import dna_output
from alphagenome_research.model.metadata.metadata import AlphaGenomeOutputMetadata

from alphagenome_ft import (
    get_custom_head_config,
    get_predefined_head_config,
    is_custom_head,
    is_predefined_head,
    list_predefined_heads,
    normalize_head_name,
)


@dataclasses.dataclass(frozen=True)
class TrackInfo:
    """Description of a single BigWig target channel."""

    name: str
    path: Path
    nonzero_mean: float | None = None
    strand: str | None = None  # Explicit +, -, or .; None infers from the label.

    def __post_init__(self) -> None:
        if self.strand not in (None, "+", "-", "."):
            raise ValueError(f"Invalid strand {self.strand!r} for track {self.name!r}.")


SPLICE_KINDS = (
    "splice_sites_classification",
    "splice_sites_usage",
    "splice_sites_junction",
)


@dataclasses.dataclass(frozen=True)
class SpliceSourceSpec:
    """Splice-specific data source, shared across a group of splice heads.

    Distinct from :class:`TrackInfo`/``targets`` (per-file continuous BigWig
    tracks): splice targets are derived at batch time from STAR junction
    files (+ optional SSU parquets / GTF annotation), not read directly from
    a single file per channel.
    """

    star_junctions: Sequence[str]
    ssu: Sequence[str] | None = None
    gtf: str | None = None
    min_unique_reads: int = 1
    filter_to_junctions: bool = True
    max_splice_sites: int = 256
    # junction ("splice_sites_junction") kind only:
    junction_position_source: str = "annotated"  # "annotated" | "predicted"
    junction_top_k: int = 512  # alphagenome_research DEFAULT_NUM_SPLICE_SITES
    splice_site_threshold: float = 0.1  # alphagenome_research DEFAULT_SPLICE_SITE_THRESHOLD
    classification_head_id: str | None = None  # required when junction_position_source="predicted"


@dataclasses.dataclass(frozen=True)
class HeadSpec:
    """Parsed head definition used by training, registration, and validation."""

    head_id: str  # Unique head instance identifier used in training/checkpoints.
    source: str  # Head source type: "custom" or "predefined".
    kind: str | None  # Predefined head kind (e.g., "rna_seq"); None for custom.
    tracks: Sequence[TrackInfo]  # Target tracks used as supervision for this head.
    config: Any | None = None  # Runtime predefined-head config object when applicable.
    metadata: Mapping[dna_client.Organism, AlphaGenomeOutputMetadata] | None = None  # Optional per-organism track metadata.
    splice_source: SpliceSourceSpec | None = None  # Set only for splice kinds; drives the splice targets.


def _resolve_target_path(path_value: str, base_dir: Path | None) -> str:
    target_path = Path(path_value).expanduser()
    if base_dir is not None and not target_path.is_absolute():
        target_path = base_dir / target_path
    return str(target_path)


def _normalize_targets_config_paths(
    config: Mapping[str, Any],
    *,
    base_dir: Path | None = None,
) -> Mapping[str, Any]:
    """Normalize target entries and resolve relative target paths.

    Args:
        config: Parsed config mapping that may include a ``heads`` list.
        base_dir: Directory used to resolve relative target ``path`` values.

    Returns:
        A normalized config mapping where each ``targets`` entry is a mapping
        with a resolved ``path`` string.
    """
    if config is None:
        return {}

    heads_cfg = config.get('heads')
    if heads_cfg is None:
        return config

    if base_dir is not None:
        base_dir = Path(base_dir).expanduser().resolve()

    normalized_heads: list[dict[str, Any]] = []
    for head in heads_cfg:
        if not isinstance(head, Mapping):
            raise ValueError(f'Each head entry must be a mapping: {head!r}')

        head_dict = dict(head)
        # Splice data files resolve like BigWig target paths.
        for key in ('star_junctions', 'ssu'):
            if head_dict.get(key) is not None:
                head_dict[key] = [_resolve_target_path(str(p), base_dir) for p in head_dict[key]]
        if head_dict.get('gtf') is not None:
            head_dict['gtf'] = _resolve_target_path(str(head_dict['gtf']), base_dir)
        targets = head_dict.get('targets')
        if targets is None:
            normalized_heads.append(head_dict)
            continue

        if not isinstance(targets, Sequence) or isinstance(targets, (str, bytes)):
            raise ValueError(
                f'Head "{head_dict.get("id", "<unknown>")}" field "targets" '
                'must be a list of track paths or target mappings.'
            )

        normalized_targets: list[dict[str, Any]] = []
        for item in targets:
            if isinstance(item, str):
                normalized_targets.append(
                    {'path': _resolve_target_path(item, base_dir)}
                )
                continue

            if not isinstance(item, Mapping):
                raise ValueError(
                    f'Each target entry must be a string path or mapping: {item!r}'
                )
            if 'path' not in item:
                raise ValueError(f'Each target must include "path": {item!r}')

            target_item = dict(item)
            target_item['path'] = _resolve_target_path(
                str(target_item['path']), base_dir
            )
            normalized_targets.append(target_item)

        head_dict['targets'] = normalized_targets
        normalized_heads.append(head_dict)

    normalized_config = dict(config)
    normalized_config['heads'] = normalized_heads
    return normalized_config


def _load_targets_config_file(path: Path) -> Mapping[str, Any]:
    """Load a YAML/JSON targets config file without path normalization.

    Args:
        path: Path to a YAML or JSON config file.

    Returns:
        Parsed config mapping.
    """
    if not path.exists():
        raise FileNotFoundError(f'Target config not found: {path}')
    text = path.read_text()
    if path.suffix.lower() in {'.yml', '.yaml'}:
        if yaml is None:
            raise ImportError('pyyaml is required to parse YAML configs.')
        config = yaml.safe_load(text)
    else:
        config = json.loads(text)
    return config


def load_targets_config(
    source: Path | Mapping[str, Any],
    base_dir: Path | None = None,
) -> Mapping[str, Any]:
    """Load/normalize targets config from a path or in-memory mapping.

    Args:
        source: Either a path to targets config file or a parsed config mapping.
        base_dir: Optional base directory used for relative target paths.

    Returns:
        A normalized config mapping.
    """
    if isinstance(source, Mapping):
        return _normalize_targets_config_paths(
            source,
            base_dir=base_dir,
        )
    if isinstance(source, Path):
        resolved_source = source.expanduser().resolve()
        effective_base_dir = (
            base_dir.expanduser().resolve()
            if base_dir is not None
            else resolved_source.parent
        )
        return _normalize_targets_config_paths(
            _load_targets_config_file(resolved_source),
            base_dir=effective_base_dir,
        )
    raise TypeError(
        f'load_targets_config expected Path or Mapping, got {type(source)!r}.'
    )


def _parse_targets(entries: Sequence[Mapping[str, Any]]) -> list[TrackInfo]:
    """Convert normalized target entries into validated TrackInfo objects."""
    tracks: list[TrackInfo] = []
    for item in entries:
        if not isinstance(item, Mapping):
            raise ValueError(
                "Entries under 'targets' must be mappings with a 'path'."
            )
        if 'path' not in item:
            raise ValueError(f'Each target must include "path": {item!r}')
        path = Path(str(item['path']))
        name = str(item.get('label') or path.stem)
        nonzero_mean = item.get('nonzero_mean')
        if nonzero_mean is not None:
            nonzero_mean = float(nonzero_mean)
        if not path.exists():
            raise FileNotFoundError(f'Target file not found: {path}')
        tracks.append(TrackInfo(name=name, path=path, nonzero_mean=nonzero_mean, strand=item.get("strand")))
    return tracks


def _metadata_from_names(
    names: Sequence[str],
    organism: dna_client.Organism | None,
    output_type: dna_output.OutputType,
) -> Mapping[dna_client.Organism, AlphaGenomeOutputMetadata]:
    """Build an AlphaGenomeOutputMetadata mapping from a list of track names.

    The returned metadata is used to initialise predefined heads, which derive
    their output dimensionality (num_tracks) from the row count of the
    ``AlphaGenomeOutputMetadata`` field matching ``output_type``. We always
    return a valid mapping so that the head constructor does not receive an
    empty dict.
    """
    df = pd.DataFrame(
        {
            "name": list(names),
            "strand": ["+"] * len(names),
        }
    )
    # AlphaGenomeOutputMetadata stores per-output-type DataFrames as named
    # fields whose names are the lower-cased OutputType enum member names
    # (e.g. OutputType.RNA_SEQ -> field "rna_seq").
    field_name = output_type.name.lower()
    ag_metadata = AlphaGenomeOutputMetadata(**{field_name: df})
    if organism is not None:
        return {organism: ag_metadata}
    return {org: ag_metadata for org in dna_client.Organism}


def _infer_track_strand(name: str) -> str:
    """Infer a track strand from supported forward/reverse label suffixes."""
    lowered = name.lower()
    if lowered.endswith(("_forward", ".forward", "_plus", ".plus", "+")):
        return "+"
    if lowered.endswith(("_reverse", ".reverse", "_minus", ".minus", "-")):
        return "-"
    return "."


def _compute_missing_track_means(
    tracks: Sequence[TrackInfo],
    intervals: Sequence[Any] | None,
    *,
    max_samples: int | None,
) -> list[TrackInfo]:
    """Populate an entirely unspecified mean set from training intervals."""
    if intervals is None or any(track.nonzero_mean is not None for track in tracks):
        return list(tracks)

    from alphagenome_ft.finetune.data import compute_track_nonzero_means

    strands = [
        track.strand
        if track.strand is not None
        else _infer_track_strand(track.name)
        for track in tracks
    ]
    plus_indices = [index for index, strand in enumerate(strands) if strand == "+"]
    minus_indices = [index for index, strand in enumerate(strands) if strand == "-"]
    if len(plus_indices) != len(minus_indices):
        raise ValueError(
            "Stranded targets require equal numbers of + and - tracks before "
            "nonzero means can be estimated."
        )
    means = compute_track_nonzero_means(
        [track.path for track in tracks],
        intervals,
        max_samples=max_samples,
        strand_pair_groups=list(zip(plus_indices, minus_indices)),
    )
    return [
        dataclasses.replace(track, nonzero_mean=float(means[index]))
        for index, track in enumerate(tracks)
    ]


def _build_track_metadata(
    tracks: Sequence[TrackInfo],
    organism: dna_client.Organism | None,
    output_type: dna_output.OutputType,
) -> Mapping[dna_client.Organism, AlphaGenomeOutputMetadata]:
    """Build an AlphaGenomeOutputMetadata mapping from user-provided tracks.

    Includes a ``nonzero_mean`` column (defaulting missing per-track values to
    1.0, i.e. no scaling) whenever any track provides one. This matters: the
    real predefined head classes (``alphagenome_research.model.heads``, e.g.
    ``BaseResolutionHead.__init__``'s ``_get_track_means``) read
    ``metadata['nonzero_mean']`` directly to rescale predictions/targets on
    every forward pass - without this column they silently fall back to
    all-ones (no scaling).

    Strand comes from ``TrackInfo.strand`` when given, otherwise from common
    forward/reverse label suffixes (``_infer_track_strand``, as upstream).
    Stranded targets need equal numbers of + and - tracks (reverse-complement
    reindexing). Pass ``strand: "."`` per target to reproduce the pre-merge
    behaviour of this branch (no strand awareness).
    """
    strands = [
        track.strand if track.strand is not None else _infer_track_strand(track.name)
        for track in tracks
    ]
    if strands.count("+") != strands.count("-"):
        raise ValueError(
            "Stranded targets require equal numbers of + and - tracks for reverse-complement "
            "reindexing. Supply paired tracks in matching order, or strand='.' for unstranded targets."
        )
    columns: dict[str, list[Any]] = {
        "name": [track.name for track in tracks],
        "strand": strands,
    }
    if any(track.nonzero_mean is not None for track in tracks):
        columns["nonzero_mean"] = [
            1.0 if track.nonzero_mean is None else float(track.nonzero_mean)
            for track in tracks
        ]
    df = pd.DataFrame(columns)
    field_name = output_type.name.lower()
    ag_metadata = AlphaGenomeOutputMetadata(**{field_name: df})
    if organism is not None:
        return {organism: ag_metadata}
    return {org: ag_metadata for org in dna_client.Organism}


# Fixed 5-class label order matching the classification target built by
# ``alphagenome_ft.finetune.splice_data`` (and the JAX SpliceSitesClassificationHead).
_SPLICE_CLASSIFICATION_LABELS = (
    "donor_pos", "acceptor_pos", "donor_neg", "acceptor_neg", "none",
)


def _parse_splice_source(entry: Mapping[str, Any], head_id: str) -> "SpliceSourceSpec":
    star_junctions = entry.get("star_junctions")
    if not star_junctions:
        raise ValueError(
            f'Splice head "{head_id}" must include a non-empty "star_junctions" list.'
        )
    ssu = entry.get("ssu")
    for label, paths in (("star_junctions", star_junctions), ("ssu", ssu)):
        if paths is None:
            continue
        for path_value in paths:
            if not Path(str(path_value)).exists():
                raise FileNotFoundError(f'Splice head "{head_id}" {label} file not found: {path_value}')
    gtf = entry.get("gtf")
    if gtf is not None and not Path(str(gtf)).exists():
        raise FileNotFoundError(f'Splice head "{head_id}" gtf file not found: {gtf}')

    junction_position_source = str(entry.get("junction_position_source", "annotated"))
    if junction_position_source not in ("annotated", "predicted"):
        raise ValueError(
            f'Splice head "{head_id}" junction_position_source must be '
            f'"annotated" or "predicted", got {junction_position_source!r}.'
        )
    classification_head_id = entry.get("classification_head_id")
    if junction_position_source == "predicted" and not classification_head_id:
        raise ValueError(
            f'Splice head "{head_id}" has junction_position_source="predicted" '
            'and must include "classification_head_id" (the id of the '
            'splice_sites_classification head to source positions from).'
        )

    return SpliceSourceSpec(
        star_junctions=[str(p) for p in star_junctions],
        ssu=[str(p) for p in ssu] if ssu is not None else None,
        gtf=str(gtf) if gtf is not None else None,
        min_unique_reads=int(entry.get("min_unique_reads", 1)),
        filter_to_junctions=bool(entry.get("filter_to_junctions", True)),
        max_splice_sites=int(entry.get("max_splice_sites", 256)),
        junction_position_source=junction_position_source,
        junction_top_k=int(entry.get("junction_top_k", 512)),
        splice_site_threshold=float(entry.get("splice_site_threshold", 0.1)),
        classification_head_id=(
            str(classification_head_id) if classification_head_id is not None else None
        ),
    )


def _build_splice_metadata(
    kind_name: str,
    splice_source: "SpliceSourceSpec",
    organism: dna_client.Organism | None,
    output_type: dna_output.OutputType,
) -> tuple[Mapping[dna_client.Organism, AlphaGenomeOutputMetadata], int]:
    """Build splice head metadata + implied ``num_tracks`` from a splice source.

    Track counts follow the shapes produced by ``splice_data.SpliceTargets``:
      * classification: fixed 5 classes (Donor+/Acceptor+/Donor-/Acceptor-/None).
      * usage: 2 channels (pos/neg strand) per sample (SSU file if given, else
        STAR junction file).
      * junction: 1 "tissue" per STAR junction file; the head's actual track
        count is ``2 * num_tissues`` (computed internally from metadata row
        count), so metadata carries one row per tissue, not per channel.
    """
    if kind_name == "splice_sites_classification":
        names = list(_SPLICE_CLASSIFICATION_LABELS)
    elif kind_name == "splice_sites_usage":
        samples = splice_source.ssu if splice_source.ssu is not None else splice_source.star_junctions
        sample_names = [Path(p).stem for p in samples]
        names = [n for name in sample_names for n in (f"{name}_pos", f"{name}_neg")]
    elif kind_name == "splice_sites_junction":
        names = [Path(p).stem for p in splice_source.star_junctions]
    else:
        raise ValueError(f"Unsupported splice kind {kind_name!r}.")
    metadata = _metadata_from_names(names, organism, output_type)
    return metadata, len(names)


def prepare_head_specs(
    config: Mapping[str, Any],
    *,
    organism: str | None = None,
    nonzero_mean_intervals: Sequence[Any] | None = None,
    nonzero_mean_max_samples: int | None = 1000,
) -> list[HeadSpec]:
    """Build head specs and optionally estimate missing predefined-head scales.

    When ``nonzero_mean_intervals`` is provided and none of a predefined head's
    targets declares ``nonzero_mean``, scales are computed from those training
    intervals using the same nonzero-value reduction as ``alphagenome-pytorch``.
    If any target declares a value, the existing explicit/default-1.0 behavior
    is retained for that head.
    """
    heads_cfg = config.get('heads')
    if not heads_cfg:
        raise ValueError('Config must define a non-empty "heads" list.')

    organism_enum: dna_client.Organism | None = None
    if organism is not None:
        try:
            organism_enum = dna_client.Organism[organism]
        except KeyError as exc:
            raise ValueError(f'Unknown organism "{organism}".') from exc

    specs: list[HeadSpec] = []
    seen_ids: set[str] = set()
    for entry in heads_cfg:
        if 'id' not in entry:
            raise ValueError('Each head entry must include an "id".')
        if 'source' not in entry:
            raise ValueError(
                f'Head "{entry["id"]}" must include a "source" (custom|predefined).'
            )

        head_id = normalize_head_name(entry['id'])
        if head_id in seen_ids:
            raise ValueError(f'Duplicate head id "{head_id}" in config.')
        seen_ids.add(head_id)

        source = str(entry['source']).lower()
        kind_name = entry.get('kind')
        if source == 'predefined' and kind_name in SPLICE_KINDS:
            splice_source = _parse_splice_source(entry, head_id)
            # num_tracks is a no-op override here: the base HeadConfig for
            # splice kinds has no num_tracks field (see get_predefined_head_config).
            head_config = get_predefined_head_config(kind_name, num_tracks=1)
            metadata, _num_tracks = _build_splice_metadata(
                kind_name, splice_source, organism_enum, head_config.output_type
            )
            specs.append(
                HeadSpec(
                    head_id=head_id,
                    source='predefined',
                    kind=str(kind_name),
                    tracks=(),
                    config=head_config,
                    metadata=metadata,
                    splice_source=splice_source,
                )
            )
            continue

        targets = entry.get('targets')
        if targets is None:
            raise ValueError(f'Head "{head_id}" must include "targets".')
        tracks = _parse_targets(targets)
        if source == 'custom':
            if not is_custom_head(head_id):
                raise ValueError(
                    f'Custom head "{head_id}" is not registered. '
                    'Register it before loading this config.'
                )
            head_config = get_custom_head_config(head_id)
            if head_config is not None and len(tracks) != head_config.num_tracks:
                raise ValueError(
                    f'Custom head "{head_id}" expects {head_config.num_tracks} '
                    f'target tracks, got {len(tracks)}.'
                )
            specs.append(
                HeadSpec(
                    head_id=head_id,
                    source='custom',
                    kind=None,
                    tracks=tracks,
                )
            )
        elif source == 'predefined':
            tracks = _compute_missing_track_means(
                tracks,
                nonzero_mean_intervals,
                max_samples=nonzero_mean_max_samples,
            )
            kind_name = entry.get('kind')
            if not kind_name:
                raise ValueError(
                    f'Predefined head "{head_id}" must include "kind".'
                )
            overrides = {
                field: entry.get(field)
                for field in (
                    'resolutions',
                    'apply_squashing',
                    'embedding_channels',
                    'num_tissues',
                    'num_tracks',
                )
                if field in entry
            }
            if not is_predefined_head(kind_name):
                valid_kinds = ", ".join(sorted(list_predefined_heads()))
                raise ValueError(
                    f'Unknown predefined head kind "{kind_name}". '
                    f"Available kinds: {valid_kinds}"
                )
            if 'num_tracks' in overrides:
                if int(overrides['num_tracks']) != len(tracks):
                    raise ValueError(
                        f'Predefined head "{head_id}" num_tracks '
                        f'{overrides["num_tracks"]} does not match targets '
                        f'({len(tracks)}).'
                    )
            head_config = get_predefined_head_config(
                kind_name,
                num_tracks=len(tracks),
                resolutions=overrides.get('resolutions'),
                apply_squashing=overrides.get('apply_squashing'),
                embedding_channels=overrides.get('embedding_channels'),
                num_tissues=overrides.get('num_tissues'),
            )
            metadata = _build_track_metadata(tracks, organism_enum, head_config.output_type)
            specs.append(
                HeadSpec(
                    head_id=head_id,
                    source='predefined',
                    kind=str(kind_name),
                    tracks=tracks,
                    config=head_config,
                    metadata=metadata,
                )
            )
        else:
            raise ValueError(
                f'Head "{head_id}" has invalid source "{source}" (custom|predefined).'
            )

    if not specs:
        raise ValueError('Config did not produce any heads.')
    _validate_predicted_junction_sources(specs)
    return specs


def _validate_predicted_junction_sources(specs: Sequence[HeadSpec]) -> None:
    """Cross-head check: predicted-mode junction heads need their classification head."""
    classification_ids = {
        spec.head_id for spec in specs if spec.kind == "splice_sites_classification"
    }
    for spec in specs:
        if spec.kind != "splice_sites_junction" or spec.splice_source is None:
            continue
        if spec.splice_source.junction_position_source != "predicted":
            continue
        cls_id = spec.splice_source.classification_head_id
        if cls_id not in classification_ids:
            raise ValueError(
                f'Junction head "{spec.head_id}" has junction_position_source='
                f'"predicted" and classification_head_id={cls_id!r}, but no '
                f'"splice_sites_classification" head with that id is present '
                f'in this config. Available classification heads: '
                f'{sorted(classification_ids) or "none"}.'
            )


def validate_head_specs(specs: Sequence[HeadSpec]) -> None:
    """Optional sanity-check pass for HeadSpec sequences before training."""
    _validate_predicted_junction_sources(specs)
    seen_ids: set[str] = set()
    for spec in specs:
        if not spec.head_id:
            raise ValueError('Head spec has empty "head_id".')
        if spec.head_id in seen_ids:
            raise ValueError(f'Duplicate head id "{spec.head_id}" in head specs.')
        seen_ids.add(spec.head_id)

        if spec.splice_source is not None:
            if not spec.splice_source.star_junctions:
                raise ValueError(
                    f'Splice head "{spec.head_id}" must include "star_junctions".'
                )
            continue

        if not spec.tracks:
            raise ValueError(
                f'Head "{spec.head_id}" must include at least one target track.'
            )
        for track in spec.tracks:
            if not track.path.exists():
                raise FileNotFoundError(
                    f'Head "{spec.head_id}" target file not found: {track.path}'
                )

        if spec.source == 'custom':
            if not is_custom_head(spec.head_id):
                raise ValueError(
                    f'Custom head "{spec.head_id}" is not registered. '
                    'Register it before loading this config.'
                )
            custom_config = get_custom_head_config(spec.head_id)
            if custom_config is not None and len(spec.tracks) != custom_config.num_tracks:
                raise ValueError(
                    f'Custom head "{spec.head_id}" expects {custom_config.num_tracks} '
                    f'target tracks, got {len(spec.tracks)}.'
                )
        elif spec.source == 'predefined':
            if not spec.kind:
                raise ValueError(
                    f'Predefined head "{spec.head_id}" missing kind.'
                )
            if not is_predefined_head(spec.kind):
                raise ValueError(
                    f'Predefined head "{spec.head_id}" has unknown kind '
                    f'"{spec.kind}".'
                )
            if spec.config is None:
                raise ValueError(
                    f'Predefined head "{spec.head_id}" missing config.'
                )
            if hasattr(spec.config, 'num_tracks'):
                expected_num_tracks = int(spec.config.num_tracks)
                if expected_num_tracks != len(spec.tracks):
                    raise ValueError(
                        f'Predefined head "{spec.head_id}" expects '
                        f'{expected_num_tracks} target tracks, got '
                        f'{len(spec.tracks)}.'
                    )
        else:
            raise ValueError(
                f'Head "{spec.head_id}" has invalid source "{spec.source}" '
                '(custom|predefined).'
            )
