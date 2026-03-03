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


@dataclasses.dataclass(frozen=True)
class HeadSpec:
    """Parsed head definition used by training, registration, and validation."""

    head_id: str  # Unique head instance identifier used in training/checkpoints.
    source: str  # Head source type: "custom" or "predefined".
    kind: str | None  # Predefined head kind (e.g., "rna_seq"); None for custom.
    tracks: Sequence[TrackInfo]  # Target tracks used as supervision for this head.
    config: Any | None = None  # Runtime predefined-head config object when applicable.
    metadata: Mapping[dna_client.Organism, AlphaGenomeOutputMetadata] | None = None  # Optional per-organism track metadata.


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
        tracks.append(TrackInfo(name=name, path=path, nonzero_mean=nonzero_mean))
    return tracks


def _build_track_metadata(
    tracks: Sequence[TrackInfo],
    organism: dna_client.Organism | None,
    output_type: dna_output.OutputType,
) -> Mapping[dna_client.Organism, AlphaGenomeOutputMetadata]:
    """Build an AlphaGenomeOutputMetadata mapping from user-provided tracks.

    The returned metadata is used to initialise predefined heads, which derive
    their output dimensionality (num_tracks) from it.  We always return a valid
    mapping so that the head constructor does not receive an empty dict.
    ``nonzero_mean`` values stored on each track are kept for use by the data
    pipeline but are not required for the metadata object itself.
    """
    df = pd.DataFrame(
        {
            "name": [track.name for track in tracks],
            "strand": ["+"] * len(tracks),
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


def prepare_head_specs(
    config: Mapping[str, Any],
    *,
    organism: str | None = None,
) -> list[HeadSpec]:
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
    return specs


def validate_head_specs(specs: Sequence[HeadSpec]) -> None:
    """Optional sanity-check pass for HeadSpec sequences before training."""
    seen_ids: set[str] = set()
    for spec in specs:
        if not spec.head_id:
            raise ValueError('Head spec has empty "head_id".')
        if spec.head_id in seen_ids:
            raise ValueError(f'Duplicate head id "{spec.head_id}" in head specs.')
        seen_ids.add(spec.head_id)

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
