"""Configuration and head setup for fine-tuning.

Expected YAML/JSON format:

heads:
  - id: my_custom_head
    source: custom
    bigwigs:
      - /data/tracks/track1.bw
      - /data/tracks/track2.bw
    # Custom heads must be registered in code before loading this config.

  - id: my_rna_instance  # instance alias used for training/checkpoints
    source: predefined
    kind: rna_seq         # predefined head kind
    # Optional predefined-head overrides:
    # resolutions: [1, 128]
    # apply_squashing: true # true for RNA-seq, false for the others
    # Use either `targets` with metadata or `bigwigs` without metadata. 
    targets:
      - bigwig: /data/rnaseq/track_001.bw
        label: track_001  # optional; defaults to filename stem
        nonzero_mean: 0.8  # optional; defaults to 1.0 if any are provided
      - bigwig: /data/rnaseq/track_002.bw
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

from alphagenome_ft import (
    get_custom_head_config,
    get_predefined_head_config,
    is_custom_head,
    is_predefined_head,
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
    """Configuration for a finetuning head."""

    head_id: str
    source: str
    kind: str | None
    tracks: Sequence[TrackInfo]
    config: Any | None = None
    metadata: Mapping[dna_client.Organism, pd.DataFrame] | None = None


def load_targets_config(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f'Target config not found: {path}')
    text = path.read_text()
    if path.suffix.lower() in {'.yml', '.yaml'}:
        if yaml is None:
            raise ImportError('pyyaml is required to parse YAML configs.')
        return yaml.safe_load(text)
    return json.loads(text)


def parse_bigwigs(entries: Sequence[str]) -> list[TrackInfo]:
    tracks: list[TrackInfo] = []
    for item in entries:
        if not isinstance(item, str):
            raise ValueError(
                "Entries under 'bigwigs' must be file paths (strings). "
                "Use 'targets' for metadata."
            )
        path = Path(item)
        if not path.exists():
            raise FileNotFoundError(f'BigWig not found: {path}')
        tracks.append(TrackInfo(name=path.stem, path=path))
    return tracks


def parse_targets(entries: Sequence[Mapping[str, Any]]) -> list[TrackInfo]:
    tracks: list[TrackInfo] = []
    for item in entries:
        if not isinstance(item, Mapping):
            raise ValueError(
                "Entries under 'targets' must be mappings with a 'bigwig' path."
            )
        if 'bigwig' not in item:
            raise ValueError(f'Each target must include "bigwig": {item!r}')
        path = Path(item['bigwig'])
        name = str(item.get('label') or path.stem)
        nonzero_mean = item.get('nonzero_mean')
        if nonzero_mean is not None:
            nonzero_mean = float(nonzero_mean)
        if not path.exists():
            raise FileNotFoundError(f'BigWig not found: {path}')
        tracks.append(TrackInfo(name=name, path=path, nonzero_mean=nonzero_mean))
    return tracks


def _build_track_metadata(
    tracks: Sequence[TrackInfo],
    organism: dna_client.Organism | None,
) -> Mapping[dna_client.Organism, pd.DataFrame] | None:
    nonzero_means = [track.nonzero_mean for track in tracks]
    if all(value is None for value in nonzero_means):
        return None
    nonzero_means = [
        1.0 if value is None else float(value) for value in nonzero_means
    ]
    df = pd.DataFrame(
        {
            "name": [track.name for track in tracks],
            "nonzero_mean": nonzero_means,
        }
    )
    if organism is not None:
        return {organism: df}
    return {org: df for org in dna_client.Organism}


def build_head_specs(
    targets_config: Path,
    *,
    organism: str | None = None,
) -> list[HeadSpec]:
    config = load_targets_config(targets_config)
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
        bigwigs = entry.get('bigwigs')
        if targets is not None and bigwigs is not None:
            raise ValueError(
                f'Head "{head_id}" must use only one of "targets" or "bigwigs".'
            )
        if targets is None and bigwigs is None:
            raise ValueError(
                f'Head "{head_id}" must include "targets" (preferred) or "bigwigs".'
            )
        if targets is not None:
            tracks = parse_targets(targets)
        else:
            tracks = parse_bigwigs(bigwigs)
        if source == 'custom':
            if 'config' in entry:
                raise ValueError(
                    f'Custom head "{head_id}" does not accept a "config" block.'
                )
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
            if 'config' in entry:
                raise ValueError(
                    f'Predefined head "{head_id}" no longer accepts nested "config". '
                    'Move override fields to the head entry.'
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
                raise ValueError(
                    f'Unknown predefined head kind "{kind_name}".'
                )
            allowed_fields = {
                'resolutions',
                'apply_squashing',
                'embedding_channels',
                'num_tissues',
                'num_tracks',
            }
            unknown = set(overrides) - allowed_fields
            if unknown:
                raise ValueError(
                    f'Unknown config field(s) for predefined head "{head_id}": '
                    f'{sorted(unknown)}'
                )
            if 'num_tracks' in overrides:
                if int(overrides['num_tracks']) != len(tracks):
                    raise ValueError(
                        f'Predefined head "{head_id}" num_tracks '
                        f'{overrides["num_tracks"]} does not match bigwigs '
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
            metadata = _build_track_metadata(tracks, organism_enum)
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


def validate_heads(specs: Sequence[HeadSpec]) -> None:
    """Validate that required custom heads are registered before training."""
    for spec in specs:
        if spec.source != 'custom':
            if spec.source == 'predefined' and spec.config is None:
                raise ValueError(
                    f'Predefined head "{spec.head_id}" missing config.'
                )
            continue
        if not is_custom_head(spec.head_id):
            raise ValueError(
                f'Custom head "{spec.head_id}" is not registered. '
                'Register it before loading this config.'
            )
