"""Configuration and head setup for fine-tuning.

Expected YAML/JSON format:

heads:
  - name: my_custom_head
    source: custom
    targets:
      - /data/tracks/track1.bw
      - path: /data/tracks/track2.bw
        name: track2
    # Custom heads must be registered in code before loading this config.

  - name: my_rna_instance  # instance name
    source: predefined
    config:
      name: rna_seq  # predefined head kind (config name)
      resolutions: [1, 128]
      apply_squashing: true
      # num_tracks is inferred from targets if omitted.
    targets:
      - /data/rnaseq/track_001.bw
      - /data/rnaseq/track_002.bw
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

from alphagenome.models import dna_output
from alphagenome_research.model import heads as predefined_heads

from alphagenome_ft import (
    get_custom_head_config,
    is_custom_head,
    is_predefined_head,
    normalize_head_name,
)


@dataclasses.dataclass(frozen=True)
class TrackInfo:
    """Description of a single BigWig target channel."""

    name: str
    path: Path


@dataclasses.dataclass(frozen=True)
class HeadSpec:
    """Configuration for a finetuning head."""

    head_name: str
    source: str
    tracks: Sequence[TrackInfo]
    config: Any | None = None


def load_targets_config(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f'Target config not found: {path}')
    text = path.read_text()
    if path.suffix.lower() in {'.yml', '.yaml'}:
        if yaml is None:
            raise ImportError('pyyaml is required to parse YAML configs for targets.')
        return yaml.safe_load(text)
    return json.loads(text)


def parse_track_list(entries: Sequence[str | Mapping[str, Any]]) -> list[TrackInfo]:
    tracks: list[TrackInfo] = []
    for item in entries:
        if isinstance(item, str):
            path = Path(item)
            name = path.stem
        elif isinstance(item, Mapping):
            path = Path(item['path'])
            name = str(item.get('name', path.stem))
        else:
            raise ValueError(f'Unsupported track entry: {item!r}')
        if not path.exists():
            raise FileNotFoundError(f'BigWig not found: {path}')
        tracks.append(TrackInfo(name=name, path=path))
    return tracks


def build_head_specs(targets_config: Path) -> list[HeadSpec]:
    config = load_targets_config(targets_config)
    heads_cfg = config.get('heads') or config.get('targets')
    if not heads_cfg:
        raise ValueError('Config must define a non-empty "heads" list.')

    specs: list[HeadSpec] = []
    seen_names: set[str] = set()
    for entry in heads_cfg:
        if 'name' not in entry:
            raise ValueError('Each head entry must include a "name".')
        if 'source' not in entry:
            raise ValueError(
                f'Head "{entry["name"]}" must include a "source" (custom|predefined).'
            )

        head_name = normalize_head_name(entry['name'])
        if head_name in seen_names:
            raise ValueError(f'Duplicate head name "{head_name}" in config.')
        seen_names.add(head_name)

        source = str(entry['source']).lower()
        targets = entry.get('targets')
        if targets is None:
            raise ValueError(f'Head "{head_name}" must include "targets".')
        tracks = parse_track_list(targets)
        config_block = entry.get('config')
        if config_block is not None and not isinstance(config_block, Mapping):
            raise ValueError(f'Head "{head_name}" config must be a mapping.')

        if source == 'custom':
            if not is_custom_head(head_name):
                raise ValueError(
                    f'Custom head "{head_name}" is not registered. '
                    'Register it before loading this config.'
                )
            if config_block:
                raise ValueError(
                    f'Custom head "{head_name}" does not accept a config block.'
                )
            head_config = get_custom_head_config(head_name)
            if head_config is not None and len(tracks) != head_config.num_tracks:
                raise ValueError(
                    f'Custom head "{head_name}" expects {head_config.num_tracks} '
                    f'target tracks, got {len(tracks)}.'
                )
            specs.append(
                HeadSpec(
                    head_name=head_name,
                    source='custom',
                    tracks=tracks,
                )
            )
        elif source == 'predefined':
            if not config_block or 'name' not in config_block:
                raise ValueError(
                    f'Predefined head "{head_name}" must include config.name.'
                )
            kind_name = config_block['name']
            if not is_predefined_head(kind_name):
                raise ValueError(
                    f'Unknown predefined head kind "{kind_name}".'
                )
            kind_enum = _normalize_predefined_kind(kind_name)
            if kind_enum is None:
                raise ValueError(
                    f'Unknown predefined head kind "{kind_name}".'
                )
            base_config = predefined_heads.get_head_config(kind_enum)
            overrides = dict(config_block)
            overrides.pop('name', None)
            _validate_predefined_overrides(kind_enum, base_config, overrides)
            if 'num_tracks' in overrides:
                if int(overrides['num_tracks']) != len(tracks):
                    raise ValueError(
                        f'Predefined head "{head_name}" num_tracks '
                        f'{overrides["num_tracks"]} does not match targets '
                        f'({len(tracks)}).'
                    )
            overrides['num_tracks'] = len(tracks)
            head_config = dataclasses.replace(base_config, **overrides)
            specs.append(
                HeadSpec(
                    head_name=head_name,
                    source='predefined',
                    tracks=tracks,
                    config=head_config,
                )
            )
        else:
            raise ValueError(
                f'Head "{head_name}" has invalid source "{source}" (custom|predefined).'
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
                    f'Predefined head "{spec.head_name}" missing config.'
                )
            continue
        if not is_custom_head(spec.head_name):
            raise ValueError(
                f'Custom head "{spec.head_name}" is not registered. '
                'Register it before loading this config.'
            )


def _normalize_predefined_kind(kind_name: Any):
    if isinstance(kind_name, predefined_heads.HeadName):
        return kind_name
    for candidate in predefined_heads.HeadName:
        if str(kind_name).lower() in {candidate.value.lower(), candidate.name.lower()}:
            return candidate
    return None


def _validate_predefined_overrides(kind_enum, base_config, overrides: dict[str, Any]) -> None:
    allowed_fields = {field.name for field in dataclasses.fields(base_config)}
    unknown = set(overrides) - allowed_fields
    if unknown:
        raise ValueError(
            f'Unknown config field(s) for predefined head "{kind_enum.value}": '
            f'{sorted(unknown)}'
        )
    for field_name, enum_cls in (
        ('type', predefined_heads.HeadType),
        ('output_type', dna_output.OutputType),
    ):
        if field_name in overrides:
            overrides[field_name] = _coerce_enum(overrides[field_name], enum_cls)
            if overrides[field_name] != getattr(base_config, field_name):
                raise ValueError(
                    f'Predefined head "{kind_enum.value}" has fixed {field_name}.'
                )
    if hasattr(base_config, 'bundle') and 'bundle' in overrides:
        from alphagenome_research.io import bundles
        overrides['bundle'] = _coerce_enum(overrides['bundle'], bundles.BundleName)
        if overrides['bundle'] != getattr(base_config, 'bundle'):
            raise ValueError(
                f'Predefined head "{kind_enum.value}" has fixed bundle.'
            )
    if 'num_tracks' in overrides:
        overrides['num_tracks'] = int(overrides['num_tracks'])


def _coerce_enum(value: Any, enum_cls):
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls[value]
        except KeyError:
            return enum_cls(value)
    return enum_cls(value)
