"""Configuration and head setup for fine-tuning."""

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

from alphagenome_ft import HeadConfig, HeadType, register_custom_head, templates


@dataclasses.dataclass(frozen=True)
class TrackInfo:
    """Description of a single BigWig target channel."""

    name: str
    path: Path


@dataclasses.dataclass(frozen=True)
class HeadSpec:
    """Configuration for a finetuning head."""

    head_name: str
    output_type: dna_output.OutputType
    tracks: Sequence[TrackInfo]
    template: type = templates.StandardHead


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


def build_head_specs(args) -> list[HeadSpec]:
    config = load_targets_config(args.targets_config)
    heads_cfg = config.get('targets') or config.get('heads')
    if not heads_cfg:
        raise ValueError('Targets config must define a non-empty "targets" list.')

    template_map = {
        'standard': templates.StandardHead,
        'transformer': templates.TransformerHead,
        'encoder': templates.EncoderOnlyHead,
    }

    specs: list[HeadSpec] = []
    for entry in heads_cfg:
        head_name = entry['name']
        output_type_name = entry['output_type'].upper()
        template_name = entry.get('template', 'standard').lower()
        tracks = parse_track_list(entry['bigwigs'])
        template_cls = template_map.get(template_name)
        if template_cls is None:
            raise ValueError(f'Unknown template "{template_name}" for head {head_name}.')
        try:
            output_type = dna_output.OutputType[output_type_name]
        except KeyError as exc:
            raise ValueError(
                f'Unknown output type "{output_type_name}" for head {head_name}.'
            ) from exc
        specs.append(
            HeadSpec(
                head_name=head_name,
                output_type=output_type,
                tracks=tracks,
                template=template_cls,
            )
        )

    if not specs:
        raise ValueError('Targets config did not produce any heads.')
    return specs


def register_heads(specs: Sequence[HeadSpec]) -> None:
    for spec in specs:
        register_custom_head(
            spec.head_name,
            spec.template,
            HeadConfig(
                type=HeadType.GENOME_TRACKS,
                name=spec.head_name,
                output_type=spec.output_type,
                num_tracks=len(spec.tracks),
                metadata={'tracks': [track.name for track in spec.tracks]},
            ),
        )
