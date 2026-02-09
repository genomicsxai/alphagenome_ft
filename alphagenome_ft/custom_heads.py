"""
Custom head management for AlphaGenome finetuning.

Provides base classes and utilities for defining and registering custom prediction heads.

Main functions:
- `register_custom_head`: Register a custom head class and config.
- `register_predefined_head`: Register a predefined head config.
- `create_registered_head`: Instantiate a registered head by name.
- `list_registered_heads`: List all heads registered by the user.
- `list_predefined_heads`: List predefined head kinds.
- `get_registered_head_config`: Retrieve the config for a registered head.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
import enum
from typing import Any, TypeAlias

import haiku as hk
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PyTree
import pandas as pd

from alphagenome.models import dna_output
from alphagenome_research.model import embeddings as embeddings_module
from alphagenome_research.model import heads as predefined_heads

from alphagenome_research.model.heads import (
    HeadName as PredefinedHeadName,
    HeadConfig as PredefinedHeadConfig,
)

_NUM_ORGANISMS = 2


class CustomHeadType(enum.Enum):
    """Types of prediction heads."""
    GENOME_TRACKS = "genome_tracks"
    CONTACT_MAPS = "contact_maps"
    SPLICE_SITES = "splice_sites"
    CUSTOM = "custom"


@dataclass
class CustomHeadConfig:
    """Configuration for a custom prediction head."""
    type: CustomHeadType
    output_type: dna_output.OutputType
    num_tracks: int
    # Optional: Haiku module name for the head. If None, the registry key
    # ``head_name`` will be used as the module name when constructing the head.
    name: str | None = None
    metadata: Mapping | None = None


# Backward compatibility aliases.
HeadType = CustomHeadType
HeadConfig = CustomHeadConfig


HeadNameLike: TypeAlias = str | enum.Enum
HeadConfigLike: TypeAlias = PredefinedHeadConfig | CustomHeadConfig
HeadLike: TypeAlias = "predefined_heads.Head | CustomHead"
HeadFactory: TypeAlias = Callable[[HeadConfigLike, Mapping | None, int], HeadLike]


class CustomHead(ABC, hk.Module):
    """Base class for custom prediction heads.

    Subclass this to create your own prediction heads for finetuning.

    A Haiku Module (`hk.Module`) - integrates with JAX/Haiku.

    Example:
        ```python
        class MyMPRAHead(CustomHead):
            def __init__(self, *, name, output_type, num_tracks, num_organisms, metadata):
                super().__init__(name=name, num_tracks=num_tracks,
                               output_type=output_type, num_organisms=num_organisms,
                               metadata=metadata)

            def predict(self, embeddings, organism_index, **kwargs):
                x = embeddings.get_sequence_embeddings(resolution=1)
                # Add your prediction layers
                output = hk.Linear(self._num_tracks)(x)
                return {'predictions': output}

            def loss(self, predictions, batch):
                # Implement your loss
                return {'loss': jnp.array(0.0)}
        ```
    """

    def __init__(
        self,
        *,
        name: str,
        output_type: dna_output.OutputType,
        num_tracks: int,
        num_organisms: int = _NUM_ORGANISMS,
        metadata: Mapping | None = None,
    ):
        super().__init__(name=name)
        self._output_type = output_type
        self._num_tracks = num_tracks
        self._num_organisms = num_organisms
        self._metadata = metadata or {}

    @abstractmethod
    def predict(
        self,
        embeddings: embeddings_module.Embeddings,
        organism_index: Int[Array, "B"],
        **kwargs,
    ) -> PyTree:
        """Generate predictions from embeddings.

        Args:
            embeddings: Multi-resolution embeddings from the AlphaGenome backbone.
            organism_index: Organism indices for each batch element.
            **kwargs: Additional arguments.

        Returns:
            Dictionary containing predictions and any intermediate outputs.
        """
        pass

    @abstractmethod
    def loss(self, predictions: PyTree, batch: PyTree) -> PyTree:
        """Compute loss for the predictions.

        Args:
            predictions: Output from predict().
            batch: Training batch containing targets and metadata.

        Returns:
            Dictionary containing 'loss' and any other metrics.
        """
        pass

    def __call__(
        self,
        embeddings: embeddings_module.Embeddings,
        organism_index: Int[Array, "B"],
        **kwargs,
    ) -> PyTree:
        """Forward pass through the head."""
        return self.predict(embeddings, organism_index, **kwargs)


# Head registries: instance name -> (factory, config).
_HEAD_REGISTRY: dict[str, HeadFactory] = {}
_HEAD_CONFIG_REGISTRY: dict[str, HeadConfigLike] = {}
_HEAD_METADATA_REGISTRY: dict[str, Mapping | None] = {}


# --- Normalization and classification helpers ---
def _normalize_predefined_head_name(head_name: HeadNameLike) -> PredefinedHeadName | None:
    if isinstance(head_name, predefined_heads.HeadName):
        return head_name
    if isinstance(head_name, str):
        for candidate in predefined_heads.HeadName:
            if head_name == candidate.value:
                return candidate
            if head_name.lower() == candidate.value.lower():
                return candidate
            if head_name.upper() == candidate.name:
                return candidate
            if head_name.lower() == candidate.name.lower():
                return candidate
    return None


def normalize_head_name(head_name: HeadNameLike) -> str:
    """Normalize a head identifier to its string name."""
    predefined_head = _normalize_predefined_head_name(head_name)
    if predefined_head is not None:
        return predefined_head.value
    if isinstance(head_name, str):
        return head_name
    if isinstance(head_name, enum.Enum):
        return str(head_name.value)
    raise TypeError(f"Unsupported head identifier: {head_name!r}")


def is_predefined_head(head_name: HeadNameLike) -> bool:
    """Check if a head name corresponds to a predefined head kind."""
    return _normalize_predefined_head_name(head_name) is not None


def is_custom_config(config: HeadConfigLike) -> bool:
    """Return True if config is a custom head config."""
    return isinstance(config, CustomHeadConfig)


def is_predefined_config(config: HeadConfigLike) -> bool:
    """Return True if config is a predefined head config."""
    return isinstance(config, PredefinedHeadConfig)


# --- Core registry operations ---
def list_registered_heads() -> list[str]:
    """List all registered head instance names."""
    return list(_HEAD_CONFIG_REGISTRY.keys())


def list_predefined_heads() -> list[str]:
    """List all predefined head kind names."""
    return [head.value for head in predefined_heads.HeadName]


def is_head_registered(head_name: HeadNameLike) -> bool:
    """Check if any head config is registered for the given name."""
    normalized = normalize_head_name(head_name)
    return normalized in _HEAD_CONFIG_REGISTRY


def get_registered_head_config(head_name: HeadNameLike) -> HeadConfigLike:
    """Get configuration for a registered head."""
    normalized = normalize_head_name(head_name)
    if normalized not in _HEAD_CONFIG_REGISTRY:
        raise ValueError(
            f"Head '{head_name}' is not registered. "
            f"Available heads: {list_registered_heads()}"
        )
    return _HEAD_CONFIG_REGISTRY[normalized]


def create_registered_head(
    head_name: HeadNameLike,
    *,
    metadata: Mapping | None = None,
    num_organisms: int = _NUM_ORGANISMS,
) -> HeadLike:
    """Instantiate any registered head by instance name."""
    normalized = normalize_head_name(head_name)
    if normalized not in _HEAD_REGISTRY:
        raise ValueError(
            f"Head '{head_name}' is not registered. "
            f"Available heads: {list_registered_heads()}"
        )
    config = get_registered_head_config(normalized)
    factory = _HEAD_REGISTRY[normalized]
    return factory(config, metadata, num_organisms)


# --- Custom head operations ---
def register_custom_head(
    head_name: str,
    head_class: type[CustomHead],
    head_config: CustomHeadConfig,
) -> None:
    """Register a custom head for use in finetuning.

    Args:
        head_name: Unique instance name for the head.
        head_class: Class implementing the CustomHead interface.
        head_config: Configuration for the head.

    Example:
        ```python
        register_custom_head(
            "my_mpra",
            MyMPRAHead,
            CustomHeadConfig(
                type=CustomHeadType.GENOME_TRACKS,
                name="my_mpra",
                output_type=dna_output.OutputType.RNA_SEQ,
                num_tracks=1,
            ),
        )
        ```
    """
    normalized_name = normalize_head_name(head_name)
    if normalized_name in _HEAD_REGISTRY:
        print(f"Warning: Overwriting existing custom head '{normalized_name}'")

    if not head_config.name:
        head_config = replace(head_config, name=normalized_name)

    def _custom_factory(
        config: HeadConfigLike,
        metadata: Mapping | None,
        num_organisms: int,
    ) -> HeadLike:
        if not isinstance(config, CustomHeadConfig):
            raise TypeError(
                f"Expected custom head config for '{normalized_name}', got {type(config)!r}."
            )
        return head_class(
            name=config.name,
            output_type=config.output_type,
            num_tracks=config.num_tracks,
            num_organisms=num_organisms,
            metadata=metadata or config.metadata,
        )

    _HEAD_REGISTRY[normalized_name] = _custom_factory
    _HEAD_CONFIG_REGISTRY[normalized_name] = head_config


def get_custom_head_config(head_name: str) -> CustomHeadConfig | None:
    """Get configuration for a registered custom head."""
    config = _HEAD_CONFIG_REGISTRY.get(normalize_head_name(head_name))
    if isinstance(config, CustomHeadConfig):
        return config
    return None


def is_custom_head(head_name: str) -> bool:
    """Check if a head name corresponds to a registered custom head."""
    config = _HEAD_CONFIG_REGISTRY.get(normalize_head_name(head_name))
    return isinstance(config, CustomHeadConfig)


def list_custom_heads() -> list[str]:
    """List all registered custom head instance names."""
    return [
        name
        for name, config in _HEAD_CONFIG_REGISTRY.items()
        if isinstance(config, CustomHeadConfig)
    ]


def create_custom_head(
    head_name: str,
    metadata: Mapping | None = None,
    num_organisms: int = _NUM_ORGANISMS,
) -> CustomHead:
    """Instantiate a registered custom head."""
    config = get_custom_head_config(head_name)
    if config is None:
        raise ValueError(
            f"Custom head '{head_name}' not registered. "
            f"Available custom heads: {list_custom_heads()}"
        )

    head = create_registered_head(
        head_name,
        metadata=metadata,
        num_organisms=num_organisms,
    )
    if not isinstance(head, CustomHead):
        raise TypeError(
            f"Head '{head_name}' resolved to a predefined head, not a custom head."
        )
    return head


# --- Predefined head operations ---
def register_predefined_head(
    head_name: HeadNameLike,
    config: PredefinedHeadConfig,
    *,
    metadata: Mapping | None = None,
) -> None:
    """Register a predefined head config under an instance name.

    The registered ``head_name`` is also used as the Haiku module name so
    parameter paths consistently include the user-facing alias.
    """
    normalized_name = normalize_head_name(head_name)
    kind_name = _normalize_predefined_head_name(config.name)
    if kind_name is None and config.name != normalized_name:
        raise ValueError(
            f"Config name '{config.name}' is not a known predefined head kind "
            f"and does not match registered alias '{normalized_name}'."
        )
    config_with_alias_name = replace(config, name=normalized_name)

    def _predefined_factory(
        cfg: HeadConfigLike,
        metadata: Mapping | None,
        _num_organisms: int,
    ) -> HeadLike:
        if not isinstance(cfg, PredefinedHeadConfig):
            raise TypeError(
                f"Expected predefined head config for '{normalized_name}', got {type(cfg)!r}."
            )
        return predefined_heads.create_head(cfg, metadata)

    _HEAD_REGISTRY[normalized_name] = _predefined_factory
    _HEAD_CONFIG_REGISTRY[normalized_name] = config_with_alias_name
    _HEAD_METADATA_REGISTRY[normalized_name] = metadata


def get_registered_head_metadata(head_name: HeadNameLike) -> Mapping | None:
    """Get optional metadata for a registered head."""
    normalized = normalize_head_name(head_name)
    return _HEAD_METADATA_REGISTRY.get(normalized)


def get_predefined_head_config(
    head_name: HeadNameLike,
    *,
    num_tracks: int,
    resolutions: list[int] | None = None,
    apply_squashing: bool | None = None,
    embedding_channels: int | None = None,
    num_tissues: int | None = None,
) -> PredefinedHeadConfig:
    """Build a predefined head config with required num_tracks and safe overrides."""
    kind_enum = _normalize_predefined_head_name(head_name)
    if kind_enum is None:
        raise ValueError(f"Head '{head_name}' is not a predefined head.")
    base_config = predefined_heads.get_head_config(kind_enum)
    overrides: dict[str, Any] = {"num_tracks": int(num_tracks)}

    def _maybe_override(field: str, value: Any) -> None:
        if value is None:
            return
        if not hasattr(base_config, field):
            raise ValueError(
                f'Predefined head "{kind_enum.value}" does not support "{field}".'
            )
        overrides[field] = value

    _maybe_override("resolutions", list(resolutions) if resolutions is not None else None)
    _maybe_override("apply_squashing", apply_squashing)
    _maybe_override("embedding_channels", embedding_channels)
    _maybe_override("num_tissues", num_tissues)

    return replace(base_config, **overrides)


def deserialize_predefined_head_config(
    config_dict: Mapping[str, Any],
) -> PredefinedHeadConfig:
    """Rebuild a predefined head config from a serialized dict."""
    from alphagenome_research.io import bundles

    def _coerce_enum(value: Any, enum_cls):
        if isinstance(value, enum_cls):
            return value
        if isinstance(value, str):
            try:
                return enum_cls[value]
            except KeyError:
                return enum_cls(value)
        return enum_cls(value)

    head_type = _coerce_enum(config_dict["type"], predefined_heads.HeadType)
    output_type = _coerce_enum(config_dict["output_type"], dna_output.OutputType)
    name = config_dict["name"]
    num_tracks = int(config_dict["num_tracks"])

    if head_type == predefined_heads.HeadType.GENOME_TRACKS:
        bundle = _coerce_enum(config_dict["bundle"], bundles.BundleName)
        return predefined_heads.GenomeTracksHeadConfig(
            type=head_type,
            name=name,
            output_type=output_type,
            num_tracks=num_tracks,
            resolutions=list(config_dict["resolutions"]),
            apply_squashing=bool(config_dict["apply_squashing"]),
            bundle=bundle,
        )
    if head_type == predefined_heads.HeadType.SPLICE_SITES_JUNCTION:
        return predefined_heads.SpliceSitesJunctionHeadConfig(
            type=head_type,
            name=name,
            output_type=output_type,
            num_tracks=num_tracks,
            embedding_channels=int(config_dict["embedding_channels"]),
            num_tissues=int(config_dict["num_tissues"]),
        )
    return predefined_heads.HeadConfig(
        type=head_type,
        name=name,
        output_type=output_type,
        num_tracks=num_tracks,
    )


def create_predefined_head_from_config(
    config: PredefinedHeadConfig,
    metadata: Mapping,
) -> predefined_heads.Head:
    """Instantiate a predefined head from an explicit config."""
    return predefined_heads.create_head(config, metadata)


def create_predefined_head(
    head_name: HeadNameLike,
    metadata: Mapping,
    *,
    num_organisms: int = _NUM_ORGANISMS,
) -> predefined_heads.Head:
    """Instantiate a predefined head using a registered config."""
    head = create_registered_head(
        head_name,
        metadata=metadata,
        num_organisms=num_organisms,
    )
    if not isinstance(head, predefined_heads.Head):
        raise TypeError(
            f"Head '{head_name}' resolved to a custom head, not a predefined head."
        )
    return head
