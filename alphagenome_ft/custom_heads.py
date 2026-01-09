"""
Custom head management for AlphaGenome finetuning.

Provides base classes and utilities for defining and registering custom prediction heads.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
import enum

import haiku as hk
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PyTree
import pandas as pd

from alphagenome.models import dna_output
from alphagenome_research.model import embeddings as embeddings_module

#symbolic names for the head types
class HeadType(enum.Enum):
    """Types of prediction heads."""
    GENOME_TRACKS = 'genome_tracks'
    CONTACT_MAPS = 'contact_maps'
    SPLICE_SITES = 'splice_sites'
    CUSTOM = 'custom'


@dataclass
class HeadConfig:
    """Configuration for a prediction head."""
    type: HeadType
    name: str
    output_type: dna_output.OutputType
    num_tracks: int
    metadata: Mapping | None = None


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
        num_organisms: int,
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
        organism_index: Int[Array, 'B'],
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
        organism_index: Int[Array, 'B'],
        **kwargs,
    ) -> PyTree:
        """Forward pass through the head."""
        return self.predict(embeddings, organism_index, **kwargs)


# Registry for custom heads - register with register_custom_head()
_CUSTOM_HEAD_REGISTRY: dict[str, type[CustomHead]] = {}
_CUSTOM_HEAD_CONFIG_REGISTRY: dict[str, HeadConfig] = {}


def register_custom_head(
    head_name: str,
    head_class: type[CustomHead],
    head_config: HeadConfig,
) -> None:
    """Register a custom head for use in finetuning.
    
    Args:
        head_name: Unique name for the head.
        head_class: Class implementing the CustomHead interface.
        head_config: Configuration for the head.
        
    Example:
        ```python
        register_custom_head(
            'my_mpra',
            MyMPRAHead,
            HeadConfig(
                type=HeadType.GENOME_TRACKS,
                name='my_mpra',
                output_type=dna_output.OutputType.RNA_SEQ,
                num_tracks=1,
            )
        )
        ```
    """
    if head_name in _CUSTOM_HEAD_REGISTRY:
        print(f"Warning: Overwriting existing custom head '{head_name}'")
    
    _CUSTOM_HEAD_REGISTRY[head_name] = head_class
    _CUSTOM_HEAD_CONFIG_REGISTRY[head_name] = head_config


def get_custom_head_config(head_name: str) -> HeadConfig | None:
    """Get configuration for a registered custom head.
    
    Args:
        head_name: Name of the custom head.
        
    Returns:
        HeadConfig if registered, None otherwise.
    """
    return _CUSTOM_HEAD_CONFIG_REGISTRY.get(head_name)


def is_custom_head(head_name: str) -> bool:
    """Check if a head name corresponds to a registered custom head.
    
    Args:
        head_name: Name to check.
        
    Returns:
        True if the head is registered as a custom head.
    """
    return head_name in _CUSTOM_HEAD_REGISTRY


def create_custom_head(
    head_name: str,
    metadata: Mapping | None = None,
    num_organisms: int = 8,  # Default for AlphaGenome
) -> CustomHead:
    """Instantiate a registered custom head.
    
    Args:
        head_name: Name of the registered custom head.
        metadata: Optional metadata to pass to the head.
        num_organisms: Number of organisms supported.
        
    Returns:
        Instantiated custom head.
        
    Raises:
        ValueError: If head_name is not registered.
    """
    if head_name not in _CUSTOM_HEAD_REGISTRY:
        raise ValueError(
            f"Custom head '{head_name}' not registered. "
            f"Available heads: {list(_CUSTOM_HEAD_REGISTRY.keys())}"
        )
    
    head_class = _CUSTOM_HEAD_REGISTRY[head_name]
    config = _CUSTOM_HEAD_CONFIG_REGISTRY[head_name]
    
    return head_class(
        name=config.name,
        output_type=config.output_type,
        num_tracks=config.num_tracks,
        num_organisms=num_organisms,
        metadata=metadata or config.metadata,
    )


def list_custom_heads() -> list[str]:
    """List all registered custom heads.
    
    Returns:
        List of registered custom head names.
    """
    return list(_CUSTOM_HEAD_REGISTRY.keys())

