"""
Custom AlphaGenome model wrapper with finetuning support.

Provides an extended model class that supports custom heads and parameter freezing.

Most common use case will be create_model_with_custom_heads() which creates a new model with custom heads only.

The data flow for this is:

DNA Sequence (one-hot encoded)
    ↓
AlphaGenome Backbone (pretrained, frozen)
    ├─ SequenceEncoder
    ├─ TransformerTower  
    └─ SequenceDecoder
    ↓
Multi-resolution Embeddings
    ├─ embeddings_1bp (high resolution)
    ├─ embeddings_128bp (low resolution)
    └─ embeddings_pair (pairwise)
    ↓
Custom Head (trainable)
    ├─ Your prediction layers
    └─ Your loss function
    ↓
Predictions + Loss


A key function that makes this approach work is `_forward_with_custom_heads` which gets the embeddings from the 
backbone and runs the custom heads on them. It's a Haiku transform_with_state function that returns the predictions 
and embeddings.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from typing import Any

import haiku as hk
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PyTree

# Optional import for bfloat16 matrix multiplication patch
try:
    import jax_bfloat_mm_patch  # noqa: F401
except ImportError:
    pass

from alphagenome.models import dna_output, dna_client
from alphagenome_research.model import dna_model, model as model_lib, embeddings as embeddings_module
from alphagenome_research.model.metadata import metadata as metadata_lib

from alphagenome_ft import parameter_utils
from alphagenome_ft import custom_heads as custom_heads_module


class _PredictionsDict:
    """Wrapper for predictions that allows mixing OutputType enum keys with string keys.
    
    This is needed when add new heads to existing model heads because JAX's tree utilities 
    can't handle dictionaries with mixed key types (OutputType enums vs strings). This 
    class stores them separately but allows dict-like access.
    """
    def __init__(self, standard_predictions, custom_predictions):
        self._standard = standard_predictions
        self._custom = custom_predictions
    
    def __getitem__(self, key):
        # Try standard predictions first (OutputType enum keys)
        if isinstance(key, dna_output.OutputType):
            if key in self._standard:
                return self._standard[key]
            raise KeyError(f"Key {key} not found in standard predictions")
        # Then try custom predictions (string keys)
        if key in self._custom:
            return self._custom[key]
        raise KeyError(f"Key {key} not found in predictions")
    
    def __contains__(self, key):
        if isinstance(key, dna_output.OutputType):
            return key in self._standard
        else:
            return key in self._custom
    
    def keys(self):
        """Return all keys (standard OutputType keys and custom string keys)."""
        return list(self._standard.keys()) + list(self._custom.keys())
    
    def get(self, key, default=None):
        if isinstance(key, dna_output.OutputType):
            return self._standard.get(key, default)
        else:
            return self._custom.get(key, default)
    
    def items(self):
        """Return all items."""
        return list(self._standard.items()) + list(self._custom.items())


class CustomAlphaGenomeModel:
    """Extended AlphaGenome model with custom head and parameter freezing support.
    
    This class wraps the original AlphaGenomeModel and adds:
    - Custom head support
    - Parameter freezing/unfreezing methods
    - Parameter inspection utilities
    
    Usage:
        ```python
        # Define and register custom head
        class MyHead(CustomHead):
            ...
        
        register_custom_head('my_head', MyHead, config)
        
        # Create model with custom head
        model = create_model_with_custom_heads(
            'all_folds',
            custom_heads=['my_head'],
        )
        
        # Freeze everything except custom head
        model.freeze_except_head('my_head')
        
        # Use model for training
        ...
        ```
    """
    
    def __init__(
        self,
        base_model: dna_model.AlphaGenomeModel,
        params: PyTree,
        state: PyTree,
        custom_forward_fn: Any | None = None,
        custom_heads_list: Sequence[str] | None = None,
    ):
        """Initialize the custom model.
        
        Args:
            base_model: Original AlphaGenomeModel to wrap.
            params: Model parameters (potentially with custom heads).
            state: Model state.
            custom_forward_fn: Optional custom forward function.
            custom_heads_list: List of custom head names in this model.
        """
        # Copy attributes from base model
        self._device_context = base_model._device_context
        self._metadata = base_model._metadata
        self._one_hot_encoder = base_model._one_hot_encoder
        self._fasta_extractors = base_model._fasta_extractors
        self._output_metadata_by_organism = base_model._output_metadata_by_organism
        self._variant_scorers = base_model._variant_scorers
        
        # Get the actual device from the device context
        device = self._device_context._device
        
        # Set parameters and state
        self._params = jax.device_put(params, device)
        self._state = jax.device_put(state, device)
        
        # Set forward functions
        if custom_forward_fn is not None:
            # Wrap custom forward function to process predictions like base model
            # This ensures predictions go through extract_predictions() and reverse_complement()
            from alphagenome_research.model.dna_model import extract_predictions
            from alphagenome_research.model import augmentation
            
            # Capture custom_heads_list in closure
            custom_heads_set = set(custom_heads_list or [])
            
            def wrapped_predict(
                params, state, sequences, organism_indices,
                *, negative_strand_mask, strand_reindexing
            ):
                # Get raw predictions from custom forward function
                raw_predictions = custom_forward_fn(params, state, sequences, organism_indices)
                
                # Separate standard predictions and custom head predictions
                standard_predictions = {k: v for k, v in raw_predictions.items() 
                                       if k not in custom_heads_set and k != 'embeddings_1bp'}
                custom_head_predictions = {k: v for k, v in raw_predictions.items() 
                                          if k in custom_heads_set}
                
                # Only process standard predictions if they exist
                # (models with custom heads only won't have standard predictions)
                if standard_predictions:
                    # Process standard predictions through extract_predictions and reverse_complement
                    extracted = extract_predictions(standard_predictions)
                    reversed_preds = augmentation.reverse_complement(
                        extracted,
                        negative_strand_mask,
                        strand_reindexing=strand_reindexing,
                        sequence_length=sequences.shape[1],
                    )
                else:
                    # No standard predictions - return empty dict
                    reversed_preds = {}
                
                # Return a wrapper that allows dict-like access but stores keys separately
                # This avoids JAX tree processing issues with mixed key types
                return _PredictionsDict(reversed_preds, custom_head_predictions)
            
            # Don't jit the wrapper - jit the inner forward function instead
            # The wrapper returns a custom object that JAX can't process as a pytree
            self._predict = wrapped_predict
        else:
            self._predict = base_model._predict
        
        self._predict_variant = base_model._predict_variant
        
        # Store custom heads info
        self._custom_heads = custom_heads_list or []
        
        # Store base model for delegation
        self._base_model = base_model
    
    # ========================================================================
    # Parameter Freezing Methods
    # ========================================================================
    
    def freeze_parameters(
        self,
        freeze_paths: Sequence[str] | None = None,
        freeze_prefixes: Sequence[str] | None = None,
    ) -> None:
        """Freeze specific parameters by path or prefix.
        
        Args:
            freeze_paths: Exact parameter paths to freeze.
            freeze_prefixes: Path prefixes to freeze.
        """
        self._params = parameter_utils.freeze_parameters(
            self._params, freeze_paths, freeze_prefixes
        )
    
    def unfreeze_parameters(
        self,
        unfreeze_paths: Sequence[str] | None = None,
        unfreeze_prefixes: Sequence[str] | None = None,
    ) -> None:
        """Unfreeze specific parameters by path or prefix.
        
        Args:
            unfreeze_paths: Exact parameter paths to unfreeze.
            unfreeze_prefixes: Path prefixes to unfreeze.
        """
        self._params = parameter_utils.unfreeze_parameters(
            self._params, unfreeze_paths, unfreeze_prefixes
        )
    
    def freeze_backbone(self) -> None:
        """Freeze the backbone (encoder, transformer, decoder) but keep heads trainable."""
        self._params = parameter_utils.freeze_backbone(self._params)
    
    def freeze_all_heads(self, except_heads: Sequence[str] | None = None) -> None:
        """Freeze all heads except specified ones.
        
        Args:
            except_heads: Head names to keep trainable.
        """
        self._params = parameter_utils.freeze_all_heads(
            self._params, except_heads=except_heads
        )
    
    def freeze_except_head(self, trainable_head: str) -> None:
        """Freeze everything except a specific head.
        
        Args:
            trainable_head: Name of the head to keep trainable.
        """
        self._params = parameter_utils.freeze_except_head(
            self._params, trainable_head=trainable_head
        )
    
    # ========================================================================
    # Parameter Inspection Methods
    # ========================================================================
    
    def get_parameter_paths(self) -> list[str]:
        """Get all parameter paths in the model.
        
        Returns:
            List of all parameter paths.
        """
        return parameter_utils.get_parameter_paths(self._params)
    
    def get_head_parameter_paths(self) -> list[str]:
        """Get all head parameter paths.
        
        Returns:
            List of head parameter paths.
        """
        return parameter_utils.get_head_parameter_paths(self._params)
    
    def get_backbone_parameter_paths(self) -> list[str]:
        """Get all backbone parameter paths.
        
        Returns:
            List of backbone parameter paths.
        """
        return parameter_utils.get_backbone_parameter_paths(self._params)
    
    def count_parameters(self) -> int:
        """Count total number of parameters.
        
        Returns:
            Total parameter count.
        """
        return parameter_utils.count_parameters(self._params)
    
    # ========================================================================
    # Delegate to base model for other methods
    # ========================================================================
    
    def predict(self, *args, **kwargs):
        """Predict using the model. Delegates to base model."""
        return self._base_model.predict(*args, **kwargs)
    
    def predict_variant(self, *args, **kwargs):
        """Predict variant effects. Delegates to base model."""
        return self._base_model.predict_variant(*args, **kwargs)
    
    def __getattr__(self, name):
        """Delegate any other attribute access to base model."""
        return getattr(self._base_model, name)


def create_model_with_custom_heads(
    model_version: str | dna_model.ModelVersion = 'all_folds',
    *,
    custom_heads: Sequence[str],
    organism_settings: Mapping[dna_model.Organism, Any] | None = None,
    device: jax.Device | None = None,
    use_encoder_output: bool = False,
) -> CustomAlphaGenomeModel:
    """Create an AlphaGenome model with custom heads replacing standard heads.
    
    This function:
    1. Loads the pretrained AlphaGenome model
    2. Creates a new model structure with custom heads
    3. Initializes custom head parameters
    4. Keeps pretrained backbone parameters
    5. Returns a model ready for finetuning
    
    Args:
        model_version: Model version to load ('all_folds' or ModelVersion enum).
        custom_heads: List of custom head names (must be registered).
        organism_settings: Optional organism settings.
        device: Optional JAX device.
        use_encoder_output: If True, use custom forward pass that provides encoder output
            before transformer. This enables heads to access raw CNN features.
        
    Returns:
        CustomAlphaGenomeModel with custom heads and pretrained backbone.
        
    Raises:
        ValueError: If any custom head is not registered.
        
    Example:
        ```python
        from alphagenome_ft import register_custom_head, CustomHead, HeadConfig, HeadType
        from alphagenome.models import dna_output
        
        # 1. Define custom head
        class MyHead(CustomHead):
            def predict(self, embeddings, organism_index, **kwargs):
                x = embeddings.get_sequence_embeddings(resolution=1)
                return {'predictions': hk.Linear(1)(x)}
            
            def loss(self, predictions, batch):
                return {'loss': jnp.array(0.0)}
        
        # 2. Register it
        register_custom_head(
            'my_head',
            MyHead,
            HeadConfig(
                type=HeadType.GENOME_TRACKS,
                name='my_head',
                output_type=dna_output.OutputType.RNA_SEQ,
                num_tracks=1,
            )
        )
        
        # 3. Create model
        model = create_model_with_custom_heads('all_folds', custom_heads=['my_head'])
        
        # 4. Freeze backbone for finetuning
        model.freeze_except_head('my_head')
        ```
    """
    # Validate all custom heads are registered
    for head_name in custom_heads:
        if not custom_heads_module.is_custom_head(head_name):
            raise ValueError(
                f"Custom head '{head_name}' not registered. "
                f"Available: {custom_heads_module.list_custom_heads()}"
            )
    
    # Load pretrained model
    print("Loading pretrained AlphaGenome model...")
    base_model = dna_model.create_from_kaggle(
        model_version,
        organism_settings=organism_settings,
        device=device,
    )
    print("✓ Pretrained model loaded")
    
    # Get metadata
    metadata = {}
    for organism in dna_model.Organism:
        metadata[organism] = metadata_lib.load(organism)
    
    # Create forward function with custom heads
    print(f"Initializing custom heads: {custom_heads}")
    
    # Set mixed precision policy
    import jmp
    policy = jmp.get_policy('params=float32,compute=bfloat16,output=bfloat16')
    hk.mixed_precision.set_policy(model_lib.AlphaGenome, policy)
    
    if use_encoder_output:
        # Use custom forward pass that captures encoder output
        from alphagenome_ft.embeddings_extended import ExtendedEmbeddings
        
        @hk.transform_with_state
        def _forward_with_custom_heads(dna_sequence, organism_index):
            """Forward pass with custom heads and encoder output."""
            # Apply mixed precision policies to all modules
            with hk.mixed_precision.push_policy(model_lib.AlphaGenome, policy):
                with hk.mixed_precision.push_policy(model_lib.SequenceEncoder, policy):
                    with hk.mixed_precision.push_policy(model_lib.TransformerTower, policy):
                        with hk.mixed_precision.push_policy(model_lib.SequenceDecoder, policy):
                            with hk.mixed_precision.push_policy(embeddings_module.OutputEmbedder, policy):
                                with hk.mixed_precision.push_policy(embeddings_module.OutputPair, policy):
                                    with hk.name_scope('alphagenome'):
                                        num_organisms = len(metadata)
                                        
                                        # Step 1: Run encoder
                                        trunk, intermediates = model_lib.SequenceEncoder()(dna_sequence)
                                        encoder_output = trunk  # Save before organism embedding
                                        
                                        # Step 2: Add organism embedding
                                        organism_embedding_trunk = hk.Embed(num_organisms, trunk.shape[-1])(
                                            organism_index
                                        )
                                        trunk += organism_embedding_trunk[:, None, :]
                                        
                                        # Step 3: Run transformer
                                        trunk, pair_activations = model_lib.TransformerTower()(trunk)
                                        
                                        # Step 4: Run decoder
                                        x = model_lib.SequenceDecoder()(trunk, intermediates)
                                        
                                        # Step 5: Create embeddings
                                        embeddings_128bp = embeddings_module.OutputEmbedder(num_organisms)(
                                            trunk, organism_index
                                        )
                                        embeddings_1bp = embeddings_module.OutputEmbedder(num_organisms)(
                                            x, organism_index, embeddings_128bp
                                        )
                                        
                                        # Create extended embeddings with encoder output
                                        embeddings = ExtendedEmbeddings(
                                            embeddings_1bp=embeddings_1bp,
                                            embeddings_128bp=embeddings_128bp,
                                            encoder_output=encoder_output,
                                        )
            
            # Run custom heads (outside alphagenome scope)
            predictions = {}
            num_organisms = len(metadata)
            with hk.name_scope('head'):
                for head_name in custom_heads:
                    head = custom_heads_module.create_custom_head(head_name, metadata, num_organisms=num_organisms)
                    predictions[head_name] = head(embeddings, organism_index)
            
            return predictions, embeddings
    else:
        # Standard forward pass (no encoder output)
        @hk.transform_with_state
        def _forward_with_custom_heads(dna_sequence, organism_index):
            """Forward pass with custom heads only."""
            # Create AlphaGenome trunk (encoder, transformer, decoder)
            # This will use pretrained params for the backbone
            alphagenome = model_lib.AlphaGenome(metadata)
            
            # Get embeddings from the backbone (without running standard heads)
            # We only need the embeddings, not the standard predictions
            _, embeddings = alphagenome(dna_sequence, organism_index)
            
            # Create predictions dict (only custom heads)
            predictions = {}
            
            # Run custom heads
            # Get number of organisms from metadata (should be 2: human and mouse)
            num_organisms = len(metadata)
            with hk.name_scope('head'):
                for head_name in custom_heads:
                    head = custom_heads_module.create_custom_head(head_name, metadata, num_organisms=num_organisms)
                    predictions[head_name] = head(embeddings, organism_index)
            
            return predictions, embeddings
    
    # Initialize parameters with dummy data
    print("Initializing parameters...")
    rng = jax.random.PRNGKey(42)
    dummy_seq = jnp.zeros((1, 2**17, 4), dtype=jnp.bfloat16)
    dummy_org = jnp.array([0])
    
    new_params, new_state = _forward_with_custom_heads.init(rng, dummy_seq, dummy_org)
    print("✓ Custom head parameters initialized")
    
    # Merge pretrained backbone params with new custom head params
    print("Merging pretrained backbone with custom heads...")
    
    def merge_params(pretrained: PyTree, new_with_custom: PyTree) -> PyTree:
        """Recursively merge pretrained params with new custom head params."""
        if not isinstance(new_with_custom, dict):
            # Leaf node - prefer pretrained if available
            return pretrained if pretrained is not None else new_with_custom
        
        merged = {}
        for key in new_with_custom:
            if isinstance(pretrained, dict) and key in pretrained:
                # Key exists in both - recurse
                merged[key] = merge_params(pretrained[key], new_with_custom[key])
            else:
                # New key (custom head) - use new value
                merged[key] = new_with_custom[key]
        
        # Also include any keys only in pretrained (shouldn't happen but be safe)
        if isinstance(pretrained, dict):
            for key in pretrained:
                if key not in merged:
                    merged[key] = pretrained[key]
        
        return merged
    
    merged_params = merge_params(base_model._params, new_params)
    merged_state = merge_params(base_model._state, new_state)
    
    print("✓ Parameters merged")
    
    # Create custom forward function for the model (JIT-compiled for performance and numerical consistency)
    @jax.jit
    def custom_forward(params, state, dna_sequence, organism_index):
        (predictions, _), _ = _forward_with_custom_heads.apply(
            params, state, None, dna_sequence, organism_index
        )
        return predictions
    
    # Create and return custom model
    custom_model = CustomAlphaGenomeModel(
        base_model,
        merged_params,
        merged_state,
        custom_forward_fn=custom_forward,
        custom_heads_list=list(custom_heads),
    )
    
    print("✓ Custom model created successfully")
    print(f"  Total parameters: {custom_model.count_parameters():,}")
    print(f"  Custom heads: {list(custom_heads)}")
    
    return custom_model


def wrap_pretrained_model(
    base_model: dna_model.AlphaGenomeModel,
) -> CustomAlphaGenomeModel:
    """Wrap an existing AlphaGenomeModel to add parameter freezing methods.
    
    Use this if you want the parameter management methods but don't need custom heads.
    
    Args:
        base_model: Existing AlphaGenomeModel instance.
        
    Returns:
        CustomAlphaGenomeModel wrapping the base model.
    """
    return CustomAlphaGenomeModel(
        base_model,
        base_model._params,
        base_model._state,
        custom_forward_fn=None,
        custom_heads_list=None,
    )


def add_custom_heads_to_model(
    base_model: dna_model.AlphaGenomeModel,
    custom_heads: Sequence[str],
) -> CustomAlphaGenomeModel:
    """Add custom heads to an existing pretrained model, keeping all standard heads.
    
    This function:
    1. Takes an existing model with standard heads
    2. Initializes new custom head parameters
    3. Merges them with existing parameters
    4. Returns a model with BOTH standard heads AND custom heads
    
    Args:
        base_model: Existing AlphaGenomeModel with standard heads.
        custom_heads: List of custom head names to add (must be registered).
        
    Returns:
        CustomAlphaGenomeModel with both standard and custom heads.
        
    Example:
        ```python
        # Load pretrained model with standard heads
        base_model = dna_model.create_from_kaggle('all_folds')
        
        # Register custom head
        register_custom_head('my_head', MyHead, config)
        
        # Add custom head to model (keeps standard heads)
        model = add_custom_heads_to_model(base_model, custom_heads=['my_head'])
        
        # Now model has both standard heads AND 'my_head'
        model.freeze_except_head('my_head')  # Freeze everything except custom head
        ```
    """
    # Validate all custom heads are registered
    for head_name in custom_heads:
        if not custom_heads_module.is_custom_head(head_name):
            raise ValueError(
                f"Custom head '{head_name}' not registered. "
                f"Available: {custom_heads_module.list_custom_heads()}"
            )
    
    print(f"Adding custom heads to model: {list(custom_heads)}")
    
    # Get metadata
    metadata = {}
    for organism in dna_model.Organism:
        metadata[organism] = metadata_lib.load(organism)
    
    # Create forward function that includes BOTH standard heads AND custom heads
    import jmp
    policy = jmp.get_policy('params=float32,compute=bfloat16,output=bfloat16')
    hk.mixed_precision.set_policy(model_lib.AlphaGenome, policy)
    
    @hk.transform_with_state
    def _forward_with_added_heads(dna_sequence, organism_index):
        """Forward pass with custom heads added to standard heads."""
        # Create AlphaGenome with standard heads (will use pretrained params)
        alphagenome = model_lib.AlphaGenome(metadata)
        
        # Get predictions from standard heads
        predictions, embeddings = alphagenome(dna_sequence, organism_index)
        
        # Add predictions from custom heads
        # Get number of organisms from metadata (should be 2: human and mouse)
        num_organisms = len(metadata)
        with hk.name_scope('head'):
            for head_name in custom_heads:
                head = custom_heads_module.create_custom_head(head_name, metadata, num_organisms=num_organisms)
                predictions[head_name] = head(embeddings, organism_index)
        
        return predictions, embeddings
    
    # Initialize parameters with dummy data
    print("Initializing custom head parameters...")
    rng = jax.random.PRNGKey(42)
    dummy_seq = jnp.zeros((1, 2**17, 4), dtype=jnp.bfloat16)
    dummy_org = jnp.array([0])
    
    new_params, new_state = _forward_with_added_heads.init(rng, dummy_seq, dummy_org)
    print("✓ Custom head parameters initialized")
    
    # Merge pretrained parameters with new custom head parameters
    print("Merging parameters...")
    
    def merge_params(pretrained: PyTree, new_with_custom: PyTree) -> PyTree:
        """Recursively merge pretrained params with new custom head params."""
        if not isinstance(new_with_custom, dict):
            # Leaf node - prefer pretrained if available
            return pretrained if pretrained is not None else new_with_custom
        
        merged = {}
        for key in new_with_custom:
            if isinstance(pretrained, dict) and key in pretrained:
                # Key exists in both - recurse
                merged[key] = merge_params(pretrained[key], new_with_custom[key])
            else:
                # New key (custom head) - use new value
                merged[key] = new_with_custom[key]
        
        # Also include any keys only in pretrained
        if isinstance(pretrained, dict):
            for key in pretrained:
                if key not in merged:
                    merged[key] = pretrained[key]
        
        return merged
    
    merged_params = merge_params(base_model._params, new_params)
    merged_state = merge_params(base_model._state, new_state)
    
    print("✓ Parameters merged")
    
    # Create custom forward function (JIT-compiled for performance and numerical consistency)
    @jax.jit
    def custom_forward(params, state, dna_sequence, organism_index):
        (predictions, _), _ = _forward_with_added_heads.apply(
            params, state, None, dna_sequence, organism_index
        )
        return predictions
    
    # Create and return custom model
    custom_model = CustomAlphaGenomeModel(
        base_model,
        merged_params,
        merged_state,
        custom_forward_fn=custom_forward,
        custom_heads_list=list(custom_heads),
    )
    
    print("✓ Custom heads added successfully")
    print(f"  Total parameters: {custom_model.count_parameters():,}")
    
    return custom_model

