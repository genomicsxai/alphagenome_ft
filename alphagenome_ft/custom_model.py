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
import json
import os
from pathlib import Path
from typing import Any

import haiku as hk
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PyTree
import orbax.checkpoint as ocp

# Optional import for bfloat16 matrix multiplication patch
try:
    import jax_bfloat_mm_patch  # noqa: F401
except ImportError:
    pass

# Optional imports for plotting
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False
    plt = None
    mpatches = None
    np = None

# Optional import for logomaker
try:
    import logomaker
    _HAS_LOGOMAKER = True
except ImportError:
    _HAS_LOGOMAKER = False
    logomaker = None

from alphagenome.models import dna_output, dna_client
from dataclasses import replace

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
        head_configs: dict[str, Any] | None = None,
    ):
        """Initialize the custom model.
        
        Args:
            base_model: Original AlphaGenomeModel to wrap.
            params: Model parameters (potentially with custom heads).
            state: Model state.
            custom_forward_fn: Optional custom forward function.
            custom_heads_list: List of custom head names in this model.
            head_configs: Dictionary mapping head names to head configs for loss computation.
        """
        # Copy attributes from base model
        self._device_context = base_model._device_context
        self._metadata = base_model._metadata
        self._one_hot_encoder = base_model._one_hot_encoder
        self._fasta_extractors = base_model._fasta_extractors
        self._output_metadata_by_organism = base_model._output_metadata_by_organism
        self._variant_scorers = base_model._variant_scorers
        self._head_configs = head_configs or {}
        
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
                *, negative_strand_mask, strand_reindexing, rng=None
            ):
                # Get raw predictions from custom forward function
                # Haiku transforms require RNG as third argument: apply(params, state, rng, *args)
                if rng is None:
                    rng = jax.random.PRNGKey(0)  # Dummy RNG if not provided
                
                    result = custom_forward_fn(params, state, rng, sequences, organism_indices)
                
                # Custom forward function may return (predictions_dict, embeddings) tuple or just predictions
                if isinstance(result, tuple):
                    raw_predictions, _ = result
                else:
                    raw_predictions = result
                
                # Separate standard predictions and custom head predictions
                # raw_predictions should be a dict
                if isinstance(raw_predictions, dict):
                    standard_predictions = {k: v for k, v in raw_predictions.items() 
                                           if k not in custom_heads_set and k != 'embeddings_1bp'}
                    custom_head_predictions = {k: v for k, v in raw_predictions.items() 
                                              if k in custom_heads_set}
                else:
                    # If not a dict, assume all are custom head predictions
                    standard_predictions = {}
                    custom_head_predictions = raw_predictions if isinstance(raw_predictions, dict) else {}
                
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
            self._custom_forward_fn = None  # No custom forward function
        
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
    
    def get_head_config(self, head_name: str) -> dict:
        """Get the head configuration for a given head name.
        
        Args:
            head_name: Name of the head.
            
        Returns:
            Head configuration dict.
            
        Raises:
            KeyError: If head_name not found.
        """
        if head_name not in self._head_configs:
            raise KeyError(
                f"Head '{head_name}' not found in model. "
                f"Available heads: {list(self._head_configs.keys())}"
            )
        return self._head_configs[head_name]
    
    def create_loss_fn_for_head(self, head_name: str):
        """Create a loss function for a custom head.
        
        This creates a function that can compute loss by instantiating the head
        within a transform context. Use this in your training loop.
        
        Args:
            head_name: Name of the custom head.
            
        Returns:
            A function(predictions, batch) -> loss_dict that computes the loss.
        """
        # Verify head exists
        _ = self.get_head_config(head_name)
        
        # Create a transformed function
        def loss_computation(predictions_and_batch):
            """Compute loss within transform."""
            predictions, batch = predictions_and_batch
            head = custom_heads_module.create_custom_head(
                head_name,
                metadata=None,  # Use head's config metadata, not organism metadata
                num_organisms=len(self._metadata)
            )
            return head.loss(predictions, batch)
        
        # Transform without apply_rng since we don't need randomness
        transformed = hk.without_apply_rng(hk.transform(
            lambda p_and_b: loss_computation(p_and_b)
        ))
        
        def loss_fn(predictions, batch):
            """Compute loss for predictions and batch."""
            # Transform expects a single argument, so pack them
            loss_dict = transformed.apply({}, (predictions, batch))
            return loss_dict
        
        return loss_fn
    
    # ========================================================================
    # Checkpoint Save/Load Methods
    # ========================================================================
    
    def save_checkpoint(
        self,
        checkpoint_dir: str | Path,
        *,
        save_full_model: bool = False,
        save_minimal_model: bool = False,
    ) -> None:
        """Save custom head parameters and configuration.
        
        By default, only saves the custom head parameters (efficient for finetuning).
        Optionally can save the full model including backbone parameters, or a minimal
        model with only encoder + custom heads (skips transformer/decoder).
        
        Args:
            checkpoint_dir: Directory to save checkpoint files.
            save_full_model: If True, saves all parameters including backbone.
                If False (default), only saves custom head parameters.
            save_minimal_model: If True, saves encoder + custom head parameters only
                (skips transformer and decoder to save disk space). Only works when
                use_encoder_output=True was used to create the model.
                
        Example:
            ```python
            # After training
            model.save_checkpoint('checkpoints/my_model')
            
            # Save minimal model (encoder + heads only)
            model.save_checkpoint('checkpoints/my_model', save_minimal_model=True)
            
            # Later, load the checkpoint
            model = load_checkpoint(
                'checkpoints/my_model',
                base_model_version='all_folds'
            )
            ```
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate save options
        if save_full_model and save_minimal_model:
            raise ValueError("Cannot set both save_full_model=True and save_minimal_model=True")
        
        # Save metadata about the checkpoint
        metadata = {
            'custom_heads': self._custom_heads,
            'head_configs': {
                name: {
                    'type': config.type.name,
                    'name': config.name,
                    'output_type': config.output_type.name,
                    'num_tracks': config.num_tracks,
                    'metadata': config.metadata,
                }
                for name, config in self._head_configs.items()
            },
            'save_full_model': save_full_model,
            'save_minimal_model': save_minimal_model,
            'use_encoder_output': hasattr(self, '_custom_forward_fn') and self._custom_forward_fn is not None,
        }
        
        with open(checkpoint_dir / 'config.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Determine what parameters to save
        if save_full_model:
            # Save all parameters
            params_to_save = self._params
            state_to_save = self._state
        elif save_minimal_model:
            # Save only encoder + custom head parameters (skip transformer/decoder)
            params_to_save = {}
            state_to_save = {}
            
            # Helper function to recursively extract parameters matching a condition
            def extract_matching(params_dict, condition_fn, result_dict, current_path=""):
                """Recursively extract parameters matching a condition."""
                if isinstance(params_dict, dict):
                    for key, value in params_dict.items():
                        key_str = str(key)
                        full_path = f"{current_path}/{key_str}" if current_path else key_str
                        
                        if condition_fn(full_path, key_str):
                            # This parameter matches - add it to result
                            if current_path:
                                # Need to reconstruct nested path
                                parts = current_path.split('/')
                                target = result_dict
                                for part in parts:
                                    if part and part not in target:
                                        target[part] = {}
                                    if part:
                                        target = target[part]
                                target[key] = value
                            else:
                                result_dict[key] = value
                        elif isinstance(value, dict):
                            # Recurse into nested dict
                            new_path = full_path
                            extract_matching(value, condition_fn, result_dict, new_path)
            
            # Condition: keep encoder parameters (sequence_encoder) but not transformer/decoder
            def is_encoder_or_custom_head(path_str, key_str):
                # Check for encoder
                if 'sequence_encoder' in path_str and 'transformer_tower' not in path_str and 'sequence_decoder' not in path_str:
                    return True
                # Check for custom heads
                for head_name in self._custom_heads:
                    if head_name in path_str or head_name in key_str:
                        return True
                return False
            
            # Extract parameters
            extract_matching(self._params, is_encoder_or_custom_head, params_to_save)
            if self._state:
                extract_matching(self._state, is_encoder_or_custom_head, state_to_save)
            
            # Also handle the common nested structure explicitly
            if 'alphagenome' in self._params and isinstance(self._params['alphagenome'], dict):
                if 'sequence_encoder' in self._params['alphagenome']:
                    if 'alphagenome' not in params_to_save:
                        params_to_save['alphagenome'] = {}
                    params_to_save['alphagenome']['sequence_encoder'] = \
                        self._params['alphagenome']['sequence_encoder']
                
                # Extract custom heads
                if 'head' in self._params['alphagenome']:
                    for head_name in self._custom_heads:
                        if head_name in self._params['alphagenome']['head']:
                            if 'alphagenome' not in params_to_save:
                                params_to_save['alphagenome'] = {}
                            if 'head' not in params_to_save['alphagenome']:
                                params_to_save['alphagenome']['head'] = {}
                            params_to_save['alphagenome']['head'][head_name] = \
                                self._params['alphagenome']['head'][head_name]
            
            # Same for state
            if self._state and 'alphagenome' in self._state and isinstance(self._state['alphagenome'], dict):
                if 'sequence_encoder' in self._state['alphagenome']:
                    if 'alphagenome' not in state_to_save:
                        state_to_save['alphagenome'] = {}
                    state_to_save['alphagenome']['sequence_encoder'] = \
                        self._state['alphagenome']['sequence_encoder']
                
                if 'head' in self._state['alphagenome']:
                    for head_name in self._custom_heads:
                        if head_name in self._state['alphagenome']['head']:
                            if 'alphagenome' not in state_to_save:
                                state_to_save['alphagenome'] = {}
                            if 'head' not in state_to_save['alphagenome']:
                                state_to_save['alphagenome']['head'] = {}
                            state_to_save['alphagenome']['head'][head_name] = \
                                self._state['alphagenome']['head'][head_name]
        else:
            # Only save custom head parameters
            params_to_save = {}
            state_to_save = {}
            
            # Extract head parameters - check all possible parameter structures
            # Structure 1: Flat keys like 'head/mpra_head/...' (use_encoder_output=True mode)
            # This happens when custom heads are created with hk.name_scope('head') outside alphagenome scope
            if isinstance(self._params, dict):
                for key, value in self._params.items():
                    if isinstance(key, str):
                        # Check if this key belongs to any of our custom heads
                        for head_name in self._custom_heads:
                            if key.startswith(f'head/{head_name}/') or key == f'head/{head_name}':
                                params_to_save[key] = value
            
            # Structure 2: alphagenome/head (encoder-only mode, nested)
            if 'alphagenome/head' in self._params:
                for head_name in self._custom_heads:
                    if head_name in self._params['alphagenome/head']:
                        if 'alphagenome/head' not in params_to_save:
                            params_to_save['alphagenome/head'] = {}
                        params_to_save['alphagenome/head'][head_name] = \
                            self._params['alphagenome/head'][head_name]
            
            # Structure 3: alphagenome -> head (standard mode, nested)
            if 'alphagenome' in self._params and 'head' in self._params['alphagenome']:
                for head_name in self._custom_heads:
                    if head_name in self._params['alphagenome']['head']:
                        if 'alphagenome' not in params_to_save:
                            params_to_save['alphagenome'] = {}
                        if 'head' not in params_to_save['alphagenome']:
                            params_to_save['alphagenome']['head'] = {}
                        params_to_save['alphagenome']['head'][head_name] = \
                            self._params['alphagenome']['head'][head_name]
            
            # Extract head state if it exists (check all structures)
            # Structure 1: Flat keys
            if isinstance(self._state, dict):
                for key, value in self._state.items():
                    if isinstance(key, str):
                        for head_name in self._custom_heads:
                            if key.startswith(f'head/{head_name}/') or key == f'head/{head_name}':
                                state_to_save[key] = value
            
            # Structure 2: alphagenome/head
            if 'alphagenome/head' in self._state:
                for head_name in self._custom_heads:
                    if head_name in self._state['alphagenome/head']:
                        if 'alphagenome/head' not in state_to_save:
                            state_to_save['alphagenome/head'] = {}
                        state_to_save['alphagenome/head'][head_name] = \
                            self._state['alphagenome/head'][head_name]
            
            # Structure 3: alphagenome -> head
            if 'alphagenome' in self._state and 'head' in self._state['alphagenome']:
                for head_name in self._custom_heads:
                    if head_name in self._state['alphagenome']['head']:
                        if 'alphagenome' not in state_to_save:
                            state_to_save['alphagenome'] = {}
                        if 'head' not in state_to_save['alphagenome']:
                            state_to_save['alphagenome']['head'] = {}
                        state_to_save['alphagenome']['head'][head_name] = \
                            self._state['alphagenome']['head'][head_name]
        
        # Save parameters using orbax
        # Remove existing checkpoint directory if it exists (to allow overwriting)
        checkpoint_path = checkpoint_dir / 'checkpoint'
        if checkpoint_path.exists():
            import shutil
            shutil.rmtree(checkpoint_path)
        
        checkpointer = ocp.StandardCheckpointer()
        checkpointer.save(
            str(checkpoint_path),
            (params_to_save, state_to_save)
        )
        # Wait for async save to complete
        checkpointer.wait_until_finished()
        
        print(f"✓ Checkpoint saved to {checkpoint_dir}")
        if save_full_model:
            print("  Saved: Full model (encoder + transformer + decoder + custom heads)")
        elif save_minimal_model:
            print("  Saved: Minimal model (encoder + custom heads only, no transformer/decoder)")
        else:
            print(f"  Saved: Custom heads only {self._custom_heads}")
    
    def get_head_parameters(self, head_name: str) -> PyTree:
        """Extract parameters for a specific head.
        
        Args:
            head_name: Name of the head.
            
        Returns:
            Parameter tree for the head.
        """
        if head_name not in self._custom_heads:
            raise KeyError(f"Head '{head_name}' not found in custom heads")
        
        # Search through all parameter keys to find the head
        # Haiku can create different path structures depending on the transform
        def find_head_params(params_dict, target_head_name, path_prefix=''):
            """Recursively search for head parameters in nested or flat dict structure."""
            if not isinstance(params_dict, dict):
                return None
            
            # For nested dicts: check if this level contains the head name directly
            if target_head_name in params_dict:
                return params_dict[target_head_name]
            
            # For flat dicts with '/' keys: look for keys containing the head name
            # e.g., 'alphagenome/head/test_mpra_head/...'
            head_key_pattern = f'head/{target_head_name}'
            matching_keys = {k: v for k, v in params_dict.items() 
                            if isinstance(k, str) and head_key_pattern in k}
            if matching_keys:
                # Reconstruct nested structure from flat keys
                result = {}
                for flat_key, value in matching_keys.items():
                    # Extract the path after the head name
                    parts = flat_key.split('/')
                    try:
                        head_idx = parts.index(target_head_name)
                        # Build nested dict from parts after head name
                        if head_idx < len(parts) - 1:
                            current = result
                            for part in parts[head_idx+1:-1]:
                                if part not in current:
                                    current[part] = {}
                                current = current[part]
                            current[parts[-1]] = value
                        else:
                            # This is the head itself
                            result = value if not isinstance(value, dict) else {**result, **value}
                    except ValueError:
                        continue
                if result:
                    return result
            
            # Search recursively in all sub-dictionaries
            for key, value in params_dict.items():
                if isinstance(value, dict):
                    new_prefix = f"{path_prefix}/{key}" if path_prefix else key
                    result = find_head_params(value, target_head_name, new_prefix)
                    if result is not None:
                        return result
            
            return None
        
        head_params = find_head_params(self._params, head_name)
        if head_params is not None:
            return head_params
        
        # If still not found, raise error with available keys for debugging
        def get_all_keys(d, prefix=''):
            """Get all keys in nested dict for debugging."""
            keys = []
            if isinstance(d, dict):
                for k, v in d.items():
                    full_key = f"{prefix}/{k}" if prefix else k
                    keys.append(full_key)
                    if isinstance(v, dict):
                        keys.extend(get_all_keys(v, full_key))
            return keys
        
        available_keys = get_all_keys(self._params)
        raise ValueError(
            f"Parameters for head '{head_name}' not found in model. "
            f"Available keys: {available_keys[:10]}..."  # Show first 10 keys
        )
    
    # ========================================================================
    # Delegate to base model for other methods
    # ========================================================================
    
    def predict(self, *args, **kwargs):
        """Predict using the model. Delegates to base model."""
        return self._base_model.predict(*args, **kwargs)
    
    def predict_variant(self, *args, **kwargs):
        """Predict variant effects. Delegates to base model."""
        return self._base_model.predict_variant(*args, **kwargs)
    
    def compute_input_gradients(
        self,
        sequence: Float[Array, 'B S 4'],
        organism_index: Int[Array, 'B'],
        *,
        head_name: str | None = None,
        output_index: int | None = None,
        negative_strand_mask: Array | None = None,
        strand_reindexing: Array | None = None,
        return_predictions: bool = False,
        gradients_x_input: bool = False,
    ) -> Float[Array, 'B S 4'] | tuple[Float[Array, 'B S 4'], Any]:
        """Compute gradients of the model output with respect to the input sequence.
        
        This method computes how sensitive the model's predictions are to changes in
        the input DNA sequence, which is useful for:
        - Understanding which positions in the sequence are most important
        - Visualizing sequence motifs and regulatory elements
        - Interpreting model predictions
        
        Args:
            sequence: Input DNA sequence (one-hot encoded), shape (batch, seq_len, 4).
            organism_index: Organism indices for each batch element, shape (batch,).
            head_name: Name of the custom head to compute gradients for. If None and
                there's only one custom head, uses that head. If None and there are
                multiple heads, raises ValueError.
            output_index: Index of the output track to compute gradients for (for
                multi-track outputs). If None, computes gradients for the mean of all tracks.
            negative_strand_mask: Optional mask for negative strand sequences.
            strand_reindexing: Optional reindexing array for strand handling.
            return_predictions: If True, also return the predictions along with gradients.
            gradients_x_input: If True, multiply gradients by input sequence. This is a
                common attribution method that helps clean up attribution maps by only
                showing gradients where the input is non-zero (i.e., where bases are present).
        
        Returns:
            If return_predictions=False: gradients with shape (batch, seq_len, 4)
            If return_predictions=True: tuple of (gradients, predictions)
        
        Raises:
            ValueError: If head_name is None and there are multiple custom heads.
            ValueError: If head_name is specified but not found in custom heads.
        
        Example:
            ```python
            from alphagenome_ft import load_checkpoint
            from alphagenome_ft import register_custom_head, HeadConfig, HeadType
            from alphagenome.models import dna_output
            import jax.numpy as jnp
            
            # Register the custom head (required before loading checkpoint)
            # The head config should match what was used during training
            register_custom_head(
                'mpra_head',
                EncoderMPRAHead,
                HeadConfig(
                    type=HeadType.GENOME_TRACKS,
                    name='mpra_head',
                    output_type=dna_output.OutputType.RNA_SEQ,
                    num_tracks=1,
                    metadata={}
                )
            )
            
            # Load trained model from checkpoint
            model = load_checkpoint(
                'path/to/checkpoint/directory',
                base_model_version='all_folds'
            )
            
            # One-hot encode your sequence (shape: batch, seq_len, 4)
            sequence = jnp.array([...])  # Your one-hot encoded sequence
            organism_index = jnp.array([0])  # 0 = human, 1 = mouse
            
            # Compute gradients
            gradients = model.compute_input_gradients(
                sequence=sequence,
                organism_index=organism_index,
                head_name='mpra_head'  # Optional if only one head
            )
            
            # Get gradients and predictions
            gradients, predictions = model.compute_input_gradients(
                sequence=sequence,
                organism_index=organism_index,
                head_name='mpra_head',
                return_predictions=True
            )
            ```
        """
        # Validate head_name
        if head_name is None:
            if len(self._custom_heads) == 0:
                raise ValueError(
                    "No custom heads found. Cannot compute gradients without a head."
                )
            elif len(self._custom_heads) == 1:
                head_name = self._custom_heads[0]
            else:
                raise ValueError(
                    f"Multiple custom heads found: {self._custom_heads}. "
                    "Please specify head_name."
                )
        elif head_name not in self._custom_heads:
            raise ValueError(
                f"Head '{head_name}' not found. Available heads: {self._custom_heads}"
            )
        
        # Set default values for optional arguments
        if negative_strand_mask is None:
            negative_strand_mask = jnp.zeros((sequence.shape[0],), dtype=jnp.bool_)
        if strand_reindexing is None:
            # Get strand_reindexing from metadata if available
            if hasattr(self._base_model, '_metadata'):
                # Try to get from first organism's metadata
                first_org = list(self._base_model._metadata.keys())[0]
                strand_reindexing = jax.device_put(
                    self._base_model._metadata[first_org].strand_reindexing
                )
            else:
                # Create empty reindexing array
                strand_reindexing = jnp.array([], dtype=jnp.int32)
        
        # Define function to compute output for gradient computation
        def compute_output(seq):
            """Compute model output for a given sequence batch."""
            # For gradient computation, directly call the forward function
            # This avoids issues with the wrapped predict function
            if hasattr(self, '_custom_forward_fn') and self._custom_forward_fn is not None:
                # Directly call the custom forward function
                # It may return either (predictions, embeddings) tuple or just predictions
                rng_key = jax.random.PRNGKey(0)
                result = self._custom_forward_fn(
                    self._params,
                    self._state,
                    rng_key,
                    seq,
                    organism_index,
                )
                
                # Handle both tuple and single return value
                if isinstance(result, tuple):
                    predictions_dict, _ = result
                else:
                    predictions_dict = result
                
                # Extract output for the specified head
                if isinstance(predictions_dict, dict):
                    output = predictions_dict.get(head_name)
                    available_keys = list(predictions_dict.keys())
                else:
                    output = None
                    available_keys = f"Type: {type(predictions_dict)}"
            else:
                # Use standard predict (full model path)
                predictions = self._predict(
                    self._params,
                    self._state,
                    seq,
                    organism_index,
                    negative_strand_mask=negative_strand_mask,
                    strand_reindexing=strand_reindexing,
                )
                
                # Extract output for the specified head
                if isinstance(predictions, _PredictionsDict):
                    output = predictions._custom.get(head_name)
                    available_keys = list(predictions.keys())
                elif hasattr(predictions, 'get'):
                    output = predictions.get(head_name)
                    available_keys = list(predictions.keys()) if isinstance(predictions, dict) else f"Type: {type(predictions)}"
                else:
                    # Try direct access
                    output = predictions[head_name] if head_name in predictions else None
                    available_keys = list(predictions.keys()) if isinstance(predictions, dict) else f"Type: {type(predictions)}"
            
            if output is None:
                raise ValueError(
                    f"Output for head '{head_name}' not found in predictions. "
                    f"Available keys: {available_keys}"
                )
            
            # Handle multi-track outputs
            if output_index is not None:
                # Select specific output track
                if output.ndim > 1:
                    output = output[..., output_index]
                else:
                    raise ValueError(
                        f"output_index specified but output is 1D. "
                        f"Output shape: {output.shape}"
                    )
            else:
                # Use mean of all tracks if multi-dimensional
                if output.ndim > 1:
                    output = jnp.mean(output, axis=-1)
            
            # Return sum over batch and sequence for gradient computation
            # This gives us gradients for each position in each sequence
            return jnp.sum(output)
        
        # Compute gradients using JAX
        # jax.grad computes gradients with respect to the first argument (sequence)
        grad_fn = jax.grad(compute_output)
        gradients = grad_fn(sequence)
        
        # Optionally compute gradients x input for cleaner attribution
        if gradients_x_input:
            gradients = gradients * sequence
        
        # Get predictions if requested
        if return_predictions:
            # Use the same forward path as gradient computation
            if hasattr(self, '_custom_forward_fn') and self._custom_forward_fn is not None:
                rng_key = jax.random.PRNGKey(0)
                result = self._custom_forward_fn(
                    self._params,
                    self._state,
                    rng_key,
                    sequence,
                    organism_index,
                )
                # Handle both tuple and single return value
                if isinstance(result, tuple):
                    predictions_dict, _ = result
                else:
                    predictions_dict = result
                # Extract predictions for the head
                if isinstance(predictions_dict, dict):
                    predictions = predictions_dict.get(head_name)
                else:
                    predictions = predictions_dict
            else:
                predictions = self._predict(
                    self._params,
                    self._state,
                    sequence,
                    organism_index,
                    negative_strand_mask=negative_strand_mask,
                    strand_reindexing=strand_reindexing,
                )
            return gradients, predictions
        else:
            return gradients
    
    def compute_deepshap_attributions(
        self,
        sequence: Float[Array, 'B S 4'],
        organism_index: Int[Array, 'B'],
        *,
        head_name: str | None = None,
        output_index: int | None = None,
        n_references: int = 20,
        reference_type: str = 'shuffle',
        random_state: int | None = None,
    ) -> Float[Array, 'B S 4']:
        """Compute DeepSHAP-style attributions using reference sequences.
        
        NOTE: This is a simplified approximation of DeepSHAP, not the full algorithm.
        The full DeepSHAP (as in [TangerMEME](https://github.com/jmschrei/tangermeme/)) 
        implements the DeepLIFT rescale rule for non-linear operations and uses 
        hypothetical attributions for categorical data. This implementation computes 
        gradient differences between the original sequence and averaged reference 
        gradients, which provides similar but not identical results to true DeepSHAP.
        
        This method averages gradients across multiple reference sequences, providing
        more stable attributions compared to simple gradients.
        
        Args:
            sequence: Input DNA sequence (one-hot encoded), shape (batch, seq_len, 4).
            organism_index: Organism indices for each batch element, shape (batch,).
            head_name: Name of the custom head to compute attributions for. If None and
                there's only one custom head, uses that head. If None and there are
                multiple heads, raises ValueError.
            output_index: Index of the output track to compute attributions for (for
                multi-track outputs). If None, computes attributions for the mean of all tracks.
            n_references: Number of reference sequences to generate (default: 20).
            reference_type: Type of reference sequences. Options:
                - 'shuffle': Dinucleotide-preserving shuffle (default)
                - 'uniform': Uniform random sequences
                - 'gc_match': GC-matched random sequences
            random_state: Random seed for reproducibility.
        
        Returns:
            Attributions with shape (batch, seq_len, 4).
        """
        import numpy as np
        
        # Validate head_name (same logic as compute_input_gradients)
        if head_name is None:
            if len(self._custom_heads) == 0:
                raise ValueError(
                    "No custom heads found. Cannot compute attributions without a head."
                )
            elif len(self._custom_heads) == 1:
                head_name = self._custom_heads[0]
            else:
                raise ValueError(
                    f"Multiple custom heads found: {self._custom_heads}. "
                    "Please specify head_name."
                )
        elif head_name not in self._custom_heads:
            raise ValueError(
                f"Head '{head_name}' not found. Available heads: {self._custom_heads}"
            )
        
        # Generate reference sequences
        batch_size, seq_len, _ = sequence.shape
        references = []
        
        if random_state is not None:
            np.random.seed(random_state)
        
        seq_np = np.asarray(sequence)
        
        for b in range(batch_size):
            seq_batch = seq_np[b:b+1]  # (1, seq_len, 4)
            batch_refs = []
            
            for _ in range(n_references):
                if reference_type == 'shuffle':
                    # Dinucleotide-preserving shuffle
                    base_map = {0: 'A', 1: 'T', 2: 'C', 3: 'G'}
                    seq_str = ''.join([base_map[np.argmax(seq_batch[0, i])] 
                                      for i in range(seq_len)])
                    
                    # Shuffle by swapping random adjacent dinucleotides
                    seq_list = list(seq_str)
                    for _ in range(seq_len // 2):
                        i = np.random.randint(0, len(seq_list) - 1)
                        seq_list[i], seq_list[i+1] = seq_list[i+1], seq_list[i]
                    ref_str = ''.join(seq_list)
                    
                    # Convert back to one-hot
                    ref_onehot = np.zeros((1, seq_len, 4), dtype=np.float32)
                    for i, base in enumerate(ref_str):
                        if base in base_map:
                            base_idx = list(base_map.keys())[list(base_map.values()).index(base)]
                            ref_onehot[0, i, base_idx] = 1.0
                    
                elif reference_type == 'uniform':
                    # Uniform random sequence
                    ref_onehot = np.random.multinomial(1, [0.25, 0.25, 0.25, 0.25], 
                                                      size=(1, seq_len)).astype(np.float32)
                elif reference_type == 'gc_match':
                    # GC-matched random sequence
                    gc_content = np.sum(seq_batch[0, :, [2, 3]]) / seq_len  # C + G
                    ref_onehot = np.zeros((1, seq_len, 4), dtype=np.float32)
                    for i in range(seq_len):
                        if np.random.random() < gc_content:
                            ref_onehot[0, i, np.random.choice([2, 3])] = 1.0
                        else:
                            ref_onehot[0, i, np.random.choice([0, 1])] = 1.0
                else:
                    raise ValueError(f"Unknown reference_type: {reference_type}")
                
                batch_refs.append(ref_onehot)
            
            references.append(np.concatenate(batch_refs, axis=0))  # (n_refs, seq_len, 4)
        
        # Set default values for optional arguments (same as compute_input_gradients)
        negative_strand_mask = jnp.zeros((sequence.shape[0],), dtype=jnp.bool_)
        if hasattr(self._base_model, '_metadata'):
            # Try to get from first organism's metadata
            first_org = list(self._base_model._metadata.keys())[0]
            strand_reindexing = jax.device_put(
                self._base_model._metadata[first_org].strand_reindexing
            )
        else:
            # Create empty reindexing array
            strand_reindexing = jnp.array([], dtype=jnp.int32)
        
        # Compute gradients for each reference and average
        all_attributions = []
        
        for b in range(batch_size):
            orig_seq = sequence[b:b+1]  # (1, seq_len, 4)
            ref_seqs = jnp.array(references[b])  # (n_refs, seq_len, 4)
            
            # Compute gradient function
            def compute_output(seq):
                """Compute model output for gradient computation."""
                if hasattr(self, '_custom_forward_fn') and self._custom_forward_fn is not None:
                    rng_key = jax.random.PRNGKey(0)
                    result = self._custom_forward_fn(
                        self._params,
                        self._state,
                        rng_key,
                        seq,
                        organism_index[b:b+1],
                    )
                    if isinstance(result, tuple):
                        predictions_dict, _ = result
                    else:
                        predictions_dict = result
                    if isinstance(predictions_dict, dict):
                        output = predictions_dict.get(head_name)
                    else:
                        output = predictions_dict
                else:
                    predictions = self._predict(
                        self._params,
                        self._state,
                        seq,
                        organism_index[b:b+1],
                        negative_strand_mask=negative_strand_mask[b:b+1],
                        strand_reindexing=strand_reindexing,
                    )
                    if isinstance(predictions, _PredictionsDict):
                        output = predictions._custom.get(head_name)
                    elif hasattr(predictions, 'get'):
                        output = predictions.get(head_name)
                    else:
                        output = predictions[head_name] if head_name in predictions else None
                
                if output is None:
                    raise ValueError(f"Output for head '{head_name}' not found")
                
                # Handle multi-track outputs (same logic as compute_input_gradients)
                if output_index is not None:
                    # Select specific output track
                    if output.ndim > 1:
                        output = output[..., output_index]
                    else:
                        raise ValueError(
                            f"output_index specified but output is 1D. "
                            f"Output shape: {output.shape}"
                        )
                else:
                    # Use mean of all tracks if multi-dimensional
                    if output.ndim > 1:
                        output = jnp.mean(output, axis=-1)
                
                return jnp.sum(output)
            
            grad_fn = jax.grad(compute_output)
            
            # Compute gradient for original
            orig_grad = grad_fn(orig_seq)  # Shape: (1, seq_len, 4)
            orig_grad = orig_grad.squeeze(0)  # Remove batch dimension: (seq_len, 4)
            
            # Compute gradients for all references and average
            ref_grads = []
            for r in range(n_references):
                ref_seq = ref_seqs[r:r+1]
                ref_grad = grad_fn(ref_seq)  # Shape: (1, seq_len, 4)
                ref_grad = ref_grad.squeeze(0)  # Remove batch dimension: (seq_len, 4)
                ref_grads.append(ref_grad)
            
            ref_grads = jnp.stack(ref_grads)  # (n_refs, seq_len, 4)
            ref_grad_mean = jnp.mean(ref_grads, axis=0)  # (seq_len, 4)
            
            # DeepSHAP attribution: difference from reference mean
            attribution = orig_grad - ref_grad_mean  # (seq_len, 4)
            all_attributions.append(attribution)
        
        attributions = jnp.stack(all_attributions)  # (batch, seq_len, 4)
        return attributions
    
    def plot_attribution_map(
        self,
        sequence: Float[Array, 'B S 4'],
        gradients: Float[Array, 'B S 4'],
        *,
        sequence_str: str | None = None,
        batch_idx: int = 0,
        figsize: tuple[int, int] = (20, 4),
        save_path: str | Path | None = None,
        dpi: int = 150,
        show_sequence: bool = False,
        colormap: str = 'RdBu_r',
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> None:
        """Plot an attribution map showing gradients/importance scores across the sequence.
        
        This creates a visualization showing:
        - The DNA sequence (if provided or can be decoded from one-hot)
        - A heatmap of attribution scores (gradients) for each position and base
        - Summary attribution scores per position
        
        Args:
            sequence: Input DNA sequence (one-hot encoded), shape (batch, seq_len, 4).
            gradients: Gradient/attribution scores, shape (batch, seq_len, 4).
            sequence_str: Optional DNA sequence string. If not provided, will be decoded
                from one-hot encoding.
            batch_idx: Index of batch element to plot (default: 0).
            figsize: Figure size (width, height) in inches.
            save_path: Optional path to save the figure. If None, displays the figure.
            dpi: Resolution for saved figure.
            show_sequence: If True, display the DNA sequence below the heatmap.
            colormap: Matplotlib colormap name for the heatmap.
            vmin: Minimum value for colormap scaling. If None, uses data min.
            vmax: Maximum value for colormap scaling. If None, uses data max.
        
        Raises:
            ImportError: If matplotlib is not installed.
        
        Example:
            ```python
            from alphagenome_ft import load_checkpoint
            from src import EncoderMPRAHead
            from alphagenome_ft import register_custom_head, HeadConfig, HeadType
            from alphagenome.models import dna_output
            import jax.numpy as jnp
            
            # Register and load model (see compute_input_gradients example)
            register_custom_head('mpra_head', EncoderMPRAHead, ...)
            model = load_checkpoint('path/to/checkpoint', base_model_version='all_folds')
            
            # Compute gradients with gradients × input for cleaner attribution
            sequence = jnp.array([...])  # (batch, seq_len, 4)
            gradients = model.compute_input_gradients(
                sequence=sequence,
                organism_index=jnp.array([0]),
                head_name='mpra_head',
                gradients_x_input=True
            )
            
            # Plot attribution map
            model.plot_attribution_map(
                sequence=sequence,
                gradients=gradients,
                sequence_str="ATCGATCG...",  # Optional
                save_path='attribution_map.png'
            )
            ```
        """
        if not _HAS_MATPLOTLIB:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib"
            )
        
        # Extract single batch element
        seq_batch = jnp.asarray(sequence)[batch_idx]  # (seq_len, 4)
        grad_batch = jnp.asarray(gradients)[batch_idx]  # (seq_len, 4)
        
        # Convert to numpy for plotting
        seq_np = np.asarray(seq_batch)
        grad_np = np.asarray(grad_batch)
        
        # Decode sequence string if not provided (need seq_len for validation)
        if sequence_str is None:
            base_map = {0: 'A', 1: 'T', 2: 'C', 3: 'G'}
            sequence_str = ''.join([base_map[np.argmax(seq_np[i])] for i in range(seq_np.shape[0])])
        
        seq_len = len(sequence_str)
        expected_seq_len = seq_np.shape[0]
        
        # Ensure grad_np has correct shape (seq_len, 4)
        # Handle case where gradients might have extra dimensions
        if grad_np.ndim > 2:
            # If shape is (1, seq_len, 4) or similar, squeeze out batch dimension
            grad_np = grad_np.squeeze()
        elif grad_np.ndim == 1:
            # If somehow only 1D, this is an error
            raise ValueError(f"Unexpected gradient shape: {grad_np.shape}. Expected (seq_len={expected_seq_len}, 4)")
        
        # Check if dimensions need to be swapped or if shape is wrong
        if grad_np.ndim == 2:
            if grad_np.shape[0] == 4 and grad_np.shape[1] == expected_seq_len:
                # Shape is (4, seq_len), transpose to (seq_len, 4)
                grad_np = grad_np.T
            elif grad_np.shape[0] != expected_seq_len or grad_np.shape[1] != 4:
                # Shape doesn't match expected (seq_len, 4)
                # Try to diagnose the issue
                if grad_np.shape == (1, 4):
                    raise ValueError(
                        f"Gradient has wrong shape (1, 4) - appears to be missing sequence dimension. "
                        f"Expected (seq_len={expected_seq_len}, 4). "
                        f"Original gradients shape: {jnp.asarray(gradients).shape}, "
                        f"batch_idx: {batch_idx}"
                    )
                else:
                    raise ValueError(
                        f"Gradient shape mismatch: grad_np.shape={grad_np.shape}, "
                        f"seq_np.shape={seq_np.shape}, expected (seq_len={expected_seq_len}, 4). "
                        f"Original gradients shape: {jnp.asarray(gradients).shape}"
                    )
        bases = ['A', 'T', 'C', 'G']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
        
        # Plot 1: Heatmap of gradients per base
        ax1 = axes[0]
        im = ax1.imshow(
            grad_np.T,  # Transpose so bases are rows, positions are columns
            aspect='auto',
            cmap=colormap,
            interpolation='nearest',
            vmin=vmin if vmin is not None else grad_np.min(),
            vmax=vmax if vmax is not None else grad_np.max(),
        )
        
        ax1.set_yticks(range(4))
        ax1.set_yticklabels(bases)
        ax1.set_xlabel('Position in Sequence')
        ax1.set_ylabel('Base')
        ax1.set_title('Attribution Map: Gradients per Position and Base')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Attribution Score', rotation=270, labelpad=20)
        
        # Plot 2: Summary attribution per position (sum across bases)
        ax2 = axes[1]
        # Ensure grad_np has correct shape before summing
        if grad_np.shape[0] != seq_len:
            # If first dimension is not seq_len, might need to reshape
            if grad_np.shape[-1] == seq_len and grad_np.shape[-2] == 4:
                # Shape might be (4, seq_len), transpose it
                grad_np = grad_np.T
            else:
                raise ValueError(
                    f"Gradient shape mismatch: grad_np.shape={grad_np.shape}, "
                    f"expected (seq_len={seq_len}, 4)"
                )
        position_scores = np.sum(grad_np, axis=1)  # Sum across bases (axis=1) for each position
        # Ensure position_scores is 1D
        position_scores = position_scores.flatten()
        ax2.plot(range(seq_len), position_scores, 'k-', linewidth=1.5)
        ax2.fill_between(range(seq_len), position_scores, 0, alpha=0.3)
        ax2.set_xlabel('Position in Sequence')
        ax2.set_ylabel('Total Attribution')
        ax2.set_title('Position-wise Attribution (sum across bases)')
        ax2.grid(True, alpha=0.3)
        
        # Add sequence aligned with positions if requested
        if show_sequence:
            # Display each base character aligned with its position
            # Use a small font size and place at bottom of plot
            y_pos = ax2.get_ylim()[0] - (ax2.get_ylim()[1] - ax2.get_ylim()[0]) * 0.1
            for i, base in enumerate(sequence_str):
                ax2.text(
                    i, y_pos,
                    base,
                    ha='center',
                    va='top',
                    fontfamily='monospace',
                    fontsize=6,  # Small font size
                    color='black'
                )
        
        plt.tight_layout()
        
        # Save or show
        if save_path is not None:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Attribution map saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_sequence_logo(
        self,
        sequence: Float[Array, 'B S 4'],
        gradients: Float[Array, 'B S 4'],
        *,
        batch_idx: int = 0,
        figsize: tuple[int, int] = (20, 3),
        save_path: str | Path | None = None,
        dpi: int = 150,
        threshold: float = 0.0,
        logo_type: str = 'information',
        mask_to_sequence: bool = True,
        use_absolute: bool = False,
    ) -> None:
        """Plot a sequence logo-style visualization of attribution scores using logomaker.
        
        This creates a professional sequence logo visualization where the height
        of each base letter is proportional to its attribution score, using the
        logomaker library for publication-quality figures.
        
        Args:
            sequence: Input DNA sequence (one-hot encoded), shape (batch, seq_len, 4).
            gradients: Gradient/attribution scores, shape (batch, seq_len, 4).
            batch_idx: Index of batch element to plot (default: 0).
            figsize: Figure size (width, height) in inches.
            save_path: Optional path to save the figure. If None, displays the figure.
            dpi: Resolution for saved figure.
            threshold: Minimum attribution score to display (default: 0.0).
            logo_type: Type of logo to create. Options: 'information' (bits), 
                'probability' (normalized probabilities), 'weight' (raw scores).
                Default: 'information'.
            mask_to_sequence: If True, multiply attributions by one-hot sequence to only
                show the present nucleotide at each position. This is standard for 
                DeepSHAP/DeepLIFT logos.
                Default: True.
            use_absolute: If True (default), use absolute attribution values for the logo
                heights (standard magnitudes-only view). If False, keep signed
                attributions so positive/negative contributions are preserved (useful
                for DeepSHAP-style signed logos).
        
        Raises:
            ImportError: If matplotlib or logomaker is not installed.
        
        Example:
            ```python
            from alphagenome_ft import load_checkpoint
            from src import EncoderMPRAHead
            from alphagenome_ft import register_custom_head, HeadConfig, HeadType
            from alphagenome.models import dna_output
            import jax.numpy as jnp
            
            # Register and load model (see compute_input_gradients example)
            register_custom_head('mpra_head', EncoderMPRAHead, ...)
            model = load_checkpoint('path/to/checkpoint', base_model_version='all_folds')
            
            # Compute gradients
            sequence = jnp.array([...])  # (batch, seq_len, 4)
            gradients = model.compute_input_gradients(
                sequence=sequence,
                organism_index=jnp.array([0]),
                head_name='mpra_head',
                gradients_x_input=True
            )
            
            # Plot sequence logo
            model.plot_sequence_logo(
                sequence=sequence,
                gradients=gradients,
                save_path='sequence_logo.png'
            )
            
            # For DeepSHAP attributions, use mask_to_sequence=True to show only
            # the present nucleotide (like TangerMEME's hypothetical=False)
            deepshap_attr = model.compute_deepshap_attributions(
                sequence=sequence,
                organism_index=jnp.array([0]),
                head_name='mpra_head'
            )
            model.plot_sequence_logo(
                sequence=sequence,
                gradients=deepshap_attr,
                save_path='deepshap_logo.png',
                mask_to_sequence=True  # Only show present nucleotide
            )
            ```
        """
        if not _HAS_MATPLOTLIB:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib"
            )
        if not _HAS_LOGOMAKER:
            raise ImportError(
                "logomaker is required for sequence logo plotting. Install with: pip install logomaker"
            )
        
        # Extract single batch element
        seq_batch = jnp.asarray(sequence)[batch_idx]  # (seq_len, 4)
        grad_batch = jnp.asarray(gradients)[batch_idx]  # (seq_len, 4)
        
        # Convert to numpy
        seq_np = np.asarray(seq_batch)
        grad_np = np.asarray(grad_batch)
        
        seq_len = grad_np.shape[0]
        
        # Mask to sequence if requested (like TangerMEME's hypothetical=False)
        # This multiplies attributions by the one-hot sequence, showing only
        # the present nucleotide at each position
        if mask_to_sequence:
            grad_np = grad_np * seq_np
        
        # Prepare data for logomaker
        # Convert gradients/attributions to a DataFrame with columns A, T, C, G.
        # For many attribution use-cases, absolute values are standard to show
        # magnitude of importance, but some analyses (e.g. DeepSHAP) benefit from
        # preserving the sign. This is controlled via use_absolute.
        if use_absolute:
            values_np = np.abs(grad_np)
        else:
            values_np = grad_np
        
        # Create DataFrame for logomaker
        import pandas as pd
        logo_df = pd.DataFrame(
            values_np,
            columns=['A', 'T', 'C', 'G']
        )
        
        # Apply threshold
        if threshold > 0:
            logo_df = logo_df.where(logo_df >= threshold, 0)
        
        # Normalize based on logo_type
        # When mask_to_sequence=True, we only have one base per position, so
        # information content normalization would give max (2 bits) for all positions.
        # Instead, use raw attribution values as heights (like TangerMEME does).
        if mask_to_sequence:
            # For masked logos, use raw values directly (no normalization)
            # The height represents the attribution value for the present base
            # This matches TangerMEME's behavior when hypothetical=False
            pass  # logo_df already contains the masked attribution values
        elif logo_type == 'probability':
            # Normalize to probabilities per position
            # Add small epsilon to all positions to avoid division issues
            row_sums = logo_df.sum(axis=1) + 1e-10
            # If a row sums to near zero (all zeros), set uniform probabilities
            zero_rows = row_sums < 1e-8
            if zero_rows.any():
                # Set uniform probabilities for zero rows
                logo_df.loc[zero_rows] = 0.25  # Uniform probability for 4 bases
                row_sums[zero_rows] = 1.0
            logo_df = logo_df.div(row_sums, axis=0)
        elif logo_type == 'information':
            # Convert to information content (bits)
            # First normalize to probabilities
            prob_df = logo_df.div(logo_df.sum(axis=1) + 1e-10, axis=0)
            # Calculate information content: I = 2 + sum(p * log2(p))
            # For DNA, max information is 2 bits (4 bases)
            # Convert to numpy for calculation (pandas doesn't support keepdims)
            prob_np = np.asarray(prob_df)
            entropy = np.sum(prob_np * np.log2(prob_np + 1e-10), axis=1, keepdims=True)
            info_content = prob_np * (2 + entropy)
            # Convert back to DataFrame
            logo_df = pd.DataFrame(info_content, columns=logo_df.columns, index=logo_df.index)
        # else: 'weight' - use raw scores as-is
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create logo using logomaker
        # Try to use Arial, but fall back to common fonts if not available
        font_name = 'Arial'  # Default preference
        try:
            import matplotlib.font_manager as fm
            # Check available fonts
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            # Try fonts in order of preference
            preferred_fonts = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Helvetica']
            for preferred in preferred_fonts:
                if preferred in available_fonts:
                    font_name = preferred
                    break
            else:
                # If none of the preferred fonts are available, use the first sans-serif font found
                sans_fonts = [f for f in available_fonts if 'sans' in f.lower() or 'Sans' in f]
                if sans_fonts:
                    font_name = sans_fonts[0]
                else:
                    # Last resort: use DejaVu Sans (usually available via matplotlib)
                    font_name = 'DejaVu Sans'
        except Exception:
            # If font checking fails, use DejaVu Sans (matplotlib default)
            font_name = 'DejaVu Sans'
        
        logo = logomaker.Logo(
            logo_df,
            ax=ax,
            color_scheme='classic',  # Standard DNA color scheme
            font_name=font_name,
            show_spines=True,
            baseline_width=0.5,
        )
        
        # Customize appearance
        ax.set_xlabel('Position in Sequence', fontsize=12)
        if mask_to_sequence:
            # When masked to sequence, we're using raw attribution values
            ax.set_ylabel('Attribution', fontsize=12)
            ax.set_title('Sequence Logo: DeepSHAP Attributions', fontsize=14)
        elif logo_type == 'information':
            ax.set_ylabel('Information Content (bits)', fontsize=12)
            ax.set_title('Sequence Logo: Attribution Scores (Information Content)', fontsize=14)
        elif logo_type == 'probability':
            ax.set_ylabel('Probability', fontsize=12)
            ax.set_title('Sequence Logo: Attribution Scores (Probability)', fontsize=14)
        else:
            ax.set_ylabel('Attribution Score', fontsize=12)
            ax.set_title('Sequence Logo: Attribution Scores', fontsize=14)
        
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        plt.tight_layout()
        
        # Save or show
        if save_path is not None:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Sequence logo saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def __getattr__(self, name):
        """Delegate any other attribute access to base model."""
        return getattr(self._base_model, name)


def create_model_with_custom_heads(
    model_version: str | dna_model.ModelVersion = 'all_folds',
    *,
    custom_heads: Sequence[str],
    organism_settings: Mapping[dna_model.Organism, Any] | None = None,
    device: jax.Device | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    use_encoder_output: bool = False,
    detach_backbone: bool = False,
    init_seq_len: int | None = None,
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
        checkpoint_path: Optional local checkpoint directory path. If provided,
            the model will be loaded from this path using dna_model.create and
            Kaggle will not be used.
        use_encoder_output: If True, use custom forward pass that provides encoder output
            before transformer. This enables heads to access raw CNN features.
        detach_backbone: If True, stop gradients at the backbone embeddings so
            heads-only training avoids backprop through the backbone.
        
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
    if checkpoint_path is not None:
        print(f"  Using local checkpoint at: {checkpoint_path}")
        base_model = dna_model.create(
            checkpoint_path,
            organism_settings=organism_settings,
            device=device,
        )
    else:
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
    
    def _stop_gradient_embeddings(embeddings):
        if embeddings is None:
            return None
        values = {}
        for field in ('embeddings_1bp', 'embeddings_128bp', 'embeddings_pair', 'encoder_output'):
            if hasattr(embeddings, field):
                value = getattr(embeddings, field)
                values[field] = None if value is None else jax.lax.stop_gradient(value)
        return replace(embeddings, **values)

    if use_encoder_output:
        # Use custom forward pass that captures encoder output
        # This skips transformer/decoder for short sequences that would fail
        from alphagenome_ft.embeddings_extended import ExtendedEmbeddings
        
        @hk.transform_with_state
        def _forward_with_custom_heads(dna_sequence, organism_index):
            """Forward pass with encoder output only (no transformer/decoder)."""
            # Apply mixed precision policies to encoder
            with hk.mixed_precision.push_policy(model_lib.AlphaGenome, policy):
                with hk.mixed_precision.push_policy(model_lib.SequenceEncoder, policy):
                    with hk.name_scope('alphagenome'):
                        num_organisms = len(metadata)
                        
                        # Step 1: Run encoder ONLY
                        trunk, intermediates = model_lib.SequenceEncoder()(dna_sequence)
                        encoder_output = trunk  # Save encoder output
                        
                        # Create extended embeddings with ONLY encoder output
                        # No transformer/decoder for short sequences
                        embeddings = ExtendedEmbeddings(
                            embeddings_1bp=None,  # Not available without decoder
                            embeddings_128bp=None,  # Not available without transformer
                            encoder_output=encoder_output,  # Raw encoder output
                        )
                        if detach_backbone:
                            embeddings = _stop_gradient_embeddings(embeddings)
            
            # Run custom heads (outside alphagenome scope)
            predictions = {}
            num_organisms = len(metadata)
            with hk.name_scope('head'):
                for head_name in custom_heads:
                    head = custom_heads_module.create_custom_head(head_name, metadata=None, num_organisms=num_organisms)
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
            if detach_backbone:
                embeddings = _stop_gradient_embeddings(embeddings)
            
            # Create predictions dict (only custom heads)
            predictions = {}
            
            # Run custom heads
            # Get number of organisms from metadata (should be 2: human and mouse)
            num_organisms = len(metadata)
            with hk.name_scope('head'):
                for head_name in custom_heads:
                    head = custom_heads_module.create_custom_head(head_name, metadata=None, num_organisms=num_organisms)
                    predictions[head_name] = head(embeddings, organism_index)
            
            return predictions, embeddings
    
    # Initialize parameters with dummy data
    # Use init_seq_len if provided (important for flatten pooling with cached embeddings)
    # Otherwise use default full sequence length
    seq_len = init_seq_len if init_seq_len is not None else 2**17
    print(f"Initializing parameters... (seq_len={seq_len})")
    rng = jax.random.PRNGKey(42)
    dummy_seq = jnp.zeros((1, seq_len, 4), dtype=jnp.bfloat16)
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
    def custom_forward(params, state, rng, dna_sequence, organism_index):
        (predictions, _), _ = _forward_with_custom_heads.apply(
            params, state, None, dna_sequence, organism_index
        )
        return predictions
    
    # Store head configs for loss computation
    head_configs = {}
    for head_name in custom_heads:
        head_configs[head_name] = custom_heads_module.get_custom_head_config(head_name)
    
    # Create and return custom model
    custom_model = CustomAlphaGenomeModel(
        base_model,
        merged_params,
        merged_state,
        custom_forward_fn=custom_forward,
        custom_heads_list=list(custom_heads),
        head_configs=head_configs,
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
                head = custom_heads_module.create_custom_head(head_name, metadata=None, num_organisms=num_organisms)
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
    def custom_forward(params, state, rng, dna_sequence, organism_index):
        (predictions, _), _ = _forward_with_added_heads.apply(
            params, state, None, dna_sequence, organism_index
        )
        return predictions
    
    # Store head configs for loss computation
    head_configs = {}
    for head_name in custom_heads:
        head_configs[head_name] = custom_heads_module.get_custom_head_config(head_name)
    
    # Create and return custom model
    custom_model = CustomAlphaGenomeModel(
        base_model,
        merged_params,
        merged_state,
        custom_forward_fn=custom_forward,
        custom_heads_list=list(custom_heads),
        head_configs=head_configs,
    )
    
    print("✓ Custom heads added successfully")
    print(f"  Total parameters: {custom_model.count_parameters():,}")
    
    return custom_model


def load_checkpoint(
    checkpoint_dir: str | Path,
    *,
    base_model_version: str | dna_model.ModelVersion = 'all_folds',
    organism_settings: Mapping[dna_model.Organism, Any] | None = None,
    device: jax.Device | None = None,
    base_checkpoint_path: str | os.PathLike[str] | None = None,
) -> CustomAlphaGenomeModel:
    """Load a saved custom head checkpoint.
    
    This function loads a checkpoint saved with `model.save_checkpoint()`.
    It handles both full model checkpoints and custom-heads-only checkpoints.
    
    Args:
        checkpoint_dir: Directory containing the checkpoint files.
        base_model_version: Base model version to use (only needed for heads-only checkpoints).
        organism_settings: Optional organism settings.
        device: Optional JAX device.
        base_checkpoint_path: Optional local checkpoint directory for the base
            AlphaGenome model. If provided, Kaggle will not be used when
            instantiating the base model.
        
    Returns:
        CustomAlphaGenomeModel with loaded parameters.
        
    Example:
        ```python
        # Load a checkpoint
        model = load_checkpoint(
            'checkpoints/my_model',
            base_model_version='all_folds'
        )
        
        # Continue training or use for inference
        predictions = model.predict(...)
        ```
        
    Raises:
        FileNotFoundError: If checkpoint files are not found.
        ValueError: If checkpoint configuration is invalid.
    """
    checkpoint_dir = Path(checkpoint_dir).resolve()  # Convert to absolute path
    
    # Handle device selection - validate and fallback if needed
    if device is None:
        # Default device selection
        try:
            # Try GPU first
            device = jax.devices('gpu')[0]
        except (IndexError, RuntimeError):
            try:
                # Fallback to TPU
                device = jax.devices('tpu')[0]
            except (IndexError, RuntimeError):
                # Fallback to CPU
                device = jax.devices('cpu')[0]
    else:
        # Validate that the provided device exists
        available_devices = jax.local_devices()
        if device not in available_devices:
            # Device not found - try to find a compatible one
            device_platform = str(device).split(':')[0] if ':' in str(device) else str(device)
            print(f"Warning: Device {device} not found. Available devices: {[str(d) for d in available_devices]}")
            # Try to find a device with the same platform
            matching_devices = [d for d in available_devices if device_platform in str(d)]
            if matching_devices:
                device = matching_devices[0]
                print(f"Using device: {device}")
            else:
                # Fallback to first available device
                device = available_devices[0]
                print(f"Falling back to: {device}")
    
    # Load configuration
    config_path = checkpoint_dir / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    custom_heads = config['custom_heads']
    save_full_model = config.get('save_full_model', False)
    save_minimal_model = config.get('save_minimal_model', False)
    use_encoder_output = config.get('use_encoder_output', False)
    
    # If use_encoder_output is not in config (old checkpoint), try to infer it
    # Encoder-only models have flat 'head/{head_name}/...' keys outside alphagenome scope
    if not use_encoder_output:
        # We'll check the parameter structure after loading to infer this
        # For now, we'll set a flag to check later
        _infer_use_encoder_output = True
    else:
        _infer_use_encoder_output = False
    
    # Re-register custom heads from config
    from . import custom_heads as custom_heads_module
    from alphagenome.models import dna_output
    from .custom_heads import HeadConfig, HeadType
    
    print(f"Loading checkpoint from {checkpoint_dir}")
    print(f"  Custom heads: {custom_heads}")
    if save_minimal_model:
        print(f"  Model type: Minimal (encoder + heads only)")
    elif save_full_model:
        print(f"  Model type: Full model")
    else:
        print(f"  Model type: Heads only")
    
    for head_name, head_config_dict in config['head_configs'].items():
        # Re-register head with checkpoint's metadata to ensure correct configuration
        # This is important because the head might have been registered with different metadata
        # (e.g., empty metadata {} in tmp.py) which would use defaults that don't match the checkpoint
        try:
            # Get the head class from the registry
            head_class = custom_heads_module._CUSTOM_HEAD_REGISTRY[head_name]
        except KeyError:
            raise RuntimeError(
                f"Head '{head_name}' is not registered. "
                f"Please import and register the head class before loading the checkpoint. "
                f"Example:\n"
                f"  from your_module import {head_name.title().replace('_', '')}Head\n"
                f"  register_custom_head('{head_name}', {head_name.title().replace('_', '')}Head, config)"
            )
        
        # Create HeadConfig from checkpoint's saved config
        checkpoint_head_config = HeadConfig(
            type=HeadType[head_config_dict['type']],
            name=head_config_dict['name'],
            output_type=getattr(dna_output.OutputType, head_config_dict['output_type']),
            num_tracks=head_config_dict['num_tracks'],
            metadata=head_config_dict.get('metadata', {})
        )
        
        # Re-register with checkpoint's config (this overwrites any previous registration)
        custom_heads_module.register_custom_head(head_name, head_class, checkpoint_head_config)
        print(f"  Head '{head_name}' re-registered with checkpoint metadata")
    
    # Load checkpoint parameters
    checkpointer = ocp.StandardCheckpointer()
    checkpoint_path = checkpoint_dir / 'checkpoint'
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Convert to absolute path (required by Orbax)
    checkpoint_path = checkpoint_path.resolve()
    
    # Restore checkpoint - Orbax may try to restore to the original device (e.g., GPU)
    # We need to handle device mapping if the original device is not available
    try:
        restore_result = checkpointer.restore(checkpoint_path)
        # Handle case where restore returns a tuple, list, or single value
        if isinstance(restore_result, (tuple, list)) and len(restore_result) == 2:
            loaded_params, loaded_state = restore_result
        elif isinstance(restore_result, (tuple, list)) and len(restore_result) == 1:
            # Some checkpoints might only have params
            loaded_params = restore_result[0]
            loaded_state = {}
        else:
            # Single value (just params)
            loaded_params = restore_result
            loaded_state = {}
    except ValueError as e:
        if "was not found in jax.local_devices()" in str(e):
            # Device mismatch - checkpoint was saved on a different device
            # Orbax stores device info and tries to restore to that device
            # Solution: Use Orbax's restore_args to specify target device
            print(f"  Warning: Checkpoint was saved on a different device. Restoring and moving to {device}...")
            
            # Try using restore_args to specify target device
            try:
                from orbax.checkpoint import type_handlers
                from orbax.checkpoint import args as ocp_args
                
                # Create restore args that specify the target device
                # This tells Orbax to restore arrays to the specified device
                restore_args = ocp_args.CheckpointArgs(
                    restore_type=ocp_args.RestoreType.RESTORE,
                    # Use ArrayRestoreArgs to specify device
                )
                
                # Try restore with explicit device handling
                # If restore_args doesn't work, we'll fall back to manual approach
                restore_result = checkpointer.restore(
                    checkpoint_path,
                    restore_args=restore_args
                )
                if isinstance(restore_result, (tuple, list)) and len(restore_result) == 2:
                    loaded_params, loaded_state = restore_result
                elif isinstance(restore_result, (tuple, list)) and len(restore_result) == 1:
                    loaded_params = restore_result[0]
                    loaded_state = {}
                else:
                    loaded_params = restore_result
                    loaded_state = {}
            except (AttributeError, TypeError, ImportError):
                # restore_args approach not available or doesn't work
                # Fall back to: restore to any device, then move
                available_devices = jax.local_devices()
                restore_device = available_devices[0]
                
                # Use a context manager to temporarily set default device
                # This might help Orbax use the available device
                try:
                    # Try using context manager if available
                    with jax.default_device(restore_device):
                        restore_result = checkpointer.restore(checkpoint_path)
                        if isinstance(restore_result, (tuple, list)) and len(restore_result) == 2:
                            loaded_params, loaded_state = restore_result
                        elif isinstance(restore_result, (tuple, list)) and len(restore_result) == 1:
                            loaded_params = restore_result[0]
                            loaded_state = {}
                        else:
                            loaded_params = restore_result
                            loaded_state = {}
                except (AttributeError, TypeError):
                    # Context manager not available, try setting default device directly
                    original_default = None
                    try:
                        if hasattr(jax, 'default_device'):
                            original_default = jax.default_device()
                            jax.default_device(restore_device)
                        restore_result = checkpointer.restore(checkpoint_path)
                        if isinstance(restore_result, (tuple, list)) and len(restore_result) == 2:
                            loaded_params, loaded_state = restore_result
                        elif isinstance(restore_result, (tuple, list)) and len(restore_result) == 1:
                            loaded_params = restore_result[0]
                            loaded_state = {}
                        else:
                            loaded_params = restore_result
                            loaded_state = {}
                    finally:
                        if original_default is not None and hasattr(jax, 'default_device'):
                            jax.default_device(original_default)
                except ValueError as restore_error:
                    # Still fails - Orbax is enforcing device metadata
                    # Provide helpful error message
                    raise ValueError(
                        f"Could not restore checkpoint: device mismatch. "
                        f"Checkpoint was saved on a device not available. "
                        f"Available devices: {[str(d) for d in jax.local_devices()]}. "
                        f"\n\nSolutions:"
                        f"\n  1) Run on the original device (GPU if checkpoint was saved on GPU)"
                        f"\n  2) Re-save the checkpoint on CPU: Load on original device and save with save_minimal_model=True"
                        f"\n  3) Use the convert_to_minimal_model.py script to convert the checkpoint"
                    ) from restore_error
                
                # Move all arrays to target device
                def move_to_device(x):
                    if isinstance(x, (jnp.ndarray, jax.Array)):
                        return jax.device_put(x, device)
                    return x
                
                loaded_params = jax.tree_util.tree_map(move_to_device, loaded_params)
                if loaded_state:
                    loaded_state = jax.tree_util.tree_map(move_to_device, loaded_state)
            
            print(f"  ✓ Checkpoint restored and moved to {device}")
        else:
            raise
    
    # Infer use_encoder_output from parameter structure if not in config
    if _infer_use_encoder_output:
        # Check if we have flat 'head/{head_name}/...' keys (encoder-only mode)
        # vs nested 'alphagenome/head/{head_name}/...' keys (standard mode)
        def has_flat_head_keys(params_dict):
            """Check if params have flat head keys (encoder-only mode)."""
            if not isinstance(params_dict, dict):
                return False
            for key in params_dict.keys():
                key_str = str(key)
                # Check for flat 'head/{head_name}/...' pattern
                if key_str.startswith('head/') and any(head_name in key_str for head_name in custom_heads):
                    return True
                # Also check nested dicts recursively
                if isinstance(params_dict[key], dict):
                    if has_flat_head_keys(params_dict[key]):
                        return True
            return False
        
        # Check both params and state
        has_flat_params = has_flat_head_keys(loaded_params)
        has_flat_state = has_flat_head_keys(loaded_state) if loaded_state else False
        
        # Also check: if save_minimal_model=True, it's likely encoder-only (minimal models
        # are typically used for encoder-only MPRA heads)
        # But we'll prioritize the parameter structure check
        use_encoder_output = has_flat_params or has_flat_state
        
        if use_encoder_output:
            print(f"  Inferred use_encoder_output=True from parameter structure (flat head keys detected)")
        elif save_minimal_model:
            # For minimal models, if we can't detect from structure, assume encoder-only
            # (minimal models are typically used for encoder-only MPRA heads)
            use_encoder_output = True
            print(f"  Inferred use_encoder_output=True (minimal model checkpoint, likely encoder-only)")
    
    if save_full_model:
        # Full model checkpoint - load base model structure and use saved params.
        # There are two cases:
        #   1) Standard AlphaGenome full model (no encoder-only head): we can use the
        #      base model's existing forward function directly.
        #   2) Encoder-only MPRA full model (created with use_encoder_output=True):
        #      the checkpoint contains parameters for a custom encoder-only transform
        #      (SequenceEncoder + ExtendedEmbeddings + EncoderMPRAHead), not for the
        #      standard AlphaGenome forward. In this case we must recreate the same
        #      encoder-only Haiku transform that was used during training and use it
        #      consistently for inference and gradients.
        print("Loading full model from checkpoint...")
        if base_checkpoint_path is not None:
            print(f"  Using local base checkpoint at: {base_checkpoint_path}")
            base_model = dna_model.create(
                base_checkpoint_path,
                organism_settings=organism_settings,
                device=device,
            )
        else:
            base_model = dna_model.create_from_kaggle(
                base_model_version,
                organism_settings=organism_settings,
                device=device,
            )
        
        if use_encoder_output:
            # Encoder-only MPRA full model: recreate the encoder-only forward used in
            # create_model_with_custom_heads(..., use_encoder_output=True,...).
            from alphagenome_research.model import model as model_lib
            from alphagenome_research.model.metadata import metadata as metadata_lib
            from alphagenome_ft.embeddings_extended import ExtendedEmbeddings
            import jmp
            import haiku as hk
            
            # Load metadata (same as in create_model_with_custom_heads).
            metadata = {}
            for organism in dna_model.Organism:
                metadata[organism] = metadata_lib.load(organism)
                
            # Infer original training sequence length from checkpoint parameters
            # For flatten pooling, check the first hidden layer's input size
            inferred_seq_len = None
            encoder_feature_size = 1536  # Standard encoder output feature size per position
            
            # Try to find hidden_0/w parameter - check both flat string keys and nested structures
            for head_name in custom_heads:
                    param_value = None
                    param_path = None
                    
                    # Method 1: Try direct flat key access (e.g., 'head/mpra_head/~predict/hidden_0/w')
                    flat_key = f'head/{head_name}/~predict/hidden_0/w'
                    if isinstance(loaded_params, dict) and flat_key in loaded_params:
                        param_value = loaded_params[flat_key]
                        param_path = flat_key
                    
                    # Method 2: Try nested access (e.g., loaded_params['head'][head_name]['~predict']['hidden_0']['w'])
                    if param_value is None and isinstance(loaded_params, dict):
                        try:
                            if 'head' in loaded_params:
                                head_dict = loaded_params['head']
                                if isinstance(head_dict, dict) and head_name in head_dict:
                                    predict_dict = head_dict[head_name]
                                    if isinstance(predict_dict, dict) and '~predict' in predict_dict:
                                        hidden_dict = predict_dict['~predict']
                                        if isinstance(hidden_dict, dict) and 'hidden_0' in hidden_dict:
                                            w_dict = hidden_dict['hidden_0']
                                            if isinstance(w_dict, dict) and 'w' in w_dict:
                                                param_value = w_dict['w']
                                                param_path = f'head/{head_name}/~predict/hidden_0/w (nested)'
                        except (KeyError, TypeError, AttributeError):
                            pass
                    
                    # Method 3: Recursive search for any key containing 'hidden_0' and '/w'
                    if param_value is None:
                        def find_any_hidden_0_w(d, path=""):
                            if isinstance(d, dict):
                                for k, v in d.items():
                                    k_str = str(k)
                                    p = f"{path}/{k_str}" if path else k_str
                                    if 'hidden_0' in k_str and '/w' in k_str:
                                        if isinstance(v, (jnp.ndarray, jax.Array)) and len(v.shape) == 2:
                                            return v, p
                                    if isinstance(v, dict):
                                        result = find_any_hidden_0_w(v, p)
                                        if result[0] is not None:
                                            return result
                            return None, None
                        
                        param_value, param_path = find_any_hidden_0_w(loaded_params)
                    
                    if param_value is not None and isinstance(param_value, (jnp.ndarray, jax.Array)) and len(param_value.shape) == 2:
                        # Found the weight matrix
                        input_size = param_value.shape[0]
                        # For flatten pooling: input_size = num_positions * encoder_feature_size
                        # Check if this looks like flatten pooling (divisible by encoder_feature_size)
                        if input_size % encoder_feature_size == 0:
                            num_positions = input_size // encoder_feature_size
                            # Encoder has 128bp resolution per position
                            inferred_seq_len = num_positions * 128
                            print(f"  Inferred original training sequence length: {inferred_seq_len}bp "
                                  f"(from {num_positions} encoder positions, input_size={input_size}, path={param_path})")
                            break

            # Mixed precision policy (match training).
            policy = jmp.get_policy('params=float32,compute=bfloat16,output=bfloat16')
            hk.mixed_precision.set_policy(model_lib.AlphaGenome, policy)
            
            # Store inferred_seq_len in a way that will be captured by closure
            # Calculate expected positions once here to ensure it's captured correctly
            # Use a sentinel value of -1 to indicate "no padding needed" (JAX-friendly)
            if inferred_seq_len is not None and inferred_seq_len > 0:
                _expected_positions = (inferred_seq_len + 127) // 128  # Ceiling division
            else:
                _expected_positions = -1  # Use -1 as sentinel (JAX-friendly, not None)

            @hk.transform_with_state
            def _forward_with_custom_heads(dna_sequence, organism_index):
                    """Encoder-only forward: SequenceEncoder -> ExtendedEmbeddings -> custom heads."""
                    with hk.mixed_precision.push_policy(model_lib.AlphaGenome, policy):
                        with hk.mixed_precision.push_policy(model_lib.SequenceEncoder, policy):
                            with hk.name_scope('alphagenome'):
                                # Run encoder ONLY (no transformer/decoder).
                                trunk, intermediates = model_lib.SequenceEncoder()(dna_sequence)
                                encoder_output = trunk  # Shape: (batch, num_positions, encoder_feature_size)
                                
                                # Pad or truncate encoder_output to match expected size (for flatten pooling)
                                # Note: Encoder outputs positions at 128bp resolution (S//128)
                                # CRITICAL: Always pad/truncate to match checkpoint's expected size for flatten pooling
                                # This ensures the head sees the correct input size (num_positions * 1536)
                                if _expected_positions > 0:
                                    current_pos = encoder_output.shape[1]
                                    batch_size = encoder_output.shape[0]
                                    feature_size = encoder_output.shape[2]
                                    
                                    # Convert to JAX ints for comparison
                                    current_pos_jax = jnp.int32(current_pos)
                                    expected_pos_jax = jnp.int32(_expected_positions)
                                    
                                    # Always pad to exactly _expected_positions
                                    # Use dynamic_update_slice - it correctly handles smaller source arrays
                                    def pad_impl():
                                        # Create a zero array of the exact target size
                                        padded_output = jnp.zeros((batch_size, _expected_positions, feature_size), dtype=encoder_output.dtype)
                                        
                                        # Insert encoder_output at the beginning
                                        # dynamic_update_slice will copy as much as fits, leaving the rest as zeros
                                        padded_output = jax.lax.dynamic_update_slice(
                                            padded_output,
                                            encoder_output,
                                            (0, 0, 0)  # Start indices for (batch, position, feature)
                                        )
                                        
                                        return padded_output
                                    
                                    def truncate_impl():
                                        return jax.lax.dynamic_slice_in_dim(
                                            encoder_output, 0, _expected_positions, axis=1
                                        )
                                    
                                    def no_change_impl():
                                        return encoder_output
                                    
                                    encoder_output = jax.lax.cond(
                                        current_pos_jax < expected_pos_jax,
                                        pad_impl,
                                        lambda: jax.lax.cond(
                                            current_pos_jax > expected_pos_jax,
                                            truncate_impl,
                                            no_change_impl
                                        )
                                    )
                                # Wrap into ExtendedEmbeddings with encoder_output only.
                                embeddings = ExtendedEmbeddings(
                                    embeddings_1bp=None,
                                    embeddings_128bp=None,
                                    encoder_output=encoder_output,
                                )
                    
                    # Run custom heads outside the alphagenome scope.
                    predictions = {}
                    num_organisms = len(metadata)
                    with hk.name_scope('head'):
                        for head_name in custom_heads:
                            head = custom_heads_module.create_custom_head(
                                head_name, metadata=None, num_organisms=num_organisms
                            )
                            predictions[head_name] = head(embeddings, organism_index)

                    return predictions, embeddings

            # JIT-compiled wrapper that matches the interface expected by CustomAlphaGenomeModel.
            @jax.jit
            def custom_forward(params, state, rng, dna_sequence, organism_index):
                (predictions, _), _ = _forward_with_custom_heads.apply(
                    params, state, None, dna_sequence, organism_index
                )
                return predictions

            custom_forward_fn = custom_forward

            # Create custom model with encoder-only forward and loaded params/state.
            custom_model = CustomAlphaGenomeModel(
                base_model,
                loaded_params,
                loaded_state,
                custom_forward_fn=custom_forward_fn,
                custom_heads_list=custom_heads,
                head_configs={
                    name: custom_heads_module.get_custom_head_config(name)
                    for name in custom_heads
                },
            )
        else:
            # Standard full model: use base model's existing forward function.
            custom_forward_fn = None
            custom_model = CustomAlphaGenomeModel(
                base_model,
                loaded_params,
                loaded_state,
                custom_forward_fn=custom_forward_fn,
                custom_heads_list=custom_heads,
                head_configs={
                    name: custom_heads_module.get_custom_head_config(name)
                    for name in custom_heads
                },
            )
    elif save_minimal_model:
        # Minimal model checkpoint - encoder + custom heads only
        # Need to load base model and merge encoder + head params
        print("Loading minimal model from checkpoint (encoder + custom heads only)...")
        if base_checkpoint_path is not None:
            print(f"  Using local base checkpoint at: {base_checkpoint_path}")
            base_model = dna_model.create(
                base_checkpoint_path,
                organism_settings=organism_settings,
                device=device,
            )
        else:
            base_model = dna_model.create_from_kaggle(
                base_model_version,
                organism_settings=organism_settings,
                device=device,
            )
        
        # Merge minimal params (encoder + heads) with base model params
        # Keep transformer/decoder from base model, use saved encoder + heads
        def merge_minimal_params(base_params, minimal_params):
            """Merge minimal params (encoder + heads) into base params."""
            merged = jax.tree_util.tree_map(lambda x: x, base_params)  # Deep copy
            
            # Helper to set nested value
            def set_nested(d, path_parts, value):
                current = d
                for part in path_parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[path_parts[-1]] = value
            
            # Merge encoder parameters
            if 'alphagenome' in minimal_params and 'sequence_encoder' in minimal_params['alphagenome']:
                if 'alphagenome' not in merged:
                    merged['alphagenome'] = {}
                merged['alphagenome']['sequence_encoder'] = minimal_params['alphagenome']['sequence_encoder']
            
            # Merge custom head parameters
            if 'alphagenome' in minimal_params and 'head' in minimal_params['alphagenome']:
                if 'alphagenome' not in merged:
                    merged['alphagenome'] = {}
                if 'head' not in merged['alphagenome']:
                    merged['alphagenome']['head'] = {}
                for head_name in minimal_params['alphagenome']['head']:
                    merged['alphagenome']['head'][head_name] = minimal_params['alphagenome']['head'][head_name]
            
            # Also handle flat structure
            for key, value in minimal_params.items():
                key_str = str(key)
                if 'sequence_encoder' in key_str or any(head_name in key_str for head_name in custom_heads):
                    merged[key] = value
            
            return merged
        
        merged_params = merge_minimal_params(base_model._params, loaded_params)
        merged_state = merge_minimal_params(base_model._state, loaded_state) if loaded_state else base_model._state
        
        # Check if this is an encoder-only model (use_encoder_output=True)
        # If so, we must recreate the encoder-only forward function
        if use_encoder_output:
            # Encoder-only MPRA minimal model: recreate the encoder-only forward used in
            # create_model_with_custom_heads(..., use_encoder_output=True,...).
            from alphagenome_research.model import model as model_lib
            from alphagenome_research.model.metadata import metadata as metadata_lib
            from alphagenome_ft.embeddings_extended import ExtendedEmbeddings
            import jmp
            import haiku as hk

            # Load metadata (same as in create_model_with_custom_heads).
            metadata = {}
            for organism in dna_model.Organism:
                metadata[organism] = metadata_lib.load(organism)

            # Infer original training sequence length from checkpoint parameters
            # For flatten pooling, check the first hidden layer's input size
            inferred_seq_len = None
            encoder_feature_size = 1536  # Standard encoder output feature size per position
            
            # Try to find hidden_0/w parameter using JAX tree utilities to flatten and search
            for head_name in custom_heads:
                param_value = None
                param_path = None
                
                # Flatten the parameter tree to get all (path, value) pairs
                def flatten_with_paths(tree, path=()):
                    """Flatten tree and return (path_tuple, value) pairs."""
                    if isinstance(tree, dict):
                        for k, v in tree.items():
                            yield from flatten_with_paths(v, path + (k,))
                    else:
                        yield (path, tree)
                
                # Search through all flattened parameters
                for params_dict in [merged_params, loaded_params]:
                    for path_tuple, value in flatten_with_paths(params_dict):
                        # Convert path tuple to string
                        path_str = '/'.join(str(p) for p in path_tuple)
                        # Check if this is hidden_0/w
                        if 'hidden_0' in path_str and path_str.endswith('/w'):
                            if isinstance(value, (jnp.ndarray, jax.Array)) and len(value.shape) == 2:
                                param_value = value
                                param_path = path_str
                                break
                    if param_value is not None:
                        break
                
                if param_value is not None and isinstance(param_value, (jnp.ndarray, jax.Array)) and len(param_value.shape) == 2:
                    # Found the weight matrix
                    input_size = param_value.shape[0]
                    # For flatten pooling: input_size = num_positions * encoder_feature_size
                    # Check if this looks like flatten pooling (divisible by encoder_feature_size)
                    if input_size % encoder_feature_size == 0:
                        num_positions = input_size // encoder_feature_size
                        # Encoder has 128bp resolution per position
                        inferred_seq_len = num_positions * 128
                        print(f"  Inferred original training sequence length: {inferred_seq_len}bp "
                              f"(from {num_positions} encoder positions, input_size={input_size}, path={param_path})")
                        break

            # Mixed precision policy (match training).
            policy = jmp.get_policy('params=float32,compute=bfloat16,output=bfloat16')
            hk.mixed_precision.set_policy(model_lib.AlphaGenome, policy)
            
            # Store inferred_seq_len in a way that will be captured by closure
            # Calculate expected positions once here to ensure it's captured correctly
            # Use a sentinel value of -1 to indicate "no padding needed" (JAX-friendly)
            if inferred_seq_len is not None and inferred_seq_len > 0:
                _expected_positions = (inferred_seq_len + 127) // 128  # Ceiling division
            else:
                _expected_positions = -1  # Use -1 as sentinel (JAX-friendly, not None)

            @hk.transform_with_state
            def _forward_with_custom_heads(dna_sequence, organism_index):
                    """Encoder-only forward: SequenceEncoder -> ExtendedEmbeddings -> custom heads."""
                    with hk.mixed_precision.push_policy(model_lib.AlphaGenome, policy):
                        with hk.mixed_precision.push_policy(model_lib.SequenceEncoder, policy):
                            with hk.name_scope('alphagenome'):
                                # Run encoder ONLY (no transformer/decoder).
                                trunk, intermediates = model_lib.SequenceEncoder()(dna_sequence)
                                encoder_output = trunk  # Shape: (batch, num_positions, encoder_feature_size)
                                
                                # Pad or truncate encoder_output to match expected size (for flatten pooling)
                                # Note: Encoder outputs positions at 128bp resolution (S//128)
                                # CRITICAL: Always pad/truncate to match checkpoint's expected size for flatten pooling
                                # This ensures the head sees the correct input size (num_positions * 1536)
                                if _expected_positions > 0:
                                    current_pos = encoder_output.shape[1]
                                    batch_size = encoder_output.shape[0]
                                    feature_size = encoder_output.shape[2]
                                    
                                    # Convert to JAX ints for comparison
                                    current_pos_jax = jnp.int32(current_pos)
                                    expected_pos_jax = jnp.int32(_expected_positions)
                                    
                                    # Always pad to exactly _expected_positions
                                    # Use dynamic_update_slice - it correctly handles smaller source arrays
                                    def pad_impl():
                                        # Create a zero array of the exact target size
                                        padded_output = jnp.zeros((batch_size, _expected_positions, feature_size), dtype=encoder_output.dtype)
                                        
                                        # Insert encoder_output at the beginning
                                        # dynamic_update_slice will copy as much as fits, leaving the rest as zeros
                                        padded_output = jax.lax.dynamic_update_slice(
                                            padded_output,
                                            encoder_output,
                                            (0, 0, 0)  # Start indices for (batch, position, feature)
                                        )
                                        
                                        return padded_output
                                    
                                    def truncate_impl():
                                        return jax.lax.dynamic_slice_in_dim(
                                            encoder_output, 0, _expected_positions, axis=1
                                        )
                                    
                                    def no_change_impl():
                                        return encoder_output
                                    
                                    encoder_output = jax.lax.cond(
                                        current_pos_jax < expected_pos_jax,
                                        pad_impl,
                                        lambda: jax.lax.cond(
                                            current_pos_jax > expected_pos_jax,
                                            truncate_impl,
                                            no_change_impl
                                        )
                                    )
                                # Wrap into ExtendedEmbeddings with encoder_output only.
                                embeddings = ExtendedEmbeddings(
                                    embeddings_1bp=None,
                                    embeddings_128bp=None,
                                    encoder_output=encoder_output,
                                )
                    
                    # Run custom heads outside the alphagenome scope.
                    predictions = {}
                    num_organisms = len(metadata)
                    with hk.name_scope('head'):
                        for head_name in custom_heads:
                            head = custom_heads_module.create_custom_head(
                                head_name, metadata=None, num_organisms=num_organisms
                            )
                            predictions[head_name] = head(embeddings, organism_index)

                    return predictions, embeddings

            # JIT-compiled wrapper that matches the interface expected by CustomAlphaGenomeModel.
            @jax.jit
            def custom_forward(params, state, rng, dna_sequence, organism_index):
                (predictions, _), _ = _forward_with_custom_heads.apply(
                    params, state, None, dna_sequence, organism_index
                )
                return predictions

            custom_forward_fn = custom_forward
        else:
            # Standard minimal model: use base model's existing forward function.
            custom_forward_fn = None
        
        # Create custom model with merged parameters
        custom_model = CustomAlphaGenomeModel(
            base_model,
            merged_params,
            merged_state,
            custom_forward_fn=custom_forward_fn,
            custom_heads_list=custom_heads,
            head_configs={
                name: custom_heads_module.get_custom_head_config(name)
                for name in custom_heads
            },
        )
        if use_encoder_output:
            print("✓ Minimal model loaded (encoder + custom heads, encoder-only forward)")
        else:
            print("✓ Minimal model loaded (encoder + custom heads, transformer/decoder from base model)")
    else:
        # Heads-only checkpoint - need to merge with base model
        print(f"Loading base model '{base_model_version}'...")
        
        # Determine if we need encoder output
        # Check if any head name suggests encoder-only mode (simple heuristic)
        use_encoder_output = any('encoder' in name.lower() for name in custom_heads)
        
        # Create model with custom heads
        custom_model = create_model_with_custom_heads(
            base_model_version,
            custom_heads=custom_heads,
            organism_settings=organism_settings,
            device=device,
            checkpoint_path=base_checkpoint_path,
            use_encoder_output=use_encoder_output,
        )
        
        # Merge loaded head parameters into model
        def merge_head_params(model_params: PyTree, loaded_head_params: PyTree) -> PyTree:
            """Merge loaded head parameters into model parameters."""
            import copy
            merged = copy.deepcopy(model_params)
            
            # Structure 1: Flat keys like 'head/{head_name}/...' (use_encoder_output=True mode)
            # This happens when custom heads are created with hk.name_scope('head') outside alphagenome scope
            if isinstance(loaded_head_params, dict):
                # Check if we have flat keys starting with 'head/'
                head_keys = {k: v for k, v in loaded_head_params.items() 
                            if isinstance(k, str) and k.startswith('head/')}
                if head_keys:
                    # Merge flat keys directly
                    for key, value in head_keys.items():
                        merged[key] = value
            
            # Structure 2: alphagenome/head (encoder-only mode, nested)
            if 'alphagenome/head' in loaded_head_params:
                if 'alphagenome/head' not in merged:
                    merged['alphagenome/head'] = {}
                
                for head_name, head_params in loaded_head_params['alphagenome/head'].items():
                    merged['alphagenome/head'][head_name] = head_params
            
            # Structure 3: alphagenome -> head (standard mode, nested)
            if 'alphagenome' in loaded_head_params:
                if isinstance(loaded_head_params['alphagenome'], dict):
                    if 'head' in loaded_head_params['alphagenome']:
                        if 'alphagenome' not in merged:
                            merged['alphagenome'] = {}
                        if not isinstance(merged['alphagenome'], dict):
                            merged['alphagenome'] = {}
                        if 'head' not in merged['alphagenome']:
                            merged['alphagenome']['head'] = {}
                        
                        for head_name, head_params in loaded_head_params['alphagenome']['head'].items():
                            merged['alphagenome']['head'][head_name] = head_params
            
            return merged
        
        custom_model._params = merge_head_params(custom_model._params, loaded_params)
        custom_model._state = merge_head_params(custom_model._state, loaded_state)
        
        # Re-put on device
        device = custom_model._device_context._device
        custom_model._params = jax.device_put(custom_model._params, device)
        custom_model._state = jax.device_put(custom_model._state, device)
    
    print("✓ Checkpoint loaded successfully")
    print(f"  Total parameters: {custom_model.count_parameters():,}")
    
    return custom_model
