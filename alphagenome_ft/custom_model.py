"""
Custom AlphaGenome model wrapper with finetuning support.

Provides an extended model class that supports custom and predefined heads and parameter freezing.

Most common use case will be create_model_with_heads() which creates a new model with the specified heads only.

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
Head (trainable)
    ├─ Your prediction layers
    └─ Your loss function
    ↓
Predictions + Loss


A key function that makes this approach work is `_forward_with_custom_heads` which gets the embeddings from the
backbone and runs the requested heads on them. It's a Haiku transform_with_state function that returns the predictions
and embeddings.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import enum
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

from alphagenome.models import dna_output, dna_client
from dataclasses import dataclass, replace

from alphagenome_research.model import dna_model, model as model_lib, embeddings as embeddings_module
from alphagenome_research.model.metadata import metadata as metadata_lib

from alphagenome_ft import parameter_utils
from alphagenome_ft import custom_heads as custom_heads_module


def _resolve_user_metadata(
    *,
    head_name: str,
    head_config: custom_heads_module.HeadConfigLike,
) -> Mapping[enum.Enum, Any] | None:
    """Return user-provided metadata, validated against num_tracks."""
    metadata = custom_heads_module.get_registered_head_metadata(head_name)
    if metadata is None:
        return None

    if isinstance(metadata, Mapping):
        for organism, meta in metadata.items():
            if meta is None:
                continue
            if len(meta) != head_config.num_tracks:
                raise ValueError(
                    f"Head '{head_name}' metadata has {len(meta)} tracks "
                    f"for {getattr(organism, 'name', organism)}, expected "
                    f"{head_config.num_tracks}."
                )
        return metadata

    if len(metadata) != head_config.num_tracks:
        raise ValueError(
            f"Head '{head_name}' metadata has {len(metadata)} tracks, "
            f"expected {head_config.num_tracks}."
        )
    return {organism: metadata for organism in dna_client.Organism}


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
        """Return all keys (standard OutputType keys and string keys for added heads)."""
        return list(self._standard.keys()) + list(self._custom.keys())

    def get(self, key, default=None):
        if isinstance(key, dna_output.OutputType):
            return self._standard.get(key, default)
        else:
            return self._custom.get(key, default)

    def items(self):
        """Return all items."""
        return list(self._standard.items()) + list(self._custom.items())


@dataclass(frozen=True)
class _HeadConfigEntry:
    source: str  # Literally 'custom' or 'predefined'
    config: Any


class CustomAlphaGenomeModel:
    """Extended AlphaGenome model with custom/predefined head support and parameter freezing.

    This class wraps the original AlphaGenomeModel and adds:
    - Head support (custom or predefined)
    - Parameter freezing/unfreezing methods
    - Parameter inspection utilities

    Usage:
        ```python
        # Define and register custom head
        class MyHead(CustomHead):
            ...

        register_custom_head('my_head', MyHead, config)

        # Create model with custom head
        model = create_model_with_heads(
            'all_folds',
            heads=['my_head'],
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
            params: Model parameters (potentially with additional heads).
            state: Model state.
            custom_forward_fn: Optional custom forward function.
            custom_heads_list: List of head names (custom or predefined) in this model.
            head_configs: Dictionary mapping head names to head configs for loss computation.
        """
        # Copy attributes from base model
        self._device_context = base_model._device_context
        self._metadata = base_model._metadata
        self._one_hot_encoder = base_model._one_hot_encoder
        self._fasta_extractors = base_model._fasta_extractors
        self._output_metadata_by_organism = base_model._output_metadata_by_organism
        self._variant_scorers = base_model._variant_scorers
        self._head_configs = {}
        if head_configs:
            for name, entry in head_configs.items():
                if isinstance(entry, _HeadConfigEntry):
                    self._head_configs[name] = entry
                else:
                    self._head_configs[name] = _HeadConfigEntry(
                        source='custom',
                        config=entry,
                    )

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

            # Capture heads list in closure
            custom_heads_set = set(custom_heads_list or [])

            def wrapped_predict(
                params, state, sequences, organism_indices,
                *, negative_strand_mask, strand_reindexing
            ):
                # Get raw predictions from custom forward function
                raw_predictions = custom_forward_fn(params, state, sequences, organism_indices)

                # Separate standard predictions and added head predictions
                standard_predictions = {k: v for k, v in raw_predictions.items()
                                       if k not in custom_heads_set and k != 'embeddings_1bp'}
                custom_head_predictions = {k: v for k, v in raw_predictions.items()
                                          if k in custom_heads_set}

                # Only process standard predictions if they exist
                # (models with added heads only won't have standard predictions)
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

        # Store requested heads info
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
        entry = self._head_configs[head_name]
        return entry.config

    def create_loss_fn_for_head(self, head_name: str):
        """Create a loss function for a head.

        This creates a function that can compute loss by instantiating the head
        within a transform context. Use this in your training loop.

        Args:
            head_name: Name of the head.

        Returns:
            A function(predictions, batch) -> loss_dict that computes the loss.
        """
        # Verify head exists
        _ = self.get_head_config(head_name)

        # Create a transformed function
        def loss_computation(predictions_and_batch):
            """Compute loss within transform."""
            predictions, batch = predictions_and_batch
            entry = self._head_configs[head_name]
            if entry.source == 'predefined':
                head_config = entry.config
                head_metadata = _resolve_user_metadata(
                    head_name=head_name,
                    head_config=head_config,
                ) or {}
                head = custom_heads_module.create_predefined_head_from_config(
                    head_config,
                    metadata=head_metadata,
                )
                if not isinstance(batch, Mapping):
                    raise TypeError(
                        f"Expected batch mapping for head '{head_name}', got {type(batch)!r}."
                    )
                # Build a minimal DataBatch for GenomeTracksHead loss.
                from alphagenome_research.model import schemas
                from alphagenome_research.io import bundles as bundles_lib
                import jax.numpy as jnp

                if 'targets' not in batch or 'organism_index' not in batch:
                    raise ValueError(
                        f"Batch for head '{head_name}' must include 'targets' "
                        "and 'organism_index'."
                    )
                targets = batch['targets']
                organism_index = batch['organism_index']
                if not hasattr(head_config, 'bundle') or head_config.bundle is None:
                    raise ValueError(
                        f"Predefined head '{head_name}' is missing bundle info."
                    )
                bundle = head_config.bundle
                mask = jnp.ones(
                    (targets.shape[0], 1, targets.shape[-1]), dtype=bool
                )

                data_kwargs: dict[str, Any] = {
                    'organism_index': organism_index,
                }
                if bundle == bundles_lib.BundleName.ATAC:
                    data_kwargs.update(atac=targets, atac_mask=mask)
                elif bundle == bundles_lib.BundleName.RNA_SEQ:
                    data_kwargs.update(rna_seq=targets, rna_seq_mask=mask)
                elif bundle == bundles_lib.BundleName.DNASE:
                    data_kwargs.update(dnase=targets, dnase_mask=mask)
                elif bundle == bundles_lib.BundleName.PROCAP:
                    data_kwargs.update(procap=targets, procap_mask=mask)
                elif bundle == bundles_lib.BundleName.CAGE:
                    data_kwargs.update(cage=targets, cage_mask=mask)
                elif bundle == bundles_lib.BundleName.CHIP_TF:
                    data_kwargs.update(chip_tf=targets, chip_tf_mask=mask)
                elif bundle == bundles_lib.BundleName.CHIP_HISTONE:
                    data_kwargs.update(chip_histone=targets, chip_histone_mask=mask)
                else:
                    raise ValueError(
                        f"Unsupported bundle {bundle!r} for head '{head_name}'."
                    )
                batch = schemas.DataBatch(**data_kwargs)
            else:
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
    ) -> None:
        """Save head parameters and configuration.

        By default, only saves the head parameters (efficient for finetuning).
        Optionally can save the full model including backbone parameters.

        Args:
            checkpoint_dir: Directory to save checkpoint files.
            save_full_model: If True, saves all parameters including backbone.
                If False (default), only saves head parameters.

        Example:
            ```python
            # After training
            model.save_checkpoint('checkpoints/my_model')

            # Later, load the checkpoint
            model = load_checkpoint(
                'checkpoints/my_model',
                base_model_version='all_folds'
            )
            ```
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        def _serialize_value(value):
            if isinstance(value, enum.Enum):
                return value.name
            if isinstance(value, (list, tuple)):
                return [_serialize_value(v) for v in value]
            if isinstance(value, dict):
                return {k: _serialize_value(v) for k, v in value.items()}
            return value

        def _serialize_head_config(config_obj):
            import dataclasses

            if dataclasses.is_dataclass(config_obj):
                return {
                    field.name: _serialize_value(getattr(config_obj, field.name))
                    for field in dataclasses.fields(config_obj)
                }
            if isinstance(config_obj, dict):
                return _serialize_value(config_obj)
            return {'value': _serialize_value(config_obj)}

        # Save metadata about the checkpoint
        metadata = {
            'custom_heads': self._custom_heads,
            'head_configs': {
                name: {
                    'source': entry.source,
                    **_serialize_head_config(entry.config),
                }
                for name, entry in self._head_configs.items()
            },
            'save_full_model': save_full_model,
        }

        with open(checkpoint_dir / 'config.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        # Determine what parameters to save
        if save_full_model:
            # Save all parameters
            params_to_save = self._params
            state_to_save = self._state
        else:
            # Only save head parameters
            params_to_save = {}
            state_to_save = {}

            # Extract head parameters - check all possible parameter structures
            # Structure 1: Flat keys like 'head/mpra_head/...' (use_encoder_output=True mode)
            # This happens when heads are created with hk.name_scope('head') outside alphagenome scope
            if isinstance(self._params, dict):
                for key, value in self._params.items():
                    if isinstance(key, str):
                        # Check if this key belongs to any of our heads
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
            print("  Saved: Full model (backbone + heads)")
        else:
            print(f"  Saved: Heads only {self._custom_heads}")

    def get_head_parameters(self, head_name: str) -> PyTree:
        """Extract parameters for a specific head.

        Args:
            head_name: Name of the head.

        Returns:
            Parameter tree for the head.
        """
        if head_name not in self._custom_heads:
            raise KeyError(f"Head '{head_name}' not found in configured heads")

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

    def __getattr__(self, name):
        """Delegate any other attribute access to base model."""
        return getattr(self._base_model, name)


def create_model_with_heads(
    model_version: str | dna_model.ModelVersion = 'all_folds',
    *,
    heads: Sequence[Any],
    organism_settings: Mapping[dna_model.Organism, Any] | None = None,
    device: jax.Device | None = None,
    use_encoder_output: bool = False,
    detach_backbone: bool = False,
    include_standard_heads: bool = False,
    init_seq_len: int = 2**14,
) -> CustomAlphaGenomeModel:
    """Create an AlphaGenome model with specified heads replacing standard heads.

    This function:
    1. Loads the pretrained AlphaGenome model
    2. Creates a new model structure with the requested heads
    3. Initializes head parameters
    4. Keeps pretrained backbone parameters
    5. Returns a model ready for finetuning

    Args:
        model_version: Model version to load ('all_folds' or ModelVersion enum).
        heads: List of head names (custom or predefined; custom must be registered).
        organism_settings: Optional organism settings.
        device: Optional JAX device.
        use_encoder_output: If True, use custom forward pass that provides encoder output
            before transformer. This enables heads to access raw CNN features.
        detach_backbone: If True, stop gradients at the backbone embeddings so
            heads-only training avoids backprop through the backbone.
        include_standard_heads: If True, compute the standard pretrained heads
            in addition to the requested heads. If False, skip standard heads
            to save compute/memory.

    Returns:
        CustomAlphaGenomeModel with requested heads and pretrained backbone.

    Raises:
        ValueError: If a custom head is not registered or a predefined head is unknown.

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
        model = create_model_with_heads('all_folds', heads=['my_head'])

        # 4. Freeze backbone for finetuning
        model.freeze_except_head('my_head')
        ```
    """
    normalized_heads = [custom_heads_module.normalize_head_name(name) for name in heads]

    # Validate all heads are registered
    for head_name in normalized_heads:
        if not custom_heads_module.is_head_registered(head_name):
            raise ValueError(
                f"Head '{head_name}' not found. "
                f"Available heads: {custom_heads_module.list_registered_heads()}"
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

    # Create forward function with requested heads
    print(f"Initializing heads: {normalized_heads}")

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

            # Run heads (outside alphagenome scope)
            predictions = {}
            num_organisms = len(metadata)
            with hk.name_scope('head'):
                for head_name in normalized_heads:
                    head_config = custom_heads_module.get_registered_head_config(head_name)
                    if custom_heads_module.is_custom_config(head_config):
                        head = custom_heads_module.create_registered_head(
                            head_name,
                            metadata=None,
                            num_organisms=num_organisms,
                        )
                    else:
                        head_metadata = _resolve_user_metadata(
                            head_name=head_name,
                            head_config=head_config,
                        ) or {}
                        head = custom_heads_module.create_predefined_head_from_config(
                            head_config,
                            metadata=head_metadata,
                        )
                    predictions[head_name] = head(embeddings, organism_index)

            return predictions, embeddings
    else:
        # Standard forward pass (no encoder output)
        @hk.transform_with_state
        def _forward_with_custom_heads(dna_sequence, organism_index):
            """Forward pass with requested heads only."""
            # Create AlphaGenome trunk (encoder, transformer, decoder)
            # This will use pretrained params for the backbone
            standard_heads = None if include_standard_heads else ()
            alphagenome = model_lib.AlphaGenome(metadata, heads=standard_heads)

            # Get embeddings from the backbone (without running standard heads)
            # We only need the embeddings, not the standard predictions
            _, embeddings = alphagenome(dna_sequence, organism_index)
            if detach_backbone:
                embeddings = _stop_gradient_embeddings(embeddings)

            # Create predictions dict (only requested heads)
            predictions = {}

            # Run heads
            # Get number of organisms from metadata (should be 2: human and mouse)
            num_organisms = len(metadata)
            with hk.name_scope('head'):
                for head_name in normalized_heads:
                    head_config = custom_heads_module.get_registered_head_config(head_name)
                    if custom_heads_module.is_custom_config(head_config):
                        head = custom_heads_module.create_registered_head(
                            head_name,
                            metadata=None,
                            num_organisms=num_organisms,
                        )
                    else:
                        head_metadata = _resolve_user_metadata(
                            head_name=head_name,
                            head_config=head_config,
                        ) or {}
                        head = custom_heads_module.create_predefined_head_from_config(
                            head_config,
                            metadata=head_metadata,
                        )
                    predictions[head_name] = head(embeddings, organism_index)

            return predictions, embeddings

    # Initialize parameters with dummy data
    print(f"Initializing parameters... (seq_len={init_seq_len})")
    rng = jax.random.PRNGKey(42)
    dummy_seq = jnp.zeros((1, init_seq_len, 4), dtype=jnp.bfloat16)
    dummy_org = jnp.array([0])

    new_params, new_state = _forward_with_custom_heads.init(rng, dummy_seq, dummy_org)
    print("✓ Head parameters initialized")

    # Merge pretrained backbone params with new head params
    print("Merging pretrained backbone with heads...")

    def merge_params(pretrained: PyTree, new_with_custom: PyTree) -> PyTree:
        """Recursively merge pretrained params with new head params."""
        if not isinstance(new_with_custom, dict):
            # Leaf node - prefer pretrained if available
            return pretrained if pretrained is not None else new_with_custom

        merged = {}
        for key in new_with_custom:
            if isinstance(pretrained, dict) and key in pretrained:
                # Key exists in both - recurse
                merged[key] = merge_params(pretrained[key], new_with_custom[key])
            else:
                # New key (head) - use new value
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

    # Store head configs for loss computation
    head_configs = {}
    for head_name in normalized_heads:
        config = custom_heads_module.get_registered_head_config(head_name)
        source = 'custom' if custom_heads_module.is_custom_config(config) else 'predefined'
        head_configs[head_name] = _HeadConfigEntry(
            source=source,
            config=config,
        )

    # Create and return custom model
    custom_model = CustomAlphaGenomeModel(
        base_model,
        merged_params,
        merged_state,
        custom_forward_fn=custom_forward,
        custom_heads_list=list(normalized_heads),
        head_configs=head_configs,
    )

    print("✓ Model created successfully")
    print(f"  Total parameters: {custom_model.count_parameters():,}")
    print(f"  Heads: {list(normalized_heads)}")

    return custom_model


def create_model_with_custom_heads(
    model_version: str | dna_model.ModelVersion = 'all_folds',
    *,
    custom_heads: Sequence[Any],
    organism_settings: Mapping[dna_model.Organism, Any] | None = None,
    device: jax.Device | None = None,
    use_encoder_output: bool = False,
    detach_backbone: bool = False,
    include_standard_heads: bool = False,
    init_seq_len: int = 2**20,
) -> CustomAlphaGenomeModel:
    """Backward-compatible wrapper for create_model_with_heads()."""
    return create_model_with_heads(
        model_version,
        heads=custom_heads,
        organism_settings=organism_settings,
        device=device,
        use_encoder_output=use_encoder_output,
        detach_backbone=detach_backbone,
        include_standard_heads=include_standard_heads,
        init_seq_len=init_seq_len,
    )


def wrap_pretrained_model(
    base_model: dna_model.AlphaGenomeModel,
) -> CustomAlphaGenomeModel:
    """Wrap an existing AlphaGenomeModel to add parameter freezing methods.

    Use this if you want the parameter management methods but don't need additional heads.

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


def add_heads_to_model(
    base_model: dna_model.AlphaGenomeModel,
    heads: Sequence[Any],
) -> CustomAlphaGenomeModel:
    """Add heads to an existing pretrained model, keeping all standard heads.

    This function:
    1. Takes an existing model with standard heads
    2. Initializes new head parameters
    3. Merges them with existing parameters
    4. Returns a model with BOTH standard heads AND added heads

    Args:
        base_model: Existing AlphaGenomeModel with standard heads.
        heads: List of head names to add (custom or predefined; custom must be registered).

    Returns:
        CustomAlphaGenomeModel with both standard and added heads.

    Example:
        ```python
        # Load pretrained model with standard heads
        base_model = dna_model.create_from_kaggle('all_folds')

        # Register custom head
        register_custom_head('my_head', MyHead, config)

        # Add head to model (keeps standard heads)
        model = add_heads_to_model(base_model, heads=['my_head'])

        # Now model has both standard heads AND 'my_head'
        model.freeze_except_head('my_head')  # Freeze everything except custom head
        ```
    """
    normalized_heads = [custom_heads_module.normalize_head_name(name) for name in heads]

    # Validate all heads are registered
    for head_name in normalized_heads:
        if not custom_heads_module.is_head_registered(head_name):
            raise ValueError(
                f"Head '{head_name}' not found. "
                f"Available heads: {custom_heads_module.list_registered_heads()}"
            )

    print(f"Adding heads to model: {list(normalized_heads)}")

    # Get metadata
    metadata = {}
    for organism in dna_model.Organism:
        metadata[organism] = metadata_lib.load(organism)

    # Create forward function that includes BOTH standard heads AND added heads
    import jmp
    policy = jmp.get_policy('params=float32,compute=bfloat16,output=bfloat16')
    hk.mixed_precision.set_policy(model_lib.AlphaGenome, policy)

    @hk.transform_with_state
    def _forward_with_added_heads(dna_sequence, organism_index):
        """Forward pass with added heads appended to standard heads."""
        # Create AlphaGenome with standard heads (will use pretrained params)
        alphagenome = model_lib.AlphaGenome(metadata)

        # Get predictions from standard heads
        predictions, embeddings = alphagenome(dna_sequence, organism_index)

        # Add predictions from added heads
        # Get number of organisms from metadata (should be 2: human and mouse)
        num_organisms = len(metadata)
        with hk.name_scope('head'):
            for head_name in normalized_heads:
                if head_name in predictions:
                    continue
                head_config = custom_heads_module.get_registered_head_config(head_name)
                if custom_heads_module.is_custom_config(head_config):
                    head = custom_heads_module.create_registered_head(
                        head_name,
                        metadata=None,
                        num_organisms=num_organisms,
                    )
                else:
                    head_metadata = _resolve_user_metadata(
                        head_name=head_name,
                        head_config=head_config,
                    ) or {}
                    head = custom_heads_module.create_predefined_head_from_config(
                        head_config,
                        metadata=head_metadata,
                    )
                predictions[head_name] = head(embeddings, organism_index)

        return predictions, embeddings

    # Initialize parameters with dummy data
    print("Initializing head parameters...")
    rng = jax.random.PRNGKey(42)
    dummy_seq = jnp.zeros((1, 2**17, 4), dtype=jnp.bfloat16)
    dummy_org = jnp.array([0])

    new_params, new_state = _forward_with_added_heads.init(rng, dummy_seq, dummy_org)
    print("✓ Head parameters initialized")

    # Merge pretrained parameters with new head parameters
    print("Merging parameters...")

    def merge_params(pretrained: PyTree, new_with_custom: PyTree) -> PyTree:
        """Recursively merge pretrained params with new head params."""
        if not isinstance(new_with_custom, dict):
            # Leaf node - prefer pretrained if available
            return pretrained if pretrained is not None else new_with_custom

        merged = {}
        for key in new_with_custom:
            if isinstance(pretrained, dict) and key in pretrained:
                # Key exists in both - recurse
                merged[key] = merge_params(pretrained[key], new_with_custom[key])
            else:
                # New key (head) - use new value
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

    # Store head configs for loss computation
    head_configs = {}
    for head_name in normalized_heads:
        config = custom_heads_module.get_registered_head_config(head_name)
        source = 'custom' if custom_heads_module.is_custom_config(config) else 'predefined'
        head_configs[head_name] = _HeadConfigEntry(
            source=source,
            config=config,
        )

    # Create and return custom model
    custom_model = CustomAlphaGenomeModel(
        base_model,
        merged_params,
        merged_state,
        custom_forward_fn=custom_forward,
        custom_heads_list=list(normalized_heads),
        head_configs=head_configs,
    )

    print("✓ Heads added successfully")
    print(f"  Total parameters: {custom_model.count_parameters():,}")

    return custom_model


def add_custom_heads_to_model(
    base_model: dna_model.AlphaGenomeModel,
    custom_heads: Sequence[Any],
) -> CustomAlphaGenomeModel:
    """Backward-compatible wrapper for add_heads_to_model()."""
    return add_heads_to_model(base_model, heads=custom_heads)


def load_checkpoint(
    checkpoint_dir: str | Path,
    *,
    base_model_version: str | dna_model.ModelVersion = 'all_folds',
    organism_settings: Mapping[dna_model.Organism, Any] | None = None,
    device: jax.Device | None = None,
) -> CustomAlphaGenomeModel:
    """Load a saved head checkpoint.

    This function loads a checkpoint saved with `model.save_checkpoint()`.
    It handles both full model checkpoints and heads-only checkpoints.

    Args:
        checkpoint_dir: Directory containing the checkpoint files.
        base_model_version: Base model version to use (only needed for heads-only checkpoints).
        organism_settings: Optional organism settings.
        device: Optional JAX device.

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
    checkpoint_dir = Path(checkpoint_dir)

    # Load configuration
    config_path = checkpoint_dir / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    custom_heads = config['custom_heads']
    save_full_model = config['save_full_model']

    # Validate heads from config
    from . import custom_heads as custom_heads_module

    print(f"Loading checkpoint from {checkpoint_dir}")
    print(f"  Heads: {custom_heads}")
    print(f"  Full model: {save_full_model}")

    for head_name, head_config_dict in config['head_configs'].items():
        source = head_config_dict.get('source', 'custom')
        if source == 'predefined':
            config_obj = custom_heads_module.deserialize_predefined_head_config(
                head_config_dict
            )
            custom_heads_module.register_predefined_head(
                head_name, config_obj
            )
            continue
        # Check if custom head is already registered
        if not custom_heads_module.is_custom_head(head_name):
            raise RuntimeError(
                f"Head '{head_name}' is not registered. "
                f"Please import and register the head class before loading the checkpoint. "
                f"Example:\n"
                f"  from your_module import {head_name.title().replace('_', '')}Head\n"
                f"  register_custom_head('{head_name}', {head_name.title().replace('_', '')}Head, config)"
            )

    # Load checkpoint parameters
    checkpointer = ocp.StandardCheckpointer()
    checkpoint_path = checkpoint_dir / 'checkpoint'

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    loaded_params, loaded_state = checkpointer.restore(checkpoint_path)

    if save_full_model:
        # Full model checkpoint - load base model structure and use saved params
        print("Loading full model from checkpoint...")
        base_model = dna_model.create_from_kaggle(
            base_model_version,
            organism_settings=organism_settings,
            device=device,
        )

        # Create custom model with loaded parameters
        custom_model = CustomAlphaGenomeModel(
            base_model,
            loaded_params,
            loaded_state,
            custom_heads_list=custom_heads,
            head_configs={
                name: _HeadConfigEntry(
                    source='custom' if custom_heads_module.is_custom_config(
                        custom_heads_module.get_registered_head_config(name)
                    ) else 'predefined',
                    config=custom_heads_module.get_registered_head_config(name),
                )
                for name in custom_heads
            },
        )
    else:
        # Heads-only checkpoint - need to merge with base model
        print(f"Loading base model '{base_model_version}'...")

        # Determine if we need encoder output
        # Check if any head name suggests encoder-only mode (simple heuristic)
        use_encoder_output = any('encoder' in name.lower() for name in custom_heads)

        # Create model with requested heads
        custom_model = create_model_with_heads(
            base_model_version,
            heads=custom_heads,
            organism_settings=organism_settings,
            device=device,
            use_encoder_output=use_encoder_output,
        )

        # Merge loaded head parameters into model
        def merge_head_params(model_params: PyTree, loaded_head_params: PyTree) -> PyTree:
            """Merge loaded head parameters into model parameters."""
            import copy
            merged = copy.deepcopy(model_params)

            # Structure 1: Flat keys like 'head/{head_name}/...' (use_encoder_output=True mode)
            # This happens when heads are created with hk.name_scope('head') outside alphagenome scope
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
