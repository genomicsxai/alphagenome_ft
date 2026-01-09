"""
Tests for model prediction consistency.

Verifies that wrapped models produce identical predictions to the base model.
"""
import jax
import jax.numpy as jnp
import pytest
from alphagenome.models import dna_output


class TestPredictionConsistency:
    """Test that wrapped models produce identical predictions to base model."""
    
    def test_wrapped_model_predictions_match_base(
        self,
        base_model,
        wrapped_model_with_head,
        test_sequence,
        organism_index,
        test_interval,
        strand_reindexing,
    ):
        """Verify wrapped model produces identical predictions for standard heads."""
        
        # Get predictions from base model
        with base_model._device_context:
            base_predictions = base_model._predict(
                base_model._params,
                base_model._state,
                test_sequence,
                organism_index,
                negative_strand_mask=jnp.array([test_interval.negative_strand]),
                strand_reindexing=jax.device_put(
                    strand_reindexing,
                    base_model._device_context._device
                ),
            )
        
        # Get predictions from wrapped model
        with wrapped_model_with_head._device_context:
            wrapped_predictions = wrapped_model_with_head._predict(
                wrapped_model_with_head._params,
                wrapped_model_with_head._state,
                test_sequence,
                organism_index,
                negative_strand_mask=jnp.array([test_interval.negative_strand]),
                strand_reindexing=jax.device_put(
                    strand_reindexing,
                    wrapped_model_with_head._device_context._device
                ),
            )
        
        # Verify all standard output types match
        all_match = True
        mismatches = []
        
        for output_type in base_predictions.keys():
            if output_type not in wrapped_predictions:
                all_match = False
                mismatches.append(f"{output_type.name}: Not in wrapped predictions")
                continue
            
            base_vals = base_predictions[output_type]
            wrapped_vals = wrapped_predictions[output_type]
            
            # Handle dict format (shouldn't happen for standard heads but just in case)
            if isinstance(base_vals, dict):
                base_vals = base_vals['predictions']
                wrapped_vals = wrapped_vals['predictions']
            
            # Check shapes
            if base_vals.shape != wrapped_vals.shape:
                all_match = False
                mismatches.append(
                    f"{output_type.name}: Shape mismatch "
                    f"{base_vals.shape} vs {wrapped_vals.shape}"
                )
                continue
            
            # Handle NaN values
            base_nan_mask = jnp.isnan(base_vals)
            wrapped_nan_mask = jnp.isnan(wrapped_vals)
            
            # Check NaN patterns match
            if not jnp.all(base_nan_mask == wrapped_nan_mask):
                all_match = False
                mismatches.append(f"{output_type.name}: NaN patterns differ")
                continue
            
            # Compare non-NaN values
            non_nan_mask = ~base_nan_mask
            if jnp.any(non_nan_mask):
                base_non_nan = base_vals[non_nan_mask]
                wrapped_non_nan = wrapped_vals[non_nan_mask]
                
                max_diff = float(jnp.max(jnp.abs(base_non_nan - wrapped_non_nan)))
                are_close = jnp.allclose(
                    base_non_nan,
                    wrapped_non_nan,
                    rtol=1e-5,
                    atol=1e-8
                )
                
                if not are_close:
                    all_match = False
                    mismatches.append(
                        f"{output_type.name}: Values differ (max diff: {max_diff:.2e})"
                    )
        
        # Assert all predictions match
        assert all_match, (
            f"Wrapped model predictions differ from base model:\n" +
            "\n".join(mismatches)
        )
    
    def test_custom_head_output_exists(
        self,
        wrapped_model_with_head,
        test_sequence,
        organism_index,
        test_interval,
        strand_reindexing,
    ):
        """Verify custom head produces output."""
        
        with wrapped_model_with_head._device_context:
            predictions = wrapped_model_with_head._predict(
                wrapped_model_with_head._params,
                wrapped_model_with_head._state,
                test_sequence,
                organism_index,
                negative_strand_mask=jnp.array([test_interval.negative_strand]),
                strand_reindexing=jax.device_put(
                    strand_reindexing,
                    wrapped_model_with_head._device_context._device
                ),
            )
        
        # Check custom head output exists
        assert 'test_mpra_head' in predictions
        
        # Check output structure
        mpra_output = predictions['test_mpra_head']
        assert isinstance(mpra_output, dict)
        assert 'predictions' in mpra_output
        
        # Check output shape
        pred_array = mpra_output['predictions']
        assert pred_array.ndim == 3  # (batch, sequence, tracks)
        assert pred_array.shape[0] == 1  # batch size
        assert pred_array.shape[2] == 1  # num_tracks
    
    def test_custom_only_model_has_correct_structure(
        self,
        custom_only_model,
        base_model,
    ):
        """Verify model with only custom heads has correct parameter structure."""
        
        # Check that backbone parameters exist and match base model
        test_param_keys = [
            'alphagenome/embed/embeddings',
            'alphagenome/sequence_encoder/stem_conv/w',
        ]
        
        for key in test_param_keys:
            # Check key exists
            assert key in custom_only_model._params, f"Missing parameter: {key}"
            assert key in base_model._params, f"Missing parameter in base: {key}"
            
            # Check parameters are identical
            custom_param = custom_only_model._params[key]
            base_param = base_model._params[key]
            
            assert jnp.allclose(custom_param, base_param), (
                f"Parameter {key} differs from base model"
            )
        
        # Check custom head parameters exist
        custom_head_keys = [
            k for k in custom_only_model._params.keys()
            if 'test_mpra_head' in k
        ]
        assert len(custom_head_keys) > 0, "No custom head parameters found"

