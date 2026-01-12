"""
Tests for encoder-only mode (short sequence support).

Verifies that models with use_encoder_output=True work correctly with short sequences.
"""
import jax
import jax.numpy as jnp
import pytest
import haiku as hk
from alphagenome.models import dna_output

from alphagenome_ft import (
    CustomHead,
    HeadConfig,
    HeadType,
    register_custom_head,
    create_model_with_custom_heads,
)


class SimpleEncoderHead(CustomHead):
    """Test head that uses only encoder output."""
    
    def __init__(self, *, name, output_type, num_tracks, num_organisms, metadata):
        super().__init__(
            name=name,
            num_tracks=num_tracks,
            output_type=output_type,
            num_organisms=num_organisms,
            metadata=metadata,
        )
    
    def predict(self, embeddings, organism_index, **kwargs):
        """Predict from encoder output only."""
        if not hasattr(embeddings, 'encoder_output') or embeddings.encoder_output is None:
            raise ValueError("encoder_output not available")
        
        # Use encoder output at 128bp resolution
        x = embeddings.encoder_output  # (batch, seq_len//128, D)
        
        # Simple prediction
        x = hk.Linear(128, name='fc1')(x)
        x = jax.nn.relu(x)
        predictions = hk.Linear(self._num_tracks, name='output')(x)
        
        return predictions  # (batch, seq_len//128, num_tracks)
    
    def loss(self, predictions, batch):
        """Compute loss."""
        targets = batch.get('targets')
        if targets is None:
            return {'loss': jnp.array(0.0)}
        
        # Average predictions over sequence
        pred_values = jnp.mean(predictions, axis=1)
        
        if targets.ndim == 1:
            targets = targets[:, None]
        
        mse_loss = jnp.mean((pred_values - targets) ** 2)
        return {'loss': mse_loss, 'mse': mse_loss}


@pytest.fixture(scope="function")
def encoder_head_registered():
    """Register encoder-only test head."""
    head_name = 'test_encoder_head'
    config = HeadConfig(
        type=HeadType.GENOME_TRACKS,
        name=head_name,
        output_type=dna_output.OutputType.RNA_SEQ,
        num_tracks=1,
    )
    register_custom_head(head_name, SimpleEncoderHead, config)
    return head_name


@pytest.fixture(scope="function")
def encoder_only_model(encoder_head_registered, device):
    """Create model with encoder-only mode."""
    return create_model_with_custom_heads(
        'all_folds',
        custom_heads=[encoder_head_registered],
        device=device,
        use_encoder_output=True,  # NEW: Encoder-only mode
    )


class TestEncoderOnlyMode:
    """Test encoder-only forward pass for short sequences."""
    
    def test_encoder_only_model_creation(self, encoder_only_model):
        """Verify encoder-only model is created successfully."""
        assert encoder_only_model is not None
        assert 'test_encoder_head' in encoder_only_model._custom_heads
        
        # Check parameters exist
        param_count = encoder_only_model.count_parameters()
        assert param_count > 100_000_000, (
            f"Model should have encoder parameters, got {param_count:,}"
        )
    
    def test_short_sequence_inference(self, encoder_only_model):
        """Verify short sequences can be processed (this would fail without encoder-only mode)."""
        # Create a SHORT sequence (300 bp) that would fail in transformer
        short_seq = jnp.zeros((1, 300, 4), dtype=jnp.float32)
        short_seq = short_seq.at[:, :, 0].set(1.0)  # All A's
        
        organism_index = jnp.array([0])
        
        # This should work with encoder-only mode
        with encoder_only_model._device_context:
            predictions = encoder_only_model._predict(
                encoder_only_model._params,
                encoder_only_model._state,
                short_seq,
                organism_index,
                negative_strand_mask=jnp.array([False]),
                strand_reindexing=jax.device_put(
                    encoder_only_model._metadata[encoder_only_model._organism_enum.HOMO_SAPIENS].strand_reindexing,
                    encoder_only_model._device_context._device
                ),
            )
        
        # Check prediction exists
        assert 'test_encoder_head' in predictions
        pred_array = predictions['test_encoder_head']
        
        # Check shape is at 128bp resolution
        # 300bp sequence -> 300//128 = 2 bins (with padding)
        assert pred_array.ndim == 3
        assert pred_array.shape[0] == 1  # batch
        assert pred_array.shape[2] == 1  # num_tracks
        # Sequence dimension should be small (seq_len // 128)
        assert pred_array.shape[1] < 10, (
            f"Expected ~2-3 bins for 300bp sequence, got {pred_array.shape[1]}"
        )
    
    def test_encoder_embeddings_only(self, encoder_only_model, test_sequence, organism_index):
        """Verify only encoder_output is populated, not transformer/decoder embeddings."""
        # Run prediction and capture embeddings
        # Note: We can't directly access embeddings from _predict, so we test indirectly
        # by verifying the head works (which requires encoder_output)
        
        with encoder_only_model._device_context:
            predictions = encoder_only_model._predict(
                encoder_only_model._params,
                encoder_only_model._state,
                test_sequence,
                organism_index,
                negative_strand_mask=jnp.array([False]),
                strand_reindexing=jax.device_put(
                    encoder_only_model._metadata[encoder_only_model._organism_enum.HOMO_SAPIENS].strand_reindexing,
                    encoder_only_model._device_context._device
                ),
            )
        
        # If encoder_output wasn't available, the head would have raised an error
        assert 'test_encoder_head' in predictions
        assert predictions['test_encoder_head'] is not None
    
    def test_parameter_freezing_with_encoder_only(self, encoder_only_model):
        """Verify parameter freezing works in encoder-only mode."""
        # Freeze all except head
        encoder_only_model.freeze_except_head('test_encoder_head')
        
        # Get trainable parameters
        trainable_paths = encoder_only_model.get_trainable_parameter_paths()
        frozen_paths = encoder_only_model.get_frozen_parameter_paths()
        
        # Head should be trainable
        head_params = [p for p in trainable_paths if 'test_encoder_head' in p]
        assert len(head_params) > 0, "Head parameters should be trainable"
        
        # Backbone should be frozen
        backbone_params = [p for p in frozen_paths if 'alphagenome' in p]
        assert len(backbone_params) > 0, "Backbone parameters should be frozen"
