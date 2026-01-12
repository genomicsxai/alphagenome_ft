"""
Template custom heads for AlphaGenome finetuning.

These templates demonstrate how to access different embedding types from AlphaGenome.
All templates use a simple architecture: Linear → Activation → Linear → Output

The key difference is WHICH embeddings they access:
1. StandardHead - Uses 1bp embeddings (decoder output: local + global context)
2. TransformerHead - Uses 128bp embeddings (transformer output: global context only)
3. EncoderOnlyHead - Uses encoder output (CNN features only, for short sequences)

Copy and modify these templates for your specific task.
"""

import jax
import jax.numpy as jnp
import haiku as hk
from alphagenome_ft import CustomHead


class StandardHead(CustomHead):
    """Template head using 1bp resolution embeddings (decoder output).
    
    This is the DEFAULT and most common choice.
    
    Embeddings: 1bp resolution from decoder
    - Shape: (batch, sequence_length, 1536)
    - Contains: Local features (CNN) + Global context (Transformer)
    - Resolution: Base-pair level
    
    Use this when:
    - You need high-resolution (1bp) predictions
    - Your sequences are standard length (> 100kb)
    - Default choice for most tasks
    
    Architecture: Encoder → Transformer → Decoder ✓
    
    Example use case: ChIP-seq peaks, TF binding, variant effects
    """
    
    def predict(self, embeddings, organism_index, **kwargs):
        """Generate predictions from 1bp embeddings.
        
        Args:
            embeddings: Multi-resolution embeddings from AlphaGenome
            organism_index: Organism indices (0=human, 1=mouse)
            
        Returns:
            Predictions with shape (batch, sequence_length, num_tracks)
        """
        # KEY: Access 1bp resolution embeddings (decoder output)
        x = embeddings.get_sequence_embeddings(resolution=1)
        # Shape: (batch, sequence_length, 1536)
        
        # Simple architecture: Linear → ReLU → Linear
        x = hk.Linear(256, name='hidden')(x)
        x = jax.nn.relu(x)
        predictions = hk.Linear(self._num_tracks, name='output')(x)
        
        # Return per-position predictions
        return predictions
    
    def loss(self, predictions, batch):
        """Compute loss for predictions.
        
        Args:
            predictions: Model predictions (batch, sequence_length, num_tracks)
            batch: Batch dictionary with 'targets' key
            
        Returns:
            Dictionary with 'loss' key and optional metrics
        """
        targets = batch.get('targets')
        if targets is None:
            return {'loss': jnp.array(0.0)}
        
        # Mean squared error loss
        mse_loss = jnp.mean((predictions - targets) ** 2)
        
        return {
            'loss': mse_loss,
            'mse': mse_loss,
        }


class TransformerHead(CustomHead):
    """Template head using 128bp resolution embeddings (transformer output).
    
    Embeddings: 128bp resolution from transformer
    - Shape: (batch, sequence_length//128, 3072)
    - Contains: Global context via attention (no decoder)
    - Resolution: 128 base-pair bins
    
    Use this when:
    - You need global context across long sequences
    - Your task operates at lower resolution (100+ bp scale)
    - You want pure attention features without local decoder mixing
    
    Architecture: Encoder → Transformer ✓ (skips Decoder)
    
    Example use case: Gene expression, chromatin state, TAD boundaries
    """
    
    def predict(self, embeddings, organism_index, **kwargs):
        """Generate predictions from 128bp embeddings.
        
        Args:
            embeddings: Multi-resolution embeddings from AlphaGenome
            organism_index: Organism indices
            
        Returns:
            Predictions with shape (batch, sequence_length//128, num_tracks)
        """
        # KEY: Access 128bp resolution embeddings (transformer output)
        x = embeddings.get_sequence_embeddings(resolution=128)
        # Shape: (batch, sequence_length//128, 3072)
        
        # Simple architecture: Linear → ReLU → Linear
        x = hk.Linear(256, name='hidden')(x)
        x = jax.nn.relu(x)
        predictions = hk.Linear(self._num_tracks, name='output')(x)
        
        # Return predictions at 128bp resolution
        return predictions
    
    def loss(self, predictions, batch):
        """Compute loss for predictions."""
        targets = batch.get('targets')
        if targets is None:
            return {'loss': jnp.array(0.0)}
        
        # MSE loss
        mse_loss = jnp.mean((predictions - targets) ** 2)
        
        return {
            'loss': mse_loss,
            'mse': mse_loss,
        }


class EncoderOnlyHead(CustomHead):
    """Template head using encoder output only (CNN features, no transformer).
    
    Embeddings: Encoder output (raw CNN features)
    - Shape: (batch, sequence_length//128, 1536)
    - Contains: Pure CNN features (NO attention, NO decoder)
    - Resolution: 128 base-pair bins
    
    Use this when:
    - Working with SHORT sequences (< 1000 bp)
    - Sequences would fail in transformer (incompatible lengths)
    - You want pure CNN features without global attention
    
    Architecture: Encoder ONLY ✓ (skips Transformer and Decoder)
    
    CRITICAL: Must create model with use_encoder_output=True:
        model = create_model_with_custom_heads(
            'all_folds',
            custom_heads=['my_head'],
            use_encoder_output=True,  # ← Required!
        )
    
    Example use case: MPRA (short sequences), promoter activity, motifs
    """
    
    def predict(self, embeddings, organism_index, **kwargs):
        """Generate predictions from encoder output.
        
        Args:
            embeddings: ExtendedEmbeddings with encoder_output field
            organism_index: Organism indices
            
        Returns:
            Predictions with shape (batch, sequence_length//128, num_tracks)
        """
        # Verify encoder output is available
        if not hasattr(embeddings, 'encoder_output') or embeddings.encoder_output is None:
            raise ValueError(
                "EncoderOnlyHead requires encoder_output. "
                "Create model with use_encoder_output=True"
            )
        
        # KEY: Access encoder output (raw CNN features, before transformer)
        x = embeddings.encoder_output
        # Shape: (batch, sequence_length//128, 1536)
        
        # Simple architecture: Linear → ReLU → Linear
        x = hk.Linear(256, name='hidden')(x)
        x = jax.nn.relu(x)
        predictions = hk.Linear(self._num_tracks, name='output')(x)
        
        # Return predictions at 128bp resolution
        return predictions
    
    def loss(self, predictions, batch):
        """Compute loss for predictions."""
        targets = batch.get('targets')
        if targets is None:
            return {'loss': jnp.array(0.0)}
        
        # For scalar targets, pool predictions over sequence
        if targets.ndim == 1:
            pred_values = jnp.mean(predictions, axis=1)  # Pool over sequence
            targets = targets[:, None]
        else:
            pred_values = predictions
        
        # MSE loss
        mse_loss = jnp.mean((pred_values - targets) ** 2)
        
        return {
            'loss': mse_loss,
            'mse': mse_loss,
        }
