"""
Extended embeddings that include encoder output before transformer.
"""
from alphagenome import typing
import chex
from jaxtyping import Array, Float


@typing.jaxtyped
@chex.dataclass(frozen=True, kw_only=True)
class ExtendedEmbeddings:
    """Extended AlphaGenome embeddings including encoder output.
    
    Provides access to multiple stages of the AlphaGenome architecture:
    - encoder_output: Raw output from SequenceEncoder (before transformer)
    - embeddings_128bp: Output from TransformerTower (global context)
    - embeddings_1bp: Output from SequenceDecoder (local + global)
    - embeddings_pair: Pairwise representations from transformer
    """
    
    # Standard embeddings (from alphagenome_research)
    embeddings_1bp: Float[Array, 'B S 1536'] | None = None
    embeddings_128bp: Float[Array, 'B S//128 3072'] | None = None
    
    # NEW: Encoder output (before transformer)
    encoder_output: Float[Array, 'B S//128 D'] | None = None
    
    def get_sequence_embeddings(self, resolution: int) -> Float[Array, 'B S D']:
        """Get embeddings at specified resolution.
        
        Args:
            resolution: Resolution in base pairs (1, 128, or 'encoder')
            
        Returns:
            Embeddings at the requested resolution.
        """
        if resolution == 128:
            return self.embeddings_128bp
        elif resolution == 1:
            return self.embeddings_1bp
        elif resolution == 'encoder':
            # Special case: return raw encoder output (before transformer)
            return self.encoder_output
        else:
            raise ValueError(
                f'Unsupported resolution: {resolution}. '
                f'Use 1, 128, or "encoder"'
            )
    
    def has_encoder_output(self) -> bool:
        """Check if encoder output is available."""
        return self.encoder_output is not None

