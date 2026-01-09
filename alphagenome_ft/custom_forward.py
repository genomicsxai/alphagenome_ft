"""
Custom forward pass that exposes encoder output before transformer.
"""
import haiku as hk
from alphagenome_research.model import model as model_lib
from alphagenome_research.model import embeddings as embeddings_module
from alphagenome_ft.embeddings_extended import ExtendedEmbeddings
from jaxtyping import Array, Float, Int


def forward_with_encoder_output(
    alphagenome: model_lib.AlphaGenome,
    dna_sequence: Float[Array, 'B S 4'],
    organism_index: Int[Array, 'B'],
) -> ExtendedEmbeddings:
    """Run AlphaGenome forward pass and capture encoder output.
    
    This manually runs the encoder, transformer, and decoder to capture
    the intermediate encoder output before it goes through the transformer.
    
    Args:
        alphagenome: AlphaGenome model instance
        dna_sequence: Input DNA sequence (one-hot encoded)
        organism_index: Organism indices (0=human, 1=mouse)
        
    Returns:
        ExtendedEmbeddings containing standard embeddings plus encoder output
    """
    # Step 1: Run encoder
    encoder = model_lib.SequenceEncoder()
    trunk_encoder, intermediates = encoder(dna_sequence)
    # trunk_encoder shape: (B, S//128, D) - OUTPUT BEFORE TRANSFORMER
    
    # Add organism embedding to encoder output
    num_organisms = alphagenome._num_organisms
    organism_embedding_trunk = hk.Embed(num_organisms, trunk_encoder.shape[-1])(
        organism_index
    )
    trunk_with_org = trunk_encoder + organism_embedding_trunk[:, None, :]
    
    # Step 2: Run transformer
    transformer = model_lib.TransformerTower()
    trunk_transformer, pair_activations = transformer(trunk_with_org)
    # trunk_transformer shape: (B, S//128, D) - OUTPUT AFTER TRANSFORMER
    
    # Step 3: Run decoder
    decoder = model_lib.SequenceDecoder()
    x = decoder(trunk_transformer, intermediates)
    # x shape: (B, S, D) - OUTPUT AT 1BP RESOLUTION
    
    # Step 4: Create output embeddings (standard AlphaGenome process)
    embeddings_128bp = embeddings_module.OutputEmbedder(num_organisms)(
        trunk_transformer, organism_index
    )
    embeddings_1bp = embeddings_module.OutputEmbedder(num_organisms)(
        x, organism_index, embeddings_128bp
    )
    embeddings_pair = embeddings_module.OutputPair(num_organisms)(
        pair_activations, organism_index
    )
    
    # Step 5: Return extended embeddings with encoder output
    return ExtendedEmbeddings(
        embeddings_1bp=embeddings_1bp,
        embeddings_128bp=embeddings_128bp,
        embeddings_pair=embeddings_pair,
        encoder_output=trunk_encoder,  # Raw encoder output (before transformer)
    )

