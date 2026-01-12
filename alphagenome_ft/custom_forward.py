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
    
    This manually replicates the AlphaGenome forward pass to capture
    the intermediate encoder output before it goes through the transformer.
    
    IMPORTANT: This function must be called within a Haiku transform that:
    1. Has mixed precision policies applied to all modules (see test for example)
    2. Uses `hk.name_scope('alphagenome')` to match parameter paths
    3. Creates modules in the same order as AlphaGenome.__call__
    
    See tests/test_model_predictions.py::test_custom_forward_matches_standard_forward
    for a complete working example of how to use this function correctly.
    
    Args:
        alphagenome: AlphaGenome model instance (used for num_organisms, not called)
        dna_sequence: Input DNA sequence (one-hot encoded)
        organism_index: Organism indices (0=human, 1=mouse)
        
    Returns:
        ExtendedEmbeddings containing standard embeddings plus encoder output
        
    Example:
        ```python
        import jmp
        import haiku as hk
        from alphagenome_research.model import model as model_lib
        from alphagenome_research.model import embeddings as embeddings_module
        
        @hk.transform_with_state
        def custom_forward(dna_sequence, organism_index, metadata):
            policy = jmp.get_policy('params=float32,compute=bfloat16,output=bfloat16')
            with hk.mixed_precision.push_policy(model_lib.AlphaGenome, policy):
                with hk.mixed_precision.push_policy(model_lib.SequenceEncoder, policy):
                    with hk.mixed_precision.push_policy(model_lib.TransformerTower, policy):
                        with hk.mixed_precision.push_policy(model_lib.SequenceDecoder, policy):
                            with hk.mixed_precision.push_policy(embeddings_module.OutputEmbedder, policy):
                                with hk.mixed_precision.push_policy(embeddings_module.OutputPair, policy):
                                    with hk.name_scope('alphagenome'):
                                        alphagenome = model_lib.AlphaGenome(metadata)
                                        return forward_with_encoder_output(
                                            alphagenome, dna_sequence, organism_index
                                        )
        ```
    """
    # Get number of organisms from the alphagenome instance
    num_organisms = alphagenome._num_organisms
    
    # Step 1: Run encoder - exactly as AlphaGenome does
    trunk, intermediates = model_lib.SequenceEncoder()(dna_sequence)
    
    # Save encoder output (before organism embedding and transformer)
    encoder_output = trunk
    
    # Add organism embedding
    organism_embedding_trunk = hk.Embed(num_organisms, trunk.shape[-1])(
        organism_index
    )
    trunk += organism_embedding_trunk[:, None, :]
    
    # Step 2: Run transformer
    trunk, pair_activations = model_lib.TransformerTower()(trunk)
    
    # Step 3: Run decoder
    x = model_lib.SequenceDecoder()(trunk, intermediates)
    
    # Step 4: Create output embeddings (same as AlphaGenome)
    embeddings_128bp = embeddings_module.OutputEmbedder(num_organisms)(
        trunk, organism_index
    )
    embeddings_1bp = embeddings_module.OutputEmbedder(num_organisms)(
        x, organism_index, embeddings_128bp
    )
    
    # Step 5: Return extended embeddings with encoder output
    # Note: ExtendedEmbeddings does not include pair embeddings
    return ExtendedEmbeddings(
        embeddings_1bp=embeddings_1bp,
        embeddings_128bp=embeddings_128bp,
        encoder_output=encoder_output,  # Raw encoder output (before org embedding and transformer)
    )

