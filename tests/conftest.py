"""
Pytest configuration and fixtures for alphagenome_ft tests.
"""

import pytest
import jax
import jax.numpy as jnp
import haiku as hk
from alphagenome.data import genome
from alphagenome.models import dna_output
from alphagenome_research.model import dna_model

from alphagenome_ft import (
    CustomHead,
    HeadConfig,
    HeadType,
    register_custom_head,
    wrap_pretrained_model,
    add_heads_to_model,
    create_model_with_heads,
)

from tests.kaggle_util import kaggle_credentials_available


# ============================================================================
# Test Custom Head Definition
# ============================================================================

class MPRAHeadForTesting(CustomHead):
    """Test custom head for MPRA predictions (used in fixtures)."""
    
    def __init__(self, *, name, output_type, num_tracks, num_organisms, metadata):
        super().__init__(
            name=name,
            num_tracks=num_tracks,
            output_type=output_type,
            num_organisms=num_organisms,
            metadata=metadata,
        )
    
    def predict(self, embeddings, organism_index, **kwargs):
        """Predict MPRA activity from embeddings."""
        # Get 1bp resolution embeddings
        x = embeddings.get_sequence_embeddings(resolution=1)
        
        # Simple prediction layers
        x = hk.Linear(256, name='hidden')(x)
        x = jax.nn.relu(x)
        predictions = hk.Linear(self._num_tracks, name='output')(x)
        
        return {'predictions': predictions}
    
    def loss(self, predictions, batch):
        """Compute MSE loss."""
        targets = batch.get('targets')
        if targets is None:
            return {'loss': jnp.array(0.0)}
        
        pred_values = predictions['predictions']
        mse_loss = jnp.mean((pred_values - targets) ** 2)
        
        return {'loss': mse_loss, 'mse': mse_loss}


# ============================================================================
# Kaggle credential helpers
# ============================================================================


def _has_kaggle_credentials() -> bool:
    """Return True if Kaggle credentials are in the environment or ``~/.kaggle/kaggle.json``."""
    return kaggle_credentials_available()


def require_kaggle_credentials() -> None:
    """Skip the current test if Kaggle credentials are not available.

    Many tests rely on ``dna_model.create_from_kaggle``, which triggers an
    interactive kagglehub login when credentials are missing. In CI / non‑TTY
    environments this causes hangs and getpass errors, so we proactively
    skip such tests unless credentials exist (see ``tests/kaggle_util.py``).
    """
    if not _has_kaggle_credentials():
        pytest.skip(
            "Kaggle credentials not found. Set KAGGLE_USERNAME and KAGGLE_KEY, "
            "or place username/key in ~/.kaggle/kaggle.json. "
            "Skipping tests that require downloading the AlphaGenome model."
        )


# ============================================================================
# Pytest Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def device():
    """Get compute device (CPU for testing)."""
    return jax.devices('cpu')[0]


@pytest.fixture(scope="session")
def base_model(device):
    """Load pretrained AlphaGenome model (reused across tests)."""
    require_kaggle_credentials()
    return dna_model.create_from_kaggle('all_folds', device=device)


@pytest.fixture(scope="session")
def test_interval():
    """Test genomic interval."""
    return genome.Interval(chromosome='chr22', start=35677410, end=36725986)


@pytest.fixture(scope="session")
def test_sequence(base_model, test_interval):
    """Extract and encode test sequence."""
    sequence_str = base_model._fasta_extractors[
        dna_model.Organism.HOMO_SAPIENS
    ].extract(test_interval)
    sequence_onehot = base_model._one_hot_encoder.encode(sequence_str)
    return jnp.array(sequence_onehot)[jnp.newaxis]  # Add batch dimension


@pytest.fixture(scope="session")
def organism_index():
    """Organism index for human."""
    return jnp.array([0])  # 0 = HOMO_SAPIENS


@pytest.fixture(scope="session")
def strand_reindexing(base_model):
    """Strand reindexing for predictions (device-placed mapping per OutputType)."""
    dev = base_model._device_context._device
    sr = base_model._metadata[dna_model.Organism.HOMO_SAPIENS].strand_reindexing
    return jax.device_put(sr, dev)


@pytest.fixture(scope="session")
def all_standard_requested_outputs():
    """All :class:`~alphagenome.models.dna_output.OutputType` values for ``_predict``."""
    return tuple(dna_output.OutputType)


@pytest.fixture(scope="function")
def mpra_head_config():
    """Configuration for test MPRA head."""
    return HeadConfig(
        type=HeadType.GENOME_TRACKS,
        name='test_mpra_head',
        output_type=dna_output.OutputType.RNA_SEQ,
        num_tracks=1,
    )


@pytest.fixture(scope="function")
def registered_mpra_head(mpra_head_config):
    """Register test MPRA head and clean up after test."""
    head_name = 'test_mpra_head'
    register_custom_head(head_name, MPRAHeadForTesting, mpra_head_config)
    yield head_name
    # Cleanup is handled by the registry (overwrites are allowed)


@pytest.fixture(scope="function")
def wrapped_model_with_head(base_model, registered_mpra_head):
    """Create wrapped model with custom head."""
    wrapped = wrap_pretrained_model(base_model)
    wrapped = add_heads_to_model(wrapped, heads=[registered_mpra_head])
    return wrapped


@pytest.fixture(scope="function")
def custom_only_model(registered_mpra_head, device):
    """Create model with only custom head."""
    require_kaggle_credentials()
    return create_model_with_heads(
        'all_folds',
        heads=[registered_mpra_head],
        device=device,
    )


