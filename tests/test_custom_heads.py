"""
Tests for custom head registration and functionality.
"""
import pytest
from alphagenome.models import dna_output
from alphagenome_ft import (
    CustomHead,
    HeadConfig,
    HeadType,
    get_predefined_head_config,
    get_registered_head_config,
    register_predefined_head,
    register_custom_head,
    is_custom_head,
    get_custom_head_config,
    list_custom_heads,
)
from tests.conftest import MPRAHeadForTesting


class TestHeadRegistry:
    """Test custom head registration and registry functions."""
    
    def test_register_custom_head(self, mpra_head_config):
        """Test registering a custom head."""
        head_name = 'test_registration_head'
        
        # Register head
        register_custom_head(head_name, MPRAHeadForTesting, mpra_head_config)
        
        # Verify it's registered
        assert is_custom_head(head_name)
    
    def test_is_custom_head(self, registered_mpra_head):
        """Test checking if head is registered."""
        # Should be registered
        assert is_custom_head(registered_mpra_head)
        
        # Should not be registered
        assert not is_custom_head('nonexistent_head')
    
    def test_get_custom_head_config(self, registered_mpra_head, mpra_head_config):
        """Test retrieving head configuration."""
        config = get_custom_head_config(registered_mpra_head)
        
        assert isinstance(config, HeadConfig)
        assert config.name == mpra_head_config.name
        assert config.type == mpra_head_config.type
        assert config.output_type == mpra_head_config.output_type
        assert config.num_tracks == mpra_head_config.num_tracks
    
    def test_list_custom_heads(self, registered_mpra_head):
        """Test listing all registered heads."""
        heads = list_custom_heads()
        
        assert isinstance(heads, list)
        assert registered_mpra_head in heads
    
    def test_overwrite_head_registration(self, mpra_head_config):
        """Test that re-registering a head overwrites the previous one."""
        head_name = 'test_overwrite_head'
        
        # Register once
        register_custom_head(head_name, MPRAHeadForTesting, mpra_head_config)
        first_config = get_custom_head_config(head_name)
        
        # Register again with different config
        new_config = HeadConfig(
            type=HeadType.GENOME_TRACKS,
            name=head_name,
            output_type=dna_output.OutputType.ATAC,  # Different output type
            num_tracks=2,  # Different num_tracks
        )
        register_custom_head(head_name, MPRAHeadForTesting, new_config)
        second_config = get_custom_head_config(head_name)
        
        # Verify it was overwritten
        assert second_config.output_type != first_config.output_type
        assert second_config.num_tracks != first_config.num_tracks


class TestHeadConfig:
    """Test HeadConfig functionality."""
    
    def test_head_config_creation(self):
        """Test creating a HeadConfig."""
        config = HeadConfig(
            type=HeadType.GENOME_TRACKS,
            name='test_head',
            output_type=dna_output.OutputType.RNA_SEQ,
            num_tracks=1,
        )
        
        assert config.type == HeadType.GENOME_TRACKS
        assert config.name == 'test_head'
        assert config.output_type == dna_output.OutputType.RNA_SEQ
        assert config.num_tracks == 1
    
    def test_head_config_with_metadata(self):
        """Test creating a HeadConfig with metadata."""
        metadata = {'key': 'value', 'number': 42}
        
        config = HeadConfig(
            type=HeadType.GENOME_TRACKS,
            name='test_head',
            output_type=dna_output.OutputType.RNA_SEQ,
            num_tracks=1,
            metadata=metadata,
        )
        
        assert config.metadata == metadata


class TestCustomHeadBase:
    """Test CustomHead base class."""
    
    def test_custom_head_has_required_methods(self):
        """Test that CustomHead requires predict and loss methods."""
        # This is enforced by Python's abstract base class
        # Attempting to instantiate without implementing methods should fail
        
        # MPRAHeadForTesting implements both methods, so it should work
        # This is tested implicitly by other tests
        assert hasattr(MPRAHeadForTesting, 'predict')
        assert hasattr(MPRAHeadForTesting, 'loss')
    
    def test_custom_head_in_model(self, wrapped_model_with_head):
        """Test custom head initialization within a model."""
        # Custom heads are Haiku modules and must be initialized within hk.transform()
        # This test verifies that the head was properly initialized in the wrapped model
        
        # Check that custom head parameters exist (this means it was initialized)
        custom_head_params = [
            k for k in wrapped_model_with_head._params.keys()
            if 'test_mpra_head' in k
        ]
        
        assert len(custom_head_params) > 0, "Custom head was not properly initialized"


class TestPredefinedHeadAliases:
    """Test predefined head registration with aliases."""

    def test_register_predefined_head_uses_alias_as_name(self):
        alias = 'test_rna_alias_head'
        cfg = get_predefined_head_config('rna_seq', num_tracks=2)

        register_predefined_head(alias, cfg)
        registered = get_registered_head_config(alias)

        assert registered.name == alias

    def test_register_predefined_head_accepts_alias_named_config(self):
        alias = 'test_atac_alias_head'
        cfg = get_predefined_head_config('atac', num_tracks=1)
        cfg = type(cfg)(**{**cfg.__dict__, 'name': alias})

        register_predefined_head(alias, cfg)
        registered = get_registered_head_config(alias)

        assert registered.name == alias

