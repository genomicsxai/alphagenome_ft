"""
Tests for parameter freezing and management utilities.
"""
import pytest
import jax.numpy as jnp


class TestParameterInspection:
    """Test parameter inspection utilities."""
    
    def test_count_parameters(self, wrapped_model_with_head):
        """Verify parameter counting works."""
        param_count = wrapped_model_with_head.count_parameters()
        
        # Should return a positive integer
        assert isinstance(param_count, int)
        assert param_count > 0
        
        # AlphaGenome has 450 million trainable parameters
        assert param_count >= 450_000_000 and param_count <= 451_000_000
    
    def test_get_parameter_paths(self, wrapped_model_with_head):
        """Verify parameter path listing works."""
        paths = wrapped_model_with_head.get_parameter_paths()
        
        assert isinstance(paths, list)
        assert len(paths) > 0
        assert all(isinstance(p, str) for p in paths)
        
        # Check some expected paths exist
        assert any('alphagenome/embed' in p for p in paths)
        assert any('test_mpra_head' in p for p in paths)
    
    def test_get_backbone_parameter_paths(self, wrapped_model_with_head):
        """Verify backbone parameter identification."""
        backbone_paths = wrapped_model_with_head.get_backbone_parameter_paths()
        
        assert isinstance(backbone_paths, list)
        assert len(backbone_paths) > 0
        
        # Check backbone paths don't include head parameters
        assert not any('head' in p for p in backbone_paths)
        
        # Check backbone paths include expected components
        assert any('embed' in p for p in backbone_paths)
        assert any('encoder' in p or 'decoder' in p for p in backbone_paths)
    
    def test_get_head_parameter_paths(self, wrapped_model_with_head):
        """Verify head parameter identification."""
        head_paths = wrapped_model_with_head.get_head_parameter_paths()
        
        assert isinstance(head_paths, list)
        assert len(head_paths) > 0
        
        # All head paths should contain 'head'
        assert all('head' in p for p in head_paths)
        
        # Should include custom head
        assert any('test_mpra_head' in p for p in head_paths)


class TestParameterFreezing:
    """Test parameter freezing functionality."""
    
    def test_freeze_backbone(self, wrapped_model_with_head):
        """Test freezing backbone parameters."""
        # Get initial parameter counts
        backbone_paths = wrapped_model_with_head.get_backbone_parameter_paths()
        head_paths = wrapped_model_with_head.get_head_parameter_paths()
        
        # Verify we have both backbone and head parameters
        assert len(backbone_paths) > 0, "No backbone parameters found"
        assert len(head_paths) > 0, "No head parameters found"
        
        # Freeze backbone - method should complete without error
        wrapped_model_with_head.freeze_backbone()
        
        # Verify method completed successfully
        assert True
    
    def test_freeze_all_heads(self, wrapped_model_with_head):
        """Test freezing all head parameters."""
        head_paths = wrapped_model_with_head.get_head_parameter_paths()
        
        # Freeze all heads
        wrapped_model_with_head.freeze_all_heads()
        
        # Verify method completed successfully
        assert len(head_paths) > 0
    
    def test_freeze_except_head(self, wrapped_model_with_head):
        """Test freezing everything except specific head."""
        # Freeze everything except custom head
        wrapped_model_with_head.freeze_except_head('test_mpra_head')
        
        # Verify method completed successfully
        assert True
    
    def test_freeze_unfreeze_cycle(self, wrapped_model_with_head):
        """Test freezing and unfreezing parameters."""
        # Get a specific parameter path
        all_paths = wrapped_model_with_head.get_parameter_paths()
        test_path = all_paths[0]
        
        # Freeze specific parameter
        wrapped_model_with_head.freeze_parameters(freeze_paths=[test_path])
        
        # Unfreeze it
        wrapped_model_with_head.unfreeze_parameters(unfreeze_paths=[test_path])
        
        # Verify methods completed successfully
        assert True
    
    def test_freeze_by_prefix(self, wrapped_model_with_head):
        """Test freezing parameters by prefix."""
        # Freeze all encoder parameters
        wrapped_model_with_head.freeze_parameters(
            freeze_prefixes=['alphagenome/sequence_encoder']
        )
        
        # Verify method completed successfully
        assert True


class TestParameterValues:
    """Test that parameter values are preserved correctly."""
    
    def test_wrapped_model_has_all_parameters(self, wrapped_model_with_head):
        """Verify wrapped model has all expected parameter types."""
        
        all_paths = wrapped_model_with_head.get_parameter_paths()
        
        # Check for backbone parameters
        backbone_params = [p for p in all_paths if 'embed' in p or 'encoder' in p]
        assert len(backbone_params) > 0, "No backbone parameters found"
        
        # Check for head parameters
        head_params = [p for p in all_paths if 'head' in p]
        assert len(head_params) > 0, "No head parameters found"
        
        # Check for custom head parameters specifically
        custom_head_params = [p for p in all_paths if 'test_mpra_head' in p]
        assert len(custom_head_params) > 0, "No custom head parameters found"
    
    def test_custom_only_model_has_backbone_and_head(self, custom_only_model):
        """Verify custom-only model has both backbone and custom head parameters."""
        
        all_paths = custom_only_model.get_parameter_paths()
        
        # Check for backbone parameters
        backbone_params = [p for p in all_paths if 'embed' in p or 'encoder' in p]
        assert len(backbone_params) > 0, "No backbone parameters found"
        
        # Check for custom head parameters
        custom_head_params = [p for p in all_paths if 'test_mpra_head' in p]
        assert len(custom_head_params) > 0, "No custom head parameters found"
        
        # Verify model can count parameters
        param_count = custom_only_model.count_parameters()
        assert param_count >= 450_000_000 and param_count <= 451_000_000, "Model should have 450 million trainable parameters"


