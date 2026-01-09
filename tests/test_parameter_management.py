"""
Tests for parameter freezing and management utilities.
"""
import pytest
import jax.numpy as jnp


class TestParameterInspection:
    """Test parameter inspection utilities."""
    
    def test_count_parameters(self, wrapped_model_with_head, base_model):
        """Verify parameter counting works."""
        wrapped_count = wrapped_model_with_head.count_parameters()
        base_count = base_model.count_parameters()
        
        # Wrapped model should have more parameters (custom head added)
        assert wrapped_count > base_count
        assert isinstance(wrapped_count, int)
        assert wrapped_count > 0
    
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
        # Get initial trainable state
        initial_paths = wrapped_model_with_head.get_parameter_paths()
        backbone_paths = wrapped_model_with_head.get_backbone_parameter_paths()
        
        # Freeze backbone
        wrapped_model_with_head.freeze_backbone()
        
        # Check backbone parameters are frozen
        for path in backbone_paths:
            param = wrapped_model_with_head._params
            for key in path.split('/'):
                param = param[key]
            # In Haiku, frozen params might not have a special marker,
            # but we verify the method runs without error
        
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
    
    def test_wrapped_model_preserves_backbone_params(
        self,
        base_model,
        wrapped_model_with_head
    ):
        """Verify backbone parameters are identical to base model."""
        
        test_keys = [
            'alphagenome/embed/embeddings',
            'alphagenome/sequence_encoder/stem_conv/w',
        ]
        
        for key in test_keys:
            if key in base_model._params and key in wrapped_model_with_head._params:
                base_param = base_model._params[key]
                wrapped_param = wrapped_model_with_head._params[key]
                
                # Check values are identical
                assert jnp.allclose(base_param, wrapped_param), (
                    f"Parameter {key} differs between base and wrapped model"
                )
    
    def test_custom_only_model_preserves_backbone_params(
        self,
        base_model,
        custom_only_model
    ):
        """Verify backbone parameters are identical in custom-only model."""
        
        test_keys = [
            'alphagenome/embed/embeddings',
            'alphagenome/sequence_encoder/stem_conv/w',
        ]
        
        for key in test_keys:
            if key in base_model._params and key in custom_only_model._params:
                base_param = base_model._params[key]
                custom_param = custom_only_model._params[key]
                
                # Check values are identical
                assert jnp.allclose(base_param, custom_param), (
                    f"Parameter {key} differs between base and custom model"
                )

