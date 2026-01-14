"""
Tests for checkpoint save and load functionality.
"""
import pytest
import tempfile
import shutil
from pathlib import Path
import json
import jax.numpy as jnp

from alphagenome_ft import (
    register_custom_head,
    create_model_with_custom_heads,
    load_checkpoint,
    get_custom_head_config,
)
from tests.conftest import MPRAHeadForTesting


class TestCheckpointSaveLoad:
    """Test checkpoint saving and loading."""
    
    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Create temporary directory for checkpoint files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        # Cleanup after test
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def trained_model(self, registered_mpra_head, device):
        """Create a model that simulates being trained."""
        model = create_model_with_custom_heads(
            'all_folds',
            custom_heads=[registered_mpra_head],
            device=device,
        )
        # Freeze backbone to simulate finetuning setup
        model.freeze_except_head(registered_mpra_head)
        return model
    
    def test_save_heads_only_checkpoint(self, trained_model, temp_checkpoint_dir):
        """Test saving checkpoint with only custom heads."""
        checkpoint_path = temp_checkpoint_dir / 'heads_only'
        
        # Save checkpoint
        trained_model.save_checkpoint(
            checkpoint_path,
            save_full_model=False
        )
        
        # Verify checkpoint files exist
        assert checkpoint_path.exists()
        assert (checkpoint_path / 'config.json').exists()
        assert (checkpoint_path / 'checkpoint').exists()
        
        # Verify config contents
        with open(checkpoint_path / 'config.json', 'r') as f:
            config = json.load(f)
        
        assert 'custom_heads' in config
        assert 'head_configs' in config
        assert 'save_full_model' in config
        assert config['save_full_model'] is False
        assert len(config['custom_heads']) == 1
        assert 'test_mpra_head' in config['custom_heads']
    
    def test_save_full_model_checkpoint(self, trained_model, temp_checkpoint_dir):
        """Test saving checkpoint with full model."""
        checkpoint_path = temp_checkpoint_dir / 'full_model'
        
        # Save checkpoint
        trained_model.save_checkpoint(
            checkpoint_path,
            save_full_model=True
        )
        
        # Verify checkpoint files exist
        assert checkpoint_path.exists()
        assert (checkpoint_path / 'config.json').exists()
        assert (checkpoint_path / 'checkpoint').exists()
        
        # Verify config indicates full model
        with open(checkpoint_path / 'config.json', 'r') as f:
            config = json.load(f)
        
        assert config['save_full_model'] is True
    
    def test_load_heads_only_checkpoint(
        self,
        trained_model,
        temp_checkpoint_dir,
        registered_mpra_head,
        mpra_head_config,
        device
    ):
        """Test loading a heads-only checkpoint."""
        checkpoint_path = temp_checkpoint_dir / 'heads_only'
        
        # Save checkpoint
        trained_model.save_checkpoint(checkpoint_path, save_full_model=False)
        
        # Verify we can load it
        # Note: Head must be registered before loading
        loaded_model = load_checkpoint(
            checkpoint_path,
            base_model_version='all_folds',
            device=device
        )
        
        # Verify model structure
        assert loaded_model is not None
        assert hasattr(loaded_model, '_params')
        assert hasattr(loaded_model, '_state')
        assert hasattr(loaded_model, '_custom_heads')
        assert 'test_mpra_head' in loaded_model._custom_heads
        
        # Verify parameter count matches
        assert loaded_model.count_parameters() == trained_model.count_parameters()
    
    def test_load_full_model_checkpoint(
        self,
        trained_model,
        temp_checkpoint_dir,
        registered_mpra_head,
        device
    ):
        """Test loading a full model checkpoint."""
        checkpoint_path = temp_checkpoint_dir / 'full_model'
        
        # Save checkpoint
        trained_model.save_checkpoint(checkpoint_path, save_full_model=True)
        
        # Load checkpoint
        loaded_model = load_checkpoint(
            checkpoint_path,
            base_model_version='all_folds',
            device=device
        )
        
        # Verify model structure
        assert loaded_model is not None
        assert hasattr(loaded_model, '_params')
        assert hasattr(loaded_model, '_custom_heads')
        
        # Verify parameter count matches
        assert loaded_model.count_parameters() == trained_model.count_parameters()
    
    def test_loaded_model_predictions(
        self,
        trained_model,
        temp_checkpoint_dir,
        test_sequence,
        organism_index,
        strand_reindexing,
        device
    ):
        """Test that loaded model can make predictions."""
        checkpoint_path = temp_checkpoint_dir / 'test_predictions'
        
        # Save checkpoint
        trained_model.save_checkpoint(checkpoint_path, save_full_model=False)
        
        # Get predictions from original model
        with trained_model._device_context:
            original_predictions = trained_model._predict(
                trained_model._params,
                trained_model._state,
                test_sequence,
                organism_index,
                negative_strand_mask=jnp.zeros(len(test_sequence), dtype=bool),
                strand_reindexing=strand_reindexing,
            )
        
        # Load checkpoint
        loaded_model = load_checkpoint(
            checkpoint_path,
            base_model_version='all_folds',
            device=device
        )
        
        # Get predictions from loaded model
        with loaded_model._device_context:
            loaded_predictions = loaded_model._predict(
                loaded_model._params,
                loaded_model._state,
                test_sequence,
                organism_index,
                negative_strand_mask=jnp.zeros(len(test_sequence), dtype=bool),
                strand_reindexing=strand_reindexing,
            )
        
        # Verify predictions match (or are very close due to numerical precision)
        # Note: We check custom head predictions
        assert 'test_mpra_head' in original_predictions
        assert 'test_mpra_head' in loaded_predictions
        
        original_pred = original_predictions['test_mpra_head']
        loaded_pred = loaded_predictions['test_mpra_head']
        
        # Check shape matches
        assert original_pred['predictions'].shape == loaded_pred['predictions'].shape
        
        # Check values are close (allow small numerical differences)
        assert jnp.allclose(
            original_pred['predictions'],
            loaded_pred['predictions'],
            rtol=1e-5,
            atol=1e-6
        )
    
    def test_checkpoint_head_config_preservation(
        self,
        trained_model,
        temp_checkpoint_dir,
        mpra_head_config,
        device
    ):
        """Test that head configuration is preserved in checkpoint."""
        checkpoint_path = temp_checkpoint_dir / 'config_test'
        
        # Save checkpoint
        trained_model.save_checkpoint(checkpoint_path)
        
        # Load config file
        with open(checkpoint_path / 'config.json', 'r') as f:
            saved_config = json.load(f)
        
        # Verify head config is saved
        assert 'head_configs' in saved_config
        assert 'test_mpra_head' in saved_config['head_configs']
        
        head_cfg = saved_config['head_configs']['test_mpra_head']
        assert head_cfg['name'] == mpra_head_config.name
        assert head_cfg['type'] == mpra_head_config.type.name
        assert head_cfg['output_type'] == mpra_head_config.output_type.name
        assert head_cfg['num_tracks'] == mpra_head_config.num_tracks
    
    def test_get_head_parameters(self, trained_model):
        """Test extracting head parameters from model."""
        # Get head parameters
        head_params = trained_model.get_head_parameters('test_mpra_head')
        
        # Verify parameters exist
        assert head_params is not None
        # Parameters should be a nested dict structure
        assert isinstance(head_params, dict)
    
    def test_checkpoint_missing_config_error(self, temp_checkpoint_dir, device):
        """Test that loading fails gracefully when config is missing."""
        checkpoint_path = temp_checkpoint_dir / 'invalid'
        checkpoint_path.mkdir()
        
        # Try to load without config.json
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_checkpoint(checkpoint_path, device=device)
    
    def test_checkpoint_missing_checkpoint_error(
        self,
        temp_checkpoint_dir,
        mpra_head_config,
        device
    ):
        """Test that loading fails gracefully when checkpoint files are missing."""
        checkpoint_path = temp_checkpoint_dir / 'invalid'
        checkpoint_path.mkdir()
        
        # Create config but no checkpoint
        config = {
            'custom_heads': ['test_mpra_head'],
            'head_configs': {
                'test_mpra_head': {
                    'type': mpra_head_config.type.name,
                    'name': mpra_head_config.name,
                    'output_type': mpra_head_config.output_type.name,
                    'num_tracks': mpra_head_config.num_tracks,
                    'metadata': {},
                }
            },
            'save_full_model': False,
        }
        
        with open(checkpoint_path / 'config.json', 'w') as f:
            json.dump(config, f)
        
        # Try to load without checkpoint files
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            load_checkpoint(checkpoint_path, device=device)
    
    def test_multiple_save_load_cycles(
        self,
        trained_model,
        temp_checkpoint_dir,
        test_sequence,
        organism_index,
        strand_reindexing,
        device
    ):
        """Test saving and loading multiple times preserves model."""
        # Save checkpoint 1
        checkpoint1 = temp_checkpoint_dir / 'checkpoint1'
        trained_model.save_checkpoint(checkpoint1)
        
        # Load checkpoint 1
        model1 = load_checkpoint(checkpoint1, base_model_version='all_folds', device=device)
        
        # Save checkpoint 2 from loaded model
        checkpoint2 = temp_checkpoint_dir / 'checkpoint2'
        model1.save_checkpoint(checkpoint2)
        
        # Load checkpoint 2
        model2 = load_checkpoint(checkpoint2, base_model_version='all_folds', device=device)
        
        # Get predictions from all three models
        models = [trained_model, model1, model2]
        predictions = []
        
        for model in models:
            with model._device_context:
                pred = model._predict(
                    model._params,
                    model._state,
                    test_sequence,
                    organism_index,
                    negative_strand_mask=jnp.zeros(len(test_sequence), dtype=bool),
                    strand_reindexing=strand_reindexing,
                )
                predictions.append(pred['test_mpra_head']['predictions'])
        
        # All predictions should be very close
        assert jnp.allclose(predictions[0], predictions[1], rtol=1e-5, atol=1e-6)
        assert jnp.allclose(predictions[1], predictions[2], rtol=1e-5, atol=1e-6)
    
    def test_checkpoint_with_string_path(self, trained_model, temp_checkpoint_dir):
        """Test that checkpoint works with string paths (not just Path objects)."""
        checkpoint_path_str = str(temp_checkpoint_dir / 'string_path')
        
        # Save with string path
        trained_model.save_checkpoint(checkpoint_path_str)
        
        # Verify it was created
        assert Path(checkpoint_path_str).exists()
        assert (Path(checkpoint_path_str) / 'config.json').exists()
