# AlphaGenome Finetuning (`alphagenome-ft`)

A lightweight Python package for finetuning [Google DeepMind's AlphaGenome](https://github.com/google-deepmind/alphagenome_research/) model with custom prediction heads and parameter freezing capabilities, **without modifying the original codebase**.

## Features

- **Custom Prediction Heads**: Define and register your own task-specific prediction heads
- **Parameter Freezing**: Flexible parameter management (freeze backbone, heads, or specific layers)
- **Easy Integration**: Works seamlessly with pretrained AlphaGenome models (Simple wrapper classes)
- **Parameter Inspection**: Utilities to explore and count model parameters
- **JAX/Haiku Native**: Built on the same framework as AlphaGenome

## Installation

### From PyPI (when published)
```bash
pip install alphagenome-ft
```

### From Source
```bash
git clone https://github.com/yourusername/alphagenome_ft.git
cd alphagenome_ft
pip install -e .
```

### Requirements

- Python ≥ 3.10
- JAX ≥ 0.4.0
- Haiku ≥ 0.0.10
- AlphaGenome Research (install from [GitHub](https://github.com/google-deepmind/alphagenome_research/))

```bash
pip install git+https://github.com/google-deepmind/alphagenome_research.git
```

## Quick Start

### Option 1: Add Custom Head to Pretrained Model

Keep all standard AlphaGenome heads and add your own:

```python
import jax
import jax.numpy as jnp
import haiku as hk
from alphagenome.models import dna_output
from alphagenome_research.model import dna_model
from alphagenome_ft import (
    CustomHead,
    HeadConfig,
    HeadType,
    register_custom_head,
    wrap_pretrained_model,
    add_custom_heads_to_model,
)

# 1. Define your custom head
class MPRAHead(CustomHead):
    """Predict MPRA (Massively Parallel Reporter Assay) activity."""
    
    def predict(self, embeddings, organism_index, **kwargs):
        # Get 1bp resolution embeddings
        x = embeddings.get_sequence_embeddings(resolution=1)
        
        # Add your prediction layers
        x = hk.Linear(256, name='hidden')(x)
        x = jax.nn.relu(x)
        predictions = hk.Linear(self._num_tracks, name='output')(x)
        
        return {'predictions': predictions}
    
    def loss(self, predictions, batch):
        targets = batch.get('targets')
        if targets is None:
            return {'loss': jnp.array(0.0)}
        
        mse = jnp.mean((predictions['predictions'] - targets) ** 2)
        return {'loss': mse, 'mse': mse}

# 2. Register the head
register_custom_head(
    'mpra_head',
    MPRAHead,
    HeadConfig(
        type=HeadType.GENOME_TRACKS,
        name='mpra_head',
        output_type=dna_output.OutputType.RNA_SEQ,
        num_tracks=1,
    )
)

# 3. Load pretrained model and add custom head
base_model = dna_model.create_from_kaggle('all_folds')
model = wrap_pretrained_model(base_model)
model = add_custom_heads_to_model(model, custom_heads=['mpra_head'])

# 4. Freeze backbone for finetuning
model.freeze_backbone()

# Now train only the custom head!
```

### Option 2: Create Model with Custom Head Only

Replace standard heads with your custom head:

```python
from alphagenome_ft import create_model_with_custom_heads

# After registering your custom head (see above)
model = create_model_with_custom_heads(
    'all_folds',
    custom_heads=['mpra_head'],
    device=jax.devices('cpu')[0],  # Optional: specify device
)

# Freeze everything except custom head
model.freeze_except_head('mpra_head')
```

## AlphaGenome Architecture

Understanding the architecture helps design custom heads:

```
DNA Sequence (B, S, 4)
    ↓
┌─────────────────────────────────────┐
│ BACKBONE (can be frozen)            │
│  ├─ SequenceEncoder ←────────────┐  │
│  ├─ TransformerTower (9 blocks)  │  │
│  └─ SequenceDecoder              │  │
└──────────────────────────────────┼──┘
    ↓                              │
┌──────────────────────────────────┼──┐
│ EMBEDDINGS (multi-resolution)    │  │
│  ├─ embeddings_1bp:   (B, S, 1536)  │
│  ├─ embeddings_128bp: (B, S/128, 3072) │
│  ├─ embeddings_pair:  (B, S/2048, S/2048, 128) │
│  └─ encoder_output*:  (B, S/128, D)  │  *Advanced
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ HEADS (task-specific)               │
│  ├─ Standard: ATAC, RNA-seq, etc.   │
│  └─ Custom: YOUR_HEAD_HERE ← Add!   │
└─────────────────────────────────────┘
```

### Advanced: Access Earlier Architecture Layers

By default, custom heads receive embeddings **after** the decoder. For research or comparison, you can finetune from earlier layers:

```python
from alphagenome_ft.embeddings_extended import ExtendedEmbeddings
from alphagenome_ft.custom_forward import forward_with_encoder_output

# Custom head that uses encoder output (before transformer)
class EncoderOnlyHead(CustomHead):
    def predict(self, embeddings, organism_index, **kwargs):
        # Access raw encoder output (CNN features, no attention)
        x = embeddings.encoder_output  # (B, S/128, D)
        # ... your prediction layers ...
```

**Available representations:**
- `embeddings_1bp` - Decoder output (local + global context via skip connections)
- `embeddings_128bp` - Transformer output (global attention context)  
- `encoder_output`* - Encoder output (pure CNN features, before transformer)

*Requires using `ExtendedEmbeddings` and custom forward pass. See `alphagenome_ft/custom_forward.py` for implementation.

## API Reference

### Custom Head Base Class

```python
class CustomHead:
    def predict(self, embeddings, organism_index, **kwargs):
        """Generate predictions from embeddings.
        
        Args:
            embeddings: Multi-resolution embeddings object with methods:
                - get_sequence_embeddings(resolution=1)   # 1bp resolution
                - get_sequence_embeddings(resolution=128) # 128bp resolution
                - get_pair_embeddings()                   # Pairwise embeddings
            organism_index: (B,) array of organism indices
            
        Returns:
            dict: Must contain 'predictions' key
        """
        raise NotImplementedError
    
    def loss(self, predictions, batch):
        """Compute loss for predictions.
        
        Args:
            predictions: Output from predict()
            batch: dict with 'targets' and other training data
            
        Returns:
            dict: Must contain 'loss' key, can include other metrics
        """
        raise NotImplementedError
```

### Parameter Management

```python
# Freeze/unfreeze by path or prefix
model.freeze_parameters(freeze_paths=['alphagenome/encoder/...'])
model.unfreeze_parameters(unfreeze_prefixes=['alphagenome/head/'])

# Convenient presets
model.freeze_backbone()                    # Freeze encoder/transformer/decoder
model.freeze_all_heads(except_heads=['my_head'])  # Freeze all heads except one
model.freeze_except_head('my_head')        # Freeze everything except one head

# Inspection
paths = model.get_parameter_paths()        # All parameter paths
head_paths = model.get_head_parameter_paths()     # Just head parameters
backbone_paths = model.get_backbone_parameter_paths()  # Just backbone
count = model.count_parameters()           # Total parameter count
```

### Head Registration

```python
from alphagenome_ft import (
    register_custom_head,
    is_custom_head,
    get_custom_head_config,
    list_custom_heads,
)

# Register
register_custom_head('my_head', MyHeadClass, head_config)

# Query
if is_custom_head('my_head'):
    config = get_custom_head_config('my_head')

# List all
all_heads = list_custom_heads()
```

## Testing

The package includes a comprehensive test suite using pytest.

### Run Tests

```bash
# Install test dependencies
pip install -e ".[test]"

# Run all tests
pytest

# Run with coverage
pytest --cov=alphagenome_ft --cov-report=html

# Run specific test file
pytest tests/test_custom_heads.py
```

See [`tests/README.md`](tests/README.md) for detailed testing documentation.

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. **Add tests for new functionality** (see [`tests/README.md`](tests/README.md))
4. Ensure tests pass: `pytest`
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

This project extends [AlphaGenome](https://github.com/google-deepmind/alphagenome_research/), which has its own license terms.

