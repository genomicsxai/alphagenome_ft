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

### Using Template Heads (Recommended for starting out)

We provide ready-to-use template heads for common scenarios:

```python
from alphagenome.models import dna_output
from alphagenome_research.model import dna_model
from alphagenome_ft import (
    templates,
    HeadConfig,
    HeadType,
    register_custom_head,
    create_model_with_custom_heads,
)

# 1. Register a template head (modify class for your task)
register_custom_head(
    'my_head',
    templates.StandardHead,  # Choose template: StandardHead, TransformerHead, EncoderOnlyHead
    HeadConfig(
        type=HeadType.GENOME_TRACKS,
        name='my_head',
        output_type=dna_output.OutputType.RNA_SEQ,
        num_tracks=1,
    )
)

# 2. Create model with custom head
model = create_model_with_custom_heads(
    'all_folds',
    custom_heads=['my_head'],
)

# 3. Freeze backbone for finetuning
model.freeze_except_head('my_head')
```

**Available Templates:**
- `templates.StandardHead` - Uses 1bp embeddings (decoder output: local + global features)
- `templates.TransformerHead` - Uses 128bp embeddings (transformer output: pure attention)
- `templates.EncoderOnlyHead` - Uses encoder output (CNN only, for short sequences < 1kb)

All templates use simple architecture: **Linear → ReLU → Linear**

The key difference is **which embeddings** they access. See [`alphagenome_ft/templates.py`](alphagenome_ft/templates.py) for code.

### Custom Head from Scratch

Define your own head for specialized tasks:

```python
import jax
import jax.numpy as jnp
import haiku as hk
from alphagenome_ft import CustomHead

class MyCustomHead(CustomHead):
    """Your custom prediction head."""
    
    def predict(self, embeddings, organism_index, **kwargs):
        # Get embeddings at desired resolution
        x = embeddings.get_sequence_embeddings(resolution=1)  # or 128
        
        # Add your prediction layers
        x = hk.Linear(256, name='hidden')(x)
        x = jax.nn.relu(x)
        predictions = hk.Linear(self._num_tracks, name='output')(x)
        
        return predictions
    
    def loss(self, predictions, batch):
        targets = batch.get('targets')
        if targets is None:
            return {'loss': jnp.array(0.0)}
        
        mse = jnp.mean((predictions - targets) ** 2)
        return {'loss': mse, 'mse': mse}

# Register and use as shown above
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

## Custom Head Types Guide

Choose the right head type for your task:

### Type 1: Standard Head (1bp Resolution)
**When to use:**
- High-resolution predictions (base-pair level)
- Tasks requiring both local and global context
- Standard-length sequences (> 100kb)

**Example:** ChIP-seq peaks, TF binding sites, variant effect prediction

```python
from alphagenome_ft import templates

class MyHead(templates.StandardHead):
    # Inherits 1bp resolution prediction
    pass

# Or use template directly:
register_custom_head('my_head', templates.StandardHead, config)

model = create_model_with_custom_heads('all_folds', custom_heads=['my_head'])
```

### Type 2: Transformer Head (128bp Resolution)
**When to use:**
- Lower-resolution tasks (100+ bp scale)
- Need global attention context
- Want pure transformer features without decoder

**Example:** Gene expression, chromatin state, TAD boundaries

```python
register_custom_head('my_head', templates.TransformerHead, config)
model = create_model_with_custom_heads('all_folds', custom_heads=['my_head'])
```

### Type 3: Encoder-Only Head (For Short Sequences)
**When to use:**
- **SHORT sequences** (< 1000 bp) - *won't fit through transformer*
- Want pure CNN features (no attention)
- MPRA, promoters, enhancers, motifs

**Example:** MPRA activity prediction, short regulatory elements

```python
register_custom_head('my_head', templates.EncoderOnlyHead, config)

# IMPORTANT: Must use use_encoder_output=True for short sequences
model = create_model_with_custom_heads(
    'all_folds',
    custom_heads=['my_head'],
    use_encoder_output=True,  # ← Required for encoder-only mode
)
```

**Why encoder-only mode?** Short sequences (< 1000 bp) cause shape mismatches in the transformer's attention mechanism. Encoder-only mode skips the transformer/decoder and uses only CNN features from the encoder.

## Head Type Comparison

| Head Type | Resolution | Architecture Used | Sequence Length | Embeddings Shape |
|-----------|-----------|-------------------|-----------------|------------------|
| StandardHead | 1bp | Encoder → Transformer → Decoder | > 100kb | (B, S, 1536) |
| TransformerHead | 128bp | Encoder → Transformer | > 100kb | (B, S/128, 3072) |
| EncoderOnlyHead | 128bp | Encoder only | **< 1kb** | (B, S/128, 1536) |

**All templates use the same simple architecture:** Linear(256) → ReLU → Linear(num_tracks)

### Available Embedding Resolutions

```python
# In your custom head's predict() method:

# 1bp resolution (highest detail, local + global features)
x_1bp = embeddings.get_sequence_embeddings(resolution=1)
# Shape: (batch, sequence_length, 1536)

# 128bp resolution (global attention features)
x_128bp = embeddings.get_sequence_embeddings(resolution=128)
# Shape: (batch, sequence_length//128, 3072)

# Encoder output (CNN features only, requires use_encoder_output=True)
x_encoder = embeddings.encoder_output
# Shape: (batch, sequence_length//128, D)
```

## Complete Examples

### Example 1: ChIP-seq Peak Prediction (Standard Head)

```python
from alphagenome.models import dna_output
from alphagenome_research.model import dna_model
from alphagenome_ft import templates, HeadConfig, HeadType, register_custom_head, create_model_with_custom_heads

# Register head
register_custom_head(
    'chipseq_head',
    templates.StandardHead,  # 1bp resolution
    HeadConfig(
        type=HeadType.GENOME_TRACKS,
        name='chipseq_head',
        output_type=dna_output.OutputType.ATAC,
        num_tracks=1,
    )
)

# Create model
model = create_model_with_custom_heads('all_folds', custom_heads=['chipseq_head'])
model.freeze_except_head('chipseq_head')

# Now train on your ChIP-seq data!
```

### Example 2: Gene Expression Prediction (Transformer Head)

```python
# Register head for gene expression (operates at 128bp resolution)
register_custom_head(
    'expression_head',
    templates.TransformerHead,  # 128bp resolution, global context
    HeadConfig(
        type=HeadType.GENOME_TRACKS,
        name='expression_head',
        output_type=dna_output.OutputType.RNA_SEQ,
        num_tracks=1,
    )
)

model = create_model_with_custom_heads('all_folds', custom_heads=['expression_head'])
model.freeze_except_head('expression_head')
```

### Example 3: MPRA Activity Prediction (Encoder-Only, Short Sequences)

```python
# For SHORT sequences (e.g., 200-500 bp MPRA constructs)
register_custom_head(
    'mpra_head',
    templates.EncoderOnlyHead,  # Encoder only (no transformer/decoder)
    HeadConfig(
        type=HeadType.GENOME_TRACKS,
        name='mpra_head',
        output_type=dna_output.OutputType.RNA_SEQ,
        num_tracks=1,
    )
)

# CRITICAL: use_encoder_output=True for short sequences
model = create_model_with_custom_heads(
    'all_folds',
    custom_heads=['mpra_head'],
    use_encoder_output=True,  # ← Skips transformer/decoder
)
model.freeze_except_head('mpra_head')

# Now you can process short sequences (< 1000 bp) without errors!
```

### Example 4: Custom Architecture

```python
import jax
import jax.numpy as jnp
import haiku as hk
from alphagenome_ft import CustomHead

class AttentionPoolingHead(CustomHead):
    """Custom head with attention-based pooling."""
    
    def predict(self, embeddings, organism_index, **kwargs):
        x = embeddings.get_sequence_embeddings(resolution=1)
        
        # Attention pooling
        query = hk.Linear(256, name='query')(x)
        key = hk.Linear(256, name='key')(x)
        value = hk.Linear(256, name='value')(x)
        
        # Compute attention scores
        scores = jnp.matmul(query, key.transpose((0, 2, 1)))
        scores = scores / jnp.sqrt(256)
        attn_weights = jax.nn.softmax(scores, axis=-1)
        
        # Apply attention
        x = jnp.matmul(attn_weights, value)
        
        # Output
        predictions = hk.Linear(self._num_tracks, name='output')(x)
        return predictions
    
    def loss(self, predictions, batch):
        targets = batch.get('targets')
        if targets is None:
            return {'loss': jnp.array(0.0)}
        mse = jnp.mean((predictions - targets) ** 2)
        return {'loss': mse, 'mse': mse}

# Register and use
register_custom_head('attention_head', AttentionPoolingHead, config)
model = create_model_with_custom_heads('all_folds', custom_heads=['attention_head'])
```

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

