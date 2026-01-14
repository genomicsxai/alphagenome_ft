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

### Alternative: Add Custom Head to Existing Model (Keep Standard Heads)

If you want to **keep the standard AlphaGenome heads** (ATAC, RNA-seq, etc.) and add your custom head:

```python
from alphagenome_research.model import dna_model
from alphagenome_ft import templates, HeadConfig, HeadType, register_custom_head, add_custom_heads_to_model

# 1. Load pretrained model (includes standard heads)
base_model = dna_model.create_from_kaggle('all_folds')

# 2. Register custom head
register_custom_head(
    'my_head',
    templates.StandardHead,
    HeadConfig(
        type=HeadType.GENOME_TRACKS,
        name='my_head',
        output_type=dna_output.OutputType.RNA_SEQ,
        num_tracks=1,
    )
)

# 3. Add custom head to model (keeps ALL standard heads)
model = add_custom_heads_to_model(base_model, custom_heads=['my_head'])

# 4. Freeze backbone + standard heads, train only custom head
model.freeze_except_head('my_head')

# 5. Create loss function for training (see "Training with Custom Heads" section)
loss_fn = model.create_loss_fn_for_head('my_head')
```

**When to use each approach:**
- `create_model_with_custom_heads()` - Custom heads **only** (faster, smaller)
- `add_custom_heads_to_model()` - Custom heads **+ standard heads** (useful for multi-task)

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

### Training with Custom Heads

When training your model, you need to compute loss using your custom head's `loss()` method. Since heads are Haiku modules that can only be instantiated within transforms, use the `create_loss_fn_for_head()` method:

```python
import jax
import jax.numpy as jnp

# Create model with custom head
model = create_model_with_custom_heads('all_folds', custom_heads=['my_head'])
model.freeze_except_head('my_head')

# Create a loss function for your head (do this ONCE before training)
loss_fn = model.create_loss_fn_for_head('my_head')

# Training loop
def train_step(model, batch, optimizer_state):
    """Single training step."""
    def loss_fn_inner(params):
        # Get model predictions
        predictions = model._predict(
            params,
            model._state,
            batch['sequences'],
            batch['organism_index'],
            negative_strand_mask=jnp.zeros(len(batch['sequences']), dtype=bool),
            strand_reindexing=model._metadata[Organism.HOMO_SAPIENS].strand_reindexing,
        )
        
        # Get predictions for our custom head
        head_predictions = predictions['my_head']
        
        # Compute loss using the head's loss function
        loss_dict = loss_fn(head_predictions, batch)  # ← Uses head's loss() method
        return loss_dict['loss']
    
    # Compute gradients
    loss, grads = jax.value_and_grad(loss_fn_inner)(model._params)
    
    # Update parameters with optimizer
    updates, optimizer_state = optimizer.update(grads, optimizer_state)
    model._params = optax.apply_updates(model._params, updates)
    
    return loss, optimizer_state

# Train
for epoch in range(num_epochs):
    for batch in dataloader:
        loss, optimizer_state = train_step(model, batch, optimizer_state)
        print(f"Loss: {loss}")
```

**Key points:**
- Call `model.create_loss_fn_for_head(head_name)` **once** before training to get a reusable loss function
- The returned `loss_fn` automatically instantiates the head within a transform and calls its `loss()` method
- Use this `loss_fn` inside your gradient computation

For a complete working example with gradient accumulation and Weights & Biases integration, see the [MPRA finetuning example](../alphagenome_FT_MPRA/src/README.md).

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

## Saving and Loading Checkpoints

After training, save your custom head parameters for later use.

### Saving a Checkpoint

```python
# After training your model
model.save_checkpoint(
    'checkpoints/my_model',
    save_full_model=False  # Default: only save custom heads (efficient)
)
```

**Options:**
- `save_full_model=False` (default): Only saves custom head parameters (~MBs)
  - **Recommended for finetuning** - much smaller checkpoints
  - Requires loading the base model when restoring
- `save_full_model=True`: Saves entire model including backbone (~GBs)
  - Self-contained checkpoint
  - Larger file size but no need for base model
  - Use if unfreezing the base model

### Loading a Checkpoint

```python
from alphagenome_ft import load_checkpoint

# Load a heads-only checkpoint (requires base model)
model = load_checkpoint(
    'checkpoints/my_model',
    base_model_version='all_folds'  # Which base model to use
)

# Now use for inference or continue training
predictions = model.predict(...)
```

**Important:** Before loading, you must register the custom head classes:

```python
# Import and register your custom head class
from your_module import MyCustomHead
from alphagenome_ft import register_custom_head

register_custom_head('my_head', MyCustomHead, config)

# Now load checkpoint
model = load_checkpoint('checkpoints/my_model')
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
pytest tests/test_checkpoint.py
```

### Test Categories

- **`test_custom_heads.py`** - Custom head registration and configuration
- **`test_model_predictions.py`** - Model prediction consistency
- **`test_encoder_only_mode.py`** - Encoder-only mode for short sequences
- **`test_parameter_management.py`** - Parameter freezing and inspection
- **`test_checkpoint.py`** - Checkpoint save/load functionality

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

