# AlphaGenome Finetuning (`alphagenome-ft`)

A lightweight Python package for finetuning [Google DeepMind's AlphaGenome](https://github.com/google-deepmind/alphagenome_research/) model with custom prediction heads and parameter freezing capabilities, **without modifying the original codebase**.

## Features

- **Custom Prediction Heads**: Define and register your own task-specific prediction heads
- **Parameter Freezing**: Flexible parameter management (freeze backbone, heads, or specific layers)
- **Easy Integration**: Works seamlessly with pretrained AlphaGenome models (Simple wrapper classes)
- **Parameter Inspection**: Utilities to explore and count model parameters
- **Attribution Analysis**: Utilities to calculate attributions based on gradients or _in silico_ mutagenesis (ISM)
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
There are three options to add new heads to AlphaGenome.

### Option A: Use a Predefined AlphaGenome Head

You can reuse the predefined head kinds from `alphagenome_research` without importing
`HeadName` by passing the string value. Supported strings are:
`atac`, `dnase`, `procap`, `cage`, `rna_seq`, `chip_tf`, `chip_histone`, `contact_maps`,
`splice_sites_classification`, `splice_sites_usage`, `splice_sites_junction`. For the details of each head, refer to [alphagenome_research](https://github.com/google-deepmind/alphagenome_research/blob/main/src/alphagenome_research/model/heads.py).

```python
from alphagenome_ft import (
    get_predefined_head_config,
    register_predefined_head,
    create_model_with_heads,
)

# 1. Build a predefined head config (num_tracks must match the number of your target tracks)
rna_config = get_predefined_head_config(
    "rna_seq",
    num_tracks=4,
)

# 2. Register it under an instance name you will train
register_predefined_head("K562_rna_seq", rna_config)

# 3. Create a model that uses the registered instance
model = create_model_with_heads("all_folds", heads=["K562_rna_seq"])
model.freeze_except_head("K562_rna_seq")
```

Note if you have a local AlphaGenome weights version you want to use instead of getting the weights from Kaggle use:

```python
model = create_model_with_heads(
    'all_folds',
    heads=['my_head'],
    checkpoint_path="full/path/to/weights",
)
```

### Option B: Use Template Heads

We provide ready-to-use template heads for common scenarios, see `./alphagenome_ft/templates.py`:

```python
from alphagenome.models import dna_output
from alphagenome_research.model import dna_model
from alphagenome_ft import (
    templates,
    CustomHeadConfig,
    CustomHeadType,
    register_custom_head,
    create_model_with_heads,
)

# 1. Register a template head (modify class for your task)
register_custom_head(
    'my_head',
    templates.StandardHead,  # Choose template: StandardHead, TransformerHead, EncoderOnlyHead
    CustomHeadConfig(
        type=CustomHeadType.GENOME_TRACKS,
        output_type=dna_output.OutputType.RNA_SEQ,
        num_tracks=1,
    )
)

# 2. Create model with custom head
model = create_model_with_heads(
    'all_folds',
    heads=['my_head'],
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

### Option C: Define Custom Head from Scratch

Define your own head when you need specific head architectures and/or loss:

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

# Register and use in the same way as Option B
```

### Note: Add Custom Head to Existing Model (Keep pre-trained Heads)

The three approaches above create models that include only the heads you explicitly provide. If you want to **keep AlphaGenome's pre-trained heads** (ATAC, RNA-seq, etc.) alongside your custom head:

```python
from alphagenome.models import dna_output
from alphagenome_research.model import dna_model
from alphagenome_ft import (
    templates,
    CustomHeadConfig,
    CustomHeadType,
    register_custom_head,
    add_heads_to_model,
)

# 1. Load pretrained model (includes standard heads)
base_model = dna_model.create_from_kaggle('all_folds')

# 2. Register custom or predefined head
register_custom_head(
    'my_head',
    templates.StandardHead,
    CustomHeadConfig(
        type=CustomHeadType.GENOME_TRACKS,
        output_type=dna_output.OutputType.RNA_SEQ,
        num_tracks=1,
    )
)

# 3. Add custom head to model (keeps ALL standard heads)
model = add_heads_to_model(base_model, heads=['my_head'])

# 4. Freeze backbone + standard heads, train only custom head
model.freeze_except_head('my_head')

# 5. Create loss function for training (see "Training with Custom Heads" section)
loss_fn = model.create_loss_fn_for_head('my_head')
```

**When to use each approach:**
- `create_model_with_heads()` - Heads **only** (faster, smaller)
- `add_heads_to_model()` - Added heads **+ pre-trained heads** (useful when referring to the original tracks)


## AlphaGenome Architecture

Understanding the architecture helps design custom heads:

```
DNA Sequence (B, L, 4)
    ↓
┌─────────────────────────────────────┐
│ BACKBONE (can be frozen)            │
│  ├─ SequenceEncoder ←────────────┐  │
│  ├─ TransformerTower (9 blocks)  │  │
│  └─ SequenceDecoder              │  │
└──────────────────────────────────┼──┘
    ↓                              │
┌──────────────────────────────────┼─────────────┐
│ EMBEDDINGS (multi-resolution)    │             │
│  ├─ embeddings_1bp:   (B, L, 1536)             │
│  ├─ embeddings_128bp: (B, L/128, 3072)         │
│  ├─ embeddings_pair:  (B, L/2048, S/2048, 128) │
│  └─ encoder_output:   (B, L/128, D)            │
└────────────────────────────────────────────────┘
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

model = create_model_with_heads('all_folds', heads=['my_head'])
```

### Type 2: Transformer Head (128bp Resolution)
**When to use:**
- Lower-resolution tasks (100+ bp scale)
- Need global attention context
- Want pure transformer features without decoder

**Example:** Gene expression, chromatin state, TAD boundaries

```python
register_custom_head('my_head', templates.TransformerHead, config)
model = create_model_with_heads('all_folds', heads=['my_head'])
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
model = create_model_with_heads(
    'all_folds',
    heads=['my_head'],
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

### Example 1: ChIP-seq Peak Prediction

```python
from alphagenome_ft import (
    get_predefined_head_config,
    register_predefined_head,
    create_model_with_heads,
)

# Build and register a predefined ChIP head (num_tracks must match targets)
chip_config = get_predefined_head_config("chip_tf", num_tracks=1)
register_predefined_head("my_chipseq_head", chip_config)

# Create model
model = create_model_with_heads('all_folds', heads=['my_chipseq_head'])
model.freeze_except_head('my_chipseq_head')

# Now train on your ChIP-seq data!
```

### Example 2: Gene Expression Prediction

```python
from alphagenome_ft import (
    get_predefined_head_config,
    register_predefined_head,
    create_model_with_heads,
)

# Build and register a predefined RNA-seq head (num_tracks must match targets)
rna_config = get_predefined_head_config("rna_seq", num_tracks=1)
register_predefined_head("my_expression_head", rna_config)

model = create_model_with_heads('all_folds', heads=['my_expression_head'])
model.freeze_except_head('my_expression_head')
```

### Example 3: MPRA Activity Prediction (Encoder-Only, Short Sequences)

```python
# For SHORT sequences (e.g., 200-500 bp MPRA constructs)
register_custom_head(
    'mpra_head',
    templates.EncoderOnlyHead,  # Encoder only (no transformer/decoder)
    CustomHeadConfig(
        type=CustomHeadType.GENOME_TRACKS,
        output_type=dna_output.OutputType.RNA_SEQ,
        num_tracks=1,
    )
)

# CRITICAL: use_encoder_output=True for short sequences
model = create_model_with_heads(
    'all_folds',
    heads=['mpra_head'],
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
model = create_model_with_heads('all_folds', heads=['attention_head'])
```

## API Reference

### Custom Head Base Class

```python
# alphagenome_ft/custom_heads.py
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
            PyTree: Any prediction structure consumed by your loss function
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
model.freeze_backbone()                    # Freeze all backbone components (default)
model.freeze_backbone(freeze_prefixes=['sequence_encoder'])  # Freeze only encoder
model.freeze_backbone(freeze_prefixes=['transformer_tower'])  # Freeze only transformer
model.freeze_backbone(freeze_prefixes=['sequence_decoder'])  # Freeze only decoder
model.freeze_backbone(freeze_prefixes=['sequence_encoder', 'transformer_tower'])  # Freeze encoder + transformer
model.freeze_all_heads(except_heads=['my_head'])  # Freeze all heads except one
model.freeze_except_head('my_head')        # Freeze everything except one head

# Inspection
paths = model.get_parameter_paths()        # All parameter paths
head_paths = model.get_head_parameter_paths()     # Just head parameters
backbone_paths = model.get_backbone_parameter_paths()  # Just backbone
count = model.count_parameters()           # Total parameter count
```

**Modular Backbone Freezing:**

The `freeze_backbone()` method now supports modular freezing of backbone components. This allows you to selectively freeze only specific parts of the backbone (encoder, transformer, or decoder) while keeping others trainable. This is useful for progressive finetuning strategies:

```python
# Example: Progressive finetuning strategy
# 1. Start with only head trainable
model.freeze_backbone()  # Freeze all backbone

# 2. Unfreeze decoder for fine-grained adaptation
model.unfreeze_parameters(unfreeze_prefixes=['sequence_decoder'])

# 3. Later, unfreeze transformer for global context adaptation
model.unfreeze_parameters(unfreeze_prefixes=['transformer_tower'])

# 4. Finally, unfreeze encoder for full finetuning
model.unfreeze_parameters(unfreeze_prefixes=['sequence_encoder'])

# Or use freeze_backbone with specific prefixes from the start:
model.freeze_backbone(freeze_prefixes=['sequence_encoder', 'transformer_tower'])  # Only decoder trainable
```

### Training with Custom Heads

When training your model, you need to compute loss using your custom head's `loss()` method. Since heads are Haiku modules that can only be instantiated within transforms, use the `create_loss_fn_for_head()` method:

```python
import jax
import jax.numpy as jnp

# Create model with custom head
model = create_model_with_heads('all_folds', heads=['my_head'])
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
    get_predefined_head_config,
    register_predefined_head,
    register_custom_head,
    is_head_registered,
    get_registered_head_config,
    list_custom_heads,
)

# Register predefined head
head_config = get_predefined_head_config(
    "atac",
    num_tracks=4,
)
register_predefined_head("my_head", head_config)

# OR register custom head
register_custom_head('my_head', MyHeadClass, head_config)


# Query
if is_head_registered('my_head'):
    config = get_registered_head_config('my_head')

# List registered custom heads
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

## Attribution Analysis

After training, you can compute attributions to understand which sequence features drive your model's predictions. The package supports multiple attribution methods:

- **DeepSHAP**: Reference-based attribution using gradient differences
- **Gradient × Input**: Gradient multiplied by input (standard gradient-based attribution)
- **Gradient**: Raw gradients (information content)
- **ISM** (_in silico_ mutagenesis): Importance scores from single-nucleotide variants

### Basic Attribution Example

```python
import jax.numpy as jnp
from alphagenome_ft import load_checkpoint

# Load trained model
model = load_checkpoint('checkpoints/my_model', base_model_version='all_folds')

# Prepare a sequence (one-hot encoded, shape: (batch, length, 4))
sequence = jnp.array([...])  # Your sequence here
organism_index = jnp.array([0])  # Organism index (0 = human)

# Compute DeepSHAP attributions
attributions = model.compute_deepshap_attributions(
    sequence=sequence,
    organism_index=organism_index,
    head_name='my_head',
    n_references=20,  # Number of reference sequences
    reference_type='shuffle',  # 'shuffle', 'uniform', or 'gc_match'
    random_state=42,
)

# Attributions shape: (batch, seq_len, 4) - one score per base per position
print(f"Attributions shape: {attributions.shape}")

# Alternative: Gradient × Input attributions
attributions_grad = model.compute_input_gradients(
    sequence=sequence,
    organism_index=organism_index,
    head_name='my_head',
    gradients_x_input=True,  # Multiply gradient by input
)

# Alternative: ISM attributions (wildtype-base importance)
attributions_ism = model.compute_ism_attributions(
    sequence=sequence,
    organism_index=organism_index,
    head_name='my_head',
)
```

### Visualization

Generate attribution maps and sequence logos:

```python
# Decode sequence to string for visualization
sequence_str = "ATCGATCG..."  # Your sequence as string

# Plot attribution heatmap
model.plot_attribution_map(
    sequence=sequence,
    gradients=attributions,  # Attributions from above
    sequence_str=sequence_str,
    save_path='attribution_map.png'
)

# Plot sequence logo (shows base preferences)
model.plot_sequence_logo(
    sequence=sequence,
    gradients=attributions,
    save_path='sequence_logo.png',
    logo_type='weight',  # 'weight' for raw scores, 'information' for bits
    mask_to_sequence=False,  # Show all bases or only input sequence
    use_absolute=False,  # Preserve signed values
)
```

### Complete Example: Analyzing a Single Sequence

```python
import jax.numpy as jnp
from alphagenome_ft import load_checkpoint

# Helper function to encode DNA sequence (AlphaGenome order: A, C, G, T)
def one_hot_encode(sequence: str) -> jnp.ndarray:
    """Convert DNA sequence string to one-hot encoding."""
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_len = len(sequence)
    one_hot = jnp.zeros((1, seq_len, 4), dtype=jnp.float32)
    for i, base in enumerate(sequence.upper()):
        if base in base_map:
            one_hot = one_hot.at[0, i, base_map[base]].set(1.0)
    return one_hot

# 1. Load model
model = load_checkpoint('checkpoints/my_model', base_model_version='all_folds')

# 2. Prepare sequence
sequence_str = "ATCGATCGATCG..."  # Your DNA sequence
sequence_onehot = one_hot_encode(sequence_str)  # Convert to one-hot
organism_index = jnp.array([0])  # Human

# 3. Compute attributions
attributions = model.compute_deepshap_attributions(
    sequence=sequence_onehot,
    organism_index=organism_index,
    head_name='my_head',
    n_references=20,
)

# 4. Visualize
model.plot_attribution_map(
    sequence=sequence_onehot,
    gradients=attributions,
    sequence_str=sequence_str,
    save_path='attribution_map.png'
)

model.plot_sequence_logo(
    sequence=sequence_onehot,
    gradients=attributions,
    save_path='sequence_logo.png',
    logo_type='weight',
    mask_to_sequence=False,
)
```

### Attribution Methods Comparison

| Method | Use Case | Notes |
|--------|----------|-------|
| **DeepSHAP** | General-purpose, robust | Reference-based, handles non-linearities well |
| **Gradient × Input** | Fast, interpretable | Standard gradient-based method |
| **Gradient** | Information content | Shows relative importance, not signed |
| **ISM** | Variant effect prediction | Computes importance from SNP effects |

**Note:** For multi-track outputs, use `output_index` to specify which track to attribute:

```python
# For a head with 2 tracks, attribute to track 0
attributions = model.compute_deepshap_attributions(
    sequence=sequence,
    organism_index=organism_index,
    head_name='my_head',
    output_index=0,  # Attribute to first track
)
```

For complete examples with motif analysis, background shuffling, and batch processing, see the [MPRA attribution script](../alphagenome_FT_MPRA/scripts/compute_attributions.py) and [DeepSTARR attribution script](../alphagenome_FT_MPRA/scripts/compute_attributions_starrseq.py).

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
