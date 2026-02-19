# AlphaGenome Finetuning (`alphagenome-ft`)

A lightweight Python package for finetuning [Google DeepMind's AlphaGenome](https://github.com/google-deepmind/alphagenome_research/) model with custom prediction heads and parameter freezing capabilities, **without modifying the original codebase**.

**Project leads - [Alan Murphy](https://al-murphy.github.io/), [Masayuki (Moon) Nagai](https://masayukinagai.github.io/), [Alejandro Buendia](https://abuendia.github.io/)**

## Use cases

- If you want to apply AlphaGenome to your MPRA (or other perturbation) data of interest, see [Encoder-only / short sequences (MPRA)](#workflow-3-encoder-only--short-sequences-mpra).
- If you want to apply AlphaGenome to your own genome-wide assay, start with [Heads-only finetuning (frozen backbone)](#workflow-1-heads-only-finetuning-frozen-backbone); then [LoRA-style adapters](#workflow-4-lora-style-adapters) or [Full-model finetuning](#workflow-2-full-model-finetuning) if needed.

## Contents

- **Overview**
  - [Features](#features)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
- **Workflows** (in recommended run order)
  1. [Heads-only finetuning (frozen backbone)](#workflow-1-heads-only-finetuning-frozen-backbone) — train a new head on top of a frozen model
  2. [Full-model finetuning](#workflow-2-full-model-finetuning) — unfreeze the backbone (e.g. progressive unfreezing)
  3. [Encoder-only / short sequences (MPRA)](#workflow-3-encoder-only--short-sequences-mpra) — finetune on short sequences (&lt; 1 kb)
  4. [LoRA-style adapters](#workflow-4-lora-style-adapters) — low-rank adapter layers
  5. [Attribution analysis](#after-training-attribution-analysis) — interpret predictions after training
- **Reference**
  - [AlphaGenome Architecture](#alphagenome-architecture)
  - [Head types and embeddings](#custom-head-types-guide)
  - [Add head / custom head from scratch](#add-head-to-existing-model-keep-standard-heads)
  - [API Reference](#api-reference)
  - [Saving and loading checkpoints](#saving-and-loading-checkpoints)
  - [Attribution analysis (full detail)](#attribution-analysis)
- **Other**
  - [Testing](#testing)
  - [Contributing](#contributing)

**How to use this README:** Sections are ordered by how you typically run things. Start with adapting to MPRA or heads-only finetuning; then add full-model workflows if needed. Run attribution and interpretation **after** you have a trained model. Step-by-step tutorials live in [`docs/`](docs/).

## Features

- **Custom Prediction Heads**: Define and register your own task-specific prediction heads
- **Parameter Freezing**: Flexible parameter management (freeze backbone, heads, or specific layers)
- **Easy Integration**: Works seamlessly with pretrained AlphaGenome models (simple wrapper classes)
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
git clone https://github.com/genomicsxai/alphagenome_ft.git
cd alphagenome_ft
pip install -e .
```

### Requirements

- Python ≥ 3.10
- JAX ≥ 0.4.0
- Haiku ≥ 0.0.10
- AlphaGenome Research (install from [GitHub](https://github.com/google-deepmind/alphagenome_research/))
- Core Python dependencies (see `pyproject.toml` for exact versions):
  - `aiohttp`
  - `optax`
  - `requests`

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


## Workflows

### Workflow 1: Encoder-only / short sequences (MPRA)

**When to use:** Short sequences (&lt; ~1 kb): MPRA, promoters, enhancers. Uses encoder (CNN) only.  
**Tutorial:** [Encoder-only finetuning](docs/encoder_only_perturbation.md). Use `templates.EncoderOnlyHead` and **`use_encoder_output=True`** in `create_model_with_heads(...)`. Run zero-shot after training; see [MPRA repo](https://github.com/genomicsxai/alphagenome_FT_MPRA).

---

### Workflow 2: Heads-only finetuning (frozen backbone)

**When to use:** New task (ChIP-seq, gene expression, etc.) on standard-length sequences; train only a new head, backbone frozen.  
**Tutorial:** [Frozen backbone, new head](docs/frozen_backbone_new_head.md).

---

### Workflow 3: LoRA-style adapters

**When to use:** Low-rank adapters on the backbone. **Tutorial:** [LoRA-style adapters](docs/lora_adapters.md).

---

### Workflow 4: Full-model finetuning

**When to use:** Adapt the backbone (e.g. after heads-only or for a different distribution).  
**Tutorial:** [Full-model finetuning (unfreezing the backbone)](docs/full_model_finetuning.md). Unfreeze via `unfreeze_parameters(unfreeze_prefixes=[...])` or `freeze_backbone(freeze_prefixes=[...])`; save with `save_checkpoint(..., save_full_model=True)`.

---

### After training: Attribution analysis

Compute attributions after training to see which sequence features drive predictions.  
**Methods:** DeepSHAP*, Gradient × Input, Gradient, ISM.  
Load a checkpoint, then use `compute_deepshap_attributions`, `compute_input_gradients`, or `compute_ism_attributions`; visualize with `plot_attribution_map` and `plot_sequence_logo`.

Full API, examples (basic, visualization, single-sequence pipeline), method comparison, and multi-track `output_index`: **[Attribution analysis](docs/attribution.md)**.

**NOTE**: DeepSHAP* - The implementation is DeepSHAP-like in that it uses a reference sequence but is note a faithful reimplmenntation.

---

## References

### AlphaGenome Architecture

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
┌──────────────────────────────────┼─────────────┐
│ EMBEDDINGS (multi-resolution)    │             │
│  ├─ embeddings_1bp:   (B, S, 1536)             │
│  ├─ embeddings_128bp: (B, S/128, 3072)         │
│  ├─ embeddings_pair:  (B, S/2048, S/2048, 128) │
│  └─ encoder_output:   (B, S/128, D)            │
└────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ HEADS (task-specific)               │
│  ├─ Standard: ATAC, RNA-seq, etc.   │
│  └─ Custom: YOUR_HEAD_HERE ← Add!   │
└─────────────────────────────────────┘
```

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

### Modular Backbone Freezing

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

### Saving a Checkpoint

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
