## Tutorial: Full-Model Finetuning (Unfreezing the Backbone)

This tutorial shows how to **finetune the entire AlphaGenome model**, including the backbone:

- **Backbone** (encoder, transformer, decoder): partially or fully trainable
- **Heads**: trainable

This is more expressive but also more compute‑intensive than heads‑only finetuning. It is useful when:
- Your data distribution is very different from AlphaGenome’s pretraining data.
- You need to adapt **global context** (transformer) or **local features** (encoder/decoder).

### 1. Start from a Heads-Only Setup

We recommend starting from a working **heads-only pipeline** (see the `Frozen Backbone, New Head` tutorial) and then gradually unfreezing backbone parts.

Assume you already have:

- A `targets_*.yaml` file describing your heads.
- A BED/BED.gz file with intervals.
- A `BigWigDataModule` and `head_specs`.

### 2. Create a Model with Your Heads

```python
from alphagenome_ft import create_model_with_heads

head_ids = [spec.head_id for spec in head_specs]

model = create_model_with_heads(
    "all_folds",
    heads=head_ids,
)
```

At this point, the model has:
- The full AlphaGenome backbone.
- Only the heads you requested in `heads=head_ids`.

**Caution about `all_folds` and benchmarks:** `"all_folds"` uses the AlphaGenome distilled model trained on all four genomic folds. This is ideal for practical finetuning, but if your task’s train/valid/test splits overlap AlphaGenome’s (and Borzoi’s) held‑out regions, you can get **data leakage and inflated test performance**. If you care about strict benchmarking that matches the splits used in [AlphaGenome](https://www.nature.com/articles/s41586-025-10014-0) and [Borzoi](https://www.nature.com/articles/s41588-024-02053-6), load one of the fold‑specific AlphaGenome models instead and align your train/valid/test partitions to those folds.

### 3. Choose a Finetuning Strategy

You have several options for how much of the backbone to train (with increasing compute requirements):

- **Strategy A: Heads only** (baseline)
  - Freeze encoder, transformer, and decoder.
- **Strategy B: Decoder-only**
  - Train the decoder, keep encoder + transformer frozen.
- **Strategy C: Transformer+Decoder**
  - Train transformer and decoder, keep encoder frozen.
- **Strategy D: Full backbone**
  - Train encoder, transformer, and decoder.

These are all built on `parameter_utils.freeze_backbone` and `unfreeze_parameters`.

### 4. Progressive Unfreezing Example

You can use **progressive unfreezing**: start with heads-only, then gradually unfreeze deeper parts of the network.

```python
from alphagenome_ft import parameter_utils

# Step 1: start with heads-only (backbone frozen)
model._params = parameter_utils.freeze_backbone(model._params)

# Optionally also freeze all heads except one, e.g.:
# model._params = parameter_utils.freeze_all_heads(model._params, except_heads=["K562_rna_seq"])

# Train with heads_only=True first (see Section 6).
```

Later, you can **unfreeze** more components:

```python
# Step 2: unfreeze decoder for fine-grained adaptation
model._params = parameter_utils.unfreeze_parameters(
    model._params,
    unfreeze_prefixes=["sequence_decoder"],
)

# Step 3: unfreeze transformer for global context adaptation
model._params = parameter_utils.unfreeze_parameters(
    model._params,
    unfreeze_prefixes=["transformer_tower"],
)

# Step 4: unfreeze encoder for full finetuning
model._params = parameter_utils.unfreeze_parameters(
    model._params,
    unfreeze_prefixes=["sequence_encoder"],
)
```

Alternatively, you can **freeze only specific parts** from the start:

```python
# Train decoder only
model._params = parameter_utils.freeze_backbone(
    model._params,
    freeze_prefixes=["sequence_encoder", "transformer_tower"],
)
```

### 5. Optimizer Setup for Full Finetuning

When you want to update backbone parameters, call `train` with `heads_only=False`. This uses a single AdamW optimizer over all parameters:

```python
from alphagenome_ft.finetune import train
from pathlib import Path

checkpoint_dir = Path("checkpoints/full_finetune")

train(
    model=model,
    data_module=data_module,
    head_specs=head_specs,
    learning_rate=1e-4,        # usually lower LR than heads-only
    weight_decay=1e-5,
    num_epochs=10,
    heads_only=False,          # ← allow backbone updates
    checkpoint_dir=checkpoint_dir,
    organism="HOMO_SAPIENS",
    best_metric="valid_loss",
    best_metric_mode="min",
    early_stopping_patience=2,
    early_stopping_min_delta=0.0,
    verbose=True,
)
```

Notes:
- For full-model finetuning, a **smaller learning rate** is typically safer.
- Checkpoints may be larger if you choose to save the full model (see below).

### 6. Saving Full-Model Checkpoints

By default, `model.save_checkpoint(..., save_full_model=False)` only saves **head parameters** for efficiency.

For full-model finetuning, you may want to save the **entire model**:

```python
model.save_checkpoint(
    "checkpoints/full_finetune/best_full",
    save_full_model=True,
)
```

This produces a self‑contained checkpoint that does not require reloading the base model from Kaggle.

### 7. Sequence Lengths (32k / 64k / 128k / 1M)

The same windowing logic from heads-only finetuning applies:

- Use the `window_size` argument in `prepare_intervals_from_split` to control sequence length.
- Typical choices:
  - `32768` (32k)
  - `65536` (64k)
  - `131072` (128k)
  - `1048576` (~1 Mbp)

Example:

```python
intervals = prepare_intervals_from_split(
    bed_path=Path("data/intervals_chr22.bed.gz"),
    window_size=65536,   # 64k windows
)

data_module = BigWigDataModule(
    intervals=intervals,
    fasta_path=Path("data/hg38.fa"),
    head_specs=head_specs,
    batch_size=2,
    shuffle=True,
)
```

### 8. When to Use Full-Model Finetuning

Consider full-model finetuning when:
- Your assay/organism/domain is far from AlphaGenome’s pretraining distribution.
- You see **systematic errors** that suggest the backbone features need to adapt.
- You have sufficient data and compute to avoid overfitting.

If you only want lightweight adaptation on top of a frozen backbone (e.g. for very small datasets), consider **LoRA-style adapters** (see the LoRA tutorial) or stick with heads-only finetuning.
