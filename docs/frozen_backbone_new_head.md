## Tutorial: Frozen Backbone, New Head (Heads-Only Finetuning)

This tutorial shows how to finetune **only a new prediction head** on top of a **frozen AlphaGenome backbone**, using `alphagenome-ft`.

This is the most common and compute-efficient setup:
- **Backbone** (encoder, transformer, decoder): frozen
- **New head**: trainable

You can use this for:
- Adding a new track (e.g. new ChIP-seq / ATAC-seq / RNA-seq track)
- Training a custom head for your own target (e.g. regression/classification task)

### 1. Install Dependencies

Make sure you have AlphaGenome and `alphagenome-ft` installed:

```bash
pip install git+https://github.com/google-deepmind/alphagenome_research.git
pip install alphagenome-ft  # or install from source in this repo
```

### 2. Define Your Targets (BigWig or Metadata)

For large-scale genomic finetuning, we recommend using **BigWig tracks** and a simple YAML/JSON config.

Example YAML (`targets_rna.yaml`):

```yaml
heads:
  - id: K562_rna_seq
    source: predefined
    kind: rna_seq
    targets:
      - bigwig: /path/to/K562_RNAseq_track1.bw
        label: K562_RNA_1
      - bigwig: /path/to/K562_RNAseq_track2.bw
        label: K562_RNA_2
```

- `id`: the **instance name** you will train (used in checkpoints, metrics, etc.).
- `source: predefined`: use AlphaGenome’s built‑in head templates.
- `kind`: one of `atac`, `dnase`, `procap`, `cage`, `rna_seq`, `chip_tf`, `chip_histone`, `contact_maps`, `splice_sites_classification`, `splice_sites_usage`, `splice_sites_junction`.
- `targets`: one entry per BigWig channel.

### 3. Create Genomic Intervals (Sequence Windows)

AlphaGenome works on up to **~1 Mbp** windows, but you can also train on shorter windows (e.g. 32k / 64k / 128k).

Create a BED (or BED.gz) file that defines your train/valid/test windows:

```text
chr22   35677410  36725986  train
chr22   36725986  37774562  train
chr22   37774562  38823138  valid
chr22   38823138  39871714  test
```

- Column 1–3: `chrom`, `start`, `end`
- Column 4: split label (`train`, `valid`, `test`)

You can optionally let `alphagenome-ft` **re-center windows** to a fixed **window size**:

- `window_size=None`: use the intervals as-is (e.g. 1 Mbp windows).
- `window_size=131072`: use 128k windows (centered on each interval).
- Other common sizes: `32768` (32k), `65536` (64k), `262144` (256k).

### 4. Build Heads and Data Module

Use the `alphagenome_ft.finetune` utilities to:
- Parse your targets config
- Load genomic intervals
- Create a `BigWigDataModule`

```python
from pathlib import Path

from alphagenome_ft import create_model_with_heads
from alphagenome_ft.finetune import (
    build_head_specs,
    validate_heads,
    load_intervals,
    BigWigDataModule,
    register_predefined_heads,
    train,
)

# 1. Parse target config and register predefined heads
targets_config = Path("configs/targets_rna.yaml")
head_specs = build_head_specs(targets_config, organism="HOMO_SAPIENS")
validate_heads(head_specs)
register_predefined_heads(head_specs)

# 2. Load genomic intervals and build data module
intervals = load_intervals(
    bed=Path("data/intervals_chr22.bed.gz"),
    window_size=131072,          # 128k windows (can be 32k, 64k, 1M, etc.)
)

data_module = BigWigDataModule(
    intervals=intervals,
    fasta_path=Path("data/hg38.fa"),
    head_specs=head_specs,
    batch_size=4,
    shuffle=True,
    window_size=131072,
)
```

### 5. Create Model with Frozen Backbone + New Head

Now create a model that **only includes the heads you want to train**, and freeze the backbone:

```python
# Extract head ids to train
head_ids = [spec.head_id for spec in head_specs]

# Create a model with only these heads
model = create_model_with_heads(
    "all_folds",
    heads=head_ids,
)

# Freeze everything except the new heads
for head_name in head_ids:
    model.freeze_except_head(head_name)
```

- Internally this uses the same AlphaGenome backbone, but keeps only your custom/predefined heads.
- `freeze_except_head` uses `parameter_utils` to freeze backbone + other heads.

### 6. Run Heads-Only Training

Use the built‑in training loop with `heads_only=True`:

```python
from pathlib import Path

checkpoint_dir = Path("checkpoints/rna_heads_only")

train(
    model=model,
    data_module=data_module,
    head_specs=head_specs,
    learning_rate=1e-3,
    weight_decay=1e-4,
    num_epochs=10,
    heads_only=True,        # ← ensures optimizer only updates head params
    checkpoint_dir=checkpoint_dir,
    organism="HOMO_SAPIENS",
    best_metric="valid_loss",
    best_metric_mode="min",
    early_stopping_patience=3,
    early_stopping_min_delta=0.0,
    verbose=True,
)
```

What this does:
- Freezes all backbone parameters.
- Optimizes only parameters under `head/<head_id>/...`.
- Streams windows from your BED file and BigWig tracks.
- Saves checkpoints with **head-only** parameters by default (small files).

### 7. Training on Different Sequence Lengths

To work with different sequence lengths (still below ~1 Mbp), adjust:

- `window_size` in `load_intervals` and `BigWigDataModule`.
- Your BED intervals if you want exact window boundaries.

Examples:

- **32k windows**:
  - `window_size=32768`
- **64k windows**:
  - `window_size=65536`
- **128k windows**:
  - `window_size=131072`
- **1M windows**:
  - `window_size=1048576`

The rest of the pipeline (data loading, model, training loop) stays identical.

### 8. Loading a Trained Head for Inference

Use `load_checkpoint` from the main API:

```python
from alphagenome_ft import load_checkpoint

model = load_checkpoint(
    "checkpoints/rna_heads_only/best",
    base_model_version="all_folds",
)

# Use for inference on new sequences
preds = model.predict_batch_of_sequences(
    sequences=...,          # one-hot encoded DNA, shape (B, L, 4)
    organism_index=...,     # e.g. jnp.zeros((sequences.shape[0],), dtype=jnp.int32) for human
)
```

**Note on data leakage and benchmarks:** `base_model_version="all_folds"` uses the AlphaGenome distilled model trained on all four genomic folds. This is ideal for practical use, but if your new task’s train/valid/test splits overlap AlphaGenome’s (and Borzoi’s) held‑out regions, this can lead to **data leakage and inflated test performance**. If you care about strict benchmarking that matches the splits used in [AlphaGenome](https://www.nature.com/articles/s41586-025-10014-0) and [Borzoi](https://www.nature.com/articles/s41588-024-02053-6), load one of the fold‑specific AlphaGenome models instead and align your train/valid/test partitions to those folds.

### 9. When to Use Heads-Only Finetuning

Use this tutorial when:
- You have **many targets** but similar regulatory logic.
- You want to keep **AlphaGenome’s backbone fixed**.
- You want **small checkpoints** that can be moved around easily.
- You are adding a new track to an existing genome-wide prediction task.

If you need the backbone to adapt (e.g. domain shift, very different assay), see the **Full-Model Finetuning** tutorial.

