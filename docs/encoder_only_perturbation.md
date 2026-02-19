## Tutorial: Encoder-Only Finetuning for Perturbation / MPRA Data

This tutorial focuses on **short-sequence perturbation assays** such as:
- MPRA / lentiMPRA constructs (e.g. 150–300 bp)
- Promoter / enhancer tiles
- Other targeted regulatory elements where the sequence is **< 1 kb**

For these settings, AlphaGenome’s full transformer stack is not needed and may produce shape mismatches. Instead, you can use **encoder-only mode**:

- **Backbone**: AlphaGenome **encoder only** (CNN downsampling)
- **Head**: Custom or template head on top of encoder embeddings

**Why encoder-only mode?** Short sequences (< 1000 bp) in perturbation data like MPRA do not consider genomic contextso it doesn't make sense to 
pass them through a transformer's attention mechanism. Encoder-only mode skips the transformer/decoder and uses only the convolution and pooling 
features from the encoder - we have shown that this still captures complex _cis_-regultory logic.

### 1. When to Use Encoder-Only Mode

Use encoder-only mode when:
- Your sequences are **shorter than ~1000 bp**.
- You want efficient training on many small constructs.
- You care primarily about **local regulatory logic** rather than megabase context.

For full-genome windows (32k–1M), use the other finetuning tutorials.

### 2. High-Level Setup

For encoder-only mode, you will:

1. Register an **encoder-only head** (either via template or custom class).
2. Create a model using `use_encoder_output=True`.
3. Train on your perturbation dataset (e.g. MPRA).
4. Optionally refer to the dedicated MPRA repository for full pipelines.

### 3. Using the Encoder-Only Template Head

The package provides `templates.EncoderOnlyHead`, which uses CNN encoder features only.

```python
from alphagenome.models import dna_output
from alphagenome_ft import (
    templates,
    CustomHeadConfig,
    CustomHeadType,
    register_custom_head,
    create_model_with_heads,
)

# 1. Register an encoder-only head
register_custom_head(
    "mpra_head",
    templates.EncoderOnlyHead,
    CustomHeadConfig(
        type=CustomHeadType.GENOME_TRACKS,
        output_type=dna_output.OutputType.RNA_SEQ,
        num_tracks=1,
    ),
)

# 2. Create a model that uses encoder output only
model = create_model_with_heads(
    "all_folds",
    heads=["mpra_head"],
    use_encoder_output=True,   # ← CRITICAL for encoder-only mode
)

# 3. Optionally freeze backbone to start with heads-only finetuning
model.freeze_except_head("mpra_head")
```

### 4. Training Loop (Short Sequences)

For MPRA-like data, you will typically have **short sequences and scalar or low-dimensional outputs** (e.g. log expression).

You can either:
- Use your own data loader and a custom training loop with `model.create_loss_fn_for_head`, or
- Follow the more complete MPRA scripts in the external repository.

Minimal example with a custom loop:

```python
import jax
import jax.numpy as jnp
import optax

from alphagenome_ft import CustomHead

# Suppose you have: sequences_onehot: (B, L, 4), targets: (B, 1)

loss_fn = model.create_loss_fn_for_head("mpra_head")

optimizer = optax.adamw(learning_rate=1e-3, weight_decay=1e-4)
opt_state = optimizer.init(model._params)

def train_step(params, state, opt_state, batch_sequences, batch_targets):
    def loss_inner(current_params):
        preds_dict = model._predict(
            current_params,
            state,
            batch_sequences,
            jnp.zeros((batch_sequences.shape[0],), dtype=jnp.int32),  # organism_index
            negative_strand_mask=jnp.zeros((batch_sequences.shape[0],), dtype=bool),
            strand_reindexing=model._metadata[next(iter(model._metadata))].strand_reindexing,
        )
        preds = preds_dict["mpra_head"]
        loss_dict = loss_fn(
            preds,
            {"targets": batch_targets, "organism_index": None},
        )
        return loss_dict["loss"]

    loss, grads = jax.value_and_grad(loss_inner)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss
```

You can wrap this into a standard epoch/batch training loop over your MPRA dataset.

### 5. Sequence Length Choices for Perturbation Assays

For perturbation assays:

- Typical sequence lengths: **200–600 bp**.
- You can pad or crop to a fixed length (e.g. 512 or 1024 bp).
- Ensure all sequences share the same length before batching.

There is no 1 Mbp limit in practice here, because you are working on short constructs. Encoder-only mode is efficient and avoids transformer overhead.

### 6. End-to-End MPRA & STARR-Seq Examples

For complete, production-ready pipelines (configs, training scripts, evaluation, attributions) using encoder-only heads on perturbation datasets, see:

- Main README of the [MPRA finetuning repo](https://github.com/Al-Murphy/alphagenome_FT_MPRA).
- `src/README.md`, `scripts/README.md`, and `configs/README.md` in that repo for:
  - MPRA heads (`mpra_heads.py`)
  - Training utilities (`training.py`)
  - Config-driven finetuning scripts (`scripts/finetune_*.py`)
  - Attribution analysis scripts.

These examples show how `alphagenome-ft` is used in practice for lentiMPRA and DeepSTARR with encoder-only heads.

### 7. When to Use Encoder-Only vs Full Backbone

Use **encoder-only mode** when:
- Sequences are **short** (< 1 kb).
- You primarily care about **local motif logic**.
- You want a lighter, faster model.

Use a **full backbone** (possibly with smaller windows) when:
- You need **long-range interactions** across tens to hundreds of kilobases.
- Your assay is genome-wide and benefits from full AlphaGenome context.

