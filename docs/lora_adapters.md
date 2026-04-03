## Tutorial: LoRA Adapters in `alphagenome-ft`

This tutorial explains how to use the **built‑in LoRA (Low‑Rank Adaptation) utilities** in `alphagenome-ft` to perform parameter‑efficient finetuning of AlphaGenome, and how this is instantiated in the ATAC demo notebook `notebooks/finetune_atac_lora.ipynb`.

The package now provides:
- **LoRA building blocks** in `alphagenome_ft.lora` (`LoRAConfig`, `LoRALinear`, parameter‑count helpers).
- **Parameter inspection and freezing utilities** in `alphagenome_ft.parameter_utils`.
- A standard training loop (`alphagenome_ft.finetune.train.train`) that works with LoRA‑equipped heads.

### 1. When to Use LoRA vs Heads‑Only Finetuning

**Heads‑only finetuning**:
- Adds a new task‑specific head on top of frozen AlphaGenome embeddings.
- Trains only the head parameters; the backbone remains frozen.
- Works well when a shallow mapping from embeddings to targets is sufficient.

**LoRA finetuning**:
- Still keeps the AlphaGenome backbone **frozen**, but adds small trainable low‑rank matrices inside your head’s linear projections.
- Increases expressivity with **very few additional parameters**, often improving over heads‑only training while retaining low memory and compute cost.

Use **LoRA** when:
- Full‑model finetuning is too expensive.
- Heads‑only finetuning has saturated and you need a richer adapter without unfreezing the backbone.

### 2. Built‑In LoRA Utilities (`alphagenome_ft.lora`)

`alphagenome_ft.lora` provides:

- **`LoRAConfig`**  
  - `rank`: low‑rank dimension (e.g. 4, 8, 16).  
  - `alpha`: scaling factor; effective scale is \( \alpha / \text{rank} \). Setting `alpha == rank` yields scale 1.

- **`LoRALinear`**  
  - A Haiku module that augments a linear projection with a LoRA adapter.  
  - Parameters:
    - Base weight `w` (typically kept frozen).
    - Low‑rank matrices `lora_a`, `lora_b` (trainable).  
  - Optional bias term `b` controlled by `with_bias`.  
  - Designed so that the parameters are named `w`, `lora_a`, `lora_b`, matching AlphaGenome’s linear‑layer conventions and integrating cleanly with checkpoints.

- **`get_lora_parameter_paths(params)`**  
  - Returns all parameter paths in a Haiku parameter tree whose leaf name is `lora_a` or `lora_b`.  
  - Useful for verifying that LoRA adapters are present where you expect them.

- **`count_lora_parameters(params)`**  
  - Counts the total number of scalar parameters across all `lora_a` and `lora_b` arrays.  
  - Lets you quantify how lightweight your adapters are compared to the full model.

The underlying forward computation in `LoRALinear` is:

\[
  y = x W \;+\; (x A) B \cdot \frac{\alpha}{r},
\]

where:
- \( W \) is the frozen base weight (`w`),
- \( A \) is `lora_a` (shape \((\text{in\_dim}, r)\)),
- \( B \) is `lora_b` (shape \((r, \text{out\_dim})\)),
- \( r \) is the LoRA rank.

In the reference implementation:
- `lora_a` is initialised from a small normal distribution.
- `lora_b` is initialised to zeros so the adapter starts as a **no‑op**, matching the frozen‑backbone baseline at step 0.

### 3. Example: ATAC/RNA Head with LoRA

The notebook `notebooks/finetune_atac_lora.ipynb` shows a concrete **LoRA head** that adapts AlphaGenome to ATAC tracks, but the same pattern applies to RNA or other genome‑track targets.

Key ideas:
- Use **1 bp embeddings** from the AlphaGenome decoder: `embeddings.get_sequence_embeddings(resolution=1)` giving shape `(B, S, 1536)`.
- Apply a single `LoRALinear` to project from 1536 features to `NUM_TRACKS` outputs.
- Wrap this in a `CustomHead` whose `predict` method uses `LoRALinear` and whose `loss` compares predictions to per‑base targets.
- Register the head via `register_custom_head` and build `head_specs` with `source: "custom"`.

Conceptually, the `predict` method of such a head looks like:

```python
from alphagenome_ft import lora

class LoRAHead(CustomHead):
    def predict(self, embeddings, organism_index, **kwargs):
        x   = embeddings.get_sequence_embeddings(resolution=1)          # (B, S, 1536)
        cfg = lora.LoRAConfig(rank=LORA_RANK, alpha=LORA_ALPHA)
        x   = lora.LoRALinear(self._num_tracks, cfg)(x)                 # (B, S, num_tracks)
        return {"predictions": jax.nn.softplus(x)}
```

The notebook then:
- Defines `TARGETS_CONFIG` for BigWig tracks and converts it to `head_specs`.
- Creates a model with `create_model_with_heads(MODEL_VERSION, heads=[HEAD_NAME])`.

### 4. Inspecting and Counting LoRA Parameters

Before training, you can verify where LoRA lives and how large it is:

- **Backbone paths**:  
  `parameter_utils.get_backbone_parameter_paths(model._params)`  
  lists all arrays belonging to the frozen AlphaGenome backbone.

- **LoRA paths**:  
  `lora.get_lora_parameter_paths(model._params)`  
  returns all leaf paths ending in `lora_a` or `lora_b`.

- **Parameter counts**:  
  Combine `parameter_utils.count_parameters(model._params)` with  
  `lora.count_lora_parameters(model._params)` to compute the **LoRA parameter fraction**:

```python
total_params = parameter_utils.count_parameters(model._params)
lora_params  = lora.count_lora_parameters(model._params)
print(f"LoRA fraction: {lora_params / total_params:.4%}")
```

This gives a quick sanity check that your adapters remain small relative to the backbone.

### 5. Freezing the Backbone and Training Only LoRA

Once the LoRA head is attached, the notebook freezes the backbone while keeping the LoRA adapters trainable:

- **Freeze everything except LoRA**  
  Use `parameter_utils.freeze_except_lora`:

```python
from alphagenome_ft import parameter_utils

model._params = parameter_utils.freeze_except_lora(model._params)
```

This applies `jax.lax.stop_gradient` to all parameters except those whose leaf name is `lora_a` or `lora_b`. The LoRA matrices remain trainable; the backbone (and any non‑LoRA head weights) are frozen. **Training-time** freezing of the backbone still requires **optimizer masking** for custom loops; `stop_gradient` alone is not enough (see [heads_only_optimizer.md](heads_only_optimizer.md)).

- **Train with `heads_only=True`**  
  When calling `alphagenome_ft.finetune.train.train`, pass `heads_only=True`. The trainer uses `create_optimizer` with masking so non‑head parameters receive **zero updates** (no SGD / no weight decay on the backbone):


```python
train(
    model,
    data_module,
    head_specs,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    num_epochs=NUM_EPOCHS,
    seed=SEED,
    max_train_steps=MAX_TRAIN_STEPS,
    heads_only=True,
    checkpoint_dir=CHECKPOINT_DIR,
    organism=ORGANISM,
    best_metric=BEST_METRIC,
    best_metric_mode=BEST_METRIC_MODE,
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    early_stopping_min_delta=EARLY_STOPPING_MIN_DELTA,
    verbose=VERBOSE,
)
```

In the ATAC demo notebook this is combined with `BigWigDataModule` and fold‑based genomic intervals, but the same pattern works for other targets and loaders.

**Custom Optax loop** (replace `HEAD_NAME` with your registered head id; LoRA tensors live under that head’s path):

```python
import optax
from alphagenome_ft import create_optimizer

optimizer = create_optimizer(
    model._params,
    trainable_head_names=(HEAD_NAME,),
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    heads_only=True,
)
opt_state = optimizer.init(model._params)
# updates, opt_state = optimizer.update(grads, opt_state, model._params)
# model._params = optax.apply_updates(model._params, updates)
```

### 6. LoRA and Sequence Length (up to 1 Mbp)

LoRA operates on **weights**, not directly on sequence length, so you can reuse any of the sequence window sizes described elsewhere:

- 32k, 64k, 128k, and up to ~1 Mbp windows.
- Control sequence length via `window_size` in the finetuning data utilities (e.g. `prepare_intervals_from_fold`) or your own loaders.

Your choice of **where** to place LoRA (which heads or layers) is independent of the chosen window size.

### 7. Recommended Workflow

1. **Start with heads‑only finetuning** on your task using the standard `GENOME_TRACKS` heads.
2. If performance plateaus:
   - Introduce a LoRA head (as in `finetune_atac_lora.ipynb`) for your task.
   - Use `LoRAConfig` to keep the adapter small (e.g. rank 4–16).
3. Monitor:
   - Validation metrics (compare heads‑only vs LoRA).
   - LoRA parameter fraction (via `count_lora_parameters`) to ensure parameter efficiency.

### 8. Notes and Caveats (Model Versions and Folds)

- **Model choice and data leakage**  
  The `"all_folds"` distilled AlphaGenome model is convenient for practical adapter experiments, but if your train/valid/test partitions overlap AlphaGenome’s (and Borzoi’s) held‑out genomic regions, you can get **data leakage and inflated test performance**.  
  For strict benchmarking that matches the splits used in [AlphaGenome](https://www.nature.com/articles/s41586-025-10014-0) and [Borzoi](https://www.nature.com/articles/s41588-024-02053-6):
  - Prefer fold‑specific models (e.g. `"fold_0"`, `"fold_1"`, etc.).
  - Align your train/valid/test intervals with the corresponding folds.

- **Extensibility**  
  The current implementation equips **heads** with LoRA adapters via `LoRALinear`. You can extend the same pattern to more complex multi‑layer heads or additional projections if needed, reusing the same utilities for parameter inspection, counting, and freezing.


