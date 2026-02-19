## Tutorial: LoRA‑Style Adapters on Top of AlphaGenome

This tutorial outlines how to use **LoRA‑style low‑rank adapters** with AlphaGenome models wrapped by `alphagenome-ft`.

The package gives you:
- A clean way to **inspect parameter paths** (via `parameter_utils`).
- A simple wrapper (`CustomAlphaGenomeModel`) around the AlphaGenome backbone.

LoRA itself is implemented as **user code** in your project, on top of Haiku modules. This tutorial sketches a reference pattern.

### 1. When to Use LoRA-Style Adapters

LoRA‑style adapters are useful when:
- You want to adapt AlphaGenome to a new domain with **very few trainable parameters**.
- Full-model finetuning is too expensive.
- Heads-only finetuning is not expressive enough.

The idea:
- Keep original backbone parameters **frozen**.
- Add small **low-rank matrices** to selected layers (e.g. attention projections, MLPs).

### 2. Identify Target Layers with `parameter_utils`

First, inspect parameter paths to decide where to attach LoRA adapters:

```python
from alphagenome_ft import create_model_with_heads
from alphagenome_ft import parameter_utils

model = create_model_with_heads("all_folds", heads=["my_head"])

paths = parameter_utils.get_backbone_parameter_paths(model._params)
for p in paths[:40]:
    print(p)
```

**Caution about `all_folds` and benchmarks:** `"all_folds"` uses the AlphaGenome distilled model trained on all four genomic folds. This is ideal for practical adapter experiments, but if your task’s train/valid/test splits overlap AlphaGenome’s (and Borzoi’s) held‑out regions, you can get **data leakage and inflated test performance**. If you care about strict benchmarking that matches the splits used in [AlphaGenome](https://www.nature.com/articles/s41586-025-10014-0) and [Borzoi](https://www.nature.com/articles/s41588-024-02053-6), load one of the fold‑specific AlphaGenome models instead and align your train/valid/test partitions to those folds.

Typical paths you might see:
- `alphagenome/sequence_encoder/...`
- `alphagenome/transformer_tower/block_0/attention/q_proj/w`
- `alphagenome/transformer_tower/block_0/mlp/dense_0/w`

Pick a small subset of these (e.g. Q/K/V projections or MLP layers) for LoRA.

### 3. Define a Haiku LoRA Module

Below is a **minimal Haiku module** that wraps a linear transformation with low-rank adapters:

```python
import jax.numpy as jnp
import haiku as hk

class LoRALinear(hk.Module):
    def __init__(self, out_dim: int, rank: int = 8, alpha: float = 1.0, name=None):
        super().__init__(name=name)
        self.out_dim = out_dim
        self.rank = rank
        self.alpha = alpha

    def __call__(self, x):
        in_dim = x.shape[-1]

        # Base weight (will be frozen)
        w = hk.get_parameter("w", shape=(in_dim, self.out_dim), init=hk.initializers.VarianceScaling())

        # LoRA adapters (trainable, low rank)
        a = hk.get_parameter("lora_a", shape=(in_dim, self.rank), init=hk.initializers.RandomNormal(0.01))
        b = hk.get_parameter("lora_b", shape=(self.rank, self.out_dim), init=hk.initializers.RandomNormal(0.01))

        base = x @ w
        delta = (x @ a) @ b * (self.alpha / self.rank)
        return base + delta
```

In practice you would integrate this into a custom attention/MLP block, reusing AlphaGenome embeddings as input.

### 4. Freeze Backbone, Train Only Adapters

Once you have a model that includes LoRA modules (e.g. in your own wrapper around AlphaGenome embeddings), you can:

1. **Freeze all original backbone parameters** using `parameter_utils.freeze_backbone`.
2. Ensure only LoRA parameters (e.g. `lora_a`, `lora_b`) are trainable.

Example pattern:

```python
from alphagenome_ft import parameter_utils

# 1. Freeze standard AlphaGenome backbone params
model._params = parameter_utils.freeze_backbone(model._params)

# 2. Optionally freeze all heads except a specific one
model._params = parameter_utils.freeze_all_heads(model._params, except_heads=["my_head"])
```

Your optimizer should then operate over the full parameter tree; frozen parameters will not receive gradients (and thus are effectively excluded).

### 5. LoRA with Sequence Lengths up to 1 Mbp

LoRA adapters operate on **layer weights**, not on sequence length. You can reuse any of the sequence window sizes described in other tutorials:

- 32k, 64k, 128k, up to ~1 Mbp windows.
- Control sequence length via `window_size` in the `finetune` data utilities, or in your own data loader.

The LoRA strategy is orthogonal to window length; choose window sizes that match your task, then decide which layers to equip with adapters.

### 6. Recommended Workflow

1. **Start with heads-only finetuning** (frozen backbone).
2. If performance plateaus and you suspect backbone limitations, try:
   - Adding small LoRA adapters to a **few transformer blocks**.
   - Training only those adapters + heads.
3. Monitor:
   - Validation metrics.
   - Parameter count (to confirm LoRA stays lightweight).

### 7. Notes and Caveats

- This repository does **not** ship a full AlphaGenome‑internal LoRA reparameterization; instead it provides:
  - Clean parameter inspection (to target modules).
  - Backbone freezing utilities.
  - A standard training loop you can adapt.
- Detailed, model‑internal LoRA implementations can be added in your own codebase, following the pattern above.

