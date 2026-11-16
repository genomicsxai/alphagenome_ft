# LoRA adapters

`alphagenome-ft` supports LoRA inside AlphaGenome's transformer backbone. The
implementation mirrors the default LoRA mode in `alphagenome-pytorch`:

- adapters are added to the query and value projections in all nine attention
  blocks (`q_proj`/`v_proj` in PyTorch, `q_layer`/`v_layer` in JAX);
- rank defaults to 8 and alpha defaults to 16, for a scale of `alpha / rank`;
- A uses the same bounds as PyTorch `nn.Linear`'s Kaiming-uniform default and B
  is initialized to zero, so the adapted trunk is an exact pretrained no-op at
  initialization;
- the pretrained trunk is frozen while the selected heads and adapter matrices
  are trained; and
- gradient clipping defaults to a global norm of 1.0 over trainable parameters.

At rank 8 with both default targets, the backbone has 308,736 trainable LoRA
scalars.

## Create a backbone-LoRA model

Register or prepare the heads normally, then pass `BackboneLoRAConfig` when the
model is created:

```python
from alphagenome_ft import BackboneLoRAConfig, create_model_with_heads

lora_config = BackboneLoRAConfig(
    rank=8,
    alpha=16,
    targets=("q_proj", "v_proj"),
    gradient_checkpointing=True,  # recommended for long sequences
)

model = create_model_with_heads(
    "all_folds",
    heads=[HEAD_NAME],
    backbone_lora_config=lora_config,
    head_init_scheme="pytorch",  # match newly initialized PyTorch track heads
    init_seq_len=SEQUENCE_LENGTH,
)
```

`head_init_scheme="pytorch"` matches the initialization distribution of new
predefined track heads. The default, `"jax"`, preserves native Haiku head
initialization. This setting does not change the LoRA A/B initialization above.

The PyTorch names `q_proj` and `v_proj` are accepted deliberately. Their JAX
parameter paths remain the native `q_layer` and `v_layer` paths, which lets the
pretrained `w` leaves load unchanged while adding `lora_a` and `lora_b`.
Encoder-only mode and `detach_backbone=True` are incompatible with backbone
LoRA because both bypass or stop gradients before the transformer.

## Train adapters and heads

The built-in trainer recognizes `model._backbone_lora_config`. When
`heads_only` and `train_lora` are left as `None`, it enables both automatically:

```python
from alphagenome_ft.finetune import train

train(
    model,
    data_module,
    head_specs,
    learning_rate=1e-4,
    weight_decay=0.0,
    num_epochs=10,
    checkpoint_dir=CHECKPOINT_DIR,
)
```

This trains only the requested heads plus all `lora_a`/`lora_b` leaves. The
base query/value weights and the rest of the trunk get zero gradients and zero
optimizer updates. Passing `train_lora=True, heads_only=False` is rejected,
because it would silently turn the run into full-model finetuning.

For an explicit call, use:

```python
train(
    model,
    data_module,
    head_specs,
    learning_rate=1e-4,
    weight_decay=0.0,
    num_epochs=10,
    heads_only=True,
    train_lora=True,
    max_grad_norm=1.0,
)
```

`max_grad_norm=1.0` matches the PyTorch pipeline. Set it to `None` or `0` to
recover the older JAX behavior without clipping.

Matching these settings does not imply bitwise equality between training runs.
Framework RNGs and mixed-precision kernels differ. PyTorch gradient clipping
also divides by `norm + 1e-6`, while Optax clips using the norm without that
additive term.

### Custom Optax loops

Use the package optimizer rather than applying AdamW to the full parameter
tree:

```python
from alphagenome_ft import create_optimizer

optimizer = create_optimizer(
    model._params,
    trainable_head_names=(HEAD_NAME,),
    learning_rate=1e-4,
    weight_decay=0.0,
    heads_only=True,
    train_lora=True,
    gradient_clip_global_norm=1.0,
)
opt_state = optimizer.init(model._params)
```

Clipping is computed only over trainable head and LoRA leaves, matching
`torch.nn.utils.clip_grad_norm_(trainable_params, ...)`. `freeze_backbone()` or
`jax.lax.stop_gradient` on an already-created parameter tree is not, by itself,
a persistent optimizer mask.

## Inspect adapter parameters

```python
from alphagenome_ft import count_lora_parameters, get_lora_parameter_paths

paths = get_lora_parameter_paths(model._params)
count = count_lora_parameters(model._params)
print(count, paths[:4])
```

For the default nine-block Q/V configuration, `count` is 308,736.

## Checkpoints

The default delta checkpoint (`save_full_model=False`) stores custom heads and
LoRA leaves, but not pretrained base weights. Reload it with the same base
checkpoint or model version:

```python
model.save_checkpoint(CHECKPOINT_DIR, save_full_model=False)
```

Full-model LoRA checkpoints are also supported and restore the LoRA-aware
forward function. Minimal encoder-only checkpoints are intentionally rejected
for backbone LoRA because they omit the transformer where the adapters live.

## Legacy head-level LoRA

`LoRAConfig` and `LoRALinear` remain available for custom heads that contain
their own low-rank projections. That is distinct from the PyTorch-aligned
backbone mode above. A head-level adapter can be useful for experimentation,
but it does not adapt AlphaGenome's attention projections.

For leakage-sensitive evaluation, use a fold-specific base model and match the
training, validation, and test intervals to that fold. The `all_folds` model is
convenient for applications but is not appropriate for held-out fold claims.
