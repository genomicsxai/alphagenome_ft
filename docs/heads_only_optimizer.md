# Heads-only training: building the optimizer

`model.freeze_except_head(...)`, `freeze_backbone()`, and `freeze_parameters()` apply `jax.lax.stop_gradient` to parameter **values** at call time. That does **not** stop the backbone from receiving **Optax updates** in a later step when your loss still depends on those weights (for example `jax.value_and_grad(loss)(params)` over the full tree).

To keep the backbone fixed during training you must use **optimizer masking**: only trainable head leaves get Adam/AdamW updates; everything else gets zero updates (including no weight decay on frozen weights).

## Custom training loop

```python
import optax
from alphagenome_ft import create_optimizer

# After model.freeze_except_head("my_head") (optional but sets a hint on the model)
optimizer = create_optimizer(
    model._params,
    trainable_head_names=("my_head",),  # registered head id(s); use a tuple
    learning_rate=1e-3,
    weight_decay=1e-4,
    heads_only=True,
    optimizer_type="adamw",  # or "adam"
    gradient_clip_global_norm=None,  # optional, e.g. 1.0
)
opt_state = optimizer.init(model._params)

# Each step, after computing grads w.r.t. params:
# updates, opt_state = optimizer.update(grads, opt_state, model._params)
# model._params = optax.apply_updates(model._params, updates)
```

Same thing via the model wrapper:

```python
optimizer = model.create_optimizer(
    trainable_head_names=("my_head",),
    learning_rate=1e-3,
    weight_decay=1e-4,
    heads_only=True,
)
opt_state = optimizer.init(model._params)
```

## Multiple trainable heads

Pass every head you want to update:

```python
trainable_head_names=("K562_rna_seq", "other_head")
```

## Built-in BigWig training

`alphagenome_ft.finetune.train.train(..., heads_only=True)` constructs this optimizer internally. You do not need to call `create_optimizer` yourself when using that entry point.

## LoRA and encoder-only workflows

- **LoRA:** Trainable `lora_a` / `lora_b` leaves usually live under `head/<your_head_name>/...`, so use `trainable_head_names=(HEAD_NAME,)` with `heads_only=True` the same way as for a plain head.
- **Encoder-only / MPRA-style loops:** If you hand-write a `train_step` with `jax.grad`, still use `create_optimizer(..., heads_only=True)` instead of a bare `optax.adamw` on the full tree.

## Implementation

See `alphagenome_ft.optimizer_utils.create_optimizer` and the tests in `tests/test_optimizer_masking.py`.
