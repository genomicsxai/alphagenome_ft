# Changelog

## Unreleased

### Behavior changes

- `finetune.train()` now defaults to `max_grad_norm=1.0`. Set it to `None` or
  `0` to retain the previous behavior without gradient clipping. Both optimizer
  factories clip over trainable parameters only, excluding frozen gradients.
- YAML `nonzero_mean` values now reach the predefined track head and affect
  target scaling, loss, and prediction unscaling; previously they were ignored.
  When loading older checkpoints trained through that YAML path, register the
  predefined heads with the original unit scale: omit those previously ignored
  means from the YAML/track metadata or set them to `1.0`. This preserves the
  scaling used during training; retaining the old non-unit YAML values changes
  prediction unscaling.
- Models built with `backbone_lora_config` now default to training heads and
  adapters only. `train_lora=True, heads_only=False` raises rather than silently
  updating the pretrained backbone.

### Fixes and additions

- Zero-initialized backbone LoRA preserves native bf16 trunk embeddings
  exactly, including with gradient checkpointing.
- Full-model LoRA checkpoints retain the adapters and custom-head forward.
  Minimal encoder-only checkpoints reject backbone LoRA at save time.
- Targets accept explicit `strand: "+"`, `strand: "-"`, or `strand: "."`;
  explicit values override label inference, and unmatched strand counts raise.
- `prepare_head_specs(..., nonzero_mean_intervals=training_intervals)` can
  estimate means when a head has no explicit means. Use training intervals
  only; without this argument, missing means retain the existing unit scale.
- Nonzero-mean estimation treats NaNs as zero and rejects positive or negative
  infinity. The PyTorch helper's current `nan_to_num` behavior instead saturates
  infinities; parity applies to finite input data.
