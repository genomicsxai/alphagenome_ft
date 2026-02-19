# Attribution Analysis

After training, you can compute attributions to understand which sequence features drive your model's predictions. The package supports multiple attribution methods:

- **DeepSHAP**: Reference-based attribution using gradient differences
- **Gradient × Input**: Gradient multiplied by input (standard gradient-based attribution)
- **Gradient**: Raw gradients (information content)
- **ISM** (_in silico_ mutagenesis): Importance scores from single-nucleotide variants

**NOTE**: DeepSHAP - The implementation is DeepSHAP-like in that it uses a reference sequence but is note a faithful reimplmenntation.

## Basic Attribution Example

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

## Visualization

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

## Complete Example: Analyzing a Single Sequence

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

## Attribution Methods Comparison

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

For complete examples with motif analysis, background shuffling, and batch processing, see the MPRA finetuning repository's attribution scripts (for example `scripts/compute_attributions_lentimpra.py` and `scripts/compute_attributions_starrseq.py`).
