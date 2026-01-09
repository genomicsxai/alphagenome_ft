"""
Utilities for managing and manipulating AlphaGenome model parameters.

Provides functions for freezing/unfreezing parameters and inspecting parameter trees.

AlphaGenome Model Architecture (for reference):
DNA Sequence (B, S, 4)
    ↓
SequenceEncoder (convolutional downsampling)
    ↓
TransformerTower (9 transformer blocks with pairwise attention)
    ↓
SequenceDecoder (convolutional upsampling)
    ↓
Embeddings:
  - embeddings_1bp: (B, S, 1536) - High resolution
  - embeddings_128bp: (B, S//128, 3072) - Low resolution  
  - embeddings_pair: (B, S//2048, S//2048, 128) - Pairwise
    ↓
Heads (task-specific predictions):
  - ATAC, DNASE, RNA_SEQ, etc.
  - YOUR_CUSTOM_HEAD ← Add here
1. **Backbone**: Encoder + Transformer + Decoder 
2. **Embeddings**: Multi-resolution representations 
3. **Heads**: Task-specific prediction layers


Parameter Paths:
In JAX/Haiku, parameters are organized in a PyTree (nested dictionary structure). Each parameter has a "path" like:
- `alphagenome/sequence_encoder/conv1/w` (backbone)
- `head/atac/output/b` (standard head)
- `head/mpra_head/~predict/hidden/w` (custom head)

Core Functions:
1. `_keypath_to_str(path_tuple)` - Converts JAX keypath tuples to readable strings.
2. `freeze_parameters(params, freeze_paths, freeze_prefixes)` - Freeze specific parameters by applying `jax.lax.stop_gradient`.
3. `unfreeze_parameters(params, unfreeze_paths, unfreeze_prefixes)` - Remove stop_gradient from parameters (make them trainable again).
4. `freeze_backbone(params)` - Freeze encoder, transformer, and decoder (the backbone).
5. `freeze_all_heads(params, except_heads)` - Freeze all heads except specified ones.
6. `freeze_except_head(params, trainable_head)` - Freeze everything except a specific head (common finetuning pattern).
7. `get_parameter_paths(params)` - Get all parameter paths in the tree.
8. `get_head_parameter_paths(params)` - Get all head parameter paths.
9. `get_backbone_parameter_paths(params)` - Get all backbone parameter paths.
10. `count_parameters(params)` - Count total number of parameters.
"""

from collections.abc import Callable, Mapping, Sequence
import jax
import jax.numpy as jnp
from jaxtyping import PyTree
from jax._src.tree_util import DictKey, GetAttrKey, SequenceKey


def _keypath_to_str(path_tuple: tuple) -> str:
    """Converts a JAX PyTree keypath tuple to a string.
    
    Args:
        path_tuple: Tuple of JAX key objects from tree_map_with_path.
        
    Returns:
        String representation of the path (e.g., 'alphagenome/head/atac/w').
    """
    parts = []
    for key in path_tuple:
        if isinstance(key, DictKey):
            parts.append(str(key.key))
        elif isinstance(key, GetAttrKey):
            parts.append(str(key.name))
        elif isinstance(key, SequenceKey):
            parts.append(str(key.idx))
        else:
            parts.append(str(key))
    return '/'.join(parts)


def freeze_parameters(
    params: PyTree,
    freeze_paths: Sequence[str] | None = None,
    freeze_prefixes: Sequence[str] | None = None,
) -> PyTree:
    """Freeze specific parameters by applying stop_gradient.
    
    Args:
        params: Parameter tree to freeze.
        freeze_paths: Exact parameter paths to freeze.
        freeze_prefixes: Path prefixes to freeze (e.g., 'alphagenome/head' freezes all heads).
        
    Returns:
        Parameter tree with stop_gradient applied to specified parameters.
    """
    freeze_paths = freeze_paths or []
    freeze_prefixes = freeze_prefixes or []
    
    def should_freeze(path_tuple: tuple) -> bool:
        """Check if a parameter path should be frozen."""
        path_str = _keypath_to_str(path_tuple)
        
        # Check exact paths
        for freeze_path in freeze_paths:
            if path_str == freeze_path:
                return True
        
        # Check prefixes
        for prefix in freeze_prefixes:
            if path_str.startswith(prefix) or f'/{prefix}/' in path_str or path_str.endswith(f'/{prefix}'):
                return True
        
        return False
    
    def freeze_fn(path, value):
        if should_freeze(path):
            return jax.lax.stop_gradient(value)
        return value
    
    return jax.tree_util.tree_map_with_path(freeze_fn, params)


def unfreeze_parameters(
    params: PyTree,
    unfreeze_paths: Sequence[str] | None = None,
    unfreeze_prefixes: Sequence[str] | None = None,
) -> PyTree:
    """Remove stop_gradient from specific parameters.
    
    Note: This removes stop_gradient wrappers, making parameters trainable again.
    
    Args:
        params: Parameter tree to unfreeze.
        unfreeze_paths: Exact parameter paths to unfreeze.
        unfreeze_prefixes: Path prefixes to unfreeze.
        
    Returns:
        Parameter tree with stop_gradient removed from specified parameters.
    """
    # In JAX, we can't directly "remove" stop_gradient, but we can identity map
    # the values we want to unfreeze. In practice, this is used after freeze_parameters
    # to selectively unfreeze certain paths.
    unfreeze_paths = unfreeze_paths or []
    unfreeze_prefixes = unfreeze_prefixes or []
    
    def should_unfreeze(path_tuple: tuple) -> bool:
        """Check if a parameter path should be unfrozen."""
        path_str = _keypath_to_str(path_tuple)
        
        # Check exact paths
        for path in unfreeze_paths:
            if path_str == path:
                return True
        
        # Check prefixes
        for prefix in unfreeze_prefixes:
            if path_str.startswith(prefix) or f'/{prefix}/' in path_str or path_str.endswith(f'/{prefix}'):
                return True
        
        return False
    
    def unfreeze_fn(path, value):
        if should_unfreeze(path):
            # Identity map to ensure no stop_gradient wrapper
            return jax.lax.stop_gradient(value) * 0 + value
        return value
    
    return jax.tree_util.tree_map_with_path(unfreeze_fn, params)


def freeze_backbone(params: PyTree) -> PyTree:
    """Freeze the backbone (encoder, transformer, decoder) but keep heads trainable.
    
    Args:
        params: Parameter tree to modify.
        
    Returns:
        Parameter tree with frozen backbone.
    """
    return freeze_parameters(
        params,
        freeze_prefixes=['sequence_encoder', 'transformer_tower', 'sequence_decoder']
    )


def freeze_all_heads(params: PyTree, except_heads: Sequence[str] | None = None) -> PyTree:
    """Freeze all heads except specified ones.
    
    Args:
        params: Parameter tree to modify.
        except_heads: Head names to keep trainable.
        
    Returns:
        Parameter tree with frozen heads (except specified ones).
    """
    except_heads = except_heads or []
    
    def should_freeze(path_tuple: tuple) -> bool:
        """Check if a head parameter should be frozen."""
        path_str = _keypath_to_str(path_tuple)
        
        # Check if it's a head parameter
        is_head = '/head/' in path_str or path_str.startswith('head/')
        if not is_head:
            return False
        
        # Check if it's in the exception list
        for head_name in except_heads:
            if f'/head/{head_name}/' in path_str or path_str.startswith(f'head/{head_name}/'):
                return False
        
        return True
    
    def freeze_fn(path, value):
        if should_freeze(path):
            return jax.lax.stop_gradient(value)
        return value
    
    return jax.tree_util.tree_map_with_path(freeze_fn, params)


def freeze_except_head(params: PyTree, trainable_head: str) -> PyTree:
    """Freeze everything except a specific head.
    
    Args:
        params: Parameter tree to modify.
        trainable_head: Name of the head to keep trainable.
        
    Returns:
        Parameter tree with only the specified head trainable.
    """
    # Freeze backbone
    frozen = freeze_backbone(params)
    
    # Freeze all other heads
    all_paths = get_parameter_paths(params)
    head_paths_to_freeze = [
        p for p in all_paths
        if ('/head/' in p or p.startswith('head/')) and 
           not (f'/head/{trainable_head}/' in p or p.startswith(f'head/{trainable_head}/'))
    ]
    
    frozen = freeze_parameters(frozen, freeze_paths=head_paths_to_freeze)
    
    return frozen


def get_parameter_paths(params: PyTree) -> list[str]:
    """Get all parameter paths in the tree.
    
    Args:
        params: Parameter tree to inspect.
        
    Returns:
        List of all parameter paths as strings.
    """
    paths = []
    
    def collect_path(path, value):
        if hasattr(value, 'shape'):  # It's a leaf node (array)
            paths.append(_keypath_to_str(path))
    
    jax.tree_util.tree_map_with_path(collect_path, params)
    return paths


def get_head_parameter_paths(params: PyTree) -> list[str]:
    """Get all head parameter paths.
    
    Args:
        params: Parameter tree to inspect.
        
    Returns:
        List of head parameter paths.
    """
    all_paths = get_parameter_paths(params)
    return [p for p in all_paths if '/head/' in p or p.startswith('head/')]


def get_backbone_parameter_paths(params: PyTree) -> list[str]:
    """Get all backbone parameter paths.
    
    Args:
        params: Parameter tree to inspect.
        
    Returns:
        List of backbone parameter paths.
    """
    all_paths = get_parameter_paths(params)
    return [
        p for p in all_paths
        if any(component in p for component in ['sequence_encoder', 'transformer_tower', 'sequence_decoder'])
    ]


def get_other_parameter_paths(params: PyTree) -> list[str]:
    """Get all non-backbone, non-head parameter paths.
    
    Args:
        params: Parameter tree to inspect.
        
    Returns:
        List of other parameter paths.
    """
    all_paths = get_parameter_paths(params)
    backbone_paths = set(get_backbone_parameter_paths(params))
    head_paths = set(get_head_parameter_paths(params))
    
    return [p for p in all_paths if p not in backbone_paths and p not in head_paths]


def count_parameters(params: PyTree) -> int:
    """Count total number of parameters.
    
    Args:
        params: Parameter tree to count.
        
    Returns:
        Total number of parameters.
    """
    flat_params = jax.tree_util.tree_flatten(params)[0]
    return sum(p.size for p in flat_params if hasattr(p, 'size'))

