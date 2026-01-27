"""
AlphaGenome Finetuning Extensions

This package provides utilities for finetuning the AlphaGenome model with custom heads
and parameter freezing capabilities, without modifying the original alphagenome_research codebase.
"""

from alphagenome_ft.custom_model import (
    CustomAlphaGenomeModel,
    create_model_with_heads,
    create_model_with_custom_heads,
    wrap_pretrained_model,
    add_heads_to_model,
    add_custom_heads_to_model,
    load_checkpoint,
)
from alphagenome_ft.custom_heads import (
    CustomHead,
    HeadConfig,
    HeadType,
    create_predefined_head,
    is_predefined_head,
    list_predefined_heads,
    register_predefined_head,
    deserialize_predefined_head_config,
    get_head_config,
    create_head,
    is_head_registered,
    is_custom_config,
    get_predefined_head_config,
    normalize_head_name,
    register_custom_head,
    get_custom_head_config,
    is_custom_head,
    list_custom_heads,
)
from alphagenome_ft import templates
from alphagenome_ft import finetune

__all__ = [
    # Model classes
    'CustomAlphaGenomeModel',
    'create_model_with_heads',
    'create_model_with_custom_heads',
    'wrap_pretrained_model',
    'add_heads_to_model',
    'add_custom_heads_to_model',
    'load_checkpoint',
    # Head classes and utilities
    'CustomHead',
    'HeadConfig',
    'HeadType',
    'create_predefined_head',
    'is_predefined_head',
    'list_predefined_heads',
    'register_predefined_head',
    'deserialize_predefined_head_config',
    'get_head_config',
    'create_head',
    'is_head_registered',
    'is_custom_config',
    'get_predefined_head_config',
    'normalize_head_name',
    'register_custom_head',
    'get_custom_head_config',
    'is_custom_head',
    'list_custom_heads',
    # Templates module (contains example head implementations)
    'templates',
    # Finetuning workflow module
    'finetune',
]
