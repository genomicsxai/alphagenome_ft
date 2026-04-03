"""
AlphaGenome Finetuning Extensions

This package provides utilities for finetuning the AlphaGenome model with custom heads
and parameter freezing capabilities, without modifying the original alphagenome_research codebase.
"""

try:
    from importlib.metadata import version as _version, PackageNotFoundError as _PackageNotFoundError
    try:
        __version__ = _version("alphagenome-ft")
    except _PackageNotFoundError:
        __version__ = "unknown"
except ImportError:
    __version__ = "unknown"

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
    CustomHeadConfig,
    CustomHeadType,
    HeadConfig,
    HeadType,
    create_predefined_head,
    create_predefined_head_from_config,
    is_predefined_head,
    is_predefined_config,
    list_registered_heads,
    list_predefined_heads,
    register_predefined_head,
    deserialize_predefined_head_config,
    get_registered_head_config,
    get_registered_head_metadata,
    create_registered_head,
    is_head_registered,
    is_custom_config,
    get_predefined_head_config,
    normalize_head_name,
    register_custom_head,
    create_custom_head,
    get_custom_head_config,
    is_custom_head,
    list_custom_heads,
)
from alphagenome_ft import templates
from alphagenome_ft import finetune
from alphagenome_ft import lora
from alphagenome_ft.lora import (
    LoRAConfig,
    LoRALinear,
    get_lora_parameter_paths,
    count_lora_parameters,
)
from alphagenome_ft.parameter_utils import freeze_except_lora
from alphagenome_ft.optimizer_utils import (
    create_optimizer,
    label_params_for_trainable_heads,
    parameter_path_to_str,
)

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
    'CustomHeadConfig',
    'CustomHeadType',
    'HeadConfig',
    'HeadType',
    'create_predefined_head',
    'create_predefined_head_from_config',
    'is_predefined_head',
    'is_predefined_config',
    'list_registered_heads',
    'list_predefined_heads',
    'register_predefined_head',
    'deserialize_predefined_head_config',
    'get_registered_head_config',
    'get_registered_head_metadata',
    'create_registered_head',
    'is_head_registered',
    'is_custom_config',
    'get_predefined_head_config',
    'normalize_head_name',
    'register_custom_head',
    'create_custom_head',
    'get_custom_head_config',
    'is_custom_head',
    'list_custom_heads',
    # Templates module (contains example head implementations)
    'templates',
    # Finetuning workflow module
    'finetune',
    # LoRA utilities
    'lora',
    'LoRAConfig',
    'LoRALinear',
    'get_lora_parameter_paths',
    'count_lora_parameters',
    'freeze_except_lora',
    # Optimizer masking (true backbone freeze during training)
    'create_optimizer',
    'label_params_for_trainable_heads',
    'parameter_path_to_str',
]