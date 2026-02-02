"""
Configuration utilities for loading from environment variables and config files.
"""

import os
from typing import Optional, Any
from pathlib import Path


def str_to_bool(value: str) -> bool:
    """Convert string to boolean"""
    return value.lower() in ('true', '1', 'yes', 'on')


def load_env_file(env_file: str = ".env") -> dict:
    """
    Load environment variables from .env file.

    Args:
        env_file: Path to .env file

    Returns:
        Dict of environment variables
    """
    env_vars = {}
    env_path = Path(env_file)

    if not env_path.exists():
        return env_vars

    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            # Parse KEY=VALUE
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()

                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]

                env_vars[key] = value

    return env_vars


def get_env(key: str, default: Any = None, cast_type: type = str) -> Any:
    """
    Get environment variable with type casting.

    Args:
        key: Environment variable name
        default: Default value if not found
        cast_type: Type to cast to (str, int, float, bool)

    Returns:
        Environment variable value with proper type
    """
    value = os.getenv(key)

    if value is None:
        return default

    # Type casting
    if cast_type == bool:
        return str_to_bool(value)
    elif cast_type == int:
        return int(value)
    elif cast_type == float:
        return float(value)
    else:
        return value


def load_config_from_env(config_class):
    """
    Load configuration from environment variables.

    This function reads environment variables and updates the config class.
    Environment variable names should match the config field names.

    Args:
        config_class: Configuration class instance

    Returns:
        Updated configuration instance
    """
    # Load .env file if exists
    env_vars = load_env_file()

    # Set environment variables from .env file
    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value

    # Update config from environment variables
    config_dict = {}

    # Retrieval parameters
    config_dict['TOP_K_ANCHORS'] = get_env('TOP_K_ANCHORS', cast_type=int)
    config_dict['TOP_N_RETRIEVAL'] = get_env('TOP_N_RETRIEVAL', cast_type=int)
    config_dict['MAX_HOPS'] = get_env('MAX_HOPS', cast_type=int)
    config_dict['ENERGY_DECAY_RATE'] = get_env('ENERGY_DECAY_RATE', cast_type=float)
    config_dict['ANCHOR_ENERGY_DISTRIBUTION'] = get_env('ANCHOR_ENERGY_DISTRIBUTION')
    config_dict['MERGE_STRATEGY'] = get_env('MERGE_STRATEGY')
    config_dict['MERGE_OLD_WEIGHT'] = get_env('MERGE_OLD_WEIGHT', cast_type=float)

    # Graph construction
    config_dict['MAX_SIM_NEIGHBORS_INTRA_DOC'] = get_env('MAX_SIM_NEIGHBORS_INTRA_DOC', cast_type=int)
    config_dict['MAX_SIM_NEIGHBORS_INTER_DOC'] = get_env('MAX_SIM_NEIGHBORS_INTER_DOC', cast_type=int)
    config_dict['CAUSE_WINDOW_SIZE'] = get_env('CAUSE_WINDOW_SIZE', cast_type=int)
    config_dict['CAUSE_NEIGHBOR_HOPS'] = get_env('CAUSE_NEIGHBOR_HOPS', cast_type=int)

    # Plasticity
    config_dict['HEBBIAN_LEARNING_RATE'] = get_env('HEBBIAN_LEARNING_RATE', cast_type=float)
    config_dict['TIME_DECAY_FACTOR'] = get_env('TIME_DECAY_FACTOR', cast_type=float)
    config_dict['MIN_EDGE_WEIGHT'] = get_env('MIN_EDGE_WEIGHT', cast_type=float)
    config_dict['LTP_ACTIVATION_THRESHOLD'] = get_env('LTP_ACTIVATION_THRESHOLD', cast_type=float)

    # Node configuration
    config_dict['USE_ENTITY_NODES'] = get_env('USE_ENTITY_NODES', cast_type=bool)
    config_dict['USE_CHUNK_NODES'] = get_env('USE_CHUNK_NODES', cast_type=bool)

    # LLM configuration
    config_dict['llm_name'] = get_env('LLM_NAME')
    config_dict['llm_base_url'] = get_env('LLM_BASE_URL')
    config_dict['llm_temperature'] = get_env('LLM_TEMPERATURE', cast_type=float)
    config_dict['llm_max_new_tokens'] = get_env('LLM_MAX_NEW_TOKENS', cast_type=int)

    # Embedding configuration
    config_dict['embedding_model_name'] = get_env('EMBEDDING_MODEL_NAME')
    config_dict['embedding_base_url'] = get_env('EMBEDDING_BASE_URL')
    config_dict['embedding_batch_size'] = get_env('EMBEDDING_BATCH_SIZE', cast_type=int)

    # Text processing
    config_dict['chunk_size'] = get_env('CHUNK_SIZE', cast_type=int)
    config_dict['chunk_overlap'] = get_env('CHUNK_OVERLAP', cast_type=int)

    # Storage
    config_dict['save_dir'] = get_env('SAVE_DIR')
    config_dict['force_index_from_scratch'] = get_env('FORCE_INDEX_FROM_SCRATCH', cast_type=bool)
    config_dict['cache_llm_responses'] = get_env('CACHE_LLM_RESPONSES', cast_type=bool)

    # Logging
    config_dict['log_level'] = get_env('LOG_LEVEL')

    # Remove None values
    config_dict = {k: v for k, v in config_dict.items() if v is not None}

    # Update config instance
    for key, value in config_dict.items():
        if hasattr(config_class, key):
            setattr(config_class, key, value)

    return config_class


__all__ = ['load_env_file', 'get_env', 'load_config_from_env', 'str_to_bool']
