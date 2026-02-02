"""
YAML configuration file loader.
"""

import yaml
from pathlib import Path
from typing import Optional
import os


def load_yaml_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def config_from_yaml(yaml_path: str = "config.yaml"):
    """
    Create MemoryConfig from YAML file.

    Args:
        yaml_path: Path to YAML config file

    Returns:
        MemoryConfig instance
    """
    from ..config.memory_config import MemoryConfig

    config_dict = load_yaml_config(yaml_path)

    # Extract and flatten configuration
    params = {}

    # API configuration
    if 'api' in config_dict:
        api = config_dict['api']

        # Set API key as environment variable
        if 'openai_api_key' in api and api['openai_api_key']:
            os.environ['OPENAI_API_KEY'] = api['openai_api_key']

        if 'llm' in api:
            params['llm_name'] = api['llm'].get('name')
            params['llm_base_url'] = api['llm'].get('base_url')
            params['llm_temperature'] = api['llm'].get('temperature')
            params['llm_max_new_tokens'] = api['llm'].get('max_new_tokens')

        if 'embedding' in api:
            params['embedding_model_name'] = api['embedding'].get('model_name')
            params['embedding_base_url'] = api['embedding'].get('base_url')
            params['embedding_batch_size'] = api['embedding'].get('batch_size')
            params['embedding_dimension'] = api['embedding'].get('dimension')

    # Retrieval configuration
    if 'retrieval' in config_dict:
        ret = config_dict['retrieval']
        params['TOP_K_ANCHORS'] = ret.get('top_k_anchors')
        params['TOP_N_RETRIEVAL'] = ret.get('top_n_retrieval')
        params['MAX_HOPS'] = ret.get('max_hops')
        params['ENERGY_DECAY_RATE'] = ret.get('energy_decay_rate')
        params['ANCHOR_ENERGY_DISTRIBUTION'] = ret.get('anchor_energy_distribution')
        params['MERGE_STRATEGY'] = ret.get('merge_strategy')
        params['MERGE_OLD_WEIGHT'] = ret.get('merge_old_weight')

    # Graph configuration
    if 'graph' in config_dict:
        graph = config_dict['graph']

        if 'sim_edges' in graph:
            params['MAX_SIM_NEIGHBORS_INTRA_DOC'] = graph['sim_edges'].get('max_neighbors_intra_doc')
            params['MAX_SIM_NEIGHBORS_INTER_DOC'] = graph['sim_edges'].get('max_neighbors_inter_doc')

        if 'cause_edges' in graph:
            params['CAUSE_WINDOW_SIZE'] = graph['cause_edges'].get('window_size')
            params['CAUSE_NEIGHBOR_HOPS'] = graph['cause_edges'].get('neighbor_hops')

        if 'nodes' in graph:
            params['USE_ENTITY_NODES'] = graph['nodes'].get('use_entity_nodes')
            params['USE_CHUNK_NODES'] = graph['nodes'].get('use_chunk_nodes')
            params['USE_TRIPLE_NODES'] = graph['nodes'].get('use_triple_nodes')

    # Plasticity configuration
    if 'plasticity' in config_dict:
        plas = config_dict['plasticity']

        if 'ltp' in plas:
            params['HEBBIAN_LEARNING_RATE'] = plas['ltp'].get('learning_rate')
            params['LTP_ACTIVATION_THRESHOLD'] = plas['ltp'].get('activation_threshold')
            params['TRACK_ACTIVATION_PATH'] = plas['ltp'].get('track_activation_path')

        if 'ltd' in plas:
            params['TIME_DECAY_FACTOR'] = plas['ltd'].get('time_decay_factor')
            params['MIN_EDGE_WEIGHT'] = plas['ltd'].get('min_edge_weight')

    # Text processing configuration
    if 'text_processing' in config_dict:
        text = config_dict['text_processing']

        if 'chunking' in text:
            params['chunk_size'] = text['chunking'].get('chunk_size')
            params['chunk_overlap'] = text['chunking'].get('chunk_overlap')

    # Storage configuration
    if 'storage' in config_dict:
        storage = config_dict['storage']
        params['save_dir'] = storage.get('save_dir')
        params['force_index_from_scratch'] = storage.get('force_index_from_scratch')
        params['cache_llm_responses'] = storage.get('cache_llm_responses')

    # System configuration
    if 'system' in config_dict:
        sys = config_dict['system']
        params['log_level'] = sys.get('log_level')
        params['verbose'] = sys.get('verbose')

        # Set environment variables
        if sys.get('hf_home'):
            os.environ['HF_HOME'] = sys['hf_home']
        if sys.get('cuda_visible_devices'):
            os.environ['CUDA_VISIBLE_DEVICES'] = sys['cuda_visible_devices']

    # Remove None values
    params = {k: v for k, v in params.items() if v is not None}

    # Create config
    return MemoryConfig(**params)


__all__ = ['load_yaml_config', 'config_from_yaml']
