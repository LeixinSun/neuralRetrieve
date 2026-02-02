"""
Neuro-Gated Graph Memory System

A neurobiologically inspired retrieval mechanism implementing spreading activation
and dynamic edge weight modulation.
"""

__version__ = "0.1.0"

from .config.memory_config import MemoryConfig
from .core import NeuroGraphMemory
from .utils.yaml_loader import config_from_yaml

__all__ = ["MemoryConfig", "NeuroGraphMemory", "config_from_yaml"]
