"""
Ingestion layer for the Neuro-Gated Graph Memory System.
"""

from .ingestion_engine import IngestionEngine
from .chunker import TextChunker
from .entity_extractor import EntityExtractor
from .edge_builder import EdgeBuilder

__all__ = ["IngestionEngine", "TextChunker", "EntityExtractor", "EdgeBuilder"]
