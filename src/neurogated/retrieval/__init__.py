"""
Neuro Retriever - Main retrieval engine.

Combines KernelGenerator and SpreadingActivation to implement the
"Query as Modulation Kernel" + "Retrieval as Energy Flow" paradigm.
"""

import logging
from typing import List

from ..data_structures import RetrievalResult
from ..storage import GraphStore
from ..llm import BaseLLM
from ..config.memory_config import MemoryConfig
from .kernel_generator import KernelGenerator
from .spreading_activation import SpreadingActivation

logger = logging.getLogger(__name__)


class NeuroRetriever:
    """
    Main retrieval engine combining kernel generation and spreading activation.
    """

    def __init__(self, graph_store: GraphStore, llm: BaseLLM, config: MemoryConfig):
        """
        Initialize neuro retriever.

        Args:
            graph_store: Graph storage instance
            llm: LLM instance for kernel generation
            config: Memory configuration
        """
        self.graph_store = graph_store
        self.config = config

        # Initialize components
        self.kernel_generator = KernelGenerator(llm)
        self.spreading_activation = SpreadingActivation(graph_store, config)

    def retrieve(self, query: str, query_embedding: List[float]) -> RetrievalResult:
        """
        Retrieve memory nodes for a query.

        Pipeline:
        1. Generate modulation kernel from query (LLM)
        2. Perform spreading activation with kernel
        3. Return top-N chunks

        Args:
            query: Query text
            query_embedding: Query vector

        Returns:
            RetrievalResult with retrieved nodes
        """
        logger.info(f"Retrieving for query: {query[:100]}...")

        # Step 1: Generate kernel
        kernel = self.kernel_generator.generate(query)

        # Step 2: Spreading activation
        result = self.spreading_activation.retrieve(
            query_embedding=query_embedding,
            kernel=kernel,
            return_chunks_only=True
        )

        logger.info(f"Retrieved {len(result.nodes)} nodes")

        return result

    def get_activated_edges(self):
        """Get edges activated in the last retrieval (for LTP)"""
        return self.spreading_activation.get_activated_edges()


__all__ = ["NeuroRetriever", "KernelGenerator", "SpreadingActivation"]
