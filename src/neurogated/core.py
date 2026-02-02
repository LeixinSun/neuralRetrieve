"""
Neuro-Gated Graph Memory System - Main Entry Point

This is the core class that integrates all components:
- Storage Layer (GraphStore + EmbeddingStore)
- Ingestion Layer (document → graph)
- Retrieval Layer (KernelGenerator + SpreadingActivation)
- Plasticity Layer (LTP/LTD)
"""

import logging
import os
from typing import List, Optional

from .config.memory_config import MemoryConfig
from .data_structures import MemoryNode, MemoryEdge, NodeType, EdgeType, RetrievalResult
from .storage import GraphStore, EmbeddingStore
from .llm import get_llm, BaseLLM
from .retrieval import NeuroRetriever
from .plasticity import PlasticityEngine
from .ingestion import IngestionEngine
from .embedding_model import _get_embedding_model_class

logger = logging.getLogger(__name__)


class NeuroGraphMemory:
    """
    Main class for the Neuro-Gated Graph Memory System.

    This class provides the high-level API for:
    - Adding documents (ingestion)
    - Retrieving information (spreading activation)
    - Learning from feedback (plasticity)
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        Initialize the memory system.

        Args:
            config: Memory configuration (uses defaults if None)
        """
        self.config = config or MemoryConfig()

        logger.info("Initializing Neuro-Gated Graph Memory System")
        logger.info(f"Save directory: {self.config.save_dir}")

        # Create save directory
        os.makedirs(self.config.save_dir, exist_ok=True)

        # Initialize LLM
        self.llm: BaseLLM = get_llm(self.config)

        # Initialize embedding model
        embedding_model_class = _get_embedding_model_class(self.config.embedding_model_name)
        self.embedding_model = embedding_model_class(
            global_config=self.config,
            embedding_model_name=self.config.embedding_model_name
        )

        # Initialize storage
        self.graph_store = GraphStore(
            save_dir=self.config.save_dir,
            force_rebuild=self.config.force_index_from_scratch
        )

        # Initialize embedding stores (Chunk + Entity)
        self.chunk_embedding_store = EmbeddingStore(
            embedding_model=self.embedding_model,
            db_filename=os.path.join(self.config.save_dir, "chunk_embeddings"),
            batch_size=self.config.embedding_batch_size,
            namespace="chunk"
        )

        self.entity_embedding_store = EmbeddingStore(
            embedding_model=self.embedding_model,
            db_filename=os.path.join(self.config.save_dir, "entity_embeddings"),
            batch_size=self.config.embedding_batch_size,
            namespace="entity"
        )

        # Initialize ingestion engine
        self.ingestion_engine = IngestionEngine(
            graph_store=self.graph_store,
            chunk_embedding_store=self.chunk_embedding_store,
            entity_embedding_store=self.entity_embedding_store,
            llm=self.llm,
            config=self.config
        )

        # Initialize retrieval engine
        self.retriever = NeuroRetriever(
            graph_store=self.graph_store,
            llm=self.llm,
            config=self.config
        )

        # Initialize plasticity engine
        self.plasticity = PlasticityEngine(
            graph_store=self.graph_store,
            config=self.config
        )

        logger.info("System initialized successfully")

    def add_document(self, text: str, document_id: Optional[str] = None) -> dict:
        """
        Add a document to the memory system.

        Pipeline:
        1. Chunk the document
        2. Extract entities (LLM-based NER)
        3. Build SEQ edges (Chunk → Chunk)
        4. Build SIM edges (Chunk ↔ Chunk, Entity ↔ Entity)
        5. Build CAUSE edges (LLM-driven)

        Args:
            text: Document text
            document_id: Optional document identifier

        Returns:
            Dict with ingestion statistics
        """
        logger.info(f"Adding document: {document_id or 'unnamed'}")

        # Use ingestion engine
        stats = self.ingestion_engine.ingest_document(text, document_id)

        return stats

    def retrieve(self, query: str) -> List[str]:
        """
        Retrieve relevant information for a query.

        Pipeline:
        1. Encode query to vector
        2. Generate modulation kernel (LLM)
        3. Perform spreading activation
        4. Return top-N chunk contents
        5. Trigger LTP on activated edges

        Args:
            query: Query text

        Returns:
            List of retrieved chunk contents
        """
        logger.info(f"Retrieving for query: {query[:100]}...")

        # Encode query
        query_embedding = self.embedding_model.encode([query])[0]

        # Perform retrieval
        result = self.retriever.retrieve(query, query_embedding)

        # Trigger LTP on activated edges
        activated_edges = self.retriever.get_activated_edges()
        self.plasticity.reinforce_path(activated_edges)

        # Extract content from chunks
        contents = [node.content for node in result.nodes]

        logger.info(f"Retrieved {len(contents)} chunks")

        return contents

    def feedback(self, relevant_node_ids: List[str]):
        """
        Provide feedback on relevant nodes (for additional LTP).

        This method strengthens edges between the specified relevant nodes,
        implementing explicit user feedback for Hebbian learning.

        Args:
            relevant_node_ids: List of node IDs that were relevant
        """
        logger.info(f"Received feedback for {len(relevant_node_ids)} nodes")

        if len(relevant_node_ids) < 2:
            logger.debug("Need at least 2 nodes for feedback reinforcement")
            return

        # Find edges between the relevant nodes and reinforce them
        edges_to_reinforce = []

        for i, source_id in enumerate(relevant_node_ids):
            for target_id in relevant_node_ids[i + 1:]:
                # Check if edge exists in either direction
                edge = self.graph_store.get_edge(source_id, target_id)
                if edge:
                    edges_to_reinforce.append(edge)

                edge_reverse = self.graph_store.get_edge(target_id, source_id)
                if edge_reverse:
                    edges_to_reinforce.append(edge_reverse)

        if edges_to_reinforce:
            self.plasticity.reinforce_path(edges_to_reinforce)
            logger.info(f"Reinforced {len(edges_to_reinforce)} edges based on feedback")

    def maintenance(self):
        """
        Perform system maintenance (LTD decay).

        This should be called periodically to prune weak edges.
        """
        logger.info("Running maintenance (LTD)")
        self.plasticity.maintenance()

    def save(self):
        """Save the memory system to disk"""
        logger.info("Saving memory system")
        self.graph_store.save_graph()
        # Embedding stores auto-save on insert

    def get_stats(self):
        """Get system statistics"""
        stats = self.graph_store.get_stats()
        logger.info(f"System stats: {stats}")
        return stats


__all__ = ["NeuroGraphMemory"]
