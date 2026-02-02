"""
Ingestion Engine - Document to Graph Pipeline.
"""

import logging
from typing import List, Optional

from ..data_structures import MemoryNode, MemoryEdge, NodeType
from ..storage import GraphStore, EmbeddingStore
from ..llm import BaseLLM
from ..config.memory_config import MemoryConfig
from ..utils.hash_utils import compute_hash_id
from .chunker import TextChunker
from .entity_extractor import EntityExtractor
from .edge_builder import EdgeBuilder

logger = logging.getLogger(__name__)


class IngestionEngine:
    """
    Converts documents into graph structure.

    Pipeline:
    1. Chunk document
    2. Extract entities (LLM-based NER)
    3. Create nodes (Chunk + Entity)
    4. Build edges (SEQ + SIM + CAUSE)
    """

    def __init__(
        self,
        graph_store: GraphStore,
        chunk_embedding_store: EmbeddingStore,
        entity_embedding_store: EmbeddingStore,
        llm: BaseLLM,
        config: MemoryConfig
    ):
        """
        Initialize ingestion engine.

        Args:
            graph_store: Graph storage
            chunk_embedding_store: Chunk embedding storage
            entity_embedding_store: Entity embedding storage
            llm: LLM instance
            config: Memory configuration
        """
        self.graph_store = graph_store
        self.chunk_embedding_store = chunk_embedding_store
        self.entity_embedding_store = entity_embedding_store
        self.llm = llm
        self.config = config

        # Initialize components
        self.chunker = TextChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        self.entity_extractor = EntityExtractor(llm)
        self.edge_builder = EdgeBuilder(llm, config)

    def ingest_document(self, text: str, document_id: Optional[str] = None) -> dict:
        """
        Ingest a document into the graph.

        Args:
            text: Document text
            document_id: Optional document identifier

        Returns:
            Dict with ingestion statistics
        """
        logger.info(f"Ingesting document: {document_id or 'unnamed'}")

        stats = {
            "document_id": document_id,
            "chunk_nodes_created": 0,
            "entity_nodes_created": 0,
            "seq_edges_created": 0,
            "sim_edges_created": 0,
            "cause_edges_created": 0
        }

        # Step 1: Chunk document
        chunks = self.chunker.chunk(text, document_id)
        logger.info(f"Created {len(chunks)} chunks")

        # Step 2: Create chunk nodes
        chunk_nodes = []
        chunk_texts = []

        for chunk_text, metadata in chunks:
            # Compute hash ID
            node_id = compute_hash_id(chunk_text, prefix="chunk-")

            # Check if already exists
            if self.graph_store.get_node(node_id):
                logger.debug(f"Chunk node {node_id[:8]} already exists, skipping")
                continue

            # Create node (embedding will be added later)
            node = MemoryNode(
                id=node_id,
                node_type=NodeType.CHUNK,
                content=chunk_text,
                metadata=metadata
            )

            chunk_nodes.append(node)
            chunk_texts.append(chunk_text)

        # Batch encode chunks
        if chunk_texts:
            logger.info(f"Encoding {len(chunk_texts)} chunks...")
            self.chunk_embedding_store.insert_strings(chunk_texts)

            # Get embeddings and add to nodes
            for node, text in zip(chunk_nodes, chunk_texts):
                hash_id = compute_hash_id(text, prefix="chunk-")
                idx = self.chunk_embedding_store.hash_id_to_idx.get(hash_id)
                if idx is not None:
                    node.embedding = self.chunk_embedding_store.embeddings[idx]

            # Add nodes to graph
            for node in chunk_nodes:
                self.graph_store.add_node(node)

            stats["chunk_nodes_created"] = len(chunk_nodes)

        # Step 3: Extract entities
        if self.config.USE_ENTITY_NODES:
            entity_nodes = self._extract_and_create_entity_nodes(chunk_texts, document_id)
            stats["entity_nodes_created"] = len(entity_nodes)
        else:
            entity_nodes = []

        # Step 4: Build SEQ edges (Chunk → Chunk)
        if chunk_nodes:
            seq_edges = self.edge_builder.build_seq_edges(chunk_nodes)
            for edge in seq_edges:
                self.graph_store.add_edge(edge)
            stats["seq_edges_created"] = len(seq_edges)

        # Step 5: Build SIM edges (Chunk ↔ Chunk, Entity ↔ Entity)
        # Get existing nodes for similarity comparison
        existing_chunk_nodes = [
            node for node in self.graph_store.nodes.values()
            if node.node_type == NodeType.CHUNK and node.id not in [n.id for n in chunk_nodes]
        ]
        existing_entity_nodes = [
            node for node in self.graph_store.nodes.values()
            if node.node_type == NodeType.ENTITY and node.id not in [n.id for n in entity_nodes]
        ]

        # Build SIM edges for chunks
        if chunk_nodes:
            chunk_sim_edges = self.edge_builder.build_sim_edges(chunk_nodes, existing_chunk_nodes)
            for edge in chunk_sim_edges:
                self.graph_store.add_edge(edge)
            stats["sim_edges_created"] += len(chunk_sim_edges)

        # Build SIM edges for entities
        if entity_nodes:
            entity_sim_edges = self.edge_builder.build_sim_edges(entity_nodes, existing_entity_nodes)
            for edge in entity_sim_edges:
                self.graph_store.add_edge(edge)
            stats["sim_edges_created"] += len(entity_sim_edges)

        # Step 6: Build CAUSE edges (optional, controlled by config)
        if not self.config.SKIP_CAUSE_EDGES:
            all_nodes = chunk_nodes + entity_nodes
            existing_nodes = list(self.graph_store.nodes.values())
            # Filter out nodes we just added
            existing_nodes = [n for n in existing_nodes if n.id not in [node.id for node in all_nodes]]

            if all_nodes:
                cause_edges = self.edge_builder.build_cause_edges(all_nodes, existing_nodes)
                for edge in cause_edges:
                    self.graph_store.add_edge(edge)
                stats["cause_edges_created"] = len(cause_edges)

        logger.info(f"Ingestion complete: {stats}")

        return stats

    def _extract_and_create_entity_nodes(
        self,
        chunk_texts: List[str],
        document_id: Optional[str]
    ) -> List[MemoryNode]:
        """
        Extract entities and create entity nodes.

        Args:
            chunk_texts: List of chunk texts
            document_id: Document identifier

        Returns:
            List of entity nodes
        """
        entity_nodes = []
        all_entities = set()

        # Extract entities from all chunks
        for chunk_text in chunk_texts:
            entities = self.entity_extractor.extract(chunk_text)
            all_entities.update(entities)

        logger.info(f"Extracted {len(all_entities)} unique entities")

        # Create entity nodes
        entity_texts = list(all_entities)

        if entity_texts:
            # Batch encode entities
            self.entity_embedding_store.insert_strings(entity_texts)

            # Create nodes
            for entity_text in entity_texts:
                node_id = compute_hash_id(entity_text, prefix="entity-")

                # Check if already exists
                if self.graph_store.get_node(node_id):
                    continue

                # Get embedding
                hash_id = compute_hash_id(entity_text, prefix="entity-")
                idx = self.entity_embedding_store.hash_id_to_idx.get(hash_id)
                embedding = None
                if idx is not None:
                    embedding = self.entity_embedding_store.embeddings[idx]

                # Create node
                node = MemoryNode(
                    id=node_id,
                    node_type=NodeType.ENTITY,
                    content=entity_text,
                    embedding=embedding,
                    metadata={"document_id": document_id}
                )

                entity_nodes.append(node)
                self.graph_store.add_node(node)

        return entity_nodes


__all__ = ["IngestionEngine"]
