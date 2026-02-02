"""
Graph storage layer using python-igraph.

This module manages the graph structure (nodes + edges) with support for
different node types and edge types.
"""

import os
import logging
from typing import List, Dict, Tuple, Optional, Set
import igraph as ig
import pickle
from collections import defaultdict

from ..data_structures import MemoryNode, MemoryEdge, NodeType, EdgeType
from ..utils.hash_utils import retrieve_knn

logger = logging.getLogger(__name__)


class GraphStore:
    """
    Graph storage using igraph for efficient graph operations.

    Manages nodes (Chunk + Entity) and edges (SEQ + SIM + CAUSE) with
    support for persistence and vector search.
    """

    def __init__(self, save_dir: str, force_rebuild: bool = False):
        """
        Initialize graph store.

        Args:
            save_dir: Directory for saving graph
            force_rebuild: If True, ignore existing graph and rebuild
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.graph_path = os.path.join(save_dir, "graph.pickle")

        # Node and edge mappings
        self.nodes: Dict[str, MemoryNode] = {}  # node_id -> MemoryNode
        self.edges: Dict[Tuple[str, str, EdgeType], MemoryEdge] = {}  # (src, tgt, type) -> MemoryEdge

        # Initialize graph
        if not force_rebuild and os.path.exists(self.graph_path):
            self._load_graph()
        else:
            self.graph = ig.Graph(directed=False)  # Undirected graph
            logger.info("Initialized new graph")

    def add_node(self, node: MemoryNode) -> bool:
        """
        Add a node to the graph.

        Args:
            node: MemoryNode to add

        Returns:
            True if node was added, False if it already exists
        """
        if node.id in self.nodes:
            return False

        self.nodes[node.id] = node

        # Add to igraph
        self.graph.add_vertex(name=node.id, node_type=node.node_type.value)

        return True

    def add_edge(self, edge: MemoryEdge) -> bool:
        """
        Add an edge to the graph.

        Args:
            edge: MemoryEdge to add

        Returns:
            True if edge was added, False if it already exists
        """
        edge_key = (edge.source_id, edge.target_id, edge.edge_type)

        if edge_key in self.edges:
            # Update existing edge weight (for accumulation)
            self.edges[edge_key].weight = min(1.0, self.edges[edge_key].weight + edge.weight)
            return False

        # Check nodes exist
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            logger.warning(f"Cannot add edge: nodes {edge.source_id} or {edge.target_id} not found")
            return False

        self.edges[edge_key] = edge

        # Add to igraph
        try:
            self.graph.add_edge(
                edge.source_id,
                edge.target_id,
                edge_type=edge.edge_type.value,
                weight=edge.weight
            )
        except Exception as e:
            logger.error(f"Failed to add edge to igraph: {e}")
            return False

        return True

    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        """Get node by ID"""
        return self.nodes.get(node_id)

    def get_edge(self, source_id: str, target_id: str, edge_type: Optional[EdgeType] = None) -> Optional[MemoryEdge]:
        """
        Get edge between two nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Optional specific edge type to look for

        Returns:
            MemoryEdge if found, None otherwise
        """
        if edge_type is not None:
            return self.edges.get((source_id, target_id, edge_type))

        # Search for any edge type between these nodes
        for etype in EdgeType:
            edge = self.edges.get((source_id, target_id, etype))
            if edge:
                return edge
        return None

    def get_neighbors(self, node_id: str, edge_type: Optional[EdgeType] = None) -> List[Tuple[str, MemoryEdge]]:
        """
        Get neighbors of a node.

        Args:
            node_id: Node ID
            edge_type: Optional filter by edge type

        Returns:
            List of (neighbor_id, edge) tuples
        """
        if node_id not in self.nodes:
            return []

        neighbors = []

        try:
            vertex = self.graph.vs.find(name=node_id)
            neighbor_vertices = self.graph.neighbors(vertex)

            for neighbor_idx in neighbor_vertices:
                neighbor_id = self.graph.vs[neighbor_idx]["name"]

                # Find edge
                for (src, tgt, etype), edge in self.edges.items():
                    if (src == node_id and tgt == neighbor_id) or (src == neighbor_id and tgt == node_id):
                        if edge_type is None or etype == edge_type:
                            neighbors.append((neighbor_id, edge))

        except Exception as e:
            logger.error(f"Failed to get neighbors for {node_id}: {e}")

        return neighbors

    def vector_search(
        self,
        query_embedding: List[float],
        node_type: Optional[NodeType] = None,
        top_k: int = 5
    ) -> List[Tuple[MemoryNode, float]]:
        """
        Vector search for nearest neighbors.

        Args:
            query_embedding: Query vector
            node_type: Optional filter by node type
            top_k: Number of results to return

        Returns:
            List of (node, similarity_score) tuples
        """
        # Filter nodes by type
        candidate_nodes = [
            node for node in self.nodes.values()
            if (node_type is None or node.node_type == node_type) and node.embedding is not None
        ]

        if not candidate_nodes:
            return []

        # Extract embeddings
        embeddings = [node.embedding for node in candidate_nodes]

        # KNN search
        indices, scores = retrieve_knn(query_embedding, embeddings, min(top_k, len(embeddings)))

        # Return nodes with scores
        results = [(candidate_nodes[idx], scores[i]) for i, idx in enumerate(indices)]

        return results

    def get_all_edges(self, edge_type: Optional[EdgeType] = None) -> List[MemoryEdge]:
        """
        Get all edges, optionally filtered by type.

        Args:
            edge_type: Optional filter by edge type

        Returns:
            List of edges
        """
        if edge_type is None:
            return list(self.edges.values())
        else:
            return [edge for (_, _, etype), edge in self.edges.items() if etype == edge_type]

    def remove_edge(self, source_id: str, target_id: str, edge_type: EdgeType) -> bool:
        """
        Remove an edge from the graph.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Edge type

        Returns:
            True if edge was removed
        """
        edge_key = (source_id, target_id, edge_type)

        if edge_key not in self.edges:
            return False

        del self.edges[edge_key]

        # Remove from igraph
        try:
            edge_id = self.graph.get_eid(source_id, target_id)
            self.graph.delete_edges([edge_id])
        except Exception as e:
            logger.warning(f"Failed to remove edge from igraph: {e}")

        return True

    def save_graph(self):
        """Save graph to disk"""
        try:
            # Save igraph
            self.graph.write_pickle(self.graph_path)

            # Save node and edge mappings
            mappings_path = os.path.join(self.save_dir, "mappings.pkl")
            with open(mappings_path, "wb") as f:
                pickle.dump({
                    "nodes": self.nodes,
                    "edges": self.edges
                }, f)

            logger.info(f"Saved graph with {len(self.nodes)} nodes and {len(self.edges)} edges")

        except Exception as e:
            logger.error(f"Failed to save graph: {e}")

    def _load_graph(self):
        """Load graph from disk"""
        try:
            # Load igraph
            self.graph = ig.Graph.Read_Pickle(self.graph_path)

            # Load mappings
            mappings_path = os.path.join(self.save_dir, "mappings.pkl")
            with open(mappings_path, "rb") as f:
                data = pickle.load(f)
                self.nodes = data["nodes"]
                self.edges = data["edges"]

            logger.info(f"Loaded graph with {len(self.nodes)} nodes and {len(self.edges)} edges")

        except Exception as e:
            logger.error(f"Failed to load graph: {e}")
            self.graph = ig.Graph(directed=False)

    def get_stats(self) -> Dict[str, any]:
        """Get graph statistics"""
        chunk_count = sum(1 for node in self.nodes.values() if node.node_type == NodeType.CHUNK)
        entity_count = sum(1 for node in self.nodes.values() if node.node_type == NodeType.ENTITY)

        edge_counts = defaultdict(int)
        for (_, _, etype), _ in self.edges.items():
            edge_counts[etype.value] += 1

        return {
            "total_nodes": len(self.nodes),
            "chunk_nodes": chunk_count,
            "entity_nodes": entity_count,
            "total_edges": len(self.edges),
            "seq_edges": edge_counts.get("sequential", 0),
            "sim_edges": edge_counts.get("similarity", 0),
            "cause_edges": edge_counts.get("causal", 0)
        }
