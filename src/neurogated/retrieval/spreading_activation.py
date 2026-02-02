"""
Spreading Activation - Core retrieval algorithm.

This module implements the energy diffusion process across the memory graph,
following the design from DESIGN.md.
"""

import logging
from typing import List, Dict, Tuple, Set
from collections import defaultdict

from ..data_structures import MemoryNode, MemoryEdge, NodeType, EdgeType, RetrievalResult
from ..storage import GraphStore
from ..utils.hash_utils import softmax
from ..config.memory_config import MemoryConfig

logger = logging.getLogger(__name__)


class SpreadingActivation:
    """
    Implements the spreading activation algorithm for memory retrieval.

    Energy flows from initial anchor nodes through the graph, modulated by
    the kernel weights and edge weights, following the formula:
    Flow = Energy * EdgeWeight * KernelMod * DecayRate
    """

    def __init__(self, graph_store: GraphStore, config: MemoryConfig):
        """
        Initialize spreading activation.

        Args:
            graph_store: Graph storage instance
            config: Memory configuration
        """
        self.graph_store = graph_store
        self.config = config

        # Track activated edges for LTP
        self.activated_edges: List[MemoryEdge] = []

    def retrieve(
        self,
        query_embedding: List[float],
        kernel: Dict,
        return_chunks_only: bool = True
    ) -> RetrievalResult:
        """
        Perform spreading activation retrieval.

        Args:
            query_embedding: Query vector
            kernel: Modulation kernel from KernelGenerator
            return_chunks_only: If True, only return chunk nodes

        Returns:
            RetrievalResult with retrieved nodes and scores
        """
        # Reset activated edges tracking
        self.activated_edges = []

        # Step 1: Priming - Find initial anchor nodes (mixed: Chunk + Entity)
        anchors = self._get_anchor_nodes(query_embedding)

        if not anchors:
            logger.warning("No anchor nodes found")
            return RetrievalResult(node_ids=[], nodes=[], scores=[], metadata={})

        # Step 2: Initialize energy pool with Softmax distribution
        activations = self._initialize_activations(anchors)

        logger.info(f"Initialized {len(activations)} anchor nodes")

        # Step 3: Spreading activation loop
        for hop in range(self.config.MAX_HOPS):
            activations = self._propagate_energy(activations, kernel, hop)
            logger.debug(f"Hop {hop + 1}: {len(activations)} active nodes")

        # Step 4: Harvest results
        result = self._harvest_results(activations, return_chunks_only)

        # Add metadata
        result.metadata["kernel"] = kernel
        result.metadata["num_hops"] = self.config.MAX_HOPS
        result.metadata["num_activated_edges"] = len(self.activated_edges)

        return result

    def _get_anchor_nodes(self, query_embedding: List[float]) -> List[Tuple[MemoryNode, float]]:
        """
        Get initial anchor nodes using vector search (mixed: Chunk + Entity).

        Args:
            query_embedding: Query vector

        Returns:
            List of (node, similarity_score) tuples
        """
        # Search both Chunk and Entity nodes
        chunk_anchors = self.graph_store.vector_search(
            query_embedding,
            node_type=NodeType.CHUNK,
            top_k=self.config.TOP_K_ANCHORS
        )

        entity_anchors = self.graph_store.vector_search(
            query_embedding,
            node_type=NodeType.ENTITY,
            top_k=self.config.TOP_K_ANCHORS
        )

        # Combine and sort by score
        all_anchors = chunk_anchors + entity_anchors
        all_anchors.sort(key=lambda x: x[1], reverse=True)

        # Take top-K overall
        return all_anchors[:self.config.TOP_K_ANCHORS]

    def _initialize_activations(self, anchors: List[Tuple[MemoryNode, float]]) -> Dict[str, float]:
        """
        Initialize energy pool with Softmax distribution.

        Args:
            anchors: List of (node, similarity_score) tuples

        Returns:
            Dict mapping node_id -> energy
        """
        # Extract scores
        scores = [score for _, score in anchors]

        # Apply softmax
        if self.config.ANCHOR_ENERGY_DISTRIBUTION == "softmax":
            energies = softmax(scores)
        elif self.config.ANCHOR_ENERGY_DISTRIBUTION == "normalized":
            total = sum(scores)
            energies = [s / total for s in scores]
        elif self.config.ANCHOR_ENERGY_DISTRIBUTION == "uniform":
            energies = [1.0 / len(anchors)] * len(anchors)
        else:
            energies = softmax(scores)  # Default

        # Create activation dict
        activations = {}
        for (node, _), energy in zip(anchors, energies):
            activations[node.id] = energy

        return activations

    def _propagate_energy(
        self,
        activations: Dict[str, float],
        kernel: Dict,
        hop: int
    ) -> Dict[str, float]:
        """
        Propagate energy for one hop.

        Args:
            activations: Current activation levels
            kernel: Modulation kernel
            hop: Current hop number

        Returns:
            Updated activations
        """
        new_activations = defaultdict(float)

        # For each active node
        for node_id, energy in activations.items():
            if energy < 1e-6:  # Skip negligible energy
                continue

            # Get neighbors
            neighbors = self.graph_store.get_neighbors(node_id)

            # Propagate to each neighbor
            for neighbor_id, edge in neighbors:
                # Get kernel modulation for this edge type
                kernel_mod = kernel["weights"].get(edge.edge_type, 1.0)

                # Calculate energy flow
                flow = energy * edge.weight * kernel_mod * self.config.ENERGY_DECAY_RATE

                # Accumulate energy (simple addition as per design)
                new_activations[neighbor_id] += flow

                # Track activated edge if flow exceeds threshold
                if flow > self.config.LTP_ACTIVATION_THRESHOLD:
                    if self.config.TRACK_ACTIVATION_PATH:
                        edge.activate()  # Mark as activated
                        self.activated_edges.append(edge)

        # Merge with old activations using weighted average
        if self.config.MERGE_STRATEGY == "weighted_average":
            merged = {}
            all_nodes = set(activations.keys()) | set(new_activations.keys())

            for node_id in all_nodes:
                old_energy = activations.get(node_id, 0.0)
                new_energy = new_activations.get(node_id, 0.0)

                merged[node_id] = (
                    self.config.MERGE_OLD_WEIGHT * old_energy +
                    (1 - self.config.MERGE_OLD_WEIGHT) * new_energy
                )

            return merged

        elif self.config.MERGE_STRATEGY == "replace":
            return dict(new_activations)

        elif self.config.MERGE_STRATEGY == "accumulate":
            merged = dict(activations)
            for node_id, energy in new_activations.items():
                merged[node_id] = merged.get(node_id, 0.0) + energy
            return merged

        elif self.config.MERGE_STRATEGY == "max":
            merged = dict(activations)
            for node_id, energy in new_activations.items():
                merged[node_id] = max(merged.get(node_id, 0.0), energy)
            return merged

        else:
            # Default to weighted average
            return dict(new_activations)

    def _harvest_results(
        self,
        activations: Dict[str, float],
        return_chunks_only: bool
    ) -> RetrievalResult:
        """
        Harvest final results from activations.

        Strategy: Return top-k Chunks + Chunks connected to top Entity nodes, deduplicated.

        Args:
            activations: Final activation levels
            return_chunks_only: If True, only return chunk nodes

        Returns:
            RetrievalResult
        """
        # Separate activations by node type
        chunk_activations = {}
        entity_activations = {}

        for node_id, energy in activations.items():
            node = self.graph_store.get_node(node_id)
            if node is None:
                continue

            if node.node_type == NodeType.CHUNK:
                chunk_activations[node_id] = energy
            elif node.node_type == NodeType.ENTITY:
                entity_activations[node_id] = energy

        # Get top-k chunks directly
        top_chunks = sorted(chunk_activations.items(), key=lambda x: x[1], reverse=True)
        top_chunk_ids = set([node_id for node_id, _ in top_chunks[:self.config.TOP_N_RETRIEVAL]])

        # Get chunks connected to top entities (entity routing)
        top_entities = sorted(entity_activations.items(), key=lambda x: x[1], reverse=True)
        top_entity_ids = [node_id for node_id, _ in top_entities[:self.config.TOP_N_RETRIEVAL]]

        routed_chunk_ids = set()
        for entity_id in top_entity_ids:
            neighbors = self.graph_store.get_neighbors(entity_id)
            for neighbor_id, _ in neighbors:
                neighbor = self.graph_store.get_node(neighbor_id)
                if neighbor and neighbor.node_type == NodeType.CHUNK:
                    routed_chunk_ids.add(neighbor_id)

        # Merge and deduplicate
        final_chunk_ids = top_chunk_ids | routed_chunk_ids

        # Sort by activation score
        final_chunks = []
        for chunk_id in final_chunk_ids:
            node = self.graph_store.get_node(chunk_id)
            score = chunk_activations.get(chunk_id, 0.0)
            if node:
                final_chunks.append((node, score))

        final_chunks.sort(key=lambda x: x[1], reverse=True)

        # Take top-N
        final_chunks = final_chunks[:self.config.TOP_N_RETRIEVAL]

        # Build result
        nodes = [node for node, _ in final_chunks]
        scores = [score for _, score in final_chunks]
        node_ids = [node.id for node in nodes]

        return RetrievalResult(
            node_ids=node_ids,
            nodes=nodes,
            scores=scores
        )

    def get_activated_edges(self) -> List[MemoryEdge]:
        """
        Get edges that were activated during the last retrieval.

        Returns:
            List of activated edges (for LTP)
        """
        return self.activated_edges
