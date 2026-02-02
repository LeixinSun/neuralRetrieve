"""
Plasticity Engine - LTP/LTD implementation.

This module implements Hebbian learning (LTP) and time-based forgetting (LTD)
for dynamic edge weight adjustment.
"""

import logging
from datetime import datetime, timedelta
from typing import List

from ..data_structures import MemoryEdge, EdgeType
from ..storage import GraphStore
from ..config.memory_config import MemoryConfig

logger = logging.getLogger(__name__)


class PlasticityEngine:
    """
    Implements synaptic plasticity mechanisms:
    - LTP (Long-Term Potentiation): Strengthen frequently used edges
    - LTD (Long-Term Depression): Weaken unused edges over time
    """

    def __init__(self, graph_store: GraphStore, config: MemoryConfig):
        """
        Initialize plasticity engine.

        Args:
            graph_store: Graph storage instance
            config: Memory configuration
        """
        self.graph_store = graph_store
        self.config = config

    def reinforce_path(self, activated_edges: List[MemoryEdge]):
        """
        Strengthen edges that were activated during retrieval (LTP).

        This implements Hebbian learning: "neurons that fire together, wire together"

        Args:
            activated_edges: List of edges that were activated
        """
        if not activated_edges:
            logger.debug("No edges to reinforce")
            return

        reinforced_count = 0

        for edge in activated_edges:
            # Increase weight
            old_weight = edge.weight
            new_weight = min(1.0, edge.weight + self.config.HEBBIAN_LEARNING_RATE)

            edge.weight = new_weight

            # Update in graph store
            # Note: The edge object is already updated in memory, but we log it
            if new_weight > old_weight:
                reinforced_count += 1

        logger.info(f"Reinforced {reinforced_count} edges via LTP")

    def decay_unused(self):
        """
        Apply time-based decay to unused edges (LTD).

        Special rule: SIM edges do NOT decay (semantic similarity is objective).

        This implements passive forgetting for edges that haven't been used recently.
        """
        current_time = datetime.now()
        decayed_count = 0
        pruned_count = 0

        # Get all edges
        all_edges = self.graph_store.get_all_edges()

        edges_to_remove = []

        for edge in all_edges:
            # Skip SIM edges (they don't decay)
            if edge.edge_type == EdgeType.SIM:
                continue

            # Check if edge was recently activated
            if edge.last_activated is None:
                # Never activated, use creation time
                time_since_use = current_time - edge.created_at
            else:
                time_since_use = current_time - edge.last_activated

            # Apply decay if not recently used (e.g., > 1 day)
            decay_threshold = timedelta(days=1)

            if time_since_use > decay_threshold:
                old_weight = edge.weight
                new_weight = edge.weight * self.config.TIME_DECAY_FACTOR

                edge.weight = new_weight
                decayed_count += 1

                # Prune if weight falls below threshold
                if new_weight < self.config.MIN_EDGE_WEIGHT:
                    edges_to_remove.append((edge.source_id, edge.target_id, edge.edge_type))
                    pruned_count += 1

        # Remove pruned edges
        for source_id, target_id, edge_type in edges_to_remove:
            self.graph_store.remove_edge(source_id, target_id, edge_type)

        logger.info(f"LTD: Decayed {decayed_count} edges, pruned {pruned_count} weak edges")

    def maintenance(self):
        """
        Perform regular maintenance (decay unused edges).

        This should be called periodically (e.g., after each retrieval or daily).
        """
        self.decay_unused()


__all__ = ["PlasticityEngine"]
