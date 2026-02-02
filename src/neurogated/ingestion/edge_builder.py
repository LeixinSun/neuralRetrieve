"""
Edge builder for constructing SEQ/SIM/CAUSE edges.
"""

import json
import logging
import re
from typing import List, Tuple, Dict
import numpy as np

from ..data_structures import MemoryNode, MemoryEdge, NodeType, EdgeType
from ..llm import BaseLLM
from ..prompts.prompt_manager import PromptTemplateManager
from ..utils.hash_utils import cosine_sim
from ..config.memory_config import MemoryConfig

logger = logging.getLogger(__name__)


class EdgeBuilder:
    """
    Builds edges between nodes following the design specifications.
    """

    def __init__(self, llm: BaseLLM, config: MemoryConfig):
        """
        Initialize edge builder.

        Args:
            llm: LLM instance for causal detection
            config: Memory configuration
        """
        self.llm = llm
        self.config = config
        self.prompt_manager = PromptTemplateManager()

    def build_seq_edges(self, chunk_nodes: List[MemoryNode]) -> List[MemoryEdge]:
        """
        Build sequential edges between consecutive chunks.

        Args:
            chunk_nodes: List of chunk nodes in order

        Returns:
            List of SEQ edges
        """
        edges = []

        for i in range(len(chunk_nodes) - 1):
            edge = MemoryEdge(
                source_id=chunk_nodes[i].id,
                target_id=chunk_nodes[i + 1].id,
                edge_type=EdgeType.SEQ,
                weight=1.0,
                metadata={"edge_builder": "seq"}
            )
            edges.append(edge)

        logger.info(f"Built {len(edges)} SEQ edges")
        return edges

    def build_sim_edges(
        self,
        nodes: List[MemoryNode],
        existing_nodes: List[MemoryNode] = None
    ) -> List[MemoryEdge]:
        """
        Build similarity edges using layered strategy:
        - Intra-document: MAX_SIM_NEIGHBORS_INTRA_DOC
        - Inter-document: MAX_SIM_NEIGHBORS_INTER_DOC

        Args:
            nodes: New nodes to connect
            existing_nodes: Existing nodes in the graph

        Returns:
            List of SIM edges
        """
        edges = []

        if existing_nodes is None:
            existing_nodes = []

        all_nodes = existing_nodes + nodes

        for node in nodes:
            if node.embedding is None:
                continue

            # Separate intra-doc and inter-doc candidates
            same_doc_nodes = []
            other_doc_nodes = []

            node_doc_id = node.metadata.get("document_id")

            for candidate in all_nodes:
                if candidate.id == node.id:
                    continue
                if candidate.embedding is None:
                    continue
                if candidate.node_type != node.node_type:
                    continue  # Only connect same type (Chunk-Chunk or Entity-Entity)

                candidate_doc_id = candidate.metadata.get("document_id")

                if node_doc_id and candidate_doc_id == node_doc_id:
                    same_doc_nodes.append(candidate)
                else:
                    other_doc_nodes.append(candidate)

            # Build intra-doc edges
            intra_edges = self._build_sim_edges_for_candidates(
                node,
                same_doc_nodes,
                self.config.MAX_SIM_NEIGHBORS_INTRA_DOC
            )
            edges.extend(intra_edges)

            # Build inter-doc edges
            inter_edges = self._build_sim_edges_for_candidates(
                node,
                other_doc_nodes,
                self.config.MAX_SIM_NEIGHBORS_INTER_DOC
            )
            edges.extend(inter_edges)

        logger.info(f"Built {len(edges)} SIM edges")
        return edges

    def _build_sim_edges_for_candidates(
        self,
        node: MemoryNode,
        candidates: List[MemoryNode],
        top_k: int
    ) -> List[MemoryEdge]:
        """
        Build SIM edges to top-K most similar candidates.

        Args:
            node: Source node
            candidates: Candidate nodes
            top_k: Number of edges to create

        Returns:
            List of SIM edges
        """
        if not candidates or top_k == 0:
            return []

        # Compute similarities
        similarities = []
        for candidate in candidates:
            sim = cosine_sim(node.embedding, candidate.embedding)
            similarities.append((candidate, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Take top-K
        top_candidates = similarities[:top_k]

        # Create edges
        edges = []
        for candidate, sim in top_candidates:
            edge = MemoryEdge(
                source_id=node.id,
                target_id=candidate.id,
                edge_type=EdgeType.SIM,
                weight=float(sim),  # Use similarity as weight
                metadata={"edge_builder": "sim", "similarity": float(sim)}
            )
            edges.append(edge)

        return edges

    def build_cause_edges(
        self,
        nodes: List[MemoryNode],
        existing_nodes: List[MemoryNode] = None
    ) -> List[MemoryEdge]:
        """
        Build causal edges using hybrid strategy:
        1. Sliding window for local causality
        2. Graph neighbors for extended causality

        Args:
            nodes: New nodes
            existing_nodes: Existing nodes in the graph

        Returns:
            List of CAUSE edges
        """
        edges = []

        # Strategy 1: Sliding window
        window_edges = self._build_cause_edges_window(nodes)
        edges.extend(window_edges)

        # Strategy 2: Graph neighbors (TODO: requires graph structure)
        # For now, skip this part

        logger.info(f"Built {len(edges)} CAUSE edges")
        return edges

    def _build_cause_edges_window(self, nodes: List[MemoryNode]) -> List[MemoryEdge]:
        """
        Build causal edges using sliding window.

        Args:
            nodes: List of nodes

        Returns:
            List of CAUSE edges
        """
        edges = []
        window_size = self.config.CAUSE_WINDOW_SIZE

        for i in range(len(nodes)):
            # Get window of neighbors
            start = max(0, i - window_size // 2)
            end = min(len(nodes), i + window_size // 2 + 1)

            for j in range(start, end):
                if i == j:
                    continue

                # Check causality using LLM
                has_causality, confidence = self._detect_causality(nodes[i], nodes[j])

                if has_causality and confidence > 0.5:
                    # Create bidirectional edges
                    edge1 = MemoryEdge(
                        source_id=nodes[i].id,
                        target_id=nodes[j].id,
                        edge_type=EdgeType.CAUSE,
                        weight=confidence,
                        metadata={"edge_builder": "cause", "confidence": confidence}
                    )
                    edge2 = MemoryEdge(
                        source_id=nodes[j].id,
                        target_id=nodes[i].id,
                        edge_type=EdgeType.CAUSE,
                        weight=confidence,
                        metadata={"edge_builder": "cause", "confidence": confidence}
                    )
                    edges.extend([edge1, edge2])

        return edges

    def _detect_causality(self, node_a: MemoryNode, node_b: MemoryNode) -> Tuple[bool, float]:
        """
        Detect causal relationship between two nodes using LLM.

        Args:
            node_a: First node
            node_b: Second node

        Returns:
            Tuple of (has_causality, confidence)
        """
        # Render prompt
        messages = self.prompt_manager.render(
            "causal_detection",
            text_a=node_a.content,
            text_b=node_b.content
        )

        # Call LLM
        response, metadata = self.llm.infer(messages)

        # Parse response
        try:
            result = self._parse_causality_response(response)
            return result["has_causality"], result["confidence"]

        except Exception as e:
            logger.warning(f"Failed to parse causality response: {e}")
            return False, 0.0

    def _parse_causality_response(self, response: str) -> Dict:
        """
        Parse LLM response for causality detection.

        Args:
            response: LLM response text

        Returns:
            Dict with has_causality and confidence
        """
        # Try to extract JSON
        json_match = re.search(r'\{[^{}]*"has_causality"[^{}]*\}', response, re.DOTALL)

        if json_match:
            data = json.loads(json_match.group())
            return {
                "has_causality": data.get("has_causality", False),
                "confidence": float(data.get("confidence", 0.0))
            }
        else:
            raise ValueError("Could not find JSON in response")
