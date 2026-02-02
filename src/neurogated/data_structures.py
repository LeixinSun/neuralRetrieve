"""
Core data structures for the Neuro-Gated Graph Memory System.

This module defines MemoryNode and MemoryEdge based on the aligned design decisions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional
from uuid import uuid4


class NodeType(Enum):
    """Types of memory nodes"""
    CHUNK = "chunk"      # Document chunks for similarity and final return
    ENTITY = "entity"    # Named entities for semantic connections


class EdgeType(Enum):
    """Types of memory edges"""
    SEQ = "sequential"   # Temporal sequence (Chunk → Chunk)
    SIM = "similarity"   # Semantic similarity (Chunk ↔ Chunk, Entity ↔ Entity)
    CAUSE = "causal"     # Causal logic (Chunk ↔ Chunk, Entity ↔ Entity)


@dataclass
class MemoryNode:
    """
    Represents a memory node (neuron) in the graph.

    Attributes:
        id: Unique identifier (hash-based for deduplication)
        node_type: Type of node (CHUNK or ENTITY)
        content: Text content
        embedding: Dense vector representation
        base_energy: Base energy level (decays over time if unused)
        last_accessed: Timestamp for forgetting curve
        metadata: Additional information (e.g., document_id, entity_type)
    """
    id: str
    node_type: NodeType
    content: str
    embedding: Optional[List[float]] = None
    base_energy: float = 1.0
    last_accessed: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate node after initialization"""
        assert self.base_energy >= 0, "base_energy must be non-negative"
        assert self.node_type in NodeType, f"Invalid node_type: {self.node_type}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "node_type": self.node_type.value,
            "content": self.content,
            "embedding": self.embedding,
            "base_energy": self.base_energy,
            "last_accessed": self.last_accessed.isoformat(),
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryNode":
        """Create from dictionary"""
        return cls(
            id=data["id"],
            node_type=NodeType(data["node_type"]),
            content=data["content"],
            embedding=data.get("embedding"),
            base_energy=data.get("base_energy", 1.0),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            metadata=data.get("metadata", {})
        )


@dataclass
class MemoryEdge:
    """
    Represents a synaptic connection between memory nodes.

    Attributes:
        source_id: ID of source node
        target_id: ID of target node
        edge_type: Type of edge (SEQ, SIM, or CAUSE)
        weight: Dynamic strength (0.0 - 1.0), subject to LTP/LTD
        created_at: Creation timestamp
        last_activated: Last time this edge was used in retrieval
        activation_count: Number of times this edge has been activated
        metadata: Additional information
    """
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    last_activated: Optional[datetime] = None
    activation_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate edge after initialization"""
        assert 0.0 <= self.weight <= 1.0, f"weight must be in [0, 1], got {self.weight}"
        assert self.edge_type in EdgeType, f"Invalid edge_type: {self.edge_type}"
        assert self.source_id != self.target_id, "Self-loops are not allowed"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "weight": self.weight,
            "created_at": self.created_at.isoformat(),
            "last_activated": self.last_activated.isoformat() if self.last_activated else None,
            "activation_count": self.activation_count,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEdge":
        """Create from dictionary"""
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            edge_type=EdgeType(data["edge_type"]),
            weight=data.get("weight", 1.0),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_activated=datetime.fromisoformat(data["last_activated"]) if data.get("last_activated") else None,
            activation_count=data.get("activation_count", 0),
            metadata=data.get("metadata", {})
        )

    def activate(self):
        """Mark this edge as activated (for LTP tracking)"""
        self.last_activated = datetime.now()
        self.activation_count += 1


@dataclass
class RetrievalResult:
    """
    Result of a retrieval operation.

    Attributes:
        node_ids: List of retrieved node IDs
        nodes: List of retrieved MemoryNode objects
        scores: Activation scores for each node
        metadata: Additional information (e.g., kernel weights, path traces)
    """
    node_ids: List[str]
    nodes: List[MemoryNode]
    scores: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_chunks(self) -> List[MemoryNode]:
        """Get only chunk nodes from results"""
        return [node for node in self.nodes if node.node_type == NodeType.CHUNK]

    def get_entities(self) -> List[MemoryNode]:
        """Get only entity nodes from results"""
        return [node for node in self.nodes if node.node_type == NodeType.ENTITY]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "node_ids": self.node_ids,
            "nodes": [node.to_dict() for node in self.nodes],
            "scores": self.scores,
            "metadata": self.metadata
        }
