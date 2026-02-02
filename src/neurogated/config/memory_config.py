"""
Configuration module for Neuro-Gated Graph Memory System.

This module defines all hyperparameters and configuration options based on the
aligned design decisions from DESIGN.md discussions.

Configuration can be loaded from:
1. Environment variables
2. .env file
3. Direct instantiation with parameters
"""

from dataclasses import dataclass, field
from typing import Optional
import os


def _get_env_or_default(key: str, default, cast_type=str):
    """Get environment variable or return default"""
    value = os.getenv(key)
    if value is None:
        return default
    if cast_type == bool:
        return value.lower() in ('true', '1', 'yes', 'on')
    return cast_type(value)


@dataclass
class MemoryConfig:
    """
    Central configuration class for the Neuro-Gated Graph Memory System.
    All "magic numbers" are extracted here for easy tuning.
    """

    # ==================== Retrieval Parameters ====================
    TOP_K_ANCHORS: int = field(default_factory=lambda: _get_env_or_default('TOP_K_ANCHORS', 5, int))
    """Number of anchor nodes from vector search (initial priming)"""

    TOP_N_RETRIEVAL: int = field(default_factory=lambda: _get_env_or_default('TOP_N_RETRIEVAL', 3, int))
    """Final number of memory nodes to return"""

    MAX_HOPS: int = field(default_factory=lambda: _get_env_or_default('MAX_HOPS', 2, int))
    """Maximum hops for energy diffusion (iteration depth)"""

    ENERGY_DECAY_RATE: float = field(default_factory=lambda: _get_env_or_default('ENERGY_DECAY_RATE', 0.5, float))
    """Energy decay per hop during spreading activation"""

    ANCHOR_ENERGY_DISTRIBUTION: str = field(default_factory=lambda: _get_env_or_default('ANCHOR_ENERGY_DISTRIBUTION', "softmax"))
    """How to distribute initial energy to anchors: 'softmax', 'normalized', 'uniform'"""

    MERGE_STRATEGY: str = field(default_factory=lambda: _get_env_or_default('MERGE_STRATEGY', "weighted_average"))
    """How to merge old and new activations: 'weighted_average', 'replace', 'accumulate', 'max'"""

    MERGE_OLD_WEIGHT: float = field(default_factory=lambda: _get_env_or_default('MERGE_OLD_WEIGHT', 0.7, float))
    """Weight for old activations in weighted_average merge (new weight = 1 - old_weight)"""

    # ==================== Graph Construction ====================
    # SIM Edge Parameters (Layered Strategy)
    MAX_SIM_NEIGHBORS_INTRA_DOC: int = 3
    """Maximum similarity edges within the same document"""

    MAX_SIM_NEIGHBORS_INTER_DOC: int = 2
    """Maximum similarity edges across different documents"""

    # CAUSE Edge Parameters (Hybrid Strategy)
    CAUSE_WINDOW_SIZE: int = 5
    """Sliding window size for local causal edge detection"""

    CAUSE_NEIGHBOR_HOPS: int = 1
    """Number of hops for graph neighbor expansion in causal detection"""

    SKIP_CAUSE_EDGES: bool = True
    """Whether to skip CAUSE edge building (set to False to enable, but it's expensive)"""

    # ==================== Plasticity Parameters ====================
    HEBBIAN_LEARNING_RATE: float = 0.1
    """Weight increment for LTP (Long-Term Potentiation)"""

    TIME_DECAY_FACTOR: float = 0.99
    """Natural forgetting coefficient for LTD (Long-Term Depression)"""

    MIN_EDGE_WEIGHT: float = 0.1
    """Minimum edge weight before pruning"""

    LTP_ACTIVATION_THRESHOLD: float = 0.05
    """Minimum energy flow for an edge to be considered 'activated' and eligible for LTP"""

    TRACK_ACTIVATION_PATH: bool = True
    """Whether to track the actual path during spreading activation for LTP"""

    # ==================== Node Granularity ====================
    USE_ENTITY_NODES: bool = True
    """Whether to create entity nodes (extracted from text)"""

    USE_CHUNK_NODES: bool = True
    """Whether to create chunk nodes (document segments)"""

    USE_TRIPLE_NODES: bool = False
    """Whether to create triple nodes (subject-predicate-object)"""

    # ==================== LLM Configuration ====================
    llm_name: str = "gpt-4o-mini"
    """LLM model name"""

    llm_base_url: Optional[str] = None
    """Custom LLM endpoint URL (None = use default OpenAI)"""

    llm_temperature: float = 0.0
    """Sampling temperature for LLM"""

    llm_max_new_tokens: int = 2048
    """Maximum tokens to generate"""

    llm_response_format: dict = field(default_factory=lambda: {"type": "json_object"})
    """Response format specification"""

    # ==================== Embedding Configuration ====================
    embedding_model_name: str = "text-embedding-3-small"
    """Embedding model name"""

    embedding_base_url: Optional[str] = None
    """Custom embedding endpoint URL"""

    embedding_batch_size: int = 32
    """Batch size for embedding encoding"""

    embedding_dimension: int = 1536
    """Embedding vector dimension"""

    embedding_return_as_normalized: bool = True
    """Whether to normalize embeddings"""

    embedding_max_seq_len: int = 2048
    """Max sequence length for embedding model"""

    azure_embedding_endpoint: Optional[str] = None
    """Azure embedding endpoint URL (if using Azure OpenAI)"""

    # ==================== Storage Configuration ====================
    save_dir: str = "outputs"
    """Directory for saving graph and embeddings"""

    force_index_from_scratch: bool = False
    """If True, ignore existing storage and rebuild from scratch"""

    cache_llm_responses: bool = True
    """Whether to cache LLM responses"""

    # ==================== Text Processing ====================
    chunk_size: int = 512
    """Number of tokens per chunk"""

    chunk_overlap: int = 128
    """Number of overlapping tokens between chunks"""

    # ==================== Debugging & Logging ====================
    log_level: str = "INFO"
    """Logging level: DEBUG, INFO, WARNING, ERROR"""

    verbose: bool = False
    """Whether to print detailed progress information"""

    def __post_init__(self):
        """Validate configuration after initialization"""
        assert self.TOP_K_ANCHORS > 0, "TOP_K_ANCHORS must be positive"
        assert self.TOP_N_RETRIEVAL > 0, "TOP_N_RETRIEVAL must be positive"
        assert self.MAX_HOPS > 0, "MAX_HOPS must be positive"
        assert 0 < self.ENERGY_DECAY_RATE <= 1, "ENERGY_DECAY_RATE must be in (0, 1]"
        assert 0 <= self.HEBBIAN_LEARNING_RATE <= 1, "HEBBIAN_LEARNING_RATE must be in [0, 1]"
        assert 0 < self.TIME_DECAY_FACTOR <= 1, "TIME_DECAY_FACTOR must be in (0, 1]"
        assert self.MIN_EDGE_WEIGHT >= 0, "MIN_EDGE_WEIGHT must be non-negative"
        assert (
            self.MERGE_STRATEGY
            in ["weighted_average", "replace", "accumulate", "max"]
        ), f"Invalid MERGE_STRATEGY: {self.MERGE_STRATEGY}"
        assert (
            self.ANCHOR_ENERGY_DISTRIBUTION in ["softmax", "normalized", "uniform"]
        ), f"Invalid ANCHOR_ENERGY_DISTRIBUTION: {self.ANCHOR_ENERGY_DISTRIBUTION}"

    def to_dict(self):
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
