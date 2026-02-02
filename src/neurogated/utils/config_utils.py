"""
Configuration utilities for embedding models.
"""

import os
from dataclasses import dataclass, field
from typing import Literal, Union, Optional


@dataclass
class BaseConfig:
    """
    Base configuration class for embedding models.
    This is a simplified version adapted from HippoRAG.
    """
    # LLM specific attributes
    llm_name: str = field(
        default="gpt-4o-mini",
        metadata={"help": "LLM model name"}
    )
    llm_base_url: Optional[str] = field(
        default=None,
        metadata={"help": "Base URL for the LLM model, if None, uses OpenAI service"}
    )
    embedding_base_url: Optional[str] = field(
        default=None,
        metadata={"help": "Base URL for embedding model, if None, uses OpenAI service"}
    )
    max_new_tokens: int = field(
        default=2048,
        metadata={"help": "Max new tokens to generate"}
    )
    temperature: float = field(
        default=0.0,
        metadata={"help": "Temperature for sampling"}
    )
    response_format: Union[dict, None] = field(
        default_factory=lambda: {"type": "json_object"},
        metadata={"help": "Response format specification"}
    )

    # Embedding specific attributes
    embedding_model_name: str = field(
        default="text-embedding-3-small",
        metadata={"help": "Embedding model name"}
    )
    embedding_batch_size: int = field(
        default=32,
        metadata={"help": "Batch size for embedding encoding"}
    )
    embedding_return_as_normalized: bool = field(
        default=True,
        metadata={"help": "Whether to normalize embeddings"}
    )
    embedding_max_seq_len: int = field(
        default=2048,
        metadata={"help": "Max sequence length for embedding model"}
    )
    embedding_model_dtype: Literal["float16", "float32", "bfloat16", "auto"] = field(
        default="auto",
        metadata={"help": "Data type for local embedding model"}
    )

    # Storage specific attributes
    force_index_from_scratch: bool = field(
        default=False,
        metadata={"help": "If True, rebuild from scratch"}
    )
    save_dir: str = field(
        default="outputs",
        metadata={"help": "Directory to save all data"}
    )
