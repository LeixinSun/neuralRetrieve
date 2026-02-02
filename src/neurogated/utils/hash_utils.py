"""
Utility functions for the Neuro-Gated Graph Memory System.
"""

import hashlib
import numpy as np
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity


def compute_hash_id(content: str, prefix: str = "") -> str:
    """
    Compute MD5 hash ID for content-based deduplication.

    Args:
        content: Text content to hash
        prefix: Prefix for the hash (e.g., "chunk-", "entity-")

    Returns:
        Hash ID string
    """
    return prefix + hashlib.md5(content.encode()).hexdigest()


def cosine_sim(vec1: List[float], vec2: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score (0-1)
    """
    return float(cosine_similarity([vec1], [vec2])[0][0])


def retrieve_knn(
    query_embedding: List[float],
    embeddings: List[List[float]],
    top_k: int
) -> Tuple[List[int], List[float]]:
    """
    Retrieve top-k nearest neighbors using cosine similarity.

    Args:
        query_embedding: Query vector
        embeddings: List of candidate vectors
        top_k: Number of neighbors to retrieve

    Returns:
        Tuple of (indices, similarities)
    """
    if not embeddings:
        return [], []

    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    top_scores = similarities[top_indices]

    return top_indices.tolist(), top_scores.tolist()


def softmax(scores: List[float], temperature: float = 1.0) -> List[float]:
    """
    Apply softmax to convert scores to probability distribution.

    Args:
        scores: List of scores
        temperature: Temperature parameter (higher = more uniform)

    Returns:
        Softmax probabilities
    """
    scores = np.array(scores) / temperature
    exp_scores = np.exp(scores - np.max(scores))  # Numerical stability
    return (exp_scores / exp_scores.sum()).tolist()


def normalize_scores(scores: List[float]) -> List[float]:
    """
    Normalize scores to sum to 1.0.

    Args:
        scores: List of scores

    Returns:
        Normalized scores
    """
    total = sum(scores)
    if total == 0:
        return [1.0 / len(scores)] * len(scores)
    return [s / total for s in scores]
