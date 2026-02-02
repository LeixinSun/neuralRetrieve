"""
Miscellaneous utility functions and data classes.
"""

from dataclasses import dataclass
from hashlib import md5
from typing import Dict, Any, List, Optional


@dataclass
class NerRawOutput:
    """Raw output from Named Entity Recognition"""
    chunk_id: str
    response: str
    unique_entities: List[str]
    metadata: Dict[str, Any]


@dataclass
class TripleRawOutput:
    """Raw output from Triple extraction"""
    chunk_id: str
    response: str
    triples: List[List[str]]
    metadata: Dict[str, Any]


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """
    Compute the MD5 hash of the given content string and optionally prepend a prefix.

    Args:
        content: The input string to be hashed
        prefix: A string to prepend to the resulting hash (default: "")

    Returns:
        A string consisting of the prefix followed by the hexadecimal representation of the MD5 hash

    Example:
        >>> compute_mdhash_id("hello world", "chunk-")
        'chunk-5eb63bbbe01eeed093cb22bb8f5acdc3'
    """
    return prefix + md5(content.encode()).hexdigest()
