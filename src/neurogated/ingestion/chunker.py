"""
Text chunking utilities.
"""

import logging
from typing import List, Tuple
import tiktoken

logger = logging.getLogger(__name__)


class TextChunker:
    """
    Chunks text into smaller segments with overlap.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128, encoding_name: str = "cl100k_base"):
        """
        Initialize text chunker.

        Args:
            chunk_size: Number of tokens per chunk
            chunk_overlap: Number of overlapping tokens between chunks
            encoding_name: Tokenizer encoding name
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)

    def chunk(self, text: str, document_id: str = None) -> List[Tuple[str, dict]]:
        """
        Chunk text into segments.

        Args:
            text: Text to chunk
            document_id: Optional document identifier

        Returns:
            List of (chunk_text, metadata) tuples
        """
        # Tokenize
        tokens = self.encoding.encode(text)

        if len(tokens) <= self.chunk_size:
            # Text is small enough, return as single chunk
            return [(text, {"document_id": document_id, "chunk_index": 0, "total_chunks": 1})]

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(tokens):
            # Get chunk tokens
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]

            # Decode back to text
            chunk_text = self.encoding.decode(chunk_tokens)

            # Create metadata
            metadata = {
                "document_id": document_id,
                "chunk_index": chunk_index,
                "start_token": start,
                "end_token": end,
                "total_tokens": len(chunk_tokens)
            }

            chunks.append((chunk_text, metadata))

            # Move to next chunk with overlap
            start += self.chunk_size - self.chunk_overlap
            chunk_index += 1

        # Add total_chunks to all metadata
        for _, meta in chunks:
            meta["total_chunks"] = len(chunks)

        logger.debug(f"Chunked text into {len(chunks)} chunks")

        return chunks
