"""
OpenAI LLM implementation with response caching.
"""

import hashlib
import json
import logging
import os
import sqlite3
from typing import List, Dict, Tuple, Any
from openai import OpenAI

from .base import BaseLLM

logger = logging.getLogger(__name__)


class OpenAILLM(BaseLLM):
    """
    OpenAI LLM implementation with SQLite-based response caching.
    """

    def __init__(self, config):
        super().__init__(config)

        # Initialize OpenAI client
        client_kwargs = {
            "api_key": os.getenv("OPENAI_API_KEY")
        }

        # Add base_url if specified
        if config.llm_base_url:
            client_kwargs["base_url"] = config.llm_base_url
            logger.info(f"Using custom OpenAI base URL: {config.llm_base_url}")

        self.client = OpenAI(**client_kwargs)

        # Setup cache
        if config.cache_llm_responses:
            self.cache_enabled = True
            self.cache_db_path = os.path.join(config.save_dir, "llm_cache.db")
            self._init_cache()
        else:
            self.cache_enabled = False

    def _init_cache(self):
        """Initialize SQLite cache database"""
        os.makedirs(os.path.dirname(self.cache_db_path), exist_ok=True)

        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                cache_key TEXT PRIMARY KEY,
                response TEXT NOT NULL,
                metadata TEXT NOT NULL
            )
        """)

        conn.commit()
        conn.close()

    def _compute_cache_key(self, messages: List[Dict], **kwargs) -> str:
        """Compute cache key from messages and parameters"""
        cache_data = {
            "messages": messages,
            "model": self.llm_name,
            "temperature": self.temperature,
            **kwargs
        }
        return hashlib.sha256(json.dumps(cache_data, sort_keys=True).encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Tuple[str, Dict, bool]:
        """Get response from cache"""
        if not self.cache_enabled:
            return None, {}, False

        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT response, metadata FROM cache WHERE cache_key = ?", (cache_key,))
            result = cursor.fetchone()

            conn.close()

            if result:
                response, metadata_str = result
                metadata = json.loads(metadata_str)
                metadata["cache_hit"] = True
                return response, metadata, True

        except Exception as e:
            logger.warning(f"Cache read error: {e}")

        return None, {}, False

    def _save_to_cache(self, cache_key: str, response: str, metadata: Dict):
        """Save response to cache"""
        if not self.cache_enabled:
            return

        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()

            cursor.execute(
                "INSERT OR REPLACE INTO cache (cache_key, response, metadata) VALUES (?, ?, ?)",
                (cache_key, response, json.dumps(metadata))
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    def infer(self, messages: List[Dict[str, str]], **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        Synchronous inference with caching.

        Args:
            messages: List of message dicts
            **kwargs: Additional parameters

        Returns:
            Tuple of (response_text, metadata)
        """
        # Check cache
        cache_key = self._compute_cache_key(messages, **kwargs)
        cached_response, cached_metadata, cache_hit = self._get_from_cache(cache_key)

        if cache_hit:
            logger.debug(f"Cache hit for key {cache_key[:8]}...")
            return cached_response, cached_metadata

        # Call OpenAI API
        try:
            response = self.client.chat.completions.create(
                model=self.llm_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
                response_format=self.config.llm_response_format,
                **kwargs
            )

            response_text = response.choices[0].message.content
            metadata = {
                "finish_reason": response.choices[0].finish_reason,
                "usage": response.usage.model_dump() if response.usage else {},
                "cache_hit": False
            }

            # Save to cache
            self._save_to_cache(cache_key, response_text, metadata)

            return response_text, metadata

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return "", {"error": str(e), "cache_hit": False}

    async def ainfer(self, messages: List[Dict[str, str]], **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        Asynchronous inference (not implemented yet, falls back to sync).

        Args:
            messages: List of message dicts
            **kwargs: Additional parameters

        Returns:
            Tuple of (response_text, metadata)
        """
        # TODO: Implement async version
        return self.infer(messages, **kwargs)
