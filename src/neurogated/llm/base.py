"""
Base LLM interface for the Neuro-Gated Graph Memory System.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """
    Abstract base class for LLM implementations.

    All LLM providers should implement this interface.
    """

    def __init__(self, config):
        """
        Initialize LLM with configuration.

        Args:
            config: MemoryConfig instance
        """
        self.config = config
        self.llm_name = config.llm_name
        self.temperature = config.llm_temperature
        self.max_new_tokens = config.llm_max_new_tokens

    @abstractmethod
    def infer(self, messages: List[Dict[str, str]], **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        Synchronous inference.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters

        Returns:
            Tuple of (response_text, metadata)
        """
        pass

    @abstractmethod
    async def ainfer(self, messages: List[Dict[str, str]], **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        Asynchronous inference.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters

        Returns:
            Tuple of (response_text, metadata)
        """
        pass

    def batch_infer(self, messages_list: List[List[Dict[str, str]]], **kwargs) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Batch inference (default implementation using sequential calls).

        Args:
            messages_list: List of message lists
            **kwargs: Additional parameters

        Returns:
            List of (response_text, metadata) tuples
        """
        results = []
        for messages in messages_list:
            result = self.infer(messages, **kwargs)
            results.append(result)
        return results
