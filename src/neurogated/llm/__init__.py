"""
LLM factory and utilities.
"""

from .base import BaseLLM
from .openai_llm import OpenAILLM


def get_llm(config) -> BaseLLM:
    """
    Factory function to get LLM implementation based on config.

    Args:
        config: MemoryConfig instance

    Returns:
        BaseLLM implementation
    """
    llm_name = config.llm_name.lower()

    if "gpt" in llm_name or "openai" in llm_name:
        return OpenAILLM(config)
    else:
        # Default to OpenAI
        return OpenAILLM(config)


__all__ = ["BaseLLM", "OpenAILLM", "get_llm"]
