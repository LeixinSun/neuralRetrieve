"""
Kernel Generator - LLM-driven edge weight modulation.

This module implements the "Query as Modulation Kernel" concept from DESIGN.md.
"""

import json
import logging
from typing import Dict
import re

from ..llm import BaseLLM
from ..prompts.prompt_manager import PromptTemplateManager
from ..data_structures import EdgeType

logger = logging.getLogger(__name__)


class KernelGenerator:
    """
    Generates modulation kernels for edge weights based on query intent.

    The kernel dynamically adjusts edge type weights (SEQ/SIM/CAUSE) based on
    the user's query, enabling context-sensitive retrieval.
    """

    def __init__(self, llm: BaseLLM):
        """
        Initialize kernel generator.

        Args:
            llm: LLM instance for generating kernels
        """
        self.llm = llm
        self.prompt_manager = PromptTemplateManager()

    def generate(self, query: str, max_retries: int = 2) -> Dict:
        """
        Generate modulation kernel for a query.

        Args:
            query: User query text
            max_retries: Maximum number of retries on parse failure

        Returns:
            Dict with structure:
            {
                "weights": {
                    EdgeType.SEQ: float,
                    EdgeType.SIM: float,
                    EdgeType.CAUSE: float
                },
                "justification": str
            }
        """
        # Render prompt
        messages = self.prompt_manager.render("kernel_generation", query=query)

        last_error = None
        for attempt in range(max_retries + 1):
            # Call LLM
            response, metadata = self.llm.infer(messages)

            # Parse response
            try:
                kernel = self._parse_kernel_response(response)
                logger.info(f"Generated kernel for query: {query[:50]}...")
                logger.debug(f"Kernel weights: {kernel['weights']}")
                logger.debug(f"Justification: {kernel['justification']}")
                return kernel

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(f"Failed to parse kernel response (attempt {attempt + 1}/{max_retries + 1}): {e}, retrying...")
                    # Add a hint to the messages for retry
                    if attempt == 0:
                        messages.append({
                            "role": "assistant",
                            "content": response
                        })
                        messages.append({
                            "role": "user",
                            "content": "Please return ONLY a valid JSON object with the exact format specified. No additional text."
                        })
                else:
                    logger.error(f"Failed to parse kernel response after {max_retries + 1} attempts: {e}")
                    logger.debug(f"Last response: {response}")

        # Fallback to uniform weights
        return self._get_default_kernel()

    def _parse_kernel_response(self, response: str) -> Dict:
        """
        Parse LLM response to extract kernel weights.

        Args:
            response: LLM response text

        Returns:
            Parsed kernel dict
        """
        # Try to extract JSON - handle nested braces
        # First try to find JSON block with code fence
        code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if code_block_match:
            try:
                kernel_data = json.loads(code_block_match.group(1))
                if "weights" in kernel_data:
                    return self._process_kernel_data(kernel_data)
            except json.JSONDecodeError:
                pass

        # Try to find raw JSON object
        # Find the outermost braces containing "weights"
        start_idx = response.find('{')
        if start_idx != -1:
            brace_count = 0
            end_idx = start_idx
            for i, char in enumerate(response[start_idx:], start_idx):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break

            json_str = response[start_idx:end_idx]
            try:
                kernel_data = json.loads(json_str)
                if "weights" in kernel_data:
                    return self._process_kernel_data(kernel_data)
            except json.JSONDecodeError:
                pass

        raise ValueError("Could not find JSON in response")

    def _process_kernel_data(self, kernel_data: Dict) -> Dict:
        """Process parsed kernel data and convert to proper format."""
        # Convert string keys to EdgeType
        weights = {}
        for key, value in kernel_data["weights"].items():
            if key == "SEQ":
                weights[EdgeType.SEQ] = float(value)
            elif key == "SIM":
                weights[EdgeType.SIM] = float(value)
            elif key == "CAUSE":
                weights[EdgeType.CAUSE] = float(value)

        # Validate weights
        for edge_type, weight in weights.items():
            if not (0.0 <= weight <= 2.0):
                logger.warning(f"Weight {weight} for {edge_type} out of range [0, 2], clipping")
                weights[edge_type] = max(0.0, min(2.0, weight))

        return {
            "weights": weights,
            "justification": kernel_data.get("justification", "")
        }

    def _get_default_kernel(self) -> Dict:
        """
        Get default kernel with uniform weights.

        Returns:
            Default kernel dict
        """
        return {
            "weights": {
                EdgeType.SEQ: 1.0,
                EdgeType.SIM: 1.0,
                EdgeType.CAUSE: 1.0
            },
            "justification": "Default uniform weights (kernel generation failed)"
        }
