"""
Entity extraction using LLM-based NER.
"""

import json
import logging
import re
from typing import List, Tuple

from ..llm import BaseLLM
from ..prompts.prompt_manager import PromptTemplateManager

logger = logging.getLogger(__name__)


class EntityExtractor:
    """
    Extracts named entities from text using LLM-based NER.
    """

    def __init__(self, llm: BaseLLM):
        """
        Initialize entity extractor.

        Args:
            llm: LLM instance for NER
        """
        self.llm = llm
        self.prompt_manager = PromptTemplateManager()

    def extract(self, text: str) -> List[str]:
        """
        Extract named entities from text.

        Args:
            text: Text to extract entities from

        Returns:
            List of entities in format "EntityName (TYPE)"
        """
        # Render NER prompt
        messages = self.prompt_manager.render("ner", passage=text)

        # Call LLM
        response, metadata = self.llm.infer(messages)

        # Parse response
        try:
            entities = self._parse_ner_response(response)
            logger.debug(f"Extracted {len(entities)} entities from text")
            return entities

        except Exception as e:
            logger.error(f"Failed to parse NER response: {e}")
            logger.debug(f"Response: {response}")
            return []

    def _parse_ner_response(self, response: str) -> List[str]:
        """
        Parse LLM response to extract entities.

        Args:
            response: LLM response text

        Returns:
            List of entity strings
        """
        # Try to extract JSON
        json_match = re.search(r'\{[^{}]*"named_entities"[^{}]*\}', response, re.DOTALL)

        if json_match:
            data = json.loads(json_match.group())
            entities = data.get("named_entities", [])

            # Deduplicate while preserving order
            seen = set()
            unique_entities = []
            for entity in entities:
                if entity not in seen:
                    seen.add(entity)
                    unique_entities.append(entity)

            return unique_entities

        else:
            raise ValueError("Could not find JSON in response")

    def batch_extract(self, texts: List[str]) -> List[List[str]]:
        """
        Extract entities from multiple texts.

        Args:
            texts: List of texts

        Returns:
            List of entity lists
        """
        results = []
        for text in texts:
            entities = self.extract(text)
            results.append(entities)
        return results
