"""
Prompt template manager for the Neuro-Gated Graph Memory System.
"""

from typing import List, Dict


class PromptTemplateManager:
    """
    Manages prompt templates for different LLM tasks.
    """

    def __init__(self):
        """Initialize prompt templates"""
        self.templates = {
            "ner": self._ner_template,
            "kernel_generation": self._kernel_generation_template,
            "causal_detection": self._causal_detection_template,
        }

    def render(self, name: str, **kwargs) -> List[Dict[str, str]]:
        """
        Render a prompt template.

        Args:
            name: Template name
            **kwargs: Template variables

        Returns:
            List of message dicts
        """
        if name not in self.templates:
            raise ValueError(f"Unknown template: {name}")

        return self.templates[name](**kwargs)

    def _ner_template(self, passage: str) -> List[Dict[str, str]]:
        """Named Entity Recognition template"""
        return [
            {
                "role": "system",
                "content": "You are an expert at extracting named entities from text. Extract all named entities and their types (PERSON, ORG, LOC, etc.)."
            },
            {
                "role": "user",
                "content": f"""Extract all named entities from the following passage. Return a JSON object with a "named_entities" array, where each entity is formatted as "EntityName (TYPE)".

Passage:
{passage}

Return JSON:
{{
    "named_entities": ["Entity1 (TYPE1)", "Entity2 (TYPE2)", ...]
}}"""
            }
        ]

    def _kernel_generation_template(self, query: str) -> List[Dict[str, str]]:
        """Kernel generation template for query modulation"""
        return [
            {
                "role": "system",
                "content": "You are an expert at analyzing query intent and assigning edge type weights for a neural memory system."
            },
            {
                "role": "user",
                "content": f"""Instruction
Analyze the following query and assign a weight between 0.0 and 2.0 to each of the three relationship types. These weights should reflect the relative emphasis that should be placed on each type when retrieving relevant materials to answer the query.

SEQ (Sequential/Temporal): For queries about order, sequence, timelines, or processes. Examples: "What happened next?", "List the steps to...", "Historical development of...".
SIM (Similarity/Semantic): For queries about related concepts, analogies, or functional similarities. Examples: "What is similar to...?", "Other examples like...", "What falls under the same category?".
CAUSE (Causal/Logical): For queries about reasons, effects, mechanisms, or explanations. Examples: "Why did...?", "How does X lead to Y?", "What is the impact of...?".
Weight Guidelines
0.0: This relationship type is largely irrelevant to the query's intent. Materials with this focus should not be prioritized.
1.0: This relationship type is moderately relevant. Materials with this focus should be considered.
2.0: This relationship type is central to the query's intent. Materials with this focus should be highly prioritized.
Query: "{query}"

Return JSON with weights and justification:
{{
    "weights": {{
        "SEQ": <float 0.0-2.0>,
        "SIM": <float 0.0-2.0>,
        "CAUSE": <float 0.0-2.0>
    }},
    "justification": "<brief explanation of why these weights were chosen>"
}}"""
            }
        ]

    def _causal_detection_template(self, text_a: str, text_b: str) -> List[Dict[str, str]]:
        """Causal relationship detection template"""
        return [
            {
                "role": "system",
                "content": "You are an expert at detecting causal relationships between text segments."
            },
            {
                "role": "user",
                "content": f"""Determine if there is a causal relationship between the following two text segments. Consider both direct causation (A causes B) and indirect causation (A influences B).

Text A:
{text_a}

Text B:
{text_b}

Return JSON:
{{
    "has_causality": <true or false>,
    "confidence": <float 0.0-1.0>,
    "direction": "<A_to_B, B_to_A, or bidirectional>",
    "explanation": "<brief explanation>"
}}"""
            }
        ]
