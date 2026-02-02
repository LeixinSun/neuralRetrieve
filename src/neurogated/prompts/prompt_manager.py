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
                "content": f"""Analyze the following query and assign weights (0.0-2.0) to each edge type based on the query's intent:

- **SEQ (sequential/temporal)**: For queries about order, timeline, process, "what happened next"
- **SIM (similarity/semantic)**: For queries about related concepts, analogies, "what is similar to"
- **CAUSE (causal/logical)**: For queries about reasons, consequences, explanations, "why", "how does X affect Y"

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
