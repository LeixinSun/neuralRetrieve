"""
Evaluation metrics for retrieval and QA.
"""

import logging
from typing import List
import re

logger = logging.getLogger(__name__)


class RetrievalRecall:
    """
    Compute recall for retrieval results.
    """

    @staticmethod
    def compute(retrieved_docs: List[List[str]], gold_docs: List[List[str]]) -> float:
        """
        Compute average recall.

        Args:
            retrieved_docs: List of retrieved document lists
            gold_docs: List of gold document lists

        Returns:
            Average recall score
        """
        if not retrieved_docs or not gold_docs:
            return 0.0

        recalls = []

        for retrieved, gold in zip(retrieved_docs, gold_docs):
            if not gold:
                continue

            # Normalize texts for comparison
            retrieved_set = set([RetrievalRecall._normalize(doc) for doc in retrieved])
            gold_set = set([RetrievalRecall._normalize(doc) for doc in gold])

            # Compute recall
            overlap = len(retrieved_set & gold_set)
            recall = overlap / len(gold_set) if gold_set else 0.0

            recalls.append(recall)

        return sum(recalls) / len(recalls) if recalls else 0.0

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for comparison"""
        return text.lower().strip()


class QAExactMatch:
    """
    Compute exact match for QA results.
    """

    @staticmethod
    def compute(predictions: List[str], gold_answers: List[List[str]]) -> float:
        """
        Compute average exact match score.

        Args:
            predictions: List of predicted answers
            gold_answers: List of gold answer lists

        Returns:
            Average exact match score
        """
        if not predictions or not gold_answers:
            return 0.0

        em_scores = []

        for pred, golds in zip(predictions, gold_answers):
            # Normalize
            pred_norm = QAExactMatch._normalize(pred)
            golds_norm = [QAExactMatch._normalize(g) for g in golds]

            # Check if prediction matches any gold answer
            em = 1.0 if pred_norm in golds_norm else 0.0
            em_scores.append(em)

        return sum(em_scores) / len(em_scores)

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for comparison"""
        # Remove articles
        text = re.sub(r'\b(a|an|the)\b', ' ', text.lower())
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text


class QAF1Score:
    """
    Compute F1 score for QA results.
    """

    @staticmethod
    def compute(predictions: List[str], gold_answers: List[List[str]]) -> float:
        """
        Compute average F1 score.

        Args:
            predictions: List of predicted answers
            gold_answers: List of gold answer lists

        Returns:
            Average F1 score
        """
        if not predictions or not gold_answers:
            return 0.0

        f1_scores = []

        for pred, golds in zip(predictions, gold_answers):
            # Compute F1 against each gold answer and take max
            max_f1 = 0.0

            for gold in golds:
                f1 = QAF1Score._compute_f1(pred, gold)
                max_f1 = max(max_f1, f1)

            f1_scores.append(max_f1)

        return sum(f1_scores) / len(f1_scores)

    @staticmethod
    def _compute_f1(pred: str, gold: str) -> float:
        """Compute F1 between two strings"""
        pred_tokens = QAF1Score._normalize(pred).split()
        gold_tokens = QAF1Score._normalize(gold).split()

        if not pred_tokens or not gold_tokens:
            return 0.0

        common = set(pred_tokens) & set(gold_tokens)

        if not common:
            return 0.0

        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gold_tokens)

        f1 = 2 * (precision * recall) / (precision + recall)

        return f1

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text"""
        text = re.sub(r'\b(a|an|the)\b', ' ', text.lower())
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join(text.split())
        return text


__all__ = ["RetrievalRecall", "QAExactMatch", "QAF1Score"]
