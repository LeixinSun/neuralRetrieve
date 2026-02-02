"""
Evaluation metrics for retrieval and QA.
"""

import logging
from typing import List, Dict, Tuple
import re
import json
import os
from datetime import datetime

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
            retrieved_docs: List of retrieved document lists (can be chunks)
            gold_docs: List of gold document lists (full documents)

        Returns:
            Average recall score
        """
        if not retrieved_docs or not gold_docs:
            return 0.0

        recalls = []

        for retrieved, gold in zip(retrieved_docs, gold_docs):
            if not gold:
                continue

            # Check if any retrieved chunk is contained in any gold doc
            hits = 0
            for gold_doc in gold:
                gold_norm = RetrievalRecall._normalize(gold_doc)
                for ret_doc in retrieved:
                    ret_norm = RetrievalRecall._normalize(ret_doc)
                    # Check if retrieved chunk is part of gold doc (substring match)
                    if ret_norm in gold_norm or gold_norm in ret_norm:
                        hits += 1
                        break

            recall = hits / len(gold) if gold else 0.0
            recalls.append(recall)

        return sum(recalls) / len(recalls) if recalls else 0.0

    @staticmethod
    def compute_at_k(
        retrieved_docs: List[List[str]],
        gold_docs: List[List[str]],
        k_list: List[int] = [1, 2, 5, 10]
    ) -> Dict[str, float]:
        """
        Compute Recall@k for multiple k values.

        Args:
            retrieved_docs: List of retrieved document lists (can be chunks)
            gold_docs: List of gold document lists (full documents)
            k_list: List of k values to compute recall at

        Returns:
            Dictionary with Recall@k scores
        """
        if not retrieved_docs or not gold_docs:
            return {f"Recall@{k}": 0.0 for k in k_list}

        results = {f"Recall@{k}": [] for k in k_list}

        for retrieved, gold in zip(retrieved_docs, gold_docs):
            if not gold:
                continue

            for k in k_list:
                retrieved_at_k = retrieved[:k] if len(retrieved) >= k else retrieved

                # Check if any retrieved chunk matches any gold doc
                hits = 0
                for gold_doc in gold:
                    gold_norm = RetrievalRecall._normalize(gold_doc)
                    for ret_doc in retrieved_at_k:
                        ret_norm = RetrievalRecall._normalize(ret_doc)
                        # Substring match: chunk in doc or doc in chunk
                        if ret_norm in gold_norm or gold_norm in ret_norm:
                            hits += 1
                            break

                recall = hits / len(gold) if gold else 0.0
                results[f"Recall@{k}"].append(recall)

        # Average across all queries
        return {key: sum(vals) / len(vals) if vals else 0.0 for key, vals in results.items()}

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for comparison - collapse whitespace"""
        # Collapse all whitespace (newlines, tabs, multiple spaces) to single space
        return ' '.join(text.lower().split())


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


class ExperimentResults:
    """
    Store and display experiment results.
    """

    def __init__(self, dataset_name: str, save_dir: str):
        self.dataset_name = dataset_name
        self.save_dir = save_dir
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.graph_stats = {}
        self.retrieval_results = {}
        self.qa_results = {}
        self.query_details = []
        self.config = {}

    def set_graph_stats(self, stats: Dict):
        """Set graph statistics"""
        self.graph_stats = stats

    def set_config(self, config: Dict):
        """Set configuration used"""
        self.config = config

    def set_retrieval_results(self, results: Dict[str, float]):
        """Set retrieval evaluation results"""
        self.retrieval_results = results

    def set_qa_results(self, em: float, f1: float):
        """Set QA evaluation results"""
        self.qa_results = {"exact_match": em, "f1_score": f1}

    def add_query_detail(self, query: str, retrieved: List[str], gold: List[str] = None):
        """Add detail for a single query"""
        self.query_details.append({
            "query": query,
            "retrieved": retrieved[:3] if retrieved else [],  # Top 3 for display
            "gold": gold[:3] if gold else [],
            "num_retrieved": len(retrieved) if retrieved else 0
        })

    def print_summary(self):
        """Print a formatted summary of results"""
        print("\n" + "=" * 80)
        print("EXPERIMENT RESULTS SUMMARY")
        print("=" * 80)

        print(f"\nDataset: {self.dataset_name}")
        print(f"Timestamp: {self.timestamp}")

        # Graph Statistics
        print("\n" + "-" * 40)
        print("Graph Statistics:")
        print("-" * 40)
        for key, value in self.graph_stats.items():
            print(f"  {key}: {value}")

        # Retrieval Results
        if self.retrieval_results:
            print("\n" + "-" * 40)
            print("Retrieval Performance:")
            print("-" * 40)
            for key, value in sorted(self.retrieval_results.items()):
                print(f"  {key}: {value:.4f}")

        # QA Results
        if self.qa_results:
            print("\n" + "-" * 40)
            print("QA Performance:")
            print("-" * 40)
            print(f"  Exact Match: {self.qa_results['exact_match']:.4f}")
            print(f"  F1 Score: {self.qa_results['f1_score']:.4f}")

        # Sample Query Details
        if self.query_details:
            print("\n" + "-" * 40)
            print("Sample Query Results (first 3):")
            print("-" * 40)
            for i, detail in enumerate(self.query_details[:3]):
                print(f"\n  Query {i+1}: {detail['query'][:60]}...")
                print(f"  Retrieved {detail['num_retrieved']} chunks")
                if detail['retrieved']:
                    print(f"  Top result: {detail['retrieved'][0][:80]}...")

        print("\n" + "=" * 80)

    def save_to_file(self):
        """Save results to JSON file"""
        os.makedirs(self.save_dir, exist_ok=True)

        results = {
            "dataset": self.dataset_name,
            "timestamp": self.timestamp,
            "graph_stats": self.graph_stats,
            "retrieval_results": self.retrieval_results,
            "qa_results": self.qa_results,
            "num_queries": len(self.query_details),
            "config": self.config
        }

        # Save summary
        summary_path = os.path.join(self.save_dir, "experiment_results.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Save detailed results
        details_path = os.path.join(self.save_dir, "query_details.json")
        with open(details_path, "w", encoding="utf-8") as f:
            json.dump(self.query_details, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {summary_path}")

        return summary_path


__all__ = ["RetrievalRecall", "QAExactMatch", "QAF1Score", "ExperimentResults"]
