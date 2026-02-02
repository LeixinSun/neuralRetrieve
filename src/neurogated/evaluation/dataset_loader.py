"""
Dataset loader for HippoRAG-format datasets.
"""

import json
import logging
from typing import List, Dict, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Loads datasets in HippoRAG format.
    """

    def __init__(self, dataset_dir: str = "dataset"):
        """
        Initialize dataset loader.

        Args:
            dataset_dir: Directory containing datasets
        """
        self.dataset_dir = Path(dataset_dir)

    def load_corpus(self, dataset_name: str) -> List[Dict]:
        """
        Load corpus from dataset.

        Args:
            dataset_name: Name of dataset (e.g., "sample", "musique")

        Returns:
            List of document dicts with 'title', 'text', 'idx'
        """
        corpus_path = self.dataset_dir / f"{dataset_name}_corpus.json"

        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

        with open(corpus_path, "r") as f:
            corpus = json.load(f)

        logger.info(f"Loaded {len(corpus)} documents from {corpus_path}")

        return corpus

    def load_queries(self, dataset_name: str) -> Tuple[List[str], List[List[str]], List[List[str]]]:
        """
        Load queries, gold answers, and gold documents.

        Args:
            dataset_name: Name of dataset

        Returns:
            Tuple of (queries, gold_answers, gold_docs)
        """
        query_path = self.dataset_dir / f"{dataset_name}.json"

        if not query_path.exists():
            raise FileNotFoundError(f"Query file not found: {query_path}")

        with open(query_path, "r") as f:
            samples = json.load(f)

        # Extract queries
        queries = [s["question"] for s in samples]

        # Extract gold answers
        gold_answers = []
        for sample in samples:
            if "answer" in sample:
                answer = sample["answer"]
            elif "gold_ans" in sample:
                answer = sample["gold_ans"]
            else:
                answer = []

            if isinstance(answer, str):
                answer = [answer]

            gold_answers.append(answer)

        # Extract gold documents
        gold_docs = []
        for sample in samples:
            if "paragraphs" in sample:
                docs = [
                    f"{p['title']}\n\n{p.get('text', p.get('paragraph_text', ''))}"
                    for p in sample["paragraphs"]
                    if p.get("is_supporting", True)
                ]
            else:
                docs = []

            gold_docs.append(docs)

        logger.info(f"Loaded {len(queries)} queries from {query_path}")

        return queries, gold_answers, gold_docs

    def format_documents(self, corpus: List[Dict]) -> List[Tuple[str, str]]:
        """
        Format corpus documents for ingestion.

        Args:
            corpus: List of document dicts

        Returns:
            List of (text, document_id) tuples
        """
        formatted = []

        for doc in corpus:
            text = f"{doc['title']}\n\n{doc['text']}"
            doc_id = f"doc_{doc['idx']}"
            formatted.append((text, doc_id))

        return formatted


__all__ = ["DatasetLoader"]
