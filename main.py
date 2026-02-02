"""
Main entry point for running experiments with datasets.
"""

import argparse
import logging
import os
import time
from pathlib import Path

from neurogated import NeuroGraphMemory, MemoryConfig, config_from_yaml
from neurogated.evaluation import DatasetLoader, RetrievalRecall, QAExactMatch, QAF1Score, ExperimentResults

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Neuro-Gated Graph Memory System")

    # Configuration file
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")

    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="sample", help="Dataset name")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Dataset directory")

    # LLM arguments (override config.yaml)
    parser.add_argument("--llm_name", type=str, default=None, help="LLM model name")
    parser.add_argument("--llm_base_url", type=str, default=None, help="LLM base URL")

    # Embedding arguments (override config.yaml)
    parser.add_argument("--embedding_name", type=str, default=None, help="Embedding model name")

    # System arguments (override config.yaml)
    parser.add_argument("--save_dir", type=str, default=None, help="Save directory")
    parser.add_argument("--force_index_from_scratch", action="store_true", help="Force rebuild from scratch")

    # Retrieval arguments (override config.yaml)
    parser.add_argument("--top_k_anchors", type=int, default=None, help="Number of anchor nodes")
    parser.add_argument("--top_n_retrieval", type=int, default=None, help="Number of results to return")
    parser.add_argument("--max_hops", type=int, default=None, help="Maximum hops for spreading activation")

    # Evaluation arguments
    parser.add_argument("--max_queries", type=int, default=None, help="Maximum number of queries to evaluate (None=all)")

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Neuro-Gated Graph Memory System - Experiment Runner")
    logger.info("=" * 80)

    # Load configuration from YAML file
    try:
        config = config_from_yaml(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except FileNotFoundError:
        logger.warning(f"Config file {args.config} not found, using defaults")
        config = MemoryConfig()

    # Override with command-line arguments if provided
    if args.llm_name is not None:
        config.llm_name = args.llm_name
    if args.llm_base_url is not None:
        config.llm_base_url = args.llm_base_url
    if args.embedding_name is not None:
        config.embedding_model_name = args.embedding_name
    if args.save_dir is not None:
        config.save_dir = args.save_dir
    if args.force_index_from_scratch:
        config.force_index_from_scratch = True
    if args.top_k_anchors is not None:
        config.TOP_K_ANCHORS = args.top_k_anchors
    if args.top_n_retrieval is not None:
        config.TOP_N_RETRIEVAL = args.top_n_retrieval
    if args.max_hops is not None:
        config.MAX_HOPS = args.max_hops

    # Update save directory with dataset name
    save_dir = os.path.join(config.save_dir, args.dataset)
    config.save_dir = save_dir
    config.verbose = True

    # Initialize experiment results
    results = ExperimentResults(args.dataset, save_dir)
    results.set_config(config.to_dict())

    # Initialize system
    logger.info("\n1. Initializing system...")
    init_start = time.time()
    memory = NeuroGraphMemory(config)
    init_time = time.time() - init_start
    logger.info(f"   Initialization time: {init_time:.2f}s")

    # Load dataset
    logger.info("\n2. Loading dataset...")
    dataset_dir = args.dataset_dir if args.dataset_dir else "dataset"
    dataset_loader = DatasetLoader(dataset_dir)

    try:
        corpus = dataset_loader.load_corpus(args.dataset)
        queries, gold_answers, gold_docs = dataset_loader.load_queries(args.dataset)
    except FileNotFoundError as e:
        logger.error(f"Dataset not found: {e}")
        logger.info("Please ensure dataset files are in the dataset/ directory")
        logger.info("Expected files:")
        logger.info(f"  - dataset/{args.dataset}_corpus.json")
        logger.info(f"  - dataset/{args.dataset}.json")
        return

    # Index documents
    logger.info("\n3. Indexing documents...")
    index_start = time.time()
    formatted_docs = dataset_loader.format_documents(corpus)

    for i, (text, doc_id) in enumerate(formatted_docs):
        logger.info(f"  Indexing document {i+1}/{len(formatted_docs)}: {doc_id}")
        try:
            stats = memory.add_document(text, doc_id)
            logger.debug(f"  Stats: {stats}")
        except Exception as e:
            logger.error(f"  Failed to index document {doc_id}: {e}")

    index_time = time.time() - index_start
    logger.info(f"   Total indexing time: {index_time:.2f}s")

    # Show graph stats
    logger.info("\n4. Graph statistics:")
    stats = memory.get_stats()
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    results.set_graph_stats(stats)

    # Save after indexing
    memory.save()

    # Retrieve for queries
    logger.info("\n5. Running retrieval...")
    retrieve_start = time.time()
    retrieved_docs = []

    # Limit queries if specified
    max_queries = args.max_queries if args.max_queries else len(queries)
    eval_queries = queries[:max_queries]
    eval_gold_docs = gold_docs[:max_queries] if gold_docs else None
    eval_gold_answers = gold_answers[:max_queries] if gold_answers else None

    for i, query in enumerate(eval_queries):
        logger.info(f"  Query {i+1}/{len(eval_queries)}: {query[:60]}...")
        try:
            query_results = memory.retrieve(query)
            retrieved_docs.append(query_results)
            results.add_query_detail(
                query,
                query_results,
                eval_gold_docs[i] if eval_gold_docs else None
            )
            logger.debug(f"  Retrieved {len(query_results)} chunks")
        except Exception as e:
            logger.error(f"  Retrieval failed: {e}")
            retrieved_docs.append([])

    retrieve_time = time.time() - retrieve_start
    logger.info(f"   Total retrieval time: {retrieve_time:.2f}s")

    # Evaluate retrieval (aligned with HippoRAG format)
    logger.info("\n6. Evaluating retrieval...")
    if eval_gold_docs:
        # Use same k_list as HippoRAG
        k_list = [1, 2, 5, 10, 20, 30, 50, 100]
        # Filter k values that make sense for our retrieval count
        max_retrieved = max(len(docs) for docs in retrieved_docs) if retrieved_docs else 0
        k_list = [k for k in k_list if k <= max(max_retrieved, config.TOP_N_RETRIEVAL * 2)]
        if not k_list:
            k_list = [1, 2, 3]

        recall_at_k = RetrievalRecall.compute_at_k(retrieved_docs, eval_gold_docs, k_list)

        # Format output like HippoRAG
        logger.info(f"   Evaluation results for retrieval: {recall_at_k}")
        results.set_retrieval_results(recall_at_k)

        # Also show simple recall
        simple_recall = RetrievalRecall.compute(retrieved_docs, eval_gold_docs)
        logger.info(f"   Overall Recall: {simple_recall:.4f}")

    # Run maintenance
    logger.info("\n7. Running maintenance...")
    memory.maintenance()

    # Final save
    logger.info("\n8. Saving system...")
    memory.save()

    # Save and print results
    results.save_to_file()
    results.print_summary()

    # Print timing summary
    logger.info("\n" + "-" * 40)
    logger.info("Timing Summary:")
    logger.info("-" * 40)
    logger.info(f"  Initialization: {init_time:.2f}s")
    logger.info(f"  Indexing: {index_time:.2f}s")
    logger.info(f"  Retrieval: {retrieve_time:.2f}s")
    logger.info(f"  Total: {init_time + index_time + retrieve_time:.2f}s")

    logger.info("\n" + "=" * 80)
    logger.info("Experiment completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
