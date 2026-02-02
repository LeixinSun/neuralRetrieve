"""
Simple test to verify the system works.
"""

import logging
import os
from neurogated import NeuroGraphMemory, MemoryConfig, config_from_yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_basic_functionality():
    """Test basic system functionality"""
    logger.info("Testing Neuro-Gated Graph Memory System")

    # Load config from YAML file (includes API key)
    config_path = "config.yaml"
    if os.path.exists(config_path):
        logger.info(f"Loading configuration from {config_path}")
        config = config_from_yaml(config_path)
        # Override some settings for test
        config.save_dir = "outputs/test"
        config.TOP_K_ANCHORS = 3
        config.TOP_N_RETRIEVAL = 2
        config.MAX_HOPS = 1
        config.force_index_from_scratch = True
    else:
        logger.warning(f"Config file {config_path} not found, using defaults")
        config = MemoryConfig(
            save_dir="outputs/test",
            llm_name="gpt-4o-mini",
            embedding_model_name="text-embedding-3-small",
            TOP_K_ANCHORS=3,
            TOP_N_RETRIEVAL=2,
            MAX_HOPS=1,
            force_index_from_scratch=True
        )

    # Initialize system
    logger.info("1. Initializing system...")
    memory = NeuroGraphMemory(config)

    # Add a simple document
    logger.info("2. Adding document...")
    doc = """
    Paris is the capital of France. It is known for the Eiffel Tower.
    The Eiffel Tower was built in 1889 for the World's Fair.
    """

    try:
        stats = memory.add_document(doc, "doc_1")
        logger.info(f"   Document added: {stats}")
    except Exception as e:
        logger.error(f"   Failed to add document: {e}")
        return False

    # Get stats
    logger.info("3. Graph statistics:")
    stats = memory.get_stats()
    for key, value in stats.items():
        logger.info(f"   {key}: {value}")

    # Try retrieval
    logger.info("4. Testing retrieval...")
    query = "What is the capital of France?"

    try:
        results = memory.retrieve(query)
        logger.info(f"   Query: {query}")
        logger.info(f"   Retrieved {len(results)} results")
        for i, result in enumerate(results):
            logger.info(f"   Result {i+1}: {result[:100]}...")
    except Exception as e:
        logger.error(f"   Retrieval failed: {e}")
        return False

    # Save
    logger.info("5. Saving system...")
    memory.save()

    logger.info("✅ Test passed!")
    return True


if __name__ == "__main__":
    success = test_basic_functionality()
    exit(0 if success else 1)
