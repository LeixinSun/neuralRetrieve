"""
Demo script for the Neuro-Gated Graph Memory System.

This script demonstrates the basic usage of the system.
"""

import logging
from neurogated import NeuroGraphMemory, MemoryConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Run demo"""
    logger.info("=" * 60)
    logger.info("Neuro-Gated Graph Memory System - Demo")
    logger.info("=" * 60)

    # Create configuration
    config = MemoryConfig(
        save_dir="outputs/demo",
        llm_name="gpt-4o-mini",
        TOP_K_ANCHORS=5,
        TOP_N_RETRIEVAL=3,
        MAX_HOPS=2,
        ENERGY_DECAY_RATE=0.5,
        HEBBIAN_LEARNING_RATE=0.1,
        verbose=True
    )

    # Initialize system
    logger.info("\n1. Initializing system...")
    memory = NeuroGraphMemory(config)

    # Show stats
    logger.info("\n2. System statistics:")
    stats = memory.get_stats()
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    # TODO: Add documents
    logger.info("\n3. Adding documents...")
    logger.info("  [Ingestion pipeline not yet implemented]")

    # TODO: Retrieve
    logger.info("\n4. Retrieving information...")
    logger.info("  [Retrieval requires documents to be added first]")

    # Example query (will fail gracefully)
    try:
        query = "What is the capital of France?"
        logger.info(f"  Query: {query}")
        results = memory.retrieve(query)
        logger.info(f"  Results: {results}")
    except Exception as e:
        logger.warning(f"  Retrieval failed (expected): {e}")

    # Maintenance
    logger.info("\n5. Running maintenance...")
    memory.maintenance()

    # Save
    logger.info("\n6. Saving system...")
    memory.save()

    logger.info("\n" + "=" * 60)
    logger.info("Demo completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
