"""
Simple test to verify the system can be imported and initialized (without API calls).
"""

import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all modules can be imported"""
    logger.info("=" * 60)
    logger.info("Testing Module Imports")
    logger.info("=" * 60)

    try:
        logger.info("1. Testing core imports...")
        from neurogated import NeuroGraphMemory, MemoryConfig, config_from_yaml
        logger.info("   ✓ NeuroGraphMemory")
        logger.info("   ✓ MemoryConfig")
        logger.info("   ✓ config_from_yaml")

        logger.info("\n2. Testing config loading...")
        config = config_from_yaml("config.yaml")
        logger.info(f"   ✓ Loaded config.yaml")
        logger.info(f"   - LLM: {config.llm_name}")
        logger.info(f"   - Embedding: {config.embedding_model_name}")
        logger.info(f"   - Top K: {config.TOP_K_ANCHORS}")

        logger.info("\n3. Testing config creation...")
        test_config = MemoryConfig(
            save_dir="outputs/test",
            llm_name="gpt-4o-mini",
            TOP_K_ANCHORS=3
        )
        logger.info(f"   ✓ Created MemoryConfig")
        logger.info(f"   - Save dir: {test_config.save_dir}")
        logger.info(f"   - LLM: {test_config.llm_name}")

        logger.info("\n4. Testing evaluation imports...")
        from neurogated.evaluation import DatasetLoader, RetrievalRecall
        logger.info("   ✓ DatasetLoader")
        logger.info("   ✓ RetrievalRecall")

        logger.info("\n5. Testing storage imports...")
        from neurogated.storage import GraphStore, EmbeddingStore
        logger.info("   ✓ GraphStore")
        logger.info("   ✓ EmbeddingStore")

        logger.info("\n" + "=" * 60)
        logger.info("✅ All imports successful!")
        logger.info("=" * 60)
        logger.info("\nNote: To run full tests with API calls, set OPENAI_API_KEY")
        logger.info("      export OPENAI_API_KEY='your-key-here'")
        logger.info("      uv run python test_basic.py")

        return True

    except Exception as e:
        logger.error(f"\n❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
