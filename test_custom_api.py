"""
Test script to verify custom OpenAI API base URL configuration.
"""

import os
import logging
from neurogated import MemoryConfig
from neurogated.llm import get_llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_custom_api():
    """Test custom API configuration"""
    logger.info("=" * 60)
    logger.info("Testing Custom OpenAI API Configuration")
    logger.info("=" * 60)

    # Get configuration from environment or use defaults
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        logger.error("❌ OPENAI_API_KEY not set!")
        logger.info("Please set: export OPENAI_API_KEY='your-key'")
        return False

    logger.info(f"\n📋 Configuration:")
    logger.info(f"  Base URL: {base_url}")
    logger.info(f"  API Key: {api_key[:10]}...{api_key[-4:]}")
    logger.info(f"  Model: gpt-4o-mini")

    # Create config
    config = MemoryConfig(
        llm_name="gpt-4o-mini",
        llm_base_url=base_url if base_url != "https://api.openai.com/v1" else None,
        save_dir="outputs/test_api",
        cache_llm_responses=False  # Disable cache for testing
    )

    # Initialize LLM
    logger.info("\n🔧 Initializing LLM...")
    try:
        llm = get_llm(config)
        logger.info("✅ LLM initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize LLM: {e}")
        return False

    # Test simple call
    logger.info("\n🧪 Testing simple API call...")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say 'Hello, World!' and nothing else."}
    ]

    try:
        response, metadata = llm.infer(messages)
        logger.info(f"✅ API call successful!")
        logger.info(f"  Response: {response}")
        logger.info(f"  Finish reason: {metadata.get('finish_reason')}")
        logger.info(f"  Cache hit: {metadata.get('cache_hit', False)}")
    except Exception as e:
        logger.error(f"❌ API call failed: {e}")
        return False

    # Test JSON response
    logger.info("\n🧪 Testing JSON response format...")
    messages = [
        {"role": "system", "content": "You are a helpful assistant that returns JSON."},
        {"role": "user", "content": 'Return a JSON object with a single field "status" set to "ok".'}
    ]

    try:
        response, metadata = llm.infer(messages)
        logger.info(f"✅ JSON response successful!")
        logger.info(f"  Response: {response[:100]}...")
    except Exception as e:
        logger.error(f"❌ JSON response failed: {e}")
        return False

    logger.info("\n" + "=" * 60)
    logger.info("✅ All tests passed!")
    logger.info("=" * 60)
    logger.info("\n💡 Your custom API configuration is working correctly.")
    logger.info("You can now use it with the main system:")
    logger.info(f"  uv run python main.py --llm_base_url '{base_url}'")

    return True


if __name__ == "__main__":
    success = test_custom_api()
    exit(0 if success else 1)
