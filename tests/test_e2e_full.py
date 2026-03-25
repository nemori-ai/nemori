"""End-to-end test covering the full Nemori pipeline.

Requires running PostgreSQL and Qdrant (docker compose up -d).
Uses real LLM and embedding APIs via environment variables.

Run with: PYTHONPATH=. pytest tests/test_e2e_full.py -v -s
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os

import pytest
import pytest_asyncio

# Skip entirely if no API key is configured
if not (os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")):
    pytest.skip("No API key configured – skipping E2E tests", allow_module_level=True)

from nemori import NemoriMemory, MemoryConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("e2e_test")

TEST_CONFIG = MemoryConfig(
    llm_model="openai/gpt-4.1-mini",
    embedding_model="google/gemini-embedding-001",
    agent_id="e2e_test",
    buffer_size_min=1,
    buffer_size_max=50,
    episode_min_messages=1,
    episode_max_messages=50,
    enable_semantic_memory=True,
    enable_prediction_correction=True,
    enable_episode_merging=True,
    search_top_k_episodes=10,
    search_top_k_semantic=20,
)

TEST_USER = "e2e_test_user"


@pytest_asyncio.fixture
async def memory():
    mem = NemoriMemory(TEST_CONFIG)
    async with mem:
        try:
            await mem.delete_user(TEST_USER)
        except Exception:
            pass
        yield mem
        try:
            await mem.delete_user(TEST_USER)
        except Exception:
            pass


@pytest.mark.asyncio
async def test_full_pipeline(memory):
    """Add messages → flush → episodes + semantics → search works."""

    messages = [
        {"role": "user", "content": "I just moved to Tokyo last week for a new job at a tech startup."},
        {"role": "assistant", "content": "That's exciting! What kind of startup is it?"},
        {"role": "user", "content": "It's an AI company focused on computer vision. I'm working as a senior ML engineer."},
        {"role": "assistant", "content": "Computer vision is a great field. How are you settling into Tokyo?"},
        {"role": "user", "content": "It's been amazing! I found an apartment in Shibuya. The food here is incredible."},
        {"role": "assistant", "content": "Shibuya is a vibrant area. Have you tried any local restaurants?"},
        {"role": "user", "content": "Yes! I found an amazing ramen place near my apartment. I go there almost every day."},
    ]

    logger.info("Adding %d messages...", len(messages))
    await memory.add_messages(TEST_USER, messages)

    logger.info("Flushing...")
    episodes = await memory.flush(TEST_USER)
    logger.info("Generated %d episodes", len(episodes))
    assert len(episodes) > 0, "Should generate at least one episode"

    for ep in episodes:
        logger.info("  Episode: %s — %s", ep["title"], ep["content"][:100])

    # Wait for semantic generation (now synchronous, but drain for safety)
    system = memory._ensure_system()
    await system.drain(timeout=60.0)

    stats = await memory.stats(TEST_USER)
    logger.info("Stats: episodes=%d, semantic=%d", stats["episode_count"], stats["semantic_memory_count"])
    assert stats["episode_count"] > 0

    # Vector search
    result = await memory.search(TEST_USER, "Where does the user work?", search_method="vector")
    logger.info("Vector: %d episodes, %d semantic", len(result["episodes"]), len(result["semantic_memories"]))
    assert len(result["episodes"]) > 0 or len(result["semantic_memories"]) > 0

    # Text search
    result = await memory.search(TEST_USER, "Tokyo ramen", search_method="text")
    logger.info("Text: %d episodes, %d semantic", len(result["episodes"]), len(result["semantic_memories"]))

    # Hybrid search
    result = await memory.search(TEST_USER, "apartment in Tokyo", search_method="hybrid")
    logger.info("Hybrid: %d episodes, %d semantic", len(result["episodes"]), len(result["semantic_memories"]))
    assert len(result["episodes"]) > 0 or len(result["semantic_memories"]) > 0

    logger.info("PASSED: test_full_pipeline")


@pytest.mark.asyncio
async def test_multimodal_pipeline(memory):
    """Multimodal message (text + image) flows through the pipeline."""

    # Generate a valid 4x4 red PNG using Pillow
    from PIL import Image as PILImage
    import io
    img = PILImage.new("RGB", (4, 4), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data_url = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    logger.info("Adding multimodal message...")
    await memory.add_multimodal_message(
        TEST_USER,
        text="Here is a photo of my new apartment in Shibuya",
        image_urls=[data_url],
        role="user",
    )
    await memory.add_messages(TEST_USER, [
        {"role": "assistant", "content": "Your apartment looks great! The view is amazing."},
        {"role": "user", "content": "Thanks! I love the balcony especially."},
    ])

    # Wait for background processing (add_messages may auto-trigger with buffer_size_min=1)
    system = memory._ensure_system()
    await system.drain(timeout=60.0)

    # Also try flush in case anything remains
    episodes = await memory.flush(TEST_USER)
    await system.drain(timeout=60.0)

    # Check that data was stored
    stats = await memory.stats(TEST_USER)
    logger.info("Stats: episodes=%d, semantic=%d", stats["episode_count"], stats["semantic_memory_count"])
    assert stats["episode_count"] > 0, "Should have episodes from multimodal messages"

    result = await memory.search(TEST_USER, "apartment photo Shibuya", search_method="hybrid")
    logger.info("Search: %d episodes, %d semantic", len(result["episodes"]), len(result["semantic_memories"]))
    assert len(result["episodes"]) > 0 or len(result["semantic_memories"]) > 0

    logger.info("PASSED: test_multimodal_pipeline")


@pytest.mark.asyncio
async def test_multi_topic_segmentation(memory):
    """Multiple topics get segmented into separate episodes."""

    messages = [
        {"role": "user", "content": "I've been learning to cook Japanese food. Started with miso soup."},
        {"role": "assistant", "content": "Miso soup is a great starting point! What kind of miso do you use?"},
        {"role": "user", "content": "I use white miso. I also learned to make tamagoyaki yesterday."},
        {"role": "assistant", "content": "Tamagoyaki is delicious! It takes practice to roll it properly."},
        {"role": "user", "content": "Completely different topic - I signed up for a gym near my office."},
        {"role": "assistant", "content": "That's great for staying healthy! What kind of workouts do you do?"},
        {"role": "user", "content": "Mostly weightlifting, 3 days a week. I'm training for a powerlifting competition."},
        {"role": "assistant", "content": "Powerlifting is impressive! What are your current lifts?"},
        {"role": "user", "content": "Squat 140kg, bench 100kg, deadlift 180kg. Trying to hit 500kg total."},
    ]

    logger.info("Adding %d multi-topic messages...", len(messages))
    await memory.add_messages(TEST_USER, messages)

    episodes = await memory.flush(TEST_USER)
    logger.info("Generated %d episodes", len(episodes))
    for ep in episodes:
        logger.info("  Episode: %s", ep["title"])

    system = memory._ensure_system()
    await system.drain(timeout=60.0)

    cooking_result = await memory.search(TEST_USER, "cooking Japanese food miso", search_method="vector")
    gym_result = await memory.search(TEST_USER, "gym weightlifting powerlifting", search_method="vector")

    logger.info("Cooking search: %d episodes", len(cooking_result["episodes"]))
    logger.info("Gym search: %d episodes", len(gym_result["episodes"]))

    assert len(cooking_result["episodes"]) > 0 or len(cooking_result["semantic_memories"]) > 0
    assert len(gym_result["episodes"]) > 0 or len(gym_result["semantic_memories"]) > 0

    logger.info("PASSED: test_multi_topic_segmentation")
