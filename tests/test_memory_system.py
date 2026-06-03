"""Tests for async MemorySystem."""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from nemori.core.memory_system import MemorySystem
from nemori.domain.models import Message, Episode, SemanticMemory
from nemori.search.unified import SearchResult
from nemori.config import MemoryConfig


@pytest.fixture
def deps():
    qdrant = MagicMock()
    qdrant.upsert_episode = MagicMock()
    qdrant.upsert_semantic = MagicMock()
    qdrant.delete_episode = MagicMock()
    qdrant.delete_semantic = MagicMock()
    qdrant.delete_episodes_by_user = MagicMock()
    qdrant.delete_semantic_by_user = MagicMock()
    return {
        "config": MemoryConfig(),
        "agent_id": "default",
        "db": AsyncMock(),
        "episode_store": AsyncMock(),
        "semantic_store": AsyncMock(),
        "buffer_store": AsyncMock(),
        "orchestrator": AsyncMock(),
        "embedding": AsyncMock(),
        "episode_generator": AsyncMock(),
        "semantic_generator": AsyncMock(),
        "event_bus": AsyncMock(),
        "search": AsyncMock(),
        "qdrant": qdrant,
    }


@pytest.fixture
def system(deps):
    deps["buffer_store"].count_unprocessed = AsyncMock(return_value=0)
    deps["buffer_store"].get_unprocessed = AsyncMock(return_value=[])
    deps["search"].search = AsyncMock(return_value=SearchResult())
    return MemorySystem(**deps)


@pytest.mark.asyncio
async def test_add_messages_pushes_to_buffer(system, deps):
    msgs = [Message(role="user", content="hi")]
    await system.add_messages("u1", msgs)
    deps["buffer_store"].push.assert_called_once_with("u1", "default", msgs)


@pytest.mark.asyncio
async def test_flush_processes_buffer(system, deps):
    deps["buffer_store"].get_unprocessed.return_value = [
        Message(role="user", content="hello", metadata={"buffer_id": 1}),
        Message(role="assistant", content="hi", metadata={"buffer_id": 2}),
    ]
    ep = Episode(user_id="u1", title="T", content="C", source_messages=[], embedding=[0.1] * 10)
    deps["episode_generator"].generate = AsyncMock(return_value=ep)

    result = await system.flush("u1")
    assert len(result) >= 1
    deps["episode_store"].save.assert_called()
    # Qdrant upsert should be called for episode with embedding
    deps["qdrant"].upsert_episode.assert_called()


@pytest.mark.asyncio
async def test_search_delegates(system, deps):
    await system.search("u1", "hiking")
    deps["search"].search.assert_called_once()


@pytest.mark.asyncio
async def test_delete_episode(system, deps):
    await system.delete_episode("u1", "ep-1")
    deps["episode_store"].delete.assert_called_once_with("ep-1", "u1", "default")
    deps["qdrant"].delete_episode.assert_called_once_with("ep-1")


@pytest.mark.asyncio
async def test_delete_user(system, deps):
    await system.delete_user("u1")
    deps["episode_store"].delete_by_user.assert_called_once_with("u1", "default")
    deps["semantic_store"].delete_by_user.assert_called_once_with("u1", "default")
    deps["qdrant"].delete_episodes_by_user.assert_called_once_with("u1", "default")
    deps["qdrant"].delete_semantic_by_user.assert_called_once_with("u1", "default")


@pytest.mark.asyncio
async def test_drain(system):
    await system.drain(timeout=1.0)


# --- Semantic dedup / supersession -----------------------------------------

def test_resolve_dedupe_id_above_threshold():
    hits = [{"id": "existing-1", "score": 0.92}]
    assert MemorySystem._resolve_dedupe_id(hits, 0.85, set()) == "existing-1"


def test_resolve_dedupe_id_below_threshold():
    hits = [{"id": "existing-1", "score": 0.50}]
    assert MemorySystem._resolve_dedupe_id(hits, 0.85, set()) is None


def test_resolve_dedupe_id_skips_already_used():
    hits = [{"id": "existing-1", "score": 0.99}]
    assert MemorySystem._resolve_dedupe_id(hits, 0.85, {"existing-1"}) is None


def test_resolve_dedupe_id_empty_hits():
    assert MemorySystem._resolve_dedupe_id([], 0.85, set()) is None


@pytest.mark.asyncio
async def test_dedupe_semantic_supersedes_near_duplicate(system, deps):
    deps["qdrant"].search_semantic = MagicMock(
        return_value=[{"id": "old-fact", "score": 0.95}]
    )
    mem = SemanticMemory(
        user_id="u1", content="User works at Globex",
        memory_type="identity", embedding=[0.1] * 10,
    )
    original_id = mem.id
    result = await system._dedupe_semantic("u1", [mem])
    # Near-duplicate reuses the existing id so the store UPDATEs in place.
    assert result[0].id == "old-fact"
    assert result[0].id != original_id


@pytest.mark.asyncio
async def test_dedupe_semantic_keeps_distinct_fact(system, deps):
    deps["qdrant"].search_semantic = MagicMock(
        return_value=[{"id": "old-fact", "score": 0.40}]
    )
    mem = SemanticMemory(
        user_id="u1", content="A brand new unrelated fact",
        memory_type="identity", embedding=[0.1] * 10,
    )
    original_id = mem.id
    result = await system._dedupe_semantic("u1", [mem])
    # Below threshold -> stays a fresh memory.
    assert result[0].id == original_id


@pytest.mark.asyncio
async def test_dedupe_semantic_disabled_via_config(deps):
    deps["config"] = MemoryConfig(enable_semantic_dedup=False)
    deps["buffer_store"].count_unprocessed = AsyncMock(return_value=0)
    sys_no_dedup = MemorySystem(**deps)
    deps["qdrant"].search_semantic = MagicMock(
        return_value=[{"id": "old-fact", "score": 0.99}]
    )
    mem = SemanticMemory(
        user_id="u1", content="x", memory_type="identity", embedding=[0.1] * 10,
    )
    original_id = mem.id
    result = await sys_no_dedup._dedupe_semantic("u1", [mem])
    assert result[0].id == original_id
    deps["qdrant"].search_semantic.assert_not_called()
