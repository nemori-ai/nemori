"""Tests for async MemorySystem."""
import sys
import types

# Stub out heavy transitive imports before anything touches them.
for _mod_name in [
    "chromadb",
    "src.search.chroma_search",
    "src.search.bm25_search",
    "src.search.unified_search",
    "src.search.original_message_search",
]:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = types.ModuleType(_mod_name)

# Ensure src.search package exists so unified.py can be imported directly
if "src.search" not in sys.modules:
    _search = types.ModuleType("src.search")
    _search.__path__ = ["src/search"]
    sys.modules["src.search"] = _search

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from src.core.memory_system import MemorySystem
from src.domain.models import Message, Episode, SemanticMemory
from src.search.unified import SearchResult
from src.config import MemoryConfig


@pytest.fixture
def deps():
    return {
        "config": MemoryConfig(),
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
    deps["buffer_store"].push.assert_called_once_with("u1", msgs)


@pytest.mark.asyncio
async def test_flush_processes_buffer(system, deps):
    deps["buffer_store"].get_unprocessed.return_value = [
        Message(role="user", content="hello", metadata={"buffer_id": 1}),
        Message(role="assistant", content="hi", metadata={"buffer_id": 2}),
    ]
    ep = Episode(user_id="u1", title="T", content="C", source_messages=[])
    deps["episode_generator"].generate = AsyncMock(return_value=ep)

    result = await system.flush("u1")
    assert len(result) >= 1
    deps["episode_store"].save.assert_called()


@pytest.mark.asyncio
async def test_search_delegates(system, deps):
    await system.search("u1", "hiking")
    deps["search"].search.assert_called_once()


@pytest.mark.asyncio
async def test_delete_episode(system, deps):
    await system.delete_episode("u1", "ep-1")
    deps["episode_store"].delete.assert_called_once_with("ep-1")


@pytest.mark.asyncio
async def test_delete_user(system, deps):
    await system.delete_user("u1")
    deps["episode_store"].delete_by_user.assert_called_once_with("u1")
    deps["semantic_store"].delete_by_user.assert_called_once_with("u1")


@pytest.mark.asyncio
async def test_drain(system):
    await system.drain(timeout=1.0)
