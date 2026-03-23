"""Tests for PgSemanticStore."""
import pytest
from unittest.mock import AsyncMock
from datetime import datetime
from nemori.db.semantic_store import PgSemanticStore
from nemori.domain.models import SemanticMemory


@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.fetch = AsyncMock(return_value=[])
    db.fetchrow = AsyncMock(return_value=None)
    db.execute = AsyncMock(return_value="INSERT 1")
    return db


@pytest.fixture
def store(mock_db):
    return PgSemanticStore(mock_db)


@pytest.mark.asyncio
async def test_save_calls_upsert(store, mock_db):
    sm = SemanticMemory(
        user_id="u1", content="User likes hiking",
        memory_type="preference", embedding=[0.1] * 1536,
    )
    await store.save(sm)
    mock_db.execute.assert_called_once()
    call_sql = mock_db.execute.call_args[0][0]
    assert "INSERT INTO semantic_memories" in call_sql
    assert "ON CONFLICT" in call_sql


@pytest.mark.asyncio
async def test_save_batch(store, mock_db):
    memories = [
        SemanticMemory(user_id="u1", content=f"fact {i}", memory_type="identity")
        for i in range(3)
    ]
    await store.save_batch(memories)
    assert mock_db.execute.call_count == 3


@pytest.mark.asyncio
async def test_get_returns_none_for_missing(store, mock_db):
    result = await store.get("missing-id")
    assert result is None


@pytest.mark.asyncio
async def test_list_by_user_with_type_filter(store, mock_db):
    await store.list_by_user("u1", memory_type="preference")
    call_sql = mock_db.fetch.call_args[0][0]
    assert "memory_type" in call_sql


@pytest.mark.asyncio
async def test_search_hybrid(store, mock_db):
    mock_db.fetch.return_value = []
    await store.search_hybrid("u1", "hiking", [0.1] * 1536, top_k=5)
    call_sql = mock_db.fetch.call_args[0][0]
    assert "rrf_score" in call_sql or "<=>" in call_sql
