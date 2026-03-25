"""Tests for PgEpisodeStore."""
import pytest
from unittest.mock import AsyncMock
from datetime import datetime
from nemori.db.episode_store import PgEpisodeStore
from nemori.domain.models import Episode


@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.fetch = AsyncMock(return_value=[])
    db.fetchrow = AsyncMock(return_value=None)
    db.execute = AsyncMock(return_value="INSERT 1")
    db.fetchval = AsyncMock(return_value=0)
    return db


@pytest.fixture
def store(mock_db):
    return PgEpisodeStore(mock_db)


@pytest.mark.asyncio
async def test_save_calls_upsert(store, mock_db):
    ep = Episode(
        user_id="u1", title="Test", content="Content",
        source_messages=[], embedding=[0.1] * 1536,
    )
    await store.save(ep)
    mock_db.execute.assert_called_once()
    call_sql = mock_db.execute.call_args[0][0]
    assert "INSERT INTO episodes" in call_sql
    assert "ON CONFLICT" in call_sql
    # Embedding should NOT be in the SQL anymore
    assert "embedding" not in call_sql.lower()


@pytest.mark.asyncio
async def test_get_returns_episode(store, mock_db):
    mock_db.fetchrow.return_value = {
        "id": "abc", "user_id": "u1", "agent_id": "default", "title": "T",
        "content": "C", "source_messages": [],
        "metadata": {},
        "created_at": datetime.now(), "updated_at": datetime.now(),
    }
    ep = await store.get("abc", "u1", "default")
    assert ep is not None
    assert ep.id == "abc"
    assert ep.embedding is None


@pytest.mark.asyncio
async def test_get_returns_none_for_missing(store, mock_db):
    mock_db.fetchrow.return_value = None
    ep = await store.get("missing", "u1", "default")
    assert ep is None


@pytest.mark.asyncio
async def test_delete_calls_execute(store, mock_db):
    await store.delete("abc", "u1", "default")
    mock_db.execute.assert_called_once()
    call_sql = mock_db.execute.call_args[0][0]
    assert "DELETE FROM episodes" in call_sql


@pytest.mark.asyncio
async def test_search_by_text_uses_tsquery(store, mock_db):
    mock_db.fetch.return_value = []
    await store.search_by_text("u1", "default", "hiking trip", top_k=5)
    call_sql = mock_db.fetch.call_args[0][0]
    assert "ts_rank" in call_sql or "tsv" in call_sql


@pytest.mark.asyncio
async def test_get_batch(store, mock_db):
    mock_db.fetch.return_value = []
    result = await store.get_batch(["id1", "id2"], "u1", "default")
    assert result == []
    mock_db.fetch.assert_called_once()
    call_sql = mock_db.fetch.call_args[0][0]
    assert "ANY" in call_sql
