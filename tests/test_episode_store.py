"""Tests for PgEpisodeStore."""
import pytest
from unittest.mock import AsyncMock
from datetime import datetime
from src.db.episode_store import PgEpisodeStore
from src.domain.models import Episode


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


@pytest.mark.asyncio
async def test_get_returns_episode(store, mock_db):
    mock_db.fetchrow.return_value = {
        "id": "abc", "user_id": "u1", "title": "T",
        "content": "C", "source_messages": [],
        "metadata": {}, "embedding": None,
        "created_at": datetime.now(), "updated_at": datetime.now(),
    }
    ep = await store.get("abc")
    assert ep is not None
    assert ep.id == "abc"


@pytest.mark.asyncio
async def test_get_returns_none_for_missing(store, mock_db):
    mock_db.fetchrow.return_value = None
    ep = await store.get("missing")
    assert ep is None


@pytest.mark.asyncio
async def test_delete_calls_execute(store, mock_db):
    await store.delete("abc")
    mock_db.execute.assert_called_once()
    call_sql = mock_db.execute.call_args[0][0]
    assert "DELETE FROM episodes" in call_sql


@pytest.mark.asyncio
async def test_search_by_text_uses_tsquery(store, mock_db):
    mock_db.fetch.return_value = []
    await store.search_by_text("u1", "hiking trip", top_k=5)
    call_sql = mock_db.fetch.call_args[0][0]
    assert "ts_rank" in call_sql or "tsv" in call_sql


@pytest.mark.asyncio
async def test_search_by_vector_uses_cosine(store, mock_db):
    mock_db.fetch.return_value = []
    await store.search_by_vector("u1", [0.1] * 1536, top_k=5)
    call_sql = mock_db.fetch.call_args[0][0]
    assert "<=>" in call_sql
