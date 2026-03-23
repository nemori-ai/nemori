"""Tests for PgMessageBufferStore."""
import pytest
from unittest.mock import AsyncMock
from nemori.db.buffer_store import PgMessageBufferStore
from nemori.domain.models import Message


@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.fetch = AsyncMock(return_value=[])
    db.execute = AsyncMock()
    db.fetchval = AsyncMock(return_value=0)
    return db


@pytest.fixture
def store(mock_db):
    return PgMessageBufferStore(mock_db)


@pytest.mark.asyncio
async def test_push_inserts_messages(store, mock_db):
    messages = [
        Message(role="user", content="hello"),
        Message(role="assistant", content="hi there"),
    ]
    await store.push("u1", "default", messages)
    assert mock_db.execute.call_count == 2


@pytest.mark.asyncio
async def test_count_unprocessed(store, mock_db):
    mock_db.fetchval.return_value = 5
    count = await store.count_unprocessed("u1", "default")
    assert count == 5
    call_sql = mock_db.fetchval.call_args[0][0]
    assert "NOT processed" in call_sql


@pytest.mark.asyncio
async def test_mark_processed_and_delete(store, mock_db):
    await store.mark_processed("u1", "default", [1, 2, 3])
    call_sql = mock_db.execute.call_args[0][0]
    assert "DELETE" in call_sql
