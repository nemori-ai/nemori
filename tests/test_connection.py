"""Tests for DatabaseManager."""
import pytest
from unittest.mock import AsyncMock, patch
from nemori.db.connection import DatabaseManager
from nemori.domain.exceptions import DatabaseError


@pytest.mark.asyncio
async def test_init_creates_pool():
    dm = DatabaseManager()
    mock_pool = AsyncMock()
    with patch("nemori.db.connection.asyncpg.create_pool", new_callable=AsyncMock, return_value=mock_pool):
        await dm.init("postgresql://localhost/test")
        assert dm.pool is not None
    await dm.close()


@pytest.mark.asyncio
async def test_close_releases_pool():
    dm = DatabaseManager()
    mock_pool = AsyncMock()
    with patch("nemori.db.connection.asyncpg.create_pool", new_callable=AsyncMock, return_value=mock_pool):
        await dm.init("postgresql://localhost/test")
        await dm.close()
        mock_pool.close.assert_called_once()


@pytest.mark.asyncio
async def test_operations_before_init_raise():
    dm = DatabaseManager()
    with pytest.raises(DatabaseError):
        await dm.ping()


@pytest.mark.asyncio
async def test_executemany_method_exists():
    dm = DatabaseManager()
    assert hasattr(dm, "executemany")
    # Should raise DatabaseError since not initialized
    with pytest.raises(DatabaseError):
        await dm.executemany("INSERT INTO t VALUES ($1)", [("a",)])
