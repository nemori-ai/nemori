"""Tests for async NemoriMemory facade."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from nemori.api.facade import NemoriMemory
from nemori.config import MemoryConfig


@pytest.mark.asyncio
async def test_facade_context_manager():
    with patch("nemori.api.facade.DatabaseManager") as MockDB, \
         patch("nemori.api.facade.NemoriMemory._build_system", new_callable=AsyncMock):
        mock_db_instance = AsyncMock()
        MockDB.return_value = mock_db_instance

        config = MemoryConfig(dsn="postgresql://localhost/test", llm_api_key="test")
        async with NemoriMemory(config=config) as memory:
            assert memory is not None
        mock_db_instance.close.assert_called_once()


@pytest.mark.asyncio
async def test_facade_add_messages():
    with patch("nemori.api.facade.DatabaseManager") as MockDB, \
         patch("nemori.api.facade.NemoriMemory._build_system", new_callable=AsyncMock):
        MockDB.return_value = AsyncMock()

        config = MemoryConfig(dsn="postgresql://localhost/test", llm_api_key="test")
        async with NemoriMemory(config=config) as memory:
            memory._system = AsyncMock()
            await memory.add_messages("u1", [{"role": "user", "content": "hi"}])
            memory._system.add_messages.assert_called_once()


@pytest.mark.asyncio
async def test_facade_health():
    with patch("nemori.api.facade.DatabaseManager") as MockDB, \
         patch("nemori.api.facade.NemoriMemory._build_system", new_callable=AsyncMock):
        mock_db = AsyncMock()
        mock_db.ping = AsyncMock(return_value=True)
        mock_db.pool = MagicMock()
        mock_db.pool.get_size.return_value = 10
        mock_db.pool.get_idle_size.return_value = 8
        MockDB.return_value = mock_db

        config = MemoryConfig(dsn="postgresql://localhost/test", llm_api_key="test")
        async with NemoriMemory(config=config) as memory:
            health = await memory.health()
            assert health.db is True


@pytest.mark.asyncio
async def test_facade_not_initialized_raises():
    memory = NemoriMemory()
    with pytest.raises(RuntimeError, match="not initialized"):
        await memory.add_messages("u1", [{"role": "user", "content": "hi"}])
