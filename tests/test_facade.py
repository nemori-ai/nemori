"""Tests for async NemoriMemory facade."""
import importlib.util
import os
import sys
import types

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

# Pre-register src.search as a lightweight namespace to avoid chromadb import
# via src/search/__init__.py
if "src.search" not in sys.modules:
    _search_pkg = types.ModuleType("src.search")
    _search_pkg.__path__ = [os.path.join(os.path.dirname(__file__), "..", "src", "search")]
    sys.modules["src.search"] = _search_pkg

# Load src/search/unified.py directly (bypasses chromadb-heavy __init__.py)
_spec = importlib.util.spec_from_file_location(
    "src.search.unified",
    os.path.join(os.path.dirname(__file__), "..", "src", "search", "unified.py"),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["src.search.unified"] = _mod
_spec.loader.exec_module(_mod)

from src.api.facade import NemoriMemory
from src.config import MemoryConfig


@pytest.mark.asyncio
async def test_facade_context_manager():
    with patch("src.api.facade.DatabaseManager") as MockDB, \
         patch("src.api.facade.NemoriMemory._build_system", new_callable=AsyncMock):
        mock_db_instance = AsyncMock()
        MockDB.return_value = mock_db_instance

        config = MemoryConfig(dsn="postgresql://localhost/test", llm_api_key="test")
        async with NemoriMemory(config=config) as memory:
            assert memory is not None
        mock_db_instance.close.assert_called_once()


@pytest.mark.asyncio
async def test_facade_add_messages():
    with patch("src.api.facade.DatabaseManager") as MockDB, \
         patch("src.api.facade.NemoriMemory._build_system", new_callable=AsyncMock):
        MockDB.return_value = AsyncMock()

        config = MemoryConfig(dsn="postgresql://localhost/test", llm_api_key="test")
        async with NemoriMemory(config=config) as memory:
            memory._system = AsyncMock()
            await memory.add_messages("u1", [{"role": "user", "content": "hi"}])
            memory._system.add_messages.assert_called_once()


@pytest.mark.asyncio
async def test_facade_health():
    with patch("src.api.facade.DatabaseManager") as MockDB, \
         patch("src.api.facade.NemoriMemory._build_system", new_callable=AsyncMock):
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
