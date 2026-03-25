"""Tests for async NemoriMemory facade."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from nemori.api.facade import NemoriMemory
from nemori.config import MemoryConfig


@pytest.mark.asyncio
async def test_facade_context_manager():
    with patch("nemori.api.facade.DatabaseManager") as MockDB, \
         patch("nemori.api.facade.QdrantVectorStore") as MockQdrant, \
         patch("nemori.api.facade.NemoriMemory._build_system", new_callable=AsyncMock):
        mock_db_instance = AsyncMock()
        MockDB.return_value = mock_db_instance
        mock_qdrant = MagicMock()
        MockQdrant.return_value = mock_qdrant

        config = MemoryConfig(dsn="postgresql://localhost/test", llm_api_key="test")
        async with NemoriMemory(config=config) as memory:
            assert memory is not None
        mock_db_instance.close.assert_called_once()
        mock_qdrant.close.assert_called_once()


@pytest.mark.asyncio
async def test_facade_add_messages():
    with patch("nemori.api.facade.DatabaseManager") as MockDB, \
         patch("nemori.api.facade.QdrantVectorStore") as MockQdrant, \
         patch("nemori.api.facade.NemoriMemory._build_system", new_callable=AsyncMock):
        MockDB.return_value = AsyncMock()
        MockQdrant.return_value = MagicMock()

        config = MemoryConfig(dsn="postgresql://localhost/test", llm_api_key="test")
        async with NemoriMemory(config=config) as memory:
            memory._system = AsyncMock()
            await memory.add_messages("u1", [{"role": "user", "content": "hi"}])
            memory._system.add_messages.assert_called_once()


@pytest.mark.asyncio
async def test_facade_health():
    with patch("nemori.api.facade.DatabaseManager") as MockDB, \
         patch("nemori.api.facade.QdrantVectorStore") as MockQdrant, \
         patch("nemori.api.facade.NemoriMemory._build_system", new_callable=AsyncMock):
        mock_db = AsyncMock()
        mock_db.ping = AsyncMock(return_value=True)
        mock_db.pool = MagicMock()
        mock_db.pool.get_size.return_value = 10
        mock_db.pool.get_idle_size.return_value = 8
        MockDB.return_value = mock_db
        MockQdrant.return_value = MagicMock()

        config = MemoryConfig(dsn="postgresql://localhost/test", llm_api_key="test")
        async with NemoriMemory(config=config) as memory:
            health = await memory.health()
            assert health.db is True


@pytest.mark.asyncio
async def test_facade_add_messages_preserves_timestamp_and_metadata():
    with patch("nemori.api.facade.DatabaseManager") as MockDB, \
         patch("nemori.api.facade.QdrantVectorStore") as MockQdrant, \
         patch("nemori.api.facade.NemoriMemory._build_system", new_callable=AsyncMock):
        MockDB.return_value = AsyncMock()
        MockQdrant.return_value = MagicMock()

        config = MemoryConfig(dsn="postgresql://localhost/test", llm_api_key="test")
        async with NemoriMemory(config=config) as memory:
            memory._system = AsyncMock()
            await memory.add_messages("u1", [
                {
                    "role": "user",
                    "content": "hi",
                    "timestamp": "2023-05-08T13:56:00",
                    "metadata": {"source": "test"},
                }
            ])
            memory._system.add_messages.assert_called_once()
            args = memory._system.add_messages.call_args
            msg = args[0][1][0]  # second positional arg, first message
            from datetime import datetime
            assert msg.timestamp == datetime(2023, 5, 8, 13, 56, 0)
            assert msg.metadata == {"source": "test"}


@pytest.mark.asyncio
async def test_add_multimodal_message_builds_content_array():
    """add_multimodal_message should build proper content array with compressed images."""
    with patch("nemori.api.facade.DatabaseManager") as MockDB, \
         patch("nemori.api.facade.QdrantVectorStore") as MockQdrant, \
         patch("nemori.api.facade.NemoriMemory._build_system", new_callable=AsyncMock):
        MockDB.return_value = AsyncMock()
        MockQdrant.return_value = MagicMock()
        config = MemoryConfig(dsn="postgresql://localhost/test", llm_api_key="test")
        async with NemoriMemory(config=config) as memory:
            memory._system = AsyncMock()
            import base64, io
            from PIL import Image as PILImage
            img = PILImage.new("RGB", (10, 10), "red")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            data_url = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

            await memory.add_multimodal_message(
                "u1", "Check this image", image_urls=[data_url], compress_images=True
            )
            call_args = memory._system.add_messages.call_args
            msg = call_args[0][1][0]
            assert isinstance(msg.content, list)
            assert msg.content[0]["type"] == "text"
            assert msg.content[1]["type"] == "image_url"
            assert "image/jpeg" in msg.content[1]["image_url"]["url"]


@pytest.mark.asyncio
async def test_add_multimodal_message_text_only_fallback():
    """Without image_urls, add_multimodal_message should add a plain text message."""
    with patch("nemori.api.facade.DatabaseManager") as MockDB, \
         patch("nemori.api.facade.QdrantVectorStore") as MockQdrant, \
         patch("nemori.api.facade.NemoriMemory._build_system", new_callable=AsyncMock):
        MockDB.return_value = AsyncMock()
        MockQdrant.return_value = MagicMock()
        config = MemoryConfig(dsn="postgresql://localhost/test", llm_api_key="test")
        async with NemoriMemory(config=config) as memory:
            memory._system = AsyncMock()
            await memory.add_multimodal_message("u1", "Just text")
            call_args = memory._system.add_messages.call_args
            msg = call_args[0][1][0]
            assert msg.content == "Just text"


@pytest.mark.asyncio
async def test_facade_not_initialized_raises():
    memory = NemoriMemory()
    with pytest.raises(RuntimeError, match="not initialized"):
        await memory.add_messages("u1", [{"role": "user", "content": "hi"}])
