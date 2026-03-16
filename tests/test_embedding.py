"""Tests for AsyncEmbeddingClient."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.services.embedding import AsyncEmbeddingClient


@pytest.mark.asyncio
async def test_embed_returns_float_list():
    client = AsyncEmbeddingClient(api_key="test", model="text-embedding-3-small")
    mock_response = MagicMock()
    mock_response.data = [MagicMock()]
    mock_response.data[0].embedding = [0.1, 0.2, 0.3]

    with patch.object(client, "_client") as mock_client:
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)
        result = await client.embed("hello")
        assert result == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_embed_batch_returns_list_of_lists():
    client = AsyncEmbeddingClient(api_key="test", model="text-embedding-3-small")
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(embedding=[0.1, 0.2]),
        MagicMock(embedding=[0.3, 0.4]),
    ]

    with patch.object(client, "_client") as mock_client:
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)
        result = await client.embed_batch(["hello", "world"])
        assert len(result) == 2
        assert result[0] == [0.1, 0.2]
