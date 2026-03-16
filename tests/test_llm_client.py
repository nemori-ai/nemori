"""Tests for AsyncLLMClient."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.llm.client import AsyncLLMClient


@pytest.mark.asyncio
async def test_complete_returns_string():
    client = AsyncLLMClient(api_key="test-key", base_url=None)
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Hello!"

    with patch.object(client, "_client") as mock_client:
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        result = await client.complete(
            [{"role": "user", "content": "hi"}], model="gpt-4o-mini"
        )
        assert result == "Hello!"


@pytest.mark.asyncio
async def test_complete_passes_params():
    client = AsyncLLMClient(api_key="test-key", base_url=None)
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "ok"

    with patch.object(client, "_client") as mock_client:
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        await client.complete(
            [{"role": "user", "content": "hi"}],
            model="gpt-4o", temperature=0.5, max_tokens=100,
        )
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100
