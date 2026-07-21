"""Tests for configuration wiring in the component factory."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemori.config import MemoryConfig
from nemori.factory import create_memory_system


@pytest.mark.asyncio
async def test_factory_propagates_llm_runtime_defaults():
    config = MemoryConfig(
        llm_api_key="test-key",
        embedding_api_key="test-key",
        llm_timeout=123.0,
        llm_retries=4,
    )

    with patch("nemori.factory.LLMOrchestrator") as orchestrator_cls:
        await create_memory_system(config, AsyncMock(), MagicMock())

    assert orchestrator_cls.call_args.kwargs["default_timeout"] == 123.0
    assert orchestrator_cls.call_args.kwargs["default_retries"] == 4
