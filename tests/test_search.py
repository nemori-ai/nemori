"""Tests for UnifiedSearch."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from nemori.search.unified import UnifiedSearch, SearchMethod, SearchResult


@pytest.fixture
def mock_episode_store():
    store = AsyncMock()
    store.search_by_text = AsyncMock(return_value=[])
    store.get_batch = AsyncMock(return_value=[])
    return store


@pytest.fixture
def mock_semantic_store():
    store = AsyncMock()
    store.search_by_text = AsyncMock(return_value=[])
    store.get_batch = AsyncMock(return_value=[])
    return store


@pytest.fixture
def mock_embedding():
    emb = AsyncMock()
    emb.embed = AsyncMock(return_value=[0.1] * 1536)
    return emb


@pytest.fixture
def mock_qdrant():
    qdrant = MagicMock()
    qdrant.search_episodes = MagicMock(return_value=[])
    qdrant.search_semantic = MagicMock(return_value=[])
    return qdrant


@pytest.fixture
def search(mock_episode_store, mock_semantic_store, mock_embedding, mock_qdrant):
    return UnifiedSearch(mock_episode_store, mock_semantic_store, mock_embedding, mock_qdrant)


@pytest.mark.asyncio
async def test_hybrid_search(search, mock_qdrant, mock_episode_store, mock_semantic_store):
    result = await search.search("u1", "default", "hiking", method=SearchMethod.HYBRID)
    assert isinstance(result, SearchResult)
    mock_qdrant.search_episodes.assert_called_once()
    mock_qdrant.search_semantic.assert_called_once()
    mock_episode_store.search_by_text.assert_called_once()
    mock_semantic_store.search_by_text.assert_called_once()


@pytest.mark.asyncio
async def test_vector_search(search, mock_qdrant):
    await search.search("u1", "default", "hiking", method=SearchMethod.VECTOR)
    mock_qdrant.search_episodes.assert_called_once()
    mock_qdrant.search_semantic.assert_called_once()


@pytest.mark.asyncio
async def test_text_search(search, mock_episode_store):
    await search.search("u1", "default", "hiking", method=SearchMethod.TEXT)
    mock_episode_store.search_by_text.assert_called_once()


@pytest.mark.asyncio
async def test_search_result_to_dict(search):
    result = await search.search("u1", "default", "test")
    d = result.to_dict()
    assert "episodes" in d
    assert "semantic_memories" in d
