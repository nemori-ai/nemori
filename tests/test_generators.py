"""Tests for LLM generators."""
import pytest
from unittest.mock import AsyncMock
from nemori.llm.orchestrator import LLMOrchestrator, LLMResponse, TokenUsage
from nemori.llm.generators.episode import EpisodeGenerator
from nemori.llm.generators.semantic import SemanticGenerator
from nemori.llm.generators.segmenter import BatchSegmenter
from nemori.domain.models import Message, Episode


@pytest.fixture
def mock_orchestrator():
    return AsyncMock(spec=LLMOrchestrator)


@pytest.fixture
def mock_embedding():
    emb = AsyncMock()
    emb.embed = AsyncMock(return_value=[0.1] * 1536)
    return emb


@pytest.mark.asyncio
async def test_episode_generator_returns_episode(mock_orchestrator, mock_embedding):
    mock_orchestrator.execute = AsyncMock(return_value=LLMResponse(
        content='{"title": "Test Episode", "content": "User discussed hiking.", "timestamp": "2024-01-01T10:00:00"}',
        model="gpt-4o-mini", usage=TokenUsage(), latency_ms=100, request_id="abc",
    ))
    gen = EpisodeGenerator(orchestrator=mock_orchestrator, embedding=mock_embedding)
    messages = [
        Message(role="user", content="I love hiking"),
        Message(role="assistant", content="That's great!"),
    ]
    episode = await gen.generate("u1", messages, "topic_change")
    assert isinstance(episode, Episode)
    assert episode.title == "Test Episode"
    assert episode.user_id == "u1"
    assert episode.embedding is not None


@pytest.mark.asyncio
async def test_episode_generator_fallback_on_bad_json(mock_orchestrator, mock_embedding):
    mock_orchestrator.execute = AsyncMock(return_value=LLMResponse(
        content="not valid json", model="gpt-4o-mini",
        usage=TokenUsage(), latency_ms=100, request_id="abc",
    ))
    gen = EpisodeGenerator(orchestrator=mock_orchestrator, embedding=mock_embedding)
    messages = [Message(role="user", content="test")]
    episode = await gen.generate("u1", messages, "fallback_test")
    assert isinstance(episode, Episode)
    assert episode.user_id == "u1"


@pytest.mark.asyncio
async def test_semantic_generator_returns_memories(mock_orchestrator, mock_embedding):
    mock_orchestrator.execute = AsyncMock(return_value=LLMResponse(
        content='{"statements": ["User likes hiking", "User works at Google"]}',
        model="gpt-4o-mini", usage=TokenUsage(), latency_ms=100, request_id="abc",
    ))
    gen = SemanticGenerator(orchestrator=mock_orchestrator, embedding=mock_embedding)
    episode = Episode(user_id="u1", title="T", content="C", source_messages=[])
    memories = await gen.generate("u1", episode, [], [])
    assert len(memories) == 2
    assert memories[0].content == "User likes hiking"


@pytest.mark.asyncio
async def test_episode_generator_multimodal(mock_orchestrator, mock_embedding):
    mock_orchestrator.execute = AsyncMock(return_value=LLMResponse(
        content='{"title": "Photo Discussion", "content": "User shared a mountain photo.", "timestamp": "2024-01-01T10:00:00"}',
        model="gpt-4o-mini", usage=TokenUsage(), latency_ms=100, request_id="abc",
    ))
    gen = EpisodeGenerator(orchestrator=mock_orchestrator, embedding=mock_embedding)
    messages = [
        Message(role="user", content=[
            {"type": "text", "text": "Look at this mountain!"},
            {"type": "image_url", "image_url": {"url": "https://example.com/mountain.jpg"}},
        ]),
        Message(role="assistant", content="Beautiful view!"),
    ]
    episode = await gen.generate("u1", messages, "topic_change")
    assert isinstance(episode, Episode)
    # Verify the LLM request contained image content
    call_args = mock_orchestrator.execute.call_args[0][0]
    user_msg = call_args.messages[1]
    assert isinstance(user_msg["content"], list)  # multimodal content array


@pytest.mark.asyncio
async def test_episode_generator_fallback_uses_text_content(mock_orchestrator, mock_embedding):
    """Fallback should use text_content(), not raw content."""
    mock_orchestrator.execute = AsyncMock(side_effect=Exception("LLM down"))
    gen = EpisodeGenerator(orchestrator=mock_orchestrator, embedding=mock_embedding)
    messages = [
        Message(role="user", content=[
            {"type": "text", "text": "Check this"},
            {"type": "image_url", "image_url": {"url": "https://img.png"}},
        ]),
    ]
    episode = await gen.generate("u1", messages, "test")
    # Should NOT contain raw list repr in content
    assert "[{" not in episode.content
    assert "Check this" in episode.content


@pytest.mark.asyncio
async def test_segmenter_handles_multimodal(mock_orchestrator):
    mock_orchestrator.execute = AsyncMock(return_value=LLMResponse(
        content='{"episodes": [{"indices": [1, 2], "topic": "photos"}]}',
        model="gpt-4o-mini", usage=TokenUsage(), latency_ms=100, request_id="abc",
    ))
    seg = BatchSegmenter(orchestrator=mock_orchestrator)
    messages = [
        Message(role="user", content=[
            {"type": "text", "text": "Look at this"},
            {"type": "image_url", "image_url": {"url": "https://img.png"}},
        ]),
        Message(role="assistant", content="Nice!"),
    ]
    groups = await seg.segment(messages)
    assert len(groups) == 1


@pytest.mark.asyncio
async def test_segmenter_returns_groups(mock_orchestrator):
    mock_orchestrator.execute = AsyncMock(return_value=LLMResponse(
        content='{"episodes": [{"indices": [1, 2], "topic": "hiking"}, {"indices": [3, 4], "topic": "work"}]}',
        model="gpt-4o-mini", usage=TokenUsage(), latency_ms=100, request_id="abc",
    ))
    seg = BatchSegmenter(orchestrator=mock_orchestrator)
    messages = [Message(role="user", content=f"msg {i}") for i in range(4)]
    groups = await seg.segment(messages)
    assert len(groups) == 2
    assert len(groups[0]["messages"]) == 2
