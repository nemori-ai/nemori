"""Tests for LLM generators."""
import json

import pytest
from unittest.mock import AsyncMock
from nemori.llm.orchestrator import LLMOrchestrator, LLMRequest, LLMResponse, TokenUsage
from nemori.llm.generators.episode import EpisodeGenerator
from nemori.llm.generators.merger import EpisodeMerger
from nemori.llm.generators.semantic import SemanticGenerator
from nemori.llm.generators.segmenter import BatchSegmenter
from nemori.domain.models import Message, Episode, SemanticMemory


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
    episode = await gen.generate("u1", "default", messages, "topic_change")
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
    episode = await gen.generate("u1", "default", messages, "fallback_test")
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
    memories = await gen.generate("u1", "default", episode, [])
    assert len(memories) == 2
    assert memories[0].content == "User likes hiking"
    assert mock_orchestrator.execute.await_count == 1
    assert mock_orchestrator.execute.await_args_list[0].args[0].metadata["generator"] == "semantic_direct"


@pytest.mark.asyncio
async def test_semantic_generator_uses_prediction_when_prior_knowledge_exists(
    mock_orchestrator, mock_embedding
):
    mock_orchestrator.execute = AsyncMock(side_effect=[
        LLMResponse(
            content="Predicted episode", model="test",
            usage=TokenUsage(), latency_ms=1, request_id="predict",
        ),
        LLMResponse(
            content='{"statements": ["Alice started climbing"]}', model="test",
            usage=TokenUsage(), latency_ms=1, request_id="extract",
        ),
    ])
    gen = SemanticGenerator(orchestrator=mock_orchestrator, embedding=mock_embedding)
    episode = Episode(user_id="u1", title="Climbing", content="C", source_messages=[])
    existing = [SemanticMemory(
        user_id="u1", content="Alice likes hiking", memory_type="preference"
    )]

    memories = await gen.generate("u1", "default", episode, existing)

    assert [memory.content for memory in memories] == ["Alice started climbing"]
    assert mock_orchestrator.execute.await_count == 2
    assert mock_orchestrator.execute.await_args_list[0].args[0].metadata["generator"] == "semantic_predict"
    assert mock_orchestrator.execute.await_args_list[1].args[0].metadata["generator"] == "semantic_extract"


def test_semantic_parser_accepts_common_provider_variants(mock_orchestrator, mock_embedding):
    gen = SemanticGenerator(orchestrator=mock_orchestrator, embedding=mock_embedding)

    assert gen._parse_statements('```json\n{"facts": [{"content": "User likes jazz"}]}\n```') == [
        "User likes jazz"
    ]
    assert gen._parse_statements('[{"statement": "User works at Acme"}]') == [
        "User works at Acme"
    ]
    assert gen._parse_statements('Some text before {"insights": ["User has a dog"]} after') == [
        "User has a dog"
    ]
    assert gen._parse_statements('{"statements": []}') == []


@pytest.mark.asyncio
async def test_semantic_generator_retries_malformed_json(mock_orchestrator, mock_embedding):
    mock_orchestrator.execute = AsyncMock(side_effect=[
        LLMResponse(
            content='{"statements": ["truncated', model="test",
            usage=TokenUsage(), latency_ms=1, request_id="one",
        ),
        LLMResponse(
            content='{"statements": ["User likes jazz"]}', model="test",
            usage=TokenUsage(), latency_ms=1, request_id="two",
        ),
    ])
    gen = SemanticGenerator(orchestrator=mock_orchestrator, embedding=mock_embedding)
    episode = Episode(user_id="u1", title="Music", content="Jazz", source_messages=[])

    memories = await gen.generate("u1", "default", episode, [])

    assert [memory.content for memory in memories] == ["User likes jazz"]
    assert mock_orchestrator.execute.await_count == 2
    retry_request = mock_orchestrator.execute.await_args_list[1].args[0]
    assert retry_request.temperature == 0.0
    assert "previous response was incomplete" in retry_request.messages[-1]["content"]


@pytest.mark.asyncio
async def test_episode_merger_retries_truncated_json(mock_orchestrator):
    mock_orchestrator.execute = AsyncMock(side_effect=[
        LLMResponse(
            content='{"decision":"new","reason":"truncated', model="test",
            usage=TokenUsage(), latency_ms=1, request_id="one",
        ),
        LLMResponse(
            content='{"decision":"new","merge_target_id":null,"reason":"distinct"}',
            model="test", usage=TokenUsage(), latency_ms=1, request_id="two",
        ),
    ])
    merger = EpisodeMerger.__new__(EpisodeMerger)
    merger._orchestrator = mock_orchestrator
    request = LLMRequest(
        messages=({"role": "user", "content": "decide"},),
        response_format={"type": "json_object"},
    )

    parsed = await merger._request_json(request, "merge decision")

    assert parsed["decision"] == "new"
    assert mock_orchestrator.execute.await_count == 2
    retry_request = mock_orchestrator.execute.await_args_list[1].args[0]
    assert retry_request.temperature == 0.0
    assert "previous response was incomplete" in retry_request.messages[-1]["content"]


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
    episode = await gen.generate("u1", "default", messages, "topic_change")
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
    episode = await gen.generate("u1", "default", messages, "test")
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


@pytest.mark.asyncio
async def test_segmenter_uses_paper_observation_window_by_default(mock_orchestrator):
    async def respond(request):
        prompt = request.messages[0]["content"]
        count = 1 if "You will receive 1 messages" in prompt else 20
        return LLMResponse(
            content=json.dumps({
                "episodes": [{"indices": list(range(1, count + 1)), "topic": "window"}]
            }),
            model="gpt-4o-mini", usage=TokenUsage(), latency_ms=100,
            request_id="abc",
        )

    mock_orchestrator.execute = AsyncMock(side_effect=respond)
    seg = BatchSegmenter(orchestrator=mock_orchestrator)

    await seg.segment([Message(role="user", content=f"msg {i}") for i in range(41)])

    assert mock_orchestrator.execute.await_count == 3
    prompts = [call.args[0].messages[0]["content"] for call in mock_orchestrator.execute.await_args_list]
    assert "20 messages" in prompts[0]
    assert "20 messages" in prompts[1]
    assert "1 messages" in prompts[2]


@pytest.mark.asyncio
async def test_segmenter_propagates_window_timeout_and_retries(mock_orchestrator):
    mock_orchestrator.execute = AsyncMock(return_value=LLMResponse(
        content='{"episodes": [{"indices": [1, 2], "topic": "window"}]}',
        model="gpt-4o-mini", usage=TokenUsage(), latency_ms=100, request_id="abc",
    ))
    seg = BatchSegmenter(
        orchestrator=mock_orchestrator,
        window_size=2,
        timeout=123.0,
        retries=5,
    )

    await seg.segment([Message(role="user", content=f"msg {i}") for i in range(4)])

    assert mock_orchestrator.execute.await_count == 2
    for call in mock_orchestrator.execute.await_args_list:
        request = call.args[0]
        assert request.timeout == 123.0
        assert request.retries == 5


@pytest.mark.asyncio
async def test_segmenter_logs_structured_output_failure(mock_orchestrator, caplog):
    mock_orchestrator.execute = AsyncMock(return_value=LLMResponse(
        content='{"unexpected": []}', model="gpt-4o-mini",
        usage=TokenUsage(), latency_ms=100, request_id="abc",
    ))
    seg = BatchSegmenter(orchestrator=mock_orchestrator)
    messages = [Message(role="user", content="hello")]

    with caplog.at_level("WARNING", logger="nemori"):
        groups = await seg.segment(messages)

    assert groups == [{"messages": messages, "topic": "conversation"}]
    assert "ValueError: structured output is missing an episodes list" in caplog.text


@pytest.mark.asyncio
async def test_segmenter_repairs_incomplete_partition_without_collapsing_boundaries(
    mock_orchestrator, caplog
):
    mock_orchestrator.execute = AsyncMock(return_value=LLMResponse(
        content=(
            '{"episodes": ['
            '{"indices": [1], "topic": "first topic"},'
            '{"indices": [3], "topic": "second topic"}'
            ']}'
        ),
        model="gpt-4o-mini", usage=TokenUsage(), latency_ms=100, request_id="abc",
    ))
    seg = BatchSegmenter(orchestrator=mock_orchestrator)
    messages = [
        Message(role="user", content="first"),
        Message(role="assistant", content="second"),
        Message(role="user", content="third"),
    ]

    with caplog.at_level("WARNING", logger="nemori"):
        groups = await seg.segment(messages)

    assert len(groups) == 2
    assert groups[0] == {"messages": messages[:2], "topic": "first topic"}
    assert groups[1] == {"messages": messages[2:], "topic": "second topic"}
    assert mock_orchestrator.execute.await_count == 2
    assert "Segmentation partition repaired" in caplog.text


@pytest.mark.asyncio
async def test_segmenter_retries_partition_constraint_before_repair(mock_orchestrator):
    mock_orchestrator.execute = AsyncMock(side_effect=[
        LLMResponse(
            content='{"episodes": [{"indices": [1], "topic": "partial"}]}',
            model="test", usage=TokenUsage(), latency_ms=1, request_id="one",
        ),
        LLMResponse(
            content=(
                '{"episodes": ['
                '{"indices": [1], "topic": "first"},'
                '{"indices": [2], "topic": "second"}'
                ']}'
            ),
            model="test", usage=TokenUsage(), latency_ms=1, request_id="two",
        ),
    ])
    seg = BatchSegmenter(orchestrator=mock_orchestrator)
    messages = [
        Message(role="user", content="first"),
        Message(role="assistant", content="second"),
    ]

    groups = await seg.segment(messages)

    assert [group["topic"] for group in groups] == ["first", "second"]
    assert mock_orchestrator.execute.await_count == 2
    retry_request = mock_orchestrator.execute.await_args_list[1].args[0]
    assert retry_request.temperature == 0.0
    assert "Missing indices: [2]" in retry_request.messages[-1]["content"]
