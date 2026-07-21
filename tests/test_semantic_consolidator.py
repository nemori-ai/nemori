"""Tests for the paper-aligned native semantic consolidator."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from nemori.domain.models import SemanticMemory
from nemori.llm.generators.consolidator import SemanticConsolidator
from nemori.llm.orchestrator import LLMOrchestrator, LLMResponse, TokenUsage


@pytest.fixture
def mock_orchestrator() -> AsyncMock:
    return AsyncMock(spec=LLMOrchestrator)


def memory(
    content: str,
    *,
    memory_id: str,
    memory_type: str = "preference",
    user_id: str = "u1",
    agent_id: str = "a1",
) -> SemanticMemory:
    return SemanticMemory(
        id=memory_id,
        user_id=user_id,
        agent_id=agent_id,
        content=content,
        memory_type=memory_type,
    )


def response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        model="test-model",
        usage=TokenUsage(),
        latency_ms=1,
        request_id="test",
    )


@pytest.mark.asyncio
async def test_no_candidates_defaults_to_new_without_llm(mock_orchestrator: AsyncMock) -> None:
    consolidator = SemanticConsolidator(mock_orchestrator)

    result = await consolidator.decide(memory("User likes coffee", memory_id="new"), [])

    assert result.decision == "NEW"
    assert result.target_ids == ()
    assert result.new_content is None
    mock_orchestrator.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_valid_new_decision(mock_orchestrator: AsyncMock) -> None:
    mock_orchestrator.execute.return_value = response(
        '{"decision":"NEW","reason":"These are different preferences."}'
    )
    consolidator = SemanticConsolidator(mock_orchestrator)

    result = await consolidator.decide(
        memory("User likes tea", memory_id="new"),
        [memory("User likes coffee", memory_id="old")],
    )

    assert result.decision == "NEW"
    request = mock_orchestrator.execute.await_args.args[0]
    assert request.response_format == {"type": "json_object"}
    assert request.metadata["generator"] == "semantic_consolidation"
    assert '"id": "old"' in request.messages[0]["content"]


@pytest.mark.asyncio
async def test_valid_merge_decision_from_fenced_json(mock_orchestrator: AsyncMock) -> None:
    mock_orchestrator.execute.return_value = response(
        "```json\n"
        '{"decision":"MERGE","target_ids":["old"],'
        '"new_content":"The user enjoys coffee.","reason":"Exact rephrasing."}'
        "\n```"
    )
    consolidator = SemanticConsolidator(mock_orchestrator)

    result = await consolidator.decide(
        memory("User likes coffee", memory_id="new"),
        [memory("The user enjoys coffee", memory_id="old")],
    )

    assert result.decision == "MERGE"
    assert result.target_ids == ("old",)
    assert result.new_content == "The user enjoys coffee."


@pytest.mark.asyncio
async def test_valid_conflict_delete_uses_incoming_content(mock_orchestrator: AsyncMock) -> None:
    mock_orchestrator.execute.return_value = response(
        '{"decision":"CONFLICT_DELETE","target_ids":["old"],'
        '"reason":"The same residence changed."}'
    )
    consolidator = SemanticConsolidator(mock_orchestrator)

    result = await consolidator.decide(
        memory("User lives in Shanghai", memory_id="new", memory_type="identity"),
        [memory("User lives in Beijing", memory_id="old", memory_type="identity")],
    )

    assert result.decision == "CONFLICT_DELETE"
    assert result.target_ids == ("old",)
    assert result.new_content is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_payload",
    [
        '{"decision":"merge","target_ids":["old"],"new_content":"x","reason":"bad case"}',
        '{"decision":"MERGE","target_ids":["not-a-candidate"],"new_content":"x","reason":"bad id"}',
        '{"decision":"MERGE","target_ids":["old"],"reason":"missing content"}',
        '{"decision":"CONFLICT_DELETE","target_ids":["old"],"new_content":"invented","reason":"bad replacement"}',
        '{"decision":"NEW","target_ids":["old"],"reason":"unsafe target"}',
        '[{"decision":"NEW","reason":"wrong root"}]',
        'not json',
    ],
)
async def test_invalid_output_safely_defaults_to_new(
    mock_orchestrator: AsyncMock,
    invalid_payload: str,
) -> None:
    mock_orchestrator.execute.return_value = response(invalid_payload)
    consolidator = SemanticConsolidator(mock_orchestrator)

    result = await consolidator.decide(
        memory("User likes coffee", memory_id="new"),
        [memory("User enjoys coffee", memory_id="old")],
    )

    assert result.decision == "NEW"
    assert result.target_ids == ()
    assert result.new_content is None


@pytest.mark.asyncio
async def test_merge_content_over_100_words_defaults_to_new(mock_orchestrator: AsyncMock) -> None:
    long_content = " ".join(["word"] * 101)
    mock_orchestrator.execute.return_value = response(
        '{"decision":"MERGE","target_ids":["old"],'
        f'"new_content":"{long_content}","reason":"Too verbose."}}'
    )
    consolidator = SemanticConsolidator(mock_orchestrator)

    result = await consolidator.decide(
        memory("User likes coffee", memory_id="new"),
        [memory("User enjoys coffee", memory_id="old")],
    )

    assert result.decision == "NEW"


@pytest.mark.asyncio
async def test_llm_failure_safely_defaults_to_new(mock_orchestrator: AsyncMock) -> None:
    mock_orchestrator.execute.side_effect = RuntimeError("provider unavailable")
    consolidator = SemanticConsolidator(mock_orchestrator)

    result = await consolidator.decide(
        memory("User likes coffee", memory_id="new"),
        [memory("User enjoys coffee", memory_id="old")],
    )

    assert result.decision == "NEW"


@pytest.mark.asyncio
async def test_cross_scope_candidates_are_not_sent_to_llm(mock_orchestrator: AsyncMock) -> None:
    consolidator = SemanticConsolidator(mock_orchestrator)

    result = await consolidator.decide(
        memory("User likes coffee", memory_id="new"),
        [memory("Other user likes coffee", memory_id="old", user_id="u2")],
    )

    assert result.decision == "NEW"
    mock_orchestrator.execute.assert_not_awaited()
