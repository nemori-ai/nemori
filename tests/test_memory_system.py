"""Tests for async MemorySystem."""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from nemori.core.memory_system import MemorySystem
from nemori.domain.models import Message, Episode, SemanticMemory
from nemori.search.unified import SearchResult
from nemori.config import MemoryConfig
from nemori.llm.generators.consolidator import ConsolidationDecision


@pytest.fixture
def deps():
    qdrant = MagicMock()
    qdrant.upsert_episode = MagicMock()
    qdrant.upsert_semantic = MagicMock()
    qdrant.delete_episode = MagicMock()
    qdrant.delete_semantic = MagicMock()
    qdrant.delete_episodes_by_user = MagicMock()
    qdrant.delete_semantic_by_user = MagicMock()
    return {
        "config": MemoryConfig(),
        "agent_id": "default",
        "db": AsyncMock(),
        "episode_store": AsyncMock(),
        "semantic_store": AsyncMock(),
        "buffer_store": AsyncMock(),
        "orchestrator": AsyncMock(),
        "embedding": AsyncMock(),
        "episode_generator": AsyncMock(),
        "semantic_generator": AsyncMock(),
        "event_bus": AsyncMock(),
        "search": AsyncMock(),
        "qdrant": qdrant,
    }


@pytest.fixture
def system(deps):
    deps["buffer_store"].count_unprocessed = AsyncMock(return_value=0)
    deps["buffer_store"].get_unprocessed = AsyncMock(return_value=[])
    deps["search"].search = AsyncMock(return_value=SearchResult())
    return MemorySystem(**deps)


@pytest.mark.asyncio
async def test_add_messages_pushes_to_buffer(system, deps):
    msgs = [Message(role="user", content="hi")]
    await system.add_messages("u1", msgs)
    deps["buffer_store"].push.assert_called_once_with("u1", "default", msgs)


@pytest.mark.asyncio
async def test_add_messages_waits_for_observation_window(system, deps):
    system._config.observation_window = 20
    deps["buffer_store"].count_unprocessed.return_value = 19
    system._process = AsyncMock(return_value=[])

    await system.add_messages("u1", [Message(role="user", content="hi")])
    await asyncio.sleep(0)

    system._process.assert_not_awaited()


@pytest.mark.asyncio
async def test_add_messages_triggers_at_observation_window(system, deps):
    system._config.observation_window = 20
    deps["buffer_store"].count_unprocessed.side_effect = [20, 0, 0]
    system._process = AsyncMock(return_value=[])

    await system.add_messages("u1", [Message(role="user", content="hi")])
    await asyncio.sleep(0)

    system._process.assert_awaited_once_with("u1")


@pytest.mark.asyncio
async def test_add_messages_uses_legacy_minimum_without_batch_segmentation(system, deps):
    system._config.enable_batch_segmentation = False
    system._config.buffer_size_min = 2
    deps["buffer_store"].count_unprocessed.side_effect = [2, 0, 0]
    system._process = AsyncMock(return_value=[])

    await system.add_messages("u1", [Message(role="user", content="hi")])
    await asyncio.sleep(0)

    system._process.assert_awaited_once_with("u1")


@pytest.mark.asyncio
async def test_observation_worker_deduplicates_same_user(system, deps):
    system._config.observation_window = 2
    release = asyncio.Event()
    started = asyncio.Event()
    counts = iter([2, 3, 0, 0])

    async def count_unprocessed(user_id, agent_id):
        return next(counts)

    async def process(user_id):
        started.set()
        await release.wait()
        return []

    deps["buffer_store"].count_unprocessed = AsyncMock(side_effect=count_unprocessed)
    system._process = AsyncMock(side_effect=process)

    await system.add_messages("u1", [Message(role="user", content="first")])
    await started.wait()
    await system.add_messages("u1", [Message(role="user", content="second")])

    assert system._process.await_count == 1
    assert len(system._observation_tasks) == 1

    release.set()
    await system.drain(timeout=1.0)
    assert system._process.await_count == 1


@pytest.mark.asyncio
async def test_observation_worker_continues_when_another_window_is_ready(system, deps):
    system._config.observation_window = 2
    counts = iter([2, 2, 0, 0])
    deps["buffer_store"].count_unprocessed = AsyncMock(
        side_effect=lambda user_id, agent_id: next(counts)
    )
    system._process = AsyncMock(return_value=[])

    await system.add_messages("u1", [Message(role="user", content="first")])
    await system.drain(timeout=1.0)

    assert system._process.await_count == 2


@pytest.mark.asyncio
async def test_observation_workers_for_different_users_run_concurrently(system, deps):
    system._config.observation_window = 2
    release = asyncio.Event()
    both_started = asyncio.Event()
    active_users = set()
    count_calls = {"u1": 0, "u2": 0}

    async def count_unprocessed(user_id, agent_id):
        count_calls[user_id] += 1
        return 2 if count_calls[user_id] == 1 else 0

    async def process(user_id):
        active_users.add(user_id)
        if len(active_users) == 2:
            both_started.set()
        await release.wait()
        return []

    deps["buffer_store"].count_unprocessed = AsyncMock(side_effect=count_unprocessed)
    system._process = AsyncMock(side_effect=process)

    await system.add_messages("u1", [Message(role="user", content="one")])
    await system.add_messages("u2", [Message(role="user", content="two")])
    await asyncio.wait_for(both_started.wait(), timeout=1.0)

    assert active_users == {"u1", "u2"}
    assert len(system._observation_tasks) == 2

    release.set()
    await system.drain(timeout=1.0)


@pytest.mark.asyncio
async def test_flush_processes_buffer(system, deps):
    deps["buffer_store"].get_unprocessed.return_value = [
        Message(role="user", content="hello", metadata={"buffer_id": 1}),
        Message(role="assistant", content="hi", metadata={"buffer_id": 2}),
    ]
    ep = Episode(user_id="u1", title="T", content="C", source_messages=[], embedding=[0.1] * 10)
    deps["episode_generator"].generate = AsyncMock(return_value=ep)

    result = await system.flush("u1")
    assert len(result) >= 1
    deps["episode_store"].save.assert_called()
    # Qdrant upsert should be called for episode with embedding
    deps["qdrant"].upsert_episode.assert_called()


@pytest.mark.asyncio
async def test_batch_segmentation_uses_runtime_config(system, deps):
    messages = [
        Message(role="user", content=f"message {i}", metadata={"buffer_id": i})
        for i in range(1, 8)
    ]
    deps["buffer_store"].get_unprocessed.return_value = messages
    system._config.observation_window = 7
    system._config.llm_timeout = 123.0
    system._config.llm_retries = 4
    system._enable_semantic = False

    episode = Episode(user_id="u1", title="T", content="C", source_messages=[])
    deps["episode_generator"].generate = AsyncMock(return_value=episode)

    with patch("nemori.core.memory_system.BatchSegmenter") as segmenter_cls:
        segmenter_cls.return_value.segment = AsyncMock(return_value=[{
            "messages": messages,
            "topic": "conversation",
        }])
        await system.flush("u1")

    segmenter_cls.assert_called_once_with(
        orchestrator=deps["orchestrator"],
        window_size=7,
        timeout=123.0,
        retries=4,
    )


@pytest.mark.asyncio
async def test_semantic_context_uses_paper_ks_and_threshold(system, deps):
    system._config.semantic_context_top_k = 10
    system._config.semantic_context_similarity_threshold = 0.70
    episode = Episode(
        user_id="u1", title="T", content="C", source_messages=[], embedding=[0.1]
    )
    prior = SemanticMemory(
        id="prior-high",
        user_id="u1",
        agent_id="default",
        content="Known fact",
        memory_type="identity",
    )
    deps["qdrant"].search_semantic.return_value = [
        {"id": "prior-high", "score": 0.71},
        {"id": "prior-equal", "score": 0.70},
    ]
    deps["semantic_store"].get_batch = AsyncMock(return_value=[prior])
    deps["semantic_generator"].generate = AsyncMock(return_value=[])

    await system._on_episode_created("u1", episode)

    deps["qdrant"].search_semantic.assert_called_once_with(
        "u1", "default", [0.1], top_k=10
    )
    deps["semantic_store"].get_batch.assert_awaited_once_with(
        ["prior-high"], "u1", "default"
    )
    deps["semantic_generator"].generate.assert_awaited_once_with(
        "u1", "default", episode, [prior]
    )


@pytest.mark.asyncio
async def test_semantic_native_merge_replaces_candidates_safely(system, deps):
    episode = Episode(
        user_id="u1", title="T", content="C", source_messages=[], embedding=[0.1]
    )
    incoming = SemanticMemory(
        id="incoming",
        user_id="u1",
        agent_id="default",
        content="User enjoys coffee",
        memory_type="preference",
        embedding=[0.2],
        source_episode_id=episode.id,
    )
    existing = SemanticMemory(
        id="existing",
        user_id="u1",
        agent_id="default",
        content="User likes coffee",
        memory_type="preference",
    )
    deps["qdrant"].search_semantic.side_effect = [
        [],
        [{"id": "existing", "score": 0.95}],
    ]
    deps["semantic_store"].get_batch = AsyncMock(return_value=[existing])
    deps["semantic_generator"].generate = AsyncMock(return_value=[incoming])
    system._semantic_consolidator.decide = AsyncMock(
        return_value=ConsolidationDecision(
            decision="MERGE",
            target_ids=("existing",),
            new_content="User likes coffee",
            reason="Same persistent preference",
        )
    )
    deps["embedding"].embed = AsyncMock(return_value=[0.9])

    await system._on_episode_created("u1", episode)

    system._semantic_consolidator.decide.assert_awaited_once_with(
        incoming, [existing]
    )
    deps["semantic_store"].save.assert_awaited_once_with(incoming)
    assert incoming.content == "User likes coffee"
    assert incoming.embedding == [0.9]
    assert incoming.metadata["consolidation"]["decision"] == "MERGE"
    deps["semantic_store"].delete.assert_awaited_once_with(
        "existing", "u1", "default"
    )
    deps["qdrant"].upsert_semantic.assert_called_once_with(
        "incoming", "u1", "default", [0.9]
    )
    deps["qdrant"].delete_semantic.assert_called_once_with("existing")


@pytest.mark.asyncio
async def test_search_delegates(system, deps):
    await system.search("u1", "hiking")
    deps["search"].search.assert_called_once()


@pytest.mark.asyncio
async def test_delete_episode(system, deps):
    await system.delete_episode("u1", "ep-1")
    deps["episode_store"].delete.assert_called_once_with("ep-1", "u1", "default")
    deps["qdrant"].delete_episode.assert_called_once_with("ep-1")


@pytest.mark.asyncio
async def test_delete_user(system, deps):
    await system.delete_user("u1")
    deps["episode_store"].delete_by_user.assert_called_once_with("u1", "default")
    deps["semantic_store"].delete_by_user.assert_called_once_with("u1", "default")
    deps["qdrant"].delete_episodes_by_user.assert_called_once_with("u1", "default")
    deps["qdrant"].delete_semantic_by_user.assert_called_once_with("u1", "default")


@pytest.mark.asyncio
async def test_drain(system):
    await system.drain(timeout=1.0)
