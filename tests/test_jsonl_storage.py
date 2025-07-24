"""
Tests for JSONL storage implementation.
"""

import shutil
import tempfile
from datetime import UTC, datetime

import pytest

from nemori.core.data_types import DataType, RawEventData, TemporalInfo
from nemori.core.episode import Episode, EpisodeLevel, EpisodeMetadata, EpisodeType
from nemori.storage.jsonl_storage import JSONLEpisodicMemoryRepository, JSONLRawDataRepository
from nemori.storage.storage_types import EpisodeQuery, RawDataQuery, StorageConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def jsonl_config(temp_dir):
    """Create a JSONL storage configuration."""
    return StorageConfig(
        backend_type="jsonl",
        connection_string=temp_dir,
        batch_size=10
    )


@pytest.fixture
def sample_raw_data():
    """Create sample raw event data for testing."""
    return RawEventData(
        data_id="test_data_1",
        data_type=DataType.CONVERSATION,
        content=[
            {"speaker": "alice", "content": "Hello, how are you?", "timestamp": "2024-01-01T10:00:00"},
            {"speaker": "bob", "content": "I'm doing well, thanks!", "timestamp": "2024-01-01T10:01:00"}
        ],
        source="test_source",
        temporal_info=TemporalInfo(
            timestamp=datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC),
            duration=60.0,
            timezone="UTC"
        ),
        metadata={"participants": ["alice", "bob"], "topic": "greeting"},
        processed=False,
        processing_version="1.0"
    )


@pytest.fixture
def sample_episode():
    """Create sample episode for testing."""
    return Episode(
        episode_id="test_episode_1",
        owner_id="alice_test",
        episode_type=EpisodeType.CONVERSATIONAL,
        level=EpisodeLevel.ATOMIC,
        title="Greeting Conversation",
        content="Alice and Bob exchanged greetings",
        summary="A simple greeting exchange",
        temporal_info=TemporalInfo(
            timestamp=datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC),
            duration=60.0,
            timezone="UTC"
        ),
        metadata=EpisodeMetadata(
            source_data_ids=["test_data_1"],
            source_types={DataType.CONVERSATION},
            entities=["alice", "bob"],
            topics=["greeting"],
            key_points=["hello", "how are you"]
        ),
        search_keywords=["greeting", "hello", "conversation"],
        importance_score=0.7
    )


class TestJSONLRawDataRepository:
    """Test JSONL raw data repository."""

    @pytest.mark.asyncio
    async def test_initialization(self, jsonl_config):
        """Test repository initialization."""
        repo = JSONLRawDataRepository(jsonl_config)
        await repo.initialize()

        assert repo._initialized
        assert repo.data_dir.exists()
        assert repo.raw_data_file.exists()

        await repo.close()

    @pytest.mark.asyncio
    async def test_health_check(self, jsonl_config):
        """Test health check functionality."""
        repo = JSONLRawDataRepository(jsonl_config)

        # Before initialization
        assert not await repo.health_check()

        # After initialization
        await repo.initialize()
        assert await repo.health_check()

        await repo.close()

    @pytest.mark.asyncio
    async def test_store_and_get_raw_data(self, jsonl_config, sample_raw_data):
        """Test storing and retrieving raw data."""
        repo = JSONLRawDataRepository(jsonl_config)
        await repo.initialize()

        # Store data
        stored_id = await repo.store_raw_data(sample_raw_data)
        assert stored_id == sample_raw_data.data_id

        # Retrieve data
        retrieved_data = await repo.get_raw_data(sample_raw_data.data_id)
        assert retrieved_data is not None
        assert retrieved_data.data_id == sample_raw_data.data_id
        assert retrieved_data.content == sample_raw_data.content
        assert retrieved_data.data_type == sample_raw_data.data_type

        await repo.close()

    @pytest.mark.asyncio
    async def test_store_raw_data_batch(self, jsonl_config):
        """Test batch storage of raw data."""
        repo = JSONLRawDataRepository(jsonl_config)
        await repo.initialize()

        # Create multiple data items
        data_list = []
        for i in range(3):
            data = RawEventData(
                data_id=f"test_data_{i}",
                data_type=DataType.CONVERSATION,
                content=f"test content {i}",
                source="test_source",
                temporal_info=TemporalInfo(
                    timestamp=datetime(2024, 1, 1, 10, i, 0, tzinfo=UTC),
                    duration=60.0,
                    timezone="UTC"
                ),
                metadata={"index": i}
            )
            data_list.append(data)

        # Store batch
        stored_ids = await repo.store_raw_data_batch(data_list)
        assert len(stored_ids) == 3

        # Retrieve batch
        retrieved_data = await repo.get_raw_data_batch(stored_ids)
        assert len(retrieved_data) == 3
        assert all(data is not None for data in retrieved_data)

        await repo.close()

    @pytest.mark.asyncio
    async def test_mark_as_processed(self, jsonl_config, sample_raw_data):
        """Test marking raw data as processed."""
        repo = JSONLRawDataRepository(jsonl_config)
        await repo.initialize()

        # Store unprocessed data
        await repo.store_raw_data(sample_raw_data)
        assert not sample_raw_data.processed

        # Mark as processed
        success = await repo.mark_as_processed(sample_raw_data.data_id, "2.0")
        assert success

        # Verify it's marked as processed
        retrieved_data = await repo.get_raw_data(sample_raw_data.data_id)
        assert retrieved_data.processed
        assert retrieved_data.processing_version == "2.0"

        await repo.close()

    @pytest.mark.asyncio
    async def test_get_unprocessed_data(self, jsonl_config):
        """Test getting unprocessed data."""
        repo = JSONLRawDataRepository(jsonl_config)
        await repo.initialize()

        # Store some processed and unprocessed data
        for i in range(5):
            data = RawEventData(
                data_id=f"test_data_{i}",
                data_type=DataType.CONVERSATION,
                content=f"test content {i}",
                source="test_source",
                temporal_info=TemporalInfo(
                    timestamp=datetime(2024, 1, 1, 10, i, 0, tzinfo=UTC),
                    duration=60.0,
                    timezone="UTC"
                ),
                metadata={"index": i},
                processed=(i % 2 == 0)  # Even indices are processed
            )
            await repo.store_raw_data(data)

        # Get unprocessed data
        unprocessed = await repo.get_unprocessed_data()
        assert len(unprocessed) == 2  # indices 1, 3 are unprocessed

        # Test with limit
        unprocessed_limited = await repo.get_unprocessed_data(limit=1)
        assert len(unprocessed_limited) == 1

        await repo.close()

    @pytest.mark.asyncio
    async def test_search_raw_data(self, jsonl_config):
        """Test searching raw data."""
        repo = JSONLRawDataRepository(jsonl_config)
        await repo.initialize()

        # Store test data
        for i in range(3):
            data = RawEventData(
                data_id=f"search_test_{i}",
                data_type=DataType.CONVERSATION,
                content=f"This is test content number {i}",
                source=f"source_{i}",
                temporal_info=TemporalInfo(
                    timestamp=datetime(2024, 1, 1, 10, i, 0, tzinfo=UTC),
                    duration=60.0,
                    timezone="UTC"
                ),
                metadata={"category": "test" if i < 2 else "other"}
            )
            await repo.store_raw_data(data)

        # Search by content
        query = RawDataQuery(content_contains="test content")
        results = await repo.search_raw_data(query)
        assert results.total_count == 3
        assert len(results.data) == 3

        # Search by source
        query = RawDataQuery(sources=["source_0", "source_1"])
        results = await repo.search_raw_data(query)
        assert results.total_count == 2

        # Search by metadata
        query = RawDataQuery(metadata_filters={"category": "test"})
        results = await repo.search_raw_data(query)
        assert results.total_count == 2

        await repo.close()

    @pytest.mark.asyncio
    async def test_get_stats(self, jsonl_config):
        """Test getting storage statistics."""
        repo = JSONLRawDataRepository(jsonl_config)
        await repo.initialize()

        # Store some test data
        for i in range(3):
            data_type = DataType.CONVERSATION if i < 2 else DataType.DOCUMENT
            data = RawEventData(
                data_id=f"stats_test_{i}",
                data_type=data_type,
                content=f"test content {i}",
                source="test_source",
                temporal_info=TemporalInfo(
                    timestamp=datetime(2024, 1, 1, 10, i, 0, tzinfo=UTC),
                    duration=60.0,
                    timezone="UTC"
                ),
                metadata={"index": i},
                processed=(i == 0)  # Only first one is processed
            )
            await repo.store_raw_data(data)

        # Get statistics
        stats = await repo.get_stats()
        assert stats.total_raw_data == 3
        assert stats.processed_raw_data == 1
        assert stats.raw_data_by_type[DataType.CONVERSATION] == 2
        assert stats.raw_data_by_type[DataType.DOCUMENT] == 1

        await repo.close()


class TestJSONLEpisodicMemoryRepository:
    """Test JSONL episodic memory repository."""

    @pytest.mark.asyncio
    async def test_initialization(self, jsonl_config):
        """Test repository initialization."""
        repo = JSONLEpisodicMemoryRepository(jsonl_config)
        await repo.initialize()

        assert repo._initialized
        assert repo.data_dir.exists()
        assert repo.episodes_file.exists()
        assert repo.links_file.exists()

        await repo.close()

    @pytest.mark.asyncio
    async def test_store_and_get_episode(self, jsonl_config, sample_episode):
        """Test storing and retrieving episodes."""
        repo = JSONLEpisodicMemoryRepository(jsonl_config)
        await repo.initialize()

        # Store episode
        stored_id = await repo.store_episode(sample_episode)
        assert stored_id == sample_episode.episode_id

        # Retrieve episode
        retrieved_episode = await repo.get_episode(sample_episode.episode_id)
        assert retrieved_episode is not None
        assert retrieved_episode.episode_id == sample_episode.episode_id
        assert retrieved_episode.title == sample_episode.title
        assert retrieved_episode.content == sample_episode.content

        await repo.close()

    @pytest.mark.asyncio
    async def test_get_episodes_by_owner(self, jsonl_config):
        """Test getting episodes by owner."""
        repo = JSONLEpisodicMemoryRepository(jsonl_config)
        await repo.initialize()

        # Store episodes for different owners
        for i in range(4):
            owner_id = "alice" if i < 2 else "bob"
            episode = Episode(
                episode_id=f"episode_{i}",
                owner_id=f"{owner_id}_test",
                episode_type=EpisodeType.CONVERSATIONAL,
                level=EpisodeLevel.ATOMIC,
                title=f"Episode {i}",
                content=f"Content for episode {i}",
                summary=f"Summary {i}",
                temporal_info=TemporalInfo(
                    timestamp=datetime(2024, 1, 1, 10, i, 0, tzinfo=UTC),
                    duration=60.0,
                    timezone="UTC"
                ),
                metadata=EpisodeMetadata(
                    source_data_ids=[f"data_{i}"],
                    source_types={DataType.CONVERSATION},
                    entities=[owner_id],
                    topics=[f"topic_{i}"],
                    key_points=[f"point_{i}"]
                ),
                search_keywords=[f"keyword_{i}"],
                importance_score=0.5 + i * 0.1
            )
            await repo.store_episode(episode)

        # Get Alice's episodes
        alice_results = await repo.get_episodes_by_owner("alice_test")
        assert alice_results.total_count == 2
        assert len(alice_results.episodes) == 2

        # Get Bob's episodes
        bob_results = await repo.get_episodes_by_owner("bob_test")
        assert bob_results.total_count == 2

        await repo.close()

    @pytest.mark.asyncio
    async def test_search_episodes(self, jsonl_config):
        """Test searching episodes."""
        repo = JSONLEpisodicMemoryRepository(jsonl_config)
        await repo.initialize()

        # Store test episodes
        episodes = []
        for i in range(3):
            episode = Episode(
                episode_id=f"search_episode_{i}",
                owner_id="test_owner",
                episode_type=EpisodeType.CONVERSATIONAL,
                level=EpisodeLevel.ATOMIC,
                title=f"Test Episode {i}",
                content=f"This is test content for episode {i}",
                summary=f"Test summary {i}",
                temporal_info=TemporalInfo(
                    timestamp=datetime(2024, 1, 1, 10, i, 0, tzinfo=UTC),
                    duration=60.0,
                    timezone="UTC"
                ),
                metadata=EpisodeMetadata(
                    source_data_ids=[f"data_{i}"],
                    source_types={DataType.CONVERSATION},
                    entities=["alice", "bob"],
                    topics=["conversation", f"topic_{i}"],
                    key_points=[f"point_{i}"]
                ),
                search_keywords=["test", "episode", f"keyword_{i}"],
                importance_score=0.5 + i * 0.2
            )
            episodes.append(episode)
            await repo.store_episode(episode)

        # Search by text
        query = EpisodeQuery(text_search="test content")
        results = await repo.search_episodes(query)
        assert results.total_count == 3

        # Search by keywords
        query = EpisodeQuery(keywords=["test"])
        results = await repo.search_episodes(query)
        assert results.total_count == 3

        # Search by owner
        query = EpisodeQuery(owner_ids=["test_owner"])
        results = await repo.search_episodes(query)
        assert results.total_count == 3

        # Search with limit
        query = EpisodeQuery(owner_ids=["test_owner"], limit=2)
        results = await repo.search_episodes(query)
        assert len(results.episodes) == 2
        assert results.has_more

        await repo.close()

    @pytest.mark.asyncio
    async def test_mark_episode_accessed(self, jsonl_config, sample_episode):
        """Test marking episode as accessed."""
        repo = JSONLEpisodicMemoryRepository(jsonl_config)
        await repo.initialize()

        # Store episode
        await repo.store_episode(sample_episode)
        initial_recall_count = sample_episode.recall_count

        # Mark as accessed
        success = await repo.mark_episode_accessed(sample_episode.episode_id)
        assert success

        # Check recall count increased
        retrieved_episode = await repo.get_episode(sample_episode.episode_id)
        assert retrieved_episode.recall_count == initial_recall_count + 1
        assert retrieved_episode.last_accessed is not None

        await repo.close()

    @pytest.mark.asyncio
    async def test_link_episode_to_raw_data(self, jsonl_config, sample_episode):
        """Test linking episodes to raw data."""
        repo = JSONLEpisodicMemoryRepository(jsonl_config)
        await repo.initialize()

        # Store episode
        await repo.store_episode(sample_episode)

        # Link to raw data
        raw_data_ids = ["raw_1", "raw_2", "raw_3"]
        success = await repo.link_episode_to_raw_data(sample_episode.episode_id, raw_data_ids)
        assert success

        # Get raw data for episode
        linked_raw_data = await repo.get_raw_data_for_episode(sample_episode.episode_id)
        assert linked_raw_data == raw_data_ids

        # Get episodes for raw data
        episodes = await repo.get_episodes_for_raw_data("raw_1")
        assert len(episodes) == 1
        assert episodes[0].episode_id == sample_episode.episode_id

        await repo.close()

    @pytest.mark.asyncio
    async def test_get_stats(self, jsonl_config):
        """Test getting episodic memory statistics."""
        repo = JSONLEpisodicMemoryRepository(jsonl_config)
        await repo.initialize()

        # Store episodes of different types
        for i in range(4):
            episode_type = EpisodeType.CONVERSATIONAL if i < 3 else EpisodeType.BEHAVIORAL
            level = EpisodeLevel.ATOMIC if i < 2 else EpisodeLevel.COMPOUND

            episode = Episode(
                episode_id=f"stats_episode_{i}",
                owner_id="test_owner",
                episode_type=episode_type,
                level=level,
                title=f"Stats Episode {i}",
                content=f"Content {i}",
                summary=f"Summary {i}",
                temporal_info=TemporalInfo(
                    timestamp=datetime(2024, 1, 1, 10, i, 0, tzinfo=UTC),
                    duration=60.0,
                    timezone="UTC"
                ),
                metadata=EpisodeMetadata(
                    source_data_ids=[f"data_{i}"],
                    source_types={DataType.CONVERSATION},
                    entities=["test"],
                    topics=[f"topic_{i}"],
                    key_points=[f"point_{i}"]
                ),
                search_keywords=[f"keyword_{i}"],
                importance_score=0.5
            )
            await repo.store_episode(episode)

        # Get statistics
        stats = await repo.get_stats()
        assert stats.total_episodes == 4
        assert stats.episodes_by_type[EpisodeType.CONVERSATIONAL] == 3
        assert stats.episodes_by_type[EpisodeType.BEHAVIORAL] == 1
        assert stats.episodes_by_level[EpisodeLevel.ATOMIC] == 2
        assert stats.episodes_by_level[EpisodeLevel.COMPOUND] == 2

        await repo.close()
