"""Tests for PostgreSQL storage implementation."""

import os
from datetime import datetime, timedelta

import pytest
import pytest_asyncio

from nemori.core.data_types import DataType, RawEventData, TemporalInfo
from nemori.core.episode import Episode, EpisodeLevel, EpisodeMetadata, EpisodeType
from nemori.storage import (
    PostgreSQLEpisodicMemoryRepository,
    PostgreSQLRawDataRepository,
    StorageConfig,
    create_postgresql_config,
    create_repositories,
)
from nemori.storage.storage_types import EpisodeQuery, RawDataQuery, TimeRange

# Skip PostgreSQL tests if no connection string is provided
POSTGRESQL_TEST_URL = os.getenv("POSTGRESQL_TEST_URL")
pytestmark = [
    pytest.mark.skipif(
        not POSTGRESQL_TEST_URL, reason="PostgreSQL tests require POSTGRESQL_TEST_URL environment variable"
    ),
]


@pytest_asyncio.fixture
async def postgresql_config():
    """Create PostgreSQL configuration for testing."""
    if not POSTGRESQL_TEST_URL:
        pytest.skip("PostgreSQL tests require POSTGRESQL_TEST_URL environment variable")

    return StorageConfig(
        backend_type="postgresql",
        connection_string=POSTGRESQL_TEST_URL,
        batch_size=100,
        cache_size=1000,
    )


@pytest_asyncio.fixture
async def raw_data_repo(postgresql_config):
    """Create and initialize PostgreSQL raw data repository."""
    repo = PostgreSQLRawDataRepository(postgresql_config)
    await repo.initialize()

    # Clean up any existing test data
    test_ids = ["test_data_1", "batch_test_0", "batch_test_1", "batch_test_2"]
    for test_id in test_ids:
        try:
            await repo.delete_raw_data(test_id)
        except Exception:
            pass  # Ignore if not exists

    yield repo

    # Clean up after test
    for test_id in test_ids:
        try:
            await repo.delete_raw_data(test_id)
        except Exception:
            pass  # Ignore if not exists

    await repo.close()


@pytest_asyncio.fixture
async def episode_repo(postgresql_config):
    """Create and initialize PostgreSQL episode repository."""
    repo = PostgreSQLEpisodicMemoryRepository(postgresql_config)
    await repo.initialize()

    # Clean up any existing test data
    test_episode_ids = ["test_episode_1", "batch_episode_0", "batch_episode_1", "batch_episode_2"]
    for episode_id in test_episode_ids:
        try:
            await repo.delete_episode(episode_id)
        except Exception:
            pass  # Ignore if not exists

    yield repo

    # Clean up after test
    for episode_id in test_episode_ids:
        try:
            await repo.delete_episode(episode_id)
        except Exception:
            pass  # Ignore if not exists

    await repo.close()


@pytest.fixture
def sample_raw_data():
    """Create sample raw event data."""
    return RawEventData(
        data_id="test_data_1",
        data_type=DataType.CONVERSATION,
        content={"text": "Hello world", "user": "Alice"},
        source="test_source",
        temporal_info=TemporalInfo(timestamp=datetime.now(), duration=1.5, timezone="UTC", precision="second"),
        metadata={"importance": "high"},
        processed=False,
        processing_version="1.0",
    )


@pytest.fixture
def sample_episode():
    """Create sample episode."""
    return Episode(
        episode_id="test_episode_1",
        owner_id="test_user",
        episode_type=EpisodeType.CONVERSATIONAL,
        level=EpisodeLevel.ATOMIC,
        title="Test Conversation",
        content="Alice said hello to the world",
        summary="A simple greeting",
        temporal_info=TemporalInfo(timestamp=datetime.now(), duration=1.5, timezone="UTC", precision="second"),
        metadata=EpisodeMetadata(
            source_data_ids=["test_data_1"],
            source_types={DataType.CONVERSATION},
            processing_timestamp=datetime.now(),
            processing_version="1.0",
            entities=["Alice"],
            topics=["greeting"],
            emotions=["neutral"],
            key_points=["Alice greeted the world"],
            time_references=[],
            confidence_score=0.9,
            completeness_score=0.8,
            relevance_score=0.7,
        ),
        structured_data={"greeting_type": "hello"},
        search_keywords=["hello", "world", "Alice"],
        importance_score=0.5,
    )


@pytest.mark.asyncio
class TestPostgreSQLRawDataRepository:
    """Test PostgreSQL raw data repository."""

    async def test_store_and_retrieve_raw_data(self, raw_data_repo, sample_raw_data):
        """Test storing and retrieving raw data."""
        # Store data
        stored_id = await raw_data_repo.store_raw_data(sample_raw_data)
        assert stored_id == sample_raw_data.data_id

        # Retrieve data
        retrieved = await raw_data_repo.get_raw_data(stored_id)
        assert retrieved is not None
        assert retrieved.data_id == sample_raw_data.data_id
        assert retrieved.data_type == sample_raw_data.data_type
        assert retrieved.content == sample_raw_data.content

    async def test_store_batch_raw_data(self, raw_data_repo, sample_raw_data):
        """Test batch storing raw data."""
        # Create multiple data items
        data_list = []
        for i in range(3):
            data = RawEventData(
                data_id=f"batch_test_{i}",
                data_type=DataType.CONVERSATION,
                content={"text": f"Message {i}"},
                source="batch_test",
                temporal_info=sample_raw_data.temporal_info,
                metadata={},
                processed=False,
                processing_version="1.0",
            )
            data_list.append(data)

        # Store batch
        stored_ids = await raw_data_repo.store_raw_data_batch(data_list)
        assert len(stored_ids) == 3

        # Retrieve batch
        retrieved_list = await raw_data_repo.get_raw_data_batch(stored_ids)
        assert len(retrieved_list) == 3
        assert all(item is not None for item in retrieved_list)

    async def test_search_raw_data(self, raw_data_repo, sample_raw_data):
        """Test searching raw data."""
        # Store test data
        await raw_data_repo.store_raw_data(sample_raw_data)

        # Search by data type
        query = RawDataQuery(data_types=[DataType.CONVERSATION])
        result = await raw_data_repo.search_raw_data(query)
        assert result.total_count >= 1
        assert len(result.data) >= 1

        # Search by time range
        now = datetime.now()
        time_range = TimeRange(start=now - timedelta(hours=1), end=now + timedelta(hours=1))
        query = RawDataQuery(time_range=time_range)
        result = await raw_data_repo.search_raw_data(query)
        assert result.total_count >= 1

    async def test_update_raw_data(self, raw_data_repo, sample_raw_data):
        """Test updating raw data."""
        # Store original data
        await raw_data_repo.store_raw_data(sample_raw_data)

        # Update data
        updated_data = sample_raw_data
        updated_data.content = {"text": "Updated hello world", "user": "Alice"}
        updated_data.processed = True

        success = await raw_data_repo.update_raw_data(sample_raw_data.data_id, updated_data)
        assert success

        # Verify update
        retrieved = await raw_data_repo.get_raw_data(sample_raw_data.data_id)
        assert retrieved.content["text"] == "Updated hello world"
        assert retrieved.processed is True

    async def test_mark_as_processed(self, raw_data_repo, sample_raw_data):
        """Test marking raw data as processed."""
        # Store data
        await raw_data_repo.store_raw_data(sample_raw_data)

        # Mark as processed
        success = await raw_data_repo.mark_as_processed(sample_raw_data.data_id, "2.0")
        assert success

        # Verify
        retrieved = await raw_data_repo.get_raw_data(sample_raw_data.data_id)
        assert retrieved.processed is True
        assert retrieved.processing_version == "2.0"

    async def test_delete_raw_data(self, raw_data_repo, sample_raw_data):
        """Test deleting raw data."""
        # Store data
        await raw_data_repo.store_raw_data(sample_raw_data)

        # Delete data
        success = await raw_data_repo.delete_raw_data(sample_raw_data.data_id)
        assert success

        # Verify deletion
        retrieved = await raw_data_repo.get_raw_data(sample_raw_data.data_id)
        assert retrieved is None

    async def test_get_unprocessed_data(self, raw_data_repo, sample_raw_data):
        """Test getting unprocessed data."""
        # Store unprocessed data
        await raw_data_repo.store_raw_data(sample_raw_data)

        # Get unprocessed data
        unprocessed = await raw_data_repo.get_unprocessed_data()
        assert len(unprocessed) >= 1
        assert any(item.data_id == sample_raw_data.data_id for item in unprocessed)

    async def test_health_check(self, raw_data_repo):
        """Test health check."""
        is_healthy = await raw_data_repo.health_check()
        assert is_healthy is True

    async def test_get_stats(self, raw_data_repo, sample_raw_data):
        """Test getting storage statistics."""
        # Store some data
        await raw_data_repo.store_raw_data(sample_raw_data)

        # Get stats
        stats = await raw_data_repo.get_stats()
        assert stats.total_raw_data >= 1
        assert DataType.CONVERSATION in stats.raw_data_by_type
        assert stats.raw_data_by_type[DataType.CONVERSATION] >= 1


@pytest.mark.asyncio
class TestPostgreSQLEpisodicMemoryRepository:
    """Test PostgreSQL episodic memory repository."""

    async def test_store_and_retrieve_episode(self, episode_repo, sample_episode):
        """Test storing and retrieving episodes."""
        # Store episode
        stored_id = await episode_repo.store_episode(sample_episode)
        assert stored_id == sample_episode.episode_id

        # Retrieve episode
        retrieved = await episode_repo.get_episode(stored_id)
        assert retrieved is not None
        assert retrieved.episode_id == sample_episode.episode_id
        assert retrieved.title == sample_episode.title
        assert retrieved.content == sample_episode.content

    async def test_store_batch_episodes(self, episode_repo, sample_episode):
        """Test batch storing episodes."""
        # Create multiple episodes
        episodes = []
        for i in range(3):
            episode = Episode(
                episode_id=f"batch_episode_{i}",
                owner_id="test_user",
                episode_type=EpisodeType.CONVERSATIONAL,
                level=EpisodeLevel.ATOMIC,
                title=f"Test Episode {i}",
                content=f"Content for episode {i}",
                summary=f"Summary {i}",
                temporal_info=sample_episode.temporal_info,
                metadata=sample_episode.metadata,
                structured_data={},
                search_keywords=[f"keyword{i}"],
                importance_score=0.5,
            )
            episodes.append(episode)

        # Store batch
        stored_ids = await episode_repo.store_episode_batch(episodes)
        assert len(stored_ids) == 3

        # Retrieve batch
        retrieved_list = await episode_repo.get_episode_batch(stored_ids)
        assert len(retrieved_list) == 3
        assert all(item is not None for item in retrieved_list)

    async def test_search_episodes(self, episode_repo, sample_episode):
        """Test searching episodes."""
        # Store test episode
        await episode_repo.store_episode(sample_episode)

        # Search by owner
        query = EpisodeQuery(owner_ids=["test_user"])
        result = await episode_repo.search_episodes(query)
        assert result.total_count >= 1
        assert len(result.episodes) >= 1

        # Search by text
        query = EpisodeQuery(text_search="hello")
        result = await episode_repo.search_episodes(query)
        assert result.total_count >= 1

        # Search by episode type
        query = EpisodeQuery(episode_types=[EpisodeType.CONVERSATIONAL])
        result = await episode_repo.search_episodes(query)
        assert result.total_count >= 1

    async def test_update_episode(self, episode_repo, sample_episode):
        """Test updating episodes."""
        # Store original episode
        await episode_repo.store_episode(sample_episode)

        # Update episode
        updated_episode = sample_episode
        updated_episode.title = "Updated Title"
        updated_episode.importance_score = 0.9

        success = await episode_repo.update_episode(sample_episode.episode_id, updated_episode)
        assert success

        # Verify update
        retrieved = await episode_repo.get_episode(sample_episode.episode_id)
        assert retrieved.title == "Updated Title"
        assert retrieved.importance_score == 0.9

    async def test_mark_episode_accessed(self, episode_repo, sample_episode):
        """Test marking episode as accessed."""
        # Store episode
        await episode_repo.store_episode(sample_episode)

        original_recall_count = sample_episode.recall_count

        # Mark as accessed
        success = await episode_repo.mark_episode_accessed(sample_episode.episode_id)
        assert success

        # Verify
        retrieved = await episode_repo.get_episode(sample_episode.episode_id)
        assert retrieved.recall_count == original_recall_count + 1
        assert retrieved.last_accessed is not None

    async def test_delete_episode(self, episode_repo, sample_episode):
        """Test deleting episodes."""
        # Store episode
        await episode_repo.store_episode(sample_episode)

        # Delete episode
        success = await episode_repo.delete_episode(sample_episode.episode_id)
        assert success

        # Verify deletion
        retrieved = await episode_repo.get_episode(sample_episode.episode_id)
        assert retrieved is None

    async def test_get_episodes_by_owner(self, episode_repo, sample_episode):
        """Test getting episodes by owner."""
        # Store episode
        await episode_repo.store_episode(sample_episode)

        # Get episodes by owner
        result = await episode_repo.get_episodes_by_owner("test_user")
        assert result.total_count >= 1
        assert any(ep.episode_id == sample_episode.episode_id for ep in result.episodes)

    async def test_get_recent_episodes(self, episode_repo, sample_episode):
        """Test getting recent episodes."""
        # Store episode
        await episode_repo.store_episode(sample_episode)

        # Get recent episodes
        result = await episode_repo.get_recent_episodes(hours=24)
        assert result.total_count >= 1
        assert any(ep.episode_id == sample_episode.episode_id for ep in result.episodes)

    async def test_health_check(self, episode_repo):
        """Test health check."""
        is_healthy = await episode_repo.health_check()
        assert is_healthy is True

    async def test_get_stats(self, episode_repo, sample_episode):
        """Test getting storage statistics."""
        # Store episode
        await episode_repo.store_episode(sample_episode)

        # Get stats
        stats = await episode_repo.get_stats()
        assert stats.total_episodes >= 1
        assert EpisodeType.CONVERSATIONAL in stats.episodes_by_type
        assert stats.episodes_by_type[EpisodeType.CONVERSATIONAL] >= 1


class TestPostgreSQLFactory:
    """Test PostgreSQL factory functions."""

    def test_create_postgresql_config(self):
        """Test creating PostgreSQL configuration."""
        config = create_postgresql_config(
            host="localhost", port=5432, database="test_db", username="test_user", password="test_pass"
        )

        assert config.backend_type == "postgresql"
        assert "postgresql+asyncpg://" in config.connection_string
        assert "test_user:test_pass@localhost:5432/test_db" in config.connection_string

    def test_create_postgresql_config_no_password(self):
        """Test creating PostgreSQL configuration without password."""
        config = create_postgresql_config(host="localhost", database="test_db", username="test_user")

        assert config.backend_type == "postgresql"
        assert "postgresql+asyncpg://" in config.connection_string
        assert "test_user@localhost:5432/test_db" in config.connection_string

    @pytest.mark.asyncio
    async def test_create_repositories_postgresql(self, postgresql_config):
        """Test creating repositories with PostgreSQL backend."""
        raw_repo, episode_repo = create_repositories(postgresql_config)

        assert isinstance(raw_repo, PostgreSQLRawDataRepository)
        assert isinstance(episode_repo, PostgreSQLEpisodicMemoryRepository)

        # Initialize and test
        await raw_repo.initialize()
        await episode_repo.initialize()

        assert await raw_repo.health_check()
        assert await episode_repo.health_check()

        await raw_repo.close()
        await episode_repo.close()
