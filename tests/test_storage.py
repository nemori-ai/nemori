"""
Test cases for the storage layer implementation.
"""

from datetime import datetime, timedelta

import pytest
import pytest_asyncio

from nemori.core.data_types import DataType, RawEventData, TemporalInfo
from nemori.core.episode import Episode, EpisodeLevel, EpisodeMetadata, EpisodeType
from nemori.storage import (
    EpisodeQuery,
    MemoryEpisodicMemoryRepository,
    MemoryRawDataRepository,
    RawDataQuery,
    StorageConfig,
    TimeRange,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture
def storage_config():
    """Create a storage configuration for testing."""
    return StorageConfig(
        backend_type="memory",
        cache_size=100,
        enable_full_text_search=True,
        enable_semantic_search=True,
    )


@pytest_asyncio.fixture
async def raw_repo(storage_config):
    """Create and initialize a raw data repository."""
    repo = MemoryRawDataRepository(storage_config)
    await repo.initialize()
    yield repo
    await repo.close()


@pytest_asyncio.fixture
async def episode_repo(storage_config):
    """Create and initialize an episode repository."""
    repo = MemoryEpisodicMemoryRepository(storage_config)
    await repo.initialize()
    yield repo
    await repo.close()


@pytest.fixture
def sample_raw_data():
    """Create sample raw data for testing."""
    return RawEventData(
        data_type=DataType.CONVERSATION,
        content=[
            {"speaker_id": "user", "content": "Hello, how are you?", "timestamp": "2024-01-15T10:00:00"},
            {"speaker_id": "assistant", "content": "I'm doing well, thank you!", "timestamp": "2024-01-15T10:00:15"},
        ],
        source="test_chat",
        temporal_info=TemporalInfo(timestamp=datetime.now()),
        metadata={"session_id": "test_123"},
    )


@pytest.fixture
def sample_episode():
    """Create sample episode for testing."""
    return Episode(
        owner_id="test_user",
        episode_type=EpisodeType.CONVERSATIONAL,
        level=EpisodeLevel.ATOMIC,
        title="Test Conversation",
        content="A brief test conversation between user and assistant.",
        summary="Test greeting conversation",
        temporal_info=TemporalInfo(timestamp=datetime.now()),
        metadata=EpisodeMetadata(
            entities=["user", "assistant"],
            topics=["greeting"],
            emotions=["friendly"],
        ),
        search_keywords=["test", "conversation", "greeting"],
        importance_score=0.5,
    )


class TestMemoryRawDataRepository:
    """Test cases for MemoryRawDataRepository."""

    async def test_initialization(self, storage_config):
        """Test repository initialization."""
        repo = MemoryRawDataRepository(storage_config)
        assert not await repo.health_check()  # Not initialized yet

        await repo.initialize()
        assert await repo.health_check()  # Now healthy

        await repo.close()

    async def test_store_and_retrieve_raw_data(self, raw_repo, sample_raw_data):
        """Test storing and retrieving raw data."""
        # Store data
        data_id = await raw_repo.store_raw_data(sample_raw_data)
        assert data_id == sample_raw_data.data_id

        # Retrieve data
        retrieved_data = await raw_repo.get_raw_data(data_id)
        assert retrieved_data is not None
        assert retrieved_data.data_id == sample_raw_data.data_id
        assert retrieved_data.data_type == sample_raw_data.data_type
        assert retrieved_data.content == sample_raw_data.content

    async def test_store_batch_raw_data(self, raw_repo):
        """Test batch storing of raw data."""
        data_list = [
            RawEventData(
                data_type=DataType.CONVERSATION,
                content=f"Test content {i}",
                source="test",
                temporal_info=TemporalInfo(timestamp=datetime.now()),
            )
            for i in range(3)
        ]

        data_ids = await raw_repo.store_raw_data_batch(data_list)
        assert len(data_ids) == 3

        # Verify all data was stored
        retrieved_data = await raw_repo.get_raw_data_batch(data_ids)
        assert len(retrieved_data) == 3
        assert all(data is not None for data in retrieved_data)

    async def test_search_raw_data_by_type(self, raw_repo, sample_raw_data):
        """Test searching raw data by data type."""
        await raw_repo.store_raw_data(sample_raw_data)

        # Search by conversation type
        query = RawDataQuery(data_types=[DataType.CONVERSATION])
        results = await raw_repo.search_raw_data(query)

        assert results.count == 1
        assert results.total_count == 1
        assert not results.has_more

    async def test_search_raw_data_by_content(self, raw_repo, sample_raw_data):
        """Test searching raw data by content."""
        await raw_repo.store_raw_data(sample_raw_data)

        # Search by content containing "Hello"
        query = RawDataQuery(content_contains="Hello")
        results = await raw_repo.search_raw_data(query)

        assert results.count == 1
        assert "Hello" in str(results.data[0].content)

    async def test_search_raw_data_by_time_range(self, raw_repo):
        """Test searching raw data by time range."""
        now = datetime.now()
        old_time = now - timedelta(hours=2)

        # Store data with different timestamps
        old_data = RawEventData(
            data_type=DataType.DOCUMENT,
            content="Old document",
            source="test",
            temporal_info=TemporalInfo(timestamp=old_time),
        )
        new_data = RawEventData(
            data_type=DataType.DOCUMENT,
            content="New document",
            source="test",
            temporal_info=TemporalInfo(timestamp=now),
        )

        await raw_repo.store_raw_data(old_data)
        await raw_repo.store_raw_data(new_data)

        # Search for recent data (last hour)
        time_range = TimeRange(start=now - timedelta(hours=1), end=now)
        query = RawDataQuery(time_range=time_range)
        results = await raw_repo.search_raw_data(query)

        assert results.count == 1
        assert results.data[0].content == "New document"

    async def test_mark_as_processed(self, raw_repo, sample_raw_data):
        """Test marking raw data as processed."""
        data_id = await raw_repo.store_raw_data(sample_raw_data)

        # Initially not processed
        assert not sample_raw_data.processed

        # Mark as processed
        success = await raw_repo.mark_as_processed(data_id, "v1.0")
        assert success

        # Verify processed status
        updated_data = await raw_repo.get_raw_data(data_id)
        assert updated_data.processed
        assert updated_data.processing_version == "v1.0"

    async def test_get_unprocessed_data(self, raw_repo):
        """Test retrieving unprocessed data."""
        # Store some processed and unprocessed data
        processed_data = RawEventData(
            data_type=DataType.CONVERSATION,
            content="Processed",
            source="test",
            processed=True,
            temporal_info=TemporalInfo(timestamp=datetime.now()),
        )
        unprocessed_data = RawEventData(
            data_type=DataType.CONVERSATION,
            content="Unprocessed",
            source="test",
            processed=False,
            temporal_info=TemporalInfo(timestamp=datetime.now()),
        )

        await raw_repo.store_raw_data(processed_data)
        await raw_repo.store_raw_data(unprocessed_data)

        # Get unprocessed data
        unprocessed = await raw_repo.get_unprocessed_data(DataType.CONVERSATION)
        assert len(unprocessed) == 1
        assert unprocessed[0].content == "Unprocessed"

    async def test_storage_stats(self, raw_repo, sample_raw_data):
        """Test storage statistics."""
        await raw_repo.store_raw_data(sample_raw_data)

        stats = await raw_repo.get_stats()
        assert stats.total_raw_data == 1
        assert stats.processed_raw_data == 0
        assert DataType.CONVERSATION in stats.raw_data_by_type
        assert stats.raw_data_by_type[DataType.CONVERSATION] == 1


class TestMemoryEpisodicMemoryRepository:
    """Test cases for MemoryEpisodicMemoryRepository."""

    async def test_store_and_retrieve_episode(self, episode_repo, sample_episode):
        """Test storing and retrieving episodes."""
        # Store episode
        episode_id = await episode_repo.store_episode(sample_episode)
        assert episode_id == sample_episode.episode_id

        # Retrieve episode
        retrieved_episode = await episode_repo.get_episode(episode_id)
        assert retrieved_episode is not None
        assert retrieved_episode.episode_id == sample_episode.episode_id
        assert retrieved_episode.title == sample_episode.title

    async def test_search_episodes_by_text(self, episode_repo, sample_episode):
        """Test searching episodes by text."""
        await episode_repo.store_episode(sample_episode)

        # Search by text in title
        results = await episode_repo.search_episodes(EpisodeQuery(text_search="Test", owner_ids=["test_user"]))
        assert results.count == 1
        assert results.episodes[0].title == sample_episode.title

    async def test_search_episodes_by_keywords(self, episode_repo, sample_episode):
        """Test searching episodes by keywords."""
        await episode_repo.store_episode(sample_episode)

        # Search by keywords
        results = await episode_repo.search_episodes(EpisodeQuery(keywords=["conversation"], owner_ids=["test_user"]))
        assert results.count == 1

    async def test_get_episodes_by_owner(self, episode_repo, sample_episode):
        """Test retrieving episodes by owner."""
        await episode_repo.store_episode(sample_episode)

        # Another episode with different owner
        other_episode = Episode(
            owner_id="other_user",
            episode_type=EpisodeType.CONVERSATIONAL,
            level=EpisodeLevel.ATOMIC,
            title="Other Episode",
            content="Another episode",
            temporal_info=TemporalInfo(timestamp=datetime.now()),
        )
        await episode_repo.store_episode(other_episode)

        # Get episodes for test_user
        results = await episode_repo.get_episodes_by_owner("test_user")
        assert results.count == 1
        assert results.episodes[0].owner_id == "test_user"

    async def test_episode_access_tracking(self, episode_repo, sample_episode):
        """Test episode access tracking."""
        episode_id = await episode_repo.store_episode(sample_episode)

        # Initially no accesses
        episode = await episode_repo.get_episode(episode_id)
        assert episode.recall_count == 0

        # Mark as accessed
        await episode_repo.mark_episode_accessed(episode_id)

        # Check access count
        updated_episode = await episode_repo.get_episode(episode_id)
        assert updated_episode.recall_count == 1

    async def test_episode_importance_update(self, episode_repo, sample_episode):
        """Test updating episode importance."""
        episode_id = await episode_repo.store_episode(sample_episode)

        # Update importance
        new_importance = 0.9
        success = await episode_repo.update_episode_importance(episode_id, new_importance)
        assert success

        # Verify update
        updated_episode = await episode_repo.get_episode(episode_id)
        assert updated_episode.importance_score == new_importance

    async def test_episode_raw_data_linking(self, episode_repo, sample_episode):
        """Test linking episodes to raw data."""
        episode_id = await episode_repo.store_episode(sample_episode)
        raw_data_ids = ["raw_1", "raw_2"]

        # Link episode to raw data
        success = await episode_repo.link_episode_to_raw_data(episode_id, raw_data_ids)
        assert success

        # Verify linking
        linked_raw_data = await episode_repo.get_raw_data_for_episode(episode_id)
        assert set(linked_raw_data) == set(raw_data_ids)

        # Verify reverse lookup
        episodes_for_raw = await episode_repo.get_episodes_for_raw_data("raw_1")
        assert len(episodes_for_raw) == 1
        assert episodes_for_raw[0].episode_id == episode_id

    async def test_related_episodes(self, episode_repo):
        """Test episode relationship management."""
        # Create two episodes
        episode1 = Episode(
            owner_id="test_user",
            episode_type=EpisodeType.CONVERSATIONAL,
            level=EpisodeLevel.ATOMIC,
            title="Episode 1",
            content="First episode",
            temporal_info=TemporalInfo(timestamp=datetime.now()),
        )
        episode2 = Episode(
            owner_id="test_user",
            episode_type=EpisodeType.CONVERSATIONAL,
            level=EpisodeLevel.ATOMIC,
            title="Episode 2",
            content="Second episode",
            temporal_info=TemporalInfo(timestamp=datetime.now()),
        )

        episode1_id = await episode_repo.store_episode(episode1)
        episode2_id = await episode_repo.store_episode(episode2)

        # Link episodes
        success = await episode_repo.link_related_episodes(episode1_id, episode2_id)
        assert success

        # Verify relationship
        related_to_1 = await episode_repo.get_related_episodes(episode1_id)
        assert len(related_to_1) == 1
        assert related_to_1[0].episode_id == episode2_id

        related_to_2 = await episode_repo.get_related_episodes(episode2_id)
        assert len(related_to_2) == 1
        assert related_to_2[0].episode_id == episode1_id

    async def test_complex_episode_query(self, episode_repo):
        """Test complex episode queries."""
        now = datetime.now()

        # Create episodes with different characteristics
        episodes_data = [
            {
                "owner_id": "user_1",
                "episode_type": EpisodeType.CONVERSATIONAL,
                "importance_score": 0.8,
                "timestamp": now - timedelta(hours=1),
                "title": "Important Chat",
            },
            {
                "owner_id": "user_1",
                "episode_type": EpisodeType.CREATIVE,
                "importance_score": 0.3,
                "timestamp": now - timedelta(hours=25),
                "title": "Old Creative Work",
            },
            {
                "owner_id": "user_2",
                "episode_type": EpisodeType.CONVERSATIONAL,
                "importance_score": 0.9,
                "timestamp": now - timedelta(minutes=30),
                "title": "Recent Important Chat",
            },
        ]

        for data in episodes_data:
            episode = Episode(
                owner_id=data["owner_id"],
                episode_type=data["episode_type"],
                level=EpisodeLevel.ATOMIC,
                title=data["title"],
                content=f"Content for {data['title']}",
                temporal_info=TemporalInfo(timestamp=data["timestamp"]),
                importance_score=data["importance_score"],
            )
            await episode_repo.store_episode(episode)

        # Complex query: user_1's conversational episodes from last 24 hours with importance > 0.5
        query = EpisodeQuery(
            owner_ids=["user_1"],
            episode_types=[EpisodeType.CONVERSATIONAL],
            recent_hours=24,
            min_importance=0.5,
        )

        results = await episode_repo.search_episodes(query)
        assert results.count == 1
        assert results.episodes[0].title == "Important Chat"

    async def test_backup_and_restore(self, episode_repo, sample_episode, tmp_path):
        """Test backup and restore functionality."""
        # Store episode
        await episode_repo.store_episode(sample_episode)

        # Create backup
        backup_file = tmp_path / "test_backup.json"
        success = await episode_repo.backup(str(backup_file))
        assert success
        assert backup_file.exists()

        # Clear repository
        await episode_repo.delete_episode(sample_episode.episode_id)
        stats = await episode_repo.get_stats()
        assert stats.total_episodes == 0

        # Restore from backup
        success = await episode_repo.restore(str(backup_file))
        assert success

        # Verify restoration
        restored_episode = await episode_repo.get_episode(sample_episode.episode_id)
        assert restored_episode is not None
        assert restored_episode.title == sample_episode.title
