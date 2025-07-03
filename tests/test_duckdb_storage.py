"""
Integration tests for SQLModel-based DuckDB storage implementation.

This module tests the SQLModel-based DuckDB storage layer to ensure
it maintains compatibility with the original implementation while
providing better type safety and security.
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import pytest_asyncio

from nemori.core.data_types import DataType, RawEventData, TemporalInfo
from nemori.core.episode import Episode, EpisodeLevel, EpisodeMetadata, EpisodeType
from nemori.storage import (
    EpisodeQuery,
    RawDataQuery,
    StorageConfig,
    TimeRange,
)
from nemori.storage.duckdb_storage import (
    DuckDBEpisodicMemoryRepository,
    DuckDBRawDataRepository,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test databases."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def duckdb_raw_config(temp_dir):
    """Create a SQLModel DuckDB storage configuration for raw data."""
    return StorageConfig(
        backend_type="duckdb",
        connection_string=str(temp_dir / "test_raw_sqlmodel.duckdb"),
        cache_size=100,
        enable_full_text_search=True,
        enable_semantic_search=True,
    )


@pytest.fixture
def duckdb_episode_config(temp_dir):
    """Create a SQLModel DuckDB storage configuration for episodes."""
    return StorageConfig(
        backend_type="duckdb",
        connection_string=str(temp_dir / "test_episodes_sqlmodel.duckdb"),
        cache_size=100,
        enable_full_text_search=True,
        enable_semantic_search=True,
    )


@pytest_asyncio.fixture
async def duckdb_raw_repo(duckdb_raw_config):
    """Create and initialize a SQLModel DuckDB raw data repository."""
    repo = DuckDBRawDataRepository(duckdb_raw_config)
    await repo.initialize()
    yield repo
    await repo.close()


@pytest_asyncio.fixture
async def duckdb_episode_repo(duckdb_episode_config):
    """Create and initialize a SQLModel DuckDB episode repository."""
    repo = DuckDBEpisodicMemoryRepository(duckdb_episode_config)
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


class TestDuckDBRawDataRepository:
    """Test cases for DuckDBRawDataRepository."""

    async def test_initialization_and_health_check(self, duckdb_raw_config):
        """Test repository initialization and health check."""
        repo = DuckDBRawDataRepository(duckdb_raw_config)
        assert not await repo.health_check()  # Not initialized yet

        await repo.initialize()
        assert await repo.health_check()  # Now healthy

        await repo.close()

    async def test_store_and_retrieve_raw_data(self, duckdb_raw_repo, sample_raw_data):
        """Test storing and retrieving raw data."""
        # Store data
        data_id = await duckdb_raw_repo.store_raw_data(sample_raw_data)
        assert data_id == sample_raw_data.data_id

        # Retrieve data
        retrieved_data = await duckdb_raw_repo.get_raw_data(data_id)
        assert retrieved_data is not None
        assert retrieved_data.data_id == sample_raw_data.data_id
        assert retrieved_data.data_type == sample_raw_data.data_type
        assert retrieved_data.content == sample_raw_data.content

    async def test_input_validation(self, duckdb_raw_repo):
        """Test input validation and security checks."""
        # Test invalid data_id
        with pytest.raises(ValueError, match="ID contains invalid characters"):
            invalid_data = RawEventData(
                data_type=DataType.CONVERSATION,
                content="test",
                source="test",
                temporal_info=TemporalInfo(timestamp=datetime.now()),
            )
            invalid_data.data_id = "test; DROP TABLE raw_data; --"
            await duckdb_raw_repo.store_raw_data(invalid_data)

        # Test empty data_id
        with pytest.raises(ValueError, match="ID cannot be empty"):
            invalid_data = RawEventData(
                data_type=DataType.CONVERSATION,
                content="test",
                source="test",
                temporal_info=TemporalInfo(timestamp=datetime.now()),
            )
            invalid_data.data_id = ""
            await duckdb_raw_repo.store_raw_data(invalid_data)

        # Test too long data_id
        with pytest.raises(ValueError, match="ID too long"):
            invalid_data = RawEventData(
                data_type=DataType.CONVERSATION,
                content="test",
                source="test",
                temporal_info=TemporalInfo(timestamp=datetime.now()),
            )
            invalid_data.data_id = "x" * 300
            await duckdb_raw_repo.store_raw_data(invalid_data)

    async def test_search_validation(self, duckdb_raw_repo):
        """Test search parameter validation."""
        # Test invalid search term
        with pytest.raises(ValueError, match="Search term contains invalid pattern"):
            query = RawDataQuery(content_contains="test; DROP TABLE raw_data; --")
            await duckdb_raw_repo.search_raw_data(query)

        # Test invalid limit
        with pytest.raises(ValueError, match="Limit too large"):
            query = RawDataQuery(limit=20000)
            await duckdb_raw_repo.search_raw_data(query)

        # Test invalid offset
        with pytest.raises(ValueError, match="Offset too large"):
            query = RawDataQuery(offset=2000000)
            await duckdb_raw_repo.search_raw_data(query)

    async def test_store_batch_raw_data(self, duckdb_raw_repo):
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

        data_ids = await duckdb_raw_repo.store_raw_data_batch(data_list)
        assert len(data_ids) == 3

        # Verify all data was stored
        retrieved_data = await duckdb_raw_repo.get_raw_data_batch(data_ids)
        assert len(retrieved_data) == 3
        assert all(data is not None for data in retrieved_data)

    async def test_search_raw_data_by_type(self, duckdb_raw_repo, sample_raw_data):
        """Test searching raw data by data type."""
        await duckdb_raw_repo.store_raw_data(sample_raw_data)

        # Search by conversation type
        query = RawDataQuery(data_types=[DataType.CONVERSATION])
        results = await duckdb_raw_repo.search_raw_data(query)

        assert results.total_count == 1
        assert len(results.data) == 1
        assert not results.has_more

    async def test_search_raw_data_by_content(self, duckdb_raw_repo, sample_raw_data):
        """Test searching raw data by content."""
        await duckdb_raw_repo.store_raw_data(sample_raw_data)

        # Search by content containing "Hello"
        query = RawDataQuery(content_contains="Hello")
        results = await duckdb_raw_repo.search_raw_data(query)

        assert results.total_count == 1
        assert "Hello" in str(results.data[0].content)

    async def test_search_raw_data_by_time_range(self, duckdb_raw_repo):
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

        await duckdb_raw_repo.store_raw_data(old_data)
        await duckdb_raw_repo.store_raw_data(new_data)

        # Search for recent data (last hour)
        time_range = TimeRange(start=now - timedelta(hours=1), end=now)
        query = RawDataQuery(time_range=time_range)
        results = await duckdb_raw_repo.search_raw_data(query)

        assert results.total_count == 1
        assert results.data[0].content == "New document"

    async def test_mark_as_processed(self, duckdb_raw_repo, sample_raw_data):
        """Test marking raw data as processed."""
        data_id = await duckdb_raw_repo.store_raw_data(sample_raw_data)

        # Initially not processed
        assert not sample_raw_data.processed

        # Mark as processed
        success = await duckdb_raw_repo.mark_as_processed(data_id, "v1.0")
        assert success

        # Verify processed status
        updated_data = await duckdb_raw_repo.get_raw_data(data_id)
        assert updated_data.processed
        assert updated_data.processing_version == "v1.0"

    async def test_get_unprocessed_data(self, duckdb_raw_repo):
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

        await duckdb_raw_repo.store_raw_data(processed_data)
        await duckdb_raw_repo.store_raw_data(unprocessed_data)

        # Get unprocessed data
        unprocessed = await duckdb_raw_repo.get_unprocessed_data(DataType.CONVERSATION)
        assert len(unprocessed) == 1
        assert unprocessed[0].content == "Unprocessed"

    async def test_storage_stats(self, duckdb_raw_repo, sample_raw_data):
        """Test storage statistics."""
        await duckdb_raw_repo.store_raw_data(sample_raw_data)

        stats = await duckdb_raw_repo.get_stats()
        assert stats.total_raw_data == 1
        assert stats.processed_raw_data == 0
        assert DataType.CONVERSATION in stats.raw_data_by_type
        assert stats.raw_data_by_type[DataType.CONVERSATION] == 1

    async def test_backup_and_restore(self, duckdb_raw_repo, sample_raw_data, temp_dir):
        """Test backup and restore functionality."""
        # Store data and capture the data_id
        data_id = await duckdb_raw_repo.store_raw_data(sample_raw_data)

        # Create backup
        backup_file = temp_dir / "raw_backup_sqlmodel.duckdb"
        success = await duckdb_raw_repo.backup(str(backup_file))
        assert success
        assert backup_file.exists()

        # Clear repository
        await duckdb_raw_repo.delete_raw_data(data_id)
        stats = await duckdb_raw_repo.get_stats()
        assert stats.total_raw_data == 0

        # Restore from backup
        success = await duckdb_raw_repo.restore(str(backup_file))
        assert success

        # Verify restoration
        restored_data = await duckdb_raw_repo.get_raw_data(data_id)
        assert restored_data is not None
        assert restored_data.content == sample_raw_data.content


class TestDuckDBEpisodicMemoryRepository:
    """Test cases for DuckDBEpisodicMemoryRepository."""

    async def test_initialization_and_health_check(self, duckdb_episode_config):
        """Test repository initialization and health check."""
        repo = DuckDBEpisodicMemoryRepository(duckdb_episode_config)
        assert not await repo.health_check()  # Not initialized yet

        await repo.initialize()
        assert await repo.health_check()  # Now healthy

        await repo.close()

    async def test_store_and_retrieve_episode(self, duckdb_episode_repo, sample_episode):
        """Test storing and retrieving episodes."""
        # Store episode
        episode_id = await duckdb_episode_repo.store_episode(sample_episode)
        assert episode_id == sample_episode.episode_id

        # Retrieve episode
        retrieved_episode = await duckdb_episode_repo.get_episode(episode_id)
        assert retrieved_episode is not None
        assert retrieved_episode.episode_id == sample_episode.episode_id
        assert retrieved_episode.title == sample_episode.title

    async def test_input_validation(self, duckdb_episode_repo):
        """Test input validation for episodes."""
        # Test invalid episode_id
        with pytest.raises(ValueError, match="ID contains invalid characters"):
            invalid_episode = Episode(
                owner_id="test_user",
                episode_type=EpisodeType.CONVERSATIONAL,
                level=EpisodeLevel.ATOMIC,
                title="Test",
                content="Test content",
                temporal_info=TemporalInfo(timestamp=datetime.now()),
            )
            invalid_episode.episode_id = "test; DROP TABLE episodes; --"
            await duckdb_episode_repo.store_episode(invalid_episode)

    async def test_search_episodes_by_text(self, duckdb_episode_repo, sample_episode):
        """Test searching episodes by text."""
        await duckdb_episode_repo.store_episode(sample_episode)

        # Search by text in title
        results = await duckdb_episode_repo.search_episodes(EpisodeQuery(text_search="Test", owner_ids=["test_user"]))
        assert results.total_count == 1
        assert results.episodes[0].title == sample_episode.title

    async def test_search_episodes_by_keywords(self, duckdb_episode_repo, sample_episode):
        """Test searching episodes by keywords."""
        await duckdb_episode_repo.store_episode(sample_episode)

        # Search by keywords
        results = await duckdb_episode_repo.search_episodes(
            EpisodeQuery(keywords=["conversation"], owner_ids=["test_user"])
        )
        assert results.total_count == 1

    async def test_get_episodes_by_owner(self, duckdb_episode_repo, sample_episode):
        """Test retrieving episodes by owner."""
        await duckdb_episode_repo.store_episode(sample_episode)

        # Another episode with different owner
        other_episode = Episode(
            owner_id="other_user",
            episode_type=EpisodeType.CONVERSATIONAL,
            level=EpisodeLevel.ATOMIC,
            title="Other Episode",
            content="Another episode",
            temporal_info=TemporalInfo(timestamp=datetime.now()),
        )
        await duckdb_episode_repo.store_episode(other_episode)

        # Get episodes for test_user
        results = await duckdb_episode_repo.get_episodes_by_owner("test_user")
        assert results.total_count == 1
        assert results.episodes[0].owner_id == "test_user"

    async def test_episode_access_tracking(self, duckdb_episode_repo, sample_episode):
        """Test episode access tracking."""
        episode_id = await duckdb_episode_repo.store_episode(sample_episode)

        # Initially no accesses
        episode = await duckdb_episode_repo.get_episode(episode_id)
        assert episode.recall_count == 0

        # Mark as accessed
        await duckdb_episode_repo.mark_episode_accessed(episode_id)

        # Check access count
        updated_episode = await duckdb_episode_repo.get_episode(episode_id)
        assert updated_episode.recall_count == 1

    async def test_episode_importance_update(self, duckdb_episode_repo, sample_episode):
        """Test updating episode importance."""
        episode_id = await duckdb_episode_repo.store_episode(sample_episode)

        # Update importance
        new_importance = 0.9
        success = await duckdb_episode_repo.update_episode_importance(episode_id, new_importance)
        assert success

        # Verify update
        updated_episode = await duckdb_episode_repo.get_episode(episode_id)
        assert updated_episode.importance_score == pytest.approx(new_importance, rel=1e-6)

    async def test_episode_raw_data_linking(self, duckdb_episode_repo, sample_episode):
        """Test linking episodes to raw data."""
        episode_id = await duckdb_episode_repo.store_episode(sample_episode)
        raw_data_ids = ["raw_1", "raw_2"]

        # Link episode to raw data
        success = await duckdb_episode_repo.link_episode_to_raw_data(episode_id, raw_data_ids)
        assert success

        # Verify linking
        linked_raw_data = await duckdb_episode_repo.get_raw_data_for_episode(episode_id)
        assert set(linked_raw_data) == set(raw_data_ids)

        # Verify reverse lookup
        episodes_for_raw = await duckdb_episode_repo.get_episodes_for_raw_data("raw_1")
        assert len(episodes_for_raw) == 1
        assert episodes_for_raw[0].episode_id == episode_id

    async def test_storage_stats(self, duckdb_episode_repo, sample_episode):
        """Test storage statistics."""
        await duckdb_episode_repo.store_episode(sample_episode)

        stats = await duckdb_episode_repo.get_stats()
        assert stats.total_episodes == 1
        assert EpisodeType.CONVERSATIONAL in stats.episodes_by_type
        assert stats.episodes_by_type[EpisodeType.CONVERSATIONAL] == 1
        assert EpisodeLevel.ATOMIC in stats.episodes_by_level
        assert stats.episodes_by_level[EpisodeLevel.ATOMIC] == 1

    async def test_backup_and_restore(self, duckdb_episode_repo, sample_episode, temp_dir):
        """Test backup and restore functionality."""
        # Store episode and capture the episode_id
        episode_id = await duckdb_episode_repo.store_episode(sample_episode)

        # Create backup
        backup_file = temp_dir / "episode_backup_sqlmodel.duckdb"
        success = await duckdb_episode_repo.backup(str(backup_file))
        assert success
        assert backup_file.exists()

        # Clear repository
        await duckdb_episode_repo.delete_episode(episode_id)
        stats = await duckdb_episode_repo.get_stats()
        assert stats.total_episodes == 0

        # Restore from backup
        success = await duckdb_episode_repo.restore(str(backup_file))
        assert success

        # Verify restoration
        restored_episode = await duckdb_episode_repo.get_episode(episode_id)
        assert restored_episode is not None
        assert restored_episode.title == sample_episode.title


class TestDuckDBIntegration:
    """Integration tests for SQLModel DuckDB storage layer."""

    async def test_end_to_end_workflow(self, temp_dir):
        """Test complete workflow from raw data to episodes."""
        # Create repositories
        raw_config = StorageConfig(
            backend_type="duckdb", connection_string=str(temp_dir / "integration_raw_sqlmodel.duckdb")
        )
        episode_config = StorageConfig(
            backend_type="duckdb", connection_string=str(temp_dir / "integration_episodes_sqlmodel.duckdb")
        )

        raw_repo = DuckDBRawDataRepository(raw_config)
        episode_repo = DuckDBEpisodicMemoryRepository(episode_config)

        await raw_repo.initialize()
        await episode_repo.initialize()

        try:
            # Step 1: Store raw data
            raw_data = RawEventData(
                data_type=DataType.CONVERSATION,
                content=[
                    {"speaker_id": "user", "content": "What is machine learning?", "timestamp": "2024-01-15T10:00:00"},
                    {
                        "speaker_id": "assistant",
                        "content": "Machine learning is a subset of artificial intelligence...",
                        "timestamp": "2024-01-15T10:00:30",
                    },
                ],
                source="education_chat",
                temporal_info=TemporalInfo(timestamp=datetime.now()),
                metadata={"topic": "education", "category": "AI"},
            )

            raw_data_id = await raw_repo.store_raw_data(raw_data)

            # Step 2: Create episode from raw data
            episode = Episode(
                owner_id="student_001",
                episode_type=EpisodeType.CONVERSATIONAL,
                level=EpisodeLevel.ATOMIC,
                title="Machine Learning Discussion",
                content="Student asked about machine learning and received an informative explanation.",
                summary="Q&A about machine learning fundamentals",
                temporal_info=raw_data.temporal_info,
                metadata=EpisodeMetadata(
                    source_data_ids=[raw_data_id],
                    entities=["machine learning", "AI", "student", "assistant"],
                    topics=["education", "machine learning"],
                ),
                search_keywords=["machine learning", "AI", "education", "discussion"],
                importance_score=0.8,
            )

            episode_id = await episode_repo.store_episode(episode)

            # Step 3: Link episode to raw data
            await episode_repo.link_episode_to_raw_data(episode_id, [raw_data_id])

            # Step 4: Mark raw data as processed
            await raw_repo.mark_as_processed(raw_data_id, "v1.0")

            # Step 5: Test searches
            # Search raw data
            raw_search = await raw_repo.search_raw_data(RawDataQuery(content_contains="machine learning"))
            assert raw_search.total_count == 1

            # Search episodes
            episode_search = await episode_repo.search_episodes(
                EpisodeQuery(text_search="machine learning", owner_ids=["student_001"])
            )
            assert episode_search.total_count == 1

            # Step 6: Test relationships
            episodes_for_raw = await episode_repo.get_episodes_for_raw_data(raw_data_id)
            assert len(episodes_for_raw) == 1
            assert episodes_for_raw[0].episode_id == episode_id

            raw_data_for_episode = await episode_repo.get_raw_data_for_episode(episode_id)
            assert raw_data_id in raw_data_for_episode

            # Step 7: Test statistics
            raw_stats = await raw_repo.get_stats()
            assert raw_stats.total_raw_data == 1
            assert raw_stats.processed_raw_data == 1

            episode_stats = await episode_repo.get_stats()
            assert episode_stats.total_episodes == 1

        finally:
            await raw_repo.close()
            await episode_repo.close()

    async def test_security_features(self, temp_dir):
        """Test security validation features."""
        config = StorageConfig(backend_type="duckdb", connection_string=str(temp_dir / "security_test_sqlmodel.duckdb"))

        repo = DuckDBRawDataRepository(config)
        await repo.initialize()

        try:
            # Test SQL injection prevention in IDs
            malicious_data = RawEventData(
                data_type=DataType.CONVERSATION,
                content="test",
                source="test",
                temporal_info=TemporalInfo(timestamp=datetime.now()),
            )
            malicious_data.data_id = "'; DROP TABLE raw_data; --"

            with pytest.raises(ValueError):
                await repo.store_raw_data(malicious_data)

            # Test search term sanitization
            with pytest.raises(ValueError):
                query = RawDataQuery(content_contains="test'; DROP TABLE raw_data; --")
                await repo.search_raw_data(query)

            # Test limit validation
            with pytest.raises(ValueError):
                query = RawDataQuery(limit=50000)
                await repo.search_raw_data(query)

        finally:
            await repo.close()
