"""
Integration tests for the complete Episode Manager workflow.

This module tests the full data flow from raw data ingestion through
episode building, storage, and retrieval indexing.
"""

from datetime import datetime

import pytest
import pytest_asyncio

from nemori.builders.conversation_builder import ConversationEpisodeBuilder
from nemori.core.builders import EpisodeBuilderRegistry
from nemori.core.data_types import ConversationMessage, DataType, RawEventData, TemporalInfo
from nemori.episode_manager import EpisodeManager
from nemori.retrieval import RetrievalConfig, RetrievalService, RetrievalStrategy
from nemori.storage import MemoryEpisodicMemoryRepository, MemoryRawDataRepository, StorageConfig


@pytest_asyncio.fixture
async def storage_repos():
    """Create storage repositories for testing."""
    config = StorageConfig(backend_type="memory")

    raw_repo = MemoryRawDataRepository(config)
    episode_repo = MemoryEpisodicMemoryRepository(config)

    await raw_repo.initialize()
    await episode_repo.initialize()

    yield raw_repo, episode_repo

    await raw_repo.close()
    await episode_repo.close()


@pytest_asyncio.fixture
async def retrieval_service(storage_repos):
    """Create retrieval service for testing."""
    _, episode_repo = storage_repos

    service = RetrievalService(episode_repo)

    # Register BM25 provider
    config = RetrievalConfig(storage_type="memory")
    service.register_provider(RetrievalStrategy.BM25, config)

    await service.initialize()

    yield service

    await service.close()


@pytest_asyncio.fixture
async def episode_manager(storage_repos, retrieval_service):
    """Create episode manager with all components."""
    raw_repo, episode_repo = storage_repos

    # Set up builder registry
    registry = EpisodeBuilderRegistry()
    conv_builder = ConversationEpisodeBuilder()
    registry.register(conv_builder)

    # Create episode manager
    manager = EpisodeManager(
        raw_data_repo=raw_repo,
        episode_repo=episode_repo,
        builder_registry=registry,
        retrieval_service=retrieval_service,
    )

    return manager


@pytest.fixture
def sample_conversation_data():
    """Create sample conversation data for testing."""
    messages = [
        ConversationMessage(
            content="Hey, I was thinking about starting a new machine learning project",
            speaker_id="user",
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
        ),
        ConversationMessage(
            content="That sounds interesting! What kind of ML project are you considering?",
            speaker_id="assistant",
            timestamp=datetime(2024, 1, 15, 10, 1, 0),
        ),
        ConversationMessage(
            content="I want to build a recommendation system for movies using collaborative filtering",
            speaker_id="user",
            timestamp=datetime(2024, 1, 15, 10, 2, 0),
        ),
        ConversationMessage(
            content="Great choice! You'll need user rating data and might want to use techniques like matrix factorization or neural networks",
            speaker_id="assistant",
            timestamp=datetime(2024, 1, 15, 10, 3, 0),
        ),
    ]

    # Convert message objects to dicts with ISO format timestamps
    message_dicts = []
    for msg in messages:
        msg_dict = msg.__dict__.copy()
        if msg_dict.get("timestamp"):
            msg_dict["timestamp"] = msg_dict["timestamp"].isoformat()
        message_dicts.append(msg_dict)

    return RawEventData(
        data_id="conv_001",
        data_type=DataType.CONVERSATION,
        content=message_dicts,
        source="test_conversation",
        temporal_info=TemporalInfo(datetime(2024, 1, 15, 10, 0, 0)),
        metadata={"conversation_type": "project_planning"},
    )


class TestEpisodeManagerIntegration:
    """Integration tests for the complete episode management workflow."""

    @pytest.mark.asyncio
    async def test_complete_data_flow(self, episode_manager, sample_conversation_data):
        """Test the complete data flow from raw data to searchable episodes."""

        # 1. Process raw data through the complete pipeline
        episode = await episode_manager.process_raw_data(sample_conversation_data, owner_id="user123")

        assert episode is not None
        assert episode.owner_id == "user123"
        assert "machine learning" in episode.content.lower()
        assert episode.episode_id is not None

        # 2. Verify episode was stored
        stored_episode = await episode_manager.get_episode(episode.episode_id)
        assert stored_episode is not None
        assert stored_episode.episode_id == episode.episode_id

        # 3. Verify raw data was marked as processed
        raw_data = await episode_manager.raw_data_repo.get_raw_data("conv_001")
        assert raw_data.processed is True

        # 4. Test search functionality
        search_results = await episode_manager.search_episodes("machine learning recommendation", owner_id="user123")

        assert search_results.count > 0
        assert search_results.episodes[0].episode_id == episode.episode_id

        # 5. Test episode retrieval stats
        stats = await episode_manager.get_retrieval_stats()
        assert "bm25" in stats
        assert stats["bm25"]["total_episodes"] == 1

    @pytest.mark.asyncio
    async def test_episode_lifecycle_management(self, episode_manager, sample_conversation_data):
        """Test episode creation, update, and deletion lifecycle."""

        # Create episode
        episode = await episode_manager.process_raw_data(sample_conversation_data, owner_id="user123")
        episode_id = episode.episode_id

        # Test update
        episode.title = "Updated: ML Recommendation System Discussion"
        episode.importance_score = 0.8

        success = await episode_manager.update_episode(episode_id, episode)
        assert success is True

        # Verify update in storage
        updated_episode = await episode_manager.get_episode(episode_id)
        assert "Updated:" in updated_episode.title
        assert updated_episode.importance_score == 0.8

        # Test search after update
        search_results = await episode_manager.search_episodes("recommendation system", owner_id="user123")
        assert search_results.count > 0
        assert "Updated:" in search_results.episodes[0].title

        # Test deletion
        success = await episode_manager.delete_episode(episode_id)
        assert success is True

        # Verify deletion
        deleted_episode = await episode_manager.get_episode(episode_id)
        assert deleted_episode is None

        # Verify removal from search index
        search_results = await episode_manager.search_episodes("recommendation system", owner_id="user123")
        assert search_results.count == 0

    @pytest.mark.asyncio
    async def test_multi_user_isolation(self, episode_manager, sample_conversation_data):
        """Test that episodes are properly isolated between users."""

        # Create episodes for different users
        await episode_manager.process_raw_data(sample_conversation_data, owner_id="user1")

        # Create second conversation for different user
        sample_conversation_data.data_id = "conv_002"
        await episode_manager.process_raw_data(sample_conversation_data, owner_id="user2")

        # Test user1 can only see their episodes
        user1_results = await episode_manager.search_episodes("machine learning", owner_id="user1")
        assert user1_results.count == 1
        assert user1_results.episodes[0].owner_id == "user1"

        # Test user2 can only see their episodes
        user2_results = await episode_manager.search_episodes("machine learning", owner_id="user2")
        assert user2_results.count == 1
        assert user2_results.episodes[0].owner_id == "user2"

        # Verify episodes are different
        assert user1_results.episodes[0].episode_id != user2_results.episodes[0].episode_id

    @pytest.mark.asyncio
    async def test_batch_processing(self, episode_manager):
        """Test processing multiple conversations in sequence."""

        conversations = []
        for i in range(3):
            messages = [
                ConversationMessage(
                    content=f"This is conversation {i+1} about topic {i+1}",
                    speaker_id="user",
                    timestamp=datetime(2024, 1, 15, 10, i, 0),
                ),
                ConversationMessage(
                    content=f"Response to conversation {i+1}",
                    speaker_id="assistant",
                    timestamp=datetime(2024, 1, 15, 10, i, 30),
                ),
            ]

            # Convert message objects to dicts with ISO format timestamps
            message_dicts = []
            for msg in messages:
                msg_dict = msg.__dict__.copy()
                if msg_dict.get("timestamp"):
                    msg_dict["timestamp"] = msg_dict["timestamp"].isoformat()
                message_dicts.append(msg_dict)

            conv_data = RawEventData(
                data_id=f"conv_{i+1:03d}",
                data_type=DataType.CONVERSATION,
                content=message_dicts,
                source="test_batch",
                temporal_info=TemporalInfo(datetime(2024, 1, 15, 10, i, 0)),
            )
            conversations.append(conv_data)

        # Process all conversations
        episodes = []
        for conv_data in conversations:
            episode = await episode_manager.process_raw_data(conv_data, owner_id="batch_user")
            episodes.append(episode)

        # Verify all episodes were created
        assert len(episodes) == 3
        assert all(ep is not None for ep in episodes)

        # Test search finds multiple episodes
        search_results = await episode_manager.search_episodes("conversation", owner_id="batch_user")
        assert search_results.count == 3

        # Verify retrieval stats
        stats = await episode_manager.get_retrieval_stats()
        assert stats["bm25"]["total_episodes"] == 3

    @pytest.mark.asyncio
    async def test_index_initialization(self, episode_manager, sample_conversation_data):
        """Test initializing retrieval index from existing episodes."""

        # First, create an episode without auto-indexing
        episode = await episode_manager.process_raw_data(sample_conversation_data, owner_id="user123", auto_index=False)

        # Verify it's not in the search index initially
        search_results = await episode_manager.search_episodes("machine learning", owner_id="user123")
        assert search_results.count == 0

        # Initialize the retrieval index
        await episode_manager.initialize_retrieval_index("user123")

        # Now it should be searchable
        search_results = await episode_manager.search_episodes("machine learning", owner_id="user123")
        assert search_results.count == 1
        assert search_results.episodes[0].episode_id == episode.episode_id

    @pytest.mark.asyncio
    async def test_health_check(self, episode_manager):
        """Test health check functionality."""

        health = await episode_manager.health_check()

        # Check that all components report healthy
        assert health["raw_data_storage"] is True
        assert health["episode_storage"] is True
        assert health["retrieval_bm25"] is True

    @pytest.mark.asyncio
    async def test_episode_access_tracking(self, episode_manager, sample_conversation_data):
        """Test that episode access is properly tracked."""

        # Create episode
        episode = await episode_manager.process_raw_data(sample_conversation_data, owner_id="user123")

        # Initial access count (may be > 0 due to episode creation process)
        initial_count = episode.recall_count

        # Access the episode
        accessed_episode = await episode_manager.get_episode(episode.episode_id, mark_accessed=True)

        # Access count should be incremented
        assert accessed_episode.recall_count > initial_count
        assert accessed_episode.last_accessed is not None

    @pytest.mark.asyncio
    async def test_error_handling(self, episode_manager):
        """Test error handling in various scenarios."""

        # Test with unsupported data type
        unsupported_data = RawEventData(
            data_id="unsupported_001",
            data_type=DataType.MEDIA,  # No builder registered for this
            content=[{"file_path": "/test/image.jpg"}],
            source="test",
            temporal_info=TemporalInfo(datetime.now()),
        )

        episode = await episode_manager.process_raw_data(unsupported_data, owner_id="user123")

        # Should return None for unsupported data type
        assert episode is None

        # Test search with no retrieval service
        manager_no_retrieval = EpisodeManager(
            raw_data_repo=episode_manager.raw_data_repo,
            episode_repo=episode_manager.episode_repo,
            builder_registry=episode_manager.builder_registry,
            retrieval_service=None,
        )

        with pytest.raises(RuntimeError, match="No retrieval service configured"):
            await manager_no_retrieval.search_episodes("test", "user123")
