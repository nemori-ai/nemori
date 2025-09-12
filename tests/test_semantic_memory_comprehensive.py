"""
Tests for semantic memory functionality in Nemori.

This module tests the semantic memory system including semantic nodes,
relationships, discovery, evolution, and storage operations.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from nemori.core.data_types import (
    ConversationData,
    ConversationMessage,
    DataType,
    RawEventData,
    RelationshipType,
    SemanticNode,
    SemanticRelationship,
    TemporalInfo,
)
from nemori.core.episode import Episode, EpisodeLevel, EpisodeType, EpisodeMetadata
from nemori.semantic.discovery import ContextAwareSemanticDiscoveryEngine
from nemori.semantic.evolution import SemanticEvolutionManager
from nemori.semantic.unified_retrieval import UnifiedRetrievalService
from nemori.storage.storage_types import StorageConfig, SemanticNodeQuery, SortOrder


class TestSemanticNode:
    """Test semantic node data structure."""

    def test_semantic_node_creation(self):
        """Test creating a semantic node."""
        node = SemanticNode(
            owner_id="user123",
            key="John的研究方向",
            value="大语言模型提示工程",
            context="John表示最近在研究大语言模型的提示工程",
            confidence=0.9,
        )

        assert node.owner_id == "user123"
        assert node.key == "John的研究方向"
        assert node.value == "大语言模型提示工程"
        assert node.confidence == 0.9
        assert node.version == 1
        assert len(node.evolution_history) == 0

    def test_semantic_node_evolution(self):
        """Test semantic node evolution."""
        node = SemanticNode(
            owner_id="user123",
            key="John的研究方向",
            value="大语言模型提示工程",
        )

        evolved_node = node.evolve(
            new_value="AI Agent行为规划",
            new_context="John转向AI Agent的行为规划研究",
            evolution_episode_id="episode_2",
        )

        assert evolved_node.version == 2
        assert evolved_node.value == "AI Agent行为规划"
        assert evolved_node.evolution_history == ["大语言模型提示工程"]
        assert "episode_2" in evolved_node.evolution_episode_ids

    def test_semantic_node_add_linked_episode(self):
        """Test adding linked episode to semantic node."""
        node = SemanticNode(
            owner_id="user123",
            key="test_key",
            value="test_value",
        )

        updated_node = node.add_linked_episode("episode_1")
        assert "episode_1" in updated_node.linked_episode_ids

        # Should not duplicate
        updated_node_2 = updated_node.add_linked_episode("episode_1")
        assert updated_node_2.linked_episode_ids.count("episode_1") == 1

    def test_semantic_node_validation(self):
        """Test semantic node validation."""
        # Should raise error for empty owner_id
        with pytest.raises(ValueError, match="owner_id is required"):
            SemanticNode(owner_id="", key="test", value="test")

        # Should raise error for empty key
        with pytest.raises(ValueError, match="key is required"):
            SemanticNode(owner_id="user123", key="", value="test")

        # Should raise error for invalid confidence
        with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
            SemanticNode(owner_id="user123", key="test", value="test", confidence=1.5)


class TestSemanticRelationship:
    """Test semantic relationship data structure."""

    def test_semantic_relationship_creation(self):
        """Test creating a semantic relationship."""
        relationship = SemanticRelationship(
            source_node_id="node1",
            target_node_id="node2",
            relationship_type=RelationshipType.RELATED,
            strength=0.8,
            description="Related nodes discovered together",
        )

        assert relationship.source_node_id == "node1"
        assert relationship.target_node_id == "node2"
        assert relationship.relationship_type == RelationshipType.RELATED
        assert relationship.strength == 0.8

    def test_semantic_relationship_validation(self):
        """Test semantic relationship validation."""
        # Should raise error for same source and target
        with pytest.raises(ValueError, match="source_node_id and target_node_id cannot be the same"):
            SemanticRelationship(
                source_node_id="node1",
                target_node_id="node1",
            )

        # Should raise error for invalid strength
        with pytest.raises(ValueError, match="strength must be between 0.0 and 1.0"):
            SemanticRelationship(
                source_node_id="node1",
                target_node_id="node2",
                strength=1.5,
            )

    def test_semantic_relationship_reinforce(self):
        """Test reinforcing a semantic relationship."""
        relationship = SemanticRelationship(
            source_node_id="node1",
            target_node_id="node2",
            strength=0.5,
        )

        reinforced = relationship.reinforce(0.2)
        assert reinforced.strength == 0.7

        # Should not exceed 1.0
        max_reinforced = reinforced.reinforce(0.5)
        assert max_reinforced.strength == 1.0


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider."""
    provider = AsyncMock()
    provider.generate.return_value = """
    {
        "knowledge_gaps": [
            {
                "key": "John的研究方向",
                "value": "大语言模型提示工程",
                "context": "John表示最近在研究大语言模型的提示工程",
                "gap_type": "personal_fact",
                "confidence": 0.9
            }
        ]
    }
    """
    return provider


@pytest.fixture
def mock_retrieval_service():
    """Create a mock unified retrieval service."""
    service = AsyncMock(spec=UnifiedRetrievalService)
    service.search_semantic_memories.return_value = []
    service.search_episodic_memories.return_value = []
    return service


@pytest.fixture
def sample_episode():
    """Create a sample episode for testing."""
    return Episode(
        episode_id="episode_1",
        owner_id="user123",
        episode_type=EpisodeType.CONVERSATIONAL,
        level=EpisodeLevel.ATOMIC,
        title="John讨论大语言模型研究",
        content="John表示最近在研究大语言模型的提示工程，特别关注如何让LLM更好地理解复杂指令。",
        summary="John分享了他在大语言模型提示工程方面的研究兴趣",
        temporal_info=TemporalInfo(timestamp=datetime.now()),
        metadata=EpisodeMetadata(),
    )


class TestSemanticDiscoveryEngine:
    """Test semantic discovery engine."""

    @pytest.mark.asyncio
    async def test_discover_semantic_knowledge(self, mock_llm_provider, mock_retrieval_service, sample_episode):
        """Test discovering semantic knowledge from episode."""
        engine = ContextAwareSemanticDiscoveryEngine(
            llm_provider=mock_llm_provider,
            retrieval_service=mock_retrieval_service,
        )

        original_content = "John: 我最近在研究大语言模型的提示工程\nAssistant: 很有趣的方向！你主要关注哪些方面？"

        discovered_nodes = await engine.discover_semantic_knowledge(sample_episode, original_content)

        assert len(discovered_nodes) > 0
        assert discovered_nodes[0].owner_id == "user123"
        assert discovered_nodes[0].key == "John的研究方向"
        assert discovered_nodes[0].discovery_episode_id == "episode_1"

    @pytest.mark.asyncio
    async def test_gather_discovery_context(self, mock_llm_provider, mock_retrieval_service, sample_episode):
        """Test gathering context for discovery."""
        engine = ContextAwareSemanticDiscoveryEngine(
            llm_provider=mock_llm_provider,
            retrieval_service=mock_retrieval_service,
        )

        context = await engine._gather_discovery_context(sample_episode)

        assert "related_semantic_memories" in context
        assert "related_historical_episodes" in context
        assert "current_episode" in context
        assert context["current_episode"] == sample_episode


class TestSemanticEvolutionManager:
    """Test semantic evolution manager."""

    @pytest.fixture
    def mock_semantic_storage(self):
        """Create a mock semantic storage."""
        storage = AsyncMock()
        storage.find_semantic_node_by_key.return_value = None
        storage.store_semantic_node = AsyncMock()
        storage.update_semantic_node = AsyncMock()
        return storage

    @pytest.mark.asyncio
    async def test_process_new_episode(self, mock_semantic_storage, mock_retrieval_service, sample_episode):
        """Test processing a new episode for semantic knowledge."""
        mock_discovery_engine = AsyncMock()
        mock_discovery_engine.discover_semantic_knowledge.return_value = [
            SemanticNode(
                owner_id="user123",
                key="John的研究方向",
                value="大语言模型提示工程",
                discovery_episode_id="episode_1",
            )
        ]

        manager = SemanticEvolutionManager(
            storage=mock_semantic_storage,
            discovery_engine=mock_discovery_engine,
            retrieval_service=mock_retrieval_service,
        )

        original_content = "John: 我最近在研究大语言模型的提示工程"
        processed_nodes = await manager.process_episode_for_semantics(sample_episode, original_content)

        assert len(processed_nodes) == 1
        assert processed_nodes[0].key == "John的研究方向"
        mock_semantic_storage.store_semantic_node.assert_called_once()

    @pytest.mark.asyncio
    async def test_evolve_existing_knowledge(self, mock_semantic_storage, mock_retrieval_service, sample_episode):
        """Test evolving existing semantic knowledge."""
        # Mock existing node
        existing_node = SemanticNode(
            node_id="node_1",
            owner_id="user123",
            key="John的研究方向",
            value="大语言模型提示工程",
            version=1,
        )
        mock_semantic_storage.find_semantic_node_by_key.return_value = existing_node

        # Mock discovery engine to return evolved knowledge
        mock_discovery_engine = AsyncMock()
        mock_discovery_engine.discover_semantic_knowledge.return_value = [
            SemanticNode(
                owner_id="user123",
                key="John的研究方向",
                value="AI Agent行为规划",  # Different value
                discovery_episode_id="episode_2",
            )
        ]

        manager = SemanticEvolutionManager(
            storage=mock_semantic_storage,
            discovery_engine=mock_discovery_engine,
            retrieval_service=mock_retrieval_service,
        )

        episode_2 = Episode(
            episode_id="episode_2",
            owner_id="user123",
            episode_type=EpisodeType.CONVERSATIONAL,
            level=EpisodeLevel.ATOMIC,
            title="John转向AI Agent研究",
            content="John表示已从LLM研究转向AI Agent的行为规划",
            temporal_info=TemporalInfo(timestamp=datetime.now()),
            metadata=EpisodeMetadata(),
        )

        processed_nodes = await manager.process_episode_for_semantics(episode_2, "John: 我现在转向AI Agent研究了")

        assert len(processed_nodes) == 1
        # Should have called update instead of store
        mock_semantic_storage.update_semantic_node.assert_called_once()

        # Check that the node was evolved
        updated_node = mock_semantic_storage.update_semantic_node.call_args[0][0]
        assert updated_node.version == 2
        assert updated_node.value == "AI Agent行为规划"
        assert "大语言模型提示工程" in updated_node.evolution_history


@pytest.fixture
def semantic_node_sample():
    """Sample semantic node for testing."""
    return SemanticNode(
        node_id="node_1",
        owner_id="user123",
        key="John的研究方向",
        value="大语言模型提示工程",
        context="John表示最近在研究大语言模型的提示工程",
        confidence=0.9,
        discovery_episode_id="episode_1",
        linked_episode_ids=["episode_1"],  # Add linked episode
        evolution_episode_ids=[],
    )


class TestUnifiedRetrievalService:
    """Test unified retrieval service."""

    @pytest.fixture
    def mock_episodic_storage(self):
        """Create mock episodic storage."""
        storage = AsyncMock()
        return storage

    @pytest.fixture
    def mock_semantic_storage(self):
        """Create mock semantic storage."""
        storage = AsyncMock()
        return storage

    @pytest.mark.asyncio
    async def test_search_semantic_memories(self, mock_episodic_storage, mock_semantic_storage, semantic_node_sample):
        """Test searching semantic memories."""
        mock_semantic_storage.similarity_search_semantic_nodes.return_value = [semantic_node_sample]

        service = UnifiedRetrievalService(
            episodic_storage=mock_episodic_storage,
            semantic_storage=mock_semantic_storage,
        )

        results = await service.search_semantic_memories("user123", "研究方向", limit=5)

        assert len(results) == 1
        assert results[0].key == "John的研究方向"
        mock_semantic_storage.similarity_search_semantic_nodes.assert_called_once_with(
            owner_id="user123", query="研究方向", limit=5
        )

    @pytest.mark.asyncio
    async def test_search_episodic_memories(self, mock_episodic_storage, mock_semantic_storage, sample_episode):
        """Test searching episodic memories."""
        from nemori.storage.storage_types import EpisodeSearchResult

        mock_result = EpisodeSearchResult(
            episodes=[sample_episode],
            total_count=1,
            has_more=False,
            query_time_ms=10.0,
        )
        mock_episodic_storage.search_episodes.return_value = mock_result

        service = UnifiedRetrievalService(
            episodic_storage=mock_episodic_storage,
            semantic_storage=mock_semantic_storage,
        )

        results = await service.search_episodic_memories("user123", "大语言模型", limit=5)

        assert len(results) == 1
        assert results[0].title == "John讨论大语言模型研究"
        mock_episodic_storage.search_episodes.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_semantic_episodes(
        self, mock_episodic_storage, mock_semantic_storage, semantic_node_sample, sample_episode
    ):
        """Test getting episodes associated with a semantic node."""
        mock_semantic_storage.get_semantic_node_by_id.return_value = semantic_node_sample

        service = UnifiedRetrievalService(
            episodic_storage=mock_episodic_storage,
            semantic_storage=mock_semantic_storage,
        )

        # Mock the get_episodes_by_ids method to return episodes
        service.get_episodes_by_ids = AsyncMock(return_value=[sample_episode])

        result = await service.get_semantic_episodes("node_1")

        assert "linked_episodes" in result
        assert "evolution_episodes" in result
        assert len(result["linked_episodes"]) == 1


class TestSemanticMemoryIntegration:
    """Integration tests for semantic memory system."""

    @pytest.mark.asyncio
    async def test_end_to_end_semantic_memory_workflow(self):
        """Test complete semantic memory workflow from conversation to knowledge evolution."""
        # This would be a comprehensive integration test
        # For now, we'll test the key workflow steps

        # 1. Create conversation data
        conversation_data = ConversationData(
            RawEventData(
                data_type=DataType.CONVERSATION,
                content=[
                    {
                        "speaker_id": "john",
                        "user_name": "John",
                        "content": "我最近在研究大语言模型的提示工程",
                        "timestamp": "2024-01-15T10:00:00",
                    },
                    {
                        "speaker_id": "assistant",
                        "user_name": "Assistant",
                        "content": "很有趣的方向！你主要关注哪些方面？",
                        "timestamp": "2024-01-15T10:01:00",
                    },
                ],
                temporal_info=TemporalInfo(timestamp=datetime.now()),
            )
        )

        # 2. Verify conversation data structure
        assert len(conversation_data.messages) == 2
        assert conversation_data.messages[0].speaker_id == "john"
        conversation_text = conversation_data.get_conversation_text(include_timestamps=True)
        assert "大语言模型" in conversation_text

        # 3. Semantic memory workflow would continue here with:
        # - Episode building
        # - Semantic discovery
        # - Knowledge evolution
        # - Retrieval testing

        # This demonstrates the data flow is properly set up
        assert True  # Placeholder for full integration test
