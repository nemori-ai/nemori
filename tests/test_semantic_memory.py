"""
Test for semantic memory functionality.

This test demonstrates the core semantic memory capabilities including
discovery, evolution, and retrieval.
"""

import asyncio
import pytest
from datetime import datetime

from nemori.core.data_types import ConversationData, ConversationMessage, DataType, RawEventData, TemporalInfo
from nemori.core.episode import Episode, EpisodeMetadata, EpisodeType, EpisodeLevel
from nemori.storage.storage_types import StorageConfig
from nemori.storage.duckdb_storage import DuckDBSemanticMemoryRepository
from nemori.semantic.discovery import ContextAwareSemanticDiscoveryEngine
from nemori.semantic.evolution import SemanticEvolutionManager
from nemori.semantic.unified_retrieval import UnifiedRetrievalService
from nemori.builders.enhanced_conversation_builder import EnhancedConversationEpisodeBuilder
from tests.conftest import MockLLMProvider


class TestSemanticMemory:
    """Test semantic memory functionality."""

    @pytest.fixture
    async def semantic_storage(self):
        """Create a semantic storage instance."""
        config = StorageConfig(connection_string="duckdb:///:memory:")
        storage = DuckDBSemanticMemoryRepository(config)
        await storage.initialize()
        yield storage
        await storage.close()

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM provider for testing."""
        return MockLLMProvider()

    @pytest.fixture
    async def semantic_engine(self, mock_llm):
        """Create semantic discovery engine."""
        return ContextAwareSemanticDiscoveryEngine(mock_llm)

    @pytest.fixture
    async def evolution_manager(self, semantic_storage, semantic_engine):
        """Create semantic evolution manager."""
        return SemanticEvolutionManager(semantic_storage, semantic_engine)

    @pytest.fixture
    async def unified_retrieval(self, semantic_storage):
        """Create unified retrieval service."""
        return UnifiedRetrievalService(None, semantic_storage)

    async def test_semantic_node_storage(self, semantic_storage):
        """Test basic semantic node storage and retrieval."""
        from nemori.core.data_types import SemanticNode
        
        node = SemanticNode(
            owner_id="test_user",
            key="John的研究方向",
            value="AI Agent行为规划",
            context="John最近在研究AI Agent的行为规划",
            confidence=0.9
        )
        
        # Store semantic node
        await semantic_storage.store_semantic_node(node)
        
        # Retrieve by ID
        retrieved_node = await semantic_storage.get_semantic_node_by_id(node.node_id)
        assert retrieved_node is not None
        assert retrieved_node.key == "John的研究方向"
        assert retrieved_node.value == "AI Agent行为规划"
        
        # Retrieve by key
        found_node = await semantic_storage.find_semantic_node_by_key("test_user", "John的研究方向")
        assert found_node is not None
        assert found_node.node_id == node.node_id

    async def test_semantic_discovery(self, semantic_engine):
        """Test semantic knowledge discovery."""
        episode = Episode(
            owner_id="test_user",
            title="John讨论研究方向",
            content="John表示最近在研究AI Agent的行为规划，特别关注决策机制。",
            summary="John的研究重点转向AI Agent行为规划",
            episode_type=EpisodeType.CONVERSATIONAL,
            level=EpisodeLevel.ATOMIC
        )
        
        original_content = """
        [2024-01-15 10:00] User: John，你最近在研究什么？
        [2024-01-15 10:01] John: 我最近在研究AI Agent的行为规划
        [2024-01-15 10:02] User: 这个方向很有前景
        [2024-01-15 10:03] John: 是的，特别是决策机制这块很有挑战性
        """
        
        # Mock LLM response for knowledge gap analysis
        semantic_engine.llm_provider.set_mock_response("""
        {
            "knowledge_gaps": [
                {
                    "key": "John的研究方向",
                    "value": "AI Agent行为规划",
                    "context": "专注于决策机制",
                    "gap_type": "personal_fact",
                    "confidence": 0.9
                }
            ]
        }
        """)
        
        discovered_nodes = await semantic_engine.discover_semantic_knowledge(
            episode, original_content
        )
        
        assert len(discovered_nodes) > 0
        node = discovered_nodes[0]
        assert node.key == "John的研究方向"
        assert node.value == "AI Agent行为规划"
        assert episode.episode_id in node.linked_episode_ids

    async def test_semantic_evolution(self, evolution_manager, semantic_storage):
        """Test semantic knowledge evolution."""
        from nemori.core.data_types import SemanticNode
        
        # Create initial semantic node
        initial_node = SemanticNode(
            owner_id="test_user",
            key="John的研究方向", 
            value="大语言模型提示工程",
            context="专注于让LLM理解复杂指令",
            confidence=0.8
        )
        
        await semantic_storage.store_semantic_node(initial_node)
        
        # Create new episode with evolved knowledge
        episode = Episode(
            owner_id="test_user",
            title="John转向Agent研究",
            content="John表示已从LLM研究转向AI Agent行为规划",
            episode_type=EpisodeType.CONVERSATIONAL
        )
        
        original_content = """
        [2024-04-20 14:30] User: John，还在做LLM研究吗？
        [2024-04-20 14:31] John: 我现在转向AI Agent的行为规划了
        [2024-04-20 14:32] User: 从LLM转向Agent了？
        [2024-04-20 14:33] John: 对，发现Agent的决策机制更有挑战性
        """
        
        # Mock knowledge gap analysis for evolution
        evolution_manager.discovery_engine.llm_provider.set_mock_response("""
        {
            "knowledge_gaps": [
                {
                    "key": "John的研究方向",
                    "value": "AI Agent行为规划", 
                    "context": "专注于Agent决策机制",
                    "gap_type": "personal_fact",
                    "confidence": 0.9
                }
            ]
        }
        """)
        
        # Process episode for semantics
        processed_nodes = await evolution_manager.process_episode_for_semantics(
            episode, original_content
        )
        
        assert len(processed_nodes) > 0
        evolved_node = processed_nodes[0]
        
        # Check evolution
        assert evolved_node.version == 2
        assert evolved_node.value == "AI Agent行为规划"
        assert "大语言模型提示工程" in evolved_node.evolution_history
        assert episode.episode_id in evolved_node.evolution_episode_ids

    async def test_enhanced_conversation_builder(self, evolution_manager):
        """Test enhanced conversation builder with semantic integration."""
        mock_llm = MockLLMProvider()
        builder = EnhancedConversationEpisodeBuilder(
            llm_provider=mock_llm,
            semantic_manager=evolution_manager
        )
        
        # Mock LLM responses
        mock_llm.set_mock_response("""
        {
            "title": "John discusses research focus on AI Agent behavior planning",
            "content": "On January 15, 2024, John mentioned his current research focus on AI Agent behavior planning, particularly interested in decision-making mechanisms."
        }
        """)
        
        evolution_manager.discovery_engine.llm_provider.set_mock_response("""
        {
            "knowledge_gaps": [
                {
                    "key": "John的研究方向",
                    "value": "AI Agent行为规划",
                    "context": "专注于决策机制",
                    "gap_type": "personal_fact", 
                    "confidence": 0.9
                }
            ]
        }
        """)
        
        # Create conversation data
        messages = [
            ConversationMessage(
                speaker_id="User",
                content="John，你最近在研究什么？",
                timestamp=datetime(2024, 1, 15, 10, 0)
            ),
            ConversationMessage(
                speaker_id="John",
                content="我最近在研究AI Agent的行为规划",
                timestamp=datetime(2024, 1, 15, 10, 1)
            ),
        ]
        
        raw_data = RawEventData(
            data_type=DataType.CONVERSATION,
            content=[msg.__dict__ for msg in messages],
            temporal_info=TemporalInfo(datetime(2024, 1, 15, 10, 0))
        )
        
        conversation_data = ConversationData(raw_data)
        
        # Build episode with semantic processing
        episode = await builder.build_episode(conversation_data, "test_user")
        
        # Verify episode was created
        assert episode is not None
        assert episode.owner_id == "test_user"
        assert episode.episode_type == EpisodeType.CONVERSATIONAL
        
        # Verify semantic metadata was added
        assert "discovered_semantics" in episode.metadata.custom_fields
        assert "semantic_node_ids" in episode.metadata.custom_fields
        assert episode.metadata.custom_fields["discovered_semantics"] > 0

    async def test_unified_retrieval_service(self, unified_retrieval, semantic_storage):
        """Test unified retrieval service functionality."""
        from nemori.core.data_types import SemanticNode
        
        # Add some semantic nodes
        nodes = [
            SemanticNode(
                owner_id="test_user",
                key="John的研究方向",
                value="AI Agent行为规划",
                context="专注于决策机制"
            ),
            SemanticNode(
                owner_id="test_user", 
                key="项目进展",
                value="Tanka项目已完成原型",
                context="用户界面设计已确定"
            )
        ]
        
        for node in nodes:
            await semantic_storage.store_semantic_node(node)
        
        # Test semantic memory search
        results = await unified_retrieval.search_semantic_memories(
            owner_id="test_user",
            query="研究",
            limit=10
        )
        
        assert len(results) > 0
        assert any("研究" in node.key for node in results)

    async def test_semantic_statistics(self, semantic_storage):
        """Test semantic memory statistics."""
        from nemori.core.data_types import SemanticNode
        
        # Add test data
        nodes = [
            SemanticNode(owner_id="test_user", key="key1", value="value1", confidence=0.8),
            SemanticNode(owner_id="test_user", key="key2", value="value2", confidence=0.9),
            SemanticNode(owner_id="other_user", key="key3", value="value3", confidence=0.7)
        ]
        
        for node in nodes:
            await semantic_storage.store_semantic_node(node)
        
        # Get statistics
        stats = await semantic_storage.get_semantic_statistics("test_user")
        
        assert stats["total_nodes"] == 2
        assert stats["average_confidence"] > 0.8
        assert stats["total_access_count"] == 0


# Standalone test function for running without pytest
async def run_semantic_test():
    """Run a simple semantic memory test."""
    print("🧠 Testing Semantic Memory Implementation")
    
    # Create storage
    config = StorageConfig(connection_string="duckdb:///:memory:")
    storage = DuckDBSemanticMemoryRepository(config)
    await storage.initialize()
    
    try:
        # Create and store a semantic node
        from nemori.core.data_types import SemanticNode
        
        node = SemanticNode(
            owner_id="demo_user",
            key="用户偏好",
            value="喜欢喝咖啡，不喜欢茶",
            context="在讨论饮品偏好时提到",
            confidence=0.9
        )
        
        await storage.store_semantic_node(node)
        print(f"✅ Stored semantic node: {node.key} = {node.value}")
        
        # Retrieve the node
        retrieved = await storage.get_semantic_node_by_id(node.node_id)
        print(f"✅ Retrieved node: {retrieved.key} = {retrieved.value}")
        
        # Test similarity search
        results = await storage.similarity_search_semantic_nodes("demo_user", "咖啡", 5)
        print(f"✅ Similarity search for '咖啡' found {len(results)} results")
        
        # Get statistics
        stats = await storage.get_semantic_statistics("demo_user")
        print(f"✅ Statistics: {stats['total_nodes']} nodes, avg confidence: {stats['average_confidence']:.2f}")
        
        print("🎉 All semantic memory tests passed!")
        
    finally:
        await storage.close()


if __name__ == "__main__":
    asyncio.run(run_semantic_test())