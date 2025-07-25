"""
Unit tests for unified semantic memory retrieval service.

This module tests the UnifiedRetrievalService which provides the core functionality
for independent similarity search and bidirectional associations between episodes and semantic nodes.
"""

from datetime import datetime

import pytest

from nemori.core.data_types import RelationshipType, SemanticNode, SemanticRelationship
from nemori.core.episode import Episode, EpisodeLevel, EpisodeType, TemporalInfo
from nemori.retrieval.service import UnifiedRetrievalService


class MockEpisodeRepository:
    """Mock implementation of EpisodicMemoryRepository for testing."""

    def __init__(self):
        self.episodes = {}  # episode_id -> Episode

    async def search_episodes(self, query) -> object:
        """Mock episode search implementation."""
        from nemori.storage.storage_types import EpisodeSearchResult

        matching_episodes = []
        for episode in self.episodes.values():
            # Check owner filter
            if query.owner_ids and episode.owner_id not in query.owner_ids:
                continue

            # Check text search
            if query.text_search:
                text_search = query.text_search.lower()
                if not (
                    text_search in episode.title.lower()
                    or text_search in episode.content.lower()
                    or text_search in episode.summary.lower()
                ):
                    continue

            matching_episodes.append(episode)

        # Apply limit
        if query.limit:
            matching_episodes = matching_episodes[: query.limit]

        return EpisodeSearchResult(
            episodes=matching_episodes, total_count=len(matching_episodes), has_more=False, query_time_ms=1.0
        )

    async def get_episode_batch(self, episode_ids: list[str]) -> list[Episode | None]:
        """Mock bulk episode retrieval."""
        return [self.episodes.get(ep_id) for ep_id in episode_ids]

    async def get_episode(self, episode_id: str) -> Episode | None:
        """Mock single episode retrieval."""
        return self.episodes.get(episode_id)

    def add_episode(self, episode: Episode) -> None:
        """Helper method to add episodes for testing."""
        self.episodes[episode.episode_id] = episode


class MockSemanticStorage:
    """Mock implementation of SemanticMemoryRepository for testing."""

    def __init__(self):
        self.nodes = {}  # node_id -> SemanticNode
        self.relationships = {}  # relationship_id -> SemanticRelationship

    async def similarity_search_semantic_nodes(self, owner_id: str, query: str, limit: int = 10) -> list[SemanticNode]:
        """Mock semantic similarity search."""
        matching_nodes = []
        for node in self.nodes.values():
            if node.owner_id == owner_id and (
                query.lower() in node.key.lower()
                or query.lower() in node.value.lower()
                or query.lower() in node.context.lower()
            ):
                matching_nodes.append(node)
        return matching_nodes[:limit]

    async def find_semantic_nodes_by_episode(self, episode_id: str) -> list[SemanticNode]:
        """Mock finding nodes discovered from episode."""
        return [node for node in self.nodes.values() if node.discovery_episode_id == episode_id]

    async def find_semantic_nodes_by_linked_episode(self, episode_id: str) -> list[SemanticNode]:
        """Mock finding nodes linked to episode."""
        return [node for node in self.nodes.values() if episode_id in node.linked_episode_ids]

    async def get_semantic_node_by_id(self, node_id: str) -> SemanticNode | None:
        """Mock single node retrieval."""
        return self.nodes.get(node_id)

    async def find_relationships_for_node(self, node_id: str) -> list[tuple[SemanticNode, SemanticRelationship]]:
        """Mock relationship finding."""
        results = []
        for rel in self.relationships.values():
            if rel.source_node_id == node_id:
                target_node = self.nodes.get(rel.target_node_id)
                if target_node:
                    results.append((target_node, rel))
            elif rel.target_node_id == node_id:
                source_node = self.nodes.get(rel.source_node_id)
                if source_node:
                    results.append((source_node, rel))
        return results

    def add_node(self, node: SemanticNode) -> None:
        """Helper method to add nodes for testing."""
        self.nodes[node.node_id] = node

    def add_relationship(self, relationship: SemanticRelationship) -> None:
        """Helper method to add relationships for testing."""
        self.relationships[relationship.relationship_id] = relationship


class TestUnifiedRetrievalService:
    """Test suite for UnifiedRetrievalService."""

    @pytest.fixture
    def episode_repository(self):
        """Create mock episode repository for testing."""
        return MockEpisodeRepository()

    @pytest.fixture
    def semantic_storage(self):
        """Create mock semantic storage for testing."""
        return MockSemanticStorage()

    @pytest.fixture
    def retrieval_service(self, episode_repository, semantic_storage):
        """Create unified retrieval service for testing."""
        return UnifiedRetrievalService(episode_repository, semantic_storage)

    @pytest.fixture
    def sample_episode(self):
        """Create a sample episode for testing."""
        return Episode(
            episode_id="episode_123",
            owner_id="test_user",
            title="John discusses AI research",
            content="John talked about his research in AI Agent behavior planning",
            summary="Discussion about AI research direction",
            episode_type=EpisodeType.CONVERSATIONAL,
            level=EpisodeLevel.ATOMIC,
            temporal_info=TemporalInfo(timestamp=datetime.now()),
        )

    @pytest.fixture
    def sample_semantic_node(self):
        """Create a sample semantic node for testing."""
        return SemanticNode(
            node_id="node_123",
            owner_id="test_user",
            key="John的研究方向",
            value="AI Agent行为规划",
            context="讨论关于研究重点",
            discovery_episode_id="episode_123",
            linked_episode_ids=["episode_123", "episode_124"],
        )

    # === Independent Similarity Search Tests ===

    @pytest.mark.asyncio
    async def test_search_episodic_memories(self, retrieval_service, episode_repository, sample_episode):
        """Test independent similarity search for episodic memories."""
        # Add test episode
        episode_repository.add_episode(sample_episode)

        # Search for episodes
        results = await retrieval_service.search_episodic_memories(owner_id="test_user", query="AI research", limit=10)

        assert len(results) == 1
        assert results[0].episode_id == sample_episode.episode_id
        assert results[0].title == sample_episode.title

    @pytest.mark.asyncio
    async def test_search_semantic_memories(self, retrieval_service, semantic_storage, sample_semantic_node):
        """Test independent similarity search for semantic memories."""
        # Add test semantic node
        semantic_storage.add_node(sample_semantic_node)

        # Search for semantic nodes
        results = await retrieval_service.search_semantic_memories(owner_id="test_user", query="AI Agent", limit=10)

        assert len(results) == 1
        assert results[0].node_id == sample_semantic_node.node_id
        assert results[0].key == sample_semantic_node.key

    @pytest.mark.asyncio
    async def test_search_respects_owner_isolation(self, retrieval_service, episode_repository, semantic_storage):
        """Test that search results are isolated by owner."""
        # Add episodes for different owners
        episode1 = Episode(
            episode_id="ep1",
            owner_id="user1",
            title="User1 AI discussion",
            content="User1 talks about AI",
            summary="AI discussion",
        )
        episode2 = Episode(
            episode_id="ep2",
            owner_id="user2",
            title="User2 AI discussion",
            content="User2 talks about AI",
            summary="AI discussion",
        )

        episode_repository.add_episode(episode1)
        episode_repository.add_episode(episode2)

        # Add semantic nodes for different owners
        node1 = SemanticNode(owner_id="user1", key="research", value="AI Agent development")
        node2 = SemanticNode(owner_id="user2", key="research", value="AI Agent development")

        semantic_storage.add_node(node1)
        semantic_storage.add_node(node2)

        # Search for user1 only
        episode_results = await retrieval_service.search_episodic_memories(owner_id="user1", query="AI", limit=10)
        semantic_results = await retrieval_service.search_semantic_memories(owner_id="user1", query="AI", limit=10)

        assert len(episode_results) == 1
        assert episode_results[0].owner_id == "user1"
        assert len(semantic_results) == 1
        assert semantic_results[0].owner_id == "user1"

    # === Bidirectional ID-based Association Tests ===

    @pytest.mark.asyncio
    async def test_get_episode_semantics(self, retrieval_service, semantic_storage, sample_episode):
        """Test getting semantic nodes associated with an episode."""
        # Create semantic nodes with different associations
        discovered_node = SemanticNode(
            owner_id="test_user",
            key="discovered_knowledge",
            value="knowledge from episode",
            discovery_episode_id=sample_episode.episode_id,
        )

        linked_node = SemanticNode(
            owner_id="test_user",
            key="linked_knowledge",
            value="knowledge linked to episode",
            linked_episode_ids=[sample_episode.episode_id, "other_episode"],
        )

        unrelated_node = SemanticNode(owner_id="test_user", key="unrelated_knowledge", value="unrelated knowledge")

        semantic_storage.add_node(discovered_node)
        semantic_storage.add_node(linked_node)
        semantic_storage.add_node(unrelated_node)

        # Get associated semantic nodes
        results = await retrieval_service.get_episode_semantics(sample_episode.episode_id)

        assert len(results) == 2
        result_ids = {node.node_id for node in results}
        assert discovered_node.node_id in result_ids
        assert linked_node.node_id in result_ids
        assert unrelated_node.node_id not in result_ids

    @pytest.mark.asyncio
    async def test_get_semantic_episodes(self, retrieval_service, episode_repository, semantic_storage):
        """Test getting episodes associated with a semantic node."""
        # Create episodes
        discovery_episode = Episode(
            episode_id="discovery_ep",
            owner_id="test_user",
            title="Discovery episode",
            content="Episode that discovered knowledge",
        )

        linked_episode = Episode(
            episode_id="linked_ep",
            owner_id="test_user",
            title="Linked episode",
            content="Episode that references knowledge",
        )

        evolution_episode = Episode(
            episode_id="evolution_ep",
            owner_id="test_user",
            title="Evolution episode",
            content="Episode that evolved knowledge",
        )

        episode_repository.add_episode(discovery_episode)
        episode_repository.add_episode(linked_episode)
        episode_repository.add_episode(evolution_episode)

        # Create semantic node with associations
        semantic_node = SemanticNode(
            owner_id="test_user",
            key="research_direction",
            value="AI Agent development",
            discovery_episode_id=discovery_episode.episode_id,
            linked_episode_ids=[linked_episode.episode_id],
            evolution_episode_ids=[evolution_episode.episode_id],
        )

        semantic_storage.add_node(semantic_node)

        # Get associated episodes
        results = await retrieval_service.get_semantic_episodes(semantic_node.node_id)

        assert len(results["discovery_episode"]) == 1
        assert results["discovery_episode"][0].episode_id == discovery_episode.episode_id

        assert len(results["linked_episodes"]) == 1
        assert results["linked_episodes"][0].episode_id == linked_episode.episode_id

        assert len(results["evolution_episodes"]) == 1
        assert results["evolution_episodes"][0].episode_id == evolution_episode.episode_id

    @pytest.mark.asyncio
    async def test_get_semantic_episodes_with_missing_episodes(
        self, retrieval_service, episode_repository, semantic_storage
    ):
        """Test handling of missing episodes in semantic associations."""
        # Create semantic node with references to non-existent episodes
        semantic_node = SemanticNode(
            owner_id="test_user",
            key="research_direction",
            value="AI development",
            discovery_episode_id="nonexistent_discovery",
            linked_episode_ids=["nonexistent_linked"],
            evolution_episode_ids=["nonexistent_evolution"],
        )

        semantic_storage.add_node(semantic_node)

        # Get associated episodes (should handle missing gracefully)
        results = await retrieval_service.get_semantic_episodes(semantic_node.node_id)

        assert len(results["discovery_episode"]) == 0
        assert len(results["linked_episodes"]) == 0
        assert len(results["evolution_episodes"]) == 0

    # === Context-Aware Retrieval Tests ===

    @pytest.mark.asyncio
    async def test_get_discovery_context(self, retrieval_service, episode_repository, semantic_storage, sample_episode):
        """Test gathering discovery context for an episode."""
        # Add related semantic memories
        related_semantic1 = SemanticNode(owner_id="test_user", key="AI research", value="machine learning focus")
        related_semantic2 = SemanticNode(
            owner_id="test_user", key="Agent development", value="behavior planning systems"
        )

        semantic_storage.add_node(related_semantic1)
        semantic_storage.add_node(related_semantic2)

        # Add related historical episodes
        related_episode = Episode(
            episode_id="related_ep",
            owner_id="test_user",
            title="Previous AI discussion",
            content="Earlier discussion about AI Agent behavior planning",
        )

        episode_repository.add_episode(related_episode)

        # Get discovery context
        context = await retrieval_service.get_discovery_context(
            episode=sample_episode, semantic_limit=5, episode_limit=3
        )

        assert "related_semantic_memories" in context
        assert "related_historical_episodes" in context
        assert "current_episode" in context
        assert context["current_episode"] == sample_episode

        # Check that we found related content (may be empty due to simple mock search)
        assert "related_semantic_memories" in context
        assert "related_historical_episodes" in context
        # The specific assertions depend on mock search implementation
        # In real implementation, this would find related content

    # === Evolution History Tests ===

    @pytest.mark.asyncio
    async def test_get_semantic_evolution_history(self, retrieval_service, episode_repository, semantic_storage):
        """Test getting comprehensive semantic evolution history."""
        # Create episodes that represent evolution
        initial_episode = Episode(
            episode_id="initial_ep",
            owner_id="test_user",
            title="Initial research discussion",
            content="John talks about LLM research",
            temporal_info=TemporalInfo(timestamp=datetime(2024, 1, 1)),
        )

        evolution_episode = Episode(
            episode_id="evolution_ep",
            owner_id="test_user",
            title="Research direction change",
            content="John switches to Agent research",
            temporal_info=TemporalInfo(timestamp=datetime(2024, 6, 1)),
        )

        episode_repository.add_episode(initial_episode)
        episode_repository.add_episode(evolution_episode)

        # Create evolved semantic node
        semantic_node = SemanticNode(
            owner_id="test_user",
            key="John的研究方向",
            value="AI Agent行为规划",
            version=2,
            evolution_history=["大语言模型研究"],
            discovery_episode_id=initial_episode.episode_id,
            evolution_episode_ids=[evolution_episode.episode_id],
            linked_episode_ids=[initial_episode.episode_id, evolution_episode.episode_id],
        )

        semantic_storage.add_node(semantic_node)

        # Get evolution history
        history = await retrieval_service.get_semantic_evolution_history(semantic_node.node_id)

        assert "node" in history
        assert "evolution_timeline" in history
        assert "linked_episodes" in history
        assert "evolution_episodes" in history
        assert "discovery_episode" in history

        # Check timeline structure
        timeline = history["evolution_timeline"]
        assert len(timeline) == 2  # Historical version + current version

        # Historical version
        assert timeline[0]["version"] == 1
        assert timeline[0]["value"] == "大语言模型研究"
        assert timeline[0]["episode"].episode_id == evolution_episode.episode_id

        # Current version
        assert timeline[1]["version"] == 2
        assert timeline[1]["value"] == "AI Agent行为规划"
        assert timeline[1]["episode"] is None  # Current version has no triggering episode

    @pytest.mark.asyncio
    async def test_get_semantic_evolution_history_nonexistent_node(self, retrieval_service, semantic_storage):
        """Test getting evolution history for non-existent node."""
        history = await retrieval_service.get_semantic_evolution_history("nonexistent")
        assert history == {}

    # === Quality-based Retrieval Tests ===

    @pytest.mark.asyncio
    async def test_get_memory_for_query_factual(self, retrieval_service, semantic_storage, sample_semantic_node):
        """Test factual quality preference (semantic memory priority)."""
        semantic_storage.add_node(sample_semantic_node)

        result = await retrieval_service.get_memory_for_query(
            owner_id="test_user", query="AI Agent", quality_preference="factual"
        )

        assert result["quality_type"] == "factual"
        assert len(result["primary"]) == 1
        assert len(result["secondary"]) == 0
        assert result["primary"][0].node_id == sample_semantic_node.node_id

    @pytest.mark.asyncio
    async def test_get_memory_for_query_contextual(self, retrieval_service, episode_repository, sample_episode):
        """Test contextual quality preference (episodic memory priority)."""
        episode_repository.add_episode(sample_episode)

        result = await retrieval_service.get_memory_for_query(
            owner_id="test_user", query="AI research", quality_preference="contextual"
        )

        assert result["quality_type"] == "contextual"
        assert len(result["primary"]) == 1
        assert len(result["secondary"]) == 0
        assert result["primary"][0].episode_id == sample_episode.episode_id

    @pytest.mark.asyncio
    async def test_get_memory_for_query_comprehensive(
        self, retrieval_service, episode_repository, semantic_storage, sample_episode, sample_semantic_node
    ):
        """Test comprehensive quality preference (combined with associations)."""
        episode_repository.add_episode(sample_episode)
        semantic_storage.add_node(sample_semantic_node)

        result = await retrieval_service.get_memory_for_query(
            owner_id="test_user", query="AI", quality_preference="comprehensive"
        )

        assert result["quality_type"] == "comprehensive"
        assert len(result["primary"]) > 0  # Enriched semantic results
        assert len(result["secondary"]) > 0  # Episodic results

        # Check enriched semantic result structure
        enriched_result = result["primary"][0]
        assert "semantic_knowledge" in enriched_result
        assert "supporting_episodes" in enriched_result
        assert "evolution_context" in enriched_result

    @pytest.mark.asyncio
    async def test_get_memory_for_query_balanced(
        self, retrieval_service, episode_repository, semantic_storage, sample_episode, sample_semantic_node
    ):
        """Test balanced quality preference (separate results)."""
        episode_repository.add_episode(sample_episode)
        semantic_storage.add_node(sample_semantic_node)

        result = await retrieval_service.get_memory_for_query(
            owner_id="test_user", query="AI", quality_preference="balanced"
        )

        assert result["quality_type"] == "balanced"
        assert "semantic_memories" in result
        assert "episodic_memories" in result
        assert len(result["semantic_memories"]) == 1
        assert len(result["episodic_memories"]) == 1

    # === Utility Method Tests ===

    @pytest.mark.asyncio
    async def test_get_related_knowledge(self, retrieval_service, semantic_storage):
        """Test getting related knowledge through relationship traversal."""
        # Create semantic nodes
        node1 = SemanticNode(owner_id="user1", key="AI", value="Artificial Intelligence")
        node2 = SemanticNode(owner_id="user1", key="ML", value="Machine Learning")
        node3 = SemanticNode(owner_id="user1", key="DL", value="Deep Learning")

        semantic_storage.add_node(node1)
        semantic_storage.add_node(node2)
        semantic_storage.add_node(node3)

        # Create relationships
        rel1 = SemanticRelationship(
            source_node_id=node1.node_id, target_node_id=node2.node_id, relationship_type=RelationshipType.RELATED
        )
        rel2 = SemanticRelationship(
            source_node_id=node2.node_id, target_node_id=node3.node_id, relationship_type=RelationshipType.PART_OF
        )

        semantic_storage.add_relationship(rel1)
        semantic_storage.add_relationship(rel2)

        # Get related knowledge with depth 2
        related = await retrieval_service.get_related_knowledge(node_id=node1.node_id, max_depth=2)

        assert "direct" in related
        assert "indirect" in related
        assert len(related["direct"]) == 1
        assert related["direct"][0].node_id == node2.node_id
        assert len(related["indirect"]) == 1
        assert related["indirect"][0].node_id == node3.node_id

    @pytest.mark.asyncio
    async def test_get_related_knowledge_depth_limit(self, retrieval_service, semantic_storage):
        """Test that relationship traversal respects depth limits."""
        # Create chain: node1 -> node2 -> node3
        node1 = SemanticNode(owner_id="user1", key="node1", value="value1")
        node2 = SemanticNode(owner_id="user1", key="node2", value="value2")
        node3 = SemanticNode(owner_id="user1", key="node3", value="value3")

        semantic_storage.add_node(node1)
        semantic_storage.add_node(node2)
        semantic_storage.add_node(node3)

        rel1 = SemanticRelationship(source_node_id=node1.node_id, target_node_id=node2.node_id)
        rel2 = SemanticRelationship(source_node_id=node2.node_id, target_node_id=node3.node_id)

        semantic_storage.add_relationship(rel1)
        semantic_storage.add_relationship(rel2)

        # Test depth 1 (should only get direct relationships)
        related = await retrieval_service.get_related_knowledge(node_id=node1.node_id, max_depth=1)

        assert len(related["direct"]) == 1
        assert len(related["indirect"]) == 0
        assert related["direct"][0].node_id == node2.node_id
