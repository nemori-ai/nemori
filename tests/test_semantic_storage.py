"""
Unit tests for semantic memory storage protocol.

This module tests the storage interface and exception handling for semantic memory,
ensuring the protocol is well-defined and implementations can be properly validated.
"""

from typing import Any

import pytest


from nemori.storage.repository import SemanticMemoryRepository
from nemori.storage.storage_types import (
    DuplicateKeyError,
    InvalidDataError,
    NotFoundError,
    SemanticStorageError,
)
from nemori.core.data_types import RelationshipType, SemanticNode, SemanticRelationship
class MockSemanticStorage(SemanticMemoryRepository):
    """Mock implementation of SemanticStorage for testing protocol adherence."""

    def __init__(self):
        from nemori.storage.storage_types import StorageConfig

        super().__init__(StorageConfig())
        self.nodes = {}  # node_id -> SemanticNode
        self.relationships = {}  # relationship_id -> SemanticRelationship
        self.owner_keys = {}  # (owner_id, key) -> node_id

    # Base repository methods
    async def initialize(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def health_check(self) -> bool:
        return True

    async def get_stats(self):
        from nemori.storage.storage_types import StorageStats

        return StorageStats()

    async def backup(self, destination: str) -> bool:
        # Mock implementation for testing
        return True

    async def restore(self, source: str) -> bool:
        # Mock implementation for testing
        return True

    # Missing abstract methods that need to be implemented
    async def search_semantic_nodes(self, query):
        from nemori.storage.storage_types import SemanticSearchResult

        # Simple implementation for testing
        return SemanticSearchResult()

    async def search_semantic_relationships(self, query):
        from nemori.storage.storage_types import SemanticSearchResult

        # Simple implementation for testing
        return SemanticSearchResult()

    async def store_semantic_node(self, node: SemanticNode) -> None:
        key = (node.owner_id, node.key)
        if key in self.owner_keys:
            raise DuplicateKeyError(f"Node with key {node.key} already exists for owner {node.owner_id}")

        self.nodes[node.node_id] = node
        self.owner_keys[key] = node.node_id

    async def get_semantic_node_by_id(self, node_id: str) -> SemanticNode | None:
        return self.nodes.get(node_id)

    async def find_semantic_node_by_key(self, owner_id: str, key: str) -> SemanticNode | None:
        node_id = self.owner_keys.get((owner_id, key))
        return self.nodes.get(node_id) if node_id else None

    async def update_semantic_node(self, node: SemanticNode) -> None:
        if node.node_id not in self.nodes:
            raise NotFoundError(f"Node {node.node_id} not found")

        # Update owner_keys mapping if key changed
        old_node = self.nodes[node.node_id]
        if old_node.key != node.key:
            old_key = (old_node.owner_id, old_node.key)
            new_key = (node.owner_id, node.key)
            del self.owner_keys[old_key]
            self.owner_keys[new_key] = node.node_id

        self.nodes[node.node_id] = node

    async def delete_semantic_node(self, node_id: str) -> bool:
        if node_id not in self.nodes:
            return False

        node = self.nodes[node_id]
        key = (node.owner_id, node.key)
        del self.nodes[node_id]
        del self.owner_keys[key]
        return True

    async def find_semantic_nodes_by_episode(self, episode_id: str) -> list[SemanticNode]:
        return [node for node in self.nodes.values() if node.discovery_episode_id == episode_id]

    async def find_semantic_nodes_by_linked_episode(self, episode_id: str) -> list[SemanticNode]:
        return [node for node in self.nodes.values() if episode_id in node.linked_episode_ids]

    async def similarity_search_semantic_nodes(self, owner_id: str, query: str, limit: int = 10) -> list[SemanticNode]:
        # Simple mock implementation: return nodes containing query text
        matching_nodes = []
        for node in self.nodes.values():
            if node.owner_id == owner_id and (
                query.lower() in node.key.lower()
                or query.lower() in node.value.lower()
                or query.lower() in node.context.lower()
            ):
                matching_nodes.append(node)
        return matching_nodes[:limit]

    async def store_semantic_relationship(self, relationship: SemanticRelationship) -> None:
        self.relationships[relationship.relationship_id] = relationship

    async def get_semantic_relationship_by_id(self, relationship_id: str) -> SemanticRelationship | None:
        return self.relationships.get(relationship_id)

    async def find_relationships_for_node(self, node_id: str) -> list[tuple[SemanticNode, SemanticRelationship]]:
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

    async def update_semantic_relationship(self, relationship: SemanticRelationship) -> None:
        if relationship.relationship_id not in self.relationships:
            raise NotFoundError(f"Relationship {relationship.relationship_id} not found")
        self.relationships[relationship.relationship_id] = relationship

    async def delete_semantic_relationship(self, relationship_id: str) -> bool:
        if relationship_id not in self.relationships:
            return False
        del self.relationships[relationship_id]
        return True

    async def get_semantic_nodes_by_ids(self, node_ids: list[str]) -> list[SemanticNode]:
        return [self.nodes[node_id] for node_id in node_ids if node_id in self.nodes]

    async def get_all_semantic_nodes_for_owner(self, owner_id: str) -> list[SemanticNode]:
        return [node for node in self.nodes.values() if node.owner_id == owner_id]

    async def get_semantic_statistics(self, owner_id: str) -> dict[str, Any]:
        owner_nodes = await self.get_all_semantic_nodes_for_owner(owner_id)
        owner_relationships = [
            rel
            for rel in self.relationships.values()
            if any(
                self.nodes.get(rel.source_node_id, {}).get("owner_id") == owner_id
                or self.nodes.get(rel.target_node_id, {}).get("owner_id") == owner_id
                for rel in [rel]
            )
        ]

        return {
            "total_nodes": len(owner_nodes),
            "total_relationships": len(owner_relationships),
            "average_confidence": sum(node.confidence for node in owner_nodes) / len(owner_nodes) if owner_nodes else 0,
            "total_access_count": sum(node.access_count for node in owner_nodes),
        }

    async def cleanup_orphaned_relationships(self) -> int:
        orphaned_count = 0
        to_delete = []

        for rel_id, rel in self.relationships.items():
            if rel.source_node_id not in self.nodes or rel.target_node_id not in self.nodes:
                to_delete.append(rel_id)
                orphaned_count += 1

        for rel_id in to_delete:
            del self.relationships[rel_id]

        return orphaned_count


class TestSemanticStorageProtocol:
    """Test suite for SemanticStorage protocol compliance."""

    @pytest.fixture
    def storage(self):
        """Create mock storage instance for testing."""
        return MockSemanticStorage()

    @pytest.fixture
    def sample_node(self):
        """Create a sample semantic node for testing."""
        return SemanticNode(
            owner_id="test_user",
            key="research_direction",
            value="AI Agent development",
            context="Discussion about future research plans",
        )

    @pytest.fixture
    def sample_relationship(self, sample_node):
        """Create a sample semantic relationship for testing."""
        return SemanticRelationship(
            source_node_id=sample_node.node_id,
            target_node_id="target_node_id",
            relationship_type=RelationshipType.RELATED,
            strength=0.8,
        )

    # === Semantic Node Tests ===

    @pytest.mark.asyncio
    async def test_store_semantic_node(self, storage, sample_node):
        """Test storing a semantic node."""
        await storage.store_semantic_node(sample_node)

        retrieved = await storage.get_semantic_node_by_id(sample_node.node_id)
        assert retrieved is not None
        assert retrieved.owner_id == sample_node.owner_id
        assert retrieved.key == sample_node.key
        assert retrieved.value == sample_node.value

    @pytest.mark.asyncio
    async def test_store_duplicate_key_raises_error(self, storage):
        """Test that storing duplicate (owner_id, key) raises DuplicateKeyError."""
        node1 = SemanticNode(owner_id="user1", key="research", value="LLM")
        node2 = SemanticNode(owner_id="user1", key="research", value="Agent")

        await storage.store_semantic_node(node1)

        with pytest.raises(DuplicateKeyError):
            await storage.store_semantic_node(node2)

    @pytest.mark.asyncio
    async def test_find_semantic_node_by_key(self, storage, sample_node):
        """Test finding semantic node by owner and key."""
        await storage.store_semantic_node(sample_node)

        found = await storage.find_semantic_node_by_key(sample_node.owner_id, sample_node.key)
        assert found is not None
        assert found.node_id == sample_node.node_id

    @pytest.mark.asyncio
    async def test_find_nonexistent_node_by_key(self, storage):
        """Test that finding nonexistent node returns None."""
        found = await storage.find_semantic_node_by_key("nonexistent", "key")
        assert found is None

    @pytest.mark.asyncio
    async def test_update_semantic_node(self, storage, sample_node):
        """Test updating an existing semantic node."""
        await storage.store_semantic_node(sample_node)

        updated_node = sample_node.evolve("Updated research direction", "New context", "episode_123")

        await storage.update_semantic_node(updated_node)

        retrieved = await storage.get_semantic_node_by_id(sample_node.node_id)
        assert retrieved.value == "Updated research direction"
        assert retrieved.version == 2

    @pytest.mark.asyncio
    async def test_update_nonexistent_node_raises_error(self, storage, sample_node):
        """Test that updating nonexistent node raises NotFoundError."""
        with pytest.raises(NotFoundError):
            await storage.update_semantic_node(sample_node)

    @pytest.mark.asyncio
    async def test_delete_semantic_node(self, storage, sample_node):
        """Test deleting a semantic node."""
        await storage.store_semantic_node(sample_node)

        deleted = await storage.delete_semantic_node(sample_node.node_id)
        assert deleted is True

        retrieved = await storage.get_semantic_node_by_id(sample_node.node_id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_node(self, storage):
        """Test that deleting nonexistent node returns False."""
        deleted = await storage.delete_semantic_node("nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_find_nodes_by_episode(self, storage):
        """Test finding semantic nodes by discovery episode."""
        node1 = SemanticNode(owner_id="user1", key="key1", value="value1", discovery_episode_id="episode_123")
        node2 = SemanticNode(owner_id="user1", key="key2", value="value2", discovery_episode_id="episode_123")
        node3 = SemanticNode(owner_id="user1", key="key3", value="value3", discovery_episode_id="episode_456")

        await storage.store_semantic_node(node1)
        await storage.store_semantic_node(node2)
        await storage.store_semantic_node(node3)

        found = await storage.find_semantic_nodes_by_episode("episode_123")
        assert len(found) == 2
        assert all(node.discovery_episode_id == "episode_123" for node in found)

    @pytest.mark.asyncio
    async def test_find_nodes_by_linked_episode(self, storage):
        """Test finding semantic nodes by linked episode."""
        node1 = SemanticNode(
            owner_id="user1", key="key1", value="value1", linked_episode_ids=["episode_123", "episode_456"]
        )
        node2 = SemanticNode(owner_id="user1", key="key2", value="value2", linked_episode_ids=["episode_123"])
        node3 = SemanticNode(owner_id="user1", key="key3", value="value3", linked_episode_ids=["episode_789"])

        await storage.store_semantic_node(node1)
        await storage.store_semantic_node(node2)
        await storage.store_semantic_node(node3)

        found = await storage.find_semantic_nodes_by_linked_episode("episode_123")
        assert len(found) == 2
        assert all("episode_123" in node.linked_episode_ids for node in found)

    @pytest.mark.asyncio
    async def test_similarity_search_semantic_nodes(self, storage):
        """Test similarity search for semantic nodes."""
        node1 = SemanticNode(owner_id="user1", key="AI research", value="machine learning algorithms")
        node2 = SemanticNode(owner_id="user1", key="database design", value="SQL optimization")
        node3 = SemanticNode(owner_id="user2", key="AI research", value="neural networks")

        await storage.store_semantic_node(node1)
        await storage.store_semantic_node(node2)
        await storage.store_semantic_node(node3)

        # Search for user1 with AI-related query
        results = await storage.similarity_search_semantic_nodes("user1", "AI", 10)
        assert len(results) == 1
        assert results[0].key == "AI research"
        assert results[0].owner_id == "user1"

    # === Semantic Relationship Tests ===

    @pytest.mark.asyncio
    async def test_store_semantic_relationship(self, storage, sample_relationship):
        """Test storing a semantic relationship."""
        await storage.store_semantic_relationship(sample_relationship)

        retrieved = await storage.get_semantic_relationship_by_id(sample_relationship.relationship_id)
        assert retrieved is not None
        assert retrieved.source_node_id == sample_relationship.source_node_id
        assert retrieved.target_node_id == sample_relationship.target_node_id

    @pytest.mark.asyncio
    async def test_find_relationships_for_node(self, storage):
        """Test finding relationships for a specific node."""
        # Create nodes
        node1 = SemanticNode(owner_id="user1", key="key1", value="value1")
        node2 = SemanticNode(owner_id="user1", key="key2", value="value2")
        node3 = SemanticNode(owner_id="user1", key="key3", value="value3")

        await storage.store_semantic_node(node1)
        await storage.store_semantic_node(node2)
        await storage.store_semantic_node(node3)

        # Create relationships
        rel1 = SemanticRelationship(
            source_node_id=node1.node_id, target_node_id=node2.node_id, relationship_type=RelationshipType.RELATED
        )
        rel2 = SemanticRelationship(
            source_node_id=node3.node_id, target_node_id=node1.node_id, relationship_type=RelationshipType.SIMILAR_TO
        )

        await storage.store_semantic_relationship(rel1)
        await storage.store_semantic_relationship(rel2)

        # Find relationships for node1
        relationships = await storage.find_relationships_for_node(node1.node_id)
        assert len(relationships) == 2

        # Check that we get the correct related nodes
        related_node_ids = {node.node_id for node, _ in relationships}
        assert node2.node_id in related_node_ids
        assert node3.node_id in related_node_ids

    @pytest.mark.asyncio
    async def test_update_semantic_relationship(self, storage, sample_relationship):
        """Test updating an existing semantic relationship."""
        await storage.store_semantic_relationship(sample_relationship)

        updated_relationship = sample_relationship.reinforce(0.1)
        await storage.update_semantic_relationship(updated_relationship)

        retrieved = await storage.get_semantic_relationship_by_id(sample_relationship.relationship_id)
        assert retrieved.strength == updated_relationship.strength

    @pytest.mark.asyncio
    async def test_delete_semantic_relationship(self, storage, sample_relationship):
        """Test deleting a semantic relationship."""
        await storage.store_semantic_relationship(sample_relationship)

        deleted = await storage.delete_semantic_relationship(sample_relationship.relationship_id)
        assert deleted is True

        retrieved = await storage.get_semantic_relationship_by_id(sample_relationship.relationship_id)
        assert retrieved is None

    # === Bulk Operations Tests ===

    @pytest.mark.asyncio
    async def test_get_semantic_nodes_by_ids(self, storage):
        """Test retrieving multiple semantic nodes by IDs."""
        nodes = [SemanticNode(owner_id="user1", key=f"key{i}", value=f"value{i}") for i in range(3)]

        for node in nodes:
            await storage.store_semantic_node(node)

        node_ids = [node.node_id for node in nodes[:2]]
        retrieved = await storage.get_semantic_nodes_by_ids(node_ids)

        assert len(retrieved) == 2
        retrieved_ids = {node.node_id for node in retrieved}
        assert retrieved_ids == set(node_ids)

    @pytest.mark.asyncio
    async def test_get_all_semantic_nodes_for_owner(self, storage):
        """Test retrieving all semantic nodes for an owner."""
        # Create nodes for different owners
        user1_nodes = [SemanticNode(owner_id="user1", key=f"key{i}", value=f"value{i}") for i in range(3)]
        user2_nodes = [SemanticNode(owner_id="user2", key=f"key{i}", value=f"value{i}") for i in range(2)]

        for node in user1_nodes + user2_nodes:
            await storage.store_semantic_node(node)

        user1_results = await storage.get_all_semantic_nodes_for_owner("user1")
        assert len(user1_results) == 3
        assert all(node.owner_id == "user1" for node in user1_results)

    # === Statistics and Maintenance Tests ===

    @pytest.mark.asyncio
    async def test_get_semantic_statistics(self, storage):
        """Test retrieving semantic memory statistics."""
        nodes = [
            SemanticNode(
                owner_id="user1", key=f"key{i}", value=f"value{i}", confidence=0.5 + (i * 0.1), access_count=i * 2
            )
            for i in range(3)
        ]

        for node in nodes:
            await storage.store_semantic_node(node)

        stats = await storage.get_semantic_statistics("user1")

        assert stats["total_nodes"] == 3
        assert stats["average_confidence"] == pytest.approx(0.6)  # (0.5 + 0.6 + 0.7) / 3
        assert stats["total_access_count"] == 6  # 0 + 2 + 4

    @pytest.mark.asyncio
    async def test_cleanup_orphaned_relationships(self, storage):
        """Test cleaning up relationships that reference non-existent nodes."""
        # Create a node and relationship
        node1 = SemanticNode(owner_id="user1", key="key1", value="value1")
        await storage.store_semantic_node(node1)

        relationship = SemanticRelationship(
            source_node_id=node1.node_id,
            target_node_id="nonexistent_node",  # This will be orphaned
            relationship_type=RelationshipType.RELATED,
        )
        await storage.store_semantic_relationship(relationship)

        # Cleanup orphaned relationships
        cleaned_count = await storage.cleanup_orphaned_relationships()

        assert cleaned_count == 1
        retrieved = await storage.get_semantic_relationship_by_id(relationship.relationship_id)
        assert retrieved is None


class TestSemanticStorageExceptions:
    """Test suite for semantic storage exceptions."""

    def test_semantic_storage_error(self):
        """Test base SemanticStorageError exception."""
        error = SemanticStorageError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_duplicate_key_error(self):
        """Test DuplicateKeyError exception."""
        error = DuplicateKeyError("Duplicate key")
        assert str(error) == "Duplicate key"
        assert isinstance(error, SemanticStorageError)

    def test_not_found_error(self):
        """Test NotFoundError exception."""
        error = NotFoundError("Not found")
        assert str(error) == "Not found"
        assert isinstance(error, SemanticStorageError)

    def test_invalid_data_error(self):
        """Test InvalidDataError exception."""
        error = InvalidDataError("Invalid data")
        assert str(error) == "Invalid data"
        assert isinstance(error, SemanticStorageError)
