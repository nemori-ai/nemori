"""
Unit tests for semantic memory data types.

This module tests the core data structures for semantic memory including
SemanticNode and SemanticRelationship classes.
"""

from datetime import datetime

import pytest

from nemori.core.data_types import RelationshipType, SemanticNode, SemanticRelationship


class TestSemanticNode:
    """Test suite for SemanticNode data structure."""

    def test_semantic_node_creation_with_defaults(self):
        """Test creating a semantic node with minimum required fields."""
        node = SemanticNode(owner_id="test_user", key="test_key", value="test_value")

        assert node.owner_id == "test_user"
        assert node.key == "test_key"
        assert node.value == "test_value"
        assert node.confidence == 1.0
        assert node.version == 1
        assert len(node.evolution_history) == 0
        assert len(node.linked_episode_ids) == 0
        assert len(node.evolution_episode_ids) == 0
        assert node.discovery_method == "diff_analysis"
        assert node.access_count == 0

    def test_semantic_node_creation_with_all_fields(self):
        """Test creating a semantic node with all fields specified."""
        now = datetime.now()
        node = SemanticNode(
            node_id="custom_id",
            owner_id="test_user",
            key="research_direction",
            value="AI Agent behavior planning",
            context="Discussion about research focus",
            confidence=0.8,
            version=2,
            evolution_history=["LLM prompt engineering"],
            created_at=now,
            last_updated=now,
            last_accessed=now,
            discovery_episode_id="episode_123",
            discovery_method="custom_analysis",
            linked_episode_ids=["episode_123", "episode_124"],
            evolution_episode_ids=["episode_124"],
            search_keywords=["AI", "Agent", "research"],
            embedding_vector=[0.1, 0.2, 0.3],
            access_count=5,
            relevance_score=0.9,
            importance_score=0.7,
        )

        assert node.node_id == "custom_id"
        assert node.owner_id == "test_user"
        assert node.key == "research_direction"
        assert node.value == "AI Agent behavior planning"
        assert node.context == "Discussion about research focus"
        assert node.confidence == 0.8
        assert node.version == 2
        assert node.evolution_history == ["LLM prompt engineering"]
        assert node.created_at == now
        assert node.last_updated == now
        assert node.last_accessed == now
        assert node.discovery_episode_id == "episode_123"
        assert node.discovery_method == "custom_analysis"
        assert node.linked_episode_ids == ["episode_123", "episode_124"]
        assert node.evolution_episode_ids == ["episode_124"]
        assert node.search_keywords == ["AI", "Agent", "research"]
        assert node.embedding_vector == [0.1, 0.2, 0.3]
        assert node.access_count == 5
        assert node.relevance_score == 0.9
        assert node.importance_score == 0.7

    def test_semantic_node_validation_missing_owner_id(self):
        """Test that missing owner_id raises ValueError."""
        with pytest.raises(ValueError, match="owner_id is required"):
            SemanticNode(key="test_key", value="test_value")

    def test_semantic_node_validation_missing_key(self):
        """Test that missing key raises ValueError."""
        with pytest.raises(ValueError, match="key is required"):
            SemanticNode(owner_id="test_user", value="test_value")

    def test_semantic_node_validation_missing_value(self):
        """Test that missing value raises ValueError."""
        with pytest.raises(ValueError, match="value is required"):
            SemanticNode(owner_id="test_user", key="test_key")

    def test_semantic_node_validation_invalid_confidence(self):
        """Test that invalid confidence values raise ValueError."""
        with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
            SemanticNode(owner_id="test_user", key="test_key", value="test_value", confidence=1.5)

        with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
            SemanticNode(owner_id="test_user", key="test_key", value="test_value", confidence=-0.1)

    def test_semantic_node_validation_invalid_version(self):
        """Test that invalid version values raise ValueError."""
        with pytest.raises(ValueError, match="version must be positive"):
            SemanticNode(owner_id="test_user", key="test_key", value="test_value", version=0)

    def test_semantic_node_to_dict(self):
        """Test converting semantic node to dictionary."""
        now = datetime.now()
        node = SemanticNode(
            owner_id="test_user",
            key="test_key",
            value="test_value",
            created_at=now,
            last_updated=now,
            last_accessed=now,
        )

        result = node.to_dict()

        assert result["owner_id"] == "test_user"
        assert result["key"] == "test_key"
        assert result["value"] == "test_value"
        assert result["created_at"] == now.isoformat()
        assert result["last_updated"] == now.isoformat()
        assert result["last_accessed"] == now.isoformat()
        assert "node_id" in result
        assert isinstance(result["linked_episode_ids"], list)

    def test_semantic_node_from_dict(self):
        """Test creating semantic node from dictionary."""
        now = datetime.now()
        data = {
            "owner_id": "test_user",
            "key": "test_key",
            "value": "test_value",
            "confidence": 0.8,
            "version": 2,
            "evolution_history": ["old_value"],
            "created_at": now.isoformat(),
            "last_updated": now.isoformat(),
            "last_accessed": now.isoformat(),
            "linked_episode_ids": ["episode_1"],
            "search_keywords": ["keyword1", "keyword2"],
        }

        node = SemanticNode.from_dict(data)

        assert node.owner_id == "test_user"
        assert node.key == "test_key"
        assert node.value == "test_value"
        assert node.confidence == 0.8
        assert node.version == 2
        assert node.evolution_history == ["old_value"]
        assert node.linked_episode_ids == ["episode_1"]
        assert node.search_keywords == ["keyword1", "keyword2"]

    def test_semantic_node_mark_accessed(self):
        """Test marking semantic node as accessed."""
        now = datetime.now()
        node = SemanticNode(owner_id="test_user", key="test_key", value="test_value", access_count=5, last_accessed=now)

        accessed_node = node.mark_accessed()

        assert accessed_node.access_count == 6
        assert accessed_node.last_accessed is not None
        assert accessed_node.last_accessed >= node.last_accessed
        # Original node should be unchanged (immutable operation)
        assert node.access_count == 5

    def test_semantic_node_evolve(self):
        """Test evolving semantic node to new version."""
        original_node = SemanticNode(
            owner_id="test_user", key="research_direction", value="LLM research", version=1, evolution_history=[]
        )

        evolved_node = original_node.evolve(
            new_value="AI Agent research", new_context="New research focus", evolution_episode_id="episode_123"
        )

        assert evolved_node.value == "AI Agent research"
        assert evolved_node.context == "New research focus"
        assert evolved_node.version == 2
        assert evolved_node.evolution_history == ["LLM research"]
        assert evolved_node.evolution_episode_ids == ["episode_123"]
        assert evolved_node.last_updated > original_node.last_updated
        # Original node should be unchanged
        assert original_node.version == 1
        assert original_node.value == "LLM research"

    def test_semantic_node_add_linked_episode(self):
        """Test adding linked episode to semantic node."""
        node = SemanticNode(owner_id="test_user", key="test_key", value="test_value", linked_episode_ids=["episode_1"])

        # Add new episode
        updated_node = node.add_linked_episode("episode_2")
        assert updated_node.linked_episode_ids == ["episode_1", "episode_2"]

        # Try to add existing episode (should not duplicate)
        same_node = updated_node.add_linked_episode("episode_1")
        assert same_node.linked_episode_ids == ["episode_1", "episode_2"]

    def test_semantic_node_update_confidence(self):
        """Test updating confidence score within bounds."""
        node = SemanticNode(owner_id="test_user", key="test_key", value="test_value", confidence=0.5)

        # Increase confidence
        increased = node.update_confidence(0.3)
        assert increased.confidence == 0.8

        # Try to exceed upper bound
        maxed = increased.update_confidence(0.5)
        assert maxed.confidence == 1.0

        # Decrease confidence
        decreased = node.update_confidence(-0.2)
        assert decreased.confidence == 0.3

        # Try to go below lower bound
        minimal = decreased.update_confidence(-1.0)
        assert minimal.confidence == 0.0


class TestSemanticRelationship:
    """Test suite for SemanticRelationship data structure."""

    def test_semantic_relationship_creation_with_defaults(self):
        """Test creating semantic relationship with minimum required fields."""
        relationship = SemanticRelationship(source_node_id="node_1", target_node_id="node_2")

        assert relationship.source_node_id == "node_1"
        assert relationship.target_node_id == "node_2"
        assert relationship.relationship_type == RelationshipType.RELATED
        assert relationship.strength == 0.5
        assert relationship.description == ""
        assert relationship.discovery_episode_id is None

    def test_semantic_relationship_creation_with_all_fields(self):
        """Test creating semantic relationship with all fields specified."""
        now = datetime.now()
        relationship = SemanticRelationship(
            relationship_id="custom_id",
            source_node_id="node_1",
            target_node_id="node_2",
            relationship_type=RelationshipType.EVOLVED_FROM,
            strength=0.8,
            description="Node 2 evolved from Node 1",
            created_at=now,
            last_reinforced=now,
            discovery_episode_id="episode_123",
        )

        assert relationship.relationship_id == "custom_id"
        assert relationship.source_node_id == "node_1"
        assert relationship.target_node_id == "node_2"
        assert relationship.relationship_type == RelationshipType.EVOLVED_FROM
        assert relationship.strength == 0.8
        assert relationship.description == "Node 2 evolved from Node 1"
        assert relationship.created_at == now
        assert relationship.last_reinforced == now
        assert relationship.discovery_episode_id == "episode_123"

    def test_semantic_relationship_validation_missing_source(self):
        """Test that missing source_node_id raises ValueError."""
        with pytest.raises(ValueError, match="source_node_id is required"):
            SemanticRelationship(target_node_id="node_2")

    def test_semantic_relationship_validation_missing_target(self):
        """Test that missing target_node_id raises ValueError."""
        with pytest.raises(ValueError, match="target_node_id is required"):
            SemanticRelationship(source_node_id="node_1")

    def test_semantic_relationship_validation_same_nodes(self):
        """Test that same source and target raises ValueError."""
        with pytest.raises(ValueError, match="source_node_id and target_node_id cannot be the same"):
            SemanticRelationship(source_node_id="node_1", target_node_id="node_1")

    def test_semantic_relationship_validation_invalid_strength(self):
        """Test that invalid strength values raise ValueError."""
        with pytest.raises(ValueError, match="strength must be between 0.0 and 1.0"):
            SemanticRelationship(source_node_id="node_1", target_node_id="node_2", strength=1.5)

        with pytest.raises(ValueError, match="strength must be between 0.0 and 1.0"):
            SemanticRelationship(source_node_id="node_1", target_node_id="node_2", strength=-0.1)

    def test_semantic_relationship_to_dict(self):
        """Test converting semantic relationship to dictionary."""
        now = datetime.now()
        relationship = SemanticRelationship(
            source_node_id="node_1",
            target_node_id="node_2",
            relationship_type=RelationshipType.SIMILAR_TO,
            strength=0.7,
            created_at=now,
            last_reinforced=now,
        )

        result = relationship.to_dict()

        assert result["source_node_id"] == "node_1"
        assert result["target_node_id"] == "node_2"
        assert result["relationship_type"] == "similar_to"
        assert result["strength"] == 0.7
        assert result["created_at"] == now.isoformat()
        assert result["last_reinforced"] == now.isoformat()
        assert "relationship_id" in result

    def test_semantic_relationship_from_dict(self):
        """Test creating semantic relationship from dictionary."""
        now = datetime.now()
        data = {
            "source_node_id": "node_1",
            "target_node_id": "node_2",
            "relationship_type": "evolved_from",
            "strength": 0.9,
            "description": "Evolution relationship",
            "created_at": now.isoformat(),
            "last_reinforced": now.isoformat(),
            "discovery_episode_id": "episode_123",
        }

        relationship = SemanticRelationship.from_dict(data)

        assert relationship.source_node_id == "node_1"
        assert relationship.target_node_id == "node_2"
        assert relationship.relationship_type == RelationshipType.EVOLVED_FROM
        assert relationship.strength == 0.9
        assert relationship.description == "Evolution relationship"
        assert relationship.discovery_episode_id == "episode_123"

    def test_semantic_relationship_reinforce(self):
        """Test reinforcing semantic relationship."""
        relationship = SemanticRelationship(source_node_id="node_1", target_node_id="node_2", strength=0.5)

        reinforced = relationship.reinforce(0.2)

        assert reinforced.strength == 0.7
        assert reinforced.last_reinforced > relationship.last_reinforced
        # Original should be unchanged
        assert relationship.strength == 0.5

        # Test that reinforcement caps at 1.0
        highly_reinforced = reinforced.reinforce(0.5)
        assert highly_reinforced.strength == 1.0

    def test_semantic_relationship_bidirectional_equivalence(self):
        """Test checking bidirectional equivalence of relationships."""
        rel1 = SemanticRelationship(
            source_node_id="node_1", target_node_id="node_2", relationship_type=RelationshipType.RELATED
        )

        rel2 = SemanticRelationship(
            source_node_id="node_2", target_node_id="node_1", relationship_type=RelationshipType.RELATED
        )

        rel3 = SemanticRelationship(
            source_node_id="node_1", target_node_id="node_3", relationship_type=RelationshipType.RELATED
        )

        rel4 = SemanticRelationship(
            source_node_id="node_1", target_node_id="node_2", relationship_type=RelationshipType.EVOLVED_FROM
        )

        # Test bidirectional equivalence
        assert rel1.is_bidirectional_equivalent(rel2)
        assert rel2.is_bidirectional_equivalent(rel1)

        # Test non-equivalent relationships
        assert not rel1.is_bidirectional_equivalent(rel3)
        assert not rel1.is_bidirectional_equivalent(rel4)


class TestRelationshipType:
    """Test suite for RelationshipType enum."""

    def test_relationship_type_values(self):
        """Test that all relationship types have correct string values."""
        assert RelationshipType.RELATED.value == "related"
        assert RelationshipType.EVOLVED_FROM.value == "evolved_from"
        assert RelationshipType.PART_OF.value == "part_of"
        assert RelationshipType.SIMILAR_TO.value == "similar_to"
        assert RelationshipType.OPPOSITE_TO.value == "opposite_to"
        assert RelationshipType.TEMPORAL.value == "temporal"

    def test_relationship_type_from_string(self):
        """Test creating relationship type from string value."""
        assert RelationshipType("related") == RelationshipType.RELATED
        assert RelationshipType("evolved_from") == RelationshipType.EVOLVED_FROM
        assert RelationshipType("part_of") == RelationshipType.PART_OF
        assert RelationshipType("similar_to") == RelationshipType.SIMILAR_TO
        assert RelationshipType("opposite_to") == RelationshipType.OPPOSITE_TO
        assert RelationshipType("temporal") == RelationshipType.TEMPORAL
