"""
Unit tests for Nemori Episode classes.

This module tests the Episode, EpisodeMetadata, and related functionality.
"""

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta

import pytest

from nemori.core.data_types import DataType, TemporalInfo
from nemori.core.episode import Episode, EpisodeLevel, EpisodeMetadata, EpisodeType


class TestEpisodeType:
    """Test EpisodeType enum."""

    def test_episode_type_values(self):
        """Test that all episode types have correct string values."""
        assert EpisodeType.CONVERSATIONAL.value == "conversational"
        assert EpisodeType.BEHAVIORAL.value == "behavioral"
        assert EpisodeType.SPATIAL.value == "spatial"
        assert EpisodeType.CREATIVE.value == "creative"
        assert EpisodeType.PHYSIOLOGICAL.value == "physiological"
        assert EpisodeType.SOCIAL.value == "social"
        assert EpisodeType.MIXED.value == "mixed"
        assert EpisodeType.SYNTHETIC.value == "synthetic"

    def test_episode_type_count(self):
        """Test that we have expected number of episode types."""
        assert len(EpisodeType) == 8


class TestEpisodeLevel:
    """Test EpisodeLevel enum."""

    def test_episode_level_values(self):
        """Test that all episode levels have correct integer values."""
        assert EpisodeLevel.ATOMIC.value == 1
        assert EpisodeLevel.COMPOUND.value == 2
        assert EpisodeLevel.THEMATIC.value == 3
        assert EpisodeLevel.ARCHIVAL.value == 4

    def test_episode_level_count(self):
        """Test that we have expected number of episode levels."""
        assert len(EpisodeLevel) == 4


class TestEpisodeMetadata:
    """Test EpisodeMetadata value object."""

    def test_episode_metadata_creation(self):
        """Test basic EpisodeMetadata creation."""
        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        metadata = EpisodeMetadata(
            source_data_ids=["data_1", "data_2"],
            source_types={DataType.CONVERSATION, DataType.ACTIVITY},
            processing_timestamp=timestamp,
            processing_version="2.0",
            entities=["Alice", "Tokyo"],
            topics=["travel", "planning"],
            emotions=["excited", "curious"],
            key_points=["Trip to Japan", "Cultural interests"],
            time_references=["next month", "spring"],
            duration_seconds=300.0,
            confidence_score=0.9,
            completeness_score=0.8,
            relevance_score=0.95,
            related_episode_ids=["related_1"],
            custom_fields={"session_id": "session_456"},
        )

        assert metadata.source_data_ids == ["data_1", "data_2"]
        assert metadata.source_types == {DataType.CONVERSATION, DataType.ACTIVITY}
        assert metadata.processing_timestamp == timestamp
        assert metadata.processing_version == "2.0"
        assert metadata.entities == ["Alice", "Tokyo"]
        assert metadata.topics == ["travel", "planning"]
        assert metadata.emotions == ["excited", "curious"]
        assert metadata.key_points == ["Trip to Japan", "Cultural interests"]
        assert metadata.time_references == ["next month", "spring"]
        assert metadata.duration_seconds == 300.0
        assert metadata.confidence_score == 0.9
        assert metadata.completeness_score == 0.8
        assert metadata.relevance_score == 0.95
        assert metadata.related_episode_ids == ["related_1"]
        assert metadata.custom_fields == {"session_id": "session_456"}

    def test_episode_metadata_defaults(self):
        """Test EpisodeMetadata with default values."""
        metadata = EpisodeMetadata()

        assert metadata.source_data_ids == []
        assert metadata.source_types == set()
        assert isinstance(metadata.processing_timestamp, datetime)
        assert metadata.processing_version == "1.0"
        assert metadata.entities == []
        assert metadata.topics == []
        assert metadata.emotions == []
        assert metadata.key_points == []
        assert metadata.time_references == []
        assert metadata.duration_seconds is None
        assert metadata.confidence_score == 1.0
        assert metadata.completeness_score == 1.0
        assert metadata.relevance_score == 1.0
        assert metadata.related_episode_ids == []
        assert metadata.custom_fields == {}

    def test_episode_metadata_to_dict(self):
        """Test EpisodeMetadata serialization to dict."""
        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        metadata = EpisodeMetadata(
            source_data_ids=["data_1"],
            source_types={DataType.CONVERSATION},
            processing_timestamp=timestamp,
            entities=["Alice"],
            topics=["travel"],
            custom_fields={"key": "value"},
        )

        result = metadata.to_dict()

        assert result["source_data_ids"] == ["data_1"]
        assert result["source_types"] == ["conversation"]
        assert result["processing_timestamp"] == "2024-01-15T10:30:00"
        assert result["entities"] == ["Alice"]
        assert result["topics"] == ["travel"]
        assert result["custom_fields"] == {"key": "value"}

    def test_episode_metadata_immutable(self):
        """Test that EpisodeMetadata is frozen/immutable."""
        metadata = EpisodeMetadata()

        with pytest.raises(FrozenInstanceError):
            metadata.entities = ["new_entity"]


class TestEpisode:
    """Test Episode class."""

    def create_sample_episode(self) -> Episode:
        """Create a sample episode for testing."""
        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        temporal_info = TemporalInfo(timestamp=timestamp, duration=300.0)

        metadata = EpisodeMetadata(
            source_data_ids=["data_123"],
            source_types={DataType.CONVERSATION},
            entities=["Alice", "Japan"],
            topics=["travel", "planning"],
            emotions=["excited"],
            custom_fields={"session_id": "session_456"},
        )

        return Episode(
            episode_id="episode_123",
            owner_id="user_123",
            episode_type=EpisodeType.CONVERSATIONAL,
            level=EpisodeLevel.ATOMIC,
            title="Japan Travel Planning",
            content="Alice discussed her upcoming trip to Japan, expressing interest in traditional culture and authentic food experiences.",
            summary="Travel planning conversation about Japan",
            temporal_info=temporal_info,
            metadata=metadata,
            structured_data={"conversation_type": "travel_planning"},
            search_keywords=["japan", "travel", "culture", "food"],
            importance_score=0.8,
        )

    def test_episode_creation(self):
        """Test basic Episode creation."""
        episode = self.create_sample_episode()

        assert episode.episode_id == "episode_123"
        assert episode.owner_id == "user_123"
        assert episode.episode_type == EpisodeType.CONVERSATIONAL
        assert episode.level == EpisodeLevel.ATOMIC
        assert episode.title == "Japan Travel Planning"
        assert "Alice discussed her upcoming trip" in episode.content
        assert episode.summary == "Travel planning conversation about Japan"
        assert episode.temporal_info.timestamp == datetime(2024, 1, 15, 10, 30, 0)
        assert episode.importance_score == 0.8
        assert episode.recall_count == 0
        assert episode.last_accessed is None

    def test_episode_validation(self):
        """Test Episode validation."""
        # Test missing owner_id
        with pytest.raises(ValueError, match="owner_id is required"):
            Episode(owner_id="")

        # Test missing both title and content
        with pytest.raises(ValueError, match="Either title or content must be provided"):
            Episode(owner_id="user_123", title="", content="")

    def test_episode_auto_generation(self):
        """Test automatic generation of title and summary."""
        # Test auto-generated summary from content
        episode = Episode(
            owner_id="user_123",
            content="This is a very long content that should be truncated when generating the summary because it exceeds the maximum length that we want for summaries.",
        )

        assert (
            episode.summary
            == "This is a very long content that should be truncated when generating the summary because it exceeds ..."
        )

        # Test auto-generated title from summary
        episode2 = Episode(
            owner_id="user_123",
            content="Short content",
            summary="This is a summary that is longer than fifty characters and should be truncated for the title",
        )

        assert episode2.title == "This is a summary that is longer than fifty charac..."

    def test_episode_defaults(self):
        """Test Episode with default values."""
        episode = Episode(owner_id="user_123", content="Test content")

        assert episode.episode_type == EpisodeType.MIXED
        assert episode.level == EpisodeLevel.ATOMIC
        assert isinstance(episode.temporal_info, TemporalInfo)
        assert isinstance(episode.metadata, EpisodeMetadata)
        assert episode.structured_data == {}
        assert episode.search_keywords == []
        assert episode.embedding_vector is None
        assert episode.recall_count == 0
        assert episode.importance_score == 0.0
        assert episode.last_accessed is None

    def test_episode_to_dict(self):
        """Test Episode serialization to dict."""
        episode = self.create_sample_episode()

        result = episode.to_dict()

        assert result["episode_id"] == "episode_123"
        assert result["owner_id"] == "user_123"
        assert result["episode_type"] == "conversational"
        assert result["level"] == 1
        assert result["title"] == "Japan Travel Planning"
        assert "Alice discussed" in result["content"]
        assert result["summary"] == "Travel planning conversation about Japan"
        assert result["temporal_info"]["timestamp"] == "2024-01-15T10:30:00"
        assert result["temporal_info"]["duration"] == 300.0
        assert result["metadata"]["entities"] == ["Alice", "Japan"]
        assert result["structured_data"] == {"conversation_type": "travel_planning"}
        assert result["search_keywords"] == ["japan", "travel", "culture", "food"]
        assert result["importance_score"] == 0.8
        assert result["recall_count"] == 0
        assert result["last_accessed"] is None

    def test_episode_from_dict(self):
        """Test Episode deserialization from dict."""
        episode_dict = {
            "episode_id": "episode_123",
            "owner_id": "user_123",
            "episode_type": "conversational",
            "level": 1,
            "title": "Japan Travel Planning",
            "content": "Alice discussed her trip to Japan.",
            "summary": "Travel planning conversation",
            "temporal_info": {
                "timestamp": "2024-01-15T10:30:00",
                "duration": 300.0,
                "timezone": "UTC",
                "precision": "second",
            },
            "metadata": {
                "source_data_ids": ["data_123"],
                "source_types": ["conversation"],
                "processing_timestamp": "2024-01-15T10:35:00",
                "entities": ["Alice", "Japan"],
                "topics": ["travel"],
                "custom_fields": {"session": "session_456"},
            },
            "structured_data": {"type": "travel"},
            "search_keywords": ["japan", "travel"],
            "importance_score": 0.8,
            "recall_count": 5,
            "last_accessed": "2024-01-15T11:00:00",
        }

        episode = Episode.from_dict(episode_dict)

        assert episode.episode_id == "episode_123"
        assert episode.owner_id == "user_123"
        assert episode.episode_type == EpisodeType.CONVERSATIONAL
        assert episode.level == EpisodeLevel.ATOMIC
        assert episode.title == "Japan Travel Planning"
        assert episode.content == "Alice discussed her trip to Japan."
        assert episode.temporal_info.timestamp == datetime(2024, 1, 15, 10, 30, 0)
        assert episode.temporal_info.duration == 300.0
        assert episode.metadata.entities == ["Alice", "Japan"]
        assert episode.metadata.topics == ["travel"]
        assert episode.structured_data == {"type": "travel"}
        assert episode.search_keywords == ["japan", "travel"]
        assert episode.importance_score == 0.8
        assert episode.recall_count == 5
        assert episode.last_accessed == datetime(2024, 1, 15, 11, 0, 0)

    def test_mark_accessed(self):
        """Test mark_accessed method."""
        episode = self.create_sample_episode()

        assert episode.recall_count == 0
        assert episode.last_accessed is None

        # Mark as accessed
        episode.mark_accessed()

        assert episode.recall_count == 1
        assert isinstance(episode.last_accessed, datetime)
        assert episode.last_accessed <= datetime.now()

        # Mark as accessed again
        first_access_time = episode.last_accessed
        episode.mark_accessed()

        assert episode.recall_count == 2
        assert episode.last_accessed >= first_access_time

    def test_add_related_episode(self):
        """Test add_related_episode method."""
        episode = self.create_sample_episode()

        # Initially no related episodes
        assert episode.metadata.related_episode_ids == []

        # Add related episode
        episode.add_related_episode("related_episode_1")

        assert "related_episode_1" in episode.metadata.related_episode_ids

        # Add another related episode
        episode.add_related_episode("related_episode_2")

        assert len(episode.metadata.related_episode_ids) == 2
        assert "related_episode_1" in episode.metadata.related_episode_ids
        assert "related_episode_2" in episode.metadata.related_episode_ids

        # Adding duplicate should not increase count
        episode.add_related_episode("related_episode_1")

        assert len(episode.metadata.related_episode_ids) == 2

    def test_update_importance(self):
        """Test update_importance method."""
        episode = self.create_sample_episode()

        # Test valid importance scores
        episode.update_importance(0.5)
        assert episode.importance_score == 0.5

        episode.update_importance(1.0)
        assert episode.importance_score == 1.0

        episode.update_importance(0.0)
        assert episode.importance_score == 0.0

        # Test clamping to valid range
        episode.update_importance(-0.5)
        assert episode.importance_score == 0.0

        episode.update_importance(1.5)
        assert episode.importance_score == 1.0

    def test_get_display_text(self):
        """Test get_display_text method."""
        episode = self.create_sample_episode()

        display_text = episode.get_display_text()

        assert "[2024-01-15 10:30]" in display_text
        assert episode.title in display_text
        assert episode.summary in display_text

    def test_is_recent(self):
        """Test is_recent method."""
        # Create episode with current timestamp
        recent_episode = Episode(
            owner_id="user_123", content="Recent content", temporal_info=TemporalInfo(timestamp=datetime.now())
        )

        assert recent_episode.is_recent(24) is True
        assert recent_episode.is_recent(1) is True

        # Create episode with old timestamp
        old_episode = Episode(
            owner_id="user_123",
            content="Old content",
            temporal_info=TemporalInfo(timestamp=datetime.now() - timedelta(days=2)),
        )

        assert old_episode.is_recent(24) is False
        assert old_episode.is_recent(72) is True  # 72 hours = 3 days, so 2 days ago should be recent

    def test_matches_keywords(self):
        """Test matches_keywords method."""
        episode = self.create_sample_episode()

        # Test matching keywords
        assert episode.matches_keywords(["japan"]) is True
        assert episode.matches_keywords(["travel"]) is True
        assert episode.matches_keywords(["alice"]) is True
        assert episode.matches_keywords(["JAPAN"]) is True  # Case insensitive

        # Test non-matching keywords
        assert episode.matches_keywords(["python"]) is False
        assert episode.matches_keywords(["programming"]) is False

        # Test multiple keywords (any match)
        assert episode.matches_keywords(["python", "japan"]) is True
        assert episode.matches_keywords(["python", "programming"]) is False

        # Test empty keywords
        assert episode.matches_keywords([]) is False


# Integration tests
class TestEpisodeIntegration:
    """Integration tests for Episode working with other components."""

    def test_episode_serialization_round_trip(self):
        """Test complete serialization and deserialization cycle."""
        # Create complex episode
        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        temporal_info = TemporalInfo(timestamp=timestamp, duration=600.0, timezone="UTC", precision="second")

        metadata = EpisodeMetadata(
            source_data_ids=["data_1", "data_2"],
            source_types={DataType.CONVERSATION, DataType.ACTIVITY},
            processing_timestamp=datetime(2024, 1, 15, 10, 35, 0),
            entities=["Alice", "Bob", "Tokyo"],
            topics=["travel", "planning", "culture"],
            emotions=["excited", "curious"],
            key_points=["Trip planning", "Cultural interests", "Food experiences"],
            time_references=["next month", "spring"],
            duration_seconds=600.0,
            confidence_score=0.85,
            completeness_score=0.90,
            relevance_score=0.95,
            related_episode_ids=["related_1", "related_2"],
            custom_fields={"session_id": "session_456", "conversation_type": "travel_planning", "app_version": "2.1.0"},
        )

        original_episode = Episode(
            episode_id="episode_123",
            owner_id="user_123",
            episode_type=EpisodeType.CONVERSATIONAL,
            level=EpisodeLevel.COMPOUND,
            title="Comprehensive Japan Travel Planning Session",
            content="Alice and Bob discussed their upcoming trip to Japan with a travel assistant. They expressed strong interest in traditional culture, authentic food experiences, and visiting historical sites in Tokyo and Kyoto. The conversation covered accommodation options including traditional ryokans, transportation methods like the JR Pass, and specific cultural experiences such as tea ceremonies and temple visits.",
            summary="Detailed travel planning conversation about Japan focusing on culture and food",
            temporal_info=temporal_info,
            metadata=metadata,
            structured_data={
                "conversation_data": {
                    "participants": ["Alice", "Bob", "Travel Assistant"],
                    "message_count": 15,
                    "duration_minutes": 10,
                    "topics_covered": ["accommodation", "transportation", "cultural_activities"],
                },
                "trip_details": {
                    "destination": "Japan",
                    "cities": ["Tokyo", "Kyoto"],
                    "duration": "10 days",
                    "interests": ["culture", "food", "history"],
                },
            },
            search_keywords=[
                "japan",
                "travel",
                "culture",
                "food",
                "tokyo",
                "kyoto",
                "ryokan",
                "temple",
                "traditional",
                "planning",
            ],
            embedding_vector=[0.1, 0.2, 0.3, 0.4, 0.5],
            recall_count=3,
            importance_score=0.9,
            last_accessed=datetime(2024, 1, 15, 11, 0, 0),
        )

        # Serialize to dict
        episode_dict = original_episode.to_dict()

        # Deserialize from dict
        restored_episode = Episode.from_dict(episode_dict)

        # Verify all fields are preserved
        assert restored_episode.episode_id == original_episode.episode_id
        assert restored_episode.owner_id == original_episode.owner_id
        assert restored_episode.episode_type == original_episode.episode_type
        assert restored_episode.level == original_episode.level
        assert restored_episode.title == original_episode.title
        assert restored_episode.content == original_episode.content
        assert restored_episode.summary == original_episode.summary

        # Verify temporal info
        assert restored_episode.temporal_info.timestamp == original_episode.temporal_info.timestamp
        assert restored_episode.temporal_info.duration == original_episode.temporal_info.duration
        assert restored_episode.temporal_info.timezone == original_episode.temporal_info.timezone

        # Verify metadata
        assert restored_episode.metadata.entities == original_episode.metadata.entities
        assert restored_episode.metadata.topics == original_episode.metadata.topics
        assert restored_episode.metadata.source_types == original_episode.metadata.source_types
        assert restored_episode.metadata.custom_fields == original_episode.metadata.custom_fields

        # Verify other fields
        assert restored_episode.structured_data == original_episode.structured_data
        assert restored_episode.search_keywords == original_episode.search_keywords
        assert restored_episode.embedding_vector == original_episode.embedding_vector
        assert restored_episode.recall_count == original_episode.recall_count
        assert restored_episode.importance_score == original_episode.importance_score
        assert restored_episode.last_accessed == original_episode.last_accessed

    def test_episode_lifecycle_simulation(self):
        """Test simulating a complete episode lifecycle."""
        # Create new episode
        episode = Episode(
            owner_id="user_123",
            title="Learning Python Programming",
            content="User asked about Python basics and received explanations about variables, functions, and data types.",
            episode_type=EpisodeType.CONVERSATIONAL,
        )

        # Verify initial state
        assert episode.recall_count == 0
        assert episode.importance_score == 0.0
        assert episode.last_accessed is None
        assert episode.metadata.related_episode_ids == []

        # User accesses the episode multiple times
        episode.mark_accessed()
        episode.mark_accessed()
        episode.mark_accessed()

        assert episode.recall_count == 3
        assert episode.last_accessed is not None

        # Episode gains importance
        episode.update_importance(0.7)
        assert episode.importance_score == 0.7

        # Add related episodes
        episode.add_related_episode("python_advanced_episode")
        episode.add_related_episode("programming_concepts_episode")

        assert len(episode.metadata.related_episode_ids) == 2

        # Test keyword matching for retrieval
        assert episode.matches_keywords(["python"]) is True
        assert episode.matches_keywords(["programming"]) is True
        assert episode.matches_keywords(["variables"]) is True

        # Check if it's recent
        assert episode.is_recent(24) is True

        # Generate display text
        display_text = episode.get_display_text()
        assert "Learning Python Programming" in display_text

        # Final verification of state
        assert episode.recall_count == 3
        assert episode.importance_score == 0.7
        assert len(episode.metadata.related_episode_ids) == 2
