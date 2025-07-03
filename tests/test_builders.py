"""
Unit tests for Nemori episode builders.

This module tests the EpisodeBuilder abstract class and its implementations,
particularly the ConversationEpisodeBuilder.
"""

from datetime import datetime
from unittest.mock import patch

import pytest

from nemori.builders.conversation_builder import ConversationEpisodeBuilder
from nemori.core.builders import BatchEpisodeBuilder, EpisodeBuilder, EpisodeBuilderRegistry
from nemori.core.data_types import ConversationData, DataType, RawEventData, TemporalInfo
from nemori.core.episode import Episode, EpisodeLevel, EpisodeType


class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(self, responses: dict[str, str] = None):
        self.responses = responses or {}
        self.calls = []

    def generate(self, prompt: str, temperature: float = 0.3) -> str:
        self.calls.append({"prompt": prompt, "temperature": temperature})

        # Return appropriate response based on prompt content
        if "boundary detection expert" in prompt.lower():
            return self.responses.get("boundary", '{"should_end": true, "reason": "test boundary"}')
        else:
            return self.responses.get(
                "episode", '{"title": "Test Episode", "content": "Test content", "summary": "Test summary"}'
            )

    def test_connection(self) -> bool:
        return True

    def __repr__(self) -> str:
        return "MockLLMProvider()"


class TestEpisodeBuilder:
    """Test the abstract EpisodeBuilder class."""

    def test_episode_builder_is_abstract(self):
        """Test that EpisodeBuilder cannot be instantiated directly."""
        with pytest.raises(TypeError):
            EpisodeBuilder()

    def test_episode_builder_subclass_requirements(self):
        """Test that subclasses must implement required methods."""

        class IncompleteBuilder(EpisodeBuilder):
            pass

        with pytest.raises(TypeError):
            IncompleteBuilder()


class TestConversationEpisodeBuilder:
    """Test ConversationEpisodeBuilder implementation."""

    def create_sample_conversation_data(self) -> RawEventData:
        """Create sample conversation data for testing."""
        messages = [
            {
                "speaker_id": "user_123",
                "user_name": "Alice",
                "content": "Hi, I'm planning a trip to Japan next month. Can you help me with some recommendations?",
                "timestamp": "2024-01-15T10:30:00",
            },
            {
                "speaker_id": "assistant_ai",
                "user_name": "Travel Assistant",
                "content": "I'd be happy to help you plan your Japan trip! What kind of experiences are you most interested in - cultural sites, food, nature, or modern attractions? Japan offers incredible diversity.",
                "timestamp": "2024-01-15T10:30:15",
            },
            {
                "speaker_id": "user_123",
                "user_name": "Alice",
                "content": "I'm really interested in traditional culture and authentic food experiences.",
                "timestamp": "2024-01-15T10:31:00",
            },
        ]

        return RawEventData(
            data_type=DataType.CONVERSATION,
            content=messages,
            source="chat_application",
            temporal_info=TemporalInfo(timestamp=datetime(2024, 1, 15, 10, 30, 0), duration=60.0, timezone="UTC"),
            metadata={"session_id": "session_456"},
        )

    def test_builder_properties(self):
        """Test ConversationEpisodeBuilder properties."""
        builder = ConversationEpisodeBuilder()

        assert builder.supported_data_type == DataType.CONVERSATION
        assert builder.default_episode_type == EpisodeType.CONVERSATIONAL

    def test_can_build(self):
        """Test can_build method."""
        builder = ConversationEpisodeBuilder()

        # Test with correct data type
        conversation_data = self.create_sample_conversation_data()
        assert builder.can_build(conversation_data) is True

        # Test with wrong data type
        activity_data = RawEventData(data_type=DataType.ACTIVITY, content={"activity": "browsing"})
        assert builder.can_build(activity_data) is False

    def test_build_episode_validation(self):
        """Test that build_episode validates data type."""
        builder = ConversationEpisodeBuilder()

        activity_data = RawEventData(data_type=DataType.ACTIVITY, content={"activity": "browsing"})

        with pytest.raises(ValueError, match="Builder for DataType.CONVERSATION cannot process DataType.ACTIVITY data"):
            builder.build_episode(activity_data, for_owner="user_123")

    def test_build_episode_without_llm(self):
        """Test building episode without LLM provider (fallback mode)."""
        builder = ConversationEpisodeBuilder()
        conversation_data = self.create_sample_conversation_data()

        episode = builder.build_episode(conversation_data, for_owner="user_123")

        # Verify basic episode structure
        assert isinstance(episode, Episode)
        assert episode.owner_id == "user_123"
        assert episode.episode_type == EpisodeType.CONVERSATIONAL
        assert episode.level == EpisodeLevel.ATOMIC  # 3 messages = atomic
        assert episode.title
        assert episode.content
        assert episode.summary

        # Verify metadata
        assert len(episode.metadata.source_data_ids) == 1
        assert DataType.CONVERSATION in episode.metadata.source_types
        assert episode.metadata.custom_fields["message_count"] == 3
        assert episode.metadata.custom_fields["unique_participants"] == 2

        # Verify temporal info
        assert episode.temporal_info.timestamp == datetime(2024, 1, 15, 10, 30, 0)
        assert episode.temporal_info.duration == 60.0

    def test_build_episode_with_llm(self):
        """Test building episode with LLM provider."""
        mock_llm = MockLLMProvider(
            {
                "episode": '{"title": "Japan Travel Planning Session", "content": "Alice discussed her upcoming trip to Japan with a travel assistant, expressing interest in traditional culture and authentic food experiences.", "summary": "Travel planning conversation about Japan"}'
            }
        )

        builder = ConversationEpisodeBuilder(llm_provider=mock_llm)
        conversation_data = self.create_sample_conversation_data()

        episode = builder.build_episode(conversation_data, for_owner="user_123")

        # Verify LLM was called
        assert len(mock_llm.calls) == 1
        assert "episodic memory" in mock_llm.calls[0]["prompt"].lower()

        # Verify LLM-generated content
        assert episode.title == "Japan Travel Planning Session"
        assert "Alice discussed her upcoming trip to Japan" in episode.content
        assert episode.summary == "Travel planning conversation about Japan"

    def test_build_episode_with_llm_failure_fallback(self):
        """Test that LLM failure falls back to simple generation."""
        mock_llm = MockLLMProvider({"episode": "invalid json response"})

        builder = ConversationEpisodeBuilder(llm_provider=mock_llm)
        conversation_data = self.create_sample_conversation_data()

        episode = builder.build_episode(conversation_data, for_owner="user_123")

        # Should still create an episode using fallback
        assert isinstance(episode, Episode)
        assert episode.title
        assert episode.content
        assert episode.summary

    def test_extract_entities(self):
        """Test entity extraction from conversation."""
        builder = ConversationEpisodeBuilder()
        conversation_data = self.create_sample_conversation_data()
        typed_data = ConversationData(conversation_data)

        entities = builder._extract_entities(typed_data)

        # Should extract some entities (this is basic extraction)
        assert isinstance(entities, list)
        assert len(entities) <= 10  # Limited to 10

    def test_extract_topics(self):
        """Test topic extraction from conversation."""
        builder = ConversationEpisodeBuilder()
        conversation_data = self.create_sample_conversation_data()
        typed_data = ConversationData(conversation_data)

        topics = builder._extract_topics(typed_data)

        assert isinstance(topics, list)
        # Empty implementation returns empty list
        assert topics == []

    def test_extract_emotions(self):
        """Test emotion extraction from conversation."""
        builder = ConversationEpisodeBuilder()
        conversation_data = self.create_sample_conversation_data()
        typed_data = ConversationData(conversation_data)

        emotions = builder._extract_emotions(typed_data)

        assert isinstance(emotions, list)
        # Empty implementation returns empty list
        assert emotions == []

    def test_extract_key_points(self):
        """Test key point extraction from conversation."""
        builder = ConversationEpisodeBuilder()
        conversation_data = self.create_sample_conversation_data()
        typed_data = ConversationData(conversation_data)

        key_points = builder._extract_key_points(typed_data)

        assert isinstance(key_points, list)
        # Empty implementation returns empty list
        assert key_points == []

    def test_determine_episode_level(self):
        """Test episode level determination based on message count."""
        builder = ConversationEpisodeBuilder()

        # Test atomic level (≤5 messages)
        short_data = RawEventData(
            data_type=DataType.CONVERSATION,
            content=[{"speaker_id": "user_123", "content": f"Message {i}"} for i in range(3)],
        )
        typed_data = ConversationData(short_data)
        level = builder._determine_episode_level(typed_data)
        assert level == EpisodeLevel.ATOMIC

        # Test compound level (6-15 messages)
        medium_data = RawEventData(
            data_type=DataType.CONVERSATION,
            content=[{"speaker_id": "user_123", "content": f"Message {i}"} for i in range(10)],
        )
        typed_data = ConversationData(medium_data)
        level = builder._determine_episode_level(typed_data)
        assert level == EpisodeLevel.COMPOUND

        # Test thematic level (>15 messages)
        long_data = RawEventData(
            data_type=DataType.CONVERSATION,
            content=[{"speaker_id": "user_123", "content": f"Message {i}"} for i in range(20)],
        )
        typed_data = ConversationData(long_data)
        level = builder._determine_episode_level(typed_data)
        assert level == EpisodeLevel.THEMATIC

    def test_generate_keywords(self):
        """Test keyword generation from content."""
        builder = ConversationEpisodeBuilder()

        title = "Japan Travel Planning"
        content = "Discussion about traditional culture and authentic food experiences in Japan"
        summary = "Travel planning conversation"

        keywords = builder._generate_keywords(title, content, summary)

        assert isinstance(keywords, list)
        assert len(keywords) <= 20  # Limited to 20
        # Should extract meaningful keywords
        keyword_text = " ".join(keywords).lower()
        assert "japan" in keyword_text
        assert "travel" in keyword_text
        assert "culture" in keyword_text


class TestEpisodeBuilderRegistry:
    """Test EpisodeBuilderRegistry functionality."""

    def test_registry_register_and_get(self):
        """Test registering and retrieving builders."""
        registry = EpisodeBuilderRegistry()
        builder = ConversationEpisodeBuilder()

        registry.register(builder)

        retrieved_builder = registry.get_builder(DataType.CONVERSATION)
        assert retrieved_builder is builder

        # Test getting non-existent builder
        assert registry.get_builder(DataType.ACTIVITY) is None

    def test_registry_can_process(self):
        """Test can_process method."""
        registry = EpisodeBuilderRegistry()
        builder = ConversationEpisodeBuilder()

        assert registry.can_process(DataType.CONVERSATION) is False

        registry.register(builder)
        assert registry.can_process(DataType.CONVERSATION) is True
        assert registry.can_process(DataType.ACTIVITY) is False

    def test_registry_build_episode(self):
        """Test building episode through registry."""
        registry = EpisodeBuilderRegistry()
        builder = ConversationEpisodeBuilder()
        registry.register(builder)

        conversation_data = RawEventData(
            data_type=DataType.CONVERSATION, content=[{"speaker_id": "user_123", "content": "Hello"}]
        )

        episode = registry.build_episode(conversation_data, for_owner="user_123")
        assert isinstance(episode, Episode)

        # Test with unsupported data type
        activity_data = RawEventData(data_type=DataType.ACTIVITY, content={"activity": "browsing"})

        result = registry.build_episode(activity_data, for_owner="user_123")
        assert result is None

    def test_registry_override_warning(self):
        """Test that overriding builder shows warning."""
        registry = EpisodeBuilderRegistry()
        builder1 = ConversationEpisodeBuilder()
        builder2 = ConversationEpisodeBuilder()

        registry.register(builder1)

        with patch("builtins.print") as mock_print:
            registry.register(builder2)
            mock_print.assert_called_once_with("Warning: Overriding existing builder for DataType.CONVERSATION")

    def test_registry_get_all_builders(self):
        """Test getting all registered builders."""
        registry = EpisodeBuilderRegistry()
        builder = ConversationEpisodeBuilder()

        assert len(registry.get_all_builders()) == 0

        registry.register(builder)
        all_builders = registry.get_all_builders()

        assert len(all_builders) == 1
        assert all_builders[DataType.CONVERSATION] is builder


class TestBatchEpisodeBuilder:
    """Test BatchEpisodeBuilder functionality."""

    def test_batch_build_episodes(self):
        """Test building multiple episodes."""
        conversation_builder = ConversationEpisodeBuilder()
        builders = {DataType.CONVERSATION: conversation_builder}
        batch_builder = BatchEpisodeBuilder(builders)

        data_items = [
            RawEventData(
                data_type=DataType.CONVERSATION,
                content=[{"speaker_id": "user_123", "content": "Hello 1"}],
            ),
            RawEventData(
                data_type=DataType.CONVERSATION,
                content=[{"speaker_id": "user_123", "content": "Hello 2"}],
            ),
            RawEventData(
                data_type=DataType.ACTIVITY,  # No builder available
                content={"activity": "browsing"},
            ),
        ]

        episodes = batch_builder.build_episodes(data_items, for_owner="user_123")

        # Should build 2 episodes (skipping the activity data)
        assert len(episodes) == 2
        assert all(isinstance(ep, Episode) for ep in episodes)

    def test_batch_build_compound_episode(self):
        """Test building compound episode from multiple data items."""
        conversation_builder = ConversationEpisodeBuilder()
        builders = {DataType.CONVERSATION: conversation_builder}
        batch_builder = BatchEpisodeBuilder(builders)

        data_items = [
            RawEventData(
                data_type=DataType.CONVERSATION,
                content=[{"speaker_id": "user_123", "content": "Part 1"}],
                temporal_info=TemporalInfo(timestamp=datetime(2024, 1, 15, 10, 30, 0)),
            ),
            RawEventData(
                data_type=DataType.CONVERSATION,
                content=[{"speaker_id": "user_123", "content": "Part 2"}],
                temporal_info=TemporalInfo(timestamp=datetime(2024, 1, 15, 10, 35, 0)),
            ),
        ]

        compound_episode = batch_builder.build_compound_episode(
            data_items, for_owner="user_123", title="Combined Conversation", compound_type=EpisodeType.MIXED
        )

        assert isinstance(compound_episode, Episode)
        assert compound_episode.episode_type == EpisodeType.MIXED
        assert compound_episode.level == EpisodeLevel.COMPOUND
        assert compound_episode.title == "Combined Conversation"
        assert "Part 1" in compound_episode.content
        assert "Part 2" in compound_episode.content
        assert len(compound_episode.metadata.related_episode_ids) == 2

    def test_batch_build_compound_episode_empty_data(self):
        """Test that compound episode creation fails with empty data."""
        batch_builder = BatchEpisodeBuilder({})

        with pytest.raises(ValueError, match="Cannot create compound episode from empty data list"):
            batch_builder.build_compound_episode([], for_owner="user_123", title="Title")


# Integration tests
class TestBuildersIntegration:
    """Integration tests for builders working together."""

    def test_end_to_end_episode_building(self):
        """Test complete episode building workflow."""
        # Create registry and register builder
        registry = EpisodeBuilderRegistry()
        conversation_builder = ConversationEpisodeBuilder()
        registry.register(conversation_builder)

        # Create sample data
        conversation_data = RawEventData(
            data_type=DataType.CONVERSATION,
            content=[
                {
                    "speaker_id": "user_123",
                    "user_name": "Alice",
                    "content": "I need help with Python programming",
                    "timestamp": "2024-01-15T10:30:00",
                },
                {
                    "speaker_id": "assistant_ai",
                    "user_name": "Programming Assistant",
                    "content": "I'd be happy to help! What specific Python topic are you working on?",
                    "timestamp": "2024-01-15T10:30:05",
                },
            ],
            source="coding_chat",
            metadata={"topic": "programming"},
        )

        # Build episode
        episode = registry.build_episode(conversation_data, for_owner="user_123")

        # Verify complete episode
        assert isinstance(episode, Episode)
        assert episode.owner_id == "user_123"
        assert episode.episode_type == EpisodeType.CONVERSATIONAL
        assert "programming" in episode.content.lower() or "python" in episode.content.lower()
        assert len(episode.metadata.source_data_ids) == 1
        assert episode.metadata.custom_fields["message_count"] == 2
