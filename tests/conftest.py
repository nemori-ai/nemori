"""
Pytest configuration and shared fixtures for Nemori tests.

This module provides common test fixtures and configuration used across
all test modules in the Nemori test suite.
"""

from datetime import datetime
from typing import Any

import pytest

from nemori.core.data_types import DataType, RawEventData, TemporalInfo
from nemori.core.episode import Episode, EpisodeLevel, EpisodeMetadata, EpisodeType


@pytest.fixture
def sample_timestamp():
    """Fixture providing a consistent timestamp for tests."""
    return datetime(2024, 1, 15, 10, 30, 0)


@pytest.fixture
def sample_temporal_info(sample_timestamp):
    """Fixture providing a sample TemporalInfo object."""
    return TemporalInfo(timestamp=sample_timestamp, duration=120.0, timezone="UTC", precision="second")


@pytest.fixture
def sample_conversation_messages():
    """Fixture providing sample conversation messages."""
    return [
        {
            "user_id": "user_123",
            "user_name": "Alice",
            "content": "Hi, I'm planning a trip to Japan next month. Can you help me with some recommendations?",
            "timestamp": "2024-01-15T10:30:00",
        },
        {
            "user_id": "assistant_ai",
            "user_name": "Travel Assistant",
            "content": "I'd be happy to help you plan your Japan trip! What kind of experiences are you most interested in - cultural sites, food, nature, or modern attractions?",
            "timestamp": "2024-01-15T10:30:15",
        },
        {
            "user_id": "user_123",
            "user_name": "Alice",
            "content": "I'm really interested in traditional culture and authentic food experiences. I'll be there for about 10 days.",
            "timestamp": "2024-01-15T10:31:00",
        },
        {
            "user_id": "assistant_ai",
            "user_name": "Travel Assistant",
            "content": "Perfect! For traditional culture, I recommend visiting Kyoto for temples like Kinkaku-ji and Fushimi Inari. For authentic food, try kaiseki dining in Kyoto, fresh sushi at Tsukiji Outer Market in Tokyo, and local ramen shops.",
            "timestamp": "2024-01-15T10:31:30",
        },
    ]


@pytest.fixture
def sample_conversation_raw_data(sample_conversation_messages, sample_temporal_info):
    """Fixture providing sample conversation RawEventData."""
    return RawEventData(
        data_id="conv_data_123",
        data_type=DataType.CONVERSATION,
        content=sample_conversation_messages,
        source="travel_chat_app",
        temporal_info=sample_temporal_info,
        metadata={"session_id": "session_456", "conversation_topic": "travel_planning", "user_location": "New York", "primary_participant": "user_123"},
    )


@pytest.fixture
def sample_activity_raw_data(sample_temporal_info):
    """Fixture providing sample activity RawEventData."""
    activity_log = {
        "activity_type": "web_browsing",
        "sessions": [
            {
                "url": "https://www.japan-guide.com/e/e623.html",
                "title": "Kyoto Travel Guide",
                "duration": 300,
                "timestamp": "2024-01-15T10:30:00",
            },
            {
                "url": "https://www.booking.com/country/jp.html",
                "title": "Hotels in Japan",
                "duration": 180,
                "timestamp": "2024-01-15T10:35:00",
            },
        ],
    }

    return RawEventData(
        data_id="activity_data_123",
        data_type=DataType.ACTIVITY,
        content=activity_log,
        source="browser_extension",
        temporal_info=sample_temporal_info,
        metadata={"browser": "Chrome", "device": "MacBook Pro", "user_agent": "Mozilla/5.0...", "user_id": "user_123"},
    )


@pytest.fixture
def sample_episode_metadata(sample_timestamp):
    """Fixture providing sample EpisodeMetadata."""
    return EpisodeMetadata(
        source_data_ids=["conv_data_123"],
        source_types={DataType.CONVERSATION},
        processing_timestamp=sample_timestamp,
        processing_version="1.0",
        entities=["Alice", "Japan", "Kyoto", "Tokyo"],
        topics=["travel", "culture", "food", "planning"],
        emotions=["excited", "curious", "interested"],
        key_points=[
            "Trip to Japan for 10 days",
            "Interest in traditional culture",
            "Authentic food experiences",
            "Recommendations for Kyoto temples",
        ],
        time_references=["next month", "10 days"],
        duration_seconds=90.0,
        confidence_score=0.9,
        completeness_score=0.85,
        relevance_score=0.95,
        custom_fields={"session_id": "session_456", "conversation_type": "travel_planning"},
    )


@pytest.fixture
def sample_episode(sample_temporal_info, sample_episode_metadata):
    """Fixture providing a complete sample Episode."""
    return Episode(
        episode_id="episode_123",
        owner_id="user_123",
        episode_type=EpisodeType.CONVERSATIONAL,
        level=EpisodeLevel.ATOMIC,
        title="Japan Travel Planning Discussion",
        content="Alice discussed her upcoming 10-day trip to Japan with a travel assistant. She expressed strong interest in traditional culture and authentic food experiences. The assistant recommended visiting Kyoto for temples like Kinkaku-ji and Fushimi Inari, and suggested trying kaiseki dining, fresh sushi at Tsukiji Outer Market, and local ramen shops for authentic culinary experiences.",
        summary="Travel planning conversation about Japan focusing on culture and food",
        temporal_info=sample_temporal_info,
        metadata=sample_episode_metadata,
        structured_data={
            "conversation_data": {
                "message_count": 4,
                "participants": ["user_123", "assistant_ai"],
                "participant_names": {"user_123": "Alice", "assistant_ai": "Travel Assistant"},
                "start_time": "2024-01-15T10:30:00",
                "end_time": "2024-01-15T10:31:30",
            },
            "trip_details": {
                "destination": "Japan",
                "duration": "10 days",
                "interests": ["traditional culture", "authentic food"],
            },
        },
        search_keywords=[
            "japan",
            "travel",
            "culture",
            "food",
            "kyoto",
            "tokyo",
            "temple",
            "sushi",
            "ramen",
            "kaiseki",
            "traditional",
            "authentic",
            "planning",
        ],
        importance_score=0.7,
    )


@pytest.fixture
def mock_llm_responses():
    """Fixture providing mock LLM responses for testing."""
    return {
        "boundary_detection": '{"should_end": true, "reason": "Topic change detected - shift from travel planning to weather discussion", "confidence": 0.85, "topic_summary": "Japan travel planning"}',
        "episode_generation": '{"title": "Japan Travel Planning Session", "content": "Alice engaged in a comprehensive discussion about her upcoming trip to Japan with a travel assistant. She expressed keen interest in experiencing traditional Japanese culture and authentic cuisine. The conversation covered temple visits in Kyoto, including recommendations for Kinkaku-ji and Fushimi Inari, as well as food experiences such as kaiseki dining and fresh sushi at Tsukiji Outer Market.", "summary": "Detailed travel planning conversation focusing on cultural and culinary experiences in Japan"}',
        "entity_extraction": '{"entities": ["Alice", "Japan", "Kyoto", "Tokyo", "Kinkaku-ji", "Fushimi Inari", "Tsukiji Outer Market"], "locations": ["Japan", "Kyoto", "Tokyo"], "people": ["Alice"], "organizations": ["Tsukiji Outer Market"]}',
        "topic_classification": '{"topics": ["travel", "culture", "food", "planning", "tourism"], "primary_topic": "travel", "confidence_scores": {"travel": 0.95, "culture": 0.85, "food": 0.80}}',
    }


class MockLLMProvider:
    """Mock LLM provider for testing purposes."""

    def __init__(self, responses: dict[str, str] = None):
        self.responses = responses or {}
        self.call_history = []
        self.call_count = 0

    def generate(self, prompt: str, temperature: float = 0.3) -> str:
        """Generate a mock response based on prompt content."""
        self.call_count += 1
        call_info = {
            "call_number": self.call_count,
            "prompt": prompt,
            "temperature": temperature,
            "timestamp": datetime.now(),
        }
        self.call_history.append(call_info)

        # Determine response type based on prompt content
        prompt_lower = prompt.lower()

        if "boundary" in prompt_lower or "should_end" in prompt_lower:
            return self.responses.get("boundary_detection", '{"should_end": true, "reason": "mock boundary detection"}')
        elif "episode" in prompt_lower or "title" in prompt_lower:
            return self.responses.get(
                "episode_generation",
                '{"title": "Mock Episode Title", "content": "Mock episode content", "summary": "Mock summary"}',
            )
        elif "entity" in prompt_lower:
            return self.responses.get("entity_extraction", '{"entities": ["MockEntity1", "MockEntity2"]}')
        elif "topic" in prompt_lower:
            return self.responses.get("topic_classification", '{"topics": ["mock_topic1", "mock_topic2"]}')
        else:
            return self.responses.get("default", '{"result": "mock response"}')

    def test_connection(self) -> bool:
        """Test connection (always returns True for mock)."""
        return True

    def get_call_count(self) -> int:
        """Get the number of times generate was called."""
        return self.call_count

    def get_last_call(self) -> dict[str, Any]:
        """Get information about the last call."""
        return self.call_history[-1] if self.call_history else None

    def reset(self):
        """Reset call history and count."""
        self.call_history = []
        self.call_count = 0

    def __repr__(self) -> str:
        return f"MockLLMProvider(calls={self.call_count})"


@pytest.fixture
def mock_llm_provider(mock_llm_responses):
    """Fixture providing a MockLLMProvider with standard responses."""
    return MockLLMProvider(mock_llm_responses)


@pytest.fixture
def conversation_episode_builder_with_mock_llm(mock_llm_provider):
    """Fixture providing ConversationEpisodeBuilder with mock LLM."""
    from nemori.builders.conversation_builder import ConversationEpisodeBuilder

    return ConversationEpisodeBuilder(llm_provider=mock_llm_provider)


@pytest.fixture
def conversation_episode_builder_no_llm():
    """Fixture providing ConversationEpisodeBuilder without LLM (fallback mode)."""
    from nemori.builders.conversation_builder import ConversationEpisodeBuilder

    return ConversationEpisodeBuilder()


@pytest.fixture
def episode_builder_registry():
    """Fixture providing an empty EpisodeBuilderRegistry."""
    from nemori.core.builders import EpisodeBuilderRegistry

    return EpisodeBuilderRegistry()


# Test data validation helpers
def assert_valid_raw_user_data(raw_data: RawEventData):
    """Assert that a RawEventData object is valid."""
    assert isinstance(raw_data, RawEventData)
    assert raw_data.data_id
    assert isinstance(raw_data.data_type, DataType)
    assert raw_data.content is not None
    assert isinstance(raw_data.temporal_info, TemporalInfo)
    assert isinstance(raw_data.metadata, dict)


def assert_valid_episode(episode: Episode):
    """Assert that an Episode object is valid."""
    assert isinstance(episode, Episode)
    assert episode.owner_id
    assert episode.episode_id
    assert isinstance(episode.episode_type, EpisodeType)
    assert isinstance(episode.level, EpisodeLevel)
    assert episode.title or episode.content  # At least one must be present
    assert isinstance(episode.temporal_info, TemporalInfo)
    assert isinstance(episode.metadata, EpisodeMetadata)
    assert isinstance(episode.structured_data, dict)
    assert isinstance(episode.search_keywords, list)
    assert 0.0 <= episode.importance_score <= 1.0
    assert episode.recall_count >= 0


def assert_episodes_equal(episode1: Episode, episode2: Episode, ignore_fields: list[str] = None):
    """Assert that two episodes are equal, optionally ignoring certain fields."""
    ignore_fields = ignore_fields or []

    if "episode_id" not in ignore_fields:
        assert episode1.episode_id == episode2.episode_id
    if "owner_id" not in ignore_fields:
        assert episode1.owner_id == episode2.owner_id
    if "episode_type" not in ignore_fields:
        assert episode1.episode_type == episode2.episode_type
    if "level" not in ignore_fields:
        assert episode1.level == episode2.level
    if "title" not in ignore_fields:
        assert episode1.title == episode2.title
    if "content" not in ignore_fields:
        assert episode1.content == episode2.content
    if "summary" not in ignore_fields:
        assert episode1.summary == episode2.summary
    if "importance_score" not in ignore_fields:
        assert episode1.importance_score == episode2.importance_score
    if "recall_count" not in ignore_fields:
        assert episode1.recall_count == episode2.recall_count


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (deselect with '-m \"not integration\"')"
    )
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "llm: marks tests that require LLM providers")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
