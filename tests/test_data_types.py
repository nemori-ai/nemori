"""
Unit tests for Nemori data types.

This module contains comprehensive tests for all data type classes including
RawEventData, TypedEventData implementations, and related functionality.
"""

from datetime import datetime

import pytest

from nemori.core.data_types import (
    ConversationData,
    ConversationMessage,
    DataType,
    RawEventData,
    TemporalInfo,
    TypedEventData,
    create_typed_data,
)


class TestDataType:
    """Test DataType enum."""

    def test_data_type_values(self):
        """Test that all data types have correct string values."""
        assert DataType.CONVERSATION.value == "conversation"
        assert DataType.ACTIVITY.value == "activity"
        assert DataType.LOCATION.value == "location"
        assert DataType.MEDIA.value == "media"
        assert DataType.DOCUMENT.value == "document"
        assert DataType.SENSOR.value == "sensor"
        assert DataType.EXTERNAL.value == "external"
        assert DataType.CUSTOM.value == "custom"

    def test_data_type_count(self):
        """Test that we have expected number of data types."""
        assert len(DataType) == 8


class TestTemporalInfo:
    """Test TemporalInfo value object."""

    def test_temporal_info_creation(self):
        """Test basic TemporalInfo creation."""
        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        temporal_info = TemporalInfo(timestamp=timestamp, duration=120.0, timezone="UTC", precision="second")

        assert temporal_info.timestamp == timestamp
        assert temporal_info.duration == 120.0
        assert temporal_info.timezone == "UTC"
        assert temporal_info.precision == "second"

    def test_temporal_info_defaults(self):
        """Test TemporalInfo with default values."""
        timestamp = datetime.now()
        temporal_info = TemporalInfo(timestamp=timestamp)

        assert temporal_info.timestamp == timestamp
        assert temporal_info.duration is None
        assert temporal_info.timezone is None
        assert temporal_info.precision == "second"

    def test_temporal_info_to_dict(self):
        """Test TemporalInfo serialization to dict."""
        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        temporal_info = TemporalInfo(timestamp=timestamp, duration=120.0, timezone="UTC", precision="minute")

        result = temporal_info.to_dict()
        expected = {"timestamp": "2024-01-15T10:30:00", "duration": 120.0, "timezone": "UTC", "precision": "minute"}

        assert result == expected

    def test_temporal_info_immutable(self):
        """Test that TemporalInfo is frozen/immutable."""
        temporal_info = TemporalInfo(timestamp=datetime.now())

        with pytest.raises(AttributeError):
            temporal_info.timestamp = datetime.now()


class TestRawEventData:
    """Test RawEventData class."""

    def test_raw_event_data_creation(self):
        """Test basic RawEventData creation."""
        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        temporal_info = TemporalInfo(timestamp=timestamp)

        raw_data = RawEventData(
            data_type=DataType.CONVERSATION,
            content=["message1", "message2"],
            source="test_app",
            temporal_info=temporal_info,
            metadata={"session_id": "session_456", "participant_id": "user_123"},
        )

        assert raw_data.metadata["participant_id"] == "user_123"
        assert raw_data.data_type == DataType.CONVERSATION
        assert raw_data.content == ["message1", "message2"]
        assert raw_data.source == "test_app"
        assert raw_data.temporal_info == temporal_info
        assert raw_data.metadata == {"session_id": "session_456", "participant_id": "user_123"}
        assert not raw_data.processed
        assert raw_data.processing_version == "1.0"
        assert raw_data.data_id  # Should be auto-generated

    def test_raw_event_data_validation(self):
        """Test RawEventData validation."""
        # Test None content
        with pytest.raises(ValueError, match="content cannot be None"):
            RawEventData(data_type=DataType.CONVERSATION, content=None)

    def test_raw_event_data_defaults(self):
        """Test RawEventData with default values."""
        raw_data = RawEventData(content="test content")

        assert raw_data.data_type == DataType.CUSTOM
        assert raw_data.source == ""
        assert isinstance(raw_data.temporal_info, TemporalInfo)
        assert raw_data.metadata == {}
        assert not raw_data.processed
        assert raw_data.processing_version == "1.0"

    def test_raw_event_data_to_dict(self):
        """Test RawEventData serialization."""
        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        temporal_info = TemporalInfo(timestamp=timestamp, duration=60.0)

        raw_data = RawEventData(
            data_id="data_123",
            data_type=DataType.CONVERSATION,
            content=["message"],
            source="test_app",
            temporal_info=temporal_info,
            metadata={"key": "value"},
            processed=True,
            processing_version="2.0",
        )

        result = raw_data.to_dict()

        assert result["data_id"] == "data_123"
        assert result["data_type"] == "conversation"
        assert result["content"] == ["message"]
        assert result["source"] == "test_app"
        assert result["temporal_info"]["timestamp"] == "2024-01-15T10:30:00"
        assert result["temporal_info"]["duration"] == 60.0
        assert result["metadata"] == {"key": "value"}
        assert result["processed"] is True
        assert result["processing_version"] == "2.0"

    def test_raw_event_data_from_dict(self):
        """Test RawEventData deserialization."""
        data_dict = {
            "data_id": "data_123",
            "data_type": "conversation",
            "content": ["message"],
            "source": "test_app",
            "temporal_info": {
                "timestamp": "2024-01-15T10:30:00",
                "duration": 60.0,
                "timezone": "UTC",
                "precision": "second",
            },
            "metadata": {"key": "value"},
            "processed": True,
            "processing_version": "2.0",
        }

        raw_data = RawEventData.from_dict(data_dict)

        assert raw_data.data_id == "data_123"
        assert raw_data.data_type == DataType.CONVERSATION
        assert raw_data.content == ["message"]
        assert raw_data.source == "test_app"
        assert raw_data.temporal_info.timestamp == datetime(2024, 1, 15, 10, 30, 0)
        assert raw_data.temporal_info.duration == 60.0
        assert raw_data.metadata == {"key": "value"}
        assert raw_data.processed is True
        assert raw_data.processing_version == "2.0"


class TestConversationMessage:
    """Test ConversationMessage value object."""

    def test_conversation_message_creation(self):
        """Test basic ConversationMessage creation."""
        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        message = ConversationMessage(
            speaker_id="user_123",
            user_name="Alice",
            content="Hello world",
            timestamp=timestamp,
            metadata={"type": "greeting"},
        )

        assert message.speaker_id == "user_123"
        assert message.user_name == "Alice"
        assert message.content == "Hello world"
        assert message.timestamp == timestamp
        assert message.metadata == {"type": "greeting"}

    def test_conversation_message_defaults(self):
        """Test ConversationMessage with default values."""
        message = ConversationMessage(speaker_id="user_123")

        assert message.speaker_id == "user_123"
        assert message.user_name is None
        assert message.content == ""
        assert message.timestamp is None
        assert message.metadata == {}

    def test_conversation_message_immutable(self):
        """Test that ConversationMessage is frozen/immutable."""
        message = ConversationMessage(speaker_id="user_123", content="Hello")

        with pytest.raises(AttributeError):
            message.content = "Modified"


class TestConversationData:
    """Test ConversationData typed wrapper."""

    def create_valid_conversation_raw_data(self) -> RawEventData:
        """Helper to create valid conversation raw data."""
        messages = [
            {"speaker_id": "user_123", "user_name": "Alice", "content": "Hello!", "timestamp": "2024-01-15T10:30:00"},
            {
                "speaker_id": "assistant_ai",
                "user_name": "Assistant",
                "content": "Hi Alice! How can I help you?",
                "timestamp": "2024-01-15T10:30:05",
            },
        ]

        return RawEventData(data_type=DataType.CONVERSATION, content=messages, source="chat_app")

    def test_conversation_data_creation(self):
        """Test successful ConversationData creation."""
        raw_data = self.create_valid_conversation_raw_data()
        conversation_data = ConversationData(raw_data)

        assert conversation_data.raw_data == raw_data
        # user_id is no longer a property of TypedEventData
        assert conversation_data.data_id == raw_data.data_id
        assert isinstance(conversation_data.timestamp, datetime)

    def test_conversation_data_validation_wrong_type(self):
        """Test ConversationData validation with wrong data type."""
        raw_data = RawEventData(data_type=DataType.ACTIVITY, content=["message"])  # Wrong type

        with pytest.raises(ValueError, match="Expected CONVERSATION data type, got DataType.ACTIVITY"):
            ConversationData(raw_data)

    def test_conversation_data_validation_wrong_content(self):
        """Test ConversationData validation with wrong content type."""
        raw_data = RawEventData(data_type=DataType.CONVERSATION, content="not a list")  # Should be a list

        with pytest.raises(ValueError, match="Conversation content must be a list of messages"):
            ConversationData(raw_data)

    def test_conversation_data_messages_property(self):
        """Test ConversationData messages property."""
        raw_data = self.create_valid_conversation_raw_data()
        conversation_data = ConversationData(raw_data)

        messages = conversation_data.messages
        assert len(messages) == 2

        # Test first message
        assert isinstance(messages[0], ConversationMessage)
        assert messages[0].speaker_id == "user_123"
        assert messages[0].user_name == "Alice"
        assert messages[0].content == "Hello!"
        assert messages[0].timestamp == datetime(2024, 1, 15, 10, 30, 0)

        # Test second message
        assert messages[1].speaker_id == "assistant_ai"
        assert messages[1].user_name == "Assistant"
        assert messages[1].content == "Hi Alice! How can I help you?"

    def test_conversation_data_messages_with_objects(self):
        """Test ConversationData with ConversationMessage objects."""
        message_objects = [
            ConversationMessage(
                speaker_id="user_123", user_name="Alice", content="Hello!", timestamp=datetime(2024, 1, 15, 10, 30, 0)
            ),
            ConversationMessage(speaker_id="assistant_ai", content="Hi there!"),
        ]

        raw_data = RawEventData(data_type=DataType.CONVERSATION, content=message_objects)

        conversation_data = ConversationData(raw_data)
        messages = conversation_data.messages

        assert len(messages) == 2
        assert messages[0] == message_objects[0]
        assert messages[1] == message_objects[1]

    def test_conversation_data_messages_missing_fields(self):
        """Test ConversationData with messages missing fields."""
        messages = [
            {"speaker_id": "user_123", "content": "Hello"},  # Missing optional fields
            {"content": "World"},  # Missing user_id
            {},  # Empty message
        ]

        raw_data = RawEventData(data_type=DataType.CONVERSATION, content=messages)

        conversation_data = ConversationData(raw_data)
        parsed_messages = conversation_data.messages

        assert len(parsed_messages) == 3

        # First message
        assert parsed_messages[0].speaker_id == "user_123"
        assert parsed_messages[0].user_name is None
        assert parsed_messages[0].content == "Hello"

        # Second message with missing user_id
        assert parsed_messages[1].speaker_id == "unknown"
        assert parsed_messages[1].content == "World"

        # Third empty message
        assert parsed_messages[2].speaker_id == "unknown"
        assert parsed_messages[2].content == ""

    def test_get_conversation_text(self):
        """Test get_conversation_text method."""
        raw_data = self.create_valid_conversation_raw_data()
        conversation_data = ConversationData(raw_data)

        text = conversation_data.get_conversation_text()
        expected = "Alice: Hello!\nAssistant: Hi Alice! How can I help you?"

        assert text == expected

    def test_get_conversation_text_fallback_to_user_id(self):
        """Test get_conversation_text with missing user_name."""
        messages = [
            {"speaker_id": "user_123", "content": "Hello!"},
            {"speaker_id": "assistant_ai", "content": "Hi there!"},
        ]

        raw_data = RawEventData(data_type=DataType.CONVERSATION, content=messages)

        conversation_data = ConversationData(raw_data)
        text = conversation_data.get_conversation_text()
        expected = "user_123: Hello!\nassistant_ai: Hi there!"

        assert text == expected


class TestCreateTypedData:
    """Test the factory function for creating typed data."""

    def test_create_conversation_data(self):
        """Test creating ConversationData via factory."""
        raw_data = RawEventData(
            data_type=DataType.CONVERSATION, content=[{"speaker_id": "user_123", "content": "Hello"}]
        )

        typed_data = create_typed_data(raw_data)

        assert isinstance(typed_data, ConversationData)
        assert typed_data.raw_data == raw_data

    def test_create_unsupported_data_type(self):
        """Test creating typed data for unsupported type."""
        raw_data = RawEventData(data_type=DataType.ACTIVITY, content={"activity": "browsing"})  # Not yet implemented

        typed_data = create_typed_data(raw_data)

        # Should return generic TypedEventData
        assert isinstance(typed_data, TypedEventData)
        assert not isinstance(typed_data, ConversationData)
        assert typed_data.raw_data == raw_data

    def test_create_custom_data_type(self):
        """Test creating typed data for custom type."""
        raw_data = RawEventData(data_type=DataType.CUSTOM, content="custom content")

        typed_data = create_typed_data(raw_data)

        # Should return generic TypedEventData
        assert isinstance(typed_data, TypedEventData)
        assert typed_data.raw_data == raw_data


class TestTypedEventDataProperties:
    """Test TypedEventData base class properties."""

    def test_typed_user_data_properties(self):
        """Test that all properties work correctly."""
        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        temporal_info = TemporalInfo(timestamp=timestamp)

        raw_data = RawEventData(
            data_id="data_123",
            data_type=DataType.CONVERSATION,
            content=[{"speaker_id": "user_123", "content": "Hello"}],
            temporal_info=temporal_info,
            metadata={"session": "session_456"},
        )

        typed_data = create_typed_data(raw_data)

        assert typed_data.data_id == "data_123"
        # user_id is no longer a property of TypedEventData
        assert typed_data.timestamp == timestamp
        assert typed_data.content == [{"speaker_id": "user_123", "content": "Hello"}]
        assert typed_data.metadata == {"session": "session_456"}


# Integration tests
class TestDataTypesIntegration:
    """Integration tests for data types working together."""

    def test_end_to_end_conversation_processing(self):
        """Test complete workflow from dict to ConversationData."""
        # Start with dictionary data (e.g., from API)
        conversation_dict = {
            "speaker_id": "user_123",
            "data_type": "conversation",
            "content": [
                {
                    "speaker_id": "user_123",
                    "user_name": "Alice",
                    "content": "I need help planning my vacation",
                    "timestamp": "2024-01-15T10:30:00",
                },
                {
                    "speaker_id": "assistant_ai",
                    "user_name": "Travel Assistant",
                    "content": "I'd be happy to help! Where would you like to go?",
                    "timestamp": "2024-01-15T10:30:05",
                },
            ],
            "source": "travel_chat_app",
            "temporal_info": {"timestamp": "2024-01-15T10:30:00", "duration": 300.0, "timezone": "UTC"},
            "metadata": {"session_id": "session_456", "topic": "travel_planning"},
        }

        # Convert to RawEventData
        raw_data = RawEventData.from_dict(conversation_dict)

        # Create typed data
        typed_data = create_typed_data(raw_data)

        # Verify it's the correct type
        assert isinstance(typed_data, ConversationData)

        # Test functionality
        messages = typed_data.messages
        assert len(messages) == 2
        assert messages[0].user_name == "Alice"
        assert messages[1].user_name == "Travel Assistant"

        conversation_text = typed_data.get_conversation_text()
        expected_text = "Alice: I need help planning my vacation\nTravel Assistant: I'd be happy to help! Where would you like to go?"
        assert conversation_text == expected_text

        # Test serialization round-trip
        serialized = raw_data.to_dict()
        restored_raw_data = RawEventData.from_dict(serialized)
        restored_typed_data = create_typed_data(restored_raw_data)

        assert isinstance(restored_typed_data, ConversationData)
        assert len(restored_typed_data.messages) == 2
        assert restored_typed_data.get_conversation_text() == conversation_text
