"""
Core data type abstractions for Nemori.

This module defines the fundamental data structures that represent different
types of events and raw data that can be transformed into episodes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4


class DataType(Enum):
    """Enumeration of supported raw event data types."""

    CONVERSATION = "conversation"
    ACTIVITY = "activity"
    LOCATION = "location"
    MEDIA = "media"
    DOCUMENT = "document"
    SENSOR = "sensor"
    EXTERNAL = "external"
    CUSTOM = "custom"


@dataclass(frozen=True)
class TemporalInfo:
    """Temporal information associated with event data."""

    timestamp: datetime
    duration: float | None = None  # Duration in seconds
    timezone: str | None = None
    precision: str = "second"  # second, minute, hour, day

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "duration": self.duration,
            "timezone": self.timezone,
            "precision": self.precision,
        }


@dataclass
class RawEventData:
    """
    Abstract representation of raw event data from various sources.

    This is the unified interface for all types of events
    that can be processed into episodic memories.
    """

    # Core identification
    data_id: str = field(default_factory=lambda: str(uuid4()))
    data_type: DataType = DataType.CUSTOM

    # Core content
    content: Any = None
    source: str = ""  # Source identifier (app, device, service, etc.)

    # Temporal information
    temporal_info: TemporalInfo = field(default_factory=lambda: TemporalInfo(datetime.now()))

    # Flexible metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Processing status
    processed: bool = False
    processing_version: str = "1.0"

    def __post_init__(self):
        """Validate and normalize data after initialization."""
        if self.content is None:
            raise ValueError("content cannot be None")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "data_id": self.data_id,
            "data_type": self.data_type.value,
            "content": self.content,
            "source": self.source,
            "temporal_info": self.temporal_info.to_dict(),
            "metadata": self.metadata,
            "processed": self.processed,
            "processing_version": self.processing_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RawEventData":
        """Create instance from dictionary representation."""
        temporal_data = data.get("temporal_info", {})
        temporal_info = TemporalInfo(
            timestamp=datetime.fromisoformat(temporal_data.get("timestamp", datetime.now().isoformat())),
            duration=temporal_data.get("duration"),
            timezone=temporal_data.get("timezone"),
            precision=temporal_data.get("precision", "second"),
        )

        return cls(
            data_id=data.get("data_id", str(uuid4())),
            data_type=DataType(data.get("data_type", "custom")),
            content=data["content"],
            source=data.get("source", ""),
            temporal_info=temporal_info,
            metadata=data.get("metadata", {}),
            processed=data.get("processed", False),
            processing_version=data.get("processing_version", "1.0"),
        )


class TypedEventData(ABC):
    """
    Abstract base class for type-specific event data.

    This allows for type-specific validation and methods while
    maintaining the unified RawEventData interface.
    """

    def __init__(self, raw_data: RawEventData):
        self.raw_data = raw_data
        self._validate()

    @abstractmethod
    def _validate(self) -> None:
        """Validate that the raw data is appropriate for this type."""
        pass

    @property
    def data_id(self) -> str:
        return self.raw_data.data_id


    @property
    def timestamp(self) -> datetime:
        return self.raw_data.temporal_info.timestamp

    @property
    def content(self) -> Any:
        return self.raw_data.content

    @property
    def metadata(self) -> dict[str, Any]:
        return self.raw_data.metadata


@dataclass(frozen=True)
class ConversationMessage:
    """Represents a single message in a conversation."""

    speaker_id: str  # Who said this message
    user_name: str | None = None  # Display name (optional)
    content: str = ""
    timestamp: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ConversationData(TypedEventData):
    """Type-specific wrapper for conversation data."""

    def _validate(self) -> None:
        if self.raw_data.data_type != DataType.CONVERSATION:
            raise ValueError(f"Expected CONVERSATION data type, got {self.raw_data.data_type}")

        if not isinstance(self.raw_data.content, list):
            raise ValueError("Conversation content must be a list of messages")

    @property
    def messages(self) -> list[ConversationMessage]:
        """Get messages as structured ConversationMessage objects."""
        messages = []
        for msg_data in self.raw_data.content:
            if isinstance(msg_data, dict):
                messages.append(
                    ConversationMessage(
                        speaker_id=msg_data.get("speaker_id", msg_data.get("user_id", "unknown")),
                        user_name=msg_data.get("user_name"),
                        content=msg_data.get("content", ""),
                        timestamp=datetime.fromisoformat(msg_data["timestamp"]) if "timestamp" in msg_data else None,
                        metadata=msg_data.get("metadata", {}),
                    )
                )
            elif isinstance(msg_data, ConversationMessage):
                messages.append(msg_data)
        return messages

    def get_conversation_text(self) -> str:
        """Get the full conversation as formatted text."""
        lines = []
        for msg in self.messages:
            speaker = msg.user_name if msg.user_name else msg.speaker_id
            lines.append(f"{speaker}: {msg.content}")
        return "\n".join(lines)


# Factory function for creating typed data
def create_typed_data(raw_data: RawEventData) -> TypedEventData:
    """Factory function to create appropriate typed data wrapper."""

    type_map = {
        DataType.CONVERSATION: ConversationData,
        # TODO: Add other types as we implement them
        # DataType.ACTIVITY: ActivityData,
        # DataType.LOCATION: LocationData,
        # DataType.MEDIA: MediaData,
        # DataType.DOCUMENT: DocumentData,
        # DataType.SENSOR: SensorData,
        # DataType.EXTERNAL: ExternalData,
    }

    typed_class = type_map.get(raw_data.data_type)
    if typed_class:
        return typed_class(raw_data)
    else:
        # For unsupported types, return a generic wrapper
        class GenericTypedData(TypedEventData):
            def _validate(self) -> None:
                pass  # No specific validation for generic types

        return GenericTypedData(raw_data)
