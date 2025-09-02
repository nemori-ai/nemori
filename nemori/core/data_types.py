"""
Core data type abstractions for Nemori.

This module defines the fundamental data structures that represent different
types of events and raw data that can be transformed into episodes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
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

    def get_conversation_text(self, include_timestamps: bool = False) -> str:
        """Get the full conversation as formatted text."""
        lines = []
        for msg in self.messages:
            speaker = msg.user_name if msg.user_name else msg.speaker_id
            if include_timestamps and msg.timestamp:
                time_str = msg.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                lines.append(f"[{time_str}] {speaker}: {msg.content}")
            else:
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


# === Semantic Memory Data Types ===


class RelationshipType(Enum):
    """Types of semantic relationships between nodes."""

    RELATED = "related"  # General relationship
    EVOLVED_FROM = "evolved_from"  # One concept evolved from another
    PART_OF = "part_of"  # Part-whole relationship
    SIMILAR_TO = "similar_to"  # Similarity relationship
    OPPOSITE_TO = "opposite_to"  # Opposition relationship
    TEMPORAL = "temporal"  # Time-based relationship


@dataclass
class SemanticNode:
    """
    Represents a single piece of semantic knowledge discovered through gap analysis.

    This is the core unit of semantic memory, capturing private domain knowledge
    that evolves over time while maintaining bidirectional links to episodes.
    """

    # Core identification
    node_id: str = field(default_factory=lambda: str(uuid4()))
    owner_id: str = ""

    # Knowledge content
    key: str = ""  # Knowledge key/identifier (e.g., "John的研究方向")
    value: str = ""  # Knowledge content (e.g., "AI Agent行为规划")
    context: str = ""  # Original context where discovered

    # Confidence and evolution
    confidence: float = 1.0  # Confidence in this knowledge [0-1]
    version: int = 1  # Version number for evolution tracking
    evolution_history: list[str] = field(default_factory=list)  # Previous values

    # Temporal information
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    last_accessed: datetime | None = None

    # Discovery metadata
    discovery_episode_id: str | None = None  # Episode that led to discovery
    discovery_method: str = "diff_analysis"  # How this was discovered

    # Bidirectional associations
    linked_episode_ids: list[str] = field(default_factory=list)  # Episodes that reference this knowledge
    evolution_episode_ids: list[str] = field(default_factory=list)  # Episodes that caused evolution

    # Search optimization
    search_keywords: list[str] = field(default_factory=list)  # Keywords for similarity search
    embedding_vector: list[float] | None = None  # Optional vector for semantic search

    # Usage statistics
    access_count: int = 0
    relevance_score: float = 0.0
    importance_score: float = 0.0

    def __post_init__(self):
        """Validate semantic node after initialization."""
        if not self.owner_id:
            raise ValueError("owner_id is required")

        if not self.key:
            raise ValueError("key is required")

        if not self.value:
            raise ValueError("value is required")

        # Ensure confidence is within valid range
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0")

        # Ensure version is positive
        if self.version < 1:
            raise ValueError("version must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "node_id": self.node_id,
            "owner_id": self.owner_id,
            "key": self.key,
            "value": self.value,
            "context": self.context,
            "confidence": self.confidence,
            "version": self.version,
            "evolution_history": self.evolution_history,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "discovery_episode_id": self.discovery_episode_id,
            "discovery_method": self.discovery_method,
            "linked_episode_ids": self.linked_episode_ids,
            "evolution_episode_ids": self.evolution_episode_ids,
            "search_keywords": self.search_keywords,
            "embedding_vector": self.embedding_vector,
            "access_count": self.access_count,
            "relevance_score": self.relevance_score,
            "importance_score": self.importance_score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SemanticNode":
        """Create semantic node from dictionary representation."""
        return cls(
            node_id=data.get("node_id", str(uuid4())),
            owner_id=data["owner_id"],
            key=data["key"],
            value=data["value"],
            context=data.get("context", ""),
            confidence=data.get("confidence", 1.0),
            version=data.get("version", 1),
            evolution_history=data.get("evolution_history", []),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
            last_updated=datetime.fromisoformat(data.get("last_updated", datetime.now().isoformat())),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None,
            discovery_episode_id=data.get("discovery_episode_id"),
            discovery_method=data.get("discovery_method", "diff_analysis"),
            linked_episode_ids=data.get("linked_episode_ids", []),
            evolution_episode_ids=data.get("evolution_episode_ids", []),
            search_keywords=data.get("search_keywords", []),
            embedding_vector=data.get("embedding_vector"),
            access_count=data.get("access_count", 0),
            relevance_score=data.get("relevance_score", 0.0),
            importance_score=data.get("importance_score", 0.0),
        )

    def mark_accessed(self) -> "SemanticNode":
        """Mark this semantic node as accessed and return updated copy."""
        return replace(self, last_accessed=datetime.now(), access_count=self.access_count + 1)

    def evolve(self, new_value: str, new_context: str, evolution_episode_id: str) -> "SemanticNode":
        """Create evolved version of this semantic node."""
        return replace(
            self,
            value=new_value,
            context=new_context,
            version=self.version + 1,
            evolution_history=self.evolution_history + [self.value],
            evolution_episode_ids=self.evolution_episode_ids + [evolution_episode_id],
            last_updated=datetime.now(),
        )

    def add_linked_episode(self, episode_id: str) -> "SemanticNode":
        """Add linked episode ID if not already present."""
        if episode_id not in self.linked_episode_ids:
            return replace(
                self, linked_episode_ids=self.linked_episode_ids + [episode_id], last_accessed=datetime.now()
            )
        return self

    def update_confidence(self, confidence_delta: float) -> "SemanticNode":
        """Update confidence score within valid bounds."""
        new_confidence = max(0.0, min(1.0, self.confidence + confidence_delta))
        return replace(self, confidence=new_confidence)


@dataclass(frozen=True)
class SemanticRelationship:
    """
    Represents relationships between semantic nodes.

    Maintains simple bidirectional associations between semantic knowledge
    without complex knowledge graph overhead.
    """

    # Core identification
    relationship_id: str = field(default_factory=lambda: str(uuid4()))

    # Relationship definition
    source_node_id: str = ""
    target_node_id: str = ""
    relationship_type: RelationshipType = RelationshipType.RELATED

    # Relationship properties
    strength: float = 0.5  # Relationship strength [0-1]
    description: str = ""  # Optional description

    # Temporal information
    created_at: datetime = field(default_factory=datetime.now)
    last_reinforced: datetime = field(default_factory=datetime.now)

    # Discovery context
    discovery_episode_id: str | None = None

    def __post_init__(self):
        """Validate semantic relationship after initialization."""
        if not self.source_node_id:
            raise ValueError("source_node_id is required")

        if not self.target_node_id:
            raise ValueError("target_node_id is required")

        if self.source_node_id == self.target_node_id:
            raise ValueError("source_node_id and target_node_id cannot be the same")

        # Ensure strength is within valid range
        if not (0.0 <= self.strength <= 1.0):
            raise ValueError("strength must be between 0.0 and 1.0")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "relationship_id": self.relationship_id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "relationship_type": self.relationship_type.value,
            "strength": self.strength,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "last_reinforced": self.last_reinforced.isoformat(),
            "discovery_episode_id": self.discovery_episode_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SemanticRelationship":
        """Create semantic relationship from dictionary representation."""
        return cls(
            relationship_id=data.get("relationship_id", str(uuid4())),
            source_node_id=data["source_node_id"],
            target_node_id=data["target_node_id"],
            relationship_type=RelationshipType(data.get("relationship_type", "related")),
            strength=data.get("strength", 0.5),
            description=data.get("description", ""),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
            last_reinforced=datetime.fromisoformat(data.get("last_reinforced", datetime.now().isoformat())),
            discovery_episode_id=data.get("discovery_episode_id"),
        )

    def reinforce(self, strength_boost: float = 0.1) -> "SemanticRelationship":
        """Create reinforced version of this relationship."""
        new_strength = min(1.0, self.strength + strength_boost)
        return replace(self, strength=new_strength, last_reinforced=datetime.now())

    def is_bidirectional_equivalent(self, other: "SemanticRelationship") -> bool:
        """Check if this relationship is equivalent to another in bidirectional sense."""
        return (
            (self.source_node_id == other.source_node_id and self.target_node_id == other.target_node_id)
            or (self.source_node_id == other.target_node_id and self.target_node_id == other.source_node_id)
        ) and self.relationship_type == other.relationship_type
