"""
Unified Episode data structures for Nemori.

This module defines the Episode class - the unified output format that all
different types of user experiences are transformed into.
"""

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from .data_types import DataType, TemporalInfo


class EpisodeType(Enum):
    """Types of episodes based on their origin and characteristics."""

    CONVERSATIONAL = "conversational"  # From conversation data
    BEHAVIORAL = "behavioral"  # From activity/behavior data
    SPATIAL = "spatial"  # From location data
    CREATIVE = "creative"  # From document/media creation
    PHYSIOLOGICAL = "physiological"  # From sensor/health data
    SOCIAL = "social"  # From external social data
    MIXED = "mixed"  # From multiple data sources
    SYNTHETIC = "synthetic"  # Generated/inferred episodes


class EpisodeLevel(Enum):
    """Hierarchical levels of episode abstraction."""

    ATOMIC = 1  # Single event/interaction
    COMPOUND = 2  # Multiple related events
    THEMATIC = 3  # Pattern-based higher-level insights
    ARCHIVAL = 4  # Long-term synthesized understanding


@dataclass(frozen=True)
class EpisodeMetadata:
    """Rich metadata associated with an episode."""

    # Source information
    source_data_ids: list[str] = field(default_factory=list)
    source_types: set[DataType] = field(default_factory=set)
    processing_timestamp: datetime = field(default_factory=datetime.now)
    processing_version: str = "1.0"

    # Content analysis
    entities: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    emotions: list[str] = field(default_factory=list)
    key_points: list[str] = field(default_factory=list)

    # Temporal context
    time_references: list[str] = field(default_factory=list)
    duration_seconds: float | None = None

    # Quality metrics
    confidence_score: float = 1.0
    completeness_score: float = 1.0
    relevance_score: float = 1.0

    # Episode relationships
    related_episode_ids: list[str] = field(default_factory=list)

    # Custom metadata
    custom_fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "source_data_ids": self.source_data_ids,
            "source_types": [dt.value for dt in self.source_types],
            "processing_timestamp": self.processing_timestamp.isoformat(),
            "processing_version": self.processing_version,
            "entities": self.entities,
            "topics": self.topics,
            "emotions": self.emotions,
            "key_points": self.key_points,
            "time_references": self.time_references,
            "duration_seconds": self.duration_seconds,
            "confidence_score": self.confidence_score,
            "completeness_score": self.completeness_score,
            "relevance_score": self.relevance_score,
            "related_episode_ids": self.related_episode_ids,
            "custom_fields": self.custom_fields,
        }


@dataclass
class Episode:
    """
    Unified episodic memory representation.

    All types of user experiences are transformed into this common format,
    making them searchable and retrievable regardless of their original source.
    """

    # Core identification
    episode_id: str = field(default_factory=lambda: str(uuid4()))
    owner_id: str = ""

    # Episode classification
    episode_type: EpisodeType = EpisodeType.MIXED
    level: EpisodeLevel = EpisodeLevel.ATOMIC

    # Core narrative content
    title: str = ""
    content: str = ""
    summary: str = ""

    # Temporal information
    temporal_info: TemporalInfo = field(default_factory=lambda: TemporalInfo(datetime.now()))

    # Rich metadata
    metadata: EpisodeMetadata = field(default_factory=EpisodeMetadata)

    # Structured data (type-specific)
    structured_data: dict[str, Any] = field(default_factory=dict)

    # Search and retrieval optimization
    search_keywords: list[str] = field(default_factory=list)
    embedding_vector: list[float] | None = None

    # Recall and importance
    recall_count: int = 0
    importance_score: float = 0.0
    last_accessed: datetime | None = None

    def __post_init__(self):
        """Validate and normalize episode after initialization."""
        if not self.owner_id:
            raise ValueError("owner_id is required")

        if not self.title and not self.content:
            raise ValueError("Either title or content must be provided")

        # Auto-generate summary if not provided
        if not self.summary and self.content:
            # Simple summary - first 100 characters
            self.summary = self.content[:100] + "..." if len(self.content) > 100 else self.content

        # Auto-generate title if not provided
        if not self.title and self.summary:
            # Simple title - first line or sentence
            first_line = self.summary.split("\n")[0]
            self.title = first_line[:50] + "..." if len(first_line) > 50 else first_line

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "episode_id": self.episode_id,
            "owner_id": self.owner_id,
            "episode_type": self.episode_type.value,
            "level": self.level.value,
            "title": self.title,
            "content": self.content,
            "summary": self.summary,
            "temporal_info": self.temporal_info.to_dict(),
            "metadata": self.metadata.to_dict(),
            "structured_data": self.structured_data,
            "search_keywords": self.search_keywords,
            "embedding_vector": self.embedding_vector,
            "recall_count": self.recall_count,
            "importance_score": self.importance_score,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Episode":
        """Create episode from dictionary representation."""

        # Parse temporal info
        temporal_data = data.get("temporal_info", {})
        temporal_info = TemporalInfo(
            timestamp=datetime.fromisoformat(temporal_data.get("timestamp", datetime.now().isoformat())),
            duration=temporal_data.get("duration"),
            timezone=temporal_data.get("timezone"),
            precision=temporal_data.get("precision", "second"),
        )

        # Parse metadata
        metadata_data = data.get("metadata", {})
        metadata = EpisodeMetadata(
            source_data_ids=metadata_data.get("source_data_ids", []),
            source_types={DataType(dt) for dt in metadata_data.get("source_types", [])},
            processing_timestamp=datetime.fromisoformat(
                metadata_data.get("processing_timestamp", datetime.now().isoformat())
            ),
            processing_version=metadata_data.get("processing_version", "1.0"),
            entities=metadata_data.get("entities", []),
            topics=metadata_data.get("topics", []),
            emotions=metadata_data.get("emotions", []),
            key_points=metadata_data.get("key_points", []),
            time_references=metadata_data.get("time_references", []),
            duration_seconds=metadata_data.get("duration_seconds"),
            confidence_score=metadata_data.get("confidence_score", 1.0),
            completeness_score=metadata_data.get("completeness_score", 1.0),
            relevance_score=metadata_data.get("relevance_score", 1.0),
            related_episode_ids=metadata_data.get("related_episode_ids", []),
            custom_fields=metadata_data.get("custom_fields", {}),
        )

        return cls(
            episode_id=data.get("episode_id", str(uuid4())),
            owner_id=data["owner_id"],
            episode_type=EpisodeType(data.get("episode_type", "mixed")),
            level=EpisodeLevel(data.get("level", 1)),
            title=data.get("title", ""),
            content=data.get("content", ""),
            summary=data.get("summary", ""),
            temporal_info=temporal_info,
            metadata=metadata,
            structured_data=data.get("structured_data", {}),
            search_keywords=data.get("search_keywords", []),
            embedding_vector=data.get("embedding_vector"),
            recall_count=data.get("recall_count", 0),
            importance_score=data.get("importance_score", 0.0),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None,
        )

    def mark_accessed(self) -> None:
        """Mark this episode as accessed (for recall tracking)."""
        self.recall_count += 1
        self.last_accessed = datetime.now()

    def add_related_episode(self, episode_id: str) -> None:
        """Add a related episode ID."""
        if episode_id not in self.metadata.related_episode_ids:
            # Since metadata is frozen, we need to create a new one
            new_related = list(self.metadata.related_episode_ids) + [episode_id]
            self.metadata = replace(self.metadata, related_episode_ids=new_related)

    def update_importance(self, importance_score: float) -> None:
        """Update the importance score of this episode."""
        self.importance_score = max(0.0, min(1.0, importance_score))

    def get_display_text(self) -> str:
        """Get formatted text for display purposes."""
        timestamp_str = self.temporal_info.timestamp.strftime("%Y-%m-%d %H:%M")
        return f"[{timestamp_str}] {self.title}\n{self.summary}"

    def is_recent(self, hours: int = 24) -> bool:
        """Check if episode is recent (within specified hours)."""
        time_diff = datetime.now() - self.temporal_info.timestamp
        return time_diff.total_seconds() < (hours * 3600)

    def matches_keywords(self, keywords: list[str]) -> bool:
        """Check if episode content matches any of the given keywords."""
        text_to_search = f"{self.title} {self.content} {self.summary}".lower()
        return any(keyword.lower() in text_to_search for keyword in keywords)
