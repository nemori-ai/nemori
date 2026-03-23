"""Domain models for Nemori memory system."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import uuid

# Multimodal content types (OpenAI content array standard)
ContentPart = dict[str, Any]
MessageContent = str | list[ContentPart]


@dataclass
class Message:
    """A single conversation message."""

    role: str
    content: MessageContent
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def text_content(self, include_placeholders: bool = True) -> str:
        """Extract text parts only. Used for embedding, search, token counting."""
        if isinstance(self.content, str):
            return self.content
        parts = []
        for part in self.content:
            if part.get("type") == "text":
                parts.append(part["text"])
            elif part.get("type") == "image_url" and include_placeholders:
                parts.append("[image]")
        return " ".join(parts)

    def has_images(self) -> bool:
        """Check if message contains image content."""
        if isinstance(self.content, str):
            return False
        return any(p.get("type") == "image_url" for p in self.content)

    def image_urls(self) -> list[str]:
        """Extract image URLs from content array."""
        if isinstance(self.content, str):
            return []
        return [
            p["image_url"]["url"]
            for p in self.content
            if p.get("type") == "image_url"
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        ts = data.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        elif ts is None:
            ts = datetime.now()
        return cls(
            message_id=data.get("message_id", str(uuid.uuid4())),
            role=data["role"],
            content=data["content"],
            timestamp=ts,
            metadata=data.get("metadata", {}),
        )


@dataclass
class Episode:
    """An episodic memory derived from conversation messages."""

    user_id: str
    title: str
    content: str
    source_messages: list[dict[str, Any]]
    agent_id: str = "default"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "title": self.title,
            "content": self.content,
            "source_messages": self.source_messages,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Episode:
        created = data.get("created_at")
        if isinstance(created, str):
            created = datetime.fromisoformat(created)
        elif created is None:
            created = datetime.now()
        updated = data.get("updated_at")
        if isinstance(updated, str):
            updated = datetime.fromisoformat(updated)
        elif updated is None:
            updated = datetime.now()
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            user_id=data["user_id"],
            agent_id=data.get("agent_id", "default"),
            title=data["title"],
            content=data["content"],
            source_messages=data.get("source_messages", []),
            metadata=data.get("metadata", {}),
            created_at=created,
            updated_at=updated,
        )


@dataclass
class SemanticMemory:
    """A semantic knowledge fact extracted from episodes."""

    user_id: str
    content: str
    memory_type: str
    agent_id: str = "default"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    embedding: list[float] | None = None
    source_episode_id: str | None = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "content": self.content,
            "memory_type": self.memory_type,
            "source_episode_id": self.source_episode_id,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SemanticMemory:
        created = data.get("created_at")
        if isinstance(created, str):
            created = datetime.fromisoformat(created)
        elif created is None:
            created = datetime.now()
        updated = data.get("updated_at")
        if isinstance(updated, str):
            updated = datetime.fromisoformat(updated)
        elif updated is None:
            updated = datetime.now()
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            user_id=data["user_id"],
            agent_id=data.get("agent_id", "default"),
            content=data["content"],
            memory_type=data.get("memory_type", data.get("knowledge_type", "")),
            source_episode_id=data.get("source_episode_id"),
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {}),
            created_at=created,
            updated_at=updated,
        )


@dataclass
class HealthResult:
    """Health check result for the Nemori system."""

    db: bool
    llm: bool
    embedding: bool
    diagnostics: dict[str, Any]

    @property
    def healthy(self) -> bool:
        return self.db and self.llm and self.embedding
