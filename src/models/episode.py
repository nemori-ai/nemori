"""
Episode Data Model
Episode data model
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
from .message import Message


@dataclass
class Episode:
    """Episode data model"""
    
    # Required fields as requested by user
    title: str                                      # Episode title
    content: str                                    # Episode content
    original_messages: List[Dict[str, Any]]         # Original message list
    message_count: int                              # Message count
    boundary_reason: str                            # Boundary detection reason
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))  # Episode ID
    user_id: str = ""                               # User ID
    created_at: datetime = field(default_factory=datetime.now)          # Creation time
    timestamp: datetime = field(default_factory=datetime.now)           # Episode timestamp (independent of message time)
    
    # Optional internal fields
    metadata: Dict[str, Any] = field(default_factory=dict)  # Metadata
    tags: List[str] = field(default_factory=list)           # Tags
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "episode_id": self.episode_id,
            "user_id": self.user_id,
            "title": self.title,
            "content": self.content,
            "original_messages": self.original_messages,
            "message_count": self.message_count,
            "boundary_reason": self.boundary_reason,
            "created_at": self.created_at.isoformat(),
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Episode':
        """Create episode from dictionary"""
        # For backward compatibility, use created_at if no timestamp field
        timestamp = data.get("timestamp", data.get("created_at"))
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        return cls(
            episode_id=data.get("episode_id", str(uuid.uuid4())),
            user_id=data["user_id"],
            title=data["title"],
            content=data["content"],
            original_messages=data["original_messages"],
            message_count=data["message_count"],
            boundary_reason=data["boundary_reason"],
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data["created_at"], str) else data["created_at"],
            timestamp=timestamp,
            metadata=data.get("metadata", {}),
            tags=data.get("tags", [])
        )
    
    @classmethod
    def from_messages(cls, messages: List[Message], user_id: str, title: str = "", 
                     content: str = "", boundary_reason: str = "", 
                     timestamp: Optional[datetime] = None) -> 'Episode':
        """Create episode from message list"""
        # If no timestamp provided, use current time
        if timestamp is None:
            timestamp = datetime.now()
            
        return cls(
            user_id=user_id,
            title=title,
            content=content,
            original_messages=[msg.to_dict(clean_metadata=True) for msg in messages],
            message_count=len(messages),
            boundary_reason=boundary_reason,
            timestamp=timestamp
        )
    
    def get_messages(self) -> List[Message]:
        """Get original message list"""
        return [Message.from_dict(msg_data) for msg_data in self.original_messages]
    
    def get_conversation_text(self) -> str:
        """Get conversation text"""
        lines = []
        for msg_data in self.original_messages:
            lines.append(f"{msg_data['role']}: {msg_data['content']}")
        return "\n".join(lines)
    
    def add_tag(self, tag: str) -> None:
        """Add tag"""
        if tag not in self.tags:
            self.tags.append(tag)
    
    def remove_tag(self, tag: str) -> None:
        """Remove tag"""
        if tag in self.tags:
            self.tags.remove(tag)
    
    def has_tag(self, tag: str) -> bool:
        """Check if tag exists"""
        return tag in self.tags
    
    def get_summary(self) -> str:
        """Get episode summary"""
        return f"Episode: {self.title} ({self.message_count} messages)"
    
    def __str__(self) -> str:
        return f"Episode(id={self.episode_id[:8]}, title={self.title[:30]}, messages={self.message_count})"
    
    def __repr__(self) -> str:
        return self.__str__() 