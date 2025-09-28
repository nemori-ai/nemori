"""
Semantic Memory Data Model
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid


@dataclass
class SemanticMemory:
    """Semantic memory data model"""
    
    # Core fields
    content: str                                        # Semantic knowledge content
    knowledge_type: str                                 # Knowledge type
    user_id: str                                        # User ID
    created_at: datetime = field(default_factory=datetime.now)     # Creation time (time identifier)
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))  # Memory ID
    
    # Source information
    source_episodes: List[str] = field(default_factory=list)  # Source episode ID list
    confidence: float = 0.8                             # Confidence
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)  # Metadata
    tags: List[str] = field(default_factory=list)           # Tags
    
    # Update information
    updated_at: Optional[datetime] = None               # Update time
    revision_count: int = 1                             # Revision count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "memory_id": self.memory_id,
            "user_id": self.user_id,
            "content": self.content,
            "knowledge_type": self.knowledge_type,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "source_episodes": self.source_episodes,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "tags": self.tags,
            "revision_count": self.revision_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SemanticMemory':
        """Create semantic memory from dictionary"""
        return cls(
            memory_id=data.get("memory_id", str(uuid.uuid4())),
            user_id=data["user_id"],
            content=data["content"],
            knowledge_type=data["knowledge_type"],
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data["created_at"], str) else data["created_at"],
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
            source_episodes=data.get("source_episodes", []),
            confidence=data.get("confidence", 0.8),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
            revision_count=data.get("revision_count", 1)
        )
    
    def update_content(self, new_content: str, additional_episodes: List[str] = None) -> None:
        """Update content"""
        self.content = new_content
        self.updated_at = datetime.now()
        self.revision_count += 1
        
        if additional_episodes:
            # Merge source episodes and remove duplicates
            all_episodes = set(self.source_episodes)
            all_episodes.update(additional_episodes)
            self.source_episodes = list(all_episodes)
    
    def add_source_episode(self, episode_id: str) -> None:
        """Add source episode"""
        if episode_id not in self.source_episodes:
            self.source_episodes.append(episode_id)
    
    def remove_source_episode(self, episode_id: str) -> None:
        """Remove source episode"""
        if episode_id in self.source_episodes:
            self.source_episodes.remove(episode_id)
    
    def add_tag(self, tag: str) -> None:
        """Add tag"""
        if tag not in self.tags:
            self.tags.append(tag)
    
    def remove_tag(self, tag: str) -> None:
        """Remove tag"""
        if tag in self.tags:
            self.tags.remove(tag)
    
    def has_tag(self, tag: str) -> bool:
        """Check if has tag"""
        return tag in self.tags
    
    def get_age_days(self) -> int:
        """Get memory age (days)"""
        return (datetime.now() - self.created_at).days
    
    def is_recent(self, days: int = 30) -> bool:
        """Check if is recent memory"""
        return self.get_age_days() <= days
    
    def get_summary(self) -> str:
        """Get memory summary"""
        return f"[{self.knowledge_type}] {self.content[:50]}..."
    
    def __str__(self) -> str:
        return f"SemanticMemory(id={self.memory_id[:8]}, type={self.knowledge_type}, content={self.content[:30]}...)"
    
    def __repr__(self) -> str:
        return self.__str__() 