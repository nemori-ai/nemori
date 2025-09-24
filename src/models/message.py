"""
Message Data Models
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid


@dataclass
class Message:
    """Message data model"""
    
    role: str                           # Message role: user, assistant, system
    content: str                        # Message content
    timestamp: datetime = field(default_factory=datetime.now)  # Message timestamp
    metadata: Dict[str, Any] = field(default_factory=dict)     # Metadata
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))  # Message ID
    
    def to_dict(self, clean_metadata: bool = True) -> Dict[str, Any]:
        """
        Convert to dictionary
        
        Args:
            clean_metadata: Whether to clean redundant information from metadata
        """
        metadata = self.metadata
        
        if clean_metadata:
            metadata = self._clean_metadata(self.metadata)
        
        return {
            "message_id": self.message_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": metadata
        }
    
    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean redundant information from metadata
        
        Args:
            metadata: Original metadata
            
        Returns:
            Cleaned metadata with redundant fields removed
        """
        if not metadata:
            return {}
        
        # 需要移除的重复字段
        redundant_fields = {
            'original_text',  # 与content重复
            'timestamp'       # 与主timestamp重复
        }
        
        # 保留有价值的唯一字段
        valuable_fields = {
            'original_speaker',     # 可能与role不同，保留原始说话人信息
            'dataset_timestamp',    # 原始数据集时间格式，有追溯价值
            'image_url',           # 多模态内容
            'blip_caption',        # 图像描述
            'search_query',        # 搜索查询
            'has_multimodal_content'  # 内容类型标识
        }
        
        # 只保留有价值的字段，移除重复字段
        cleaned_metadata = {}
        for key, value in metadata.items():
            if key not in redundant_fields:
                # 只保留非空且有价值的字段
                if value is not None or key in valuable_fields:
                    cleaned_metadata[key] = value
        
        return cleaned_metadata
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary"""
        return cls(
            message_id=data.get("message_id", str(uuid.uuid4())),
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data["timestamp"], str) else data["timestamp"],
            metadata=data.get("metadata", {})
        )
    
    def __str__(self) -> str:
        return f"[{self.role}] {self.content[:50]}..."


@dataclass
class MessageBuffer:
    """Message buffer data model"""
    
    owner_id: str                       # Buffer owner ID
    messages: List[Message] = field(default_factory=list)  # Message list
    created_at: datetime = field(default_factory=datetime.now)  # Buffer creation time
    last_updated: datetime = field(default_factory=datetime.now)  # Last update time
    metadata: Dict[str, Any] = field(default_factory=dict)      # Metadata
    
    def add_message(self, message: Message) -> None:
        """Add message to buffer"""
        self.messages.append(message)
        self.last_updated = datetime.now()
    
    def add_messages(self, messages: List[Message]) -> None:
        """Batch add messages"""
        self.messages.extend(messages)
        self.last_updated = datetime.now()
    
    def clear(self) -> List[Message]:
        """Clear buffer and return cleared messages"""
        messages = self.messages.copy()
        self.messages.clear()
        self.last_updated = datetime.now()
        return messages
    
    def size(self) -> int:
        """Get buffer size"""
        return len(self.messages)
    
    def get_messages(self) -> List[Message]:
        """Get all messages in buffer"""
        return self.messages.copy()
    
    def is_empty(self) -> bool:
        """Check if buffer is empty"""
        return len(self.messages) == 0
    
    def is_timeout(self, timeout_minutes: int) -> bool:
        """Check if buffer has timed out - timeout functionality disabled"""
        # Buffer timeout functionality has been disabled, will never timeout
        return False
    
    def get_conversation_text(self) -> str:
        """Get conversation text"""
        lines = []
        for msg in self.messages:
            lines.append(f"{msg.role}: {msg.content}")
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "owner_id": self.owner_id,
            "messages": [msg.to_dict() for msg in self.messages],
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MessageBuffer':
        """Create buffer from dictionary"""
        buffer = cls(
            owner_id=data["owner_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_updated=datetime.fromisoformat(data["last_updated"]),
            metadata=data.get("metadata", {})
        )
        
        # Add messages
        for msg_data in data.get("messages", []):
            buffer.add_message(Message.from_dict(msg_data))
        
        return buffer
    
    def __len__(self) -> int:
        return len(self.messages)
    
    def __iter__(self):
        return iter(self.messages)
    
    def __str__(self) -> str:
        return f"MessageBuffer(owner={self.owner_id}, size={len(self.messages)})" 