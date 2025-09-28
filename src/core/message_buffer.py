"""
Message buffer manager module
"""

import threading
import logging
from typing import Dict, Optional
from datetime import datetime
from ..models import MessageBuffer, Message
from ..config import MemoryConfig

logger = logging.getLogger(__name__)


class MessageBufferManager:
    """Message buffer manager for handling user message buffers with user-level locking"""
    
    def __init__(self, config: MemoryConfig):
        """
        Initialize message buffer manager with optimized locking
        
        Args:
            config: Memory system configuration
        """
        self.config = config
        self._buffers: Dict[str, MessageBuffer] = {}
        self._user_locks: Dict[str, threading.RLock] = {}  # One independent lock per user
        self._global_lock = threading.RLock()  # Only used for managing user lock dictionary
        
        logger.info("Optimized message buffer manager initialized with user-level locking")
    
    def _get_user_lock(self, owner_id: str) -> threading.RLock:
        """
        Get or create a lock for specific user (thread-safe)
        
        Args:
            owner_id: User ID
            
        Returns:
            User-specific lock
        """
        # Fast path: if lock exists, return directly (avoid global lock)
        if owner_id in self._user_locks:
            return self._user_locks[owner_id]
        
        # Slow path: only use global lock when creating new lock
        with self._global_lock:
            if owner_id not in self._user_locks:
                self._user_locks[owner_id] = threading.RLock()
                logger.debug(f"Created user-specific lock for {owner_id}")
            return self._user_locks[owner_id]
    
    def get_buffer(self, owner_id: str) -> Optional[MessageBuffer]:
        """
        Get buffer for a user (user-level locking)
        
        Args:
            owner_id: User ID
            
        Returns:
            Message buffer or None if doesn't exist
        """
        user_lock = self._get_user_lock(owner_id)
        with user_lock:  # Only lock specific user, doesn't affect other users
            return self._buffers.get(owner_id)
    
    def get_or_create_buffer(self, owner_id: str) -> MessageBuffer:
        """
        Get or create buffer for a user (user-level locking)
        
        Args:
            owner_id: User ID
            
        Returns:
            Message buffer
        """
        user_lock = self._get_user_lock(owner_id)
        with user_lock:  # Only lock specific user, doesn't affect other users
            if owner_id not in self._buffers:
                self._buffers[owner_id] = MessageBuffer(owner_id=owner_id)
                logger.debug(f"Created new buffer for user {owner_id}")
            
            return self._buffers[owner_id]
    
    def add_message(self, owner_id: str, message: Message) -> int:
        """
        Add message to user's buffer (user-level locking)
        
        Args:
            owner_id: User ID
            message: Message to add
            
        Returns:
            New buffer size
        """
        user_lock = self._get_user_lock(owner_id)
        with user_lock:  # Only lock specific user, doesn't affect other users
            buffer = self.get_or_create_buffer(owner_id)
            buffer.add_message(message)
            
            logger.debug(f"Added message to buffer for user {owner_id}, new size: {buffer.size()}")
            return buffer.size()
    
    def add_messages(self, owner_id: str, messages: list[Message]) -> int:
        """
        Add multiple messages to user's buffer (user-level locking)
        
        Args:
            owner_id: User ID
            messages: Messages to add
            
        Returns:
            New buffer size
        """
        user_lock = self._get_user_lock(owner_id)
        with user_lock:  # Only lock specific user, doesn't affect other users
            buffer = self.get_or_create_buffer(owner_id)
            buffer.add_messages(messages)
            
            logger.debug(f"Added {len(messages)} messages to buffer for user {owner_id}, new size: {buffer.size()}")
            return buffer.size()
    
    def clear_buffer(self, owner_id: str) -> bool:
        """
        Clear user's buffer (user-level locking)
        
        Args:
            owner_id: User ID
            
        Returns:
            True if buffer existed and was cleared
        """
        user_lock = self._get_user_lock(owner_id)
        with user_lock:  # Only lock specific user, doesn't affect other users
            if owner_id in self._buffers:
                self._buffers[owner_id].clear()
                logger.debug(f"Cleared buffer for user {owner_id}")
                return True
            return False
    
    def delete_buffer(self, owner_id: str) -> bool:
        """
        Delete user's buffer completely (user-level locking)
        
        Args:
            owner_id: User ID
            
        Returns:
            True if buffer existed and was deleted
        """
        user_lock = self._get_user_lock(owner_id)
        with user_lock:  # Only lock specific user, doesn't affect other users
            buffer_existed = owner_id in self._buffers
            if buffer_existed:
                del self._buffers[owner_id]
                logger.debug(f"Deleted buffer for user {owner_id}")
            
            # Clean up locks no longer needed (under global lock protection)
            with self._global_lock:
                if owner_id in self._user_locks:
                    del self._user_locks[owner_id]
                    logger.debug(f"Cleaned up lock for user {owner_id}")
            
            return buffer_existed
    
    def get_buffer_size(self, owner_id: str) -> int:
        """
        Get buffer size for a user (user-level locking)
        
        Args:
            owner_id: User ID
            
        Returns:
            Buffer size (0 if buffer doesn't exist)
        """
        user_lock = self._get_user_lock(owner_id)
        with user_lock:  # Only lock specific user, doesn't affect other users
            buffer = self._buffers.get(owner_id)
            return buffer.size() if buffer else 0
    
    def is_buffer_full(self, owner_id: str) -> bool:
        """
        Check if buffer is full (reached max size) (user-level locking)
        
        Args:
            owner_id: User ID
            
        Returns:
            True if buffer is full
        """
        user_lock = self._get_user_lock(owner_id)
        with user_lock:  # Only lock specific user, doesn't affect other users
            buffer = self._buffers.get(owner_id)
            if not buffer:
                return False
            
            return buffer.size() >= self.config.buffer_size_max
    
    def is_buffer_ready(self, owner_id: str) -> bool:
        """
        Check if buffer is ready for episode creation (user-level locking)
        (reached min size - timeout check removed)
        
        Args:
            owner_id: User ID
            
        Returns:
            True if buffer is ready
        """
        user_lock = self._get_user_lock(owner_id)
        with user_lock:  # Only lock specific user, doesn't affect other users
            buffer = self._buffers.get(owner_id)
            if not buffer or buffer.is_empty():
                return False
            
            # Check size only (timeout check removed)
            return buffer.size() >= self.config.buffer_size_min
    
    def is_buffer_timeout(self, owner_id: str) -> bool:
        """
        Check if buffer has timed out - DISABLED
        
        Args:
            owner_id: User ID
            
        Returns:
            Always False (timeout functionality disabled)
        """
        # Buffer timeout functionality has been disabled
        return False
    
    def get_all_buffer_stats(self) -> Dict[str, Dict[str, any]]:
        """
        Get statistics for all buffers (optimized for minimal global locking)
        
        Returns:
            Dictionary with buffer statistics
        """
        # First get all user IDs (brief global lock)
        with self._global_lock:
            user_ids = list(self._buffers.keys())
        
        # Then get statistics for each user (user-level locks, can be parallel)
        stats = {}
        for owner_id in user_ids:
            try:
                user_lock = self._get_user_lock(owner_id)
                with user_lock:
                    buffer = self._buffers.get(owner_id)
                    if buffer:
                        stats[owner_id] = {
                            "size": buffer.size(),
                            "created_at": buffer.created_at.isoformat(),
                            "last_updated": buffer.last_updated.isoformat(),
                            "is_empty": buffer.is_empty(),
                            "is_timeout": False,  # Timeout functionality disabled, always False
                            "is_full": buffer.size() >= self.config.buffer_size_max,
                            "is_ready": buffer.size() >= self.config.buffer_size_min
                        }
            except Exception as e:
                logger.warning(f"Error getting stats for user {owner_id}: {e}")
                continue
        
        return stats
    
    def cleanup_inactive_buffers(self, max_age_hours: int = 24) -> int:
        """
        Clean up inactive buffers that haven't been updated for a long time
        (optimized for user-level locking)
        
        Args:
            max_age_hours: Maximum age in hours before cleanup
            
        Returns:
            Number of buffers cleaned up
        """
        # First get all user IDs
        with self._global_lock:
            user_ids = list(self._buffers.keys())
        
        current_time = datetime.now()
        buffers_to_delete = []
        
        # Check each user's buffer (can be parallel)
        for owner_id in user_ids:
            try:
                user_lock = self._get_user_lock(owner_id)
                with user_lock:
                    buffer = self._buffers.get(owner_id)
                    if buffer:
                        time_diff = current_time - buffer.last_updated
                        age_hours = time_diff.total_seconds() / 3600
                        
                        if age_hours > max_age_hours:
                            buffers_to_delete.append(owner_id)
            except Exception as e:
                logger.warning(f"Error checking buffer age for user {owner_id}: {e}")
                continue
        
        # Delete expired buffers
        cleaned_count = 0
        for owner_id in buffers_to_delete:
            if self.delete_buffer(owner_id):
                cleaned_count += 1
                logger.info(f"Cleaned up inactive buffer for user {owner_id}")
        
        return cleaned_count
    
    def get_manager_stats(self) -> Dict[str, any]:
        """
        Get overall manager statistics (optimized)
        
        Returns:
            Manager statistics
        """
        # Quickly get basic statistics (short global lock)
        with self._global_lock:
            total_buffers = len(self._buffers)
            total_locks = len(self._user_locks)
            user_ids = list(self._buffers.keys())
        
        # Calculate detailed statistics (user-level lock, can be parallel)
        total_messages = 0
        ready_buffers = 0
        
        for owner_id in user_ids:
            try:
                user_lock = self._get_user_lock(owner_id)
                with user_lock:
                    buffer = self._buffers.get(owner_id)
                    if buffer:
                        total_messages += buffer.size()
                        if buffer.size() >= self.config.buffer_size_min:
                            ready_buffers += 1
            except Exception as e:
                logger.warning(f"Error getting detailed stats for user {owner_id}: {e}")
                continue
        
        return {
            "total_buffers": total_buffers,
            "total_messages": total_messages,
            "ready_buffers": ready_buffers,
            "timeout_buffers": 0,  # Always 0, because timeout functionality is disabled
            "average_buffer_size": total_messages / total_buffers if total_buffers > 0 else 0,
            "active_user_locks": total_locks,
            "locking_strategy": "user-level (optimized)",
            "config": {
                "buffer_size_min": self.config.buffer_size_min,
                "buffer_size_max": self.config.buffer_size_max
                # Note: buffer_timeout_minutes has been removed
            }
        } 