"""
Semantic Memory Storage
"""

import os
import json
import logging
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime
from .base_storage import BaseStorage
from ..models import SemanticMemory

logger = logging.getLogger(__name__)


class SemanticStorage(BaseStorage):
    """Semantic memory storage implementation using JSONL files"""
    
    def __init__(self, storage_path: str):
        """
        Initialize semantic memory storage
        
        Args:
            storage_path: Base storage directory
        """
        super().__init__(storage_path)
        self.semantic_dir = os.path.join(storage_path, "semantic")
        os.makedirs(self.semantic_dir, exist_ok=True)
        
        # In-memory index for fast access
        self._memory_index: Dict[str, str] = {}  # memory_id -> file_path
        self._user_index: Dict[str, List[str]] = {}  # user_id -> [memory_ids]
        
        # File lock dictionary, one lock per user
        self._user_file_locks: Dict[str, threading.RLock] = {}
        self._lock_dict_lock = threading.Lock()  # Lock to protect the lock dictionary
        
        # Load existing index
        self._load_index()
        
        logger.info(f"Semantic storage initialized at {self.semantic_dir}")
    
    def _get_user_file_path(self, user_id: str) -> str:
        """Get file path for user's semantic memories"""
        return os.path.join(self.semantic_dir, f"{user_id}_semantic.jsonl")
    
    def _load_index(self):
        """Load index from existing files"""
        try:
            for filename in os.listdir(self.semantic_dir):
                if filename.endswith("_semantic.jsonl"):
                    user_id = filename.replace("_semantic.jsonl", "")
                    file_path = os.path.join(self.semantic_dir, filename)
                    
                    # Load memories from file
                    memories = self._load_memories_from_file(file_path)
                    memory_ids = []
                    
                    for memory in memories:
                        memory_id = memory.memory_id
                        self._memory_index[memory_id] = file_path
                        memory_ids.append(memory_id)
                    
                    self._user_index[user_id] = memory_ids
                    
        except Exception as e:
            logger.warning(f"Error loading semantic index: {e}")
    
    def _load_memories_from_file(self, file_path: str) -> List[SemanticMemory]:
        """Load semantic memories from JSONL file"""
        memories = []
        
        if not os.path.exists(file_path):
            return memories
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        memory_data = json.loads(line)
                        memory = SemanticMemory.from_dict(memory_data)
                        memories.append(memory)
        except Exception as e:
            logger.error(f"Error loading semantic memories from {file_path}: {e}")
        
        return memories
    
    def _get_user_file_lock(self, user_id: str) -> threading.RLock:
        """Get user file lock"""
        with self._lock_dict_lock:
            if user_id not in self._user_file_locks:
                self._user_file_locks[user_id] = threading.RLock()
            return self._user_file_locks[user_id]
    
    def save_semantic_memory(self, memory: SemanticMemory) -> str:
        """Save a semantic memory with proper concurrency protection"""
        try:
            user_id = memory.user_id
            memory_id = memory.memory_id
            file_path = self._get_user_file_path(user_id)
            
            # Get user file lock
            file_lock = self._get_user_file_lock(user_id)
            
            with file_lock:
                # Ensure directory exists
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # Check if memory already exists (avoid duplicates)
                if memory_id in self._memory_index:
                    logger.debug(f"Semantic memory {memory_id} already exists, skipping")
                    return memory_id
                
                # Use append mode to write, avoid overwriting existing content
                with open(file_path, 'a', encoding='utf-8') as f:
                    json.dump(memory.to_dict(), f, ensure_ascii=False)
                    f.write('\n')
                
                # Update memory index
                self._memory_index[memory_id] = file_path
                if user_id not in self._user_index:
                    self._user_index[user_id] = []
                if memory_id not in self._user_index[user_id]:
                    self._user_index[user_id].append(memory_id)
            
            logger.debug(f"Saved semantic memory {memory_id} for user {user_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Error saving semantic memory {memory.memory_id}: {e}")
            raise
    
    def save(self, item: SemanticMemory) -> str:
        """Implementation of base save method"""
        return self.save_semantic_memory(item)
    
    def load(self, item_id: str) -> Optional[SemanticMemory]:
        """Implementation of base load method"""
        try:
            if item_id not in self._memory_index:
                return None
            
            file_path = self._memory_index[item_id]
            memories = self._load_memories_from_file(file_path)
            
            for memory in memories:
                if memory.memory_id == item_id:
                    return memory
            
            return None
            
        except Exception as e:
            logger.error(f"Error loading semantic memory {item_id}: {e}")
            return None
    
    def delete(self, item_id: str) -> bool:
        """Implementation of base delete method"""
        return False  # Simplified for now
    
    def list_user_items(self, user_id: str) -> List[SemanticMemory]:
        """Get all semantic memories for a user"""
        try:
            file_path = self._get_user_file_path(user_id)
            memories = self._load_memories_from_file(file_path)
            
            # Sort by creation time (newest first)
            memories.sort(key=lambda x: x.created_at, reverse=True)
            
            return memories
            
        except Exception as e:
            logger.error(f"Error getting semantic memories for user {user_id}: {e}")
            return []
    
    def delete_user_data(self, user_id: str) -> bool:
        """Delete all data for a user"""
        try:
            file_path = self._get_user_file_path(user_id)
            
            # Get user file lock
            file_lock = self._get_user_file_lock(user_id)
            
            with file_lock:
                # Delete file if exists
                if os.path.exists(file_path):
                    os.remove(file_path)
                
                # Clean up index
                if user_id in self._user_index:
                    memory_ids = self._user_index[user_id]
                    for memory_id in memory_ids:
                        if memory_id in self._memory_index:
                            del self._memory_index[memory_id]
                    del self._user_index[user_id]
            
            # Clean up locks
            with self._lock_dict_lock:
                if user_id in self._user_file_locks:
                    del self._user_file_locks[user_id]
            
            logger.info(f"Deleted all semantic memories for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting semantic data for {user_id}: {e}")
            return False
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics for a user"""
        try:
            memories = self.list_user_items(user_id)
            
            if not memories:
                return {
                    "semantic_memory_count": 0,
                    "first_memory": None,
                    "last_memory": None
                }
            
            first_memory = min(memories, key=lambda x: x.created_at)
            last_memory = max(memories, key=lambda x: x.created_at)
            
            return {
                "semantic_memory_count": len(memories),
                "first_memory": first_memory.created_at.isoformat(),
                "last_memory": last_memory.created_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting semantic stats for user {user_id}: {e}")
            return {"error": str(e)}
    
    def find_similar_statement(
        self, 
        user_id: str, 
        content: str, 
        similarity_threshold: float = 0.9
    ) -> Optional[SemanticMemory]:
        """
        Find similar statements (for deduplication)
        
        Args:
            user_id: User ID
            content: Content to check
            similarity_threshold: Similarity threshold
            
        Returns:
            Returns similar statement if found, otherwise None
        """
        try:
            # Get all semantic memories for the user
            existing_memories = self.list_user_items(user_id)
            
            if not existing_memories:
                return None
            
            # Simplified processing here, should actually use vector similarity
            # Use exact matching for now
            for memory in existing_memories:
                # Remove timestamp prefix for comparison
                memory_content = memory.content
                if memory_content.startswith("Time:"):
                    # Extract actual content after timestamp
                    parts = memory_content.split(" ", 3)
                    if len(parts) >= 4:
                        memory_content = parts[3]
                
                test_content = content
                if test_content.startswith("Time:"):
                    parts = test_content.split(" ", 3)
                    if len(parts) >= 4:
                        test_content = parts[3]
                
                # Simple text similarity check
                if memory_content.strip().lower() == test_content.strip().lower():
                    return memory
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding similar statement: {e}")
            return None 