"""
Episode Storage
"""

import os
import json
import logging
import threading
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
from .base_storage import BaseStorage
from ..models import Episode

logger = logging.getLogger(__name__)


class EpisodeStorage(BaseStorage):
    """
    Episodic memory storage class (optimized concurrent version)
    
    Features:
    - User file-level locks to avoid file conflicts
    - Batch writing to reduce I/O operations
    - Asynchronous save support
    - Improved error handling
    """
    
    def __init__(self, storage_path: str):
        """
        Initialize episode storage
        
        Args:
            storage_path: Base storage directory
        """
        super().__init__(storage_path, "episodes")
        
        # User file-level locks (avoid concurrent writes to the same file)
        self._user_file_locks: Dict[str, threading.RLock] = {}
        self._locks_manager = threading.RLock()
        
        # Batch write buffer (kept for compatibility but writes are now synchronous)
        self._write_buffer: Dict[str, List[Episode]] = {}
        self._buffer_lock = threading.RLock()
        self._buffer_size_threshold = 1  # Force immediate flush
        self._buffer_time_threshold = 0.0
        self._last_flush_time: Dict[str, float] = {}
        
        # Statistics information
        self._stats = {
            "saves": 0,
            "loads": 0,
            "batched_saves": 0,
            "cache_hits": 0,
            "file_locks_acquired": 0
        }
        self._stats_lock = threading.Lock()
        
        # Simple memory cache (reduce duplicate file reads)
        self._cache: Dict[str, List[Episode]] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._cache_ttl = 300.0  # 5 minutes cache
        self._cache_lock = threading.RLock()
        
        # In-memory index for fast access
        self._episode_index: Dict[str, str] = {}  # episode_id -> file_path
        self._user_index: Dict[str, List[str]] = {}  # user_id -> [episode_ids]
        
        # Load existing index
        self._load_index()
        
        logger.info(f"Episode storage initialized at {self.data_dir}")
    
    def _get_user_file_lock(self, owner_id: str) -> threading.RLock:
        """
        Get user file dedicated lock
        
        Args:
            owner_id: User identifier
            
        Returns:
            User file lock
        """
        # Fast path
        if owner_id in self._user_file_locks:
            return self._user_file_locks[owner_id]
        
        # Slow path
        with self._locks_manager:
            if owner_id not in self._user_file_locks:
                self._user_file_locks[owner_id] = threading.RLock()
                with self._stats_lock:
                    self._stats["file_locks_acquired"] += 1
            return self._user_file_locks[owner_id]
    
    def _load_index(self):
        """Load index from existing files"""
        try:
            for filename in os.listdir(self.data_dir):
                if filename.endswith("_episodes.jsonl"):
                    user_id = filename.replace("_episodes.jsonl", "")
                    file_path = self.data_dir / filename
                    
                    # Load episodes from JSONL file
                    episodes = self._load_episodes_from_jsonl(file_path)
                    episode_ids = []
                    
                    for episode in episodes:
                        episode_id = episode.episode_id
                        self._episode_index[episode_id] = str(file_path)
                        episode_ids.append(episode_id)
                    
                    self._user_index[user_id] = episode_ids
                    
                elif filename.endswith(".json"):
                    # Backward compatibility: handle old JSON files
                    user_id = filename.replace(".json", "")
                    file_path = self.data_dir / filename
                    
                    # Load episodes from JSON file
                    episodes = self._load_episodes_from_json(file_path)
                    episode_ids = []
                    
                    for episode in episodes:
                        episode_id = episode.episode_id
                        self._episode_index[episode_id] = str(file_path)
                        episode_ids.append(episode_id)
                    
                    self._user_index[user_id] = episode_ids
                    
        except Exception as e:
            logger.warning(f"Error loading index: {e}")
    

    
    def save_episode(self, episode: Episode) -> str:
        """
        Save episodic memory (support batch and async)
        
        Args:
            episode: Episodic memory object
            
        Returns:
            episode_id
        """
        try:
            if not episode.episode_id:
                episode.episode_id = str(uuid.uuid4())

            owner_id = episode.user_id

            # Persist immediately to avoid index/storage divergence
            self._save_single_episode(owner_id, episode)

            with self._buffer_lock:
                # Ensure compatibility with existing cleanup paths
                self._write_buffer.pop(owner_id, None)
                self._last_flush_time.pop(owner_id, None)

            with self._stats_lock:
                self._stats["saves"] += 1

            return episode.episode_id

        except Exception as e:
            raise Exception(f"Error saving episode: {e}")
    
    def _batch_save_episodes(self, owner_id: str, episodes: List[Episode]):
        """
        Batch save episodes to JSONL file
        
        Args:
            owner_id: User identifier
            episodes: Episode list
        """
        if not episodes:
            return
        
        file_lock = self._get_user_file_lock(owner_id)
        with file_lock:
            try:
                # Use JSONL format file
                file_path = self.data_dir / f"{owner_id}_episodes.jsonl"
                
                # Read existing data
                existing_episodes = []
                if file_path.exists():
                    existing_episodes = self._load_episodes_from_jsonl(file_path)
                
                # Merge new episodes (avoid duplicates)
                existing_ids = {ep.episode_id for ep in existing_episodes}
                new_episodes = [ep for ep in episodes if ep.episode_id not in existing_ids]
                
                if new_episodes:
                    all_episodes = existing_episodes + new_episodes
                    
                    # Write to JSONL file
                    self._save_episodes_to_jsonl(file_path, all_episodes)
                    
                    # Update cache
                    with self._cache_lock:
                        self._cache[owner_id] = all_episodes
                        self._cache_timestamps[owner_id] = time.time()

                    # Update indices for the new episodes
                    user_episode_ids = self._user_index.setdefault(owner_id, [])
                    for ep in new_episodes:
                        self._episode_index[ep.episode_id] = str(file_path)
                        if ep.episode_id not in user_episode_ids:
                            user_episode_ids.append(ep.episode_id)
                    
                    with self._stats_lock:
                        self._stats["batched_saves"] += 1
                
            except Exception as e:
                logger.error(f"Error in batch save for {owner_id}: {e}")
                # Rollback to individual save on error
                for episode in episodes:
                    try:
                        self._save_single_episode(owner_id, episode)
                    except:
                        pass  # Ignore individual failures
    
    def _save_single_episode(self, owner_id: str, episode: Episode):
        """
        Save single episode to JSONL file (rollback solution)
        
        Args:
            owner_id: User identifier
            episode: Episode object
        """
        file_lock = self._get_user_file_lock(owner_id)
        with file_lock:
            file_path = self.data_dir / f"{owner_id}_episodes.jsonl"
            
            # Read existing data
            episodes = []
            if file_path.exists():
                episodes = self._load_episodes_from_jsonl(file_path)

            # Check for duplicates
            if not any(ep.episode_id == episode.episode_id for ep in episodes):
                episodes.append(episode)
                
                # Write to JSONL file
                self._save_episodes_to_jsonl(file_path, episodes)
                
                # Update cache
                with self._cache_lock:
                    self._cache[owner_id] = episodes
                    self._cache_timestamps[owner_id] = time.time()

                # Update in-memory indices for fast lookups
                self._episode_index[episode.episode_id] = str(file_path)
                user_episode_ids = self._user_index.setdefault(owner_id, [])
                if episode.episode_id not in user_episode_ids:
                    user_episode_ids.append(episode.episode_id)
    
    def _save_episodes_to_jsonl(self, file_path: Path, episodes: List[Episode]):
        """Save episodes to JSONL file"""
        try:
            os.makedirs(file_path.parent, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                for episode in episodes:
                    json.dump(episode.to_dict(), f, ensure_ascii=False)
                    f.write('\n')
                    
        except Exception as e:
            logger.error(f"Error saving episodes to JSONL {file_path}: {e}")
            raise
    
    def get_episode(self, episode_id: str, owner_id: str) -> Optional[Episode]:
        """
        Get single episodic memory (using cache)
        
        Args:
            episode_id: Episode ID
            owner_id: User identifier
            
        Returns:
            Episode object or None
        """
        try:
            episodes = self.get_user_episodes(owner_id)
            return next((ep for ep in episodes if ep.episode_id == episode_id), None)
        except Exception as e:
            raise Exception(f"Error getting episode {episode_id}: {e}")
    
    def get_user_episodes(self, owner_id: str) -> List[Episode]:
        """
        Get all episodic memories for user (optimized cache, using JSONL format)
        
        Args:
            owner_id: User identifier
            
        Returns:
            Episode list
        """
        try:
            # Flush any buffered episodes to keep storage consistent
            buffered_episodes = []
            with self._buffer_lock:
                if owner_id in self._write_buffer and self._write_buffer[owner_id]:
                    buffered_episodes = self._write_buffer[owner_id].copy()
                    self._write_buffer[owner_id].clear()
                    self._last_flush_time[owner_id] = time.time()

            if buffered_episodes:
                self._batch_save_episodes(owner_id, buffered_episodes)

            # Check cache
            with self._cache_lock:
                if owner_id in self._cache:
                    cache_time = self._cache_timestamps.get(owner_id, 0)
                    if time.time() - cache_time < self._cache_ttl:
                        with self._stats_lock:
                            self._stats["cache_hits"] += 1
                        return self._cache[owner_id].copy()
            
            # Cache miss, read from file
            file_lock = self._get_user_file_lock(owner_id)
            with file_lock:
                episodes = []
                
                # Prefer JSONL format
                jsonl_path = self.data_dir / f"{owner_id}_episodes.jsonl"
                if jsonl_path.exists():
                    episodes = self._load_episodes_from_jsonl(jsonl_path)
                else:
                    # Backward compatibility: if no JSONL file, try old JSON format
                    json_path = self.data_dir / f"{owner_id}.json"
                    if json_path.exists():
                        episodes = self._load_episodes_from_json(json_path)
                        # If data was loaded from JSON, immediately convert to JSONL format
                        if episodes:
                            self._save_episodes_to_jsonl(jsonl_path, episodes)
                            logger.info(f"Migrated {owner_id} from JSON to JSONL format")
                
                # Update cache
                with self._cache_lock:
                    self._cache[owner_id] = episodes
                    self._cache_timestamps[owner_id] = time.time()
                
                with self._stats_lock:
                    self._stats["loads"] += 1
                
                return episodes.copy()
                
        except Exception as e:
            raise Exception(f"Error loading user episodes for {owner_id}: {e}")

    def _load_episodes_from_jsonl(self, file_path: Path) -> List[Episode]:
        """Load episodes from JSONL file (compatible with existing data)"""
        episodes = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        episode_data = json.loads(line)
                        episode = Episode.from_dict(episode_data)
                        episodes.append(episode)
        except Exception as e:
            logger.error(f"Error loading episodes from JSONL {file_path}: {e}")
        
        return episodes

    def _load_episodes_from_json(self, file_path: Path) -> List[Episode]:
        """Load episodes from JSON file (new format)"""
        episodes = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                episodes = [Episode.from_dict(item) for item in data]
        except Exception as e:
            logger.error(f"Error loading episodes from JSON {file_path}: {e}")
        
        return episodes
    
    def update_episode(self, episode: Episode) -> bool:
        """
        Update episodic memory (using JSONL format)
        
        Args:
            episode: Updated Episode object
            
        Returns:
            Returns True if update successful
        """
        try:
            owner_id = episode.user_id
            file_lock = self._get_user_file_lock(owner_id)
            
            with file_lock:
                episodes = self.get_user_episodes(owner_id)
                
                # Find and update
                updated = False
                for i, ep in enumerate(episodes):
                    if ep.episode_id == episode.episode_id:
                        episodes[i] = episode
                        updated = True
                        break
                
                if updated:
                    # Write to JSONL file
                    file_path = self.data_dir / f"{owner_id}_episodes.jsonl"
                    self._save_episodes_to_jsonl(file_path, episodes)
                    
                    # Update cache
                    with self._cache_lock:
                        self._cache[owner_id] = episodes
                        self._cache_timestamps[owner_id] = time.time()

                    # Refresh index metadata
                    self._episode_index[episode.episode_id] = str(file_path)
                    user_episode_ids = self._user_index.setdefault(owner_id, [])
                    if episode.episode_id not in user_episode_ids:
                        user_episode_ids.append(episode.episode_id)
                
                return updated
                
        except Exception as e:
            raise Exception(f"Error updating episode: {e}")
    
    def delete_episode(self, episode_id: str, owner_id: str) -> bool:
        """
        Delete episodic memory (using JSONL format)
        
        Args:
            episode_id: Episode ID
            owner_id: User identifier
            
        Returns:
            Returns True if deletion successful
        """
        try:
            file_lock = self._get_user_file_lock(owner_id)
            
            with file_lock:
                episodes = self.get_user_episodes(owner_id)
                
                # Find and delete
                initial_count = len(episodes)
                episodes = [ep for ep in episodes if ep.episode_id != episode_id]
                
                if len(episodes) < initial_count:
                    # Write to JSONL file
                    file_path = self.data_dir / f"{owner_id}_episodes.jsonl"
                    self._save_episodes_to_jsonl(file_path, episodes)
                    
                    # Update cache
                    with self._cache_lock:
                        self._cache[owner_id] = episodes
                        self._cache_timestamps[owner_id] = time.time()

                    # Update in-memory indices
                    self._episode_index.pop(episode_id, None)
                    if owner_id in self._user_index:
                        try:
                            self._user_index[owner_id].remove(episode_id)
                        except ValueError:
                            pass
                    
                    return True
                
                return False
                
        except Exception as e:
            raise Exception(f"Error deleting episode: {e}")
    
    def get_user_stats(self, owner_id: str) -> Dict[str, Any]:
        """
        Get user statistics
        
        Args:
            owner_id: User identifier
            
        Returns:
            Statistics dictionary
        """
        try:
            episodes = self.get_user_episodes(owner_id)
            
            if not episodes:
                return {
                    "episode_count": 0,
                    "total_messages": 0,
                    "avg_messages_per_episode": 0,
                    "date_range": None
                }
            
            total_messages = sum(ep.message_count for ep in episodes)
            dates = [ep.created_at for ep in episodes if ep.created_at]
            
            return {
                "episode_count": len(episodes),
                "total_messages": total_messages,
                "avg_messages_per_episode": total_messages / len(episodes) if episodes else 0,
                "date_range": {
                    "earliest": min(dates).isoformat() if dates else None,
                    "latest": max(dates).isoformat() if dates else None
                }
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def delete_user_data(self, owner_id: str) -> bool:
        """
        Delete all user data (using JSONL format)
        
        Args:
            owner_id: User identifier
            
        Returns:
            Returns True if deletion successful
        """
        try:
            file_lock = self._get_user_file_lock(owner_id)
            
            with file_lock:
                # Delete JSONL file
                jsonl_path = self.data_dir / f"{owner_id}_episodes.jsonl"
                if jsonl_path.exists():
                    jsonl_path.unlink()
                
                # Delete old JSON file (if exists)
                json_path = self.data_dir / f"{owner_id}.json"
                if json_path.exists():
                    json_path.unlink()
                
                # Clean up cache
                with self._cache_lock:
                    self._cache.pop(owner_id, None)
                    self._cache_timestamps.pop(owner_id, None)
                
                # Clean up buffer
                with self._buffer_lock:
                    self._write_buffer.pop(owner_id, None)
                    self._last_flush_time.pop(owner_id, None)

                # Clean up indices
                user_episode_ids = self._user_index.pop(owner_id, [])
                for episode_id in user_episode_ids:
                    self._episode_index.pop(episode_id, None)
                
                # Clean up locks (performed outside manager lock)
                with self._locks_manager:
                    self._user_file_locks.pop(owner_id, None)
                
                return True
                
        except Exception as e:
            raise Exception(f"Error deleting user data for {owner_id}: {e}")
    
    def flush_all_buffers(self):
        """
        Flush write buffers for all users
        """
        with self._buffer_lock:
            users_to_flush = list(self._write_buffer.keys())
        
        # Flush all users' buffers in parallel
        with ThreadPoolExecutor(max_workers=min(len(users_to_flush), 5)) as executor:
            futures = []
            
            for owner_id in users_to_flush:
                with self._buffer_lock:
                    if owner_id in self._write_buffer and self._write_buffer[owner_id]:
                        episodes_to_save = self._write_buffer[owner_id].copy()
                        self._write_buffer[owner_id].clear()
                        self._last_flush_time[owner_id] = time.time()
                        
                        future = executor.submit(self._batch_save_episodes, owner_id, episodes_to_save)
                        futures.append(future)
            
            # Wait for all saves to complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    # Log error but don't interrupt other saves
                    pass
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics
        
        Returns:
            Storage statistics
        """
        with self._stats_lock:
            stats = self._stats.copy()
        
        with self._cache_lock:
            cache_stats = {
                "cached_users": len(self._cache),
                "cache_hit_ratio": (
                    stats["cache_hits"] / max(stats["loads"] + stats["cache_hits"], 1)
                ) * 100
            }
        
        with self._buffer_lock:
            buffer_stats = {
                "buffered_users": len(self._write_buffer),
                "total_buffered_episodes": sum(len(episodes) for episodes in self._write_buffer.values()),
                "buffer_threshold": self._buffer_size_threshold
            }
        
        return {
            "operations": stats,
            "cache": cache_stats,
            "buffer": buffer_stats,
            "file_locks": len(self._user_file_locks)
        }
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        # Ensure all buffers are flushed
        self.flush_all_buffers()
    
    # BaseStorage abstract method implementations
    def save(self, item: Episode) -> str:
        """Implementation of BaseStorage's save method"""
        return self.save_episode(item)
    
    def load(self, item_id: str) -> Optional[Episode]:
        """Implementation of BaseStorage's load method"""
        # Need to find the episode from all users
        try:
            # Traverse all user files to find episode
            for file_path in self.data_dir.glob("*.*"):
                if not file_path.is_file():
                    continue

                try:
                    if file_path.suffix == ".jsonl":
                        with open(file_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                ep_data = json.loads(line)
                                if ep_data.get("episode_id") == item_id:
                                    return Episode.from_dict(ep_data)
                    elif file_path.suffix == ".json":
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            for ep_data in data:
                                if ep_data.get("episode_id") == item_id:
                                    return Episode.from_dict(ep_data)
                except Exception:
                    continue
            return None
        except Exception:
            return None
    
    def delete(self, item_id: str) -> bool:
        """Implementation of BaseStorage's delete method"""
        # Need to find and delete the episode from all users
        try:
            for file_path in self.data_dir.glob("*.*"):
                if not file_path.is_file():
                    continue

                owner_id = file_path.stem.replace('_episodes', '')
                if self.delete_episode(item_id, owner_id):
                    return True
            return False
        except Exception:
            return False
    
    def list_user_items(self, user_id: str) -> List[Episode]:
        """Implementation of BaseStorage's list_user_items method"""
        return self.get_user_episodes(user_id) 
