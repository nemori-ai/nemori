"""
Lightweight Search System
Lightweight search system optimized for fast search
"""

import logging
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from ..config import MemoryConfig
from ..models import Episode, SemanticMemory
from ..utils import EmbeddingClient
from ..search import VectorSearch

logger = logging.getLogger(__name__)


class LightweightSearchSystem:
    """
    Lightweight search system optimized for high-concurrency search scenarios
    
    Features:
    - Minimized initialization overhead
    - Preload data to memory to avoid file I/O contention
    - Contains only search-essential components
    - Lock-free design supporting true concurrency
    """
    
    def __init__(self, config: MemoryConfig, preload_data: bool = True):
        """
        Initialize lightweight search system
        
        Args:
            config: System configuration
            preload_data: Whether to preload all data to memory
        """
        self.config = config
        self.preload_data = preload_data
        
        # Initialize only essential components
        self.embedding_client = EmbeddingClient(
            api_key=config.openai_api_key,
            model=config.embedding_model
        )
        
        # Vector search engine
        self.vector_search = VectorSearch(
            self.embedding_client,
            config.storage_path,
            config.embedding_dimension
        )
        
        # Preloaded data storage (lock-free, read-only)
        self.preloaded_episodes: Dict[str, List[Episode]] = {}
        self.preloaded_semantic: Dict[str, List[SemanticMemory]] = {}
        
        logger.info(f"Lightweight search system initialized (preload={preload_data})")
    
    def preload_user_data(self, user_id: str) -> bool:
        """
        Preload user data to memory (avoiding subsequent file I/O)
        
        Args:
            user_id: User ID
            
        Returns:
            Whether loading was successful
        """
        try:
            start_time = time.time()
            
            # Read directly from files, avoiding storage layer locks
            episodes = self._load_episodes_directly(user_id)
            semantic_memories = self._load_semantic_directly(user_id)
            
            # Store to memory (lock-free)
            self.preloaded_episodes[user_id] = episodes
            self.preloaded_semantic[user_id] = semantic_memories
            
            # Load vector indices
            self.vector_search.load_user_indices(user_id, episodes, semantic_memories)
            
            load_time = time.time() - start_time
            logger.info(f"Preloaded user {user_id}: {len(episodes)} episodes, "
                       f"{len(semantic_memories)} semantic memories in {load_time:.3f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Error preloading user {user_id}: {e}")
            return False
    
    def _load_episodes_directly(self, user_id: str) -> List[Episode]:
        """Load episodes directly from files, bypassing storage layer locks"""
        episodes = []
        
        # JSONL file path
        jsonl_path = Path(self.config.storage_path) / "episodes" / f"{user_id}_episodes.jsonl"
        
        if jsonl_path.exists():
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        episode_data = json.loads(line)
                        episode = Episode.from_dict(episode_data)
                        episodes.append(episode)
        
        return episodes
    
    def _load_semantic_directly(self, user_id: str) -> List[SemanticMemory]:
        """Load semantic memories directly from files, bypassing storage layer locks"""
        memories = []
        
        # JSONL file path
        jsonl_path = Path(self.config.storage_path) / "semantic" / f"{user_id}_semantic.jsonl"
        
        if jsonl_path.exists():
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        memory_data = json.loads(line)
                        memory = SemanticMemory.from_dict(memory_data)
                        memories.append(memory)
        
        return memories
    
    def search_vector(
        self,
        user_id: str,
        query: str,
        top_k_episodes: int = 10,
        top_k_semantic: int = 10
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Execute vector search (lock-free, high performance)
        
        Args:
            user_id: User ID
            query: Search query
            top_k_episodes: Number of episodic memories to return
            top_k_semantic: Number of semantic memories to return
            
        Returns:
            Search results
        """
        try:
            # Load data if not preloaded
            if user_id not in self.preloaded_episodes:
                self.preload_user_data(user_id)
            
            # Execute vector search
            episode_results = self.vector_search.search_episodes(user_id, query, top_k_episodes)
            semantic_results = self.vector_search.search_semantic_memories(user_id, query, top_k_semantic)
            
            return {
                "episodic": episode_results,
                "semantic": semantic_results
            }
            
        except Exception as e:
            logger.error(f"Error in vector search for user {user_id}: {e}")
            return {"episodic": [], "semantic": []}
    
    def batch_preload_users(self, user_ids: List[str], max_workers: Optional[int] = None) -> Dict[str, bool]:
        """
        True parallel batch preloading of multiple users' data
        
        Args:
            user_ids: List of user IDs
            max_workers: Maximum number of parallel worker threads (defaults to CPU core count)
            
        Returns:
            Loading status dictionary {user_id: success_boolean}
        """
        if not user_ids:
            return {}
        
        # Set reasonable default parallelism
        if max_workers is None:
            import os
            max_workers = min(len(user_ids), os.cpu_count() or 4, 8)  # Limit to max 8 threads to avoid overload
        
        results = {}
        start_time = time.time()
        
        logger.info(f"Starting parallel preload of {len(user_ids)} users with {max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(self.preload_user_data, user_id): user_id for user_id in user_ids}
            
            # Use progress bar to show loading progress
            with tqdm(total=len(user_ids), desc="Preloading users", unit="user") as pbar:
                for future in as_completed(futures):
                    user_id = futures[future]
                    try:
                        success = future.result(timeout=30)  # 30 second timeout
                        results[user_id] = success
                        if success:
                            pbar.set_postfix({"loaded": user_id[:12]})
                        else:
                            pbar.set_postfix({"failed": user_id[:12]})
                    except Exception as e:
                        results[user_id] = False
                        logger.error(f"Error preloading user {user_id}: {e}")
                        pbar.set_postfix({"error": user_id[:12]})
                    finally:
                        pbar.update(1)
        
        # Summarize results
        successful = sum(1 for v in results.values() if v)
        failed = len(user_ids) - successful
        total_time = time.time() - start_time
        
        logger.info(f"Parallel preload completed in {total_time:.2f}s")
        logger.info(f"Successfully preloaded: {successful}/{len(user_ids)} users")
        if failed > 0:
            logger.warning(f"Failed to preload: {failed} users")
        
        return results
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        total_episodes = sum(len(eps) for eps in self.preloaded_episodes.values())
        total_semantic = sum(len(mems) for mems in self.preloaded_semantic.values())
        
        return {
            "preloaded_users": len(self.preloaded_episodes),
            "total_episodes": total_episodes,
            "total_semantic_memories": total_semantic,
            "estimated_memory_mb": (total_episodes + total_semantic) * 0.001  # Rough estimate
        } 