"""
Memory system core module
"""

import os
import time
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from ..config import MemoryConfig
from ..models import Message, MessageBuffer, Episode, SemanticMemory
from ..utils import LLMClient, EmbeddingClient, PerformanceOptimizer
from .message_buffer import MessageBufferManager
from .boundary_detector import BoundaryDetector

logger = logging.getLogger(__name__)


class MemorySystem:
    """
    High-performance memory system main class
    
    This is the system's main interface, providing:
    - Intelligent message buffering
    - Automatic boundary detection
    - Episodic memory generation
    - Semantic memory extraction
    - Parallel search functionality
    - Optimized concurrent processing
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None, language: str = "en"):
        """
        Initialize memory system
        
        Args:
            config: System configuration, uses default if not provided
            language: Language setting for BM25 tokenization
        """
        self.config = config or MemoryConfig()
        self.language = language
        
        # Initialize clients
        self.llm_client = LLMClient(
            api_key=self.config.openai_api_key,
            model=self.config.llm_model
        )
        
        self.embedding_client = EmbeddingClient(
            api_key=self.config.openai_api_key,
            model=self.config.embedding_model
        )
        
        # Initialize performance optimizer (increased shard count)
        self.performance_optimizer = PerformanceOptimizer(
            cache_size=self.config.cache_size,
            cache_ttl=self.config.cache_ttl_seconds,
            max_workers=self.config.max_workers,
            num_cache_shards=40  # Increase shard count to reduce contention
        )
        
        # Initialize core components
        self.buffer_manager = MessageBufferManager(self.config)
        self.boundary_detector = BoundaryDetector(
            llm_client=self.llm_client,
            config=self.config
        )
        
        # Lazy initialization of storage and search components (avoid circular imports)
        self._storage = None
        self._search_engine = None
        self._episode_generator = None
        self._semantic_generator = None
        
        # User-level processing locks (avoid concurrent processing conflicts for same user)
        self._user_processing_locks: Dict[str, threading.RLock] = {}
        self._user_locks_manager = threading.RLock()
        
        # Statistics information (thread-safe)
        self.stats = {
            "messages_processed": 0,
            "episodes_created": 0,
            "semantic_memories_created": 0,
            "searches_performed": 0,
            "total_processing_time": 0.0,
            "concurrent_operations": 0
        }
        self.stats_lock = threading.Lock()
        
        # Semantic memory async generation queue
        semantic_workers = getattr(self.config, 'semantic_generation_workers', 8)  # Increase default thread count
        self._semantic_generation_executor = ThreadPoolExecutor(
            max_workers=semantic_workers,
            thread_name_prefix="semantic_gen"
        )
        self._semantic_generation_futures = {}  # Track async tasks
        self._semantic_futures_lock = threading.Lock()
        
        logger.info(f"Memory System initialized with optimized concurrency: {self.config.to_dict()}")
    
    def _get_user_processing_lock(self, owner_id: str) -> threading.RLock:
        """
        Get user-specific processing lock
        
        Args:
            owner_id: User identifier
            
        Returns:
            User-specific lock
        """
        # Fast path: if lock already exists
        if owner_id in self._user_processing_locks:
            return self._user_processing_locks[owner_id]
        
        # Slow path: create new lock
        with self._user_locks_manager:
            if owner_id not in self._user_processing_locks:
                self._user_processing_locks[owner_id] = threading.RLock()
                logger.debug(f"Created processing lock for user {owner_id}")
            return self._user_processing_locks[owner_id]
    
    @property
    def storage(self):
        """Lazy initialization of storage components"""
        if self._storage is None:
            from ..storage import EpisodeStorage, SemanticStorage
            self._storage = {
                "episode": EpisodeStorage(self.config.storage_path),
                "semantic": SemanticStorage(self.config.storage_path)
            }
        return self._storage
    
    @property
    def search_engine(self):
        """Lazy initialization of search engine"""
        if self._search_engine is None:
            from ..search import UnifiedSearchEngine
            self._search_engine = UnifiedSearchEngine(
                embedding_client=self.embedding_client,
                config=self.config,
                language=self.language
            )
        return self._search_engine
    
    def load_user_data_and_indices(self, owner_id: str):
        """
        Load user data and build/restore vector indices (supports concurrency, compatible with format changes)
        
        Args:
            owner_id: User identifier
        """
        user_lock = self._get_user_processing_lock(owner_id)
        with user_lock:  # User-level lock, doesn't block other users
            try:
                # Use cache to avoid duplicate loading
                cache_key = f"user_data_loaded_{owner_id}"
                if self.performance_optimizer.cache.contains(cache_key):
                    return
                
                # Load episodic memories
                episodes = self.storage["episode"].get_user_episodes(owner_id)
                
                # Load semantic memories
                semantic_memories = []
                if self.config.enable_semantic_memory:
                    semantic_memories = self.storage["semantic"].list_user_items(owner_id)
                
                # Check if forced index rebuild is needed（兼容性处理）
                force_rebuild = self._should_force_rebuild_indices(owner_id, episodes, semantic_memories)
                
                if episodes or semantic_memories:
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        bm25_future = None
                        vector_future = None
                        
                        if episodes:
                            if force_rebuild:
                                bm25_future = executor.submit(
                                    self.search_engine.bm25_search.index_episodes, 
                                    owner_id, episodes
                                )
                            else:
                                bm25_future = executor.submit(
                                    self.search_engine.bm25_search.index_episodes, 
                                    owner_id, episodes
                                )
                        
                        if semantic_memories:
                            if force_rebuild:
                                self.search_engine.bm25_search.index_semantic_memories(owner_id, semantic_memories)
                            else:
                                self.search_engine.bm25_search.index_semantic_memories(owner_id, semantic_memories)
                        
                        if force_rebuild:
                            vector_future = executor.submit(
                                self._force_rebuild_vector_indices,
                                owner_id, episodes, semantic_memories
                            )
                        else:                          
                            vector_future = executor.submit(
                                self.search_engine.vector_search.load_user_indices,
                                owner_id, episodes, semantic_memories
                            )
                        
                        if bm25_future:
                            bm25_future.result()
                        
                        if vector_future:
                            vector_future.result()
                    
                    logger.info(f"Loaded data and indices for user {owner_id}: {len(episodes)} episodes, {len(semantic_memories)} semantic memories (force_rebuild: {force_rebuild})")
                
                # Cache loading status
                self.performance_optimizer.cache.put(cache_key, True, ttl=3600)
                
            except Exception as e:
                logger.error(f"Error loading user data and indices for {owner_id}: {e}")
    
    def load_user_data_and_indices_for_method(self, owner_id: str, search_method: str = "vector"):
        """
        Load user data and corresponding indices based on search method (optimized version, only loads required indices)
        
        Args:
            owner_id: User identifier
            search_method: Search method ("vector", "vector_norlift", "bm25", "hybrid")
        """
        user_lock = self._get_user_processing_lock(owner_id)
        with user_lock:
            try:
                # Use cache to avoid duplicate loading
                cache_key = f"user_data_loaded_{owner_id}_{search_method}"
                if self.performance_optimizer.cache.contains(cache_key):
                    return
                
                # Load episodic memories
                episodes = self.storage["episode"].get_user_episodes(owner_id)
                
                # Load semantic memories
                semantic_memories = []
                if self.config.enable_semantic_memory:
                    semantic_memories = self.storage["semantic"].list_user_items(owner_id)
                
                # Check if forced index rebuild is needed
                force_rebuild = self._should_force_rebuild_indices(owner_id, episodes, semantic_memories)
                
                # 根据搜索方法只加载需要的索引
                if episodes or semantic_memories:
                    if search_method == "vector" or search_method == "vector_norlift":
                        # Load only vector indices
                        self._load_vector_indices_only(owner_id, episodes, semantic_memories, force_rebuild)
                    elif search_method == "bm25":
                        # Load only BM25 indices
                        self._load_bm25_indices_only(owner_id, episodes, semantic_memories, force_rebuild)
                    elif search_method == "hybrid":
                        self.load_user_data_and_indices(owner_id)
                        return
                    else:
                        raise ValueError(f"Unknown search method: {search_method}")
                    
                    logger.info(f"Loaded data and {search_method} indices for user {owner_id}: "
                              f"{len(episodes)} episodes, {len(semantic_memories)} semantic memories")
                
                # Cache loading status
                self.performance_optimizer.cache.put(cache_key, True, ttl=3600)
                
            except Exception as e:
                logger.error(f"Error loading user data and indices for {owner_id}: {e}")
    
    def _load_vector_indices_only(self, owner_id: str, episodes: List[Episode], 
                                 semantic_memories: List[SemanticMemory], force_rebuild: bool):
        try:
            if force_rebuild:
                self._force_rebuild_vector_indices(owner_id, episodes, semantic_memories)
            else:
                self.search_engine.vector_search.load_user_indices(owner_id, episodes, semantic_memories)
            
            logger.debug(f"Loaded vector indices for user {owner_id}")
            
        except Exception as e:
            logger.error(f"Error loading vector indices for {owner_id}: {e}")
            raise
    
    def _load_bm25_indices_only(self, owner_id: str, episodes: List[Episode], 
                                semantic_memories: List[SemanticMemory], force_rebuild: bool):
        try:
            if episodes:
                self.search_engine.bm25_search.index_episodes(owner_id, episodes)
            
            if semantic_memories:
                self.search_engine.bm25_search.index_semantic_memories(owner_id, semantic_memories)
            
            logger.debug(f"Loaded BM25 indices for user {owner_id}")
            
        except Exception as e:
            logger.error(f"Error loading BM25 indices for {owner_id}: {e}")
            raise
    
    def _should_force_rebuild_indices(self, owner_id: str, episodes: List[Episode], semantic_memories: List[SemanticMemory]) -> bool:
        try:
            storage_path = self.config.storage_path
            
            jsonl_path = os.path.join(storage_path, "episodes", f"{owner_id}_episodes.jsonl")
            json_path = os.path.join(storage_path, "episodes", f"{owner_id}.json")
            
            if os.path.exists(jsonl_path) and os.path.exists(json_path):
                jsonl_mtime = os.path.getmtime(jsonl_path)
                json_mtime = os.path.getmtime(json_path)
                
                if json_mtime > jsonl_mtime:
                    logger.info(f"Detected format migration for user {owner_id}, forcing index rebuild")
                    return True
            
            if episodes:
                vector_dir = os.path.join(storage_path, "episodes", "vector_db")
                embeddings_path = os.path.join(vector_dir, f"{owner_id}_embeddings.npy")
                
                if os.path.exists(embeddings_path):
                    try:
                        import numpy as np
                        embeddings = np.load(embeddings_path)
                        
                        if len(embeddings) != len(episodes):
                            logger.info(f"Vector count mismatch for user {owner_id}: {len(embeddings)} vs {len(episodes)}, forcing rebuild")
                            return True
                        
                        embeddings_mtime = os.path.getmtime(embeddings_path)
                        json_mtime = os.path.getmtime(json_path) if os.path.exists(json_path) else 0
                        
                        if json_mtime > embeddings_mtime:
                            logger.info(f"Data file newer than vector index for user {owner_id}, forcing rebuild")
                            return True
                        
                    except Exception as e:
                        logger.warning(f"Error checking vector index for user {owner_id}: {e}, forcing rebuild")
                        return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error in force rebuild check for user {owner_id}: {e}, playing safe with rebuild")
            return True
    
    def _force_rebuild_vector_indices(self, owner_id: str, episodes: List[Episode], semantic_memories: List[SemanticMemory]):
        """
        Force rebuild vector indices
        
        Args:
            owner_id: User identifier
            episodes: List of episodic memories
            semantic_memories: List of semantic memories
        """
        try:
            vector_dir = os.path.join(self.config.storage_path, "episodes", "vector_db")
            semantic_vector_dir = os.path.join(self.config.storage_path, "semantic", "vector_db")
            
            for dir_path in [vector_dir, semantic_vector_dir]:
                if os.path.exists(dir_path):
                    for filename in os.listdir(dir_path):
                        if filename.startswith(f"{owner_id}.") or filename.startswith(f"{owner_id}_"):
                            file_path = os.path.join(dir_path, filename)
                            try:
                                os.remove(file_path)
                                logger.debug(f"Removed old vector index file: {file_path}")
                            except Exception as e:
                                logger.warning(f"Failed to remove old vector index file {file_path}: {e}")
            
            if episodes:
                self.search_engine.vector_search.index_episodes(owner_id, episodes)
            
            if semantic_memories:
                self.search_engine.vector_search.index_semantic_memories(owner_id, semantic_memories)
            
            logger.info(f"Successfully rebuilt vector indices for user {owner_id}")
            
        except Exception as e:
            logger.error(f"Error rebuilding vector indices for user {owner_id}: {e}")
            # 回退到正常加载
            self.search_engine.vector_search.load_user_indices(owner_id, episodes, semantic_memories)
    
    @property
    def episode_generator(self):
        """Lazy initialization of episode generator"""
        if self._episode_generator is None:
            from ..generation import EpisodeGenerator
            self._episode_generator = EpisodeGenerator(
                llm_client=self.llm_client,
                config=self.config
            )
        return self._episode_generator
    
    @property
    def semantic_generator(self):
        """Lazy initialization of semantic generator"""
        if self._semantic_generator is None:
            from ..generation import SemanticGenerator
            self._semantic_generator = SemanticGenerator(
                llm_client=self.llm_client,
                embedding_client=self.embedding_client,
                config=self.config,
                vector_search=self.search_engine.vector_search  # 传递向量搜索引擎实例
            )
        return self._semantic_generator
    
    def add_messages(
        self, 
        owner_id: str, 
        messages: List[Dict[str, str]], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add messages to user buffer (optimized concurrent processing)
        
        Args:
            owner_id: User unique identifier
            messages: Message list, each message contains role and content
            metadata: Additional metadata
            
        Returns:
            Processing result, including buffer status and generated memories
        """
        start_time = time.time()
        
        # Use user-level lock to avoid concurrent conflicts for same user
        user_lock = self._get_user_processing_lock(owner_id)
        with user_lock:
            try:
                with self.stats_lock:
                    self.stats["concurrent_operations"] += 1
                
                # Convert message format
                message_objects = []
                for msg_data in messages:
                    # Process timestamp
                    timestamp = None
                    if "timestamp" in msg_data:
                        timestamp_value = msg_data["timestamp"]
                        if isinstance(timestamp_value, str):
                            try:
                                timestamp = datetime.fromisoformat(timestamp_value.replace('Z', '+00:00'))
                            except ValueError:
                                logger.warning(f"Invalid timestamp format: {timestamp_value}, using current time")
                                timestamp = None
                        elif isinstance(timestamp_value, datetime):
                            timestamp = timestamp_value
                    
                    # Create message object, preserving original timestamp
                    if timestamp:
                        message = Message(
                            role=msg_data["role"],
                            content=msg_data["content"],
                            timestamp=timestamp,
                            metadata=msg_data.get("metadata", {})
                        )
                    else:
                        message = Message(
                            role=msg_data["role"],
                            content=msg_data["content"],
                            metadata=msg_data.get("metadata", {})
                        )
                    message_objects.append(message)
                
                # Get or create buffer (user-level lock already protected)
                buffer = self.buffer_manager.get_or_create_buffer(owner_id)
                
                # Process result
                result = {
                    "status": "success",
                    "buffer_size_before": buffer.size(),
                    "messages_added": len(message_objects),
                    "episodes_created": [],
                    "semantic_memories_created": [],
                    "boundary_detections": []
                }
                
                # Batch process messages for efficiency
                episodes_to_create = []
                
                # Process messages one by one (supports boundary detection)
                for message in message_objects:
                    # Check if boundary detection is needed
                    should_create_episode = False
                    boundary_reason = ""
                    
                    if self.config.enable_smart_boundary and not buffer.is_empty():
                        # Improved boundary detection logic: when buffer message count exceeds threshold, exclude last message
                        detection_buffer = buffer
                        if (self.config.boundary_exclude_last_message and 
                            buffer.size() > self.config.boundary_exclude_threshold):
                            # Create a temporary buffer, only containing n-1 messages
                            temp_buffer = MessageBuffer(owner_id=buffer.owner_id)
                            all_messages = buffer.get_messages()
                            # Add all messages except the last one
                            for msg in all_messages[:-1]:
                                temp_buffer.add_message(msg)
                            detection_buffer = temp_buffer
                            logger.debug(f"Using modified buffer for boundary detection: {len(all_messages)-1} messages (excluded last message as transition)")
                        
                        # Perform boundary detection (using cache)
                        detection_result = self.performance_optimizer.cached_call(
                            self.boundary_detector.detect_boundary,
                            f"boundary_detection_{owner_id}_{buffer.size()}",  # Update cache key to distinguish different buffer states
                            detection_buffer, [message]
                        )
                        
                        result["boundary_detections"].append(detection_result)
                        
                        if detection_result.get("should_end", False):
                            should_create_episode = True
                            boundary_reason = detection_result.get("reason", "Intelligent boundary detection")
                    
                    # Check buffer size limit
                    elif buffer.size() >= self.config.buffer_size_max:
                        should_create_episode = True
                        boundary_reason = "Buffer reached maximum size"
                    
                    # If episode creation is needed, process current buffer first
                    if should_create_episode and not buffer.is_empty():
                        episodes_to_create.append({
                            "buffer_messages": buffer.get_messages().copy(),
                            "boundary_reason": boundary_reason
                        })
                        buffer.clear()  # Clear buffer
                    
                    # Add new message to buffer
                    buffer.add_message(message)
                
                # Batch create episodes (parallel processing)
                if episodes_to_create:
                    created_episodes = self._batch_create_episodes(owner_id, episodes_to_create)
                    result["episodes_created"] = created_episodes
                
                # Check if semantic memory generation is needed (async processing)
                if self.config.enable_semantic_memory and result["episodes_created"]:
                    # Trigger semantic memory generation for all created episodes
                    semantic_tasks_scheduled = 0
                    for episode_info in result["episodes_created"]:
                        if "episode_object" in episode_info:
                            # Asynchronous generation of semantic memories
                            self._schedule_semantic_generation(owner_id, episode_info["episode_object"])
                            semantic_tasks_scheduled += 1
                    
                    if semantic_tasks_scheduled > 0:
                        result["semantic_generation_scheduled"] = True
                        result["semantic_tasks_scheduled"] = semantic_tasks_scheduled
                        logger.info(f"Scheduled {semantic_tasks_scheduled} async semantic memory generation tasks for user {owner_id}")
                
                # Update final buffer size
                result["buffer_size_after"] = buffer.size()
                
                # Update statistics
                with self.stats_lock:
                    self.stats["messages_processed"] += len(message_objects)
                    self.stats["episodes_created"] += len(result["episodes_created"])
                    self.stats["semantic_memories_created"] += len(result["semantic_memories_created"])
                    self.stats["total_processing_time"] += time.time() - start_time
                    self.stats["concurrent_operations"] -= 1
                
                logger.info(f"Processed {len(message_objects)} messages for user {owner_id}")
                
                return result
                
            except Exception as e:
                with self.stats_lock:
                    self.stats["concurrent_operations"] -= 1
                logger.error(f"Error processing messages for user {owner_id}: {e}")
                return {
                    "status": "error",
                    "error": str(e),
                    "buffer_size": self.buffer_manager.get_buffer_size(owner_id)
                }
    
    def _batch_create_episodes(self, owner_id: str, episodes_data: List[Dict]) -> List[Dict[str, Any]]:
        """
        Batch create episodes (supports parallel processing)
        
        Args:
            owner_id: User identifier
            episodes_data: episodes data list
            
        Returns:
            List of created episodes information
        """
        if not episodes_data:
            return []
        
        created_episodes = []
        
        # If there is only one episode, create it directly
        if len(episodes_data) == 1:
            episode_data = episodes_data[0]
            episode_info = self._create_episode_from_messages(
                owner_id, 
                episode_data["buffer_messages"], 
                episode_data["boundary_reason"]
            )
            if episode_info:
                created_episodes.append(episode_info)
        else:
            # Multiple episodes can be created in parallel
            with ThreadPoolExecutor(max_workers=min(len(episodes_data), 3)) as executor:
                future_to_data = {
                    executor.submit(
                        self._create_episode_from_messages,
                        owner_id,
                        ep_data["buffer_messages"],
                        ep_data["boundary_reason"]
                    ): ep_data for ep_data in episodes_data
                }
                
                for future in as_completed(future_to_data):
                    try:
                        episode_info = future.result()
                        if episode_info:
                            created_episodes.append(episode_info)
                    except Exception as e:
                        logger.error(f"Error creating episode in batch: {e}")
                        continue
        
        return created_episodes
    
    def search(
        self,
        owner_id: str,
        query: str,
        top_k: Optional[int] = None,
        memory_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search user's memories (optimized concurrency)
        
        Args:
            owner_id: User identifier
            query: Search query
            top_k: Number of results to return
            memory_types: Memory types to search ["episodic", "semantic"]
            
        Returns:
            List of search results
        """
        start_time = time.time()
        
        try:
            # If top_k not specified, use maximum of both memory types
            if top_k is None:
                top_k = max(self.config.search_top_k_episodes, self.config.search_top_k_semantic)
            memory_types = memory_types or ["episodic", "semantic"]
            
            # Ensure user data and indices are loaded (using cache to avoid duplicate loading)
            self.load_user_data_and_indices(owner_id)
            
            # Execute search (using cache)
            cache_key = f"search_{owner_id}_{hash(query)}_{top_k}_{','.join(sorted(memory_types))}"
            results = self.performance_optimizer.cached_call(
                self.search_engine.search,
                cache_key,
                owner_id=owner_id,
                query=query,
                top_k=top_k,
                memory_types=memory_types
            )
            
            # Update statistics
            with self.stats_lock:
                self.stats["searches_performed"] += 1
            
            search_time = time.time() - start_time
            logger.info(f"Search completed for user {owner_id} in {search_time:.2f}s, found {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching memories for user {owner_id}: {e}")
            return []
    
    def search_all(
        self,
        user_id: str,
        query: str,
        top_k_episodes: Optional[int] = None,
        top_k_semantic: Optional[int] = None,
        search_method: str = "hybrid",
        use_optimized_loading: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search all types of memories and return categorized results (optimized concurrency)
        
        Args:
            user_id: User identifier
            query: Search query
            top_k_episodes: Number of episodic memories to return
            top_k_semantic: Number of semantic memories to return
            search_method: Search method ("hybrid", "bm25", "vector")
            use_optimized_loading: Whether to use optimized index loading (only load required indices)
            
        Returns:
            Categorized search results dictionary {"episodic": [...], "semantic": [...]}
        """
        start_time = time.time()
        
        try:
            # Use default values from configuration
            top_k_episodes = top_k_episodes or self.config.search_top_k_episodes
            top_k_semantic = top_k_semantic or self.config.search_top_k_semantic
            
            # Ensure user data and indices are loaded
            if use_optimized_loading:
                # Use optimized loading method, only load required indices
                self.load_user_data_and_indices_for_method(user_id, search_method)
            else:
                # Use original method, load all indices
                self.load_user_data_and_indices(user_id)
            
            # Parallel search for different types of memories
            with ThreadPoolExecutor(max_workers=2) as executor:
                episode_future = executor.submit(
                    self._search_episodes_by_method,
                    user_id, query, top_k_episodes, search_method
                )
                
                semantic_future = executor.submit(
                    self._search_semantic_by_method,
                    user_id, query, top_k_semantic, search_method
                )
                
                episode_results = episode_future.result()
                semantic_results = semantic_future.result()
            
            # Update statistics
            with self.stats_lock:
                self.stats["searches_performed"] += 1
            
            search_time = time.time() - start_time
            logger.info(f"Search_all completed for user {user_id} in {search_time:.2f}s, "
                       f"found {len(episode_results)} episodes and {len(semantic_results)} semantic memories")
            
            return {
                "episodic": episode_results,
                "semantic": semantic_results
            }
            
        except Exception as e:
            logger.error(f"Error in search_all for user {user_id}: {e}")
            return {"episodic": [], "semantic": []}
    
    def _search_episodes_by_method(self, user_id: str, query: str, top_k: int, search_method: str):
        """Search episodes by search method"""
        if search_method == "hybrid":
            return self.search_engine.search_episodes(user_id, query, top_k)
        elif search_method == "bm25":
            return self.search_engine.bm25_search.search_episodes(user_id, query, top_k)
        elif search_method == "vector":
            return self.search_engine.vector_search.search_episodes(user_id, query, top_k)
        elif search_method == "vector_norlift":
            return self.search_engine.search_episodes(user_id, query, top_k, "vector_norlift")
        else:
            raise ValueError(f"Unknown search method: {search_method}")
    
    def _search_semantic_by_method(self, user_id: str, query: str, top_k: int, search_method: str):
        """Search semantic memories by search method"""
        if search_method == "hybrid":
            return self.search_engine.search_semantic_memories(user_id, query, top_k)
        elif search_method == "bm25":
            return self.search_engine.bm25_search.search_semantic_memories(user_id, query, top_k)
        elif search_method == "vector":
            return self.search_engine.vector_search.search_semantic_memories(user_id, query, top_k)
        elif search_method == "vector_norlift":
            # NOR-LIFT 仅用于情景级排序，这里回退为向量检索
            return self.search_engine.vector_search.search_semantic_memories(user_id, query, top_k)
        else:
            raise ValueError(f"Unknown search method: {search_method}")
    
    def force_episode_creation(self, owner_id: str) -> Optional[Dict[str, Any]]:
        """
        Force episode creation from current buffer (supports user-level lock)
        
        Args:
            owner_id: User identifier
            
        Returns:
            Created episode information, or None if buffer is empty
        """
        user_lock = self._get_user_processing_lock(owner_id)
        with user_lock:
            try:
                buffer = self.buffer_manager.get_buffer(owner_id)
                if not buffer or buffer.is_empty():
                    return None
                
                # Get messages and clear buffer
                messages = buffer.clear()
                
                episode_info = self._create_episode_from_messages(
                    owner_id, messages, "Force episode creation"
                )
                
                # If episode creation is successful and semantic memory is enabled, schedule async generation
                if episode_info and self.config.enable_semantic_memory and "episode_object" in episode_info:
                    self._schedule_semantic_generation(owner_id, episode_info["episode_object"])
                    episode_info["semantic_generation_scheduled"] = True
                    logger.info(f"Scheduled async semantic memory generation for user {owner_id} (force creation)")
                
                return episode_info
                
            except Exception as e:
                logger.error(f"Error forcing episode creation for user {owner_id}: {e}")
                return None
    
    def clear_buffer(self, owner_id: str) -> bool:
        """
        Clear user buffer
        
        Args:
            owner_id: User identifier
            
        Returns:
            Whether operation was successful
        """
        try:
            return self.buffer_manager.clear_buffer(owner_id)
        except Exception as e:
            logger.error(f"Error clearing buffer for user {owner_id}: {e}")
            return False
    
    def get_stats(self, owner_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics (thread-safe)
        
        Args:
            owner_id: User identifier, if provided returns user-specific statistics
            
        Returns:
            Statistics dictionary
        """
        try:
            # System-level statistics (thread-safe)
            with self.stats_lock:
                system_stats = {
                    "system": self.stats.copy(),
                    "performance": self.performance_optimizer.get_stats(),
                    "config": self.config.to_dict()
                }
            
            # User-specific statistics
            if owner_id:
                buffer = self.buffer_manager.get_buffer(owner_id)
                user_stats = {
                    "buffer_size": buffer.size() if buffer else 0,
                    "buffer_created_at": buffer.created_at.isoformat() if buffer else None,
                    "buffer_last_updated": buffer.last_updated.isoformat() if buffer else None,
                }
                
                # Store statistics
                try:
                    user_stats.update(self.storage["episode"].get_user_stats(owner_id))
                    if self.config.enable_semantic_memory:
                        user_stats.update(self.storage["semantic"].get_user_stats(owner_id))
                except:
                    pass
                
                system_stats["user"] = user_stats
            
            return system_stats
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}
    
    def delete_user_data(self, owner_id: str) -> Dict[str, bool]:
        """
        Delete all user data (supports user-level lock)
        
        Args:
            owner_id: User identifier
            
        Returns:
            Deletion status for each component
        """
        user_lock = self._get_user_processing_lock(owner_id)
        with user_lock:
            results = {
                "buffer": False,
                "episodic_storage": False,
                "semantic_storage": False,
                "search_index": False
            }
            
            try:
                # Clear buffer
                results["buffer"] = self.buffer_manager.delete_buffer(owner_id)
                
                # Delete storage data
                results["episodic_storage"] = self.storage["episode"].delete_user_data(owner_id)
                
                if self.config.enable_semantic_memory:
                    results["semantic_storage"] = self.storage["semantic"].delete_user_data(owner_id)
                else:
                    results["semantic_storage"] = True
                
                # Clear search index
                results["search_index"] = self.search_engine.clear_user_index(owner_id)
                
                # Clear user processing lock
                with self._user_locks_manager:
                    if owner_id in self._user_processing_locks:
                        del self._user_processing_locks[owner_id]
                
                # Clear related cache
                cache_key = f"user_data_loaded_{owner_id}"
                if self.performance_optimizer.cache.contains(cache_key):
                    # Simple cache cleanup (sharding system will automatically expire)
                    pass
                
                logger.info(f"Deleted all data for user {owner_id}")
                
            except Exception as e:
                logger.error(f"Error deleting user data for {owner_id}: {e}")
            
            return results
    
    def _create_episode_from_messages(
        self, 
        owner_id: str, 
        messages: List[Message], 
        boundary_reason: str
    ) -> Optional[Dict[str, Any]]:
        """
        Create episode from message list (optimized version)
        
        Args:
            owner_id: User identifier
            messages: Message list
            boundary_reason: Boundary detection reason
            
        Returns:
            Created episode information
        """
        try:
            # Check minimum message count
            if len(messages) < self.config.episode_min_messages:
                return None
            
            # Generate episode (using cache)
            episode = self.performance_optimizer.cached_call(
                self.episode_generator.generate_episode,
                f"episode_gen_{owner_id}_{len(messages)}",
                user_id=owner_id,
                messages=messages,
                boundary_reason=boundary_reason
            )
            
            # Save episode
            episode_id = self.storage["episode"].save_episode(episode)
            
            # Index episode (for search, using incremental update)
            self.search_engine.add_episode(owner_id, episode)
            
            logger.info(f"Created episode {episode_id} for user {owner_id}")
            
            return {
                "episode_id": episode_id,
                "title": episode.title,
                "message_count": episode.message_count,
                "boundary_reason": boundary_reason,
                "episode_object": episode  # Add episode object for semantic memory generation
            }
            
        except Exception as e:
            logger.error(f"Error creating episode from messages: {e}")
            return None
    
    def _create_episode_from_buffer(
        self, 
        owner_id: str, 
        buffer: MessageBuffer, 
        boundary_reason: str
    ) -> Optional[Dict[str, Any]]:
        """
        Create episode from buffer (backward compatibility)
        
        Args:
            owner_id: User identifier
            buffer: Message buffer
            boundary_reason: Boundary detection reason
            
        Returns:
            Created episode information
        """
        try:
            # Check minimum message count
            if buffer.size() < self.config.episode_min_messages:
                return None
            
            # Clear buffer and get messages
            messages = buffer.clear()
            
            return self._create_episode_from_messages(owner_id, messages, boundary_reason)
            
        except Exception as e:
            logger.error(f"Error creating episode from buffer: {e}")
            return None
    
    def _schedule_semantic_generation(self, owner_id: str, new_episode: Episode):
        """
        Schedule async semantic memory generation task
        
        Args:
            owner_id: User identifier
            new_episode: Newly created episode
        """
        # Create task ID
        task_id = f"{owner_id}_{new_episode.episode_id}"
        
        # Submit async task
        future = self._semantic_generation_executor.submit(
            self._async_generate_semantic_memories,
            owner_id,
            new_episode
        )
        
        # Record task
        with self._semantic_futures_lock:
            self._semantic_generation_futures[task_id] = {
                "future": future,
                "owner_id": owner_id,
                "episode_id": new_episode.episode_id,
                "started_at": time.time()
            }
        
        # Set completion callback
        future.add_done_callback(
            lambda f: self._on_semantic_generation_complete(task_id, f)
        )
    
    def _async_generate_semantic_memories(self, owner_id: str, new_episode: Episode) -> List[Dict[str, Any]]:
        """
        Asynchronous generation of semantic memories (executed in background thread)
        
        Args:
            owner_id: User identifier
            new_episode: Newly created episode
            
        Returns:
            List of generated semantic memories
        """
        try:
            logger.info(f"Starting async semantic generation for user {owner_id}, episode {new_episode.episode_id}")
            start_time = time.time()
            
            # Get all user episodes
            all_episodes = self.storage["episode"].get_user_episodes(owner_id)
            
            # Exclude newly created episode
            existing_episodes = [ep for ep in all_episodes if ep.episode_id != new_episode.episode_id]
            
            # If prediction-correction mode is enabled, get existing semantic memories
            existing_semantic_memories = None
            if self.config.enable_prediction_correction:
                existing_semantic_memories = self.storage["semantic"].list_user_items(owner_id)
                logger.debug(f"Loaded {len(existing_semantic_memories)} existing semantic memories for prediction-correction")
            
            # Generate semantic memories
            semantic_memories = self.semantic_generator.check_and_generate_semantic_memories(
                user_id=owner_id,
                new_episode=new_episode,
                existing_episodes=existing_episodes,
                existing_semantic_memories=existing_semantic_memories
            )
            
            # Save semantic memories
            saved_memories = []
            for memory in semantic_memories:
                # Check for duplicates
                if not self._is_duplicate_semantic_memory(owner_id, memory):
                    memory_id = self.storage["semantic"].save_semantic_memory(memory)
                    
                    # Exception handling when adding to search index
                    try:
                        self.search_engine.add_semantic_memory(owner_id, memory)
                    except RuntimeError as e:
                        if "cannot schedule new futures" in str(e):
                            logger.warning(f"Search engine executor already shut down, skipping index update for memory {memory_id}")
                        else:
                            raise e
                    
                    saved_memories.append({
                        "memory_id": memory_id,
                        "knowledge_type": memory.knowledge_type,
                        "content": memory.content
                    })
            
            generation_time = time.time() - start_time
            
            # Update statistics
            with self.stats_lock:
                self.stats["semantic_memories_created"] += len(saved_memories)
            
            logger.info(f"Async semantic generation completed for user {owner_id}: "
                       f"{len(saved_memories)} memories generated in {generation_time:.2f}s")
            
            return saved_memories
            
        except Exception as e:
            logger.error(f"Error in async semantic memory generation: {e}")
            return []
    
    def _on_semantic_generation_complete(self, task_id: str, future):
        """
        Callback for completed semantic memory generation task
        
        Args:
            task_id: Task ID
            future: Completed Future object
        """
        try:
            result = future.result()
            logger.debug(f"Semantic generation task {task_id} completed with {len(result)} memories")
        except Exception as e:
            logger.error(f"Semantic generation task {task_id} failed: {e}")
        finally:
            # Clean up completed tasks
            with self._semantic_futures_lock:
                if task_id in self._semantic_generation_futures:
                    del self._semantic_generation_futures[task_id]
    
    def get_semantic_generation_status(self, owner_id: str = None) -> Dict[str, Any]:
        """
        Get status of semantic memory generation tasks
        
        Args:
            owner_id: User identifier, if provided returns only tasks for that user
            
        Returns:
            Task status information
        """
        with self._semantic_futures_lock:
            all_tasks = []
            
            for task_id, task_info in self._semantic_generation_futures.items():
                if owner_id and task_info["owner_id"] != owner_id:
                    continue
                
                status = {
                    "task_id": task_id,
                    "owner_id": task_info["owner_id"],
                    "episode_id": task_info["episode_id"],
                    "started_at": task_info["started_at"],
                    "running_time": time.time() - task_info["started_at"],
                    "done": task_info["future"].done(),
                    "running": task_info["future"].running()
                }
                
                if task_info["future"].done():
                    try:
                        result = task_info["future"].result(timeout=0)
                        status["result_count"] = len(result)
                        status["status"] = "completed"
                    except Exception as e:
                        status["error"] = str(e)
                        status["status"] = "failed"
                else:
                    status["status"] = "running" if task_info["future"].running() else "pending"
                
                all_tasks.append(status)
            
            return {
                "active_tasks": len(all_tasks),
                "tasks": all_tasks
            }
    
    def wait_for_semantic_generation(self, owner_id: str, timeout: float = 30.0) -> bool:
        """
        Wait for all semantic memory generation tasks for a specific user to complete
        
        Args:
            owner_id: User identifier
            timeout: Timeout (seconds)
            
        Returns:
            Whether all tasks were successfully completed
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Get tasks for the specific user
            user_tasks = []
            with self._semantic_futures_lock:
                for task_id, task_info in self._semantic_generation_futures.items():
                    if task_info["owner_id"] == owner_id:
                        user_tasks.append(task_info["future"])
            
            # If there are no tasks, return immediately
            if not user_tasks:
                return True
            
            # Check if all tasks are completed
            all_done = all(future.done() for future in user_tasks)
            if all_done:
                # Check if all tasks were successful
                all_success = True
                for future in user_tasks:
                    try:
                        future.result(timeout=0)
                    except Exception:
                        all_success = False
                return all_success
            
            # 短暂等待
            time.sleep(0.1)
        
        logger.warning(f"Timeout waiting for semantic generation tasks for user {owner_id}")
        return False
    
    def _is_duplicate_semantic_memory(self, owner_id: str, memory: SemanticMemory) -> bool:
        """
        Check if semantic memory is duplicate (using existing embedding vector, avoiding duplicate API calls)
        
        Args:
            owner_id: User identifier
            memory: Semantic memory
            
        Returns:
            Whether the semantic memory is duplicate
        """
        try:
            # Check if there are existing semantic memories and embedding vectors
            if (owner_id not in self.search_engine.vector_search.semantic_embeddings or 
                owner_id not in self.search_engine.vector_search.semantic_data):
                return False
            
            existing_embeddings = self.search_engine.vector_search.semantic_embeddings[owner_id]
            existing_memories = self.search_engine.vector_search.semantic_data[owner_id]
            
            if len(existing_memories) == 0:
                return False
            
            # Generate embedding for new memory (1 API call)
            memory_embedding = self.performance_optimizer.cached_call(
                self.embedding_client.embed_text,
                f"embed_{hash(memory.content)}",
                memory.content
            )
            
            # Compare with existing embeddings (no additional API calls)
            memory_embedding_np = np.array(memory_embedding)
            
            for i, existing in enumerate(existing_memories):
                if existing.knowledge_type == memory.knowledge_type:
                    # Use stored embedding vector directly, no need to regenerate
                    existing_embedding_np = existing_embeddings[i]
                    
                    # Calculate cosine similarity
                    similarity = self._cosine_similarity_np(memory_embedding_np, existing_embedding_np)
                    
                    if similarity > self.config.semantic_similarity_threshold:
                        logger.debug(f"Found duplicate semantic memory (similarity: {similarity:.3f})")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking semantic memory duplication: {e}")
            # Fall back to traditional method (for compatibility)
            return self._is_duplicate_semantic_memory_fallback(owner_id, memory)
    
    def _cosine_similarity_np(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity using numpy
        
        Args:
            vec1: Vector 1
            vec2: Vector 2
            
        Returns:
            Cosine similarity
        """
        try:
            # Normalize vectors
            vec1_norm = vec1 / np.linalg.norm(vec1)
            vec2_norm = vec2 / np.linalg.norm(vec2)
            
            # Calculate cosine similarity
            similarity = np.dot(vec1_norm, vec2_norm)
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def _is_duplicate_semantic_memory_fallback(self, owner_id: str, memory: SemanticMemory) -> bool:
        """
        Fallback method for duplicate detection (for compatibility)
        
        Args:
            owner_id: User identifier
            memory: 语义记忆
            
        Returns:
            Whether the semantic memory is duplicate
        """
        try:
            # Use vector similarity to check for duplicates
            existing_memories = self.storage["semantic"].list_user_items(owner_id)
            
            if not existing_memories:
                return False
            
            # Calculate similarity with existing memories (using cache)
            memory_embedding = self.performance_optimizer.cached_call(
                self.embedding_client.embed_text,
                f"embed_{hash(memory.content)}",
                memory.content
            )
            
            for existing in existing_memories:
                if existing.knowledge_type == memory.knowledge_type:
                    existing_embedding = self.performance_optimizer.cached_call(
                        self.embedding_client.embed_text,
                        f"embed_{hash(existing.content)}",
                        existing.content
                    )
                    
                    similarity = self.embedding_client.cosine_similarity(
                        memory_embedding, existing_embedding
                    )
                    
                    if similarity > self.config.semantic_similarity_threshold:
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking semantic memory duplication: {e}")
            return False
    
    def clear_user_cache(self, owner_id: str) -> bool:
        """
        Clear all user caches (used to solve cache inconsistency after format migration)
        
        Args:
            owner_id: User identifier
            
        Returns:
            Whether cache clearing was successful
        """
        try:
            # Clear performance optimizer cache
            cache_keys = [
                f"user_data_loaded_{owner_id}",
                f"episode_gen_{owner_id}",
                f"semantic_gen_{owner_id}",
                f"boundary_detection_{owner_id}",
            ]
            
            for key in cache_keys:
                if self.performance_optimizer.cache.contains(key):
                    self.performance_optimizer.cache.remove(key)
            
            # Clear search-related caches (fuzzy matching)
            all_cache_keys = self.performance_optimizer.cache.list_keys()
            for key in all_cache_keys:
                if owner_id in key and ("search_" in key or "_gen_" in key):
                    self.performance_optimizer.cache.remove(key)
            
            # Clear storage layer cache
            if hasattr(self.storage["episode"], '_cache'):
                with getattr(self.storage["episode"], '_cache_lock', threading.RLock()):
                    if owner_id in self.storage["episode"]._cache:
                        del self.storage["episode"]._cache[owner_id]
                    if owner_id in getattr(self.storage["episode"], '_cache_timestamps', {}):
                        del self.storage["episode"]._cache_timestamps[owner_id]
            
            # Clear search engine memory data
            if hasattr(self.search_engine, 'bm25_search'):
                # Clear BM25 search cache
                if owner_id in self.search_engine.bm25_search.episode_data:
                    del self.search_engine.bm25_search.episode_data[owner_id]
                if owner_id in self.search_engine.bm25_search.episode_indices:
                    del self.search_engine.bm25_search.episode_indices[owner_id]
                if owner_id in self.search_engine.bm25_search.episode_tokenized_texts:
                    del self.search_engine.bm25_search.episode_tokenized_texts[owner_id]
                    
                # Clear semantic memory BM25 cache
                if owner_id in self.search_engine.bm25_search.semantic_data:
                    del self.search_engine.bm25_search.semantic_data[owner_id]
                if owner_id in self.search_engine.bm25_search.semantic_indices:
                    del self.search_engine.bm25_search.semantic_indices[owner_id]
                if owner_id in self.search_engine.bm25_search.semantic_tokenized_texts:
                    del self.search_engine.bm25_search.semantic_tokenized_texts[owner_id]
            
            if hasattr(self.search_engine, 'vector_search'):
                # Clear vector search cache
                if owner_id in self.search_engine.vector_search.episode_data:
                    del self.search_engine.vector_search.episode_data[owner_id]
                if owner_id in self.search_engine.vector_search.episode_indices:
                    del self.search_engine.vector_search.episode_indices[owner_id]
                if owner_id in self.search_engine.vector_search.episode_embeddings:
                    del self.search_engine.vector_search.episode_embeddings[owner_id]
                    
                # Clear semantic memory vector cache
                if owner_id in self.search_engine.vector_search.semantic_data:
                    del self.search_engine.vector_search.semantic_data[owner_id]
                if owner_id in self.search_engine.vector_search.semantic_indices:
                    del self.search_engine.vector_search.semantic_indices[owner_id]
                if owner_id in self.search_engine.vector_search.semantic_embeddings:
                    del self.search_engine.vector_search.semantic_embeddings[owner_id]
            
            logger.info(f"Successfully cleared all caches for user {owner_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing user cache for {owner_id}: {e}")
            return False
    
    def rebuild_user_indices(self, owner_id: str) -> bool:
        """
        Force rebuild all search indices for a user (used to solve format migration issues)
        
        Args:
            owner_id: User identifier
            
        Returns:
            Whether index rebuilding was successful
        """
        try:
            # First clear cache
            self.clear_user_cache(owner_id)
            
            # Force reload data and indices
            self.load_user_data_and_indices(owner_id)
            
            logger.info(f"Successfully rebuilt indices for user {owner_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error rebuilding indices for user {owner_id}: {e}")
            return False
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        # Wait for all semantic memory generation tasks to complete
        logger.info("Waiting for async semantic generation tasks to complete...")
        # Python 3.9 and earlier versions do not support timeout parameter
        try:
            self._semantic_generation_executor.shutdown(wait=True)
        except Exception as e:
            logger.warning(f"Error during semantic generation executor shutdown: {e}")
        
        # Close search engine thread pool
        if hasattr(self, '_search_engine') and self._search_engine is not None:
            if hasattr(self._search_engine, 'executor'):
                logger.info("Shutting down search engine executor...")
                try:
                    self._search_engine.executor.shutdown(wait=True)
                except Exception as e:
                    logger.warning(f"Error during search engine executor shutdown: {e}")
        
        # Clean up resources
        if hasattr(self.performance_optimizer, 'executor'):
            try:
                self.performance_optimizer.executor.shutdown(wait=True)
            except Exception as e:
                logger.warning(f"Error during performance optimizer executor shutdown: {e}") 