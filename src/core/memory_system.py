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
from ..domain.interfaces import (
    EpisodeRepository,
    SemanticRepository,
    VectorIndex,
    LexicalIndex,
    EpisodeGenerator as EpisodeGeneratorInterface,
    SemanticGenerator as SemanticGeneratorInterface,
)
from ..services.providers import DefaultProviders
from ..services.cache import PerUserCache, SemanticEmbeddingCache
from ..services.event_bus import EventBus
from ..services.task_manager import SemanticTaskManager
from ..services.metrics import MetricsReporter, LoggingMetricsReporter
from .message_buffer import MessageBufferManager

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
    
    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        language: str = "en",
        llm_client: Optional[LLMClient] = None,
        embedding_client: Optional[EmbeddingClient] = None,
        episode_repository: Optional[EpisodeRepository] = None,
        semantic_repository: Optional[SemanticRepository] = None,
        vector_index: Optional[VectorIndex] = None,
        lexical_index: Optional[LexicalIndex] = None,
        episode_generator: Optional[EpisodeGeneratorInterface] = None,
        semantic_generator: Optional[SemanticGeneratorInterface] = None,
        metrics_reporter: Optional[MetricsReporter] = None,
    ):
        """
        Initialize memory system
        
        Args:
            config: System configuration, uses default if not provided
            language: Language setting for BM25 tokenization
            llm_client: Optional custom LLM client (useful for testing or alternative providers)
            embedding_client: Optional custom embedding client
        """
        self.config = config or MemoryConfig()
        self.language = language
        
        # Initialize clients
        providers = None
        if any(
            component is None
            for component in (
                episode_repository,
                semantic_repository,
                vector_index,
                lexical_index,
                episode_generator,
                semantic_generator,
            )
        ):
            providers = DefaultProviders(self.config, llm_client=llm_client, embedding_client=embedding_client)

        self.llm_client = llm_client or (providers.llm_client if providers else LLMClient(
            api_key=self.config.openai_api_key,
            model=self.config.llm_model,
            reasoning_effort=self.config.reasoning_effort
        ))

        self.embedding_client = embedding_client or (providers.embedding_client if providers else EmbeddingClient(
            api_key=self.config.openai_api_key,
            model=self.config.embedding_model
        ))

        if providers is None:
            from ..storage.episode_storage import EpisodeStorage
            from ..storage.semantic_storage import SemanticStorage
            from ..search.chroma_search import ChromaSearchEngine
            from ..search.bm25_search import BM25Search
            from ..generation.episode_generator import EpisodeGenerator
            from ..generation.semantic_generator import SemanticGenerator
            from ..infrastructure.repositories import EpisodeStorageRepository, SemanticStorageRepository
            from ..infrastructure.indices import Bm25Index, ChromaVectorIndex

            providers_episode_repo = EpisodeStorageRepository(EpisodeStorage(self.config.storage_path))
            providers_semantic_repo = SemanticStorageRepository(SemanticStorage(self.config.storage_path))
            providers_vector_index = ChromaVectorIndex(ChromaSearchEngine(self.embedding_client, self.config))
            providers_lexical_index = Bm25Index(BM25Search(language=self.language))
            providers_episode_gen = EpisodeGenerator(self.llm_client, self.config)
            providers_semantic_gen = SemanticGenerator(
                self.llm_client,
                self.embedding_client,
                self.config,
                vector_search=providers_vector_index._backend,
            )
        else:
            providers_episode_repo = providers.episode_repository()
            providers_semantic_repo = providers.semantic_repository()
            providers_vector_index = providers.vector_index()
            providers_lexical_index = providers.lexical_index()
            providers_episode_gen = providers.episode_generator()
            providers_semantic_gen = providers.semantic_generator()

        self._episode_repository = episode_repository or providers_episode_repo
        self._semantic_repository = semantic_repository or providers_semantic_repo
        self._vector_index = vector_index or providers_vector_index
        self._lexical_index = lexical_index or providers_lexical_index
        self._episode_generator = episode_generator or providers_episode_gen
        self._semantic_generator = semantic_generator or providers_semantic_gen
        
        # Initialize performance optimizer (increased shard count)
        self.performance_optimizer = PerformanceOptimizer(
            cache_size=self.config.cache_size,
            cache_ttl=self.config.cache_ttl_seconds,
            max_workers=self.config.max_workers,
            num_cache_shards=40  # Increase shard count to reduce contention
        )
        
        # Initialize core components
        self.buffer_manager = MessageBufferManager(self.config)
        # Lazy initialization of storage and search components (avoid circular imports)
        self._storage = None
        self._search_engine = None
        
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
        semantic_workers = getattr(self.config, 'semantic_generation_workers', 20)  # 提高默认线程数以匹配旧版本性能
        self._semantic_generation_executor = ThreadPoolExecutor(
            max_workers=semantic_workers,
            thread_name_prefix="semantic_gen"
        )
        self._semantic_generation_futures = {}  # Track async tasks
        self._semantic_futures_lock = threading.Lock()
        self.semantic_task_manager = SemanticTaskManager(self._semantic_generation_executor, max_retries=1)
        self.metrics_reporter = metrics_reporter or LoggingMetricsReporter()
        
        # 添加语义记忆缓存，避免重复加载
        self._semantic_cache_ttl = getattr(self.config, "semantic_cache_ttl", 600)
        self.semantic_memory_cache: PerUserCache[List[SemanticMemory]] = PerUserCache(
            ttl_seconds=self._semantic_cache_ttl
        )
        self.semantic_embedding_cache = SemanticEmbeddingCache()

        self._episode_cache_ttl = getattr(self.config, "episode_cache_ttl", 600)
        self.episode_cache: PerUserCache[List[Episode]] = PerUserCache(
            ttl_seconds=self._episode_cache_ttl
        )
        
        # Initialize batch segmenter and episode merger if enabled
        self._batch_segmenter = None
        self._episode_merger = None
        if self.config.enable_batch_segmentation:
            from ..generation.batch_segmenter import BatchSegmenter
            self._batch_segmenter = BatchSegmenter(self.llm_client, self.config)
            logger.info("Batch segmenter initialized")
        
        if self.config.enable_episode_merging:
            from ..generation.episode_merger import EpisodeMerger
            # Get the actual storage and search objects
            episode_storage = getattr(self._episode_repository, '_storage', None)
            vector_search_backend = getattr(self._vector_index, '_backend', None)
            if episode_storage and vector_search_backend:
                self._episode_merger = EpisodeMerger(
                    llm_client=self.llm_client,
                    embedding_client=self.embedding_client,
                    config=self.config,
                    episode_storage=episode_storage,
                    vector_search=vector_search_backend
                )
                logger.info("Episode merger initialized")
            else:
                logger.warning("Episode merger disabled: storage or vector search backend not available")

        self.event_bus = EventBus()
        self.event_bus.subscribe("episode_created", self._handle_episode_created_event)
        
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
            episode_backend = getattr(self._episode_repository, "_storage", self._episode_repository)
            semantic_backend = getattr(self._semantic_repository, "_storage", self._semantic_repository)
            self._storage = {
                "episode": episode_backend,
                "semantic": semantic_backend
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

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------
    def _handle_episode_created_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        if not self.config.enable_semantic_memory:
            return
        episode = payload.get("episode")
        owner_id = payload.get("user_id")
        if not episode or not owner_id:
            return
        try:
            self._schedule_semantic_generation(owner_id, episode)
        except Exception as exc:
            logger.error(f"Failed to schedule semantic generation for user {owner_id}: {exc}")
    
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
                episodes = self._episode_repository.list_by_user(owner_id)
                
                # Load semantic memories
                semantic_memories = []
                if self.config.enable_semantic_memory:
                    semantic_memories = self._semantic_repository.list_by_user(owner_id)
                
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
                                self.search_engine.chroma_search.load_user_indices,
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

    # ------------------------------------------------------------------
    # Deletion helpers
    # ------------------------------------------------------------------
    def delete_episode(self, owner_id: str, episode_id: str, cascade_semantic: bool = True) -> Dict[str, Any]:
        """Delete a single episode and optionally cascade to semantic memories."""
        user_lock = self._get_user_processing_lock(owner_id)
        with user_lock:
            response = {
                "status": "not_found",
                "deleted_semantic_ids": []
            }

            try:
                episodes = self._episode_repository.list_by_user(owner_id)
                target_episode = next((ep for ep in episodes if ep.episode_id == episode_id), None)
                if not target_episode:
                    return response

                if not self._episode_repository.delete(owner_id, episode_id):
                    return response

                response["status"] = "deleted"

                # Remove from indices
                self._lexical_index.remove_episode(owner_id, episode_id)
                self._vector_index.remove_episode(owner_id, episode_id)
                self.search_engine.remove_episode(owner_id, episode_id)

                # Cache invalidation
                self.episode_cache.invalidate(owner_id)
                self.performance_optimizer.cache.clear()

                # Cancel pending semantic generation tasks for the episode
                with self._semantic_futures_lock:
                    pending = [
                        task_id for task_id, ctx in self._semantic_generation_futures.items()
                        if ctx.get("owner_id") == owner_id and ctx.get("episode_id") == episode_id
                    ]
                    for task_id in pending:
                        future = self._semantic_generation_futures[task_id]["future"]
                        future.cancel()
                        del self._semantic_generation_futures[task_id]

                removed_semantics = []
                if cascade_semantic and self.config.enable_semantic_memory:
                    semantic_memories = self._semantic_repository.list_by_user(owner_id)
                    for memory in semantic_memories:
                        if episode_id in memory.source_episodes:
                            if self._semantic_repository.delete(owner_id, memory.memory_id):
                                removed_semantics.append(memory.memory_id)
                                self._lexical_index.remove_semantic(owner_id, memory.memory_id)
                                self._vector_index.remove_semantic(owner_id, memory.memory_id)
                                self.search_engine.remove_semantic_memory(owner_id, memory.memory_id)

                if removed_semantics:
                    response["deleted_semantic_ids"] = removed_semantics
                # Invalidate semantic caches regardless of cascade to avoid stale data
                self.semantic_memory_cache.invalidate(owner_id)
                self.semantic_embedding_cache.invalidate_user(owner_id)

                self.event_bus.publish(
                    "episode_deleted",
                    {
                        "user_id": owner_id,
                        "episode_id": episode_id,
                        "cascade_semantic": cascade_semantic,
                        "deleted_semantic_ids": removed_semantics,
                    }
                )

                return response

            except Exception as exc:
                logger.error(f"Error deleting episode {episode_id} for user {owner_id}: {exc}")
                response["status"] = "error"
                response["error"] = str(exc)
                return response

    def delete_semantic_memory(self, owner_id: str, memory_id: str) -> Dict[str, Any]:
        """Delete a single semantic memory for a user."""
        user_lock = self._get_user_processing_lock(owner_id)
        with user_lock:
            response = {
                "status": "not_found"
            }

            try:
                memories = self._semantic_repository.list_by_user(owner_id)
                target = next((mem for mem in memories if mem.memory_id == memory_id), None)
                if not target:
                    return response

                if not self._semantic_repository.delete(owner_id, memory_id):
                    return response

                response["status"] = "deleted"

                self._lexical_index.remove_semantic(owner_id, memory_id)
                self._vector_index.remove_semantic(owner_id, memory_id)
                self.search_engine.remove_semantic_memory(owner_id, memory_id)

                self.semantic_memory_cache.invalidate(owner_id)
                self.semantic_embedding_cache.invalidate_user(owner_id)
                self.performance_optimizer.cache.clear()

                self.event_bus.publish(
                    "semantic_memory_deleted",
                    {
                        "user_id": owner_id,
                        "memory_id": memory_id
                    }
                )

                return response

            except Exception as exc:
                logger.error(f"Error deleting semantic memory {memory_id} for user {owner_id}: {exc}")
                response["status"] = "error"
                response["error"] = str(exc)
                return response

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
                episodes = self._episode_repository.list_by_user(owner_id)
                
                # Load semantic memories
                semantic_memories = []
                if self.config.enable_semantic_memory:
                    semantic_memories = self._semantic_repository.list_by_user(owner_id)
                
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
                self.search_engine.chroma_search.load_user_indices(owner_id, episodes, semantic_memories)
            
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
            
            try:
                counts = self.search_engine.chroma_search.get_collection_counts(owner_id)
                if episodes and counts.get("episodes", 0) != len(episodes):
                    logger.info(
                        "Episode vector count mismatch for user %s: %s vs %s, forcing rebuild",
                        owner_id,
                        counts.get("episodes", 0),
                        len(episodes)
                    )
                    return True

                if (
                    self.config.enable_semantic_memory
                    and semantic_memories
                    and counts.get("semantic", 0) != len(semantic_memories)
                ):
                    logger.info(
                        "Semantic vector count mismatch for user %s: %s vs %s, forcing rebuild",
                        owner_id,
                        counts.get("semantic", 0),
                        len(semantic_memories)
                    )
                    return True

            except Exception as exc:
                logger.warning(
                    "Error checking Chroma indices for user %s: %s, forcing rebuild",
                    owner_id,
                    exc
                )
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
            self.search_engine.chroma_search.clear_user_index(owner_id)

            if episodes:
                self.search_engine.chroma_search.index_episodes(owner_id, episodes)

            if semantic_memories:
                self.search_engine.chroma_search.index_semantic_memories(owner_id, semantic_memories)

            logger.info(f"Successfully rebuilt vector indices for user {owner_id}")

        except Exception as e:
            logger.error(f"Error rebuilding vector indices for user {owner_id}: {e}")
            self.search_engine.chroma_search.load_user_indices(owner_id, episodes, semantic_memories)
    
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
                vector_search=self.search_engine.chroma_search  # 传递向量搜索引擎实例
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
                
                # Check if batch segmentation is enabled
                if self.config.enable_batch_segmentation and self._batch_segmenter:
                    # Add all messages to buffer first
                    buffer.add_messages(message_objects)
                    
                    # Check if batch threshold is reached
                    should_process, reason = self._batch_segmenter.should_create_episode(buffer.size())
                    
                    if should_process:
                        # Process batch with segmentation
                        created_episodes = self._process_batch_segmentation(owner_id, buffer, reason)
                        result["episodes_created"] = created_episodes
                        
                        # Publish events for semantic generation
                        semantic_tasks_scheduled = 0
                        for episode_info in created_episodes:
                            episode_obj = episode_info.get("episode_object")
                            if episode_obj is None:
                                continue
                            self.metrics_reporter.report(
                                "episode_created",
                                {
                                    "user_id": owner_id,
                                    "episode_id": episode_obj.episode_id,
                                    "message_count": episode_obj.message_count,
                                },
                            )
                            self.event_bus.publish(
                                "episode_created",
                                {"user_id": owner_id, "episode": episode_obj},
                            )
                            semantic_tasks_scheduled += 1
                        
                        if semantic_tasks_scheduled:
                            result["semantic_generation_scheduled"] = True
                            result["semantic_tasks_scheduled"] = semantic_tasks_scheduled
                            logger.info(
                                f"Dispatched {semantic_tasks_scheduled} episode_created events for user {owner_id}"
                            )
                        
                        # Clear buffer after processing
                        buffer.clear()
                else:
                    # Fallback: If batch segmentation is disabled, use simple buffer size trigger
                    logger.warning("Batch segmentation is disabled, using simple buffer size trigger")
                    buffer.add_messages(message_objects)
                    
                    # Check if buffer reaches max size
                    if buffer.size() >= self.config.buffer_size_max:
                        # Create single episode from all messages
                        episodes_to_create = [{
                            "buffer_messages": buffer.get_messages().copy(),
                            "boundary_reason": "Buffer reached maximum size"
                        }]
                        
                        created_episodes = self._batch_create_episodes(owner_id, episodes_to_create)
                        result["episodes_created"] = created_episodes
                        
                        # Publish events for semantic generation
                        semantic_tasks_scheduled = 0
                        for episode_info in created_episodes:
                            episode_obj = episode_info.get("episode_object")
                            if episode_obj is None:
                                continue
                            self.metrics_reporter.report(
                                "episode_created",
                                {
                                    "user_id": owner_id,
                                    "episode_id": episode_obj.episode_id,
                                    "message_count": episode_obj.message_count,
                                },
                            )
                            self.event_bus.publish(
                                "episode_created",
                                {"user_id": owner_id, "episode": episode_obj},
                            )
                            semantic_tasks_scheduled += 1

                        if semantic_tasks_scheduled:
                            result["semantic_generation_scheduled"] = True
                            result["semantic_tasks_scheduled"] = semantic_tasks_scheduled
                            logger.info(
                                f"Dispatched {semantic_tasks_scheduled} episode_created events for user {owner_id}"
                            )
                        
                        buffer.clear()
                
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
    
    def _process_batch_segmentation(
        self,
        owner_id: str,
        buffer: MessageBuffer,
        reason: str
    ) -> List[Dict[str, Any]]:
        """
        Process batch with intelligent segmentation and concurrent generation.
        
        Args:
            owner_id: User identifier
            buffer: Message buffer containing messages to segment
            reason: Reason for triggering batch processing
            
        Returns:
            List of created episodes information
        """
        if buffer.is_empty():
            return []
        
        try:
            # 1. Batch segmentation using LLM
            buffer_messages = buffer.get_messages()
            episode_groups = self._batch_segmenter.segment_batch(buffer_messages)
            
            logger.info(
                f"Batch segmentation: {len(buffer_messages)} messages → "
                f"{len(episode_groups)} episode groups"
            )
            
            # 2. Generate all episodes (可以并发)
            created_episodes = []
            
            for i, indices in enumerate(episode_groups):
                try:
                    # Extract messages by indices (1-based → 0-based)
                    group_messages = [buffer_messages[idx-1] for idx in indices]
                    
                    # Generate episode
                    episode = self.episode_generator.generate_episode(
                        user_id=owner_id,
                        messages=group_messages,
                        boundary_reason=f"Batch segmentation (group {i+1}/{len(episode_groups)})"
                    )
                    
                    # 3. Only check merge for the first episode
                    if i == 0 and self.config.enable_episode_merging and self._episode_merger:
                        try:
                            merged, merged_ep, old_id = self._episode_merger.check_and_merge(
                                new_episode=episode,
                                top_k=self.config.merge_top_k,
                                similarity_threshold=self.config.merge_similarity_threshold
                            )
                            
                            if merged and merged_ep and old_id:
                                # Use merged episode
                                episode = merged_ep
                                
                                # Delete old episode from storage and indices
                                self._delete_episode(owner_id, old_id)
                                
                                logger.info(
                                    f"Merged first episode in batch: {episode.episode_id} "
                                    f"(deleted old: {old_id[:8]}...)"
                                )
                        except Exception as e:
                            logger.warning(f"Episode merge failed for first episode: {e}, saving as new")
                    
                    # 4. Save episode
                    episode_id = self._episode_repository.save(episode)
                    
                    # 5. Index episode
                    self._lexical_index.add_episode(owner_id, episode)
                    self._vector_index.add_episode(owner_id, episode)
                    self.search_engine.add_episode(owner_id, episode)
                    
                    # Update episodes cache
                    cached = self.episode_cache.get(owner_id)
                    if cached is not None:
                        cached = cached + [episode]
                        self.episode_cache.put(owner_id, cached)
                    
                    # Add to results
                    created_episodes.append({
                        "episode_id": episode_id,
                        "title": episode.title,
                        "message_count": episode.message_count,
                        "episode_object": episode
                    })
                    
                    logger.info(
                        f"Created episode {episode_id} from batch group {i+1}/{len(episode_groups)} "
                        f"({len(group_messages)} messages)"
                    )
                    
                except Exception as e:
                    logger.error(f"Error processing episode group {i+1}: {e}")
                    continue
            
            return created_episodes
            
        except Exception as e:
            logger.error(f"Error in batch segmentation processing: {e}")
            return []
    
    def _delete_episode(self, owner_id: str, episode_id: str) -> bool:
        """
        Delete an episode from storage and all indices.
        
        Args:
            owner_id: User identifier
            episode_id: Episode ID to delete
            
        Returns:
            True if successful
        """
        try:
            # Delete from storage
            episode_storage = getattr(self._episode_repository, '_storage', None)
            if episode_storage:
                episode_storage.delete_episode(episode_id, owner_id)
            
            # Delete from vector index
            vector_search = getattr(self._vector_index, '_backend', None)
            if vector_search:
                vector_search.remove_episode(owner_id, episode_id)
            
            # Delete from lexical index (if supported)
            # BM25Search doesn't have explicit delete, it rebuilds from storage
            
            # Invalidate cache
            self.episode_cache.invalidate(owner_id)
            
            logger.info(f"Deleted episode {episode_id} for user {owner_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting episode {episode_id}: {e}")
            return False
    
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
            # Use default values from configuration only if None is passed
            if top_k_episodes is None:
                top_k_episodes = self.config.search_top_k_episodes
            if top_k_semantic is None:
                top_k_semantic = self.config.search_top_k_semantic
            
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
            self.metrics_reporter.report(
                "search",
                {
                    "user_id": user_id,
                    "query": query,
                    "episodes": len(episode_results),
                    "semantic": len(semantic_results),
                    "duration": search_time,
                },
            )
            
            return {
                "episodic": episode_results,
                "semantic": semantic_results
            }
            
        except Exception as e:
            logger.error(f"Error in search_all for user {user_id}: {e}")
            return {"episodic": [], "semantic": []}
    
    def _search_episodes_by_method(self, user_id: str, query: str, top_k: int, search_method: str):
        """Search episodes by search method"""
        # Handle top_k=0 case: return empty list
        if top_k == 0:
            return []
        
        if search_method == "hybrid":
            return self.search_engine.search_episodes(user_id, query, top_k)
        elif search_method == "bm25":
            return self.search_engine.bm25_search.search_episodes(user_id, query, top_k)
        elif search_method == "vector":
            return self.search_engine.chroma_search.search_episodes(user_id, query, top_k)
        elif search_method == "vector_norlift":
            return self.search_engine.search_episodes(user_id, query, top_k, "vector_norlift")
        else:
            raise ValueError(f"Unknown search method: {search_method}")
    
    def _search_semantic_by_method(self, user_id: str, query: str, top_k: int, search_method: str):
        """Search semantic memories by search method"""
        # Handle top_k=0 case: return empty list
        if top_k == 0:
            return []
        
        if search_method == "hybrid":
            return self.search_engine.search_semantic_memories(user_id, query, top_k)
        elif search_method == "bm25":
            return self.search_engine.bm25_search.search_semantic_memories(user_id, query, top_k)
        elif search_method == "vector":
            return self.search_engine.chroma_search.search_semantic_memories(user_id, query, top_k)
        elif search_method == "vector_norlift":
            # NOR-LIFT 仅用于情景级排序，这里回退为向量检索
            return self.search_engine.chroma_search.search_semantic_memories(user_id, query, top_k)
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
                    episode_obj = episode_info["episode_object"]
                    self.metrics_reporter.report(
                        "episode_created",
                        {
                            "user_id": owner_id,
                            "episode_id": episode_obj.episode_id,
                            "message_count": episode_obj.message_count,
                        },
                    )
                    self.event_bus.publish(
                        "episode_created",
                        {"user_id": owner_id, "episode": episode_obj},
                    )
                    episode_info["semantic_generation_scheduled"] = True
                    logger.info(f"Published episode_created event for user {owner_id} (force creation)")
                
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
            episode_id = self._episode_repository.save(episode)
            
            # Index episode (for search, using incremental update)
            self._lexical_index.add_episode(owner_id, episode)
            self._vector_index.add_episode(owner_id, episode)
            self.search_engine.add_episode(owner_id, episode)
            
            # 更新episodes缓存
            cached = self.episode_cache.get(owner_id)
            if cached is not None:
                cached = cached + [episode]
                self.episode_cache.put(owner_id, cached)
                logger.debug(f"Updated episodes cache for user {owner_id}")
            
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
        future = self.semantic_task_manager.submit(
            self._async_generate_semantic_memories,
            owner_id,
            new_episode,
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
            
            # Get all user episodes (使用缓存避免重复磁盘I/O)
            cached_episodes = self.episode_cache.get(owner_id)
            if cached_episodes is not None:
                all_episodes = cached_episodes
                logger.debug(f"Episodes cache hit for user {owner_id}")
            else:
                all_episodes = self._episode_repository.list_by_user(owner_id)
                self.episode_cache.put(owner_id, all_episodes)
                logger.debug(f"Episodes cache miss for user {owner_id}, loaded {len(all_episodes)} episodes from storage")
            
            # Exclude newly created episode
            existing_episodes = [ep for ep in all_episodes if ep.episode_id != new_episode.episode_id]
            
            # If prediction-correction mode is enabled, get existing semantic memories
            existing_semantic_memories = None
            if self.config.enable_prediction_correction:
                # 使用缓存的语义记忆
                cache_key = f"semantic_{owner_id}"
                cache_hit = False
                
                # 使用用户级别的锁，不同用户可以并行
                semantic_lock = self.semantic_embedding_cache.get_user_lock(owner_id)
                cached_semantics = self.semantic_memory_cache.get(owner_id)
                if cached_semantics is not None:
                    existing_semantic_memories = cached_semantics
                    cache_hit = True
                    logger.debug(f"Cache hit: Loaded {len(existing_semantic_memories)} semantic memories from cache")
                
                if not cache_hit:
                    # 缓存未命中，从存储加载
                    existing_semantic_memories = self._semantic_repository.list_by_user(owner_id)
                    logger.debug(f"Cache miss: Loaded {len(existing_semantic_memories)} semantic memories from storage")
                    
                    # 更新缓存
                    self.semantic_memory_cache.put(owner_id, existing_semantic_memories)
            
            if existing_semantic_memories is None:
                existing_semantic_memories = []

            # 构建语义向量缓存，避免重复读取与计算
            existing_embeddings: Dict[str, List[float]] = self.semantic_embedding_cache.list_user_embeddings(owner_id)

            if existing_semantic_memories:
                # 尝试从缓存或Chroma中批量获取已有语义的向量
                missing_ids = {mem.memory_id for mem in existing_semantic_memories if mem.memory_id not in existing_embeddings}

                if missing_ids:
                    chroma_embeddings = self.search_engine.chroma_search.get_semantic_embeddings(owner_id, list(missing_ids))
                    if chroma_embeddings:
                        for mem_id, embedding in chroma_embeddings.items():
                            if embedding is None:
                                continue
                            embedding_list = list(embedding)
                            self.semantic_embedding_cache.set(owner_id, mem_id, embedding_list)
                            existing_embeddings[mem_id] = embedding_list
                            missing_ids.discard(mem_id)

                if missing_ids:
                    # Fallback: 对仍缺失的语义记忆重新计算向量，一次性批量完成
                    remaining_pairs = [(mem.memory_id, mem.content) for mem in existing_semantic_memories if mem.memory_id in missing_ids]
                    if remaining_pairs:
                        contents = [content for _, content in remaining_pairs]
                        embed_resp = self.embedding_client.embed_texts(contents)
                        for (mem_id, _), embedding in zip(remaining_pairs, embed_resp.embeddings):
                            self.semantic_embedding_cache.set(owner_id, mem_id, embedding)
                            existing_embeddings[mem_id] = embedding
                            missing_ids.discard(mem_id)

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
                # 预先计算新语义的向量，避免重复嵌入
                new_embedding = self.embedding_client.embed_text(memory.content)
                
                # Check for duplicates
                known_embedding_keys = set(existing_embeddings.keys())
                if self._is_duplicate_semantic_memory(
                    owner_id,
                    memory,
                    existing_memories=existing_semantic_memories,
                    embedding_cache=existing_embeddings,
                    new_embedding=new_embedding
                ):
                    continue

                memory_id = self._semantic_repository.save(memory)
                
                # Exception handling when adding to search index
                try:
                    self.search_engine.add_semantic_memory(owner_id, memory, embedding=new_embedding)
                except Exception as e:
                    # 捕获所有异常，不让单个索引失败影响整个语义记忆生成
                    logger.warning(f"Failed to add semantic memory to search index (continuing): {e}")
                    # 不重新抛出异常，让任务继续完成
                
                saved_memories.append({
                    "memory_id": memory_id,
                    "knowledge_type": memory.knowledge_type,
                    "content": memory.content
                })

                # 更新缓存中的语义记忆与向量
                self.semantic_memory_cache.put(owner_id, existing_semantic_memories + [memory])
                self.semantic_embedding_cache.set(owner_id, memory.memory_id, new_embedding)
                
                # 保证后续语义去重能感知到新记忆
                existing_semantic_memories.append(memory)
                existing_embeddings[memory.memory_id] = new_embedding
            
            generation_time = time.time() - start_time
            
            # Update statistics
            with self.stats_lock:
                self.stats["semantic_memories_created"] += len(saved_memories)
            
            logger.info(f"Async semantic generation completed for user {owner_id}: "
                       f"{len(saved_memories)} memories generated in {generation_time:.2f}s")
            self.metrics_reporter.report(
                "semantic_generation",
                {
                    "user_id": owner_id,
                    "episode_id": new_episode.episode_id,
                    "generated": len(saved_memories),
                    "duration": generation_time,
                },
            )
            
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
            logger.info(f"Semantic generation task {task_id} completed successfully with {len(result)} memories")
        except Exception as e:
            logger.error(f"Semantic generation task {task_id} failed: {e}")
            # 失败的任务也算作"完成"，避免无限等待
        finally:
            # 无论成功或失败，都清理任务记录
            with self._semantic_futures_lock:
                if task_id in self._semantic_generation_futures:
                    task_info = self._semantic_generation_futures[task_id]
                    execution_time = time.time() - task_info["started_at"]
                    logger.debug(f"Cleaning up semantic generation task {task_id} after {execution_time:.2f}s")
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
            
            # 只统计真正活跃的任务（运行中或等待中，不包括已完成或失败的）
            active_count = 0
            for task in all_tasks:
                if task["status"] in ["running", "pending"]:
                    active_count += 1
            
            return {
                "active_tasks": active_count,  # 只包含真正活跃的任务
                "total_tasks": len(all_tasks),
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
        last_log_time = start_time
        
        while time.time() - start_time < timeout:
            # Get status for the specific user
            status = self.get_semantic_generation_status(owner_id)
            active_tasks = status.get("active_tasks", 0)
            
            # If there are no active tasks, we're done
            if active_tasks == 0:
                logger.debug(f"All semantic generation tasks completed for user {owner_id}")
                return True
            
            # 每5秒打印一次进度日志
            current_time = time.time()
            if current_time - last_log_time > 5.0:
                tasks = status.get("tasks", [])
                running_tasks = [t for t in tasks if t["status"] == "running"]
                failed_tasks = [t for t in tasks if t["status"] == "failed"]
                
                logger.debug(f"Waiting for user {owner_id}: {active_tasks} active, {len(running_tasks)} running, {len(failed_tasks)} failed")
                last_log_time = current_time
            
            # 短暂等待
            time.sleep(0.5)
        
        # 超时处理：强制清理挂起的任务
        logger.warning(f"Timeout waiting for semantic generation tasks for user {owner_id}, forcing cleanup...")
        
        # 获取并清理超时的任务
        with self._semantic_futures_lock:
            timeout_tasks = []
            for task_id, task_info in list(self._semantic_generation_futures.items()):
                if task_info["owner_id"] == owner_id:
                    future = task_info["future"]
                    if not future.done():
                        # 尝试取消超时的任务
                        try:
                            future.cancel()
                            logger.warning(f"Cancelled timeout task {task_id}")
                        except Exception:
                            pass
                    timeout_tasks.append(task_id)
            
            # 清理超时任务记录
            for task_id in timeout_tasks:
                if task_id in self._semantic_generation_futures:
                    del self._semantic_generation_futures[task_id]
                    logger.debug(f"Cleaned up timeout task {task_id}")
        
        return False
    
    def force_complete_semantic_generation(self, owner_id: str = None) -> Dict[str, Any]:
        """
        强制完成挂起的语义记忆生成任务
        
        Args:
            owner_id: 用户标识符，如果提供则只处理该用户的任务
            
        Returns:
            完成状态信息
        """
        logger.info(f"强制完成语义记忆生成任务，用户: {owner_id or 'all'}")
        
        completed_tasks = []
        cancelled_tasks = []
        
        with self._semantic_futures_lock:
            tasks_to_process = []
            
            for task_id, task_info in list(self._semantic_generation_futures.items()):
                if owner_id and task_info["owner_id"] != owner_id:
                    continue
                
                tasks_to_process.append((task_id, task_info))
            
            for task_id, task_info in tasks_to_process:
                future = task_info["future"]
                
                if future.done():
                    # 任务已完成，清理记录
                    try:
                        result = future.result(timeout=0)
                        completed_tasks.append({
                            "task_id": task_id,
                            "user_id": task_info["owner_id"],
                            "status": "completed",
                            "result_count": len(result) if isinstance(result, list) else 0
                        })
                    except Exception as e:
                        completed_tasks.append({
                            "task_id": task_id,
                            "user_id": task_info["owner_id"],
                            "status": "failed",
                            "error": str(e)
                        })
                    
                    del self._semantic_generation_futures[task_id]
                
                else:
                    # 任务未完成，尝试取消
                    try:
                        cancelled = future.cancel()
                        cancelled_tasks.append({
                            "task_id": task_id,
                            "user_id": task_info["owner_id"],
                            "cancelled": cancelled,
                            "running_time": time.time() - task_info["started_at"]
                        })
                        
                        del self._semantic_generation_futures[task_id]
                        
                    except Exception as e:
                        logger.warning(f"Failed to cancel task {task_id}: {e}")
        
        result = {
            "completed_tasks": completed_tasks,
            "cancelled_tasks": cancelled_tasks,
            "total_processed": len(completed_tasks) + len(cancelled_tasks)
        }
        
        logger.info(f"强制完成结果: {len(completed_tasks)} 已完成, {len(cancelled_tasks)} 已取消")
        
        return result
    
    def _is_duplicate_semantic_memory(
        self,
        owner_id: str,
        memory: SemanticMemory,
        existing_memories: Optional[List[SemanticMemory]] = None,
        embedding_cache: Optional[Dict[str, List[float]]] = None,
        new_embedding: Optional[List[float]] = None
    ) -> bool:
        """
        Check if semantic memory is duplicate (using existing embedding vector, avoiding duplicate API calls)
        
        Args:
            owner_id: User identifier
            memory: Semantic memory
            existing_memories: 已加载的语义记忆集合，若提供则不再重复读取磁盘
            embedding_cache: 语义向量缓存，避免重复嵌入
            new_embedding: 新语义的嵌入向量，避免重复计算
            
        Returns:
            Whether the semantic memory is duplicate
        """
        try:
            # ChromaDB不暴露内部embeddings，直接使用fallback方法
            return self._is_duplicate_semantic_memory_fallback(
                owner_id,
                memory,
                existing_memories=existing_memories,
                embedding_cache=embedding_cache,
                new_embedding=new_embedding
            )
        
        except Exception as e:
            logger.error(f"Error checking semantic memory duplication: {e}")
            # Fall back to traditional method (for compatibility)
            return self._is_duplicate_semantic_memory_fallback(
                owner_id,
                memory,
                existing_memories=existing_memories,
                embedding_cache=embedding_cache,
                new_embedding=new_embedding
            )
    
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
    
    def _is_duplicate_semantic_memory_fallback(
        self,
        owner_id: str,
        memory: SemanticMemory,
        existing_memories: Optional[List[SemanticMemory]] = None,
        embedding_cache: Optional[Dict[str, List[float]]] = None,
        new_embedding: Optional[List[float]] = None
    ) -> bool:
        """
        Fallback method for duplicate detection (for compatibility)
        
        Args:
            owner_id: User identifier
            memory: 语义记忆
            existing_memories: 可复用的语义记忆列表
            embedding_cache: 已缓存的语义向量
            new_embedding: 新语义向量
            
        Returns:
            Whether the semantic memory is duplicate
        """
        try:
            # Use vector similarity to check for duplicates
            if existing_memories is None:
                existing_memories = self._semantic_repository.list_by_user(owner_id)
            
            if not existing_memories:
                return False
            
            embeddings_lookup = embedding_cache if embedding_cache is not None else {}

            # Calculate similarity with existing memories (using cache)
            if new_embedding is not None:
                memory_embedding = new_embedding
            else:
                memory_embedding = self.performance_optimizer.cached_call(
                    self.embedding_client.embed_text,
                    f"embed_{hash(memory.content)}",
                    memory.content
                )
            
            for existing in existing_memories:
                if existing.knowledge_type == memory.knowledge_type:
                    existing_embedding = embeddings_lookup.get(existing.memory_id)
                    if existing_embedding is None:
                        existing_embedding = self.performance_optimizer.cached_call(
                            self.embedding_client.embed_text,
                            f"embed_{hash(existing.content)}",
                            existing.content
                        )
                        if embedding_cache is not None:
                            embeddings_lookup[existing.memory_id] = existing_embedding
                    
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
            
            if hasattr(self.search_engine, 'chroma_search'):
                # ChromaDB cache clearing is handled by clear_user_index
                try:
                    self.search_engine.chroma_search.clear_user_index(owner_id)
                except Exception as e:
                    logger.warning(f"Error clearing ChromaDB cache for {owner_id}: {e}")
            
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
