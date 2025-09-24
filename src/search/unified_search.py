"""
Unified Search Engine
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from ..models import Episode, SemanticMemory
from ..utils import EmbeddingClient
from ..config import MemoryConfig
from .bm25_search import BM25Search
from .chroma_search import ChromaSearchEngine

logger = logging.getLogger(__name__)


class UnifiedSearchEngine:
    """Unified search engine combining BM25 and vector search with incremental updates"""
    
    def __init__(self, embedding_client: EmbeddingClient, config: MemoryConfig, language: str = "en"):
        """
        Initialize unified search engine
        
        Args:
            embedding_client: Client for generating embeddings
            config: Memory system configuration
            language: Language for BM25 tokenization ("en" for English, "zh" for Chinese)
        """
        self.config = config
        self.language = language
        # Save embedding client for advanced ranking (e.g., NOR-LIFT)
        self.embedding_client = embedding_client
        
        # Initialize search engines
        self.bm25_search = BM25Search(language=language)
        self.chroma_search = ChromaSearchEngine(embedding_client, config)
        
        # Thread pool for parallel search
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        logger.info(f"Unified search engine initialized with {language} tokenization")
    
    def index_episodes(self, user_id: str, episodes: List[Episode]):
        """
        Index episodes in both search engines
        
        Args:
            user_id: User ID
            episodes: List of episodes to index
        """
        try:
            # Index in both engines in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                bm25_future = executor.submit(self.bm25_search.index_episodes, user_id, episodes)
                chroma_future = executor.submit(self.chroma_search.index_episodes, user_id, episodes)
                
                # Wait for both to complete
                bm25_future.result()
                chroma_future.result()
            
            logger.info(f"Indexed {len(episodes)} episodes in unified search for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error indexing episodes in unified search: {e}")
    
    def index_semantic_memories(self, user_id: str, memories: List[SemanticMemory]):
        """
        Index semantic memories in both search engines
        
        Args:
            user_id: User ID
            memories: List of semantic memories to index
        """
        try:
            # Index in both engines in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                bm25_future = executor.submit(self.bm25_search.index_semantic_memories, user_id, memories)
                chroma_future = executor.submit(self.chroma_search.index_semantic_memories, user_id, memories)
                
                # Wait for both to complete
                bm25_future.result()
                chroma_future.result()
            
            logger.info(f"Indexed {len(memories)} semantic memories in unified search for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error indexing semantic memories in unified search: {e}")
    
    def search_episodes(
        self, 
        user_id: str, 
        query: str, 
        top_k: int = 10,
        search_method: str = "hybrid"
    ) -> List[Dict[str, Any]]:
        """
        Search episodes using unified approach
        
        Args:
            user_id: User ID
            query: Search query
            top_k: Number of results to return
            search_method: "bm25", "vector", or "hybrid"
            
        Returns:
            List of search results
        """
        try:
            if search_method == "bm25":
                return self.bm25_search.search_episodes(user_id, query, top_k)
            elif search_method == "vector":
                # 可选启用 NOR-LIFT 排序
                if getattr(self.config, 'enable_norlift_ranking', False):
                    return self.search_episodes_norlift(user_id, query, top_k)
                return self.chroma_search.search_episodes(user_id, query, top_k)
            elif search_method == "vector_norlift":
                return self.search_episodes_norlift(user_id, query, top_k)
            elif search_method == "hybrid":
                return self._hybrid_search_episodes(user_id, query, top_k)
            else:
                logger.warning(f"Unknown search method: {search_method}, using hybrid")
                return self._hybrid_search_episodes(user_id, query, top_k)
                
        except Exception as e:
            logger.error(f"Error in unified episode search: {e}")
            return []
    
    def search_semantic_memories(
        self, 
        user_id: str, 
        query: str, 
        top_k: int = 10,
        search_method: str = "hybrid"
    ) -> List[Dict[str, Any]]:
        """
        Search semantic memories using unified approach
        
        Args:
            user_id: User ID
            query: Search query
            top_k: Number of results to return
            search_method: "bm25", "vector", or "hybrid"
            
        Returns:
            List of search results
        """
        try:
            if search_method == "bm25":
                return self.bm25_search.search_semantic_memories(user_id, query, top_k)
            elif search_method == "vector":
                return self.chroma_search.search_semantic_memories(user_id, query, top_k)
            elif search_method == "hybrid":
                return self._hybrid_search_semantic_memories(user_id, query, top_k)
            else:
                logger.warning(f"Unknown search method: {search_method}, using hybrid")
                return self._hybrid_search_semantic_memories(user_id, query, top_k)
                
        except Exception as e:
            logger.error(f"Error in unified semantic memory search: {e}")
            return []
    
    def search(
        self,
        owner_id: str,
        query: str,
        top_k: int = 10,
        memory_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        General search method for searching specified types of memories
        
        Args:
            owner_id: User identifier
            query: Search query
            top_k: Number of results to return
            memory_types: Types of memories to search ["episodic", "semantic"]
            
        Returns:
            List of search results
        """
        try:
            memory_types = memory_types or ["episodic", "semantic"]
            all_results = []
            
            # Search according to specified types
            if "episodic" in memory_types:
                episode_results = self.search_episodes(owner_id, query, top_k, "hybrid")
                all_results.extend(episode_results)
            
            if "semantic" in memory_types:
                semantic_results = self.search_semantic_memories(owner_id, query, top_k, "hybrid")
                all_results.extend(semantic_results)
            
            # Sort and return top_k results
            all_results.sort(
                key=lambda x: x.get("fused_score", x.get("score", 0)),
                reverse=True
            )
            
            return all_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in unified search: {e}")
            return []
    
    def search_all(
        self, 
        user_id: str, 
        query: str, 
        top_k_episodes: int = 5,
        top_k_semantic: int = 5,
        search_method: str = "hybrid"
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search both episodes and semantic memories
        
        Args:
            user_id: User ID
            query: Search query
            top_k_episodes: Number of episode results
            top_k_semantic: Number of semantic results
            search_method: "bm25", "vector", or "hybrid"
            
        Returns:
            Dictionary with separate episode and semantic results
        """
        try:
            # Perform searches in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                episode_future = executor.submit(
                    self.search_episodes, 
                    user_id, query, top_k_episodes, search_method
                )
                semantic_future = executor.submit(
                    self.search_semantic_memories, 
                    user_id, query, top_k_semantic, search_method
                )
                
                episode_results = episode_future.result()
                semantic_results = semantic_future.result()
            
            return {
                "episodic": episode_results,
                "semantic": semantic_results,
                "combined": self._combine_and_rank_results(episode_results, semantic_results)
            }
            
        except Exception as e:
            logger.error(f"Error in unified search all: {e}")
            return {"episodic": [], "semantic": [], "combined": []}
    
    def _hybrid_search_episodes(self, user_id: str, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Hybrid search for episodes combining BM25 and vector search
        
        Args:
            user_id: User ID
            query: Search query
            top_k: Number of results
            
        Returns:
            Hybrid search results
        """
        try:
            # Perform searches in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                bm25_future = executor.submit(
                    self.bm25_search.search_episodes, 
                    user_id, query, top_k * 2  # Get more results for fusion
                )
                vector_future = executor.submit(
                    self.chroma_search.search_episodes, 
                    user_id, query, top_k * 2
                )
                
                bm25_results = bm25_future.result()
                vector_results = vector_future.result()
            
            # Fuse results using Reciprocal Rank Fusion
            fused_results = self._reciprocal_rank_fusion(
                bm25_results, vector_results, top_k
            )
            
            return fused_results
            
        except Exception as e:
            logger.error(f"Error in hybrid episode search: {e}")
            # Fallback to BM25 search
            return self.bm25_search.search_episodes(user_id, query, top_k)
    
    def _hybrid_search_semantic_memories(self, user_id: str, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Hybrid search for semantic memories combining BM25 and vector search
        
        Args:
            user_id: User ID
            query: Search query
            top_k: Number of results
            
        Returns:
            Hybrid search results
        """
        try:
            # Perform searches in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                bm25_future = executor.submit(
                    self.bm25_search.search_semantic_memories, 
                    user_id, query, top_k * 2  # Get more results for fusion
                )
                vector_future = executor.submit(
                    self.chroma_search.search_semantic_memories, 
                    user_id, query, top_k * 2
                )
                
                bm25_results = bm25_future.result()
                vector_results = vector_future.result()
            
            # Fuse results using Reciprocal Rank Fusion
            fused_results = self._reciprocal_rank_fusion(
                bm25_results, vector_results, top_k
            )
            
            return fused_results
            
        except Exception as e:
            logger.error(f"Error in hybrid semantic memory search: {e}")
            # Fallback to BM25 search
            return self.bm25_search.search_semantic_memories(user_id, query, top_k)
    
    def _reciprocal_rank_fusion(
        self, 
        results1: List[Dict[str, Any]], 
        results2: List[Dict[str, Any]], 
        top_k: int,
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Combine search results using Reciprocal Rank Fusion
        
        Args:
            results1: First set of results (e.g., BM25)
            results2: Second set of results (e.g., vector)
            top_k: Number of final results
            k: RRF parameter
            
        Returns:
            Fused and ranked results
        """
        # Create a mapping from item ID to result
        item_scores = {}
        item_data = {}
        
        # Process first result set
        for rank, result in enumerate(results1):
            item_id = result.get("episode_id") or result.get("memory_id")
            if item_id:
                rrf_score = 1.0 / (k + rank + 1)
                item_scores[item_id] = item_scores.get(item_id, 0) + rrf_score
                item_data[item_id] = result
        
        # Process second result set
        for rank, result in enumerate(results2):
            item_id = result.get("episode_id") or result.get("memory_id")
            if item_id:
                rrf_score = 1.0 / (k + rank + 1)
                item_scores[item_id] = item_scores.get(item_id, 0) + rrf_score
                
                # Update result data with hybrid information
                if item_id not in item_data:
                    item_data[item_id] = result
                else:
                    # Mark as hybrid result
                    item_data[item_id]["search_method"] = "hybrid"
        
        # Sort by fused score and return top k
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        fused_results = []
        for item_id, fused_score in sorted_items:
            result = item_data[item_id].copy()
            result["fused_score"] = fused_score
            result["search_method"] = "hybrid"
            fused_results.append(result)
        
        return fused_results
    
    def _combine_and_rank_results(
        self, 
        episode_results: List[Dict[str, Any]], 
        semantic_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Combine and rank episode and semantic results
        
        Args:
            episode_results: Episode search results
            semantic_results: Semantic memory search results
            
        Returns:
            Combined and ranked results
        """
        all_results = episode_results + semantic_results
        
        # Sort by score (fused_score if available, otherwise original score)
        all_results.sort(
            key=lambda x: x.get("fused_score", x.get("score", 0)), 
            reverse=True
        )
        
        return all_results

    # ================= NOR-LIFT Aggregation Ranking =================
    def search_episodes_norlift(
        self,
        user_id: str,
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        NOR-LIFT: 使用情景向量与其语义记忆向量的 Noisy-OR + LIFT 聚合对情景进行排序。
        - 候选集：取情景向量 Top-Pe 与语义向量 Top-Ps 的并集所覆盖的情景
        - 评分：对候选集所有分数做 z-score 标准化→sigmoid 映射→Noisy-OR 聚合→LIFT 修正
        """
        try:
            # NOR-LIFT ranking not supported with ChromaDB, fallback to normal search
            logger.warning("NOR-LIFT ranking not supported with ChromaDB, using normal ChromaDB search")
            return self.chroma_search.search_episodes(user_id, query, top_k)

        except Exception as e:
            logger.error(f"Error in NOR-LIFT episode ranking: {e}")
            # 失败回退到普通向量检索
            return self.chroma_search.search_episodes(user_id, query, top_k)
    
    def add_episode(self, user_id: str, episode: Episode):
        """
        Add episode to both search engines using INCREMENTAL UPDATE
        
        Args:
            user_id: User ID
            episode: Episode to add
        """
        try:
            # Add to both engines in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                bm25_future = executor.submit(self.bm25_search.add_episode, user_id, episode)
                chroma_future = executor.submit(self.chroma_search.add_episode, user_id, episode)
                
                bm25_future.result()
                chroma_future.result()
            
            logger.debug(f"🚀 Incrementally added episode to unified search for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error adding episode to unified search: {e}")
    
    def add_semantic_memory(self, user_id: str, memory: SemanticMemory, embedding: Optional[List[float]] = None):
        """
        Add semantic memory to both search engines using INCREMENTAL UPDATE
        
        Args:
            user_id: User ID
            memory: Semantic memory to add
        """
        try:
            # Check if executor is still available
            if hasattr(self, 'executor') and self.executor._shutdown:
                # Executor is shutting down, use synchronous calls
                logger.debug("Executor is shutting down, using synchronous updates")
                self.bm25_search.add_semantic_memory(user_id, memory)
                self.chroma_search.add_semantic_memory(user_id, memory, embedding=embedding)
            else:
                # Add to both engines in parallel
                with ThreadPoolExecutor(max_workers=2) as executor:
                    bm25_future = executor.submit(self.bm25_search.add_semantic_memory, user_id, memory)
                    chroma_future = executor.submit(self.chroma_search.add_semantic_memory, user_id, memory, embedding)
                    
                    bm25_future.result()
                    chroma_future.result()
            
            logger.debug(f"🚀 Incrementally added semantic memory to unified search for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error adding semantic memory to unified search: {e}")
    
    def clear_user_index(self, user_id: str) -> bool:
        """
        Clear all indices for a user from both search engines
        
        Args:
            user_id: User ID
            
        Returns:
            True if cleared successfully from both engines
        """
        try:
            bm25_success = self.bm25_search.clear_user_index(user_id)
            chroma_success = self.chroma_search.clear_user_index(user_id)
            
            return bm25_success and chroma_success
            
        except Exception as e:
            logger.error(f"Error clearing unified search indices for user {user_id}: {e}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive index statistics from both search engines
        
        Returns:
            Combined index statistics
        """
        try:
            bm25_stats = self.bm25_search.get_index_stats()
            chroma_stats = self.chroma_search.get_stats()
            
            # Combine stats
            combined_stats = {
                "search_engines": {
                    "bm25": bm25_stats,
                    "chroma": chroma_stats
                },
                # Overall stats
                "total_episodes": max(
                    bm25_stats.get("total_episodes", 0),
                    len([c for c in chroma_stats.get("collections", []) if c.get("name", "").endswith("_episodes")])
                ),
                "total_semantic_memories": max(
                    bm25_stats.get("total_semantic_memories", 0),
                    len([c for c in chroma_stats.get("collections", []) if c.get("name", "").endswith("_semantic")])
                ),
                "episode_users": max(
                    bm25_stats.get("episode_users", 0),
                    len([c for c in chroma_stats.get("collections", []) if c.get("name", "").endswith("_episodes")])
                ),
                "semantic_users": max(
                    bm25_stats.get("semantic_users", 0),
                    len([c for c in chroma_stats.get("collections", []) if c.get("name", "").endswith("_semantic")])
                ),
                # Features
                "incremental_updates": True,  # ChromaDB supports incremental updates
                "optimized_updates": bm25_stats.get("optimized_updates", False),
                "chroma_available": True,  # ChromaDB is available
                "spacy_available": bm25_stats.get("spacy_available", False),
                "language": self.language,
                "unified_engine": True
            }
            
            return combined_stats
            
        except Exception as e:
            logger.error(f"Error getting unified search index stats: {e}")
            return {
                "error": str(e),
                "unified_engine": True,
                "total_episodes": 0,
                "total_semantic_memories": 0
            }
    
    def __del__(self):
        """Cleanup resources"""
        try:
            self.executor.shutdown(wait=True)
        except:
            pass 
