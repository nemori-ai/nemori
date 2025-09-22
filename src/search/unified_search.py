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
from .vector_search import VectorSearch

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
        self.vector_search = VectorSearch(embedding_client, config.storage_path, config.embedding_dimension)
        
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
                vector_future = executor.submit(self.vector_search.index_episodes, user_id, episodes)
                
                # Wait for both to complete
                bm25_future.result()
                vector_future.result()
            
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
                vector_future = executor.submit(self.vector_search.index_semantic_memories, user_id, memories)
                
                # Wait for both to complete
                bm25_future.result()
                vector_future.result()
            
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
                return self.vector_search.search_episodes(user_id, query, top_k)
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
                return self.vector_search.search_semantic_memories(user_id, query, top_k)
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
                    self.vector_search.search_episodes, 
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
                    self.vector_search.search_semantic_memories, 
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
            # 前置检查：索引可用
            if user_id not in self.vector_search.episode_data:
                logger.debug(f"No episode vector data for user {user_id}")
                return []

            # 参数
            pe = int(getattr(self.config, 'norlift_pool_size_episodes', 100))
            ps = int(getattr(self.config, 'norlift_pool_size_semantic', 200))
            tau_percentile = float(getattr(self.config, 'norlift_percentile_tau', 0.95))
            lam = float(getattr(self.config, 'norlift_sigmoid_lambda', 2.5))
            eps = float(getattr(self.config, 'norlift_epsilon', 1e-6))

            # 计算查询向量
            q_vec = np.array(self.embedding_client.embed_text(query), dtype=np.float32)
            if q_vec.size == 0:
                return []

            # 获取情景候选（Top-Pe）
            epi_index = self.vector_search.episode_indices.get(user_id)
            episodes = self.vector_search.episode_data.get(user_id, [])
            if not episodes:
                return []
            pe = min(pe, len(episodes))

            epi_scores, epi_indices = self.vector_search._search_faiss_index(
                epi_index, q_vec, pe
            )

            episode_id_to_score: Dict[str, float] = {}
            episode_id_to_obj: Dict[str, Episode] = {}
            for score, idx in zip(epi_scores, epi_indices):
                if idx is None or idx < 0 or idx >= len(episodes):
                    continue
                ep = episodes[idx]
                episode_id_to_score[ep.episode_id] = float(score)
                episode_id_to_obj[ep.episode_id] = ep

            # 获取语义记忆候选（Top-Ps）
            sem_index = self.vector_search.semantic_indices.get(user_id)
            sem_data = self.vector_search.semantic_data.get(user_id, [])
            sem_id_to_score: Dict[str, float] = {}
            sem_items = []  # (memory_obj, score)
            if sem_index is not None and sem_data:
                ps = min(ps, len(sem_data))
                sem_scores, sem_indices = self.vector_search._search_faiss_index(
                    sem_index, q_vec, ps
                )
                for score, idx in zip(sem_scores, sem_indices):
                    if idx is None or idx < 0 or idx >= len(sem_data):
                        continue
                    mem = sem_data[idx]
                    sem_id_to_score[mem.memory_id] = float(score)
                    sem_items.append((mem, float(score)))

            # 构造候选情景集合：情景TopPe ∪ 由语义TopPs指向的情景
            # 建 episode_id -> 其相关分数组合（包含情景自身 + 命中的语义记忆）
            # 同时收集所有原始分数用于全局 z-score
            all_scores: List[float] = []
            episode_scores_map: Dict[str, List[float]] = {}

            # 先放入情景 TopPe
            for ep_id, s in episode_id_to_score.items():
                episode_scores_map.setdefault(ep_id, []).append(s)
                all_scores.append(s)

            # 语义记忆命中，挂到各自 source_episodes
            for mem, s in sem_items:
                if not getattr(mem, 'source_episodes', None):
                    continue
                for src_ep_id in mem.source_episodes:
                    episode_scores_map.setdefault(src_ep_id, []).append(s)
                    all_scores.append(s)

            # 有些仅由语义记忆带入的情景，其“情景自身向量”不在 TopPe，需要补充其情景分数
            # 通过内存中的嵌入直接计算余弦相似度
            if episode_scores_map:
                ep_emb_matrix = self.vector_search.episode_embeddings.get(user_id)
                if isinstance(ep_emb_matrix, np.ndarray) and ep_emb_matrix.size > 0:
                    # 预建 id->index
                    ep_list = self.vector_search.episode_data[user_id]
                    id_to_idx = {ep.episode_id: i for i, ep in enumerate(ep_list)}

                    # 归一化查询向量
                    q_norm = q_vec / (np.linalg.norm(q_vec) + eps)

                    for ep_id in list(episode_scores_map.keys()):
                        if ep_id in episode_id_to_score:
                            # 已有情景自身分数
                            continue
                        idx = id_to_idx.get(ep_id, None)
                        if idx is None:
                            continue
                        vec = ep_emb_matrix[idx]
                        if vec is None or np.asarray(vec).size == 0:
                            continue
                        v = np.asarray(vec, dtype=np.float32)
                        v_norm = v / (np.linalg.norm(v) + eps)
                        sim = float(np.dot(v_norm, q_norm))
                        episode_scores_map[ep_id].append(sim)
                        all_scores.append(sim)

            if not episode_scores_map:
                return []

            # z-score 标准化（基于候选集所有分数）
            scores_arr = np.array(all_scores, dtype=np.float32)
            mu = float(scores_arr.mean())
            sigma = float(scores_arr.std())
            denom = sigma + eps

            # 将每个 ep 的原始分数映射为 z/p
            def sigmoid(x: float) -> float:
                # 数值稳定 Sigmoid
                if x >= 0:
                    z = np.exp(-x)
                    return float(1.0 / (1.0 + z))
                else:
                    z = np.exp(x)
                    return float(z / (1.0 + z))

            # 计算全部 z 用于分位数阈值
            z_values = [ (s - mu) / denom for s in all_scores ]
            tau_z = float(np.percentile(np.array(z_values, dtype=np.float32), tau_percentile * 100.0))

            # 预计算每个原始分数的 p（重用）
            # 为避免重复计算，建立映射：raw_score -> p 值。这里原始分数是浮点，可能重复率低；为安全用四舍五入键。
            def p_from_score(s: float) -> float:
                z = (s - mu) / denom
                return sigmoid(lam * (z - tau_z))

            # 计算每个情景的 Noisy-OR 与 LIFT
            all_p_values: List[float] = []
            per_episode_p_list: Dict[str, List[float]] = {}
            for ep_id, s_list in episode_scores_map.items():
                p_list = [p_from_score(s) for s in s_list]
                per_episode_p_list[ep_id] = p_list
                all_p_values.extend(p_list)

            if not all_p_values:
                return []

            bar_p = float(np.mean(np.array(all_p_values, dtype=np.float32)))

            def clamp01(x: float) -> float:
                if x <= eps:
                    return eps
                if x >= 1.0 - eps:
                    return 1.0 - eps
                return x

            def logit(x: float) -> float:
                x = clamp01(x)
                return float(np.log(x / (1.0 - x)))

            episode_scores: List[tuple[str, float]] = []
            for ep_id, p_list in per_episode_p_list.items():
                if not p_list:
                    continue
                # P_i via Noisy-OR
                one_minus = 1.0
                for p in p_list:
                    one_minus *= (1.0 - p)
                P_i = clamp01(1.0 - one_minus)

                m = len(p_list)  # d_i + 1
                P0 = clamp01(1.0 - (1.0 - bar_p) ** m)
                S_i = logit(P_i) - logit(P0)
                episode_scores.append((ep_id, float(S_i)))

            # 排序并组装结果
            episode_scores.sort(key=lambda x: x[1], reverse=True)
            top = episode_scores[:top_k]

            # 获取 Episode 对象（如缺失则从 episode_data 中查找）
            ep_list = self.vector_search.episode_data[user_id]
            id_to_ep = {ep.episode_id: ep for ep in ep_list}

            results: List[Dict[str, Any]] = []
            for ep_id, s in top:
                ep = id_to_ep.get(ep_id)
                if not ep:
                    continue
                results.append({
                    "type": "episodic",
                    "score": float(s),
                    "episode_id": ep.episode_id,
                    "title": ep.title,
                    "content": ep.content,
                    "original_messages": ep.original_messages,
                    "boundary_reason": ep.boundary_reason,
                    "timestamp": ep.timestamp.isoformat(),
                    "created_at": ep.created_at.isoformat(),
                    "message_count": ep.message_count,
                    "search_method": "vector_norlift"
                })

            return results

        except Exception as e:
            logger.error(f"Error in NOR-LIFT episode ranking: {e}")
            # 失败回退到普通向量检索
            return self.vector_search.search_episodes(user_id, query, top_k)
    
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
                vector_future = executor.submit(self.vector_search.add_episode, user_id, episode)
                
                bm25_future.result()
                vector_future.result()
            
            logger.debug(f"🚀 Incrementally added episode to unified search for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error adding episode to unified search: {e}")
    
    def add_semantic_memory(self, user_id: str, memory: SemanticMemory):
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
                self.vector_search.add_semantic_memory(user_id, memory)
            else:
                # Add to both engines in parallel
                with ThreadPoolExecutor(max_workers=2) as executor:
                    bm25_future = executor.submit(self.bm25_search.add_semantic_memory, user_id, memory)
                    vector_future = executor.submit(self.vector_search.add_semantic_memory, user_id, memory)
                    
                    bm25_future.result()
                    vector_future.result()
            
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
            vector_success = self.vector_search.clear_user_index(user_id)
            
            return bm25_success and vector_success
            
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
            vector_stats = self.vector_search.get_index_stats()
            
            # Combine stats
            combined_stats = {
                "search_engines": {
                    "bm25": bm25_stats,
                    "vector": vector_stats
                },
                # Overall stats
                "total_episodes": max(
                    bm25_stats.get("total_episodes", 0),
                    vector_stats.get("total_episodes", 0)
                ),
                "total_semantic_memories": max(
                    bm25_stats.get("total_semantic_memories", 0),
                    vector_stats.get("total_semantic_memories", 0)
                ),
                "episode_users": max(
                    bm25_stats.get("episode_users", 0),
                    vector_stats.get("episode_users", 0)
                ),
                "semantic_users": max(
                    bm25_stats.get("semantic_users", 0),
                    vector_stats.get("semantic_users", 0)
                ),
                # Features
                "incremental_updates": vector_stats.get("incremental_updates", False),
                "optimized_updates": bm25_stats.get("optimized_updates", False),
                "faiss_available": vector_stats.get("faiss_available", False),
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