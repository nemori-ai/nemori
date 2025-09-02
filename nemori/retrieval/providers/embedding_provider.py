"""
Embedding-based retrieval provider for Nemori episodic memory.

This module implements vector similarity search using sentence embeddings
for semantic retrieval of episodes.
"""

import json
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import tiktoken

from ...core.episode import Episode
from ...storage.repository import EpisodicMemoryRepository
from ..retrieval_types import (
    IndexStats,
    RetrievalConfig,
    RetrievalQuery,
    RetrievalResult,
    RetrievalStorageType,
    RetrievalStrategy,
)
from .base import RetrievalProvider


class EmbeddingRetrievalProvider(RetrievalProvider):
    """
    Embedding-based retrieval provider using vector similarity search.
    
    Uses sentence embeddings to enable semantic search that goes beyond
    keyword matching to find conceptually similar episodes.
    """

    def __init__(self, config: RetrievalConfig, storage_repo: EpisodicMemoryRepository):
        """Initialize embedding retrieval provider."""
        super().__init__(config, storage_repo)
        
        # User indices: owner_id -> index data
        self.user_indices: Dict[str, Dict[str, Any]] = {}
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            print(f"Warning: Failed to initialize tiktoken, falling back to character-based summary. Error: {e}")
            self.tokenizer = None
        # Embedding configuration
        self.embedding_model = None
        self.embedding_dim = None
        self.openai_client = None
        self.api_key = config.api_key
        self.base_url = config.base_url
        self.embed_model = config.embed_model
        # Storage configuration
        self.storage_type = config.storage_type
        self.storage_config = config.storage_config.copy()
        
        # Set up persistence based on storage type
        if self.storage_type == RetrievalStorageType.MEMORY:
            self.persistence_enabled = False
            self.persistence_dir = None
        elif self.storage_type == RetrievalStorageType.DISK:
            self.persistence_enabled = True
            # Get directory from storage_config, fallback to default
            self.persistence_dir = Path(self.storage_config.get("directory", ".tmp"))
            self.persistence_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.persistence_enabled = False
            self.persistence_dir = None

    @property
    def strategy(self) -> RetrievalStrategy:
        """Return embedding strategy."""
        return RetrievalStrategy.EMBEDDING
    
    async def initialize(self) -> None:
        """Initialize the embedding provider."""
        if self._initialized:
            return
            
        # Initialize embedding model
        self._initialize_embedding_model()
        
        # Load existing indices from disk if persistence is enabled
        if self.persistence_enabled:
            self._load_all_indices_from_disk()
            
        self._initialized = True

    def _load_all_indices_from_disk(self) -> None:
        """Load all existing indices from disk."""
        if not self.persistence_enabled or not self.persistence_dir or not self.persistence_dir.exists():
            return
            
        # Find all index files
        index_files = list(self.persistence_dir.glob("embedding_index_*.json"))
        
        for index_file in index_files:
            # Extract owner_id from filename
            filename = index_file.stem  # e.g., "embedding_index_user123"
            if filename.startswith("embedding_index_"):
                owner_id = filename[16:]  # Remove "embedding_index_" prefix
                success = self._load_index_from_disk(owner_id)
                if success:
                    print(f"Loaded embedding index for owner: {owner_id}")
                else:
                    print(f"Failed to load embedding index for owner: {owner_id}")

    async def close(self) -> None:
        """Close the provider and cleanup resources."""
        # Save all indices to disk before closing
        if self.persistence_enabled:
            for owner_id in self.user_indices.keys():
                self._save_index_to_disk(owner_id)
                
        self.user_indices.clear()
        self.embedding_model = None
        self.openai_client = None
        self._initialized = False

    async def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for the given text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of float values representing the embedding
        """
        if not text.strip():
            return [0.0] * self.embedding_dim
            
        if isinstance(self.embedding_model, str):
            if not self.openai_client:
                raise ValueError("OpenAI client not initialized")
            # OpenAI embedding
            response = await self.openai_client.embeddings.create(
                model=self.embed_model,
                input=text
            )
            return response.data[0].embedding
            

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            # Convert to numpy arrays
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            
            # Calculate cosine similarity
            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception:
            return 0.0



    def _initialize_embedding_model(self) -> None:
        """Initialize the embedding model based on configuration."""
        # Fallback to OpenAI embeddings if configured
        from openai import AsyncOpenAI

        openai_api_key = self.api_key
        base_url = self.base_url

        if openai_api_key or base_url:
            self.openai_client = AsyncOpenAI(api_key=openai_api_key, base_url=base_url)
            self.embedding_model = self.embed_model
            self.embedding_dim = 1024  # OpenAI text-embedding-ada-002 dimension
            print(f"Using Qwen3 embeddings (dimension: {self.embedding_dim})")

    def _get_user_index(self, owner_id: str) -> Dict[str, Any]:
        """
        Get or create user index.
        
        If the index is not in memory, it attempts to load from disk.
        If not on disk, it rebuilds from the storage repository.
        """
        if owner_id not in self.user_indices:
            self.user_indices[owner_id] = {
                "episodes": [],
                "embeddings": [],
                "episode_id_to_index": {},
                "last_updated": datetime.now(),
            }

        return self.user_indices[owner_id]

    def _build_searchable_text(self, episode: Episode) -> str:
        """
        Build searchable text from episode for embedding.
        
        Args:
            episode: Episode to process
            
        Returns:
            Searchable text string
        """
        parts = []
        
        # Title
        if episode.title:
            parts.append(episode.title)
            
        # Summary
        if episode.summary:
            parts.append(episode.summary)
            
        # Content (truncated for embedding efficiency)
        if episode.content:
            # tokens = self.tokenizer.encode(episode.content)
            content = episode.content #[:1000]  # Limit content length
            # content = self.tokenizer.decode(tokens[:512])
            parts.append(content)
            
        # Key entities and topics
        if episode.metadata.entities:
            parts.append(" ".join(episode.metadata.entities))
            
        if episode.metadata.topics:
            parts.append(" ".join(episode.metadata.topics))
            
        return " ".join(parts)

        
    async def _rebuild_embedding_index(self, owner_id: str) -> None:
        """
        [正确实现] 从存储库中为一个用户彻底重建 Embedding 索引。
        这个函数会清空现有索引，从数据库重新获取所有episodes，并为它们生成新的embeddings。
        """
        index = self._get_user_index(owner_id)
        
        print(f"   - Clearing existing in-memory index for {owner_id}...")
        index["episodes"].clear()
        index["embeddings"].clear()
        index["episode_id_to_index"].clear()
        
        # [关键步骤 2] 从数据库获取所有 episodes 作为事实来源
        print(f"   - Fetching all episodes from storage for {owner_id}...")
        result = await self.storage_repo.get_episodes_by_owner(owner_id)
        episodes_from_db = result.episodes
        
        if not episodes_from_db:
            print(f"   - No episodes found in storage for owner {owner_id}. Index will be empty.")
            index["last_updated"] = datetime.now()
            if self.persistence_enabled:
                self._save_index_to_disk(owner_id)
            return
            
        print(f"   - Found {len(episodes_from_db)} episodes. Generating embeddings...")
        
        # [关键步骤 3 & 4] 遍历所有 episodes，生成 embedding 并重新填充索引
        for i, episode in enumerate(episodes_from_db):
            # 打印进度，这对于耗时操作很重要
            if (i + 1) % 10 == 0 or i == len(episodes_from_db) - 1:
                print(f"     - Processing episode {i + 1}/{len(episodes_from_db)}...")
                
            # 为 episode 生成 embedding
            searchable_text = self._build_searchable_text(episode)
            embedding = await self._generate_embedding(searchable_text)
            
            # 获取当前列表长度作为新条目的索引位置
            # 这是保证 episode 和 embedding 一一对应的关键
            new_index_position = len(index["episodes"])
            
            # 填充所有数据结构
            index["episodes"].append(episode)
            index["embeddings"].append(embedding)
            index["episode_id_to_index"][episode.episode_id] = new_index_position

        # [关键步骤 5] 更新时间戳并保存到磁盘
        index["last_updated"] = datetime.now()
        if self.persistence_enabled:
            print(f"   - Saving newly built index to disk for {owner_id}...")
            self._save_index_to_disk(owner_id)
            
        print(f"✅ Finished rebuilding embedding index for owner: {owner_id}. Total episodes: {len(index['episodes'])}.")


    def _get_index_file_path(self, owner_id: str) -> Path | None:
        """Get the file path for a user's index."""
        if not self.persistence_enabled or not self.persistence_dir:
            return None
        return self.persistence_dir / f"embedding_index_{owner_id}.json"

    def _save_index_to_disk(self, owner_id: str) -> None:
        """Save user index to disk."""
        if not self.persistence_enabled:
            return
            
        try:
            index = self.user_indices.get(owner_id)
            if not index:
                return
                
            index_file = self._get_index_file_path(owner_id)
            if not index_file:
                return
                
            # Prepare data for serialization (exclude BM25 object)
            serializable_data = {
                "episodes": [],  # Will store episode data as dicts
                "embeddings": index["embeddings"],
                "episode_id_to_index": index["episode_id_to_index"],
                "last_updated": index["last_updated"].isoformat(),
                "metadata": {
                    "total_episodes": len(index["episodes"]),
                    "total_tokens": sum(len(doc) for doc in index["embeddings"]),
                },
            }
            # Store episode IDs (not full objects to avoid circular refs)
            for episode in index["episodes"]:
                #print("episode keys:",episode.keys())
                # 假设 'episode' 是一个字典变量
                episode_data = {
                    "episode_id": episode['episode_id'],
                    "owner_id": episode['owner_id'],
                    "title": episode['title'],
                    "content": episode['content'],
                    "summary": episode['summary'],
                    # 对于 .value 的情况，字典中通常直接存储了最终的值
                    "episode_type": episode['episode_type'],
                    "level": episode['level'],
                    # 对于嵌套属性，字典中通常是扁平化的键
                    "timestamp": episode['timestamp'], # 假设字典中的值已经是 ISO 格式的字符串
                    "duration": episode['duration'],
                    "search_keywords": episode['search_keywords'],
                    "importance_score": episode['importance_score'],
                    "recall_count": episode['recall_count'],
                }
                serializable_data["episodes"].append(episode_data)
                
            with open(index_file, "w") as f:
                json.dump(serializable_data, f, ensure_ascii=False)
                
        except Exception as e:
            print(f"Warning: Failed to save embedding index for {owner_id}: {e}")

    
    def _load_index_from_disk(self, owner_id: str) -> bool:
        """Load user index from disk. Returns True if successful."""
        if not self.persistence_enabled:
            return False
            
        try:
            index_file = self._get_index_file_path(owner_id)
            if not index_file or not index_file.exists():
                return False
                
            with open(index_file, "r") as f:
                data = json.load(f)
                
            # Recreate the index structure
            index = self._get_user_index(owner_id)
            # Episodes will be loaded from storage when needed
            index["episodes"] = data["episodes"]
            index["embeddings"] = data["embeddings"]
            index["episode_id_to_index"] = data["episode_id_to_index"]
            index["last_updated"] = datetime.fromisoformat(data["last_updated"])
            
            return True
            
        except Exception as e:
            print(f"Warning: Failed to load embedding index for {owner_id}: {e}")
            return False

    async def _reload_episodes_from_storage(self, owner_id: str) -> None:
        """Reload episodes from storage repository for the given owner."""
        try:
            if not self.storage_repo:
                print("Warning: No storage repository available to reload episodes")
                return

            # Get episodes for this owner from storage
            result = await self.storage_repo.get_episodes_by_owner(owner_id)
            episodes = result.episodes

            if not episodes:
                print(f"Warning: No episodes found in storage for owner: {owner_id}")
                return

            # Get the user index
            index = self._get_user_index(owner_id)

            # Initialize episodes array to match embeddings size (maintain synchronization)
            embeddings_size = len(index["embeddings"])
            index["episodes"] = [None] * embeddings_size

            # Re-add episodes in the same order as the embeddings
            episodes_found = 0
            for episode in episodes:
                if episode.episode_id in index["episode_id_to_index"]:
                    embeddings_index = index["episode_id_to_index"][episode.episode_id]
                    if 0 <= embeddings_index < embeddings_size:
                        index["episodes"][embeddings_index] = episode
                        episodes_found += 1

            print(f"Reloaded {episodes_found} episodes for owner: {owner_id} (embeddings size: {embeddings_size})")

            # Verify synchronization
            if episodes_found != embeddings_size:
                print(f"Warning: Episode count ({episodes_found}) doesn't match embeddings size ({embeddings_size})")

        except Exception as e:
            print(f"Error reloading episodes from storage for {owner_id}: {e}")

  

    async def add_episode(self, episode: Episode) -> None:
        """Add a new episode to the index."""
        await self.add_episodes_batch([episode])


    async def add_episodes_batch(self, episodes: list[Episode]) -> None:
        """Add multiple episodes to the index in batch."""
        if not episodes:
            return

        # Group episodes by owner
        episodes_by_owner: dict[str, list[Episode]] = {}
        for episode in episodes:
            if episode.owner_id not in episodes_by_owner:
                episodes_by_owner[episode.owner_id] = []
            episodes_by_owner[episode.owner_id].append(episode)

        # Add to each user's index
        for owner_id, user_episodes in episodes_by_owner.items():
            index = self._get_user_index(owner_id)

            for episode in user_episodes:
                # Skip if episode already exists in index
                if episode.episode_id in index["episode_id_to_index"]:
                    continue

                # Build searchable text and tokenize
                searchable_text = self._build_searchable_text(episode)
                embedding = await self._generate_embedding(searchable_text)

                # Add to index
                embeddings_index = len(index["episodes"])
                index["episodes"].append(episode)
                index["embeddings"].append(embedding)
                index["episode_id_to_index"][episode.episode_id] = embeddings_index

            await self._rebuild_embedding_index(owner_id)

    async def remove_episode(self, episode_id: str) -> bool:
        """Remove an episode from the index."""
        # Find which user index contains this episode
        for owner_id, index in self.user_indices.items():
            if episode_id in index["episode_id_to_index"]:
                embeddings_index = index["episode_id_to_index"][episode_id]

                # Remove from all structures
                del index["episodes"][embeddings_index]
                del index["embeddings"][embeddings_index]
                del index["episode_id_to_index"][episode_id]

                # Update remaining indices
                for eid, idx in index["episode_id_to_index"].items():
                    if idx > embeddings_index:
                        index["episode_id_to_index"][eid] = idx - 1

                # Rebuild embeddings index
                await self._rebuild_embedding_index(owner_id)
                return True

        return False

    async def update_episode(self, episode: Episode) -> bool:
        """Update an existing episode in the index."""
        # Remove old version and add new version
        removed = await self.remove_episode(episode.episode_id)
        if removed:
            await self.add_episode(episode)
            return True
        return False

    async def search(self, query: RetrievalQuery) -> RetrievalResult:
        """Search for relevant episodes using embedding similarity."""
        start_time = time.time()
        
        # Validate query
        self._validate_query(query)
        
        index = self._get_user_index(query.owner_id)
        # ======================== [ 新增改造代码块 START ] ========================
        # If embeddings are loaded from disk but episodes are not in memory, reload them.
        
        if index["embeddings"] is None:
            return RetrievalResult(
                episodes=[],
                scores=[],
                total_candidates=0,
                query_time_ms=(time.time() - start_time) * 1000,
                strategy_used=self.strategy,
            )
        
        if not index["episodes"] and index["embeddings"]:
            await self._reload_episodes_from_storage(query.owner_id)
            x = index["episodes"]
            y = index["embeddings"][:1]
            print(f"episodes: {x}, embeddings: {y}")
            # If still no episodes after reload, return empty result
            if not index["episodes"]:
                print(f"Warning: Could not reload episodes from storage for owner: {query.owner_id}")
                return RetrievalResult(
                    episodes=[],
                    scores=[],
                    total_candidates=len(index["embeddings"]),
                    query_time_ms=(time.time() - start_time) * 1000,
                    strategy_used=self.strategy,
                )

        # Generate query embedding
        query_embedding = await self._generate_embedding(query.text)
        if not query_embedding or all(x == 0 for x in query_embedding):
            return RetrievalResult(
                episodes=[],
                scores=[],
                total_candidates=len(index["episodes"]),
                query_time_ms=(time.time() - start_time) * 1000,
                strategy_used=self.strategy,
            )
            
        # Calculate similarities
        results = []
        # [小优化] 确保 episodes 和 embeddings 列表长度一致，避免索引越界
        min_len = min(len(index["episodes"]), len(index["embeddings"]))
        for i, (episode, embedding) in enumerate(zip(index["episodes"], index["embeddings"])):
            # Apply filters if specified
            if query.episode_types and episode.episode_type.value not in query.episode_types:
                continue
                
            if query.time_range_hours and not episode.is_recent(query.time_range_hours):
                continue
                
            if query.min_importance and episode.importance_score < query.min_importance:
                continue
                
            # Calculate similarity
            similarity = self._cosine_similarity(query_embedding, embedding)
            results.append((episode, similarity))
            
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Apply limit
        limited_results = results[:query.limit]
        
        # Split episodes and scores
        episodes = [ep for ep, _ in limited_results]
        final_scores = [score for _, score in limited_results]
        
        query_time_ms = (time.time() - start_time) * 1000
        
        return RetrievalResult(
            episodes=episodes,
            scores=final_scores,
            total_candidates=len(results),
            query_time_ms=query_time_ms,
            strategy_used=self.strategy,
            metadata={
                "query_embedding_generated": True,
                "embedding_dimension": len(query_embedding),
                "total_indexed_episodes": len(index["episodes"]),
            },
        )

    async def rebuild_index(self) -> None:
        """Rebuild the entire index from storage."""
        # Clear existing indices
        self.user_indices.clear()
        
        # Note: In a real implementation, you'd fetch episodes from storage
        # For now, indices will be rebuilt when episodes are added
        pass

    async def get_stats(self) -> IndexStats:
        """Get statistics about the embedding index."""
        total_episodes = sum(len(idx["episodes"]) for idx in self.user_indices.values())
        total_embeddings = sum(len(idx["embeddings"]) for idx in self.user_indices.values())
        
        # Estimate index size
        index_size_mb = 0.0
        for idx in self.user_indices.values():
            # Each embedding vector size estimation
            embedding_count = len(idx["embeddings"])
            if embedding_count > 0 and self.embedding_dim:
                index_size_mb += (embedding_count * self.embedding_dim * 4) / (1024 * 1024)  # 4 bytes per float
                
        last_updated = None
        if self.user_indices:
            last_updated = max(idx["last_updated"] for idx in self.user_indices.values())
            
        return IndexStats(
            total_episodes=total_episodes,
            total_documents=total_embeddings,
            index_size_mb=index_size_mb,
            last_updated=last_updated,
            provider_stats={
                "user_indices_count": len(self.user_indices),
                "embedding_dimension": self.embedding_dim,
                "embedding_model": str(self.embedding_model)[:100],  # Truncate for readability
                "similarity_metric": "cosine",
            },
        )

    async def health_check(self) -> bool:
        """Check if the embedding provider is healthy."""
        return self._initialized