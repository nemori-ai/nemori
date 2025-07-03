"""
BM25-based retrieval provider for Nemori episodic memory.

This module implements BM25 (Best Matching 25) retrieval strategy with
NLTK-based text processing and rank_bm25 for efficient BM25 scoring.
"""

import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

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


class BM25RetrievalProvider(RetrievalProvider):
    """
    BM25 retrieval provider using rank_bm25 library.

    Implements NLTK-based text processing with stemming, stopword removal,
    and maintains per-user indices for efficient retrieval.
    """

    def __init__(self, config: RetrievalConfig, storage_repo: EpisodicMemoryRepository):
        """Initialize BM25 retrieval provider."""
        super().__init__(config, storage_repo)

        # User indices: owner_id -> index data
        self.user_indices: dict[str, dict[str, Any]] = {}

        # Storage configuration based on storage type
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
            # For other storage types (duckdb, redis, etc.), disable local persistence
            # as they handle their own persistence
            self.persistence_enabled = False
            self.persistence_dir = None

        # Initialize NLTK components
        self._ensure_nltk_data()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words("english"))

    def _ensure_nltk_data(self):
        """Ensure required NLTK data is downloaded."""
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords", quiet=True)

    @property
    def strategy(self) -> RetrievalStrategy:
        """Return BM25 strategy."""
        return RetrievalStrategy.BM25

    async def initialize(self) -> None:
        """Initialize the BM25 provider."""
        if self._initialized:
            return

        # Load existing indices from disk if persistence is enabled
        if self.persistence_enabled:
            self._load_all_indices_from_disk()

        self._initialized = True

    def _load_all_indices_from_disk(self) -> None:
        """Load all existing indices from disk."""
        if not self.persistence_enabled or not self.persistence_dir or not self.persistence_dir.exists():
            return

        # Find all index files
        index_files = list(self.persistence_dir.glob("bm25_index_*.pkl"))

        for index_file in index_files:
            # Extract owner_id from filename
            filename = index_file.stem  # e.g., "bm25_index_agent"
            if filename.startswith("bm25_index_"):
                owner_id = filename[11:]  # Remove "bm25_index_" prefix
                success = self._load_index_from_disk(owner_id)
                if success:
                    print(f"Loaded BM25 index for owner: {owner_id}")
                else:
                    print(f"Failed to load BM25 index for owner: {owner_id}")

    async def close(self) -> None:
        """Close the provider and cleanup resources."""
        # Save all indices to disk before closing
        if self.persistence_enabled:
            for owner_id in self.user_indices.keys():
                self._save_index_to_disk(owner_id)

        self.user_indices.clear()
        self._initialized = False

    def _tokenize(self, text: str) -> list[str]:
        """
        NLTK-based tokenization with stemming and stopword removal.

        Args:
            text: Input text

        Returns:
            List of processed tokens (stemmed, lowercase, no stopwords)
        """
        if not text:
            return []

        # Tokenize using NLTK
        tokens = word_tokenize(text.lower())

        # Process tokens: remove stopwords, non-alphabetic, and apply stemming
        processed_tokens = []
        for token in tokens:
            # Keep only alphabetic tokens that are not stopwords and have length >= 2
            if token.isalpha() and len(token) >= 2 and token not in self.stop_words:
                # Apply stemming
                stemmed_token = self.stemmer.stem(token)
                processed_tokens.append(stemmed_token)

        return processed_tokens

    def _get_user_index(self, owner_id: str) -> dict[str, Any]:
        """Get or create user index."""
        if owner_id not in self.user_indices:
            self.user_indices[owner_id] = {
                "episodes": [],  # Store episode objects
                "corpus": [],  # Store tokenized documents for BM25
                "bm25": None,  # BM25 object
                "episode_id_to_index": {},  # Map episode_id to corpus index
                "last_updated": datetime.now(),
            }
        return self.user_indices[owner_id]

    def _build_searchable_text(self, episode: Episode) -> str:
        """
        Build searchable text from episode with weighted fields.

        Args:
            episode: Episode to process

        Returns:
            Searchable text string
        """
        parts = []

        # Title has highest weight (repeat 3 times)
        if episode.title:
            parts.extend([episode.title] * 3)

        # Content
        if episode.content:
            parts.append(episode.content)

        # Summary (repeat 2 times)
        if episode.summary:
            parts.extend([episode.summary] * 2)

        # Entities (repeat 2 times)
        if episode.metadata.entities:
            parts.extend(episode.metadata.entities * 2)

        # Topics (repeat 2 times)
        if episode.metadata.topics:
            parts.extend(episode.metadata.topics * 2)

        # Key points
        if episode.metadata.key_points:
            parts.extend(episode.metadata.key_points)

        return " ".join(str(part) for part in parts if part)

    def _rebuild_bm25_index(self, owner_id: str) -> None:
        """Rebuild BM25 index for a user."""
        index = self._get_user_index(owner_id)
        if index["corpus"]:
            index["bm25"] = BM25Okapi(index["corpus"])
            index["last_updated"] = datetime.now()

            # Save index to disk if persistence is enabled
            if self.persistence_enabled:
                self._save_index_to_disk(owner_id)

    def _get_index_file_path(self, owner_id: str) -> Path | None:
        """Get the file path for a user's index."""
        if not self.persistence_enabled or not self.persistence_dir:
            return None
        return self.persistence_dir / f"bm25_index_{owner_id}.pkl"

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
                "corpus": index["corpus"],
                "episode_id_to_index": index["episode_id_to_index"],
                "last_updated": index["last_updated"].isoformat(),
                "metadata": {
                    "total_episodes": len(index["episodes"]),
                    "total_tokens": sum(len(doc) for doc in index["corpus"]),
                },
            }

            # Serialize episodes as dictionaries (not full objects to avoid circular refs)
            for episode in index["episodes"]:
                episode_data = {
                    "episode_id": episode.episode_id,
                    "owner_id": episode.owner_id,
                    "title": episode.title,
                    "content": episode.content,
                    "summary": episode.summary,
                    "episode_type": episode.episode_type.value,
                    "level": episode.level.value,
                    "timestamp": episode.temporal_info.timestamp.isoformat(),
                    "duration": episode.temporal_info.duration,
                    "search_keywords": episode.search_keywords,
                    "importance_score": episode.importance_score,
                    "recall_count": episode.recall_count,
                }
                serializable_data["episodes"].append(episode_data)

            with open(index_file, "wb") as f:
                pickle.dump(serializable_data, f)

        except Exception as e:
            print(f"Warning: Failed to save BM25 index for {owner_id}: {e}")

    def _load_index_from_disk(self, owner_id: str) -> bool:
        """Load user index from disk. Returns True if successful."""
        if not self.persistence_enabled:
            return False

        try:
            index_file = self._get_index_file_path(owner_id)
            if not index_file or not index_file.exists():
                return False

            with open(index_file, "rb") as f:
                data = pickle.load(f)

            # Recreate the index structure
            index = self._get_user_index(owner_id)
            index["corpus"] = data["corpus"]
            index["episode_id_to_index"] = data["episode_id_to_index"]
            index["last_updated"] = datetime.fromisoformat(data["last_updated"])

            # Rebuild BM25 index from corpus
            if index["corpus"]:
                index["bm25"] = BM25Okapi(index["corpus"])

            # For now, we won't reload full episode objects as they should come from storage
            # This is just the index data needed for search
            index["episodes"] = []  # Will be populated when episodes are added

            return True

        except Exception as e:
            print(f"Warning: Failed to load BM25 index for {owner_id}: {e}")
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

            # Initialize episodes array to match corpus size (maintain synchronization)
            corpus_size = len(index["corpus"])
            index["episodes"] = [None] * corpus_size

            # Re-add episodes in the same order as the corpus
            episodes_found = 0
            for episode in episodes:
                if episode.episode_id in index["episode_id_to_index"]:
                    corpus_index = index["episode_id_to_index"][episode.episode_id]
                    if 0 <= corpus_index < corpus_size:
                        index["episodes"][corpus_index] = episode
                        episodes_found += 1

            print(f"Reloaded {episodes_found} episodes for owner: {owner_id} (corpus size: {corpus_size})")

            # Verify synchronization
            if episodes_found != corpus_size:
                print(f"Warning: Episode count ({episodes_found}) doesn't match corpus size ({corpus_size})")

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
                tokens = self._tokenize(searchable_text)

                # Add to index
                corpus_index = len(index["episodes"])
                index["episodes"].append(episode)
                index["corpus"].append(tokens)
                index["episode_id_to_index"][episode.episode_id] = corpus_index

            # Rebuild BM25 index
            self._rebuild_bm25_index(owner_id)

    async def remove_episode(self, episode_id: str) -> bool:
        """Remove an episode from the index."""
        # Find which user index contains this episode
        for owner_id, index in self.user_indices.items():
            if episode_id in index["episode_id_to_index"]:
                corpus_index = index["episode_id_to_index"][episode_id]

                # Remove from all structures
                del index["episodes"][corpus_index]
                del index["corpus"][corpus_index]
                del index["episode_id_to_index"][episode_id]

                # Update remaining indices
                for eid, idx in index["episode_id_to_index"].items():
                    if idx > corpus_index:
                        index["episode_id_to_index"][eid] = idx - 1

                # Rebuild BM25 index
                self._rebuild_bm25_index(owner_id)
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
        """Search for relevant episodes using BM25."""
        start_time = time.time()

        # Validate query
        self._validate_query(query)

        index = self._get_user_index(query.owner_id)

        # If no corpus or no BM25 index, return empty results
        if not index["corpus"] or index["bm25"] is None:
            return RetrievalResult(
                episodes=[],
                scores=[],
                total_candidates=0,
                query_time_ms=(time.time() - start_time) * 1000,
                strategy_used=self.strategy,
            )

        # If episodes array is empty but corpus exists (loaded from disk),
        # try to reload episodes from storage
        if not index["episodes"] and index["corpus"]:
            await self._reload_episodes_from_storage(query.owner_id)

            # If still no episodes after reload, return empty result
            if not index["episodes"]:
                print(f"Warning: Could not reload episodes from storage for owner: {query.owner_id}")
                return RetrievalResult(
                    episodes=[],
                    scores=[],
                    total_candidates=len(index["corpus"]),
                    query_time_ms=(time.time() - start_time) * 1000,
                    strategy_used=self.strategy,
                )

        # Normal case: both episodes and corpus are available
        if not index["episodes"]:
            return RetrievalResult(
                episodes=[],
                scores=[],
                total_candidates=0,
                query_time_ms=(time.time() - start_time) * 1000,
                strategy_used=self.strategy,
            )

        # Tokenize query
        query_tokens = self._tokenize(query.text)
        if not query_tokens:
            return RetrievalResult(
                episodes=[],
                scores=[],
                total_candidates=len(index["episodes"]),
                query_time_ms=(time.time() - start_time) * 1000,
                strategy_used=self.strategy,
            )

        # Get BM25 scores
        scores = index["bm25"].get_scores(query_tokens)

        # If all BM25 scores are 0, use simple term frequency as fallback
        if len(scores) > 0 and scores.max() == 0:
            for i in range(len(index["corpus"])):
                doc_tokens = index["corpus"][i]
                # Count how many query tokens appear in the document
                match_count = sum(1 for token in query_tokens if token in doc_tokens)
                if match_count > 0:
                    # Simple TF score: matches / doc_length
                    scores[i] = match_count / len(doc_tokens) if doc_tokens else 0

        # Create results with scores
        results = []
        for i, score in enumerate(scores):
            # Skip if episode is None (can happen after reloading from disk)
            if i >= len(index["episodes"]) or index["episodes"][i] is None:
                continue

            episode = index["episodes"][i]

            # Apply filters if specified
            if query.episode_types and episode.episode_type.value not in query.episode_types:
                continue

            if query.time_range_hours and not episode.is_recent(query.time_range_hours):
                continue

            if query.min_importance and episode.importance_score < query.min_importance:
                continue

            results.append((episode, float(score)))

        # Sort by score (descending)
        results.sort(key=lambda x: x[1], reverse=True)

        # Apply limit
        limited_results = results[: query.limit]

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
                "query_tokens": query_tokens,
                "total_indexed_episodes": len(index["episodes"]),
            },
        )

    async def rebuild_index(self) -> None:
        """Rebuild the entire index from storage."""
        # Clear existing indices
        self.user_indices.clear()

        # Note: This is a simplified implementation
        # In a real system, you'd want to fetch all episodes and rebuild
        # For now, indices will be rebuilt on-demand when users search
        pass

    async def get_stats(self) -> IndexStats:
        """Get statistics about the BM25 index."""
        total_episodes = sum(len(idx["episodes"]) for idx in self.user_indices.values())
        total_documents = sum(len(idx["corpus"]) for idx in self.user_indices.values())

        # Estimate index size (rough calculation)
        index_size_mb = 0.0
        for idx in self.user_indices.values():
            # Rough size estimation based on corpus tokens
            token_count = sum(len(doc) for doc in idx["corpus"])
            index_size_mb += token_count * 0.001  # Rough estimate: 1KB per 1000 tokens

        last_updated = None
        if self.user_indices:
            last_updated = max(idx["last_updated"] for idx in self.user_indices.values())

        return IndexStats(
            total_episodes=total_episodes,
            total_documents=total_documents,
            index_size_mb=index_size_mb,
            last_updated=last_updated,
            provider_stats={
                "user_indices_count": len(self.user_indices),
                "tokenization_method": "nltk_with_stemming",
                "weighting_strategy": "title*3, summary*2, entities*2, topics*2",
                "stemmer": "porter",
                "stopwords_removed": True,
            },
        )

    async def health_check(self) -> bool:
        """Check if the BM25 provider is healthy."""
        return self._initialized
