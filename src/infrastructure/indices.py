"""Infrastructure adapters for search indices."""

from __future__ import annotations

from typing import Dict, List, Optional
import threading

from ..domain.interfaces import LexicalIndex, VectorIndex
from ..models import Episode, SemanticMemory
from ..search.bm25_search import BM25Search
from ..search.chroma_search import ChromaSearchEngine


class Bm25Index(LexicalIndex):
    def __init__(self, backend: BM25Search) -> None:
        self._backend = backend

    def add_episode(self, user_id: str, episode: Episode) -> None:
        self._backend.add_episode(user_id, episode)

    def add_semantic(self, user_id: str, memory: SemanticMemory) -> None:
        self._backend.add_semantic_memory(user_id, memory)

    def search_episodes(self, user_id: str, query: str, top_k: int) -> List[Dict]:
        return self._backend.search_episodes(user_id, query, top_k)

    def search_semantics(self, user_id: str, query: str, top_k: int) -> List[Dict]:
        return self._backend.search_semantic_memories(user_id, query, top_k)

    def clear(self, user_id: str) -> bool:
        return self._backend.clear_user_index(user_id)

    def remove_episode(self, user_id: str, episode_id: str) -> bool:
        return self._backend.remove_episode(user_id, episode_id)

    def remove_semantic(self, user_id: str, memory_id: str) -> bool:
        return self._backend.remove_semantic_memory(user_id, memory_id)


class ChromaVectorIndex(VectorIndex):
    def __init__(self, backend: ChromaSearchEngine) -> None:
        self._backend = backend

    def add_episode(self, user_id: str, episode: Episode, embedding: Optional[List[float]] = None) -> None:
        self._backend.add_episode(user_id, episode) if embedding is None else self._backend.add_episode(user_id, episode)

    def add_semantic(self, user_id: str, memory: SemanticMemory, embedding: Optional[List[float]] = None) -> None:
        self._backend.add_semantic_memory(user_id, memory, embedding=embedding)

    def search_episodes(self, user_id: str, query: str, top_k: int) -> List[Dict]:
        return self._backend.search_episodes(user_id, query, top_k)

    def search_semantics(self, user_id: str, query: str, top_k: int) -> List[Dict]:
        return self._backend.search_semantic_memories(user_id, query, top_k)

    def clear(self, user_id: str) -> bool:
        return self._backend.clear_user_index(user_id)

    def remove_episode(self, user_id: str, episode_id: str) -> bool:
        return self._backend.remove_episode(user_id, episode_id)

    def remove_semantic(self, user_id: str, memory_id: str) -> bool:
        return self._backend.remove_semantic_memory(user_id, memory_id)


class InMemoryVectorIndex(VectorIndex):
    def __init__(self) -> None:
        self._episodes: Dict[str, List[Episode]] = {}
        self._semantics: Dict[str, List[SemanticMemory]] = {}
        self._lock = threading.Lock()

    def add_episode(self, user_id: str, episode: Episode, embedding: Optional[List[float]] = None) -> None:
        with self._lock:
            self._episodes.setdefault(user_id, []).append(episode)

    def add_semantic(self, user_id: str, memory: SemanticMemory, embedding: Optional[List[float]] = None) -> None:
        with self._lock:
            self._semantics.setdefault(user_id, []).append(memory)

    def search_episodes(self, user_id: str, query: str, top_k: int) -> List[Dict]:
        with self._lock:
            episodes = list(self._episodes.get(user_id, []))
        results = [
            {
                "episode_id": ep.episode_id,
                "title": ep.title,
                "content": ep.content,
                "score": 1.0 if query.lower() in ep.content.lower() else 0.5,
                "type": "episode",
            }
            for ep in episodes
            if query.lower() in ep.content.lower() or query.lower() in ep.title.lower()
        ]
        return results[:top_k]

    def search_semantics(self, user_id: str, query: str, top_k: int) -> List[Dict]:
        with self._lock:
            memories = list(self._semantics.get(user_id, []))
        results = [
            {
                "memory_id": mem.memory_id,
                "content": mem.content,
                "score": 1.0 if query.lower() in mem.content.lower() else 0.5,
                "type": "semantic",
            }
            for mem in memories
            if query.lower() in mem.content.lower()
        ]
        return results[:top_k]

    def clear(self, user_id: str) -> bool:
        with self._lock:
            self._episodes.pop(user_id, None)
            self._semantics.pop(user_id, None)
        return True

    def remove_episode(self, user_id: str, episode_id: str) -> bool:
        with self._lock:
            episodes = self._episodes.get(user_id)
            if not episodes:
                return False
            original_len = len(episodes)
            self._episodes[user_id] = [ep for ep in episodes if ep.episode_id != episode_id]
            return len(self._episodes[user_id]) != original_len

    def remove_semantic(self, user_id: str, memory_id: str) -> bool:
        with self._lock:
            memories = self._semantics.get(user_id)
            if not memories:
                return False
            original_len = len(memories)
            self._semantics[user_id] = [mem for mem in memories if mem.memory_id != memory_id]
            return len(self._semantics[user_id]) != original_len


class InMemoryLexicalIndex(LexicalIndex):
    def __init__(self) -> None:
        self._episodes: Dict[str, List[Episode]] = {}
        self._semantics: Dict[str, List[SemanticMemory]] = {}
        self._lock = threading.Lock()

    def add_episode(self, user_id: str, episode: Episode) -> None:
        with self._lock:
            self._episodes.setdefault(user_id, []).append(episode)

    def add_semantic(self, user_id: str, memory: SemanticMemory) -> None:
        with self._lock:
            self._semantics.setdefault(user_id, []).append(memory)

    def search_episodes(self, user_id: str, query: str, top_k: int) -> List[Dict]:
        with self._lock:
            episodes = list(self._episodes.get(user_id, []))
        results = [
            {
                "episode_id": ep.episode_id,
                "title": ep.title,
                "content": ep.content,
                "score": 1.0 if query.lower() in ep.content.lower() else 0.5,
                "type": "episode",
            }
            for ep in episodes
            if query.lower() in ep.content.lower() or query.lower() in ep.title.lower()
        ]
        return results[:top_k]

    def search_semantics(self, user_id: str, query: str, top_k: int) -> List[Dict]:
        with self._lock:
            memories = list(self._semantics.get(user_id, []))
        results = [
            {
                "memory_id": mem.memory_id,
                "content": mem.content,
                "score": 1.0 if query.lower() in mem.content.lower() else 0.5,
                "type": "semantic",
            }
            for mem in memories
            if query.lower() in mem.content.lower()
        ]
        return results[:top_k]

    def clear(self, user_id: str) -> bool:
        with self._lock:
            self._episodes.pop(user_id, None)
            self._semantics.pop(user_id, None)
        return True

    def remove_episode(self, user_id: str, episode_id: str) -> bool:
        with self._lock:
            episodes = self._episodes.get(user_id)
            if not episodes:
                return False
            original_len = len(episodes)
            self._episodes[user_id] = [ep for ep in episodes if ep.episode_id != episode_id]
            return len(self._episodes[user_id]) != original_len

    def remove_semantic(self, user_id: str, memory_id: str) -> bool:
        with self._lock:
            memories = self._semantics.get(user_id)
            if not memories:
                return False
            original_len = len(memories)
            self._semantics[user_id] = [mem for mem in memories if mem.memory_id != memory_id]
            return len(self._semantics[user_id]) != original_len
