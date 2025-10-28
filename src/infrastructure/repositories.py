"""Infrastructure adapters implementing repository interfaces."""

from __future__ import annotations

import threading
from typing import Dict, List

from ..domain.interfaces import EpisodeRepository, SemanticRepository
from ..models import Episode, SemanticMemory
from ..storage.episode_storage import EpisodeStorage
from ..storage.semantic_storage import SemanticStorage


class EpisodeStorageRepository(EpisodeRepository):
    def __init__(self, storage: EpisodeStorage) -> None:
        self._storage = storage

    def save(self, episode: Episode) -> str:
        return self._storage.save_episode(episode)

    def list_by_user(self, user_id: str) -> List[Episode]:
        return self._storage.get_user_episodes(user_id)

    def delete_user_data(self, user_id: str) -> bool:
        return self._storage.delete_user_data(user_id)

    def delete(self, user_id: str, episode_id: str) -> bool:
        return self._storage.delete(user_id, episode_id)


class SemanticStorageRepository(SemanticRepository):
    def __init__(self, storage: SemanticStorage) -> None:
        self._storage = storage

    def save(self, memory: SemanticMemory) -> str:
        return self._storage.save_semantic_memory(memory)

    def list_by_user(self, user_id: str) -> List[SemanticMemory]:
        return self._storage.list_user_items(user_id)

    def delete_user_data(self, user_id: str) -> bool:
        return self._storage.delete_user_data(user_id)

    def delete(self, user_id: str, memory_id: str) -> bool:
        return self._storage.delete_semantic_memory(user_id, memory_id)


class InMemoryEpisodeRepository(EpisodeRepository):
    def __init__(self) -> None:
        self._episodes: Dict[str, List[Episode]] = {}
        self._lock = threading.Lock()

    def save(self, episode: Episode) -> str:
        with self._lock:
            self._episodes.setdefault(episode.user_id, []).append(episode)
        return episode.episode_id

    def list_by_user(self, user_id: str) -> List[Episode]:
        with self._lock:
            return list(self._episodes.get(user_id, []))

    def delete_user_data(self, user_id: str) -> bool:
        with self._lock:
            return self._episodes.pop(user_id, None) is not None

    def delete(self, user_id: str, episode_id: str) -> bool:
        with self._lock:
            episodes = self._episodes.get(user_id)
            if not episodes:
                return False
            original_len = len(episodes)
            self._episodes[user_id] = [ep for ep in episodes if ep.episode_id != episode_id]
            return len(self._episodes[user_id]) != original_len


class InMemorySemanticRepository(SemanticRepository):
    def __init__(self) -> None:
        self._memories: Dict[str, List[SemanticMemory]] = {}
        self._lock = threading.Lock()

    def save(self, memory: SemanticMemory) -> str:
        with self._lock:
            self._memories.setdefault(memory.user_id, []).append(memory)
        return memory.memory_id

    def list_by_user(self, user_id: str) -> List[SemanticMemory]:
        with self._lock:
            return list(self._memories.get(user_id, []))

    def delete_user_data(self, user_id: str) -> bool:
        with self._lock:
            return self._memories.pop(user_id, None) is not None

    def delete(self, user_id: str, memory_id: str) -> bool:
        with self._lock:
            memories = self._memories.get(user_id)
            if not memories:
                return False
            original_len = len(memories)
            self._memories[user_id] = [mem for mem in memories if mem.memory_id != memory_id]
            return len(self._memories[user_id]) != original_len
