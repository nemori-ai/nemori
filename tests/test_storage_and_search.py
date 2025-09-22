import json
import re
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import Episode
from src.search.bm25_search import BM25Search
from src.search.vector_search import VectorSearch
from src.storage.episode_storage import EpisodeStorage
from src.utils.embedding_client import EmbeddingResponse


class DummyEmbeddingClient:
    """Deterministic embedding client for tests."""

    def __init__(self, dimension: int = 4):
        self.dimension = dimension
        self.vocab = {"apple": 0, "banana": 1, "carrot": 2, "date": 3}

    def _tokenize(self, text: str):
        return re.findall(r"\w+", text.lower())

    def _embed(self, text: str):
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in self._tokenize(text):
            if token in self.vocab:
                vector[self.vocab[token]] += 1.0
        if not np.any(vector):
            vector += 1e-6
        return vector.tolist()

    def embed_texts(self, texts):
        embeddings = [self._embed(text) for text in texts]
        return EmbeddingResponse(
            embeddings=embeddings,
            usage={},
            model="dummy",
            response_time=0.0,
        )

    def embed_text(self, text):
        return self._embed(text)


def make_episode(user_id: str, title: str, content: str) -> Episode:
    return Episode(
        user_id=user_id,
        title=title,
        content=content,
        original_messages=[{"role": "user", "content": content}],
        message_count=1,
        boundary_reason="unit-test",
    )


class StorageAndSearchTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.base_path = Path(self.tmp_dir.name) / "memory"
        self.embedding_client = DummyEmbeddingClient()
        self.user_id = "tester"

        # Ensure FAISS is disabled to avoid dependency on native libraries
        from src.search import vector_search as vector_search_module

        self._faiss_patch = mock.patch.object(vector_search_module, "FAISS_AVAILABLE", False)
        self._faiss_patch.start()
        self.addCleanup(self._faiss_patch.stop)

    def _storage(self) -> EpisodeStorage:
        return EpisodeStorage(str(self.base_path))

    def test_episode_storage_persists_immediately(self):
        storage = self._storage()
        episode = make_episode(self.user_id, "Fruit chat", "Apple and banana conversation")
        episode_id = storage.save_episode(episode)

        jsonl_path = Path(storage.data_dir) / f"{self.user_id}_episodes.jsonl"
        self.assertTrue(jsonl_path.exists(), "Episode JSONL file should exist after save")

        with jsonl_path.open("r", encoding="utf-8") as handle:
            lines = [json.loads(line) for line in handle if line.strip()]

        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["episode_id"], episode_id)

        episodes = storage.get_user_episodes(self.user_id)
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0].episode_id, episode_id)

    def test_vector_search_alignment_with_storage(self):
        storage = self._storage()
        search = VectorSearch(self.embedding_client, str(self.base_path), self.embedding_client.dimension)

        ep1 = make_episode(self.user_id, "Fruit bowl", "Apple banana smoothie")
        ep2 = make_episode(self.user_id, "Veggies", "Carrot date salad")
        ep3 = make_episode(self.user_id, "Small talk", "Chatting about the weather")

        for episode in (ep1, ep2, ep3):
            storage.save_episode(episode)

        episodes = storage.get_user_episodes(self.user_id)
        search.index_episodes(self.user_id, episodes)

        embeddings_path = Path(self.base_path) / "episodes" / "vector_db" / f"{self.user_id}_embeddings.npy"
        self.assertTrue(embeddings_path.exists(), "Embeddings file missing after indexing")

        embeddings = np.load(embeddings_path)
        self.assertEqual(embeddings.shape, (len(episodes), self.embedding_client.dimension))

        jsonl_path = Path(self.base_path) / "episodes" / f"{self.user_id}_episodes.jsonl"
        with jsonl_path.open("r", encoding="utf-8") as handle:
            episode_count = sum(1 for line in handle if line.strip())
        self.assertEqual(episode_count, embeddings.shape[0], "Embedding count mismatch with JSONL records")

        results = search.search_episodes(self.user_id, "apple", top_k=2)
        self.assertTrue(results, "Vector search returned no results")
        self.assertEqual(results[0]["episode_id"], ep1.episode_id)
        self.assertEqual(results[0]["title"], ep1.title)

    def test_bm25_search_returns_expected_episode(self):
        storage = self._storage()
        bm25 = BM25Search(language="en")

        ep1 = make_episode(self.user_id, "Fruit bowl", "Apple banana smoothie")
        ep2 = make_episode(self.user_id, "Veggies", "Carrot date salad")
        ep3 = make_episode(self.user_id, "Small talk", "Chatting about the weather")

        for episode in (ep1, ep2, ep3):
            storage.save_episode(episode)

        episodes = storage.get_user_episodes(self.user_id)
        bm25.index_episodes(self.user_id, episodes)

        results = bm25.search_episodes(self.user_id, "carrot", top_k=2)
        self.assertTrue(results, "BM25 search returned no results")
        self.assertEqual(results[0]["episode_id"], ep2.episode_id)
        self.assertEqual(results[0]["title"], ep2.title)


if __name__ == "__main__":
    unittest.main()
