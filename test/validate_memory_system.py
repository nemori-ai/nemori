#!/usr/bin/env python3
"""Utility script to validate episodic storage, vector search, and BM25 search."""

import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import Episode
from src.search import vector_search as vector_search_module
from src.search.bm25_search import BM25Search
from src.search.vector_search import VectorSearch
from src.storage.episode_storage import EpisodeStorage
from src.utils.embedding_client import EmbeddingResponse


class DummyEmbeddingClient:
    """Deterministic embedding client used for offline validation."""

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


def build_episode(user_id: str, title: str, content: str) -> Episode:
    return Episode(
        user_id=user_id,
        title=title,
        content=content,
        original_messages=[{"role": "user", "content": content}],
        message_count=1,
        boundary_reason="validation",
    )


def main() -> None:
    vector_search_module.FAISS_AVAILABLE = False

    tmp_dir = Path(tempfile.mkdtemp(prefix="nemori_memory_"))
    print(f"Working directory: {tmp_dir}")

    storage = EpisodeStorage(str(tmp_dir))
    embedding_client = DummyEmbeddingClient()
    vector = VectorSearch(embedding_client, str(tmp_dir), embedding_client.dimension)
    bm25 = BM25Search(language="en")

    user_id = "validator"
    episodes = [
        build_episode(user_id, "Fruit talk", "Apple banana smoothie"),
        build_episode(user_id, "Veggie chat", "Carrot date lunch"),
        build_episode(user_id, "Small talk", "Discussing the weather"),
    ]

    for episode in episodes:
        storage.save_episode(episode)

    stored = storage.get_user_episodes(user_id)
    print(f"Stored episodes: {len(stored)}")

    jsonl_path = tmp_dir / "episodes" / f"{user_id}_episodes.jsonl"
    with jsonl_path.open("r", encoding="utf-8") as handle:
        on_disk_records = [json.loads(line) for line in handle if line.strip()]
    print(f"JSONL records: {len(on_disk_records)}")

    vector.index_episodes(user_id, stored)
    embeddings_path = tmp_dir / "episodes" / "vector_db" / f"{user_id}_embeddings.npy"
    embeddings = np.load(embeddings_path)
    print(f"Embeddings shape: {embeddings.shape}")

    vector_results = vector.search_episodes(user_id, "apple", top_k=3)
    print("Vector search results (query='apple'):")
    if not vector_results:
        print("  (no results)")
    else:
        for item in vector_results:
            print(f"  - {item['episode_id']} | {item['title']} | score={item['score']:.4f}")

    bm25.index_episodes(user_id, stored)
    bm25_results = bm25.search_episodes(user_id, "carrot", top_k=2)
    print("BM25 search results (query='carrot'):")
    if not bm25_results:
        print("  (no results)")
    else:
        for item in bm25_results:
            print(f"  - {item['episode_id']} | {item['title']} | score={item['score']:.4f}")

    print("Validation complete. Temporary data retained for inspection.")


if __name__ == "__main__":
    main()
