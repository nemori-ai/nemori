#!/usr/bin/env python3
"""
Rebuild FAISS/embedding artifacts for episodic and semantic memories.
"""

import argparse
import json
import shutil
from pathlib import Path
import sys
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import MemoryConfig
from src.models import Episode, SemanticMemory
from src.search.vector_search import VectorSearch
from src.utils.embedding_client import EmbeddingClient


def load_jsonl(path: Path, model_cls):
    """Load JSONL file as dataclass instances."""
    items = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(model_cls.from_dict(json.loads(line)))
    return items


def reset_vector_dir(base: Path) -> None:
    """Clear and recreate vector_db directory under base."""
    vector_dir = base / "vector_db"
    if vector_dir.exists():
        shutil.rmtree(vector_dir)
    vector_dir.mkdir(parents=True, exist_ok=True)


def rebuild(storage_root: Path) -> None:
    if not storage_root.exists():
        raise SystemExit(f"Storage path not found: {storage_root}")

    load_dotenv()

    embedding_client = EmbeddingClient(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="text-embedding-3-small"
    )
    vector_search = VectorSearch(
        embedding_client=embedding_client,
        storage_path=str(storage_root),
        dimension=1536
    )

    episodes_dir = storage_root / "episodes"
    semantic_dir = storage_root / "semantic"

    reset_vector_dir(episodes_dir)
    reset_vector_dir(semantic_dir)

    print("Re-indexing episodic memories...")
    for jsonl_path in sorted(episodes_dir.glob("*_episodes.jsonl")):
        user_id = jsonl_path.stem.replace("_episodes", "")
        episodes = load_jsonl(jsonl_path, Episode)
        if not episodes:
            print(f"  [skip] {user_id}: 0 episodes")
            continue
        vector_search.index_episodes(user_id, episodes)
        print(f"  [done] {user_id}: {len(episodes)} episodes")

    print("\nRe-indexing semantic memories...")
    for jsonl_path in sorted(semantic_dir.glob("*_semantic.jsonl")):
        user_id = jsonl_path.stem.replace("_semantic", "")
        memories = load_jsonl(jsonl_path, SemanticMemory)
        if not memories:
            print(f"  [skip] {user_id}: 0 semantic memories")
            continue
        vector_search.index_semantic_memories(user_id, memories)
        print(f"  [done] {user_id}: {len(memories)} semantic memories")

    print("\nRebuild completed.")


def main():
    parser = argparse.ArgumentParser(description="Rebuild vector indices for Nemori memories")
    parser.add_argument(
        "storage",
        nargs="?",
        default="evaluation/memories_1",
        help="Root directory containing episodes/ and semantic/"
    )
    args = parser.parse_args()
    rebuild(Path(args.storage).resolve())


if __name__ == "__main__":
    main()
