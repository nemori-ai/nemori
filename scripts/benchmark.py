"""Simple benchmarking harness for Nemori memory operations."""

from __future__ import annotations

import argparse
import time
from datetime import datetime

from nemori import NemoriMemory, MemoryConfig


def run_benchmark(message_count: int, repetitions: int, backend: str) -> None:
    config = MemoryConfig(
        storage_backend=backend,
        vector_index_backend="memory" if backend == "memory" else "chroma",
        lexical_index_backend="memory" if backend == "memory" else "bm25",
    )

    with NemoriMemory(config=config) as memory:
        user_id = "benchmark"
        start = time.perf_counter()
        for iteration in range(repetitions):
            batch = [
                {
                    "role": "user" if i % 2 == 0 else "assistant",
                    "content": f"Benchmark message {iteration}-{i}",
                    "timestamp": datetime.now().isoformat(),
                }
                for i in range(message_count)
            ]
            memory.add_messages(user_id, batch)
            memory.flush(user_id)
        memory.wait_for_semantic(user_id, timeout=30)
        duration = time.perf_counter() - start
        print(f"Processed {repetitions} batches in {duration:.2f}s (backend={backend})")


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="Nemori benchmark")
    parser.add_argument("--message-count", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--backend", type=str, default="filesystem", choices=["filesystem", "memory"])
    args = parser.parse_args()
    run_benchmark(args.message_count, args.repetitions, args.backend)
