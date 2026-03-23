"""Add LongMemEval dataset conversations into Nemori memory using the async facade."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from tqdm.asyncio import tqdm

from nemori import NemoriMemory, MemoryConfig

load_dotenv()

logger = logging.getLogger(__name__)

# Suppress verbose logging from libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


def load_config(path: Path) -> MemoryConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    valid_fields = {f.name for f in dataclasses.fields(MemoryConfig)}
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    return MemoryConfig(**filtered)


async def process_question(
    memory: NemoriMemory,
    question_data: Dict[str, Any],
    question_idx: int,
    batch_size: int,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    """Process a single LongMemEval question: add all sessions then flush."""
    question_id = question_data["question_id"]
    user_id = f"question_{question_id}"

    result: Dict[str, Any] = {
        "question_id": question_id,
        "success": False,
        "episodes_created": 0,
        "total_messages": 0,
        "error": None,
    }

    async with semaphore:
        try:
            # Clean up any prior data for this question
            await memory.delete_user(user_id)

            haystack_sessions = question_data["haystack_sessions"]
            haystack_dates = question_data["haystack_dates"]
            total_messages = 0

            # Process each session
            for session, date in zip(haystack_sessions, haystack_dates):
                messages: List[Dict[str, Any]] = []
                for turn in session:
                    # Parse timestamp
                    try:
                        timestamp = datetime.strptime(date, "%Y/%m/%d (%a) %H:%M")
                    except Exception:
                        timestamp = datetime.now()

                    messages.append({
                        "role": turn["role"],
                        "content": turn["content"],
                        "timestamp": timestamp.isoformat(),
                    })

                # Add messages in batches
                for i in range(0, len(messages), batch_size):
                    batch = messages[i : i + batch_size]
                    await memory.add_messages(user_id, batch)
                    total_messages += len(batch)

            # Flush to create episodes (triggers semantic generation via event bus)
            episodes = await memory.flush(user_id)

            result["success"] = True
            result["episodes_created"] = len(episodes)
            result["total_messages"] = total_messages

        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"
            logger.error("Error processing question %s: %s", question_id, e)

    return result


async def process_dataset(
    config: MemoryConfig,
    dataset: List[Dict[str, Any]],
    batch_size: int,
    max_workers: int,
    start_idx: int = 0,
    end_idx: int | None = None,
) -> None:
    if end_idx is None:
        end_idx = len(dataset)
    questions = dataset[start_idx:end_idx]

    print(f"\nProcessing {len(questions)} questions (idx {start_idx}-{end_idx}) with concurrency={max_workers}...")

    semaphore = asyncio.Semaphore(max_workers)

    async with NemoriMemory(config) as memory:
        tasks = [
            process_question(memory, q, start_idx + i, batch_size, semaphore)
            for i, q in enumerate(questions)
        ]
        results = await tqdm.gather(*tasks, desc="Questions")

    # Print summary
    successful = sum(1 for r in results if r.get("success"))
    failed = len(results) - successful
    total_episodes = sum(r.get("episodes_created", 0) for r in results)
    total_messages = sum(r.get("total_messages", 0) for r in results)

    print(f"\n{'=' * 60}")
    print("Processing Summary")
    print(f"{'=' * 60}")
    print(f"Questions processed: {successful}/{len(questions)}")
    print(f"Failed: {failed}")
    print(f"Total episodes created: {total_episodes}")
    print(f"Total messages processed: {total_messages}")

    if failed > 0:
        print("\nFailure Details:")
        for r in results:
            if not r.get("success"):
                print(f"  - question_{r['question_id']}: {r.get('error', 'Unknown')}")

    print(f"{'=' * 60}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Add LongMemEval data to Nemori memory")
    parser.add_argument("--data_path", type=str, default="./dataset/longmemeval_s.json",
                        help="Path to longmemeval_s.json")
    parser.add_argument("--config", type=str, default="config.json",
                        help="Path to MemoryConfig JSON")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for adding messages")
    parser.add_argument("--max_workers", type=int, default=10,
                        help="Maximum concurrency")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path
    config = load_config(config_path)

    dataset = json.loads(Path(args.data_path).read_text(encoding="utf-8"))
    asyncio.run(process_dataset(
        config, dataset, args.batch_size, args.max_workers,
        args.start_idx, args.end_idx,
    ))


if __name__ == "__main__":  # pragma: no cover
    main()
