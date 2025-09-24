"""Add LoCoMo dataset conversations into Nemori memory using the simplified facade."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

from dotenv import load_dotenv
from tqdm import tqdm

from nemori import NemoriMemory, MemoryConfig
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor

load_dotenv()


def load_config(path: Path) -> MemoryConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    return MemoryConfig.from_dict(data)


def parse_timestamp(value: str) -> datetime:
    """Parse dataset timestamps such as "1:56 pm on 8 May, 2023"."""
    value = " ".join(value.split())
    if " on " not in value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return datetime.now()
    time_part, date_part = value.split(" on ")
    time_part = time_part.lower().strip()
    hour = 0
    minute = 0
    is_pm = "pm" in time_part
    time_part = time_part.replace("pm", "").replace("am", "").strip()
    if ":" in time_part:
        hour_str, minute_str = time_part.split(":", 1)
        hour = int(hour_str)
        minute = int(minute_str)
    else:
        hour = int(time_part)
    if is_pm and hour != 12:
        hour += 12
    if not is_pm and hour == 12:
        hour = 0
    months = {
        "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
        "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6,
        "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sep": 9,
        "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
    }
    parts = date_part.replace(",", "").split()
    day = 1
    month = 1
    year = datetime.now().year
    for part in parts:
        lower = part.lower()
        if lower in months:
            month = months[lower]
        elif part.isdigit():
            num = int(part)
            if num > 31:
                year = num
            else:
                day = num
    return datetime(year=year, month=month, day=day, hour=hour, minute=minute)


def build_messages(conversation: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten conversation sections into Nemori-compatible messages."""
    speaker_a = conversation.get("speaker_a", "speaker_a")
    speaker_b = conversation.get("speaker_b", "speaker_b")
    special_keys = {"speaker_a", "speaker_b"}
    messages: List[Dict[str, Any]] = []
    for key, chats in conversation.items():
        if key in special_keys or key.endswith("_date_time"):
            continue
        timestamp_raw = conversation.get(f"{key}_date_time")
        timestamp = parse_timestamp(timestamp_raw) if timestamp_raw else datetime.now()
        for chat in chats or []:
            speaker = chat.get("speaker", speaker_a)
            text = chat.get("text", "")
            parts = [text]
            if chat.get("blip_caption"):
                parts.append(f"[Image: {chat['blip_caption']}]")
            if chat.get("query"):
                parts.append(f"[Search: {chat['query']}]")
            role = speaker if speaker in (speaker_a, speaker_b) else "user"
            messages.append({
                "role": role,
                "content": " ".join(parts),
                "timestamp": timestamp.isoformat(),
                "metadata": {
                    "original_speaker": speaker,
                    "dataset_timestamp": timestamp_raw,
                    "blip_caption": chat.get("blip_caption"),
                    "search_query": chat.get("query"),
                },
            })
    return messages


def batched(iterable: Iterable[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    batch: List[Dict[str, Any]] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def process_single_conversation(
    config: MemoryConfig,
    item: Dict[str, Any],
    idx: int,
    batch_size: int,
    wait_timeout: float,
    tracker: Dict[str, Any],
) -> None:
    conversation = item.get("conversation", {})
    speaker_a = conversation.get("speaker_a", "speaker_a")
    user_id = f"{speaker_a}_{idx}"
    messages = build_messages(conversation)

    if not messages:
        print(f"[DEBUG] conversation {idx} for user {user_id} has 0 messages; no episode will be created")
    else:
        print(f"[DEBUG] conversation {idx} for user {user_id} has {len(messages)} messages")

    memory = NemoriMemory(config=config)
    try:
        for chunk in batched(messages, batch_size):
            memory.add_messages(user_id, chunk)
        episode_info = memory.flush(user_id)
        if episode_info is None:
            print(f"[DEBUG] conversation {idx} flush returned None for {user_id}")
        memory.wait_for_semantic(user_id, timeout=wait_timeout)
        episodes_dir = Path(config.storage_path) / "episodes"
        episodes_dir.mkdir(parents=True, exist_ok=True)
        file_path = episodes_dir / f"{user_id}_episodes.jsonl"
        print(f"[DEBUG] episode file for {user_id} exists after processing? {file_path.exists()}")
    finally:
        memory.close()
    tracker[user_id] = True


def process_dataset(
    config: MemoryConfig,
    dataset: List[Dict[str, Any]],
    batch_size: int,
    wait_timeout: float,
    max_workers: int,
) -> None:
    tracker: Dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_single_conversation,
                config,
                item,
                idx,
                batch_size,
                wait_timeout,
                tracker,
            )
            for idx, item in enumerate(dataset)
        ]
        for future in tqdm(futures, desc="Conversations"):
            future.result()


def main() -> None:
    parser = argparse.ArgumentParser(description="Add LoCoMo dataset into Nemori memory")
    parser.add_argument("--data", default="dataset/locomo10.json", help="Path to LoCoMo JSON dataset")
    parser.add_argument("--config", default="config.json", help="Path to MemoryConfig JSON")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--wait-timeout", type=float, default=1800.0)
    parser.add_argument("--max-workers", type=int, default=10)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path
    config = load_config(config_path)

    dataset_path = Path(args.data)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    process_dataset(config, dataset, args.batch_size, args.wait_timeout, args.max_workers)


if __name__ == "__main__":  # pragma: no cover
    main()
