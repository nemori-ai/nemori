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
import threading
import logging

load_dotenv()

# Suppress verbose logging from libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)


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
    tracker_lock: threading.Lock,
    position: int = 0,
    verbose: bool = False,
) -> None:
    conversation = item.get("conversation", {})
    speaker_a = conversation.get("speaker_a", "speaker_a")
    user_id = f"{speaker_a}_{idx}"
    messages = build_messages(conversation)

    # Track result
    result = {"user_id": user_id, "success": False, "message_count": len(messages), "error": None}
    
    if not messages:
        result["error"] = "No messages"
        with tracker_lock:
            tracker[user_id] = result
        return

    memory = NemoriMemory(config=config)
    
    # Create progress bar for this user's message processing
    chunks = list(batched(messages, batch_size))
    total_chunks = len(chunks)
    
    # Use a progress bar if verbose mode is enabled and we have messages to process
    if verbose and (total_chunks > 1 or len(messages) > 20):
        pbar = tqdm(
            total=len(messages),
            desc=f"  [{idx+1:2d}] {speaker_a}",
            position=position + 1,
            leave=False,
            bar_format="{desc:15s}: {percentage:3.0f}%|{bar:30}| {n_fmt:>4}/{total_fmt:<4} msgs",
            ncols=80,
            colour='cyan',
            disable=False  # Ensure it's enabled
        )
    else:
        pbar = None
    
    try:
        # Add messages in batches with progress tracking
        messages_processed = 0
        for chunk in chunks:
            memory.add_messages(user_id, chunk)
            messages_processed += len(chunk)
            if pbar:
                pbar.update(len(chunk))
        
        if pbar:
            pbar.close()
        
        # Create episode
        episode_info = memory.flush(user_id)
        if episode_info is None:
            result["error"] = "Failed to create episode"
            with tracker_lock:
                tracker[user_id] = result
            return
            
        # Wait for semantic generation
        memory.wait_for_semantic(user_id, timeout=wait_timeout)
        
        # Verify file creation
        episodes_dir = Path(config.storage_path) / "episodes"
        file_path = episodes_dir / f"{user_id}_episodes.jsonl"
        
        if file_path.exists():
            result["success"] = True
        else:
            result["error"] = "Episode file not found after processing"
            
    except Exception as e:
        result["error"] = str(e)
        if pbar:
            pbar.close()
    finally:
        memory.close()
        with tracker_lock:
            tracker[user_id] = result


def process_dataset(
    config: MemoryConfig,
    dataset: List[Dict[str, Any]],
    batch_size: int,
    wait_timeout: float,
    max_workers: int,
    verbose: bool = False,
) -> None:
    tracker: Dict[str, Any] = {}
    tracker_lock = threading.Lock()
    
    print(f"\n🚀 Processing {len(dataset)} conversations with {max_workers} workers...")
    
    if verbose:
        print("─" * 80)
        # Pre-calculate total messages for each user
        user_info = []
        total_messages = 0
        for idx, item in enumerate(dataset):
            conversation = item.get("conversation", {})
            speaker_a = conversation.get("speaker_a", "speaker_a")
            messages = build_messages(conversation)
            user_info.append({
                "user_id": f"{speaker_a}_{idx}",
                "speaker": speaker_a,
                "message_count": len(messages)
            })
            total_messages += len(messages)
        
        # Display user overview
        print("📋 User Overview:")
        for info in user_info[:5]:  # Show first 5 users
            print(f"  • {info['user_id']}: {info['message_count']} messages")
        if len(user_info) > 5:
            print(f"  ... and {len(user_info) - 5} more users")
        print(f"📈 Total messages to process: {total_messages}")
        print("─" * 80 + "\n")
    else:
        print("")  # Just add a blank line for cleaner output
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create futures with position information
        futures_with_info = []
        for idx, item in enumerate(dataset):
            future = executor.submit(
                process_single_conversation,
                config,
                item,
                idx,
                batch_size,
                wait_timeout,
                tracker,
                tracker_lock,
                position=idx if idx < max_workers else idx % max_workers,  # Reuse positions for overflow
                verbose=verbose,
            )
            futures_with_info.append((future, idx, item))
        
        # Main progress bar
        with tqdm(total=len(futures_with_info), 
                  desc="Overall", 
                  position=0,
                  bar_format="{desc:10s}: {percentage:3.0f}%|{bar:40}| {n_fmt:>2}/{total_fmt:<2} [{elapsed}<{remaining}]",
                  ncols=80,
                  colour='green') as main_pbar:
            
            for future, idx, item in futures_with_info:
                try:
                    future.result()
                except Exception as e:
                    # Silently handle exceptions (they're tracked in results)
                    pass
                main_pbar.update(1)
        
        # Clear the screen a bit for cleaner output
        print("\n" * 2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Processing Summary")
    print("=" * 60)
    
    successful = sum(1 for r in tracker.values() if r.get("success", False))
    failed = len(tracker) - successful
    
    print(f"✅ Successful: {successful}/{len(tracker)}")
    print(f"❌ Failed: {failed}/{len(tracker)}")
    
    if failed > 0:
        print("\n🔍 Failure Details:")
        for user_id, result in tracker.items():
            if not result.get("success", False):
                error = result.get("error", "Unknown error")
                msg_count = result.get("message_count", 0)
                print(f"  - {user_id}: {error} (messages: {msg_count})")
    
    print("\n" + "=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add LoCoMo dataset into Nemori memory")
    parser.add_argument("--data", default="dataset/locomo10.json", help="Path to LoCoMo JSON dataset")
    parser.add_argument("--config", default="config.json", help="Path to MemoryConfig JSON")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--wait-timeout", type=float, default=1800.0)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--verbose", action="store_true", help="Show detailed progress for each user")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path
    config = load_config(config_path)

    dataset_path = Path(args.data)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    process_dataset(config, dataset, args.batch_size, args.wait_timeout, args.max_workers, args.verbose)


if __name__ == "__main__":  # pragma: no cover
    main()
