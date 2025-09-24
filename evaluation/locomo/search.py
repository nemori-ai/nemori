"""Search LoCoMo memories using the Nemori facade with concurrent execution."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List
import threading

from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI
from tqdm import tqdm

from nemori import MemoryConfig, NemoriMemory

load_dotenv()

ANSWER_PROMPT = Template(
    """
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

    # CONTEXT:
    You have access to memories from two speakers in a conversation. These memories contain
    timestamped information that may be relevant to answering the question.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories from both speakers
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. If there is a question about time references (like "last year", "two months ago", etc.),
       calculate the actual date based on the memory timestamp. For example, if a memory from
       4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
    6. Always convert relative time references to specific dates, months, or years. For example,
       convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory
       timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the memories from both speakers. Do not confuse character
       names mentioned in memories with the actual users who created those memories.
    8. The answer should be less than 5-6 words.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the memories
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

    Episodic Memories:
    {{ episodic }}

    Semantic Memories:
    {{ semantic }}

    Question: {{ question }}

    Answer:
    """
)


def load_config(path: Path) -> MemoryConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    return MemoryConfig.from_dict(data)


def format_memory_lines(memories: List[Dict[str, Any]], include_original: bool = False) -> str:
    lines: List[str] = []
    for item in memories:
        timestamp = item.get("timestamp") or item.get("created_at") or ""
        content = item.get("content", "")
        lines.append(f"- [{timestamp}] {content}")
        if include_original and item.get("original_messages"):
            for msg in item["original_messages"]:
                lines.append(f"    • {msg.get('role', 'user')}: {msg.get('content', '')}")
    return "\n".join(lines)


def flatten_results(memories: Dict[str, List[Dict[str, Any]]], include_original_limit: int) -> List[Dict[str, Any]]:
    combined: List[Dict[str, Any]] = []
    for mem_type, items in memories.items():
        for idx, item in enumerate(items):
            record = {
                "type": mem_type,
                "timestamp": item.get("timestamp", item.get("created_at")),
                "content": item.get("content", item.get("summary")),
                "score": item.get("score"),
                "episode_id": item.get("episode_id"),
                "memory_id": item.get("memory_id"),
            }
            if item.get("original_messages") and idx < include_original_limit:
                record["original_messages"] = item["original_messages"]
            combined.append(record)
    return combined


class LocomoSearcher:
    def __init__(
        self,
        config: MemoryConfig,
        *,
        output_path: Path,
        top_k_episodes: int,
        top_k_semantic: int,
        search_method: str,
        include_original_messages_top_k: int,
    ) -> None:
        self.config = config
        self.output_path = output_path
        self.top_k_episodes = top_k_episodes
        self.top_k_semantic = top_k_semantic
        self.search_method = search_method
        self.include_original_messages_top_k = include_original_messages_top_k
        self.results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._memory_build_lock = threading.Lock()
        try:
            self.openai_client = OpenAI()
        except Exception:
            self.openai_client = None

    def close(self) -> None:
        pass  # Each worker creates its own NemoriMemory instance

    def answer(self, question: str, memories: Dict[str, List[Dict[str, Any]]]) -> str:
        if not self.openai_client:
            return ""
        episodic = format_memory_lines(memories.get("episodic", []), include_original=True)
        semantic = format_memory_lines(memories.get("semantic", []), include_original=False)
        prompt = ANSWER_PROMPT.render(
            question=question,
            episodic=episodic,
            semantic=semantic,
        )
        response = self.openai_client.chat.completions.create(
            model=self.config.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content if response and response.choices else ""

    def search(self, memory: NemoriMemory, user_id: str, question: str) -> Dict[str, List[Dict[str, Any]]]:
        return memory.search(
            user_id,
            question,
            top_k_episodes=self.top_k_episodes,
            top_k_semantic=self.top_k_semantic,
            search_method=self.search_method,
        )

    def process(self, dataset: List[Dict[str, Any]], max_workers: int) -> None:
        def worker(idx: int, item: Dict[str, Any]) -> None:
            with self._memory_build_lock:
                memory = NemoriMemory(config=self.config)
            try:
                conversation = item.get("conversation", {})
                user_id = f"{conversation.get('speaker_a', 'speaker')}_{idx}"
                for qa in item.get("qa", []):
                    question = qa.get("question", "")
                    memories = self.search(memory, user_id, question)
                    response = self.answer(question, memories)
                    flattened = flatten_results(memories, self.include_original_messages_top_k)
                    record = {
                        "question": question,
                        "answer": qa.get("answer"),
                        "category": qa.get("category"),
                        "evidence": qa.get("evidence", []),
                        "response": response,
                        "memories": flattened,
                        "search_method": self.search_method,
                    }
                    self.results[str(idx)].append(record)
            finally:
                memory.close()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(worker, idx, item) for idx, item in enumerate(dataset)]
            for future in tqdm(futures, desc="Questions"):
                future.result()

    def save(self) -> None:
        self.output_path.write_text(json.dumps(self.results, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search LoCoMo memories")
    parser.add_argument("--data", default="dataset/locomo10.json", help="Path to LoCoMo dataset")
    parser.add_argument("--config", default="config.json", help="Path to MemoryConfig JSON")
    parser.add_argument("--output", default="locomo/results.json")
    parser.add_argument("--top-k-episodes", type=int, default=10)
    parser.add_argument("--top-k-semantic", type=int, default=20)
    parser.add_argument("--search-method", default="vector", choices=["hybrid", "vector", "bm25"])
    parser.add_argument("--include-original-messages-top-k", type=int, default=2)
    parser.add_argument("--max-workers", type=int, default=50)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path
    config = load_config(config_path)

    dataset_path = Path(args.data)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))

    searcher = LocomoSearcher(
        config=config,
        output_path=Path(args.output),
        top_k_episodes=args.top_k_episodes,
        top_k_semantic=args.top_k_semantic,
        search_method=args.search_method,
        include_original_messages_top_k=args.include_original_messages_top_k,
    )
    try:
        searcher.process(dataset, max_workers=args.max_workers)
        searcher.save()
    finally:
        searcher.close()


if __name__ == "__main__":  # pragma: no cover
    main()
