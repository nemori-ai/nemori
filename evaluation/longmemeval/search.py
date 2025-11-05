"""Search LongMemEval memories using the Nemori facade."""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI
from tqdm import tqdm

from nemori import MemoryConfig, NemoriMemory

logger = logging.getLogger(__name__)

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
    Question Date: {{ question_date }}

    Answer:
    """
)


def load_config(path: Path) -> MemoryConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    return MemoryConfig.from_dict(data)


def format_memory_lines(
    memories: List[Dict[str, Any]], include_original_limit: int = 0
) -> str:
    lines: List[str] = []
    for idx, item in enumerate(memories):
        timestamp = item.get("timestamp") or item.get("created_at") or ""
        content = item.get("content", "")
        lines.append(f"- [{timestamp}] {content}")
        if (
            include_original_limit > 0
            and idx < include_original_limit
            and item.get("original_messages")
        ):
            lines.append("    Original Messages:")
            for msg in item["original_messages"]:
                lines.append(
                    f"    • {msg.get('role', 'user')}: {msg.get('content', '')}"
                )
    return "\n".join(lines)


def flatten_results(
    memories: Dict[str, List[Dict[str, Any]]], include_original_limit: int
) -> List[Dict[str, Any]]:
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


class LongMemEvalSearcher:
    def __init__(
        self,
        config: MemoryConfig,
        output_path: Path,
        top_k_episodes: int,
        top_k_semantic: int,
        search_method: str,
        include_original_messages_top_k: int,
    ) -> None:
        self.memory = NemoriMemory(config=config)
        self.output_path = output_path
        self.top_k_episodes = top_k_episodes
        self.top_k_semantic = top_k_semantic
        self.search_method = search_method
        self.include_original_messages_top_k = include_original_messages_top_k
        self.results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        try:
            self.openai_client = OpenAI()
        except Exception:
            self.openai_client = None

    def close(self) -> None:
        self.memory.close()

    def answer(self, question: str, question_date: str, memories: Dict[str, List[Dict[str, Any]]]) -> str:
        if not self.openai_client:
            return ""
        episodic = format_memory_lines(
            memories.get("episodic", []),
            include_original_limit=self.include_original_messages_top_k,
        )
        semantic = format_memory_lines(memories.get("semantic", []), include_original_limit=0)
        prompt = ANSWER_PROMPT.render(
            question=question,
            question_date=question_date,
            episodic=episodic,
            semantic=semantic,
        )
        
        # Check if using reasoning model (gpt-5, o1, etc.) that only supports temperature=1
        reasoning_models = ['gpt-5', 'o1', 'o1-mini', 'o1-preview', 'o3', 'o3-mini']
        is_reasoning_model = any(self.memory.config.llm_model.lower().startswith(prefix) for prefix in reasoning_models)
        
        # Build parameters - skip temperature for reasoning models
        params = {
            "model": self.memory.config.llm_model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if not is_reasoning_model:
            params["temperature"] = 0.0
        
        response = self.openai_client.chat.completions.create(**params)
        return response.choices[0].message.content if response and response.choices else ""

    def search(self, user_id: str, question: str) -> Dict[str, List[Dict[str, Any]]]:
        return self.memory.search(
            user_id,
            question,
            top_k_episodes=self.top_k_episodes,
            top_k_semantic=self.top_k_semantic,
            search_method=self.search_method,
        )

    def _hydrate_episode_results(
        self, user_id: str, episodes: List[Dict[str, Any]]
    ) -> None:
        if not episodes:
            return

        memory_system = getattr(self.memory, "_memory_system", None)
        if memory_system is None:
            return

        repository = getattr(memory_system, "_episode_repository", None)
        episode_cache: Dict[str, Any] = {}

        for idx, item in enumerate(episodes):
            metadata = item.get("metadata") or {}
            if "timestamp" not in item and metadata.get("timestamp"):
                item["timestamp"] = metadata["timestamp"]
            if "created_at" not in item and metadata.get("created_at"):
                item["created_at"] = metadata["created_at"]

            if (
                idx < self.include_original_messages_top_k
                and not item.get("original_messages")
            ):
                episode_id = item.get("episode_id") or metadata.get("episode_id")
                if not episode_id or repository is None:
                    continue

                episode_obj = None
                try:
                    if hasattr(repository, "get_episode"):
                        episode_obj = repository.get_episode(episode_id, user_id)  # type: ignore[attr-defined]
                    else:
                        if not episode_cache:
                            all_eps = repository.list_by_user(user_id)
                            episode_cache = {ep.episode_id: ep for ep in all_eps}
                        episode_obj = episode_cache.get(episode_id)
                except Exception as exc:  # pragma: no cover
                    logger.debug(
                        "Hydration failed for episode %s (user %s): %s",
                        episode_id,
                        user_id,
                        exc,
                    )
                    continue

                if episode_obj is None:
                    continue

                item.setdefault("original_messages", episode_obj.original_messages)
                item.setdefault("timestamp", episode_obj.timestamp.isoformat())
                item.setdefault("created_at", episode_obj.created_at.isoformat())
                item.setdefault("content", episode_obj.content)
                item.setdefault("title", episode_obj.title)

    def _hydrate_semantic_results(self, semantic_results: List[Dict[str, Any]]) -> None:
        for item in semantic_results:
            metadata = item.get("metadata") or {}
            if "timestamp" not in item and metadata.get("timestamp"):
                item["timestamp"] = metadata["timestamp"]
            if "created_at" not in item and metadata.get("created_at"):
                item["created_at"] = metadata["created_at"]

    def process(self, dataset: List[Dict[str, Any]]) -> None:
        for item in tqdm(dataset, desc="Questions"):
            question_id = item.get("question_id")
            user_id = f"question_{question_id}"
            question = item.get("question", "")
            question_date = item.get("question_date", "")
            memories = self.search(user_id, question)
            self._hydrate_episode_results(user_id, memories.get("episodic", []))
            self._hydrate_semantic_results(memories.get("semantic", []))
            response = self.answer(question, question_date, memories)
            flattened = flatten_results(memories, self.include_original_messages_top_k)
            record = {
                "question_id": question_id,
                "question": question,
                "question_date": question_date,
                "response": response,
                "memories": flattened,
                "search_method": self.search_method,
            }
            self.results[str(question_id)].append(record)

    def save(self) -> None:
        self.output_path.write_text(json.dumps(self.results, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search LongMemEval memories")
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--output", default="longmemeval_results.json")
    parser.add_argument("--top-k-episodes", type=int, default=10)
    parser.add_argument("--top-k-semantic", type=int, default=10)
    parser.add_argument("--search-method", default="vector", choices=["vector", "bm25", "hybrid"])
    parser.add_argument(
        "--include-original-messages-top-k", type=int, default=0,
        help="Number of top episodic memories to attach original messages for",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path
    config = load_config(config_path)

    dataset_path = Path(args.data)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))

    searcher = LongMemEvalSearcher(
        config=config,
        output_path=Path(args.output),
        top_k_episodes=args.top_k_episodes,
        top_k_semantic=args.top_k_semantic,
        search_method=args.search_method,
        include_original_messages_top_k=args.include_original_messages_top_k,
    )
    try:
        searcher.process(dataset)
        searcher.save()
    finally:
        searcher.close()


if __name__ == "__main__":  # pragma: no cover
    main()
