"""Search LongMemEval memories using the Nemori facade."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI
from tqdm import tqdm

from nemori import MemoryConfig, NemoriMemory

load_dotenv()

ANSWER_PROMPT = Template(
    """
You are an intelligent assistant with access to episodic and semantic memories.

Question Date: {{ question_date }}
Question: {{ question }}

Episodic Memories:
{{ episodic }}

Semantic Memories:
{{ semantic }}

Answer in fewer than 6 words using only the information above.
"""
)


def load_config(path: Path) -> MemoryConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    return MemoryConfig.from_dict(data)


def format_memory_lines(memories: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for item in memories:
        timestamp = item.get("timestamp") or item.get("created_at") or ""
        content = item.get("content", "")
        lines.append(f"- [{timestamp}] {content}")
    return "\n".join(lines)


def flatten_results(memories: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    combined: List[Dict[str, Any]] = []
    for mem_type, items in memories.items():
        for item in items:
            combined.append(
                {
                    "type": mem_type,
                    "timestamp": item.get("timestamp", item.get("created_at")),
                    "content": item.get("content", item.get("summary")),
                    "score": item.get("score"),
                    "episode_id": item.get("episode_id"),
                    "memory_id": item.get("memory_id"),
                }
            )
    return combined


class LongMemEvalSearcher:
    def __init__(
        self,
        config: MemoryConfig,
        output_path: Path,
        top_k_episodes: int,
        top_k_semantic: int,
        search_method: str,
    ) -> None:
        self.memory = NemoriMemory(config=config)
        self.output_path = output_path
        self.top_k_episodes = top_k_episodes
        self.top_k_semantic = top_k_semantic
        self.search_method = search_method
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
        episodic = format_memory_lines(memories.get("episodic", []))
        semantic = format_memory_lines(memories.get("semantic", []))
        prompt = ANSWER_PROMPT.render(
            question=question,
            question_date=question_date,
            episodic=episodic,
            semantic=semantic,
        )
        response = self.openai_client.chat.completions.create(
            model=self.memory.config.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content if response and response.choices else ""

    def search(self, user_id: str, question: str) -> Dict[str, List[Dict[str, Any]]]:
        return self.memory.search(
            user_id,
            question,
            top_k_episodes=self.top_k_episodes,
            top_k_semantic=self.top_k_semantic,
            search_method=self.search_method,
        )

    def process(self, dataset: List[Dict[str, Any]]) -> None:
        for item in tqdm(dataset, desc="Questions"):
            question_id = item.get("question_id")
            user_id = f"question_{question_id}"
            question = item.get("question", "")
            question_date = item.get("question_date", "")
            memories = self.search(user_id, question)
            response = self.answer(question, question_date, memories)
            flattened = flatten_results(memories)
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
    )
    try:
        searcher.process(dataset)
        searcher.save()
    finally:
        searcher.close()


if __name__ == "__main__":  # pragma: no cover
    main()
