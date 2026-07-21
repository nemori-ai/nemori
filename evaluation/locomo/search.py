"""Search LoCoMo memories using the async Nemori facade."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from jinja2 import Template
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

from nemori import MemoryConfig, NemoriMemory
from evaluation.trace_utils import emit_trace, retrieval_snapshot

logger = logging.getLogger(__name__)

CHECKPOINT_REPLACE_ATTEMPTS = 8
CHECKPOINT_RETRY_BASE_SECONDS = 0.05

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
    valid_fields = {f.name for f in dataclasses.fields(MemoryConfig)}
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    return MemoryConfig(**filtered)


def format_memory_lines(
    memories: List[Dict[str, Any]],
    include_original_limit: int = 0,
) -> str:
    lines: List[str] = []
    for idx, item in enumerate(memories):
        timestamp = item.get("created_at") or ""
        content = item.get("content", "")
        lines.append(f"- [{timestamp}] {content}")
        if (
            include_original_limit > 0
            and idx < include_original_limit
            and item.get("source_messages")
        ):
            lines.append("    Source Messages:")
            for msg in item["source_messages"]:
                lines.append(f"    - {msg.get('role', 'user')}: {msg.get('content', '')}")
    return "\n".join(lines)


def flatten_results(
    memories: Dict[str, List[Dict[str, Any]]],
    include_original_limit: int,
) -> List[Dict[str, Any]]:
    combined: List[Dict[str, Any]] = []
    # Map new keys to type labels
    type_map = {"episodes": "episodic", "semantic_memories": "semantic"}
    for mem_key, items in memories.items():
        mem_type = type_map.get(mem_key, mem_key)
        for idx, item in enumerate(items):
            record = {
                "type": mem_type,
                "timestamp": item.get("created_at"),
                "content": item.get("content"),
                "score": item.get("score"),
                "episode_id": item.get("id"),
                "memory_id": item.get("id"),
            }
            if item.get("source_messages") and idx < include_original_limit:
                record["source_messages"] = item["source_messages"]
            combined.append(record)
    return combined


def questions_for_item(
    item: Dict[str, Any],
    *,
    scored_only: bool,
) -> List[Dict[str, Any]]:
    questions = item.get("qa", [])
    if not scored_only:
        return questions
    # LoCoMo category 5 contains adversarial/unanswerable questions and is not
    # part of the official 1,540-question score. Accept both numeric and string
    # category values used by different dataset serializations.
    return [qa for qa in questions if str(qa.get("category", "")).strip() != "5"]


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
        checkpoint_every: int = 0,
        max_concurrent_answer: int = 20,
        resume: bool = True,
        scored_only: bool = False,
    ) -> None:
        self.config = config
        self.output_path = output_path
        self.top_k_episodes = top_k_episodes
        self.top_k_semantic = top_k_semantic
        self.search_method = search_method
        self.include_original_messages_top_k = include_original_messages_top_k
        self.checkpoint_every = checkpoint_every
        self.max_concurrent_answer = max_concurrent_answer
        self.resume = resume
        self.scored_only = scored_only
        self.results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.completed_questions = 0
        self.total_questions = 0
        self.checkpoint_lock = asyncio.Lock()
        try:
            self.openai_client = AsyncOpenAI(
                api_key=config.llm_api_key,
                base_url=config.llm_base_url,
            )
        except Exception:
            self.openai_client = None

    async def answer(self, question: str, memories: Dict[str, List[Dict[str, Any]]]) -> str:
        if not self.openai_client:
            return ""
        episodic = format_memory_lines(
            memories.get("episodes", []),
            include_original_limit=self.include_original_messages_top_k,
        )
        semantic = format_memory_lines(
            memories.get("semantic_memories", []),
            include_original_limit=0,
        )
        prompt = ANSWER_PROMPT.render(
            question=question,
            episodic=episodic,
            semantic=semantic,
        )
        response = await self.openai_client.chat.completions.create(
            model=self.config.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content if response and response.choices else ""

    async def process(self, dataset: List[Dict[str, Any]], max_workers: int) -> None:
        semaphore = asyncio.Semaphore(max_workers)
        answer_semaphore = asyncio.Semaphore(self.max_concurrent_answer)
        self.total_questions = sum(
            len(questions_for_item(item, scored_only=self.scored_only))
            for item in dataset
        )
        self._restore_checkpoint(dataset)

        if self.completed_questions:
            print(
                "NEMORI_PROGRESS "
                + json.dumps({
                    "phase": "search",
                    "completed": self.completed_questions,
                    "total": self.total_questions,
                    "resumed": True,
                }),
                flush=True,
            )

        async def worker(memory: NemoriMemory, idx: int, item: Dict[str, Any]) -> None:
            async with semaphore:
                conversation = item.get("conversation", {})
                user_id = f"{conversation.get('speaker_a', 'speaker')}_{idx}"
                completed_for_item = len(self.results.get(str(idx), []))
                questions = questions_for_item(item, scored_only=self.scored_only)
                for question_index, qa in enumerate(questions):
                    if question_index < completed_for_item:
                        continue
                    question = qa.get("question", "")
                    memories = await memory.search(
                        user_id,
                        question,
                        top_k_episodes=self.top_k_episodes,
                        top_k_semantic=self.top_k_semantic,
                        search_method=self.search_method,
                    )
                    async with answer_semaphore:
                        response = await self.answer(question, memories)
                    emit_trace(
                        retrieval_snapshot(
                            question=question,
                            answer=response,
                            memories=memories,
                            method=self.search_method,
                            top_k_episodes=self.top_k_episodes,
                            top_k_semantic=self.top_k_semantic,
                            include_original_limit=self.include_original_messages_top_k,
                            question_id=f"{idx}:{question_index}",
                        )
                    )
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
                    await self.checkpoint()

        async with NemoriMemory(self.config) as memory:
            tasks = [worker(memory, idx, item) for idx, item in enumerate(dataset)]
            await tqdm.gather(*tasks, desc="Questions")

    async def checkpoint(self) -> None:
        async with self.checkpoint_lock:
            self.completed_questions += 1
            print(
                "NEMORI_PROGRESS "
                + json.dumps({
                    "phase": "search",
                    "completed": self.completed_questions,
                    "total": self.total_questions,
                }),
                flush=True,
            )
            if self.checkpoint_every and (
                self.completed_questions % self.checkpoint_every == 0
                or self.completed_questions == self.total_questions
            ):
                self.save()

    def save(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.output_path.parent,
                prefix=f"{self.output_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                tmp_path = Path(handle.name)
                json.dump(self.results, handle, indent=2, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            raise

        last_error: OSError | None = None
        for attempt in range(CHECKPOINT_REPLACE_ATTEMPTS):
            try:
                tmp_path.replace(self.output_path)
                self._cleanup_stale_temp_files()
                return
            except OSError as exc:
                last_error = exc
                if attempt + 1 < CHECKPOINT_REPLACE_ATTEMPTS:
                    delay = min(
                        CHECKPOINT_RETRY_BASE_SECONDS * (2**attempt),
                        1.0,
                    )
                    time.sleep(delay)

        # Keep the fully written temporary file: the next run can safely resume
        # from it, while the previous canonical checkpoint remains untouched.
        raise OSError(
            f"Could not replace checkpoint {self.output_path} after "
            f"{CHECKPOINT_REPLACE_ATTEMPTS} attempts; recovery data was kept at "
            f"{tmp_path}"
        ) from last_error

    def _temp_candidates(self) -> List[Path]:
        if not self.output_path.parent.exists():
            return []
        return list(self.output_path.parent.glob(f"{self.output_path.name}*.tmp"))

    def _cleanup_stale_temp_files(self) -> None:
        for candidate in self._temp_candidates():
            try:
                candidate.unlink(missing_ok=True)
            except OSError:
                # A stale recovery file is harmless and may itself be locked by
                # an antivirus/indexer on Windows. Leave it for a later save.
                logger.debug("Could not remove stale checkpoint %s", candidate)

    @staticmethod
    def _matching_checkpoint(
        raw_results: Any,
        dataset: List[Dict[str, Any]],
        *,
        scored_only: bool = False,
    ) -> Dict[str, List[Dict[str, Any]]] | None:
        """Return dataset-compatible result prefixes, or reject the checkpoint."""
        if not isinstance(raw_results, dict):
            return None

        expected_keys = {str(idx) for idx in range(len(dataset))}
        if any(str(key) not in expected_keys for key in raw_results):
            return None

        matched: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for idx, item in enumerate(dataset):
            records = raw_results.get(str(idx), [])
            questions = questions_for_item(item, scored_only=scored_only)
            if not isinstance(records, list) or len(records) > len(questions):
                return None
            for question_index, record in enumerate(records):
                if not isinstance(record, dict):
                    return None
                expected_question = questions[question_index].get("question", "")
                if record.get("question") != expected_question:
                    return None
            if records:
                matched[str(idx)].extend(records)
        return matched

    def _restore_checkpoint(self, dataset: List[Dict[str, Any]]) -> None:
        self.results = defaultdict(list)
        self.completed_questions = 0
        if not self.resume:
            return

        candidates = [self.output_path, *self._temp_candidates()]
        best_path: Path | None = None
        best_results: Dict[str, List[Dict[str, Any]]] | None = None
        best_count = -1
        best_is_canonical = False

        for candidate in candidates:
            if not candidate.is_file():
                continue
            try:
                raw_results = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                logger.warning("Ignoring unreadable checkpoint %s: %s", candidate, exc)
                continue
            matched = self._matching_checkpoint(
                raw_results,
                dataset,
                scored_only=self.scored_only,
            )
            if matched is None:
                logger.warning("Ignoring checkpoint for a different dataset: %s", candidate)
                continue
            count = sum(len(records) for records in matched.values())
            is_canonical = candidate == self.output_path
            if count > best_count or (
                count == best_count and is_canonical and not best_is_canonical
            ):
                best_path = candidate
                best_results = matched
                best_count = count
                best_is_canonical = is_canonical

        if best_results is None:
            return

        self.results = defaultdict(list, best_results)
        self.completed_questions = best_count
        logger.info(
            "Resuming LoCoMo search from %s (%d/%d questions)",
            best_path,
            self.completed_questions,
            self.total_questions,
        )


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Search LoCoMo memories")
    parser.add_argument("--data", default="dataset/locomo10.json", help="Path to LoCoMo dataset")
    parser.add_argument("--config", default="config.json", help="Path to MemoryConfig JSON")
    parser.add_argument("--output", default="locomo/results.json")
    parser.add_argument("--top-k-episodes", type=int, default=10)
    parser.add_argument("--top-k-semantic", type=int, default=20)
    parser.add_argument("--search-method", default="vector", choices=["hybrid", "vector", "text"])
    parser.add_argument("--include-original-messages-top-k", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=50)
    parser.add_argument(
        "--max-concurrent-answer",
        type=int,
        default=None,
        help="Maximum concurrent answer-generation requests; defaults to --max-workers",
    )
    parser.add_argument(
        "--scored-only",
        action="store_true",
        help="Process only official scored questions (exclude category 5)",
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Resume from the newest valid checkpoint (default)",
    )
    resume_group.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore existing checkpoints and start from the first question",
    )
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
        checkpoint_every=args.checkpoint_every,
        max_concurrent_answer=max(1, args.max_concurrent_answer or args.max_workers),
        resume=args.resume,
        scored_only=args.scored_only,
    )
    asyncio.run(searcher.process(dataset, max_workers=args.max_workers))
    searcher.save()


if __name__ == "__main__":  # pragma: no cover
    main()
