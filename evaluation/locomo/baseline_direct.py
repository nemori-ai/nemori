"""Direct LoCoMo baseline without Nemori memory construction or retrieval."""

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
from typing import Any

from dotenv import load_dotenv
from jinja2 import Template
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

from nemori import MemoryConfig
from evaluation.locomo.add import build_messages
from evaluation.locomo.search import questions_for_item
from evaluation.trace_utils import emit_trace

logger = logging.getLogger(__name__)
CHECKPOINT_REPLACE_ATTEMPTS = 8
CHECKPOINT_RETRY_BASE_SECONDS = 0.05

DIRECT_PROMPT = Template(
    """
    You are answering questions about a long conversation.

    Use only the transcript below. Pay close attention to speakers and timestamps.
    Convert relative time references into specific dates when possible.
    If the answer is not present, say that it is not mentioned.
    Keep the final answer concise, usually less than 5-6 words.

    Transcript:
    {{ transcript }}

    Question: {{ question }}

    Answer:
    """
)


def load_config(path: Path) -> MemoryConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    valid_fields = {f.name for f in dataclasses.fields(MemoryConfig)}
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    return MemoryConfig(**filtered)


def transcript_for_item(item: dict[str, Any], max_chars: int = 0) -> str:
    messages = build_messages(item.get("conversation", {}))
    lines = [
        f"[{message.get('timestamp') or ''}] {message.get('role', 'speaker')}: {message.get('content', '')}"
        for message in messages
    ]
    transcript = "\n".join(lines)
    if max_chars > 0 and len(transcript) > max_chars:
        return transcript[-max_chars:]
    return transcript


class DirectLocomoBaseline:
    def __init__(
        self,
        config: MemoryConfig,
        *,
        output_path: Path,
        checkpoint_every: int,
        max_concurrent_answer: int,
        scored_only: bool,
        resume: bool,
        max_transcript_chars: int,
    ) -> None:
        self.config = config
        self.output_path = output_path
        self.checkpoint_every = checkpoint_every
        self.max_concurrent_answer = max_concurrent_answer
        self.scored_only = scored_only
        self.resume = resume
        self.max_transcript_chars = max_transcript_chars
        self.results: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.completed_questions = 0
        self.total_questions = 0
        self.checkpoint_lock = asyncio.Lock()
        self.openai_client = AsyncOpenAI(
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
            timeout=config.llm_timeout,
        )

    async def answer(self, transcript: str, question: str) -> str:
        response = await self.openai_client.chat.completions.create(
            model=self.config.llm_model,
            messages=[
                {
                    "role": "user",
                    "content": DIRECT_PROMPT.render(transcript=transcript, question=question),
                }
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content if response and response.choices else ""

    async def process(self, dataset: list[dict[str, Any]]) -> None:
        self.total_questions = sum(
            len(questions_for_item(item, scored_only=self.scored_only))
            for item in dataset
        )
        self._restore_checkpoint(dataset)
        if self.completed_questions:
            self._progress(resumed=True)

        queue: asyncio.Queue[tuple[int, dict[str, Any], int, dict[str, Any], str]] = asyncio.Queue()
        for idx, item in enumerate(dataset):
            transcript = transcript_for_item(item, self.max_transcript_chars)
            completed_for_item = len(self.results.get(str(idx), []))
            for question_index, qa in enumerate(questions_for_item(item, scored_only=self.scored_only)):
                if question_index >= completed_for_item:
                    queue.put_nowait((idx, item, question_index, qa, transcript))

        async def worker() -> None:
            while True:
                try:
                    idx, _item, question_index, qa, transcript = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    question = qa.get("question", "")
                    response = await self.answer(transcript, question)
                    emit_trace(
                        {
                            "event": "search_complete",
                            "flow": "search",
                            "nodes": ["question", "answer_prompt", "answer", "checkpoint"],
                            "summary": f"裸上下文回答：{len(transcript)} 字符 → {response[:80] or '空回答'}",
                            "question_id": f"{idx}:{question_index}",
                            "question": question[:800],
                            "answer": response[:800],
                            "baseline": "direct_context",
                            "context_chars": len(transcript),
                            "episodes": [],
                            "semantic_memories": [],
                        }
                    )
                    self.results[str(idx)].append(
                        {
                            "question": question,
                            "answer": qa.get("answer"),
                            "category": qa.get("category"),
                            "evidence": qa.get("evidence", []),
                            "response": response,
                            "memories": [],
                            "search_method": "direct_context",
                            "context_chars": len(transcript),
                        }
                    )
                    await self.checkpoint()
                finally:
                    queue.task_done()

        worker_count = min(self.max_concurrent_answer, max(1, queue.qsize()))
        await tqdm.gather(*(worker() for _ in range(worker_count)), desc="Direct Questions")

    def _progress(self, *, resumed: bool = False) -> None:
        payload = {
            "phase": "search",
            "completed": self.completed_questions,
            "total": self.total_questions,
        }
        if resumed:
            payload["resumed"] = True
        print("NEMORI_PROGRESS " + json.dumps(payload), flush=True)

    async def checkpoint(self) -> None:
        async with self.checkpoint_lock:
            self.completed_questions += 1
            self._progress()
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
                    time.sleep(min(CHECKPOINT_RETRY_BASE_SECONDS * (2**attempt), 1.0))
        raise OSError(f"Could not replace checkpoint {self.output_path}; recovery kept at {tmp_path}") from last_error

    def _temp_candidates(self) -> list[Path]:
        if not self.output_path.parent.exists():
            return []
        return list(self.output_path.parent.glob(f"{self.output_path.name}*.tmp"))

    def _cleanup_stale_temp_files(self) -> None:
        for candidate in self._temp_candidates():
            try:
                candidate.unlink(missing_ok=True)
            except OSError:
                logger.debug("Could not remove stale checkpoint %s", candidate)

    def _restore_checkpoint(self, dataset: list[dict[str, Any]]) -> None:
        if not self.resume:
            return
        candidates = [self.output_path, *self._temp_candidates()]
        best: dict[str, list[dict[str, Any]]] | None = None
        best_count = -1
        for candidate in candidates:
            if not candidate.is_file():
                continue
            try:
                raw = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeError):
                continue
            matched = self._matching_checkpoint(raw, dataset)
            if matched is None:
                continue
            count = sum(len(records) for records in matched.values())
            if count > best_count:
                best = matched
                best_count = count
        if best is not None:
            self.results = defaultdict(list, best)
            self.completed_questions = best_count

    def _matching_checkpoint(
        self,
        raw_results: Any,
        dataset: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]] | None:
        if not isinstance(raw_results, dict):
            return None
        expected_keys = {str(idx) for idx in range(len(dataset))}
        if any(str(key) not in expected_keys for key in raw_results):
            return None
        matched: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for idx, item in enumerate(dataset):
            records = raw_results.get(str(idx), [])
            questions = questions_for_item(item, scored_only=self.scored_only)
            if not isinstance(records, list) or len(records) > len(questions):
                return None
            for question_index, record in enumerate(records):
                if not isinstance(record, dict):
                    return None
                if record.get("question") != questions[question_index].get("question", ""):
                    return None
            if records:
                matched[str(idx)].extend(records)
        return matched


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run direct-context LoCoMo baseline")
    parser.add_argument("--data", default="dataset/locomo10.json")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--output", default="locomo/direct_results.json")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--max-concurrent-answer", type=int, default=1)
    parser.add_argument("--max-transcript-chars", type=int, default=0)
    parser.add_argument("--scored-only", action="store_true")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest="resume", action="store_true", default=True)
    resume_group.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path
    dataset_path = Path(args.data)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    baseline = DirectLocomoBaseline(
        load_config(config_path),
        output_path=Path(args.output),
        checkpoint_every=args.checkpoint_every,
        max_concurrent_answer=max(1, args.max_concurrent_answer),
        scored_only=args.scored_only,
        resume=args.resume,
        max_transcript_chars=max(0, args.max_transcript_chars),
    )
    asyncio.run(baseline.process(dataset))
    baseline.save()


if __name__ == "__main__":  # pragma: no cover
    main()
