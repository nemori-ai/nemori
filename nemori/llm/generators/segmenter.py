"""Batch message segmentation into episodes."""
from __future__ import annotations

import json
import logging
from dataclasses import replace
from typing import Any

from nemori.domain.models import Message
from nemori.llm.orchestrator import LLMOrchestrator, LLMRequest
from nemori.llm.prompts import PromptTemplates

logger = logging.getLogger("nemori")

# Nemori's reported experiments use an observation window of w=20.  Keep the
# value here as the direct-construction default while allowing MemoryConfig to
# override it for ablations and evaluation profiles.
DEFAULT_OBSERVATION_WINDOW = 20


class BatchSegmenter:
    """Segments a batch of messages into coherent episode groups."""

    def __init__(
        self,
        orchestrator: LLMOrchestrator,
        window_size: int = DEFAULT_OBSERVATION_WINDOW,
        timeout: float | None = None,
        retries: int | None = None,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be > 0")
        if retries is not None and retries < 1:
            raise ValueError("retries must be >= 1")
        self._orchestrator = orchestrator
        self._window_size = window_size
        self._timeout = timeout
        self._retries = retries

    async def segment(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Segment messages into groups. Returns list of {messages, topic}.

        For batches larger than the observation window, splits them into
        windows and segments each window independently.
        """
        if len(messages) <= self._window_size:
            return await self._segment_chunk(messages, offset=0)

        # Split into manageable chunks and segment each
        all_groups: list[dict[str, Any]] = []
        starts = list(range(0, len(messages), self._window_size))
        for window_number, start in enumerate(starts, 1):
            chunk = messages[start : start + self._window_size]
            groups = await self._segment_chunk(chunk, offset=start)
            all_groups.extend(groups)
            logger.info(
                "Segmentation progress: window %d/%d at offset %d -> %d episodes",
                window_number,
                len(starts),
                start,
                len(groups),
            )

        return all_groups if all_groups else [{"messages": messages, "topic": "conversation"}]

    async def _segment_chunk(
        self, messages: list[Message], offset: int = 0
    ) -> list[dict[str, Any]]:
        """Segment a single chunk of messages."""
        formatted_lines = []
        for i, msg in enumerate(messages, 1):
            ts = msg.timestamp.strftime("%Y-%m-%d %H:%M:%S") if msg.timestamp else ""
            content = msg.text_content()
            if ts:
                formatted_lines.append(f"{i}. [{ts}] {msg.role}: {content}")
            else:
                formatted_lines.append(f"{i}. {msg.role}: {content}")
        formatted = "\n".join(formatted_lines)

        prompt = PromptTemplates.get_batch_segmentation_prompt(
            count=len(messages), messages=formatted
        )

        request_options: dict[str, Any] = {}
        if self._timeout is not None:
            request_options["timeout"] = self._timeout
        if self._retries is not None:
            request_options["retries"] = self._retries

        request = LLMRequest(
            messages=({"role": "user", "content": prompt},),
            temperature=0.0,
            max_tokens=4096,
            response_format={"type": "json_object"},
            metadata={"generator": "segmenter"},
            **request_options,
        )

        try:
            current_request = request
            repairable: tuple[
                list[dict[str, Any]], set[int], set[int], set[int]
            ] | None = None
            last_error: Exception | None = None

            for attempt in range(2):
                response = await self._orchestrator.execute(current_request)
                try:
                    episodes = self._parse_episode_specs(response.content)
                    missing, duplicates, invalid = self._partition_issues(
                        episodes, len(messages)
                    )
                    if not missing and not duplicates and not invalid:
                        groups = self._groups_from_specs(episodes, messages)
                        if offset == 0 and len(messages) <= self._window_size:
                            logger.info(
                                "Segmentation progress: window 1/1 at offset 0 -> "
                                "%d episodes",
                                len(groups),
                            )
                        return groups

                    repairable = (episodes, missing, duplicates, invalid)
                    if attempt == 0:
                        logger.info(
                            "Segmentation partition incomplete at offset %d; "
                            "retrying with explicit constraints "
                            "(missing_indices=%s, duplicate_indices=%s, "
                            "invalid_indices=%s)",
                            offset,
                            sorted(missing),
                            sorted(duplicates),
                            sorted(invalid),
                        )
                        current_request = replace(
                            request,
                            messages=request.messages + (
                                {"role": "assistant", "content": response.content[:4000]},
                                {
                                    "role": "user",
                                    "content": self._partition_retry_instruction(
                                        len(messages), missing, duplicates, invalid
                                    ),
                                },
                            ),
                            temperature=0.0,
                        )
                        continue
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    last_error = exc
                    if attempt == 0:
                        logger.info(
                            "Segmentation structured output invalid at offset %d "
                            "(%s); retrying once",
                            offset,
                            exc,
                        )
                        current_request = replace(
                            request,
                            messages=request.messages + (
                                {"role": "assistant", "content": response.content[:4000]},
                                {
                                    "role": "user",
                                    "content": self._partition_retry_instruction(
                                        len(messages),
                                        set(range(1, len(messages) + 1)),
                                        set(),
                                        set(),
                                    ),
                                },
                            ),
                            temperature=0.0,
                        )
                        continue

            if repairable is not None:
                episodes, missing, duplicates, invalid = repairable
                repaired = self._repair_partition(episodes, messages)
                logger.warning(
                    "Segmentation partition repaired at offset %d after structured "
                    "retry (assigned_missing=%s, removed_duplicates=%s, "
                    "removed_invalid=%s)",
                    offset,
                    sorted(missing),
                    sorted(duplicates),
                    sorted(invalid),
                )
                return repaired

            if last_error is not None:
                raise last_error
            raise ValueError("structured output contains no usable episode groups")

        except Exception as e:
            detail = str(e).strip()
            error = f"{type(e).__name__}: {detail}" if detail else type(e).__name__
            logger.warning(
                "Segmentation failed for observation window at offset %d "
                "(%d messages); returning a single fallback group: %s",
                offset,
                len(messages),
                error,
            )
            return [{"messages": messages, "topic": "conversation"}]

    @staticmethod
    def _parse_episode_specs(content: str) -> list[dict[str, Any]]:
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(
                lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            )
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                raise
            parsed = json.loads(text[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("structured output is not a JSON object")
        raw_episodes = parsed.get("episodes")
        if not isinstance(raw_episodes, list):
            raise ValueError("structured output is missing an episodes list")

        episodes: list[dict[str, Any]] = []
        for episode in raw_episodes:
            if not isinstance(episode, dict):
                continue
            indices = episode.get("indices")
            if not isinstance(indices, list):
                continue
            episodes.append(
                {
                    "indices": [idx for idx in indices if isinstance(idx, int)],
                    "topic": str(episode.get("topic") or "conversation"),
                }
            )
        if not episodes:
            raise ValueError("structured output contains no usable episode groups")
        return episodes

    @staticmethod
    def _partition_issues(
        episodes: list[dict[str, Any]], message_count: int
    ) -> tuple[set[int], set[int], set[int]]:
        expected = set(range(1, message_count + 1))
        seen: set[int] = set()
        duplicates: set[int] = set()
        invalid: set[int] = set()
        for episode in episodes:
            for idx in episode["indices"]:
                if idx not in expected:
                    invalid.add(idx)
                    continue
                if idx in seen:
                    duplicates.add(idx)
                seen.add(idx)
        return expected - seen, duplicates, invalid

    @staticmethod
    def _partition_retry_instruction(
        message_count: int,
        missing: set[int],
        duplicates: set[int],
        invalid: set[int],
    ) -> str:
        issue_lines = []
        if missing:
            issue_lines.append(f"Missing indices: {sorted(missing)}.")
        if duplicates:
            issue_lines.append(f"Duplicate indices: {sorted(duplicates)}.")
        if invalid:
            issue_lines.append(f"Out-of-range indices: {sorted(invalid)}.")
        issues = " ".join(issue_lines)
        return (
            "Your previous partition violated the required partition constraint. "
            f"{issues} Return a corrected complete JSON object. Every integer from "
            f"1 through {message_count} must appear exactly once across all episodes: "
            "no omissions, no duplicates, and no out-of-range indices. Preserve the "
            "topic boundaries you identified. Return only the JSON object."
        )

    @staticmethod
    def _groups_from_specs(
        episodes: list[dict[str, Any]], messages: list[Message]
    ) -> list[dict[str, Any]]:
        groups = []
        for episode in episodes:
            indices = sorted(set(episode["indices"]))
            group_messages = [
                messages[idx - 1] for idx in indices if 1 <= idx <= len(messages)
            ]
            if group_messages:
                groups.append(
                    {"messages": group_messages, "topic": episode["topic"]}
                )
        if not groups:
            raise ValueError("structured output contains no usable episode groups")
        return groups

    @classmethod
    def _repair_partition(
        cls, episodes: list[dict[str, Any]], messages: list[Message]
    ) -> list[dict[str, Any]]:
        """Preserve model boundaries while deterministically covering each message."""
        message_count = len(messages)
        assigned: set[int] = set()
        repaired_specs: list[dict[str, Any]] = []
        for episode in episodes:
            unique = []
            for idx in episode["indices"]:
                if 1 <= idx <= message_count and idx not in assigned:
                    unique.append(idx)
                    assigned.add(idx)
            repaired_specs.append(
                {"indices": sorted(unique), "topic": episode["topic"]}
            )

        missing = sorted(set(range(1, message_count + 1)) - assigned)
        for idx in missing:
            nonempty = [
                (group_index, spec)
                for group_index, spec in enumerate(repaired_specs)
                if spec["indices"]
            ]
            if not nonempty:
                return [{"messages": messages, "topic": "conversation"}]
            group_index, _ = min(
                nonempty,
                key=lambda pair: (
                    min(abs(idx - existing) for existing in pair[1]["indices"]),
                    0 if any(existing < idx for existing in pair[1]["indices"]) else 1,
                    pair[0],
                ),
            )
            repaired_specs[group_index]["indices"].append(idx)
            repaired_specs[group_index]["indices"].sort()

        repaired_specs = [spec for spec in repaired_specs if spec["indices"]]
        return cls._groups_from_specs(repaired_specs, messages)
