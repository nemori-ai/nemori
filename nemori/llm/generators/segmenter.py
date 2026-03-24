"""Batch message segmentation into episodes."""
from __future__ import annotations

import json
import logging
from typing import Any

from nemori.domain.models import Message
from nemori.llm.orchestrator import LLMOrchestrator, LLMRequest
from nemori.llm.prompts import PromptTemplates

logger = logging.getLogger("nemori")


class BatchSegmenter:
    """Segments a batch of messages into coherent episode groups."""

    def __init__(self, orchestrator: LLMOrchestrator) -> None:
        self._orchestrator = orchestrator

    async def segment(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Segment messages into groups. Returns list of {messages, topic}."""
        # Format messages for prompt
        formatted_lines = []
        for i, msg in enumerate(messages, 1):
            ts = msg.timestamp.strftime("%Y-%m-%d %H:%M:%S") if msg.timestamp else ""
            if ts:
                formatted_lines.append(f"{i}. [{ts}] {msg.role}: {msg.text_content()}")
            else:
                formatted_lines.append(f"{i}. {msg.role}: {msg.text_content()}")
        formatted = "\n".join(formatted_lines)

        prompt = PromptTemplates.get_batch_segmentation_prompt(
            count=len(messages), messages=formatted
        )
        request = LLMRequest(
            messages=({"role": "user", "content": prompt},),
            metadata={"generator": "segmenter"},
        )

        try:
            response = await self._orchestrator.execute(request)
            parsed = self._parse_response(response.content)

            groups = []
            for ep in parsed.get("episodes", []):
                indices = ep.get("indices", [])
                topic = ep.get("topic", "")
                group_messages = []
                for idx in indices:
                    if 1 <= idx <= len(messages):
                        group_messages.append(messages[idx - 1])
                if group_messages:
                    groups.append({"messages": group_messages, "topic": topic})

            return groups if groups else [{"messages": messages, "topic": "conversation"}]

        except Exception as e:
            logger.warning("Segmentation failed, returning single group: %s", e)
            return [{"messages": messages, "topic": "conversation"}]

    def _parse_response(self, content: str) -> dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return json.loads(text)
