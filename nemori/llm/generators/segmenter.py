"""Batch message segmentation into episodes."""
from __future__ import annotations

import json
import logging
import re
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
        formatted_lines = []
        for i, msg in enumerate(messages, 1):
            ts = msg.timestamp.strftime("%Y-%m-%d %H:%M:%S") if msg.timestamp else ""
            content = msg.text_content()[:200]  # Truncate long messages
            if ts:
                formatted_lines.append(f"{i}. [{ts}] {msg.role}: {content}")
            else:
                formatted_lines.append(f"{i}. {msg.role}: {content}")
        formatted = "\n".join(formatted_lines)

        prompt = PromptTemplates.get_batch_segmentation_prompt(
            count=len(messages), messages=formatted
        )

        conversation: list[dict[str, str]] = [{"role": "user", "content": prompt}]
        max_retries = 3
        temperature = 0.2

        for attempt in range(max_retries):
            request = LLMRequest(
                messages=tuple(conversation),
                temperature=temperature,
                metadata={"generator": "segmenter"},
            )
            response = None
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

                if groups:
                    return groups

                logger.warning("Segmentation returned empty groups on attempt %d", attempt + 1)

            except Exception as e:
                logger.warning("Segmentation attempt %d failed: %s", attempt + 1, e)
                # Add correction to conversation for next attempt
                if attempt < max_retries - 1:
                    bad_content = response.content if response is not None else str(e)
                    conversation.append({"role": "assistant", "content": bad_content})
                    conversation.append({
                        "role": "user",
                        "content": (
                            "Your response was not valid JSON. Please provide ONLY a valid JSON object "
                            "with the episodes array, no additional text. "
                            'Format: {"episodes": [{"indices": [1,2,3], "topic": "..."}]}'
                        ),
                    })
                    temperature = min(temperature + 0.1, 0.5)

        logger.warning("Segmentation failed after %d attempts, returning single group", max_retries)
        return [{"messages": messages, "topic": "conversation"}]

    def _parse_response(self, content: str) -> dict[str, Any]:
        """Parse JSON from LLM response with multiple fallback strategies."""
        text = content.strip()

        # Strategy 1: Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Strip markdown code fences
        cleaned = text
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Strategy 3: Find the largest JSON object via brace matching
        try:
            start = text.index('{')
            depth = 0
            end = start
            for i, ch in enumerate(text[start:], start):
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            candidate = text[start:end]
            return json.loads(candidate)
        except (ValueError, json.JSONDecodeError):
            pass

        # Strategy 4: Regex for JSON objects
        try:
            matches = re.findall(r'\{[\s\S]*\}', text)
            for match in sorted(matches, key=len, reverse=True):
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        except Exception:
            pass

        raise json.JSONDecodeError("No valid JSON found in response", text, 0)
