"""Episode generation from conversation messages."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from nemori.domain.models import Message, Episode
from nemori.domain.interfaces import EmbeddingProvider
from nemori.llm.orchestrator import LLMOrchestrator, LLMRequest
from nemori.llm.prompts import PromptTemplates

logger = logging.getLogger("nemori")

_MULTIMODAL_GUIDANCE = """If images are included in this conversation:
1. Use the images to enrich your understanding of what the user was doing or discussing.
2. Describe the visual context naturally within the narrative.
3. Do NOT reference technical details like "image_url" or "screenshot #3".
4. Integrate visual information chronologically with the text conversation."""


class EpisodeGenerator:
    """Constructs episode generation prompts and parses LLM responses."""

    def __init__(self, orchestrator: LLMOrchestrator, embedding: EmbeddingProvider) -> None:
        self._orchestrator = orchestrator
        self._embedding = embedding

    async def generate(
        self, user_id: str, agent_id: str, messages: list[Message], boundary_reason: str
    ) -> Episode:
        # Format conversation
        msg_dicts = [m.to_dict() for m in messages]
        conversation = PromptTemplates.format_conversation(msg_dicts)
        prompt = PromptTemplates.get_episode_generation_prompt(conversation, boundary_reason)

        has_images = any(m.has_images() for m in messages)

        if has_images:
            # Build multimodal content array
            user_content = self._build_multimodal_prompt(messages, boundary_reason)
        else:
            user_content = prompt  # existing text prompt

        request = LLMRequest(
            messages=(
                {"role": "system", "content": "You are an episodic memory generation expert."},
                {"role": "user", "content": user_content},
            ),
            response_format={"type": "json_object"},
            metadata={"generator": "episode", "user_id": user_id},
        )

        try:
            response = await self._orchestrator.execute(request)
            parsed = self._parse_response(response.content)

            # Generate embedding
            embed_text = f"{parsed['title']} {parsed['content']}"
            embedding = await self._embedding.embed(embed_text)

            # Parse timestamp
            timestamp = datetime.now()
            if parsed.get("timestamp"):
                try:
                    timestamp = datetime.fromisoformat(parsed["timestamp"])
                except (ValueError, TypeError):
                    pass

            return Episode(
                user_id=user_id,
                title=parsed["title"],
                content=parsed["content"],
                source_messages=msg_dicts,
                agent_id=agent_id,
                embedding=embedding,
                metadata={"boundary_reason": boundary_reason},
                created_at=timestamp,
                updated_at=datetime.now(),
            )
        except Exception as e:
            logger.warning("Episode generation failed, creating fallback: %s", e)
            return self._create_fallback(user_id, agent_id, messages, boundary_reason)

    def _parse_response(self, content: str) -> dict[str, Any]:
        """Parse JSON response from LLM."""
        # Strip markdown code fences if present
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return json.loads(text)

    def _build_multimodal_prompt(
        self, messages: list[Message], boundary_reason: str
    ) -> list[dict]:
        """Build content array with text prompt + images."""
        # Format conversation with image markers
        conversation = self._format_with_image_markers(messages)
        prompt = PromptTemplates.get_episode_generation_prompt(conversation, boundary_reason)

        # Add multimodal guidance
        prompt += "\n\n" + _MULTIMODAL_GUIDANCE

        parts: list[dict] = [{"type": "text", "text": prompt}]

        # Attach images (already compressed at ingestion time)
        for msg in messages:
            for url in msg.image_urls():
                parts.append({"type": "image_url", "image_url": {"url": url}})

        return parts

    def _format_with_image_markers(self, messages: list[Message]) -> str:
        """Format conversation text, marking image positions."""
        lines = []
        for msg in messages:
            if isinstance(msg.content, str):
                lines.append(f"{msg.role}: {msg.content}")
            else:
                msg_parts = []
                for part in msg.content:
                    if part.get("type") == "text":
                        msg_parts.append(part["text"])
                    elif part.get("type") == "image_url":
                        msg_parts.append("[Image attached]")
                lines.append(f"{msg.role}: {' '.join(msg_parts)}")
        return "\n".join(lines)

    def _create_fallback(
        self, user_id: str, agent_id: str, messages: list[Message], boundary_reason: str
    ) -> Episode:
        """Create a raw episode when LLM generation fails."""
        conversation = "\n".join(f"{m.role}: {m.text_content()}" for m in messages)
        return Episode(
            user_id=user_id,
            title=f"Conversation ({len(messages)} messages)",
            content=conversation,
            source_messages=[m.to_dict() for m in messages],
            agent_id=agent_id,
            metadata={"boundary_reason": boundary_reason, "fallback": True},
        )
