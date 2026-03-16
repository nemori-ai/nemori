"""Episode generation from conversation messages."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from src.domain.models import Message, Episode
from src.domain.interfaces import EmbeddingProvider
from src.llm.orchestrator import LLMOrchestrator, LLMRequest
from src.llm.prompts import PromptTemplates

logger = logging.getLogger("nemori")


class EpisodeGenerator:
    """Constructs episode generation prompts and parses LLM responses."""

    def __init__(self, orchestrator: LLMOrchestrator, embedding: EmbeddingProvider) -> None:
        self._orchestrator = orchestrator
        self._embedding = embedding

    async def generate(
        self, user_id: str, messages: list[Message], boundary_reason: str
    ) -> Episode:
        # Format conversation
        msg_dicts = [m.to_dict() for m in messages]
        conversation = PromptTemplates.format_conversation(msg_dicts)
        prompt = PromptTemplates.get_episode_generation_prompt(conversation, boundary_reason)

        request = LLMRequest(
            messages=(
                {"role": "system", "content": "You are an episodic memory generation expert."},
                {"role": "user", "content": prompt},
            ),
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
                embedding=embedding,
                metadata={"boundary_reason": boundary_reason},
                created_at=timestamp,
                updated_at=datetime.now(),
            )
        except Exception as e:
            logger.warning("Episode generation failed, creating fallback: %s", e)
            return self._create_fallback(user_id, messages, boundary_reason)

    def _parse_response(self, content: str) -> dict[str, Any]:
        """Parse JSON response from LLM."""
        # Strip markdown code fences if present
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return json.loads(text)

    def _create_fallback(
        self, user_id: str, messages: list[Message], boundary_reason: str
    ) -> Episode:
        """Create a raw episode when LLM generation fails."""
        conversation = "\n".join(f"{m.role}: {m.content}" for m in messages)
        return Episode(
            user_id=user_id,
            title=f"Conversation ({len(messages)} messages)",
            content=conversation,
            source_messages=[m.to_dict() for m in messages],
            metadata={"boundary_reason": boundary_reason, "fallback": True},
        )
