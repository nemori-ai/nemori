"""Semantic memory extraction from episodes."""
from __future__ import annotations

import json
import logging
from typing import Any

from nemori.domain.models import Episode, SemanticMemory
from nemori.domain.interfaces import EmbeddingProvider
from nemori.llm.orchestrator import LLMOrchestrator, LLMRequest
from nemori.llm.prompts import PromptTemplates

logger = logging.getLogger("nemori")


def _extract_text(msg_dict: dict) -> str:
    """Extract text from a source_message dict, handling both str and content array."""
    content = msg_dict.get("content", "")
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if part.get("type") == "text":
            parts.append(part["text"])
        elif part.get("type") == "image_url":
            parts.append("[image]")
    return " ".join(parts)


class SemanticGenerator:
    """Extracts semantic memories from episodes."""

    def __init__(
        self,
        orchestrator: LLMOrchestrator,
        embedding: EmbeddingProvider,
        enable_prediction_correction: bool = True,
    ) -> None:
        self._orchestrator = orchestrator
        self._embedding = embedding
        self._enable_pc = enable_prediction_correction

    async def generate(
        self,
        user_id: str,
        agent_id: str,
        episode: Episode,
        existing_episodes: list[Episode],
        existing_semantics: list[SemanticMemory],
    ) -> list[SemanticMemory]:
        try:
            if self._enable_pc and existing_semantics:
                statements = await self._prediction_correction(episode, existing_semantics)
            else:
                statements = await self._direct_extraction(episode)

            if not statements:
                return []

            # Generate embeddings and create memories
            memories = []
            for stmt in statements:
                emb = await self._embedding.embed(stmt)
                memories.append(SemanticMemory(
                    user_id=user_id,
                    content=stmt,
                    memory_type=self._classify_type(stmt),
                    agent_id=agent_id,
                    embedding=emb,
                    source_episode_id=episode.id,
                ))
            return memories

        except Exception as e:
            logger.warning("Semantic generation failed: %s", e)
            return []

    async def _prediction_correction(
        self, episode: Episode, existing: list[SemanticMemory]
    ) -> list[str]:
        """Two-step: predict from knowledge, then extract deltas."""
        knowledge = [s.content for s in existing[:20]]

        # Step 1: Predict
        predict_prompt = PromptTemplates.get_prediction_prompt(
            episode.title, knowledge
        )
        predict_req = LLMRequest(
            messages=({"role": "user", "content": predict_prompt},),
            metadata={"generator": "semantic_predict"},
        )
        predict_resp = await self._orchestrator.execute(predict_req)

        # Step 2: Extract deltas
        original = "\n".join(
            f"{m.get('role', 'unknown')}: {_extract_text(m)}"
            for m in episode.source_messages
        )
        extract_prompt = PromptTemplates.EXTRACT_KNOWLEDGE_FROM_COMPARISON_PROMPT.format(
            original_messages=original,
            predicted_episode=predict_resp.content,
        )
        extract_req = LLMRequest(
            messages=({"role": "user", "content": extract_prompt},),
            metadata={"generator": "semantic_extract"},
        )
        extract_resp = await self._orchestrator.execute(extract_req)
        return self._parse_statements(extract_resp.content)

    async def _direct_extraction(self, episode: Episode) -> list[str]:
        """Single-step extraction from episode content."""
        ep_text = f"Episode 1:\nTitle: {episode.title}\nContent: {episode.content}"
        prompt = PromptTemplates.get_semantic_generation_prompt(ep_text)
        req = LLMRequest(
            messages=({"role": "user", "content": prompt},),
            metadata={"generator": "semantic_direct"},
        )
        resp = await self._orchestrator.execute(req)
        return self._parse_statements(resp.content)

    def _parse_statements(self, content: str) -> list[str]:
        """Parse JSON statements from LLM response."""
        try:
            text = content.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            data = json.loads(text)
            return data.get("statements", [])
        except (json.JSONDecodeError, AttributeError):
            logger.warning("Failed to parse semantic statements")
            return []

    @staticmethod
    def _classify_type(statement: str) -> str:
        """Simple keyword-based classification."""
        lower = statement.lower()
        if any(w in lower for w in ["name is", "works at", "job", "profession", "role is"]):
            return "identity"
        if any(w in lower for w in ["likes", "prefers", "favorite", "enjoys"]):
            return "preference"
        if any(w in lower for w in ["family", "friend", "colleague", "partner", "wife", "husband"]):
            return "relationship"
        if any(w in lower for w in ["goal", "plan", "wants to", "aims to", "intends"]):
            return "goal"
        if any(w in lower for w in ["believes", "thinks that", "values"]):
            return "belief"
        if any(w in lower for w in ["every", "always", "usually", "routine", "habit"]):
            return "habit"
        return "identity"
