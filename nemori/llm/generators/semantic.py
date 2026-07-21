"""Semantic memory extraction from episodes."""
from __future__ import annotations

import json
import logging
from dataclasses import replace
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
        existing_semantics: list[SemanticMemory],
    ) -> list[SemanticMemory]:
        try:
            # Bootstrap exactly as the released implementation does: there is
            # no useful world model to predict from until at least one semantic
            # memory exists.  Subsequent episodes use prediction correction.
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
        knowledge = [s.content for s in existing]

        # Step 1: Predict
        predict_prompt = PromptTemplates.get_prediction_prompt(
            episode.title, knowledge
        )
        predict_req = LLMRequest(
            messages=({"role": "user", "content": predict_prompt},),
            metadata={"generator": "semantic_predict"},
        )  # No response_format here — prediction output is free-form text
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
            temperature=0.0,
            response_format={"type": "json_object"},
            metadata={"generator": "semantic_extract"},
        )
        return await self._request_statements(extract_req)

    async def _direct_extraction(self, episode: Episode) -> list[str]:
        """Single-step extraction from episode content."""
        ep_text = f"Episode 1:\nTitle: {episode.title}\nContent: {episode.content}"
        prompt = PromptTemplates.get_semantic_generation_prompt(ep_text)
        req = LLMRequest(
            messages=({"role": "user", "content": prompt},),
            temperature=0.0,
            response_format={"type": "json_object"},
            metadata={"generator": "semantic_direct"},
        )
        return await self._request_statements(req)

    async def _request_statements(self, request: LLMRequest) -> list[str]:
        """Retry successful HTTP calls that returned malformed structured output."""
        last_error: Exception | None = None
        last_content = ""
        current = request
        for attempt in range(2):
            response = await self._orchestrator.execute(current)
            last_content = response.content
            try:
                return self._parse_statements(response.content)
            except (json.JSONDecodeError, AttributeError, TypeError, ValueError) as exc:
                last_error = exc
                if attempt == 0:
                    logger.info(
                        "Semantic structured output was invalid (%s); retrying once",
                        exc,
                    )
                    current = replace(
                        request,
                        messages=request.messages + (
                            {"role": "assistant", "content": response.content[:2000]},
                            {
                                "role": "user",
                                "content": (
                                    "Your previous response was incomplete or invalid JSON. "
                                    "Return one complete JSON object with a statements array. "
                                    "Return no markdown and no text outside the JSON object."
                                ),
                            },
                        ),
                        temperature=0.0,
                    )
        raise ValueError(
            "invalid semantic JSON after structured retry: "
            f"{last_error}; content={last_content[:500]!r}"
        )

    def _parse_statements(self, content: str) -> list[str]:
        """Parse JSON statements from LLM response."""
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        data = self._load_jsonish(text)
        statements, found = self._coerce_statements(data)
        if found:
            # An empty statements array is a valid prediction-error result: it
            # means this episode added no durable knowledge.
            return statements
        raise ValueError("structured output has no statements array")

    @staticmethod
    def _load_jsonish(text: str) -> Any:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start : end + 1])
            start = text.find("[")
            end = text.rfind("]")
            if start >= 0 and end > start:
                return json.loads(text[start : end + 1])
            raise

    @classmethod
    def _coerce_statements(cls, data: Any) -> tuple[list[str], bool]:
        if isinstance(data, list):
            return cls._coerce_statement_list(data), True
        if not isinstance(data, dict):
            return [], False
        for key in (
            "statements",
            "semantic_memories",
            "memories",
            "facts",
            "knowledge",
            "insights",
        ):
            value = data.get(key)
            if isinstance(value, list):
                return cls._coerce_statement_list(value), True
        for value in data.values():
            if isinstance(value, list):
                coerced = cls._coerce_statement_list(value)
                if coerced:
                    return coerced, True
        return [], False

    @staticmethod
    def _coerce_statement_list(items: list[Any]) -> list[str]:
        statements: list[str] = []
        for item in items:
            if isinstance(item, str):
                text = item.strip()
            elif isinstance(item, dict):
                text = str(
                    item.get("statement")
                    or item.get("content")
                    or item.get("memory")
                    or item.get("fact")
                    or item.get("text")
                    or ""
                ).strip()
            else:
                text = ""
            if text:
                statements.append(text)
        return statements

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
