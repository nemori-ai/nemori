"""Native semantic-memory consolidation from Nemori Appendix D.1.7."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Literal

from nemori.domain.models import SemanticMemory
from nemori.llm.orchestrator import LLMOrchestrator, LLMRequest
from nemori.llm.prompts import PromptTemplates

logger = logging.getLogger("nemori")

ConsolidationAction = Literal["NEW", "MERGE", "CONFLICT_DELETE"]


@dataclass(frozen=True)
class ConsolidationDecision:
    """A validated native-management directive.

    ``target_ids`` is empty for ``NEW``.  ``new_content`` is populated only
    for ``MERGE``; ``CONFLICT_DELETE`` retains the incoming memory content and
    removes the selected obsolete entries.
    """

    decision: ConsolidationAction
    target_ids: tuple[str, ...] = ()
    new_content: str | None = None
    reason: str = ""


class SemanticConsolidator:
    """Resolve a distilled insight against associative semantic candidates.

    The paper deliberately makes NEW the conservative default.  Accordingly,
    transport failures, malformed JSON, and every validation failure return a
    NEW directive and never authorize mutation of existing memories.
    """

    def __init__(self, orchestrator: LLMOrchestrator) -> None:
        self._orchestrator = orchestrator

    async def decide(
        self,
        new_memory: SemanticMemory,
        candidates: list[SemanticMemory],
    ) -> ConsolidationDecision:
        """Return a strictly validated NEW/MERGE/CONFLICT_DELETE directive."""
        eligible = self._eligible_candidates(new_memory, candidates)
        if not eligible:
            return self._new("No eligible associative candidates; add the new item.")

        prompt = PromptTemplates.get_semantic_consolidation_prompt(
            new_type=new_memory.memory_type,
            new_content=new_memory.content,
            candidates=self._format_candidates(eligible),
        )
        request = LLMRequest(
            messages=({"role": "user", "content": prompt},),
            response_format={"type": "json_object"},
            metadata={"generator": "semantic_consolidation"},
        )

        try:
            response = await self._orchestrator.execute(request)
            return self._parse_and_validate(
                response.content,
                candidate_ids={candidate.id for candidate in eligible},
            )
        except Exception as error:
            logger.warning(
                "Semantic consolidation failed; defaulting safely to NEW: %s",
                error,
            )
            return self._new(
                "Consolidation decision was unavailable or invalid; defaulted to NEW."
            )

    @staticmethod
    def _eligible_candidates(
        new_memory: SemanticMemory,
        candidates: list[SemanticMemory],
    ) -> list[SemanticMemory]:
        """Keep unique candidates in the same user/agent isolation scope."""
        eligible: list[SemanticMemory] = []
        seen: set[str] = set()
        for candidate in candidates:
            if (
                candidate.id == new_memory.id
                or candidate.id in seen
                or candidate.user_id != new_memory.user_id
                or candidate.agent_id != new_memory.agent_id
            ):
                continue
            seen.add(candidate.id)
            eligible.append(candidate)
        return eligible

    @staticmethod
    def _format_candidates(candidates: list[SemanticMemory]) -> str:
        payload = [
            {
                "id": candidate.id,
                "type": candidate.memory_type,
                "content": candidate.content,
            }
            for candidate in candidates
        ]
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @classmethod
    def _parse_and_validate(
        cls,
        content: str,
        candidate_ids: set[str],
    ) -> ConsolidationDecision:
        parsed = cls._parse_json_object(content)

        decision = parsed.get("decision")
        if decision not in {"NEW", "MERGE", "CONFLICT_DELETE"}:
            raise ValueError("decision must be NEW, MERGE, or CONFLICT_DELETE")

        reason = parsed.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string")
        reason = reason.strip()

        if decision == "NEW":
            cls._validate_unused_fields(parsed)
            return cls._new(reason)

        target_ids = cls._validate_target_ids(parsed.get("target_ids"), candidate_ids)

        if decision == "MERGE":
            new_content = parsed.get("new_content")
            if not isinstance(new_content, str) or not new_content.strip():
                raise ValueError("MERGE new_content must be a non-empty string")
            new_content = new_content.strip()
            if len(new_content.split()) > 100:
                raise ValueError("MERGE new_content must contain 100 words or fewer")
            return ConsolidationDecision(
                decision="MERGE",
                target_ids=target_ids,
                new_content=new_content,
                reason=reason,
            )

        # Appendix D.1.7 does not permit generated replacement content in a
        # conflict directive: the incoming distilled item is the replacement.
        new_content = parsed.get("new_content")
        if new_content not in (None, ""):
            raise ValueError("CONFLICT_DELETE must use the incoming memory content")
        return ConsolidationDecision(
            decision="CONFLICT_DELETE",
            target_ids=target_ids,
            reason=reason,
        )

    @staticmethod
    def _parse_json_object(content: str) -> dict[str, Any]:
        if not isinstance(content, str):
            raise ValueError("consolidation response must be text")
        text = content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if not lines:
                raise ValueError("consolidation response is empty")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("consolidation response must be a JSON object")
        return parsed

    @staticmethod
    def _validate_target_ids(
        raw_target_ids: Any,
        candidate_ids: set[str],
    ) -> tuple[str, ...]:
        if not isinstance(raw_target_ids, list) or not raw_target_ids:
            raise ValueError("target_ids must be a non-empty list")
        if any(not isinstance(item, str) or not item.strip() for item in raw_target_ids):
            raise ValueError("every target id must be a non-empty string")

        target_ids = tuple(item.strip() for item in raw_target_ids)
        if len(set(target_ids)) != len(target_ids):
            raise ValueError("target_ids must not contain duplicates")
        if not set(target_ids).issubset(candidate_ids):
            raise ValueError("target_ids may reference only supplied candidates")
        return target_ids

    @staticmethod
    def _validate_unused_fields(parsed: dict[str, Any]) -> None:
        target_ids = parsed.get("target_ids")
        new_content = parsed.get("new_content")
        if target_ids not in (None, []):
            raise ValueError("NEW must not contain target_ids")
        if new_content not in (None, ""):
            raise ValueError("NEW must not contain new_content")

    @staticmethod
    def _new(reason: str) -> ConsolidationDecision:
        return ConsolidationDecision(decision="NEW", reason=reason)
