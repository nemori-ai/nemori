"""Episode merger for consolidating similar episodes."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from nemori.domain.models import Episode
from nemori.domain.interfaces import EpisodeStore, EmbeddingProvider
from nemori.db.qdrant_store import QdrantVectorStore
from nemori.llm.orchestrator import LLMOrchestrator, LLMRequest
from nemori.llm.prompts import PromptTemplates

logger = logging.getLogger("nemori")


class EpisodeMerger:
    """Async episode merger that consolidates similar episodes."""

    def __init__(
        self,
        orchestrator: LLMOrchestrator,
        embedding: EmbeddingProvider,
        episode_store: EpisodeStore,
        qdrant: QdrantVectorStore,
        similarity_threshold: float = 0.85,
        merge_top_k: int = 5,
    ) -> None:
        self._orchestrator = orchestrator
        self._embedding = embedding
        self._episode_store = episode_store
        self._qdrant = qdrant
        self._similarity_threshold = similarity_threshold
        self._merge_top_k = merge_top_k

    async def check_and_merge(
        self, episode: Episode, agent_id: str
    ) -> tuple[bool, Episode | None, str | None]:
        """Check if episode should merge with existing ones.

        Returns: (merged, final_episode, old_episode_id_to_delete)
        """
        try:
            # 1. Search for similar episodes
            candidates = await self._find_similar(episode, agent_id)
            if not candidates:
                return False, None, None

            # 2. LLM decides whether to merge
            should_merge, target_id, reason = await self._decide_merge(episode, candidates)
            if not should_merge or not target_id:
                return False, None, None

            # 3. Find target
            target = next((c for c in candidates if c.id == target_id), None)
            if not target:
                return False, None, None

            # 4. Generate merged content
            merged = await self._merge_contents(target, episode, agent_id)
            logger.info("Episode merge: %s + %s -> %s", target.id[:8], episode.id[:8], merged.id[:8])
            return True, merged, target.id

        except Exception as e:
            logger.warning("Episode merge check failed: %s", e)
            return False, None, None

    async def _find_similar(self, episode: Episode, agent_id: str) -> list[Episode]:
        """Find similar episodes using Qdrant vector search."""
        if not episode.embedding:
            return []
        results = self._qdrant.search_episodes(
            episode.user_id, agent_id, episode.embedding, self._merge_top_k + 1
        )
        # Filter out self and fetch full records from PostgreSQL
        ids = [r["id"] for r in results if r["id"] != episode.id][:self._merge_top_k]
        if not ids:
            return []
        return await self._episode_store.get_batch(ids, episode.user_id, agent_id)

    async def _decide_merge(
        self, new_episode: Episode, candidates: list[Episode]
    ) -> tuple[bool, str | None, str]:
        """Use LLM to decide if merging is appropriate."""
        candidates_text = self._format_candidates(candidates)
        ts = new_episode.created_at.strftime("%Y-%m-%d %H:%M:%S") if new_episode.created_at else "unknown"
        new_time_range = f"{ts} ({len(new_episode.source_messages)} messages)"

        prompt = PromptTemplates.get_merge_decision_prompt(
            new_time_range=new_time_range,
            new_content=new_episode.content,
            candidates=candidates_text,
        )
        request = LLMRequest(
            messages=({"role": "user", "content": prompt},),
            response_format={"type": "json_object"},
            metadata={"generator": "merge_decision"},
        )
        response = await self._orchestrator.execute(request)
        parsed = self._parse_json(response.content)

        decision = parsed.get("decision", "new")
        target_id = parsed.get("merge_target_id")
        reason = parsed.get("reason", "")
        return decision == "merge" and target_id is not None, target_id, reason

    async def _merge_contents(
        self, target: Episode, new_episode: Episode, agent_id: str
    ) -> Episode:
        """Generate merged episode content via LLM."""
        target_ts = target.created_at.strftime("%Y-%m-%d %H:%M:%S") if target.created_at else "unknown"
        new_ts = new_episode.created_at.strftime("%Y-%m-%d %H:%M:%S") if new_episode.created_at else "unknown"

        prompt = PromptTemplates.get_merge_content_prompt(
            original_time_range=f"{target_ts} ({len(target.source_messages)} messages)",
            original_title=target.title,
            original_content=target.content,
            new_time_range=f"{new_ts} ({len(new_episode.source_messages)} messages)",
            new_title=new_episode.title,
            new_content=new_episode.content,
            combined_events=f"Original: {target.content}\n\nNew: {new_episode.content}",
        )
        request = LLMRequest(
            messages=({"role": "user", "content": prompt},),
            response_format={"type": "json_object"},
            metadata={"generator": "merge_content"},
        )
        response = await self._orchestrator.execute(request)
        parsed = self._parse_json(response.content)

        # Merge source messages
        merged_messages = target.source_messages + new_episode.source_messages

        # Use earliest timestamp (ensure both are aware for safe comparison)
        now = datetime.now(timezone.utc)
        ts_target = target.created_at or now
        ts_new = new_episode.created_at or now
        # Normalise to aware datetimes if either is naive
        if ts_target.tzinfo is None:
            ts_target = ts_target.replace(tzinfo=timezone.utc)
        if ts_new.tzinfo is None:
            ts_new = ts_new.replace(tzinfo=timezone.utc)
        merged_ts = min(ts_target, ts_new)
        if parsed.get("timestamp"):
            try:
                parsed_ts = datetime.fromisoformat(parsed["timestamp"])
                if parsed_ts.tzinfo is None:
                    parsed_ts = parsed_ts.replace(tzinfo=timezone.utc)
                merged_ts = parsed_ts
            except (ValueError, TypeError):
                pass

        # Generate new embedding
        embed_text = f"{parsed.get('title', '')} {parsed.get('content', '')}"
        embedding = await self._embedding.embed(embed_text)

        return Episode(
            user_id=new_episode.user_id,
            agent_id=agent_id,
            title=parsed.get("title", f"Merged: {target.title}"),
            content=parsed.get("content", f"{target.content}\n\n{new_episode.content}"),
            source_messages=merged_messages,
            embedding=embedding,
            metadata={
                "merged_from": [target.id, new_episode.id],
                "merge_timestamp": datetime.now(timezone.utc).isoformat(),
            },
            created_at=merged_ts,
            updated_at=datetime.now(timezone.utc),
        )

    def _format_candidates(self, candidates: list[Episode]) -> str:
        lines = []
        for i, ep in enumerate(candidates, 1):
            ts = ep.created_at.strftime("%Y-%m-%d %H:%M:%S") if ep.created_at else "unknown"
            lines.append(
                f"{i}. Candidate ID: {ep.id}\n"
                f"   Time: {ts} ({len(ep.source_messages)} messages)\n"
                f"   Title: {ep.title}\n"
                f"   Content: {ep.content[:200]}..."
            )
        return "\n\n".join(lines)

    def _parse_json(self, content: str) -> dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return json.loads(text)
