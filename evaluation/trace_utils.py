from __future__ import annotations

import json
from typing import Any


def emit_trace(payload: dict[str, Any]) -> None:
    """Emit one machine-readable dashboard trace without logging secrets."""
    print("NEMORI_TRACE " + json.dumps(payload, ensure_ascii=False), flush=True)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(part.get("text", ""))
            if part.get("type") == "text"
            else "[image]"
            for part in content
            if isinstance(part, dict)
        )
    return str(content or "")


def message_preview(messages: list[dict[str, Any]], limit: int = 6) -> list[dict[str, Any]]:
    return [
        {
            "role": message.get("role", "unknown"),
            "timestamp": str(message.get("timestamp") or ""),
            "content": _text(message.get("content"))[:360],
        }
        for message in messages[:limit]
    ]


async def memory_snapshot(memory: Any, user_id: str) -> dict[str, Any]:
    """Read back the just-created records so the canvas shows real transformations."""
    system = getattr(memory, "_system", None)
    if system is None:
        return {"episodes": [], "semantic_memories": []}
    agent_id = system._agent_id
    episodes = await system._episode_store.list_by_user(user_id, agent_id)
    semantics = await system._semantic_store.list_by_user(user_id, agent_id)
    episode_items = []
    for episode in episodes[-12:]:
        source = episode.source_messages or []
        episode_items.append(
            {
                "id": episode.id,
                "title": episode.title,
                "narrative": episode.content[:1200],
                "narrative_chars": len(episode.content),
                "source_message_count": len(source),
                "source_chars": sum(len(_text(item.get("content"))) for item in source),
                "source_preview": message_preview(source, limit=4),
                "boundary_reason": episode.metadata.get("boundary_reason"),
                "fallback": bool(episode.metadata.get("fallback")),
            }
        )
    semantic_items = [
        {
            "id": memory_item.id,
            "content": memory_item.content[:600],
            "type": memory_item.memory_type,
            "source_episode_id": memory_item.source_episode_id,
        }
        for memory_item in semantics[-20:]
    ]
    return {"episodes": episode_items, "semantic_memories": semantic_items}


def retrieval_snapshot(
    *,
    question: str,
    answer: str,
    memories: dict[str, list[dict[str, Any]]],
    method: str,
    top_k_episodes: int,
    top_k_semantic: int,
    include_original_limit: int,
    question_id: str | None = None,
) -> dict[str, Any]:
    episodes = memories.get("episodes", [])
    semantics = memories.get("semantic_memories", [])
    episode_items = [
        {
            "id": item.get("id"),
            "title": item.get("title"),
            "content": str(item.get("content") or "")[:900],
            "created_at": str(item.get("created_at") or ""),
            "source_message_count": len(item.get("source_messages") or []),
        }
        for item in episodes[:10]
    ]
    semantic_items = [
        {
            "id": item.get("id"),
            "content": str(item.get("content") or "")[:500],
            "type": item.get("memory_type"),
            "created_at": str(item.get("created_at") or ""),
        }
        for item in semantics[:16]
    ]
    context_chars = sum(len(item["content"]) for item in episode_items) + sum(
        len(item["content"]) for item in semantic_items
    )
    return {
        "event": "search_complete",
        "flow": "search",
        "nodes": [
            "question",
            "query_embed",
            "episode_recall",
            "semantic_recall",
            "rrf",
            "topk",
            "answer_prompt",
            "answer",
            "checkpoint",
        ],
        "summary": (
            f"检索完成：{len(episode_items)} 个情景 + "
            f"{len(semantic_items)} 条语义记忆 → {answer[:80] or '空回答'}"
        ),
        "question_id": question_id,
        "question": question[:800],
        "answer": answer[:800],
        "search_method": method,
        "top_k_episodes": top_k_episodes,
        "top_k_semantic": top_k_semantic,
        "include_original_limit": include_original_limit,
        "context_chars": context_chars,
        "episodes": episode_items,
        "semantic_memories": semantic_items,
    }
