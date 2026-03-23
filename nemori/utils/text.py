"""Text utilities."""
from __future__ import annotations


def estimate_token_count(text: str) -> int:
    """Quick token estimate based on character heuristics."""
    if not text:
        return 0
    cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    if cjk_count > len(text) * 0.3:
        return int(len(text) / 1.5)
    return len(text) // 4
