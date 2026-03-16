"""Token estimation utilities."""
from __future__ import annotations

import tiktoken


def estimate_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Estimate token count for a given text."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        return len(text) // 4
