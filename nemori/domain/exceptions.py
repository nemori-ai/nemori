"""Nemori exception hierarchy."""


class NemoriError(Exception):
    """Base exception for all Nemori errors."""


class DatabaseError(NemoriError):
    """Connection failure, query failure, migration failure."""


class LLMError(NemoriError):
    """Base for LLM call errors."""


class LLMRateLimitError(LLMError):
    """429 — caller can use retry_after to decide backoff."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class LLMAuthError(LLMError):
    """401/403 — invalid API key."""


class TokenBudgetExceeded(LLMError):
    """Token budget exhausted."""

    def __init__(self, message: str, used: int, budget: int):
        super().__init__(message)
        self.used = used
        self.budget = budget


class EmbeddingError(NemoriError):
    """Embedding generation failure."""


class ConfigError(NemoriError):
    """Configuration validation failure."""


class UserNotFoundError(NemoriError):
    """No data for the given user_id."""
