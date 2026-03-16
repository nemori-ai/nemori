"""Tests for Nemori exception hierarchy."""
import pytest
from src.domain.exceptions import (
    NemoriError,
    DatabaseError,
    LLMError,
    LLMRateLimitError,
    LLMAuthError,
    TokenBudgetExceeded,
    EmbeddingError,
    ConfigError,
    UserNotFoundError,
)


def test_all_exceptions_inherit_nemori_error():
    exceptions = [
        DatabaseError("db fail"),
        LLMError("llm fail"),
        LLMRateLimitError("rate limited"),
        LLMAuthError("auth fail"),
        TokenBudgetExceeded("budget exceeded", used=5000, budget=1000),
        EmbeddingError("embed fail"),
        ConfigError("config fail"),
        UserNotFoundError("user not found"),
    ]
    for exc in exceptions:
        assert isinstance(exc, NemoriError)


def test_llm_rate_limit_error_has_retry_after():
    exc = LLMRateLimitError("rate limited", retry_after=30.0)
    assert exc.retry_after == 30.0
    assert isinstance(exc, LLMError)


def test_llm_rate_limit_error_default_retry_after():
    exc = LLMRateLimitError("rate limited")
    assert exc.retry_after is None


def test_token_budget_exceeded_has_usage_info():
    exc = TokenBudgetExceeded("over budget", used=5000, budget=1000)
    assert exc.used == 5000
    assert exc.budget == 1000
    assert isinstance(exc, LLMError)


def test_nemori_error_is_exception():
    with pytest.raises(NemoriError):
        raise DatabaseError("test")
