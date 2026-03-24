"""Tests for domain interface protocols."""
import pytest
from nemori.domain.interfaces import (
    EpisodeStore,
    SemanticStore,
    MessageBufferStore,
    EmbeddingProvider,
    LLMProvider,
)


def test_episode_store_has_required_methods():
    required = [
        "save", "get", "list_by_user", "delete", "delete_by_user",
        "search_by_text", "get_batch",
    ]
    for method in required:
        assert hasattr(EpisodeStore, method), f"Missing: {method}"


def test_semantic_store_has_required_methods():
    required = [
        "save", "save_batch", "get", "list_by_user", "delete",
        "delete_by_user", "search_by_text", "get_batch",
    ]
    for method in required:
        assert hasattr(SemanticStore, method), f"Missing: {method}"


def test_message_buffer_store_has_required_methods():
    required = ["push", "get_unprocessed", "mark_processed", "count_unprocessed"]
    for method in required:
        assert hasattr(MessageBufferStore, method), f"Missing: {method}"


def test_embedding_provider_has_required_methods():
    required = ["embed", "embed_batch"]
    for method in required:
        assert hasattr(EmbeddingProvider, method), f"Missing: {method}"


def test_llm_provider_has_required_methods():
    required = ["complete"]
    for method in required:
        assert hasattr(LLMProvider, method), f"Missing: {method}"
