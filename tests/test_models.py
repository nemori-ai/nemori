"""Tests for domain models."""
import pytest
from datetime import datetime
from nemori.domain.models import Message, Episode, SemanticMemory, HealthResult


class TestMessage:
    def test_create_message(self):
        msg = Message(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.message_id
        assert isinstance(msg.timestamp, datetime)

    def test_message_from_dict(self):
        data = {"role": "user", "content": "hi", "timestamp": "2024-01-01T10:00:00"}
        msg = Message.from_dict(data)
        assert msg.role == "user"
        assert msg.content == "hi"

    def test_message_to_dict(self):
        msg = Message(role="assistant", content="world")
        d = msg.to_dict()
        assert d["role"] == "assistant"
        assert d["content"] == "world"
        assert "message_id" in d
        assert "timestamp" in d

    def test_multimodal_content(self):
        msg = Message(
            role="user",
            content=[
                {"type": "text", "text": "Look at this"},
                {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
            ],
        )
        assert msg.has_images()
        assert msg.image_urls() == ["https://example.com/img.png"]
        assert msg.text_content() == "Look at this [image]"
        assert msg.text_content(include_placeholders=False) == "Look at this"

    def test_text_only_message_helpers(self):
        msg = Message(role="user", content="just text")
        assert not msg.has_images()
        assert msg.image_urls() == []
        assert msg.text_content() == "just text"

    def test_multimodal_to_dict_round_trip(self):
        content = [
            {"type": "text", "text": "hello"},
            {"type": "image_url", "image_url": {"url": "https://img.png"}},
        ]
        msg = Message(role="user", content=content)
        d = msg.to_dict()
        assert d["content"] == content
        msg2 = Message.from_dict(d)
        assert msg2.content == content
        assert msg2.has_images()


class TestEpisode:
    def test_create_episode(self):
        ep = Episode(
            user_id="u1", title="Test", content="Test content",
            source_messages=[{"role": "user", "content": "hi"}],
        )
        assert ep.id
        assert ep.user_id == "u1"
        assert ep.title == "Test"
        assert ep.metadata == {}

    def test_episode_from_dict(self):
        data = {
            "id": "abc-123", "user_id": "u1", "title": "Test",
            "content": "Content", "source_messages": [],
            "created_at": "2024-01-01T10:00:00",
        }
        ep = Episode.from_dict(data)
        assert ep.id == "abc-123"

    def test_episode_to_dict(self):
        ep = Episode(user_id="u1", title="T", content="C", source_messages=[])
        d = ep.to_dict()
        assert "id" in d
        assert d["user_id"] == "u1"
        assert "created_at" in d

    def test_episode_metadata_stores_boundary_reason(self):
        ep = Episode(
            user_id="u1", title="T", content="C", source_messages=[],
            metadata={"boundary_reason": "topic_change"},
        )
        assert ep.metadata["boundary_reason"] == "topic_change"


class TestSemanticMemory:
    def test_create_semantic_memory(self):
        sm = SemanticMemory(user_id="u1", content="User likes hiking", memory_type="preference")
        assert sm.id
        assert sm.memory_type == "preference"
        assert sm.confidence == 1.0
        assert sm.source_episode_id is None

    def test_semantic_memory_with_source(self):
        sm = SemanticMemory(
            user_id="u1", content="User works at Google", memory_type="identity",
            source_episode_id="ep-123", confidence=0.9,
        )
        assert sm.source_episode_id == "ep-123"
        assert sm.confidence == 0.9

    def test_semantic_memory_to_dict(self):
        sm = SemanticMemory(user_id="u1", content="fact", memory_type="identity")
        d = sm.to_dict()
        assert d["memory_type"] == "identity"
        assert "id" in d


class TestHealthResult:
    def test_healthy_when_all_ok(self):
        hr = HealthResult(db=True, llm=True, embedding=True, diagnostics={})
        assert hr.healthy is True

    def test_unhealthy_when_db_down(self):
        hr = HealthResult(db=False, llm=True, embedding=True, diagnostics={})
        assert hr.healthy is False
