"""Tests for multimodal message flow through the pipeline."""
from __future__ import annotations

import pytest
from datetime import datetime

from nemori.domain.models import Message
from nemori.llm.generators.episode import EpisodeGenerator, MAX_IMAGES_PER_EPISODE
from nemori.llm.generators.semantic import _extract_text


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TEXT_ONLY_CONTENT = "Hello, this is a plain text message."

MULTIMODAL_CONTENT: list[dict] = [
    {"type": "text", "text": "Check out this photo"},
    {"type": "image_url", "image_url": {"url": "https://example.com/img1.png"}},
]

MULTI_IMAGE_CONTENT: list[dict] = [
    {"type": "text", "text": "Here are two images"},
    {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
    {"type": "image_url", "image_url": {"url": "https://example.com/b.png"}},
]

IMAGE_ONLY_CONTENT: list[dict] = [
    {"type": "image_url", "image_url": {"url": "https://example.com/only.png"}},
]


def _make_msg(content, role="user") -> Message:
    return Message(role=role, content=content, message_id="test-id")


# ===================================================================
# 1. Message class multimodal methods
# ===================================================================


class TestMessageTextContent:
    """Message.text_content() with various content shapes."""

    def test_string_content(self):
        msg = _make_msg(TEXT_ONLY_CONTENT)
        assert msg.text_content() == TEXT_ONLY_CONTENT

    def test_content_array_extracts_text(self):
        msg = _make_msg(MULTIMODAL_CONTENT)
        result = msg.text_content()
        assert "Check out this photo" in result

    def test_content_array_includes_image_placeholder_by_default(self):
        msg = _make_msg(MULTIMODAL_CONTENT)
        result = msg.text_content(include_placeholders=True)
        assert "[image]" in result

    def test_content_array_excludes_placeholder_when_disabled(self):
        msg = _make_msg(MULTIMODAL_CONTENT)
        result = msg.text_content(include_placeholders=False)
        assert "[image]" not in result
        assert "Check out this photo" in result

    def test_multiple_text_parts_joined(self):
        content = [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]
        msg = _make_msg(content)
        assert msg.text_content() == "first second"

    def test_image_only_content_gives_placeholder(self):
        msg = _make_msg(IMAGE_ONLY_CONTENT)
        assert msg.text_content() == "[image]"

    def test_image_only_no_placeholder(self):
        msg = _make_msg(IMAGE_ONLY_CONTENT)
        assert msg.text_content(include_placeholders=False) == ""


class TestMessageHasImages:
    def test_string_content_no_images(self):
        assert _make_msg(TEXT_ONLY_CONTENT).has_images() is False

    def test_content_array_with_image(self):
        assert _make_msg(MULTIMODAL_CONTENT).has_images() is True

    def test_text_only_array_no_images(self):
        content = [{"type": "text", "text": "just text"}]
        assert _make_msg(content).has_images() is False

    def test_image_only_content(self):
        assert _make_msg(IMAGE_ONLY_CONTENT).has_images() is True


class TestMessageImageUrls:
    def test_string_content_empty_list(self):
        assert _make_msg(TEXT_ONLY_CONTENT).image_urls() == []

    def test_single_image_url(self):
        urls = _make_msg(MULTIMODAL_CONTENT).image_urls()
        assert urls == ["https://example.com/img1.png"]

    def test_multiple_image_urls(self):
        urls = _make_msg(MULTI_IMAGE_CONTENT).image_urls()
        assert urls == ["https://example.com/a.png", "https://example.com/b.png"]

    def test_no_images_in_array(self):
        content = [{"type": "text", "text": "hi"}]
        assert _make_msg(content).image_urls() == []


class TestMessageSerialization:
    """to_dict / from_dict round-trip with multimodal content."""

    def test_to_dict_preserves_content_array(self):
        msg = _make_msg(MULTIMODAL_CONTENT)
        d = msg.to_dict()
        assert d["content"] == MULTIMODAL_CONTENT
        assert d["role"] == "user"

    def test_to_dict_preserves_string_content(self):
        msg = _make_msg(TEXT_ONLY_CONTENT)
        assert msg.to_dict()["content"] == TEXT_ONLY_CONTENT

    def test_from_dict_with_content_array(self):
        d = {
            "role": "user",
            "content": MULTIMODAL_CONTENT,
            "timestamp": "2025-01-01T00:00:00",
        }
        msg = Message.from_dict(d)
        assert msg.has_images() is True
        assert msg.text_content() == "Check out this photo [image]"

    def test_round_trip_multimodal(self):
        original = _make_msg(MULTIMODAL_CONTENT)
        restored = Message.from_dict(original.to_dict())
        assert restored.content == original.content
        assert restored.role == original.role
        assert restored.has_images() == original.has_images()
        assert restored.image_urls() == original.image_urls()

    def test_round_trip_string(self):
        original = _make_msg(TEXT_ONLY_CONTENT)
        restored = Message.from_dict(original.to_dict())
        assert restored.content == original.content


# ===================================================================
# 2. EpisodeGenerator multimodal handling
# ===================================================================


class TestEpisodeGeneratorMultimodal:
    """Test _build_multimodal_prompt and _format_with_image_markers."""

    def _make_generator(self):
        """Create generator with None deps (we only test formatting helpers)."""
        return EpisodeGenerator(orchestrator=None, embedding=None)  # type: ignore[arg-type]

    def test_format_with_image_markers_text_only(self):
        gen = self._make_generator()
        msgs = [_make_msg("hello"), _make_msg("world", role="assistant")]
        result = gen._format_with_image_markers(msgs)
        assert "user: hello" in result
        assert "assistant: world" in result

    def test_format_with_image_markers_multimodal(self):
        gen = self._make_generator()
        msgs = [_make_msg(MULTIMODAL_CONTENT)]
        result = gen._format_with_image_markers(msgs)
        assert "Check out this photo" in result
        assert "[Image attached]" in result

    def test_build_multimodal_prompt_structure(self):
        gen = self._make_generator()
        msgs = [_make_msg(MULTIMODAL_CONTENT)]
        parts = gen._build_multimodal_prompt(msgs, "topic_change")
        # First part is the text prompt
        assert parts[0]["type"] == "text"
        assert isinstance(parts[0]["text"], str)
        # Second part is the image
        assert parts[1]["type"] == "image_url"
        assert parts[1]["image_url"]["url"] == "https://example.com/img1.png"

    def test_build_multimodal_prompt_caps_images(self):
        """Images are capped at MAX_IMAGES_PER_EPISODE."""
        gen = self._make_generator()
        # Create a message with more images than the cap
        many_images = [{"type": "text", "text": "lots of images"}] + [
            {"type": "image_url", "image_url": {"url": f"https://example.com/{i}.png"}}
            for i in range(MAX_IMAGES_PER_EPISODE + 5)
        ]
        msgs = [_make_msg(many_images)]
        parts = gen._build_multimodal_prompt(msgs, "overflow")
        image_parts = [p for p in parts if p.get("type") == "image_url"]
        assert len(image_parts) == MAX_IMAGES_PER_EPISODE

    def test_build_multimodal_prompt_no_images(self):
        """Text-only messages produce a text-only prompt part."""
        gen = self._make_generator()
        msgs = [_make_msg("just text")]
        parts = gen._build_multimodal_prompt(msgs, "reason")
        image_parts = [p for p in parts if p.get("type") == "image_url"]
        assert len(image_parts) == 0
        assert parts[0]["type"] == "text"

    def test_build_multimodal_prompt_includes_guidance(self):
        gen = self._make_generator()
        msgs = [_make_msg(MULTIMODAL_CONTENT)]
        parts = gen._build_multimodal_prompt(msgs, "reason")
        text_part = parts[0]["text"]
        assert "images are included" in text_part.lower() or "visual" in text_part.lower()


# ===================================================================
# 3. SemanticGenerator _extract_text helper
# ===================================================================


class TestExtractText:
    """Module-level _extract_text from semantic.py."""

    def test_string_content(self):
        assert _extract_text({"content": "hello"}) == "hello"

    def test_content_array_text_only(self):
        msg = {"content": [{"type": "text", "text": "foo"}, {"type": "text", "text": "bar"}]}
        assert _extract_text(msg) == "foo bar"

    def test_content_array_with_image(self):
        msg = {
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "https://x.com/i.png"}},
            ]
        }
        result = _extract_text(msg)
        assert "look" in result
        assert "[image]" in result

    def test_missing_content_key(self):
        assert _extract_text({}) == ""

    def test_empty_content_array(self):
        assert _extract_text({"content": []}) == ""

    def test_image_only_content(self):
        msg = {"content": [{"type": "image_url", "image_url": {"url": "https://x.com/i.png"}}]}
        assert _extract_text(msg) == "[image]"
