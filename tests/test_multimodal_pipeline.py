"""Tests for multimodal message flow through the pipeline."""
from __future__ import annotations

import json
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from nemori.domain.models import Message, Episode
from nemori.db.buffer_store import PgMessageBufferStore
from nemori.llm.generators.episode import EpisodeGenerator
from nemori.llm.generators.segmenter import BatchSegmenter
from nemori.llm.generators.semantic import SemanticGenerator, _extract_text


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

    def test_build_multimodal_prompt_all_images_included(self):
        """All images are included (no artificial cap)."""
        gen = self._make_generator()
        num_images = 15
        many_images = [{"type": "text", "text": "lots of images"}] + [
            {"type": "image_url", "image_url": {"url": f"https://example.com/{i}.png"}}
            for i in range(num_images)
        ]
        msgs = [_make_msg(many_images)]
        parts = gen._build_multimodal_prompt(msgs, "overflow")
        image_parts = [p for p in parts if p.get("type") == "image_url"]
        assert len(image_parts) == num_images

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


# ===================================================================
# 4. Buffer storage round-trip (PgMessageBufferStore)
# ===================================================================


class TestBufferStoreMultimodalRoundTrip:
    """push() -> get_unprocessed() preserves multimodal content arrays."""

    @pytest.mark.asyncio
    async def test_push_stores_content_as_json(self):
        """push() serialises multimodal content via json.dumps."""
        mock_db = AsyncMock()
        store = PgMessageBufferStore(mock_db)
        msg = _make_msg(MULTIMODAL_CONTENT)

        await store.push("u1", "agent1", [msg])

        mock_db.execute.assert_called_once()
        call_args = mock_db.execute.call_args[0]
        # positional args: (sql, user_id, agent_id, role, content_json, timestamp)
        stored_json = call_args[4]
        assert json.loads(stored_json) == MULTIMODAL_CONTENT

    @pytest.mark.asyncio
    async def test_get_unprocessed_preserves_content_array(self):
        """get_unprocessed() returns Message objects with list content intact."""
        mock_db = AsyncMock()
        ts = datetime(2025, 1, 1, 12, 0, 0)
        mock_db.fetch.return_value = [
            {
                "id": 42,
                "role": "user",
                "content": MULTIMODAL_CONTENT,  # asyncpg returns Python objects from JSONB
                "timestamp": ts,
            }
        ]
        store = PgMessageBufferStore(mock_db)
        messages = await store.get_unprocessed("u1", "agent1")

        assert len(messages) == 1
        msg = messages[0]
        assert isinstance(msg.content, list)
        assert msg.content == MULTIMODAL_CONTENT
        assert msg.has_images() is True
        assert msg.image_urls() == ["https://example.com/img1.png"]

    @pytest.mark.asyncio
    async def test_round_trip_text_content(self):
        """push() -> get_unprocessed() round-trip with plain string content."""
        mock_db = AsyncMock()
        ts = datetime(2025, 1, 1)
        # push
        store = PgMessageBufferStore(mock_db)
        msg = _make_msg("plain text")
        await store.push("u1", "a1", [msg])
        stored_json = mock_db.execute.call_args[0][4]

        # simulate get_unprocessed returning the stored data
        mock_db.fetch.return_value = [
            {"id": 1, "role": "user", "content": json.loads(stored_json), "timestamp": ts}
        ]
        result = await store.get_unprocessed("u1", "a1")
        assert result[0].content == "plain text"

    @pytest.mark.asyncio
    async def test_round_trip_multi_image_content(self):
        """Round-trip with multiple images preserves all image URLs."""
        mock_db = AsyncMock()
        ts = datetime(2025, 6, 1)
        mock_db.fetch.return_value = [
            {"id": 7, "role": "user", "content": MULTI_IMAGE_CONTENT, "timestamp": ts}
        ]
        store = PgMessageBufferStore(mock_db)
        msgs = await store.get_unprocessed("u1", "a1")
        assert msgs[0].image_urls() == [
            "https://example.com/a.png",
            "https://example.com/b.png",
        ]


# ===================================================================
# 5. Segmenter with multimodal messages
# ===================================================================


class TestBatchSegmenterMultimodal:
    """BatchSegmenter handles messages with images via text_content()."""

    @pytest.mark.asyncio
    async def test_segment_uses_text_content_for_multimodal(self):
        """Images are replaced with [image] placeholder in the prompt sent to LLM."""
        mock_orch = AsyncMock()
        mock_orch.execute.return_value = MagicMock(
            content=json.dumps({
                "episodes": [{"indices": [1, 2], "topic": "photo discussion"}]
            })
        )
        segmenter = BatchSegmenter(mock_orch)
        msgs = [
            _make_msg(MULTIMODAL_CONTENT),
            _make_msg("That's a nice photo", role="assistant"),
        ]
        groups = await segmenter.segment(msgs)

        # Verify the prompt sent to LLM
        call_args = mock_orch.execute.call_args[0][0]
        prompt_text = call_args.messages[0]["content"]
        # Should contain the text_content() output (with [image] placeholder)
        assert "Check out this photo" in prompt_text
        assert "[image]" in prompt_text
        # Should NOT contain raw image URLs in the formatted prompt
        assert "https://example.com/img1.png" not in prompt_text

        # Verify grouping result
        assert len(groups) == 1
        assert len(groups[0]["messages"]) == 2
        assert groups[0]["topic"] == "photo discussion"

    @pytest.mark.asyncio
    async def test_segment_fallback_on_failure_preserves_multimodal(self):
        """On LLM failure, fallback returns all messages including multimodal ones."""
        mock_orch = AsyncMock()
        mock_orch.execute.side_effect = Exception("LLM down")
        segmenter = BatchSegmenter(mock_orch)
        msgs = [_make_msg(MULTIMODAL_CONTENT), _make_msg("text only")]
        groups = await segmenter.segment(msgs)

        assert len(groups) == 1
        assert groups[0]["messages"] == msgs
        assert groups[0]["messages"][0].has_images() is True

    @pytest.mark.asyncio
    async def test_segment_image_only_message(self):
        """A message with only an image is segmented using [image] placeholder."""
        mock_orch = AsyncMock()
        mock_orch.execute.return_value = MagicMock(
            content=json.dumps({
                "episodes": [{"indices": [1], "topic": "image"}]
            })
        )
        segmenter = BatchSegmenter(mock_orch)
        msgs = [_make_msg(IMAGE_ONLY_CONTENT)]
        groups = await segmenter.segment(msgs)

        prompt_text = mock_orch.execute.call_args[0][0].messages[0]["content"]
        assert "[image]" in prompt_text
        assert len(groups) == 1


# ===================================================================
# 6. add_messages() with pre-built multimodal content
# ===================================================================


class TestFacadeAddMessagesMultimodal:
    """Test add_messages() directly with a multimodal content array."""

    @pytest.mark.asyncio
    async def test_add_messages_with_content_array(self):
        """Calling add_messages with a content list creates a Message with list content."""
        with patch("nemori.api.facade.DatabaseManager") as MockDB, \
             patch("nemori.api.facade.QdrantVectorStore") as MockQdrant, \
             patch("nemori.api.facade.NemoriMemory._build_system", new_callable=AsyncMock):
            MockDB.return_value = AsyncMock()
            MockQdrant.return_value = MagicMock()

            from nemori.api.facade import NemoriMemory
            from nemori.config import MemoryConfig

            config = MemoryConfig(dsn="postgresql://localhost/test", llm_api_key="test")
            async with NemoriMemory(config=config) as memory:
                memory._system = AsyncMock()
                await memory.add_messages("u1", [
                    {"role": "user", "content": MULTIMODAL_CONTENT}
                ])
                memory._system.add_messages.assert_called_once()
                msg = memory._system.add_messages.call_args[0][1][0]
                assert isinstance(msg.content, list)
                assert msg.content == MULTIMODAL_CONTENT
                assert msg.has_images() is True
                assert msg.text_content() == "Check out this photo [image]"

    @pytest.mark.asyncio
    async def test_add_messages_multimodal_with_timestamp(self):
        """Multimodal message with explicit timestamp is preserved."""
        with patch("nemori.api.facade.DatabaseManager") as MockDB, \
             patch("nemori.api.facade.QdrantVectorStore") as MockQdrant, \
             patch("nemori.api.facade.NemoriMemory._build_system", new_callable=AsyncMock):
            MockDB.return_value = AsyncMock()
            MockQdrant.return_value = MagicMock()

            from nemori.api.facade import NemoriMemory
            from nemori.config import MemoryConfig

            config = MemoryConfig(dsn="postgresql://localhost/test", llm_api_key="test")
            async with NemoriMemory(config=config) as memory:
                memory._system = AsyncMock()
                await memory.add_messages("u1", [
                    {
                        "role": "user",
                        "content": MULTI_IMAGE_CONTENT,
                        "timestamp": "2025-06-01T10:00:00",
                    }
                ])
                msg = memory._system.add_messages.call_args[0][1][0]
                assert isinstance(msg.content, list)
                assert msg.timestamp == datetime(2025, 6, 1, 10, 0, 0)
                assert msg.image_urls() == [
                    "https://example.com/a.png",
                    "https://example.com/b.png",
                ]


# ===================================================================
# 7. Episode.to_dict() preserves image_url content
# ===================================================================


class TestEpisodeToDictMultimodal:
    """Episode.to_dict() preserves multimodal source_messages."""

    def test_to_dict_preserves_image_url_in_source_messages(self):
        source_msgs = [
            {"role": "user", "content": MULTIMODAL_CONTENT},
            {"role": "assistant", "content": "Nice photo!"},
        ]
        ep = Episode(
            user_id="u1",
            title="Photo chat",
            content="User shared a photo.",
            source_messages=source_msgs,
        )
        d = ep.to_dict()
        assert d["source_messages"] == source_msgs
        # Verify the image_url is accessible
        user_msg = d["source_messages"][0]
        assert user_msg["content"][1]["type"] == "image_url"
        assert user_msg["content"][1]["image_url"]["url"] == "https://example.com/img1.png"

    def test_to_dict_round_trip_preserves_multimodal(self):
        source_msgs = [{"role": "user", "content": MULTI_IMAGE_CONTENT}]
        ep = Episode(
            user_id="u1",
            title="Multi image",
            content="Multiple images shared.",
            source_messages=source_msgs,
        )
        restored = Episode.from_dict(ep.to_dict())
        assert restored.source_messages == source_msgs
        assert restored.source_messages[0]["content"][1]["image_url"]["url"] == "https://example.com/a.png"
        assert restored.source_messages[0]["content"][2]["image_url"]["url"] == "https://example.com/b.png"

    def test_to_dict_mixed_source_messages(self):
        """Episode with both text-only and multimodal source_messages."""
        source_msgs = [
            {"role": "user", "content": "plain text"},
            {"role": "user", "content": IMAGE_ONLY_CONTENT},
            {"role": "assistant", "content": "I see the image."},
        ]
        ep = Episode(
            user_id="u1",
            title="Mixed",
            content="Mixed conversation.",
            source_messages=source_msgs,
        )
        d = ep.to_dict()
        assert d["source_messages"][0]["content"] == "plain text"
        assert d["source_messages"][1]["content"][0]["type"] == "image_url"
        assert d["source_messages"][2]["content"] == "I see the image."


# ===================================================================
# 8. SemanticGenerator with multimodal episodes
# ===================================================================


class TestSemanticGeneratorMultimodal:
    """SemanticGenerator handles episodes with multimodal source_messages."""

    def _make_episode_with_images(self) -> Episode:
        return Episode(
            user_id="u1",
            title="Photo discussion",
            content="User shared a photo of their dog.",
            source_messages=[
                {"role": "user", "content": MULTIMODAL_CONTENT},
                {"role": "assistant", "content": "That's a cute dog!"},
            ],
        )

    @pytest.mark.asyncio
    async def test_direct_extraction_with_multimodal_episode(self):
        """Direct extraction works even if source_messages have images."""
        mock_orch = AsyncMock()
        mock_orch.execute.return_value = MagicMock(
            content=json.dumps({"statements": ["User has a dog"]})
        )
        mock_embed = AsyncMock()
        mock_embed.embed.return_value = [0.1] * 10

        gen = SemanticGenerator(mock_orch, mock_embed, enable_prediction_correction=False)
        ep = self._make_episode_with_images()
        memories = await gen.generate("u1", "a1", ep, [], [])

        assert len(memories) == 1
        assert memories[0].content == "User has a dog"

    @pytest.mark.asyncio
    async def test_prediction_correction_uses_extract_text(self):
        """_prediction_correction formats multimodal source_messages via _extract_text."""
        mock_orch = AsyncMock()
        # First call: prediction response
        predict_resp = MagicMock(content="The user likes dogs.")
        # Second call: extraction response
        extract_resp = MagicMock(
            content=json.dumps({"statements": ["User likes dogs"]})
        )
        mock_orch.execute.side_effect = [predict_resp, extract_resp]

        mock_embed = AsyncMock()
        mock_embed.embed.return_value = [0.1] * 10

        existing_semantic = MagicMock()
        existing_semantic.content = "User is friendly"

        gen = SemanticGenerator(mock_orch, mock_embed, enable_prediction_correction=True)
        ep = self._make_episode_with_images()
        memories = await gen.generate("u1", "a1", ep, [], [existing_semantic])

        assert len(memories) == 1
        # Verify the extract call used _extract_text (contains [image] placeholder)
        extract_call = mock_orch.execute.call_args_list[1][0][0]
        extract_prompt = extract_call.messages[0]["content"]
        assert "[image]" in extract_prompt
        assert "Check out this photo" in extract_prompt

    @pytest.mark.asyncio
    async def test_extract_text_called_for_each_source_message(self):
        """Each source_message is formatted via _extract_text in prediction_correction."""
        ep = self._make_episode_with_images()
        # Manually verify _extract_text works for each message
        for msg_dict in ep.source_messages:
            text = _extract_text(msg_dict)
            assert isinstance(text, str)
            assert len(text) > 0


# ===================================================================
# 9. EpisodeGenerator fallback with multimodal messages
# ===================================================================


class TestEpisodeGeneratorFallbackMultimodal:
    """EpisodeGenerator._create_fallback() with multimodal messages."""

    def _make_generator(self):
        return EpisodeGenerator(orchestrator=None, embedding=None)  # type: ignore[arg-type]

    def test_fallback_preserves_image_content_in_source_messages(self):
        """source_messages in fallback episode preserve multimodal content."""
        gen = self._make_generator()
        msgs = [
            _make_msg(MULTIMODAL_CONTENT),
            _make_msg("Got it", role="assistant"),
        ]
        ep = gen._create_fallback("u1", "a1", msgs, "timeout")

        # source_messages should be to_dict() of each message
        assert len(ep.source_messages) == 2
        assert ep.source_messages[0]["content"] == MULTIMODAL_CONTENT
        assert ep.source_messages[0]["content"][1]["type"] == "image_url"

    def test_fallback_content_uses_text_content(self):
        """The content field uses text_content() which replaces images with [image]."""
        gen = self._make_generator()
        msgs = [_make_msg(MULTIMODAL_CONTENT)]
        ep = gen._create_fallback("u1", "a1", msgs, "timeout")

        assert "Check out this photo" in ep.content
        assert "[image]" in ep.content
        # Raw URL should NOT appear in the text content field
        assert "https://example.com/img1.png" not in ep.content

    def test_fallback_title_includes_message_count(self):
        gen = self._make_generator()
        msgs = [_make_msg(MULTIMODAL_CONTENT), _make_msg("reply", role="assistant")]
        ep = gen._create_fallback("u1", "a1", msgs, "timeout")
        assert "2 messages" in ep.title

    def test_fallback_metadata_marked_as_fallback(self):
        gen = self._make_generator()
        msgs = [_make_msg(IMAGE_ONLY_CONTENT)]
        ep = gen._create_fallback("u1", "a1", msgs, "error")
        assert ep.metadata.get("fallback") is True
        assert ep.metadata.get("boundary_reason") == "error"

    def test_fallback_image_only_message(self):
        """Image-only message produces [image] in content, preserves data in source_messages."""
        gen = self._make_generator()
        msgs = [_make_msg(IMAGE_ONLY_CONTENT)]
        ep = gen._create_fallback("u1", "a1", msgs, "test")

        assert ep.content == "user: [image]"
        assert ep.source_messages[0]["content"] == IMAGE_ONLY_CONTENT
