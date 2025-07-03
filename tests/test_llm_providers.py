"""
Integration tests for LLM providers.

These tests require valid API keys in environment variables or .env file.
"""

import os

import pytest
from dotenv import load_dotenv

from nemori.llm.providers import AnthropicProvider, GeminiProvider, OpenAIProvider

# Load environment variables from .env file
load_dotenv()


class TestOpenAIProvider:
    """Test cases for OpenAI provider."""

    def test_init_with_defaults(self):
        """Test provider initialization with default parameters."""
        provider = OpenAIProvider()
        assert provider.model == "gpt-4o-mini"
        assert provider.temperature == 0.3
        assert provider.max_tokens == 16 * 1024

    def test_init_with_custom_params(self):
        """Test provider initialization with custom parameters."""
        provider = OpenAIProvider(model="gpt-4", temperature=0.7, max_tokens=1000)
        assert provider.model == "gpt-4"
        assert provider.temperature == 0.7
        assert provider.max_tokens == 1000

    def test_from_env(self):
        """Test provider creation from environment variables."""
        # Set temporary env vars
        original_model = os.getenv("OPENAI_MODEL")
        os.environ["OPENAI_MODEL"] = "gpt-4"

        try:
            provider = OpenAIProvider.from_env()
            assert provider.model == "gpt-4"
        finally:
            # Restore original env var
            if original_model:
                os.environ["OPENAI_MODEL"] = original_model
            elif "OPENAI_MODEL" in os.environ:
                del os.environ["OPENAI_MODEL"]

    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    @pytest.mark.asyncio
    async def test_generate_simple(self):
        """Test simple text generation."""
        provider = OpenAIProvider()
        response = await provider.generate("Say hello in one word.")

        assert isinstance(response, str)
        assert len(response.strip()) > 0
        print(f"OpenAI response: {response}")

    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    @pytest.mark.asyncio
    async def test_generate_with_temperature(self):
        """Test generation with custom temperature."""
        provider = OpenAIProvider()
        response = await provider.generate("Say hello", temperature=0.1)

        assert isinstance(response, str)
        assert len(response.strip()) > 0
        print(f"OpenAI response with temperature: {response}")

    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    @pytest.mark.asyncio
    async def test_connection(self):
        """Test API connection."""
        provider = OpenAIProvider()
        assert await provider.test_connection() is True


class TestAnthropicProvider:
    """Test cases for Anthropic provider."""

    def test_init_with_defaults(self):
        """Test provider initialization with default parameters."""
        provider = AnthropicProvider()
        assert provider.model == "claude-sonnet-4-20250514"
        assert provider.temperature == 0.3
        assert provider.max_tokens == 16 * 1024

    def test_init_with_custom_params(self):
        """Test provider initialization with custom parameters."""
        provider = AnthropicProvider(model="claude-3-5-sonnet-20241022", temperature=0.7, max_tokens=1000)
        assert provider.model == "claude-3-5-sonnet-20241022"
        assert provider.temperature == 0.7
        assert provider.max_tokens == 1000

    def test_from_env(self):
        """Test provider creation from environment variables."""
        # Set temporary env vars
        original_model = os.getenv("ANTHROPIC_MODEL")
        os.environ["ANTHROPIC_MODEL"] = "claude-3-5-sonnet-20241022"

        try:
            provider = AnthropicProvider.from_env()
            assert provider.model == "claude-3-5-sonnet-20241022"
        finally:
            # Restore original env var
            if original_model:
                os.environ["ANTHROPIC_MODEL"] = original_model
            elif "ANTHROPIC_MODEL" in os.environ:
                del os.environ["ANTHROPIC_MODEL"]

    @pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
    @pytest.mark.asyncio
    async def test_generate_simple(self):
        """Test simple text generation."""
        provider = AnthropicProvider()
        response = await provider.generate("Say hello in one word.")

        assert isinstance(response, str)
        assert len(response.strip()) > 0
        print(f"Anthropic response: {response}")

    @pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
    @pytest.mark.asyncio
    async def test_generate_with_temperature(self):
        """Test generation with custom temperature."""
        provider = AnthropicProvider()
        response = await provider.generate("Say hello", temperature=0.1)

        assert isinstance(response, str)
        assert len(response.strip()) > 0
        print(f"Anthropic response with temperature: {response}")

    @pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
    @pytest.mark.asyncio
    async def test_connection(self):
        """Test API connection."""
        provider = AnthropicProvider()
        assert await provider.test_connection() is True


class TestGeminiProvider:
    """Test cases for Gemini provider."""

    def test_init_with_defaults(self):
        """Test provider initialization with default parameters."""
        provider = GeminiProvider()
        assert provider.model == "gemini-2.5-flash"
        assert provider.temperature == 0.3
        assert provider.max_tokens == 16 * 1024

    def test_init_with_custom_params(self):
        """Test provider initialization with custom parameters."""
        provider = GeminiProvider(model="gemini-2.5-pro", temperature=0.7, max_tokens=1000)
        assert provider.model == "gemini-2.5-pro"
        assert provider.temperature == 0.7
        assert provider.max_tokens == 1000

    def test_from_env(self):
        """Test provider creation from environment variables."""
        # Set temporary env vars
        original_model = os.getenv("GEMINI_MODEL")
        os.environ["GEMINI_MODEL"] = "gemini-2.5-pro"

        try:
            provider = GeminiProvider.from_env()
            assert provider.model == "gemini-2.5-pro"
        finally:
            # Restore original env var
            if original_model:
                os.environ["GEMINI_MODEL"] = original_model
            elif "GEMINI_MODEL" in os.environ:
                del os.environ["GEMINI_MODEL"]

    @pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set")
    @pytest.mark.asyncio
    async def test_generate_simple(self):
        """Test simple text generation."""
        provider = GeminiProvider()
        response = await provider.generate("Say hello in one word.")

        assert isinstance(response, str)
        assert len(response.strip()) > 0
        print(f"Gemini response: {response}")

    @pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set")
    @pytest.mark.asyncio
    async def test_generate_with_temperature(self):
        """Test generation with custom temperature."""
        provider = GeminiProvider()
        response = await provider.generate("Say hello", temperature=0.1)

        assert isinstance(response, str)
        assert len(response.strip()) > 0
        print(f"Gemini response with temperature: {response}")

    @pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set")
    @pytest.mark.asyncio
    async def test_connection(self):
        """Test API connection."""
        provider = GeminiProvider()
        assert await provider.test_connection() is True


class TestProviderIntegration:
    """Integration tests using providers with conversation builder."""

    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    @pytest.mark.asyncio
    async def test_openai_with_conversation_builder(self):
        """Test OpenAI provider with conversation builder."""
        from datetime import datetime

        from nemori.builders.conversation_builder import ConversationEpisodeBuilder
        from nemori.core.data_types import (
            ConversationMessage,
            DataType,
            RawEventData,
            TemporalInfo,
        )

        # Create test conversation data
        messages = [
            ConversationMessage(
                speaker_id="user1", user_name="Alice", content="Hello, how are you?", timestamp=datetime.now()
            ),
            ConversationMessage(
                speaker_id="assistant",
                user_name="Assistant",
                content="I'm doing well, thank you! How can I help you today?",
                timestamp=datetime.now(),
            ),
        ]

        # Convert messages to dict format with proper timestamp serialization
        message_dicts = []
        for msg in messages:
            msg_dict = msg.__dict__.copy()
            if msg_dict["timestamp"]:
                msg_dict["timestamp"] = msg_dict["timestamp"].isoformat()
            message_dicts.append(msg_dict)

        raw_data = RawEventData(
            data_type=DataType.CONVERSATION,
            source="test",
            content=message_dicts,
            temporal_info=TemporalInfo(timestamp=datetime.now()),
        )

        # Create builder with OpenAI provider
        provider = OpenAIProvider()
        builder = ConversationEpisodeBuilder(llm_provider=provider)

        # Build episode
        episode = await builder.build_episode(raw_data, for_owner="user1")

        assert episode.title is not None
        assert episode.content is not None
        assert episode.summary is not None
        print(f"Generated episode title: {episode.title}")
        print(f"Generated episode content: {episode.content[:200]}...")


def test_all_providers_repr():
    """Test string representation of all providers."""
    openai_provider = OpenAIProvider()
    anthropic_provider = AnthropicProvider()
    gemini_provider = GeminiProvider()

    assert "OpenAIProvider" in str(openai_provider)
    assert "AnthropicProvider" in str(anthropic_provider)
    assert "GeminiProvider" in str(gemini_provider)


if __name__ == "__main__":
    # Run basic connectivity tests if API keys are available
    print("Testing LLM Providers...")

    import asyncio

    async def test_connections():
        if os.getenv("OPENAI_API_KEY"):
            print("Testing OpenAI connection...")
            openai = OpenAIProvider()
            if await openai.test_connection():
                print("✓ OpenAI connection successful")
            else:
                print("✗ OpenAI connection failed")

        if os.getenv("ANTHROPIC_API_KEY"):
            print("Testing Anthropic connection...")
            anthropic = AnthropicProvider()
            if await anthropic.test_connection():
                print("✓ Anthropic connection successful")
            else:
                print("✗ Anthropic connection failed")

        if os.getenv("GOOGLE_API_KEY"):
            print("Testing Gemini connection...")
            gemini = GeminiProvider()
            if await gemini.test_connection():
                print("✓ Gemini connection successful")
            else:
                print("✗ Gemini connection failed")

    asyncio.run(test_connections())
