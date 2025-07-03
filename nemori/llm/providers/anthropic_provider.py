"""
Anthropic LLM provider implementation using LangChain.
"""

import os

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from ..protocol import LLMProvider


class AnthropicProvider(LLMProvider):
    """
    Anthropic LLM provider using LangChain ChatAnthropic.

    Supports Claude models via Anthropic API.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = 16 * 1024,
        **kwargs,
    ):
        """
        Initialize Anthropic provider.

        Args:
            model: Model name (e.g., "claude-sonnet-4-20250514", "claude-3-5-sonnet-20241022")
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments passed to ChatAnthropic
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize LangChain ChatAnthropic

        self.llm = ChatAnthropic(
            model=model,
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    async def generate(self, prompt: str, temperature: float | None = None) -> str:
        """
        Generate a response for the given prompt.

        Args:
            prompt: Input prompt
            temperature: Override temperature for this request

        Returns:
            Generated response text
        """
        # Use provided temperature or default
        effective_temperature = temperature if temperature is not None else self.temperature

        # Create a temporary LLM instance if temperature differs
        if effective_temperature != self.temperature:
            effective_max_tokens = self.max_tokens if self.max_tokens is not None else 1024
            temp_llm = ChatAnthropic(
                model=self.model,
                api_key=self.llm.anthropic_api_key,
                temperature=effective_temperature,
                max_tokens=effective_max_tokens,
            )
            llm_to_use = temp_llm
        else:
            llm_to_use = self.llm

        # Generate response asynchronously
        message = HumanMessage(content=prompt)
        response = await llm_to_use.ainvoke([message])

        return response.content

    @classmethod
    def from_env(cls, model: str = "claude-sonnet-4-20250514", **kwargs) -> "AnthropicProvider":
        """
        Create provider from environment variables.

        Expected environment variables:
        - ANTHROPIC_API_KEY: Anthropic API key
        - ANTHROPIC_MODEL: (Optional) Default model name

        Args:
            model: Model name (can be overridden by ANTHROPIC_MODEL env var)
            **kwargs: Additional arguments

        Returns:
            Configured AnthropicProvider instance
        """
        env_model = os.getenv("ANTHROPIC_MODEL", model)

        return cls(model=env_model, **kwargs)

    async def test_connection(self) -> bool:
        """
        Test the connection to Anthropic API.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = await self.generate("Hello", temperature=0.1)
            return bool(response and len(response.strip()) > 0)
        except Exception as e:
            print(f"Anthropic connection test failed: {e}")
            return False

    def __repr__(self) -> str:
        return f"AnthropicProvider(model='{self.model}', temperature={self.temperature})"
