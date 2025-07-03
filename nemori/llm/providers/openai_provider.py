"""
OpenAI LLM provider implementation using LangChain.
"""

import os

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from ..protocol import LLMProvider


class OpenAIProvider(LLMProvider):
    """
    OpenAI LLM provider using LangChain ChatOpenAI.

    Supports both OpenAI API and Azure OpenAI deployments.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = 16 * 1024,
        **kwargs,
    ):
        """
        Initialize OpenAI provider.

        Args:
            model: Model name (e.g., "gpt-4o-mini", "gpt-4o")
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            base_url: Custom base URL for API (for Azure OpenAI or custom endpoints)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments passed to ChatOpenAI
        """
        self.model = model
        self.temperature = temperature
        print(f"max_tokens: {max_tokens}")
        self.max_tokens = max_tokens

        # Initialize LangChain ChatOpenAI
        self.llm = ChatOpenAI(
            model=model,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
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
            temp_llm = ChatOpenAI(
                model=self.model,
                api_key=self.llm.openai_api_key,
                base_url=self.llm.openai_api_base,
                temperature=effective_temperature,
                max_tokens=self.max_tokens,
            )
            llm_to_use = temp_llm
        else:
            llm_to_use = self.llm

        # Generate response asynchronously
        message = HumanMessage(content=prompt)
        response = await llm_to_use.ainvoke([message])

        return response.content

    @classmethod
    def from_env(cls, model: str = "gpt-4o-mini", **kwargs) -> "OpenAIProvider":
        """
        Create provider from environment variables.

        Expected environment variables:
        - OPENAI_API_KEY: OpenAI API key
        - OPENAI_BASE_URL: (Optional) Custom base URL
        - OPENAI_MODEL: (Optional) Default model name

        Args:
            model: Model name (can be overridden by OPENAI_MODEL env var)
            **kwargs: Additional arguments

        Returns:
            Configured OpenAIProvider instance
        """
        env_model = os.getenv("OPENAI_MODEL", model)
        base_url = os.getenv("OPENAI_BASE_URL")

        return cls(model=env_model, base_url=base_url, **kwargs)

    async def test_connection(self) -> bool:
        """
        Test the connection to OpenAI API.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = await self.generate("Hello", temperature=0.1)
            return bool(response and len(response.strip()) > 0)
        except Exception as e:
            print(f"OpenAI connection test failed: {e}")
            return False

    def __repr__(self) -> str:
        return f"OpenAIProvider(model='{self.model}', temperature={self.temperature})"
