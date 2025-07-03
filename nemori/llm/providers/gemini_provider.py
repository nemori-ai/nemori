"""
Google Gemini LLM provider implementation using LangChain.
"""

import os

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from ..protocol import LLMProvider


class GeminiProvider(LLMProvider):
    """
    Google Gemini LLM provider using LangChain ChatGoogleGenerativeAI.

    Supports Gemini models via Google AI API.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = 16 * 1024,
        **kwargs,
    ):
        """
        Initialize Gemini provider.

        Args:
            model: Model name (e.g., "gemini-2.5-flash", "gemini-1.5-pro")
            api_key: Google AI API key (defaults to GOOGLE_API_KEY env var)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments passed to ChatGoogleGenerativeAI
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize LangChain ChatGoogleGenerativeAI
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key or os.getenv("GOOGLE_API_KEY"),
            temperature=temperature,
            max_output_tokens=max_tokens,
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
            temp_llm = ChatGoogleGenerativeAI(
                model=self.model,
                google_api_key=self.llm.google_api_key,
                temperature=effective_temperature,
                max_output_tokens=self.max_tokens,
            )
            llm_to_use = temp_llm
        else:
            llm_to_use = self.llm

        # Generate response asynchronously
        message = HumanMessage(content=prompt)
        response = await llm_to_use.ainvoke([message])

        return response.content

    @classmethod
    def from_env(cls, model: str = "gemini-2.5-flash", **kwargs) -> "GeminiProvider":
        """
        Create provider from environment variables.

        Expected environment variables:
        - GOOGLE_API_KEY: Google AI API key
        - GEMINI_MODEL: (Optional) Default model name

        Args:
            model: Model name (can be overridden by GEMINI_MODEL env var)
            **kwargs: Additional arguments

        Returns:
            Configured GeminiProvider instance
        """
        env_model = os.getenv("GEMINI_MODEL", model)

        return cls(model=env_model, **kwargs)

    async def test_connection(self) -> bool:
        """
        Test the connection to Google AI API.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = await self.generate("Hello", temperature=0.1)
            return bool(response and len(response.strip()) > 0)
        except Exception as e:
            print(f"Gemini connection test failed: {e}")
            return False

    def __repr__(self) -> str:
        return f"GeminiProvider(model='{self.model}', temperature={self.temperature})"
