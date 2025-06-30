"""
LLM Provider Protocol for Nemori.

This module defines the abstract interface that all LLM providers must implement.
Keeping this separate avoids circular import dependencies.
"""

from typing import Protocol


class LLMProvider(Protocol):
    """
    Protocol for LLM providers used in episode generation.

    All concrete LLM provider implementations must implement this interface
    to be compatible with Nemori's episode builders.
    """

    def generate(self, prompt: str, temperature: float | None = None) -> str:
        """
        Generate a response for the given prompt.

        Args:
            prompt: Input prompt text
            temperature: Optional temperature override for this request

        Returns:
            Generated response text

        Raises:
            Exception: If generation fails
        """
        ...

    def test_connection(self) -> bool:
        """
        Test the connection to the LLM provider.

        Returns:
            True if connection successful, False otherwise
        """
        ...

    def __repr__(self) -> str:
        """String representation of the provider."""
        ...
