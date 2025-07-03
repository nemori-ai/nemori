# Claude Assistant Notes

## Project: Nemori
Episodic memory system for transforming raw user data into structured narrative episodes.

## Pre-Development Requirements
- **Always read top-level documentation first** - Understand project architecture design principles and approach
- Review README.md, architecture docs, and design patterns before making changes
- Familiarize with the episodic memory system's core concepts and data flow

## Package Management
- **Use `uv` for all package management operations**
- Install dependencies: `uv sync`
- Install in development mode: `uv pip install -e .`
- Add dependencies: `uv add <package>`
- Run tests: `uv run pytest`
- Run linting: `uv run black .` and `uv run ruff check .`
- Code formatting uses 120 character line length for better readability

## Project Structure
- Core modules: `nemori/core/` - data types, episodes, builders
- LLM providers: `nemori/llm/providers/` - OpenAI, Anthropic, Gemini
- Conversation builder: `nemori/builders/conversation_builder.py`
- Tests: `tests/` - comprehensive test suite with fixtures

## Dependencies
- LangChain packages for LLM integration
- python-dotenv for environment variables
- pytest for testing
- black and ruff for code formatting/linting

## Testing
- Use pytest markers: `unit`, `integration`, `llm`, `slow`
- LLM tests require API keys in environment variables
- Mock LLM provider available in conftest.py for testing