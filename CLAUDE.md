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

## Code Formatting Standards
- **Line length**: 120 characters maximum
- **Quote style**: Use double quotes (`"`) for strings (default Python style)
- **Tools**: Black formatter with standard settings and Ruff linter with flake8-quotes
- **Commands**:
  - Format code: `uv run black .`
  - Check/fix linting: `uv run ruff check . --fix`
  - Both together: `uv run black . && uv run ruff check . --fix`
- **Configuration**: All formatting rules are defined in `pyproject.toml`

## Project Structure
- Core modules: `nemori/core/` - data types (episodic + semantic), episodes, builders
- Storage layer: `nemori/storage/` - repository interfaces and implementations
- Retrieval system: `nemori/retrieval/` - search providers and unified services
- LLM providers: `nemori/llm/providers/` - OpenAI, Anthropic, Gemini
- Builders: `nemori/builders/` - episode builders and registry
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