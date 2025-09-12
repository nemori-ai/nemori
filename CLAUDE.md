# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Testing
- Run all tests: `pytest`
- Run specific test file: `pytest tests/test_semantic_memory.py`
- Run with coverage: `pytest --cov=nemori`
- Run integration tests: `pytest -m integration`
- Run specific test markers: `pytest -m semantic` or `pytest -m episodic`

### Code Quality
- Format code: `black nemori tests`
- Lint code: `ruff check nemori tests`
- Fix linting issues: `ruff check --fix nemori tests`

### Development
- Install dev dependencies: `pip install -e ".[dev]"`
- Run semantic memory demo: `python simple_semantic_demo.py`
- Run enhanced demo: `python enhanced_semantic_demo.py`

## Architecture Overview

Nemori is a nature-inspired episodic memory system that transforms user experiences into searchable, narrative episodic memories with semantic knowledge discovery.

### Core Architecture Layers

**Data Flow**: Raw Events → Episodes → Semantic Knowledge → Unified Retrieval

#### 1. Data Types & Episode Management (`nemori/core/`)
- **RawEventData**: Unified interface for all input data types (conversation, activity, media, etc.)
- **Episode**: Structured narrative memories with metadata, importance scoring, and temporal information
- **EpisodeManager**: High-level coordinator for the complete episode lifecycle (ingestion → processing → storage → indexing)

#### 2. Episode Building (`nemori/builders/`)
- **EpisodeBuilderRegistry**: Factory pattern for type-specific episode builders
- **ConversationEpisodeBuilder**: Transforms conversation data into episodic narratives using LLM prompts
- **EnhancedConversationEpisodeBuilder**: Advanced version with semantic integration

#### 3. Storage Layer (`nemori/storage/`)
Multiple backend implementations via repository pattern:
- **Memory**: In-memory storage for testing
- **DuckDB**: Local file-based storage (default for development)
- **PostgreSQL**: Production database storage
- **JSONL**: File-based append-only storage

All repositories implement abstract interfaces (`RawDataRepository`, `EpisodicMemoryRepository`, `SemanticMemoryRepository`).

#### 4. Retrieval System (`nemori/retrieval/`)
Extensible retrieval with multiple strategies:
- **BM25Provider**: Token-based sparse retrieval (primary method in LoCoMo benchmark)
- **EmbeddingProvider**: Dense vector similarity search
- **RetrievalService**: Coordinates multiple providers with query routing

#### 5. Semantic Memory (`nemori/semantic/`)
Advanced knowledge discovery system:
- **ContextAwareSemanticDiscoveryEngine**: Extracts private domain knowledge through episode comparison
- **SemanticEvolutionManager**: Manages knowledge evolution and confidence tracking
- **UnifiedRetrievalService**: Combines episodic and semantic search for enhanced recall

### Key Design Patterns

**Repository Pattern**: Clean separation between business logic and storage implementation
**Builder Pattern**: Type-specific episode creation with extensible registration system
**Provider Pattern**: Pluggable retrieval strategies with unified interfaces
**Factory Pattern**: Configuration-driven component instantiation

### Data Models

**SemanticNode**: Core knowledge unit with key-value structure, confidence scoring, and bidirectional episode links
**SemanticRelationship**: Simple associations between knowledge nodes
**Episode**: Rich narrative structure with temporal information, importance scoring, and access tracking

### Configuration & Setup

Storage backends are configured through `StorageConfig` dataclasses:
```python
# DuckDB (development)
config = create_duckdb_config("path/to/db.duckdb")

# PostgreSQL (production)
config = create_postgresql_config(
    host="localhost", database="nemori",
    username="user", password="pass"
)
```

### Core Philosophy

- **Episodic Memory Granularity**: Episodes align with human memory patterns and LLM training distributions
- **Minimal LLM Usage**: LoCoMo benchmark uses only 2 prompts + BM25 retrieval
- **Semantic Gap Analysis**: Private knowledge discovery through differential episode analysis
- **Bidirectional Associations**: Episodes ↔ Semantic Knowledge linking for enhanced recall

### Testing Structure

Tests are organized by component with specific markers:
- `unit`: Fast unit tests
- `integration`: Cross-component integration tests
- `semantic`: Semantic memory functionality
- `episodic`: Core episodic memory features
- `llm`: Tests requiring LLM providers (require API keys)