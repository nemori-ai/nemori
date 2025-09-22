# 🧠 Nemori: Self-Organizing Agent Memory Inspired by Cognitive Science

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

> **Nemori** addresses the fundamental challenge of Large Language Model (LLM) amnesia by introducing a novel self-organizing memory architecture inspired by human cognitive principles. Moving beyond passive storage, Nemori enables autonomous agents to actively learn and evolve their knowledge through principled memory formation.

## 🎯 Core Innovation

Nemori implements a **dual-pillar cognitive framework** that transforms raw conversational streams into structured, queryable memory:

### 🔄 Two-Step Alignment Principle
- **Boundary Alignment**: Autonomously segments conversations into semantically coherent episodes using Event Segmentation Theory
- **Representation Alignment**: Transforms raw segments into rich episodic narratives inspired by human memory formation

### 🧠 Predict-Calibrate Principle  
- **Proactive Learning**: Generates predictions about new episodes based on existing knowledge
- **Knowledge Distillation**: Learns from prediction gaps using Free-energy Principle insights
- **Adaptive Evolution**: Continuously updates semantic knowledge base through error-driven learning

## 🚀 Quick Start

### Environment Setup with uv

We use [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management.

#### 1. Install uv (macOS)

```bash
# Using Homebrew
brew install uv

# Or using curl
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 2. Create and Activate Environment

```bash
# Clone the repository
git clone https://github.com/anonymous/nemori-code.git
cd nemori-code

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows

# Install all dependencies
uv pip install -e .

# Install optional dependencies for development
uv pip install -e ".[dev,evaluation]"

# For GPU support (optional)
uv pip install -e ".[gpu]"
```

#### 3. Environment Variables

Create a `.env` file in the project root with your API keys:

```bash
# Required for LLM operations
OPENAI_API_KEY=your_openai_api_key_here

# Optional: For advanced features
ANTHROPIC_API_KEY=your_anthropic_key_here
```

**Note**: You need to manually add these environment variables as shown above. The system doesn't have write access to `.env` files for security reasons.

### Basic Usage

```python
from src import MemorySystem, MemoryConfig
from src.models import Message

# Initialize Nemori memory system
config = MemoryConfig(
    llm_model="gpt-4o-mini",
    enable_semantic_memory=True,
    enable_prediction_correction=True
)

memory_system = MemorySystem(config)

# Add conversational memory
messages = [
    Message(role="user", content="I love hiking in the mountains"),
    Message(role="assistant", content="That sounds wonderful! What's your favorite trail?"),
    Message(role="user", content="I really enjoy the Pacific Crest Trail")
]

# System automatically creates episodes and semantic knowledge
result = memory_system.add_messages(
    owner_id="user123",
    messages=messages
)

# Retrieve relevant memories
memories = memory_system.search(
    owner_id="user123", 
    query="outdoor activities preferences",
    top_k_episodes=5,
    top_k_semantic=10
)
```

## 🏗️ Architecture Overview

The Nemori system implements three core computational modules that operationalize our cognitive principles:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Nemori Architecture                          │
├─────────────────────────────────────────────────────────────────┤
│  Raw Conversation Stream                                        │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────┐    Two-Step Alignment Principle       │
│  │ Topic Segmentation  │ ◄─ Boundary Alignment                 │
│  └─────────────────────┘                                       │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────┐                                       │
│  │ Episodic Memory     │ ◄─ Representation Alignment           │
│  │ Generation          │                                       │
│  └─────────────────────┘                                       │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────┐    Predict-Calibrate Principle        │
│  │ Semantic Memory     │ ◄─ Proactive Learning                 │
│  │ Generation          │                                       │
│  └─────────────────────┘                                       │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Source Code Structure

The `src/` directory implements Nemori's modular architecture:

```
src/
├── __init__.py                    # Main exports
├── config.py                     # System configuration
├── core/                         # Core system components
│   ├── memory_system.py         # Main MemorySystem class
│   ├── boundary_detector.py     # Topic segmentation (Boundary Alignment)
│   ├── message_buffer.py        # Conversation buffering
│   └── lightweight_search_system.py  # Optimized retrieval
├── models/                       # Data models
│   ├── message.py              # Message representation
│   ├── episode.py              # Episodic memory model
│   └── semantic.py             # Semantic memory model
├── generation/                   # Memory generation modules
│   ├── episode_generator.py    # Episodic memory creation (Representation Alignment)
│   ├── semantic_generator.py   # Semantic memory creation
│   ├── prediction_correction_engine.py  # Predict-Calibrate implementation
│   └── prompts.py              # LLM prompt templates
├── search/                      # Retrieval systems
│   ├── vector_search.py        # Dense vector retrieval
│   ├── bm25_search.py          # Sparse keyword retrieval  
│   ├── unified_search.py       # Hybrid search engine
│   └── original_message_search.py  # Raw conversation search
├── storage/                     # Persistence layer
│   ├── base_storage.py         # Abstract storage interface
│   ├── episode_storage.py      # Episodic memory storage
│   └── semantic_storage.py     # Semantic memory storage
└── utils/                       # Utility components
    ├── llm_client.py           # LLM API wrapper
    ├── embedding_client.py     # Embedding generation
    └── performance.py          # Performance optimization
```

### Key Components Explained

#### 🧠 Core System (`src/core/`)
- **`memory_system.py`**: Central orchestrator implementing the dual-pillar framework
- **`boundary_detector.py`**: LLM-based intelligent boundary detection for episode segmentation
- **`message_buffer.py`**: Manages conversational message accumulation and triggering

#### 🎭 Memory Generation (`src/generation/`)
- **`episode_generator.py`**: Transforms raw conversation segments into narrative episodic memories
- **`semantic_generator.py`**: Manages semantic knowledge extraction and evolution
- **`prediction_correction_engine.py`**: Implements the core Predict-Calibrate learning cycle

#### 🔍 Search & Retrieval (`src/search/`)
- **`unified_search.py`**: Hybrid retrieval combining vector and keyword search
- **`vector_search.py`**: Dense semantic similarity search using embeddings
- **`bm25_search.py`**: Sparse keyword-based retrieval for precise matching

## 📊 Evaluation & Benchmarks

Nemori is evaluated on two comprehensive datasets:

- **[LOCOMO](evaluation/readme.md#locomo-evaluation)**: Long-context memory benchmarks (24K avg tokens)
- **[LongMemEval_S](evaluation/readme.md#longmemeval-evaluation)**: Extended memory evaluation (105K avg tokens)

### Run Evaluations

```bash
# LOCOMO evaluation workflow
python evaluation/locomo/add.py
python evaluation/locomo/search.py  
python evaluation/locomo/evals.py
python evaluation/locomo/generate_scores.py

# LongMemEval evaluation workflow
python evaluation/longmemeval/add.py
python evaluation/longmemeval/search.py
python evaluation/longmemeval/evals.py
```

See the [evaluation README](evaluation/readme.md) for detailed setup and usage instructions.

## 🎯 Key Performance Results

| Method | LLM Score | Token Efficiency | Temporal Reasoning |
|--------|-----------|------------------|-------------------|
| **Nemori** | **0.794** | 88% reduction | **0.776** |
| Full Context | 0.806 | Baseline | 0.693 |
| RAG-4096 | 0.629 | - | 0.584 |
| Mem0 | 0.680 | - | 0.598 |

*Results on LoCoMo benchmark with gpt-4.1-mini. Nemori achieves near-optimal performance with significantly reduced token usage.*

## 🛠️ Advanced Configuration

### Memory System Configuration

```python
config = MemoryConfig(
    # LLM Settings
    llm_model="gpt-4o-mini",
    embedding_model="text-embedding-3-small",
    
    # Memory Settings
    enable_semantic_memory=True,
    enable_prediction_correction=True,
    
    # Boundary Detection
    boundary_confidence_threshold=0.7,
    enable_smart_boundary=True,
    
    # Episode Settings
    episode_min_messages=2
    episode_max_messages=25,
    
    # Search Settings
    search_top_k_episodes=10,
    search_top_k_semantic=20,
    
    # Performance
    semantic_generation_workers=8,
    use_faiss=False,
    batch_size=64
)
```

### Storage Options

```python
# Local file storage (default)
config.storage_path = "./memories"

# Custom storage backend
from src.storage import BaseStorage
config.storage_backend = MyCustomStorage()
```

## 🔬 Research Background

Nemori addresses fundamental limitations in current Memory-Augmented Generation (MAG) systems:

1. **Arbitrary Granularity**: Most systems use heuristic segmentation (single messages, interaction pairs) that breaks semantic coherence
2. **Passive Knowledge Extraction**: Current methods rely on simple summarization rather than active learning from prediction errors
3. **Limited Self-Organization**: Existing approaches lack principled mechanisms for autonomous memory evolution

Our approach draws inspiration from:
- **Event Segmentation Theory**: For principled boundary detection
- **Free-energy Principle**: For prediction-error driven learning  
- **Complementary Learning Systems**: For dual episodic-semantic architecture

### Development Setup

```bash
# Install development dependencies
uv pip install -e ".[dev]"
```


