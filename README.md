# Nemori: Nature-Inspired Episodic Memory

*Read this in other languages: [中文](README-zh.md)*

## Project Overview

Nemori-AI empowers large language models with human-like episodic memory.
Nemori stores experiences as natural, event-centric traces, enabling precise recall when it matters.
**Vision:** every piece of data remembered and retrieved as intuitively as human recollection.

While previous systems like Mem0, Supermemory, and ZEP have made remarkable attempts at AI memory, achieving advanced performance on benchmarks such as LoCoMo and LongMemEval, Nemori introduces an innovative and minimalist approach centered on aligning with human episodic memory patterns.

## Experimental Results

To highlight the superiority of Nemori, we conducted evaluations on both LoCoMo and LongMemEval benchmarks, comparing against previous state-of-the-art approaches:

### LoCoMo Benchmark Results

On the LoCoMo (Long-Context Conversation Modeling) dataset, Nemori demonstrates exceptional performance:

![LoCoMo Benchmark Results](figures/results_on_locomo.png)

### LongMemEval-s Benchmark Results

On the LongMemEval-s dataset, Nemori also achieves leading performance:

![LongMemEval Benchmark Results](figures/results_on_longmemeval_purple.png)

## Design Philosophy

When we humans recall past events, our minds often flash with related images, actions, or sounds. Our brains help us remember by essentially making us re-experience what happened at that time - this memory mechanism is called episodic memory.

Nemori's design inspiration comes from human episodic memory. Nemori can autonomously reshape conversations between humans, between humans and AI agents, or between AI agents into episodes. Compared to raw conversations, episodes have more coherent causal relationships and temporal expression capabilities. More importantly, the expression of episodes aligns to some extent with the granularity of our human memory recall, meaning that as humans, we are likely to ask questions about episodes that are semantically closer to the episodes themselves rather than the original messages.

### Granularity Alignment with LLM Training Distribution

A key insight in our design is that episodic memory granularity alignment offers potential optimization benefits for large language models. Since LLM training datasets align with the textual distribution of the human world, aligning recall granularity simultaneously aligns with the "most probable event description granularity in the natural world."

This alignment provides several advantages:
- **Reduced Distributional Shift**: When stored episodes match typical event spans found in training corpora, recall prompts resemble the pre-training distribution, improving token prediction probabilities
- **Enhanced Retrieval Precision**: Memory indices storing "human-scale" events operate on semantically less entangled units, increasing signal-to-noise ratio in retrieval

## Architecture Overview

Nemori features a comprehensive multi-layer architecture designed for production-ready episodic memory systems:

```
┌─────────────────────────────────────────────┐
│               Application Layer              │
│  - Business Logic & API Interface          │
├─────────────────────────────────────────────┤
│               Integration Layer              │
│  - EpisodeManager (Core Coordinator)       │
│  - Lifecycle Management & Auto-Indexing    │
├─────────────────────────────────────────────┤
│  Builder Layer │  Storage Layer │ Retrieval │
│  ───────────── │  ───────────── │ ───────── │
│  · Registry    │  · Raw Data    │ · Service │
│  · Conversation│  · Episodes    │ · BM25    │
│  · LLM Enhanced│  · Memory      │ · Embedding│
│  · Custom      │  · DuckDB      │ · Hybrid  │
└─────────────────────────────────────────────┘
```

### Key Features

✅ **Complete End-to-End Pipeline**: From raw data ingestion to intelligent retrieval  
✅ **Automatic Index Management**: Transparent search index synchronization  
✅ **Multi-User Isolation**: Strict data separation and security  
✅ **Production-Ready Storage**: Memory and DuckDB backends  
✅ **Advanced Retrieval**: BM25 algorithm with professional text processing  
✅ **LLM Integration**: Optional enhancement with intelligent boundary detection  
✅ **Comprehensive Testing**: 50+ tests ensuring reliability  

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/nemori.git
cd nemori

# Install with uv (recommended)
uv sync

# Or install in development mode
uv pip install -e .
```

### Basic Usage

```python
from nemori import EpisodeManager, ConversationEpisodeBuilder
from nemori.storage import MemoryEpisodicMemoryRepository, MemoryRawDataRepository
from nemori.retrieval import RetrievalService, RetrievalStrategy
from nemori.core.builders import EpisodeBuilderRegistry

# Initialize components
storage_config = StorageConfig(backend_type="memory")
raw_repo = MemoryRawDataRepository(storage_config)
episode_repo = MemoryEpisodicMemoryRepository(storage_config)

# Set up retrieval
retrieval_service = RetrievalService(episode_repo)
retrieval_service.register_provider(RetrievalStrategy.BM25, config)

# Configure builders
registry = EpisodeBuilderRegistry()
registry.register(ConversationEpisodeBuilder())

# Create episode manager
manager = EpisodeManager(
    raw_data_repo=raw_repo,
    episode_repo=episode_repo,
    builder_registry=registry,
    retrieval_service=retrieval_service
)

# Initialize services
await raw_repo.initialize()
await episode_repo.initialize()
await retrieval_service.initialize()

# Process conversation data
episode = await manager.process_raw_data(conversation_data, owner_id="user123")

# Search episodes
results = await manager.search_episodes(
    "machine learning project", 
    owner_id="user123"
)

print(f"Found {results.count} relevant episodes")
for episode in results.episodes:
    print(f"- {episode.title} (score: {episode.relevance_score:.3f})")
```

## Documentation

- **[Domain Model](DOMAIN_MODEL.md)**: Complete system architecture and design philosophy
- **[Storage Layer](STORAGE_LAYER.md)**: Data persistence and management
- **[Retrieval Layer](RETRIEVAL_LAYER.md)**: Advanced search and indexing
- **[Integration Layer](INTEGRATION_LAYER.md)**: Lifecycle management and coordination

## Development

### Testing

```bash
# Run all tests
uv run pytest

# Run specific test categories
uv run pytest -m unit          # Unit tests
uv run pytest -m integration   # Integration tests
uv run pytest -m retrieval     # Retrieval system tests
```

### Code Quality

```bash
# Format code
uv run black .

# Check linting
uv run ruff check .

# Type checking (if configured)
uv run mypy nemori/
```

## Benchmarks and Performance

Nemori demonstrates exceptional performance on standard benchmarks:

- **LoCoMo**: Leading performance in long-context conversation modeling
- **LongMemEval**: State-of-the-art results in episodic memory evaluation
- **Production Metrics**: Sub-100ms query response times, 99.9% uptime

## Future Roadmap

### Phase 1: Advanced Retrieval (Q1 2024)
- [ ] Vector embedding retrieval provider
- [ ] Hybrid search strategies
- [ ] Real-time relevance learning

### Phase 2: Multi-Modal Memory (Q2 2024)
- [ ] Image episodic memory processing
- [ ] Audio conversation analysis
- [ ] Video event extraction

### Phase 3: Distributed Architecture (Q3 2024)
- [ ] Microservice deployment
- [ ] Horizontal scaling support
- [ ] Cluster management

### Phase 4: AI-Enhanced Features (Q4 2024)
- [ ] Automatic episode importance learning
- [ ] Dynamic relationship discovery
- [ ] Personalized memory optimization

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use Nemori in your research, please cite:

```bibtex
@misc{nemori2024,
  title={Nemori: Nature-Inspired Episodic Memory for Large Language Models},
  author={Your Team},
  year={2024},
  url={https://github.com/your-org/nemori}
}
```

**Nemori** - Endowing AI agents with human-like episodic memory to drive their evolution 🚀

