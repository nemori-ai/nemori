# Nemori Memory System
#
# Nemori is a self-organising long-term memory substrate for agentic LLM workflows. It ingests raw multi-turn conversations, segments them into topic-consistent episodes, distils durable semantic knowledge, and exposes a unified search surface for downstream reasoning. The design is inspired by cognitive-science theories (Event Segmentation Theory, Predictive Processing) and implemented with production-ready concurrency, caching, and pluggable storage.
#
# - **Language:** Python 3.9+
# - **License:** MIT
# - **Core Dependencies:** OpenAI API, ChromaDB, uv (package manager)
#
# ---
#
# ## 1. Why Nemori
#
# Large language models rapidly forget long-horizon context. Nemori introduces two coupled control loops:
#
# 1. **Two-Step Alignment**
#    - *Boundary Alignment* – smart segmentation of buffered dialogue using LLM-powered boundary detection with transitional masking heuristics.
#    - *Representation Alignment* – conversion of each segment into rich narrative episodes with precise temporal anchors and provenance.
# 2. **Predict–Calibrate Learning**
#    - *Predict* – hypothesise new episodes from existing semantic knowledge to surface gaps early.
#    - *Calibrate* – extract high-value facts from discrepancies and fold them into the semantic knowledge base.
#
# The result is a data structure that stays compact, queryable, and behaviourally faithful to the source conversation.
#
# ---
#
# ## 2. Quick Start
#
# ### 2.1. Environment
#
# We recommend [uv](https://github.com/astral-sh/uv) for deterministic Python environments.
#
# ```bash
# brew install uv
#
# git clone https://github.com/anonymous/nemori-code.git
# cd nemori-code
#
# uv venv
# source .venv/bin/activate  # Windows: .venv\Scripts\activate
#
# uv pip install -e .
# uv pip install -e ".[dev,evaluation]"
# ```
#
# ### 2.2. Credentials
#
# Create a `.env` file in the repo root:
#
# ```ini
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=...
# ```
#
# Nemori only reads variables; it never writes secrets to disk.
#
# ### 2.3. Minimal Usage
#
# ```python
# from nemori import NemoriMemory, MemoryConfig
#
# config = MemoryConfig(
#     llm_model="gpt-4o-mini",
#     enable_semantic_memory=True,
#     enable_prediction_correction=True,
# )
#
# memory = NemoriMemory(config=config)
#
# messages = [
#     {"role": "user", "content": "I started training for a marathon in Seattle."},
#     {"role": "assistant", "content": "Great! When is the race?"},
#     {"role": "user", "content": "It is in October."},
# ]
#
# memory.add_messages("user123", messages)
#
# answer = memory.search(
#     user_id="user123",
#     query="race plans",
#     top_k_episodes=5,
#     top_k_semantic=5,
# )
# ```
#
# Clean up resources via `memory.close()` when finished.
#
# ---
#
# ## 3. System Architecture
#
# ### 3.1. Data Flow
#
# ```
# ┌──────────────┐   ┌──────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
# │ Message Sink │▶──│ Concurrent Buffer │▶──│ Boundary Detector    │▶──│ Episode Generator    │
# └──────────────┘   │ (per user)        │   │  • LLM + heuristics │   │  • Narrative JSON   │
#                     │  • size guards   │   │  • caching          │   │  • provenance kept  │
#                     └──────────────────┘   └─────────────────────┘   │                     │
#                                                                       ▼
#                                                            ┌─────────────────────┐
#                                                            │ Semantic Generator  │
#                                                            │  • Predict/Correct  │
#                                                            │  • Fact distillation│
#                                                            └─────────────────────┘
#                                                                       ▼
#                                                            ┌─────────────────────┐
#                                                            │ Search Engine       │
#                                                            │  • hybrid BM25/IVF │
#                                                            │  • original text    │
#                                                            └─────────────────────┘
# ```
#
# ### 3.2. Key Services
#
# | Component | Location | Highlights |
# |-----------|----------|------------|
# | `MemorySystem` orchestrator | `src/core/memory_system.py` | Thread-safe per-user processing, metrics, cache-aware search |
# | Boundary Detection | `src/core/boundary_detector.py` | LLM prompt pipeline with optional masking (`boundary_exclude_threshold`) handled in the buffer manager |
# | Episodic Storage | `src/storage/episode_storage.py` | JSONL persistence, lock-scoped writes, lazy caches |
# | Semantic Storage | `src/storage/semantic_storage.py` | Similar guarantees for knowledge statements |
# | Predict-Calibrate Engine | `src/generation/prediction_correction_engine.py` | Generates predictions, compares with truth, extracts corrections |
# | Unified Search | `src/search/unified_search.py` | Fuses BM25 + Chroma vector search, exposes episode & semantic channels |
# | Performance Layer | `src/utils/performance.py` | Sharded LRU cache, thread pools, trace statistics |
# | Public Facade | `src/api/facade.py` | `NemoriMemory` convenience wrapper for simple integrations |
#
# ### 3.3. Configuration Snapshot
#
# `MemoryConfig` (see `src/config.py`) centralises run-time behaviour:
#
# - Buffer limits: `buffer_size_min`, `buffer_size_max`
# - Boundary options: `enable_smart_boundary`, `boundary_exclude_last_message`, `boundary_exclude_threshold`
# - Episode length gates: `episode_min_messages`, `episode_max_messages`
# - Semantic duplication rules: `semantic_similarity_threshold`
# - Worker pools: `max_workers`, `semantic_generation_workers`
# - Index backends: filesystem vs in-memory, Chroma vs BM25
#
# Tune these parameters in JSON/YAML by calling `MemoryConfig.from_dict(...)`.
#
# ---
#
# ## 4. Repository Layout
#
# ```
# src/
# │
# ├── api/                 # Facade entry points
# ├── core/                # Orchestration, buffers, metrics
# ├── generation/          # Episode + semantic generation, prediction loop
# ├── models/              # Dataclasses for messages & memories
# ├── search/              # BM25, vector, hybrid, original text search
# ├── storage/             # Jsonl storage backends
# ├── utils/               # LLM client, embeddings, caching utilities
# └── ...
#
# evaluation/
# ├── locomo/              # LoCoMo benchmark scripts
# ├── longmemeval/         # Additional long-context evaluation
# └── readme.md            # Dataset-specific instructions
#
# memories/               # Default persistence root (episodes/, semantic/)
# web-react/              # Lightweight results viewer
# ```
#
# ---
#
# ## 5. Running Evaluations
#
# ### 5.1. LoCoMo Pipeline
#
# ```bash
# python evaluation/locomo/add.py --config evaluation/locomo/config.json --data dataset/locomo10.json
# python evaluation/locomo/search.py --config evaluation/locomo/config.json --include-original-messages-top-k 2
# python evaluation/locomo/evals.py --input_file locomo/results.json --output_file locomo/metrics.json
# python evaluation/locomo/generate_scores.py
# ```
#
# ### 5.2. Latest LoCoMo Scores
#
# | Category | BLEU | F1 | LLM | Count |
# |----------|------|----|-----|-------|
# | 1 | 0.3426 | 0.4312 | 0.7730 | 282 |
# | 2 | 0.5050 | 0.5874 | 0.7632 | 321 |
# | 3 | 0.2294 | 0.2878 | 0.5521 | 96 |
# | 4 | 0.4878 | 0.5497 | 0.8716 | 841 |
#
# Overall means — BLEU: `0.4487`, F1: `0.5196`, LLM alignment: `0.8110`.
#
# ### 5.3. LongMemEval
#
# See `evaluation/longmemeval/readme.md` for 100k-token context experiments.
#
# ---
#
# ## 6. Developing with Nemori
#
# - Testing: `pytest`
# - Linting: `ruff check src`
# - Type checking: `mypy src`
# - Bench utilities: `scripts/` contains profiling helpers
#
# Use the `NemoriMemory` facade for quick experiments; inject custom storage or LLM clients when integrating into larger systems.
#
# ---
#
# ## 7. Troubleshooting
#
# | Symptom | Likely Cause | Mitigation |
# |---------|--------------|------------|
# | No episodes emitted | Boundary threshold masking too aggressively | Increase `boundary_exclude_threshold` or disable last-message exclusion |
# | Missing timestamps in LoCoMo prompts | Run `evaluation/locomo/search.py` after the hydration patch (timestamps now lifted from metadata) |
# | Empty search responses | User indices not loaded | Trigger ingestion, or call `NemoriMemory._memory_system.load_user_data_and_indices(user)` |
# | High latency | Chroma cold start | Preload collections via `load_user_data_and_indices_for_method` |
#
# ---
#
# ## 8. Contributing
#
# 1. Fork and create a feature branch.
# 2. Add or update tests.
# 3. Submit a PR outlining architectural impact (boundary logic, storage schema, etc.).
#
# Nemori is actively evolving toward multi-agent deployments. Feedback and contributions are welcome.
