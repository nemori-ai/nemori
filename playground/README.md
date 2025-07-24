# Nemori Playground - Quick Start Guide

This directory contains two clear demonstrations of the Nemori episodic memory system, designed to show different aspects of the system's capabilities.

## 🚀 Quick Start - Choose Your Demo

### Option 1: Basic Demo (Recommended for First Time)
**No external dependencies required** - Perfect for understanding core concepts.

```bash
# Run the basic demo
uv run python playground/basic_demo.py
```

**What it offers:**
- ✅ **No OpenAI API key required** - Uses mock data and simulated episodes
- ✅ **Human-readable JSONL storage** - Easy to inspect generated files
- ✅ **Rich conversation scenarios** - AI projects, travel planning, tech architecture
- ✅ **Complete workflow demonstration** - Storage, retrieval, and search
- ✅ **Perfect for learning** - Understand system without external dependencies

**What it demonstrates:**
- Raw conversation data storage
- Mock episode creation (simulating LLM results)
- Episode retrieval by owner
- Keyword-based search functionality
- Storage statistics and reporting

**Expected output:**
```
🚀 Nemori Basic Demo - Simulated Episode Creation
📋 This demo uses mock data and simulated episodes (no OpenAI required)
💾 All data will be saved as human-readable JSONL files
✅ Created 3 conversation scenarios
✅ Created 6 mock episodes
🔍 Demonstrating search and retrieval capabilities...
🎉 Basic demo completed successfully!
```

**Generated files:** `playground/basic_demo_results/`
- `raw_data.jsonl` - Original conversations
- `episodes.jsonl` - Generated episodes
- `episode_links.jsonl` - Episode-data relationships

### Option 2: Intelligent Demo (Real LLM Processing)
**Requires OpenAI API key** - Shows authentic AI-powered episodic memory.

```bash
# Set your OpenAI API key (required!)
export OPENAI_API_KEY="your-api-key-here"

# Run the intelligent demo
uv run python playground/intelligent_demo.py
```

**What it offers:**
- 🧠 **Real LLM processing** - Uses OpenAI GPT-4o-mini for authentic intelligence
- 🎯 **Conversation boundary detection** - AI identifies natural episode boundaries
- 🔍 **Intelligent search** - Semantic understanding of episode content
- 💾 **Storage backend options** - Choose JSONL, DuckDB, or PostgreSQL
- 📊 **Search demonstrations** - Shows intelligent retrieval capabilities

**What it demonstrates:**
- Real conversation analysis using LLM
- Intelligent episode boundary detection
- Natural language processing for memory creation
- BM25 search with AI-generated content
- Comparison between mock and LLM-generated episodes

**Storage backend selection:**
```bash
# Use JSONL (default)
uv run python playground/intelligent_demo.py

# Use DuckDB
NEMORI_STORAGE=duckdb uv run python playground/intelligent_demo.py

# Use PostgreSQL (requires setup)
export POSTGRESQL_TEST_URL="postgresql+asyncpg://postgres:postgres@localhost/nemori_demo"
NEMORI_STORAGE=postgresql uv run python playground/intelligent_demo.py
```

**Expected output:**
```
🚀 Nemori Intelligent Demo - Real LLM Processing
🧠 This demo uses OpenAI's LLM for authentic intelligence
🤖 Setting up LLM provider (OpenAI)...
✅ OpenAI connection successful!
🧠 LLM analyzing conversation for sarah...
✅ Generated episode: AI Project Milestone Review and Architecture Planning...
🔍 Demonstrating intelligent search capabilities...
🎉 Intelligent demo completed successfully!
```

## 🗄️ Storage Options

The intelligent demo supports multiple storage backends:

### JSONL (Default - Recommended for Development)
- **Human-readable JSON Lines format**
- Files created as `raw_data.jsonl`, `episodes.jsonl`, `episode_links.jsonl`
- Perfect for debugging and data inspection
- No database setup required
- Version control friendly

### DuckDB
- **Zero configuration required**
- High-performance analytical queries
- Single database file
- Great for development and testing

### PostgreSQL (Production-like)
For production-like testing:

```bash
# 1. Start PostgreSQL (using Docker)
docker run -d --name nemori-postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=nemori_demo \
  -p 5432:5432 postgres:15

# 2. Set environment variable
export POSTGRESQL_TEST_URL="postgresql+asyncpg://postgres:postgres@localhost/nemori_demo"

# 3. Run intelligent demo with PostgreSQL
NEMORI_STORAGE=postgresql uv run python playground/intelligent_demo.py
```

## 📊 Demo Comparison

| Feature | Basic Demo | Intelligent Demo |
|---------|------------|------------------|
| **OpenAI API Key** | ❌ Not required | ✅ Required |
| **Episode Generation** | Mock/Simulated | Real LLM Processing |
| **Storage Options** | JSONL only | JSONL/DuckDB/PostgreSQL |
| **Boundary Detection** | Pre-defined | AI-powered |
| **Search Quality** | Keyword-based | Semantic understanding |
| **Runtime** | ~5 seconds | ~60-120 seconds |
| **Best For** | Learning concepts | Seeing real capabilities |

## 🛠️ Troubleshooting

### Common Issues

**1. OpenAI API key missing (Intelligent Demo):**
```bash
# The intelligent demo will show helpful instructions:
❌ ERROR: OPENAI_API_KEY environment variable is required!
💡 For a demo without API requirements, try:
   python playground/basic_demo.py
```

**2. Import errors:**
```bash
# Make sure you're in the project root directory
cd /path/to/nemori

# Reinstall dependencies
uv sync
```

**3. PostgreSQL connection failed:**
```bash
# Check if PostgreSQL is running
pg_isready -h localhost -p 5432

# The intelligent demo will automatically fall back to JSONL
⚠️ Storage setup failed: connection error
🔄 Falling back to JSONL storage...
```

**4. Permission errors:**
```bash
# Make sure the playground directory is writable
chmod +w playground/
```

### Inspecting Generated Data

**JSONL files are human-readable:**
```bash
# View raw conversation data
cat playground/*/raw_data.jsonl | jq .

# View created episodes
cat playground/*/episodes.jsonl | jq .

# Check episode relationships
cat playground/*/episode_links.jsonl | jq .
```

## 🎯 Next Steps

After running the demos successfully:

1. **Start with Basic Demo** - Understand core concepts without dependencies
2. **Try Intelligent Demo** - Experience real AI-powered memory creation
3. **Experiment with Storage** - Test different backends for your use case
4. **Modify Demo Data** - Edit conversations to see how the system responds
5. **Integrate** - Use learnings to integrate Nemori in your project

## 📚 Demo Data

Both demos use rich, realistic conversation data including:
- **AI Project Discussions** - Technical planning and development coordination
- **Travel Planning** - Detailed logistics, bookings, and cultural experiences
- **Technology Architecture** - Database selection and system design decisions

The conversations feature natural topic transitions and realistic dialogue patterns, perfect for demonstrating episodic memory capabilities.

---

**🎉 Ready to experience Nemori's memory capabilities?**

- **New to Nemori?** → Start with `python playground/basic_demo.py`
- **Want to see real AI?** → Try `python playground/intelligent_demo.py` (OpenAI key required)