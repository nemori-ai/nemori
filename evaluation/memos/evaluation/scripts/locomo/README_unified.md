# Unified Memory LoCoMo Benchmark Evaluation

This directory contains the unified episodic and semantic memory evaluation scripts for the LoCoMo benchmark, combining both memory types in a single integrated system.

## Scripts Overview

### 1. `locomo_unified_memory_eval.py` - Memory Ingestion
**Purpose**: Builds unified episodic and semantic memory from LoCoMo conversation data
**Usage**: 
```bash
python locomo_unified_memory_eval.py --lib nemori --version unified_v1
```

**Features**:
- ✅ **Episodic Memory**: Conversation episodes with boundary detection  
- ✅ **Semantic Memory**: Automatic knowledge discovery and evolution
- ✅ **Embedding Retrieval**: Vector similarity search for both memory types
- ✅ **Enhanced Builder**: Unified conversation processing
- ✅ **Statistics**: Comprehensive memory analysis and reporting

### 2. `locomo_unified_responses.py` - Response Generation  
**Purpose**: Generates responses to LoCoMo questions using unified memory retrieval
**Usage**:
```bash
python locomo_unified_responses.py --lib nemori --version unified_v1
```

**Features**:
- 🔍 **Unified Retrieval**: Searches both episodic episodes and semantic concepts
- 🤖 **LLM Generation**: Contextual response generation using retrieved memories
- 📊 **Performance Metrics**: Search and response timing analysis
- 💾 **JSON Output**: Compatible with LoCoMo evaluation format

### 3. `locomo_eval.py` - Benchmark Evaluation
**Purpose**: Evaluates generated responses using multiple metrics
**Usage**:
```bash
python locomo_eval.py --lib nemori --version unified_v1 --num_runs 3
```

**Features**:
- 🎯 **LLM-as-a-Judge**: Accuracy evaluation using GPT models
- 📈 **NLP Metrics**: ROUGE, BLEU, METEOR, BERT-F1 scoring
- 🔢 **Semantic Similarity**: Embedding-based similarity calculation
- 📋 **Multi-run Analysis**: Statistical significance testing

## Complete Benchmark Workflow

### Step 1: Data Preparation
Ensure LoCoMo data files are available:
```
data/locomo/
├── locomo10.json          # Conversation data  
└── locomo_questions.json  # Evaluation questions (auto-generated if missing)
```

### Step 2: Memory Ingestion
Build unified episodic and semantic memory:
```bash
cd evaluation/memos/evaluation/scripts/locomo
python locomo_unified_memory_eval.py --lib nemori --version unified_v1
```

**Output**: 
- Creates unified memory database with episodic episodes and semantic knowledge
- Generates embedding indices for vector similarity search
- Reports memory statistics and coverage

### Step 3: Response Generation
Generate responses using unified memory retrieval:
```bash  
python locomo_unified_responses.py --lib nemori --version unified_v1
```

**Output**:
- `results/locomo/nemori-unified_v1/nemori_locomo_responses.json`
- Contains responses with episodic and semantic context

### Step 4: Benchmark Evaluation
Evaluate responses against ground truth:
```bash
python locomo_eval.py --lib nemori --version unified_v1 --num_runs 3
```

**Output**:
- `results/locomo/nemori-unified_v1/nemori_locomo_judged.json`  
- Comprehensive evaluation metrics and scores

## Unified Memory Architecture

### Episodic Memory
- **Episodes**: Structured conversation segments with temporal boundaries
- **Boundary Detection**: LLM-powered conversation segmentation
- **Embedding Search**: Vector similarity for episode retrieval
- **Speaker Perspective**: Episodes created for each conversation participant

### Semantic Memory  
- **Knowledge Discovery**: Automatic extraction of concepts, facts, and relationships
- **Knowledge Evolution**: Version tracking and confidence-based updates
- **Semantic Relationships**: Typed connections between knowledge nodes
- **Embedding Integration**: Vector representations for semantic similarity

### Unified Retrieval
- **Enhanced Query**: Simultaneous search across both memory types
- **Context Integration**: Combines episodic episodes and semantic concepts
- **Relevance Ranking**: Embedding-based similarity scoring
- **Comprehensive Context**: Rich memory context for response generation

## Configuration Options

### Memory Configuration
```python
experiment = NemoriExperiment(
    version="unified_v1",
    episode_mode="speaker",           # Episode creation per speaker
    retrievalstrategy=RetrievalStrategy.EMBEDDING  # Vector similarity search  
)
```

### LLM Configuration  
- **Model**: `gpt-4o-mini` for semantic discovery and response generation
- **Embeddings**: `text-embedding-ada-002` for vector representations
- **API**: Configurable endpoints and authentication

### Retrieval Configuration
- **Episodic Limit**: Number of relevant episodes to retrieve
- **Semantic Limit**: Number of semantic concepts to include  
- **Context Size**: Maximum context length for response generation
- **Similarity Threshold**: Minimum similarity for relevance

## Performance Metrics

### Memory Construction
- **Episodes Created**: Count of episodic memory segments
- **Semantic Concepts**: Number of discovered knowledge items
- **Processing Time**: Ingestion and indexing performance
- **Memory Coverage**: Participants and conversation coverage

### Response Quality
- **LLM-as-a-Judge**: Human-like accuracy evaluation (0-1 scale)
- **ROUGE Scores**: N-gram overlap with reference answers
- **Semantic Similarity**: Embedding cosine similarity
- **Response Time**: Search and generation latency

### Statistical Analysis
- **Multi-run Evaluation**: Multiple judgment runs for statistical significance
- **Standard Deviation**: Consistency measurement across runs
- **Confidence Intervals**: Statistical reliability assessment

## Comparison with Baseline Systems

### Advantages over Pure Episodic Systems
- **Concept Abstraction**: Semantic memory captures generalizable knowledge
- **Knowledge Evolution**: Facts and preferences update over time
- **Cross-Episode Connections**: Semantic relationships span conversations
- **Improved Relevance**: Concept-based retrieval beyond keyword matching

### Advantages over Pure Semantic Systems  
- **Temporal Context**: Episodic memory preserves when events occurred
- **Narrative Structure**: Maintains conversation flow and context
- **Personal Details**: Specific experiences and interactions preserved
- **Rich Context**: Complete conversational episodes for context

### Integration Benefits
- **Comprehensive Retrieval**: Both specific episodes and general knowledge
- **Contextual Responses**: Combines personal history with learned concepts
- **Adaptive Memory**: Episodes inform semantic discovery and vice versa
- **Improved Coverage**: Handles both specific and general queries effectively

## File Structure
```
locomo/
├── locomo_unified_memory_eval.py    # Memory ingestion script
├── locomo_unified_responses.py      # Response generation script  
├── locomo_eval.py                   # Benchmark evaluation script
├── locomo_ingestion_emb.py         # Original episodic-only ingestion
├── README_unified.md               # This documentation
└── results/
    └── locomo/
        └── nemori-{version}/
            ├── nemori_locomo_responses.json  # Generated responses
            ├── nemori_locomo_judged.json     # Evaluation results  
            └── storages/                     # Memory database files
```

## Research Applications

This unified memory system enables research into:
- **Memory Integration**: How episodic and semantic memory complement each other
- **Knowledge Discovery**: Automatic extraction from conversational data  
- **Memory Evolution**: How knowledge changes and updates over time
- **Retrieval Strategies**: Optimal combination of memory types for different queries
- **Benchmark Performance**: Comparative evaluation against existing memory systems

## Troubleshooting

### Common Issues
1. **Missing Data Files**: Ensure `locomo10.json` exists in `data/locomo/`
2. **API Key Errors**: Check OpenAI API key configuration
3. **Memory Issues**: Large datasets may require increased system resources
4. **Embedding Failures**: Verify embedding model availability and API limits

### Performance Optimization  
- **Batch Processing**: Process conversations in batches for large datasets
- **Index Caching**: Reuse embedding indices across runs when possible
- **Concurrent Processing**: Utilize asyncio for improved performance
- **Resource Monitoring**: Monitor memory and API usage during evaluation

## Future Enhancements

### Planned Features
- **Multi-modal Support**: Integration with image and audio memories
- **Advanced Relationships**: More sophisticated semantic relationship types
- **Temporal Reasoning**: Time-aware retrieval and knowledge updates
- **Interactive Evaluation**: Real-time benchmark testing interface
- **Comparative Analysis**: Automated comparison with other memory frameworks