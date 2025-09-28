# 📊 Nemori Memory System Evaluation

This directory contains evaluation scripts and datasets for testing the Nemori memory system performance against established benchmarks.

## 🎯 Overview

We evaluate our memory system using two comprehensive datasets:
- **LOCOMO**: Long-context memory benchmarks
- **LongMemEval**: Extended memory evaluation suite

## 🗂️ Directory Structure

```
evaluation/
├── dataset/               # Evaluation datasets
│   ├── locomo10.json
│   └── longmemeval_s.json
├── locomo/               # LOCOMO evaluation scripts
│   ├── add.py           # Add memories to system
│   ├── search.py        # Search and retrieve memories
│   ├── evals.py         # Evaluate results
│   ├── generate_scores.py  # Generate final scores
│   └── metrics/         # Evaluation metrics
│       ├── llm_judge.py
│       └── utils.py
├── longmemeval/         # LongMemEval evaluation scripts
│   ├── add.py          # Add memories to system
│   ├── search.py       # Search and retrieve memories
│   └── evals.py        # Evaluate results
└── readme.md           # This file
```

## 📦 Dataset Setup

### 1. Create Dataset Directory

```bash
cd evaluation
mkdir -p dataset
```

### 2. Download Datasets

#### LOCOMO Dataset
Download from the [LOCOMO GitHub repository](https://github.com/snap-research/locomo/tree/main/data):

```bash
# Download locomo10.json and locomo10_rag.json to dataset/
wget https://github.com/snap-research/locomo/raw/main/data/locomo10.json -O dataset/locomo10.json
```

#### LongMemEval Dataset
Download from [Hugging Face](https://huggingface.co/datasets/xiaowu0162/longmemeval/tree/main):

```bash
# Download longmemeval_s.json to dataset/
wget https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s.json -O dataset/longmemeval_s.json
```

## 🧪 Evaluation Workflows

### 🎪 LOCOMO Evaluation

LOCOMO evaluates long-context memory capabilities with conversational scenarios.

#### Step 1: Add Memories
```bash
python locomo/add.py
```
- Processes LOCOMO conversations
- Creates episodic and semantic memories
- Stores memories in the system

#### Step 2: Search & Retrieve
```bash
python locomo/search.py
```
- Executes memory searches for test queries
- Retrieves relevant episodic and semantic memories
- Generates candidate responses

#### Step 3: Evaluate Results
```bash
python locomo/evals.py
```
- Compares model responses with ground truth
- Calculates BLEU, F1, and LLM judge scores
- Generates detailed evaluation metrics

#### Step 4: Generate Final Scores
```bash
python locomo/generate_scores.py
```
- Aggregates evaluation results
- Produces final performance scores
- Creates summary reports

### 🧠 LongMemEval Evaluation

LongMemEval focuses on extended memory evaluation across diverse question types.

#### Step 1: Add Memories
```bash
python longmemeval/add.py
```
- Processes LongMemEval conversations
- Creates comprehensive memory representations
- Supports semantic memory generation

#### Step 2: Search & Retrieve
```bash
python longmemeval/search.py
```
- Performs memory retrieval for evaluation questions
- Supports multiple search strategies (vector, BM25, hybrid)
- Generates contextual responses

#### Step 3: Evaluate Results
```bash
python longmemeval/evals.py longmemeval/results.json
```
- Evaluates responses using LongMemEval criteria
- Supports temporal reasoning and factual questions
- Provides comprehensive accuracy metrics

## ⚙️ Configuration Options

### Memory System Configuration
- **Model**: Choose LLM model (default: `gpt-4o-mini`)
- **Vector Database**: ChromaDB for vector storage and search
- **Storage Path**: File system storage for original data (JSONL format)
- **Batch Size**: Control processing batch size
- **Workers**: Configure parallel processing threads
- **Search Method**: Vector (ChromaDB), BM25, or hybrid search
- **Semantic Memory**: Enable/disable semantic memory generation

### Vector Database Configuration
- **Vector DB Type**: `chroma` (ChromaDB)
- **Persist Directory**: ChromaDB database storage path
- **Collection Prefix**: Prefix for ChromaDB collections (default: `nemori_eval` for LOCOMO, `nemori_longmem` for LongMemEval)

### Evaluation Parameters
- **Top-K Episodes**: Number of episodic memories to retrieve
- **Top-K Semantic**: Number of semantic memories to retrieve
- **Evaluation Model**: Model used for response evaluation
- **Concurrency**: Maximum concurrent evaluation tasks

## 🔄 ChromaDB Migration

The evaluation system has been updated to use **ChromaDB** instead of FAISS for vector storage and search. This provides:

### Key Benefits
- **Unified Storage**: Single ChromaDB database manages all vector data
- **Auto-Persistence**: Built-in data persistence without manual file management
- **Rich Metadata**: Enhanced metadata querying and filtering capabilities
- **User Isolation**: Independent collections for each user/evaluation run

### Data Storage Architecture
```
evaluation_memories_v3/          # Base storage directory
├── episodes/                    # Episode data (JSONL files)
├── semantic/                    # Semantic memory data (JSONL files)
└── chroma_db/                   # ChromaDB vector database
    ├── nemori_eval_user1_episodes    # User episode vectors
    ├── nemori_eval_user1_semantic    # User semantic vectors
    └── ...
```

### Configuration Changes
The evaluation scripts now use ChromaDB-specific configuration:

```python
config = MemoryConfig(
    # Vector Database Configuration
    vector_db_type="chroma",
    chroma_persist_directory="./storage_path/chroma_db",
    chroma_collection_prefix="nemori_eval",  # or "nemori_longmem"
    
    # Other settings remain the same
    llm_model="gpt-4o-mini",
    embedding_model="text-embedding-3-small"
)
```

### Compatibility Testing
Run the compatibility test to verify your setup:

```bash
python test_chroma_compatibility.py
```

This will validate:
- Basic module imports
- ChromaDB configuration
- Memory system initialization
- Evaluation script dependencies
- ChromaDB functionality

