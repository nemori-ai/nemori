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
wget https://github.com/snap-research/locomo/raw/main/data/locomo10.json -O dataset/locomo10.json
```

#### LongMemEval Dataset
Download from [Hugging Face](https://huggingface.co/datasets/xiaowu0162/longmemeval/tree/main):

```bash
wget https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s.json -O dataset/longmemeval_s.json
```

## 🧪 Evaluation Workflows

### 🎪 LOCOMO Evaluation

LOCOMO evaluates long-context memory capabilities with conversational scenarios.

#### Step 1: Add Memories
```bash
PYTHONPATH=. python evaluation/locomo/add.py
```
- Processes LOCOMO conversations
- Creates episodic and semantic memories
- Stores memories in PostgreSQL and Qdrant

#### Step 2: Search & Retrieve
```bash
PYTHONPATH=. python evaluation/locomo/search.py
```
- Executes memory searches for test queries
- Retrieves relevant episodic and semantic memories via hybrid search
- Generates candidate responses

#### Step 3: Evaluate Results
```bash
PYTHONPATH=. python evaluation/locomo/evals.py
```
- Compares model responses with ground truth
- Calculates BLEU, F1, and LLM judge scores
- Generates detailed evaluation metrics

#### Step 4: Generate Final Scores
```bash
PYTHONPATH=. python evaluation/locomo/generate_scores.py
```
- Aggregates evaluation results
- Produces final performance scores
- Creates summary reports

### 🧠 LongMemEval Evaluation

LongMemEval focuses on extended memory evaluation across diverse question types.

#### Step 1: Add Memories
```bash
PYTHONPATH=. python evaluation/longmemeval/add.py
```
- Processes LongMemEval conversations
- Creates comprehensive memory representations

#### Step 2: Search & Retrieve
```bash
PYTHONPATH=. python evaluation/longmemeval/search.py
```
- Performs memory retrieval for evaluation questions
- Supports multiple search strategies (vector, text, hybrid)
- Generates contextual responses

#### Step 3: Evaluate Results
```bash
PYTHONPATH=. python evaluation/longmemeval/evals.py longmemeval/results.json
```
- Evaluates responses using LongMemEval criteria
- Supports temporal reasoning and factual questions
- Provides comprehensive accuracy metrics

## ⚙️ Configuration Options

### Memory System Configuration
- **Model**: Choose LLM model (default: `openai/gpt-4.1-mini` via OpenRouter)
- **PostgreSQL**: Metadata, text search (tsvector/GIN), and message buffering
- **Qdrant**: Vector storage and similarity search
- **Batch Size**: Control processing batch size
- **Search Method**: Vector (Qdrant), text (PostgreSQL tsvector), or hybrid search
- **Semantic Memory**: Enable/disable semantic memory generation

### Storage Architecture
- **PostgreSQL** – stores episode metadata, semantic records, and provides full-text search via tsvector/GIN indexes
- **Qdrant** – stores all embedding vectors with automatic dimension adaptation

### Evaluation Parameters
- **Top-K Episodes**: Number of episodic memories to retrieve
- **Top-K Semantic**: Number of semantic memories to retrieve
- **Evaluation Model**: Model used for response evaluation
- **Concurrency**: Maximum concurrent evaluation tasks
