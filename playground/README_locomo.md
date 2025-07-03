# LoComo Experiment Scripts

This directory contains two scripts for working with episodic memories using the Nemori system and the LoComo dataset:

- **`locomo.py`**: Complete end-to-end workflow for building and testing episodic memories
- **`locomo_interactive.py`**: Interactive query tool for previously saved episodic memories

## 🎯 What These Scripts Do

### locomo.py - Complete Experiment Workflow

1. **Environment Setup**: Automatically loads environment variables from project root `.env` file
2. **Data Loading**: Loads and preprocesses LoComo conversation data
3. **Episode Building**: Uses EpisodeManager + ConversationEpisodeBuilder to create episodes
4. **Storage**: Stores episodes and raw data in DuckDB storage with persistent BM25 index
5. **Retrieval Setup**: Configures BM25 retrieval provider with persistence
6. **Retrieval Testing**: Performs comprehensive retrieval experiments
7. **Analysis**: Analyzes results and creates visualizations
8. **Interactive Mode**: Offers interactive query mode after experiment completion

### locomo_interactive.py - Interactive Query Tool

1. **Database Connection**: Connects to previously created DuckDB database
2. **Index Loading**: Loads persistent BM25 index from disk
3. **Interactive Querying**: Allows real-time querying of saved episodic memories
4. **Formatted Results**: Displays search results with relevance scores and filtering

## 🚀 Usage

### Prerequisites

1. **OpenAI API Key** (optional but recommended for locomo.py):
   
   **Option A: Using .env file (Recommended)**:
   ```bash
   # Copy the example file and edit it
   cp ../.env.example ../.env
   # Then edit .env and set your OPENAI_API_KEY
   ```
   
   **Option B: Using environment variable**:
   ```bash
   export OPENAI_API_KEY=your_key_here
   ```

2. **Dataset**: Ensure `dataset/locomo10.json` exists in the playground directory

3. **Dependencies**: Make sure all nemori dependencies are installed:
   ```bash
   uv sync
   ```

### Running the Main Experiment

```bash
cd playground
python locomo.py
```

After the experiment completes, you'll be prompted to enter interactive query mode:
```
🎮 Enter interactive query mode? (y/n): y
```

### Running the Interactive Query Tool Only

If you've already run the main experiment and want to query your saved memories:

```bash
cd playground
python locomo_interactive.py
```

This script will:
- Load the existing database from `.tmp/nemori_memory.duckdb`
- Load the persistent BM25 index from `.tmp/bm25_index_agent.pkl`
- Start an interactive query session

### Interactive Query Examples

In interactive mode, you can query with natural language:

```
🔎 Query: basketball
🔎 Query: music and concerts
🔎 Query: What did they discuss about health?
🔎 Query: travel and adventure
🔎 Query: quit  # To exit
```

## 🔍 Experiment Flow

### 1. LLM Provider Setup (locomo.py)
- Attempts to configure OpenAI provider if API key is available
- Falls back to basic episode building if no LLM is available

### 2. Data Preprocessing (locomo.py)
- Loads LoComo conversations from JSON
- Converts LoComo timestamp format to ISO format
- Structures conversation data for Nemori format

### 3. Episode Building (locomo.py)
- Uses EpisodeManager for complete lifecycle management
- Processes raw data → episodes → storage → indexing
- Each conversation becomes one episode
- Creates persistent BM25 index

### 4. Retrieval Testing (locomo.py)
- Tests multiple query scenarios (music, sports, travel, etc.)
- Tests retrieval with relevance threshold filtering
- Measures retrieval performance and relevance

### 5. Interactive Query Mode (both scripts)
- Real-time querying of saved episodic memories
- Relevance-based result filtering (score > 1.0)
- Formatted result display with episode details
- Option to view additional results

### 6. Analysis & Visualization (locomo.py)
- Episode level distribution (ATOMIC/COMPOUND/THEMATIC)
- Content length analysis
- Owner distribution
- Timeline visualization
- Storage and retrieval statistics

## 📊 Key Features

### Persistent Storage & Indexing
- **Database**: `.tmp/nemori_memory.duckdb` - Stores episodes and raw data
- **BM25 Index**: `.tmp/bm25_index_agent.pkl` - Persistent search index
- **Reusable**: Query tool can work with existing data without rebuilding

### Interactive Query Features
- **Real-time Search**: Query saved memories instantly
- **Relevance Filtering**: Only shows results with BM25 score > 1.0
- **Formatted Display**: Pretty-printed results with episode details
- **Progressive Results**: Option to view additional results
- **Bilingual Interface**: English and Chinese prompts

### Search Result Information
Each search result displays:
- **BM25 Score**: Relevance score for the query
- **Episode Title**: AI-generated or extracted title
- **Owner**: Episode owner ID
- **Level**: Episode complexity level (ATOMIC/COMPOUND/THEMATIC)
- **Date & Duration**: Temporal information
- **Importance Score**: Episode importance rating
- **Keywords**: Search keywords for the episode
- **Content Preview**: Summary or content snippet

## 🎨 Visualization Components (locomo.py)

The generated visualization includes:
1. **Pie Chart**: Episode level distribution
2. **Histogram**: Content length distribution  
3. **Bar Chart**: Episodes per owner
4. **Scatter Plot**: Episode timeline with durations

## ⚙️ Configuration Options

### Main Experiment (locomo.py)
```python
await experiment.run_complete_experiment(
    data_file="locomo10.json",  # Dataset filename
    sample_size=10              # Number of conversations to process
)
```

### Interactive Query Tool (locomo_interactive.py)
```python
# Default configuration
db_dir = Path(".tmp")  # Database directory
relevance_threshold = 1.0  # Minimum BM25 score for results
```

## 🔧 Troubleshooting

### Common Issues:

1. **Missing Dataset** (locomo.py):
   - Ensure `dataset/locomo10.json` exists
   - Check file permissions

2. **Missing Database** (locomo_interactive.py):
   - Run `locomo.py` first to create the database
   - Ensure `.tmp/nemori_memory.duckdb` exists

3. **Missing BM25 Index** (locomo_interactive.py):
   - Run `locomo.py` first to create the index
   - Ensure `.tmp/bm25_index_agent.pkl` exists

4. **OpenAI API Issues** (locomo.py):
   - Verify API key is set correctly
   - Check internet connection
   - Script will continue in fallback mode if OpenAI fails

5. **Database Issues**:
   - `locomo.py` automatically cleans and recreates database
   - Check `.tmp/` directory permissions

6. **Import Errors**:
   - Ensure you're in the playground directory
   - Verify nemori package is installed: `uv sync`

## 📝 Understanding the Output

### Console Output Structure (locomo.py):
- 🚀 Initialization messages
- 🤖 LLM provider setup
- 📚 Data loading progress
- 🗄️ Storage setup confirmation
- 🏗️ Episode building progress (shows each conversation processed)
- 🔍 Retrieval experiment results
- 📊 Final statistics and analysis
- 🎮 Interactive query mode option

### Interactive Query Output (both scripts):
- 🔧 Component initialization
- 📊 Index statistics (episodes, documents, size)
- 🔍 Search results with relevance filtering
- 📋 Formatted episode details
- 📈 Query performance metrics

### Episode Building Process (locomo.py):
For each conversation, you'll see:
- Owner ID extracted from conversation
- Message count and duration
- Episode creation success/failure
- Episode title and level assignment

### Retrieval Results Display:
- **Relevance Threshold**: Only shows results with BM25 score > 1.0
- **Query Time**: Search performance in milliseconds
- **Total Candidates**: Number of episodes searched
- **Result Details**: Formatted episode information
- **Progressive Display**: Option to view more results

## 🎯 Workflow Recommendations

1. **First Time**: Run `locomo.py` to build the complete dataset
2. **Subsequent Queries**: Use `locomo_interactive.py` for fast querying
3. **Experiment Changes**: Re-run `locomo.py` to rebuild with new data
4. **Interactive Mode**: Use either script's interactive mode for exploration

This setup provides a comprehensive demonstration of the Nemori system's capabilities, from data ingestion to retrieval, using real conversation data from the LoComo dataset, with both batch processing and interactive query capabilities.