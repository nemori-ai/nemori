# Evaluation Memory Framework

This repository provides tools and scripts for evaluating the LoCoMo dataset using various models and APIs.

## Installation

1. Set the `PYTHONPATH` environment variable:
   ```bash
   export PYTHONPATH=../src
   cd evaluation
   ```

2. Install the required dependencies:
   ```bash
   poetry install --with eval
   ```

## Configuration

1. Copy the `.env-example` file to `.env`, and fill in the required environment variables according to your environment and API keys.

2. Copy the `configs-example/` directory to a new directory named `configs/`, and modify the configuration files inside it as needed. This directory contains model and API-specific settings.


## Evaluation Scripts

### LoCoMo Evaluation
⚙️ To evaluate the **LoCoMo** dataset using one of the supported memory frameworks — `memos`, `mem0`, or `zep` — run the following [script](./scripts/run_locomo_eval.sh):

```bash
# Edit the configuration in ./scripts/run_locomo_eval.sh
# Specify the model and memory backend you want to use (e.g., mem0, zep, etc.)
./scripts/run_locomo_eval.sh
```

✍️ For evaluating OpenAI's native memory feature with the LoCoMo dataset, please refer to the detailed guide: [OpenAI Memory on LoCoMo - Evaluation Guide](./scripts/locomo/openai_memory_locomo_eval_guide.md).
