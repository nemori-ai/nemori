# Local evaluation on Windows

Run commands from the repository root unless a section says otherwise.

## Installed layout

- Python environment: `.venv` (Python 3.13)
- LoCoMo: `evaluation/dataset/locomo10.json`
- LongMemEval-S: `evaluation/dataset/longmemeval_s.json`
- SWE-bench Lite: `evaluation/dataset/swebench_lite_test.json` (300 instances)
- PostgreSQL: `localhost:5432`
- Qdrant HTTP/gRPC: `localhost:6333` / `localhost:6334`

The upstream `.python-version` requests Python 3.9 while `pyproject.toml` requires
Python 3.10 or newer. Use the interpreter path explicitly:

```powershell
uv venv --python 3.13 .venv
uv sync -p .\.venv\Scripts\python.exe --extra eval --extra dashboard
```

## Configure model access

```powershell
Copy-Item .env.example .env
notepad .env
```

For OpenRouter, set all four variables below. The same key can usually be used
for both APIs.

```dotenv
LLM_API_KEY=sk-or-...
LLM_BASE_URL=https://openrouter.ai/api/v1
EMBEDDING_API_KEY=sk-or-...
EMBEDDING_BASE_URL=https://openrouter.ai/api/v1
NO_PROXY=localhost,127.0.0.1,::1
```

MiniMax or another OpenAI-compatible agent configuration can use the familiar
`AGENT_*` aliases. Nemori treats them as LLM fallbacks when `LLM_*` is absent:

```dotenv
AGENT_API_KEY=replace_with_a_new_key
AGENT_BASE_URL=https://api.minimaxi.com/v1
AGENT_MODEL=MiniMax-M2.7
AGENT_WORK_DIR=.
AGENT_PROVIDER=minimax
WEB_LLM_MODE=llm
AGENT_TEMPERATURE=0.7
```

Never commit `.env` or paste a live key into an issue/chat. Revoke and rotate a
key immediately if it is exposed.

For TokenRouter, use the SDK base URL rather than the raw cURL endpoint:

```dotenv
LLM_API_KEY=replace_with_your_tokenrouter_key
LLM_BASE_URL=https://api.tokenrouter.com/v1
NEMORI_LLM_MODEL=deepseek/deepseek-v4-flash
NEMORI_JUDGE_MODEL=deepseek/deepseek-v4-flash
AGENT_API_KEY=replace_with_your_tokenrouter_key
AGENT_BASE_URL=https://api.tokenrouter.com/v1
AGENT_MODEL=deepseek/deepseek-v4-flash
AGENT_PROVIDER=tokenrouter
```

Raw HTTP/cURL examples include `/v1/chat/completions`; the OpenAI Python SDK
adds `/chat/completions` itself. The dashboard accepts either form and
normalizes it to the SDK base URL before saving or testing.

### Local Qwen embedding with Ollama

Nemori's embedding client uses the OpenAI-compatible endpoint supported by
Ollama. The small Qwen model is a practical local smoke-test default:

```powershell
ollama pull qwen3-embedding:4b
```

```dotenv
EMBEDDING_API_KEY=ollama
EMBEDDING_BASE_URL=http://127.0.0.1:21434/v1
NEMORI_EMBEDDING_MODEL=qwen3-embedding:4b
```

The API key is required by the OpenAI client but ignored by local Ollama. The
dashboard's **Use local qwen3-embedding:4b** button fills these values and its
connection test reports model health, vector dimension, and latency. Every run
uses an isolated Qdrant collection prefix so cloud and local embedding
dimensions cannot collide.

The checked-in evaluation configs use `openai/gpt-4.1-mini` and
`google/gemini-embedding-001`, which are OpenRouter model IDs. For direct
OpenAI, change them to `gpt-4.1-mini` and `text-embedding-3-small` and omit the
base URLs.

## Start infrastructure

```powershell
docker compose up -d
docker compose ps
```

Both containers should report `healthy`.

## Real-time control room

Start the local-only dashboard from the repository root:

```powershell
.\.venv\Scripts\python.exe -m dashboard
```

Open <http://127.0.0.1:8765>. The server binds to `127.0.0.1` by default; do
not expose it to a public interface because it can edit `.env` and control
benchmark processes.

Recommended first run:

1. Open **Environment & connections**, choose a provider preset, enter the
   relevant key, save, and run the three independent LLM, Agent, and embedding
   checks. Each result shows the actual endpoint and model used.
2. Confirm PostgreSQL and Qdrant show `ONLINE`.
3. Create a LoCoMo run with `start_index=0` and `limit=1`.
4. Keep the default low ingestion concurrency, then start the run. Start always
   performs an exact LLM, embedding, and Judge preflight and will not launch a
   subprocess when any configured model fails.
5. Watch the four-stage pipeline, process log, CPU/memory, checkpointed answers,
   and final metrics. Pause/resume suspends the whole process tree; stop retains
   logs and any checkpoint file already written.
6. Increase the sample range only after the smoke run completes. A full
   LongMemEval-S run contains 500 histories and can consume substantial time
   and model budget.

For SWE-bench Lite, choose the benchmark in the same dialog and keep the first
run at one instance. `gold` validates the Docker harness; it is not an agent
score. To score an agent, provide its prediction JSONL path. The start preflight
checks Docker Engine, registry access, Ubuntu WSL2, the isolated `uv` runner,
and free disk space before launching.

Each run is persisted under `.nemori-dashboard/runs/<run-id>/` with an isolated
agent ID, config snapshot, selected dataset slice, log, intermediate answers,
metrics, and summary. Secrets are never returned by the environment API and are
redacted from streamed process output.

The **Internal workflow canvas** expands the two most important stages:

- **Build memory** shows source normalization, message buffering, topic
  segmentation, third-person episode narration, vectorization, similar-episode
  merging, prediction-correction semantic extraction, and both stores.
- **Retrieve & answer** shows query embedding, episode/semantic parallel
  retrieval, optional RRF hybrid fusion, top-k context assembly, answer prompt,
  and checkpoint persistence.

Click a node to see its real code location and decision rule. During a run,
machine-readable traces add real message previews, source-to-narrative ratios,
episode narratives, semantic facts, retrieved memories, context size, and the
generated answer. The canvas supports drag-to-pan, wheel zoom, fit-to-view, and
dark/light themes.

Important nuance: narrative generation is not a fixed character truncation.
Messages enter the persistent buffer first. Batch segmentation only runs when
one processing pass sees at least `batch_threshold` messages; its prompt favors
focused 2–15-message topic groups and each segmentation call is capped at 80
messages. Small `batch_size` / `buffer_size_min` values can therefore produce
much finer episodes. The canvas is designed to make this behavior visible.

## Optional smoke subsets

Create one-dialogue/one-question subsets before spending money on the complete
benchmarks:

```powershell
@'
import json
from pathlib import Path

p = Path("evaluation/dataset")
for source, target in [
    ("locomo10.json", "locomo1.json"),
    ("longmemeval_s.json", "longmemeval_1.json"),
]:
    data = json.loads((p / source).read_text(encoding="utf-8"))
    (p / target).write_text(
        json.dumps(data[:1], ensure_ascii=False), encoding="utf-8"
    )
'@ | .\.venv\Scripts\python.exe -
```

Use the subset filename in the commands below. Do not ingest a subset and then
the full dataset into the same volumes: the same user/question IDs will be
inserted again. To deliberately erase benchmark state, run `docker compose
down -v`, then `docker compose up -d`.

## LoCoMo

The Nemori scripts score 1,540 questions in categories 1-4 and intentionally
skip the 446 adversarial questions in category 5.

```powershell
Set-Location evaluation
$env:PYTHONPATH = ".."
$env:LLM_JUDGE_MODEL = "openai/gpt-4o-mini"  # OpenRouter; use gpt-4o-mini for direct OpenAI

..\.venv\Scripts\python.exe locomo\add.py `
  --data dataset\locomo10.json --batch-size 1 --max-workers 1

..\.venv\Scripts\python.exe locomo\search.py `
  --data dataset\locomo10.json `
  --output locomo\results.json `
  --top-k-episodes 10 --top-k-semantic 20 `
  --include-original-messages-top-k 2 `
  --search-method vector --max-workers 10

..\.venv\Scripts\python.exe locomo\evals.py `
  --input_file locomo\results.json `
  --output_file locomo\metrics.json --max_workers 10

..\.venv\Scripts\python.exe locomo\generate_scores.py
```

`generate_scores.py` prints BLEU-1, token F1, and LLM-judge accuracy by
category and overall. Scores are decimals in `[0, 1]`; the paper reports the
same values on a `[0, 100]` scale.

## LongMemEval-S

LongMemEval-S has 500 independently timestamped histories across six question
types. Its histories are much larger than LoCoMo, so start with
`longmemeval_1.json` and low ingestion concurrency.

```powershell
Set-Location evaluation
$env:PYTHONPATH = ".."

..\.venv\Scripts\python.exe longmemeval\add.py `
  --data_path dataset\longmemeval_s.json `
  --batch_size 1 --max_workers 1

..\.venv\Scripts\python.exe longmemeval\search.py `
  --data dataset\longmemeval_s.json `
  --output longmemeval\results.json `
  --top-k-episodes 10 --top-k-semantic 20 `
  --include-original-messages-top-k 2 `
  --search-method vector

..\.venv\Scripts\python.exe longmemeval\evals.py `
  longmemeval\results.json `
  --model openai/gpt-4o-mini `
  --output longmemeval\report.json --max-concurrent 10
```

Use `gpt-4o-mini` instead of `openai/gpt-4o-mini` when calling OpenAI directly.
The report contains overall accuracy, accuracy by question type, and example
errors.

## SWE-bench Lite

The dashboard exposes a single Nemori SWE pipeline. It downloads the dataset,
exports each official repository at `base_commit`, builds native Episode and
Semantic Memory from source code, retrieves issue-relevant context, runs the
Coding Agent, and sends the generated patch to `swebench==4.1.0`.

PostgreSQL and Qdrant are started automatically with Docker Compose. Ubuntu
WSL2 must provide `uv` at `$HOME/.local/bin/uv`; Docker Desktop must use Linux
containers. Configure the Agent, Nemori LLM and embedding credentials in the
web page, then click **一键运行 Nemori SWE**. No prediction file, memory file,
repository checkout or dataset path is supplied manually.

Artifacts are saved under `runs/nemori-swe/<run-id>/`. Evaluator-only fields
and future repository history never enter code memory or the Agent input.

## Stop services

```powershell
Set-Location ..
docker compose down       # keep benchmark data
# docker compose down -v  # erase PostgreSQL and Qdrant benchmark data
```
