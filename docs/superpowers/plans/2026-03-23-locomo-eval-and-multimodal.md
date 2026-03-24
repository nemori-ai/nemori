# LoCoMo Evaluation + Multimodal Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate the refactored Nemori system achieves 80%+ accuracy on the LoCoMo benchmark (using OpenRouter + gpt-4.1-mini), then add full multimodal message support.

**Architecture:** Two-phase approach. Phase A fixes evaluation infrastructure (config, timestamp bug, OpenRouter compatibility) and runs the LoCoMo pipeline iteratively until 80%+ LLM Judge accuracy. Phase B adds multimodal support (image compression, updated facade/prompts) and verifies no regression on LoCoMo.

**Tech Stack:** Python 3.11+, asyncpg, pgvector, OpenAI SDK (via OpenRouter), PIL/Pillow for image processing, pytest

---

## Critical Bug Found During Planning

**Facade `add_messages` drops timestamps** (`nemori/api/facade.py:100-105`):
```python
msg_objects = [
    Message(role=m["role"], content=m["content"]) for m in messages
]
```
This discards `timestamp` and `metadata` from input dicts. The evaluation `add.py` scripts pass timestamps but they are silently lost, causing all episodes to have `datetime.now()` timestamps instead of the actual conversation dates. This critically impacts temporal reasoning accuracy (LoCoMo categories 2 & 3).

---

## Phase A: LoCoMo Evaluation (Target: 80%+ LLM Judge Accuracy)

### Task 1: Fix Facade Timestamp Bug

**Files:**
- Modify: `nemori/api/facade.py:100-105`
- Test: `tests/test_facade.py`

- [ ] **Step 1: Write the failing test**

Uses the same inline patching pattern as existing tests in `test_facade.py`:

```python
# In tests/test_facade.py — add test for timestamp/metadata passthrough

@pytest.mark.asyncio
async def test_add_messages_preserves_timestamp_and_metadata():
    """add_messages should pass timestamp and metadata from input dicts."""
    with patch("nemori.api.facade.DatabaseManager") as MockDB, \
         patch("nemori.api.facade.NemoriMemory._build_system", new_callable=AsyncMock):
        MockDB.return_value = AsyncMock()

        config = MemoryConfig(dsn="postgresql://localhost/test", llm_api_key="test")
        async with NemoriMemory(config=config) as memory:
            memory._system = AsyncMock()
            messages = [
                {
                    "role": "user",
                    "content": "hello",
                    "timestamp": "2023-05-08T13:56:00",
                    "metadata": {"source": "test"},
                }
            ]
            await memory.add_messages("u1", messages)
            call_args = memory._system.add_messages.call_args
            msg = call_args[0][1][0]
            assert msg.timestamp.year == 2023
            assert msg.timestamp.month == 5
            assert msg.metadata.get("source") == "test"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_facade.py::test_add_messages_preserves_timestamp_and_metadata -v`
Expected: FAIL — timestamp is `datetime.now()`, not 2023

- [ ] **Step 3: Fix `add_messages` in facade**

Modify `nemori/api/facade.py:100-105`:
```python
async def add_messages(self, user_id: str, messages: list[dict[str, Any]]) -> None:
    system = self._ensure_system()
    msg_objects = []
    for m in messages:
        kwargs: dict[str, Any] = {"role": m["role"], "content": m["content"]}
        if "timestamp" in m:
            ts = m["timestamp"]
            if isinstance(ts, str):
                from datetime import datetime
                ts = datetime.fromisoformat(ts)
            kwargs["timestamp"] = ts
        if "metadata" in m:
            kwargs["metadata"] = m["metadata"]
        msg_objects.append(Message(**kwargs))
    await system.add_messages(user_id, msg_objects)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_facade.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add nemori/api/facade.py tests/test_facade.py
git commit -m "fix: preserve timestamp and metadata in facade add_messages"
```

---

### Task 2: Update Evaluation Config for OpenRouter

OpenRouter model names:
- LLM: `openai/gpt-4.1-mini` (on OpenRouter)
- Embedding: `text-embedding-3-small` (on OpenRouter: `openai/text-embedding-3-small`)
- LLM Judge: `openai/gpt-4.1-mini` (same model, via OpenRouter)

**Files:**
- Modify: `evaluation/locomo/config.json`
- Modify: `evaluation/longmemeval/config.json`

- [ ] **Step 1: Update locomo config.json**

```json
{
  "dsn": "postgresql://localhost/nemori",
  "llm_model": "openai/gpt-4.1-mini",
  "llm_base_url": "https://openrouter.ai/api/v1",
  "embedding_model": "openai/text-embedding-3-small",
  "embedding_base_url": "https://openrouter.ai/api/v1",
  "buffer_size_min": 1,
  "buffer_size_max": 20,
  "episode_min_messages": 1,
  "episode_max_messages": 20,
  "enable_semantic_memory": true,
  "enable_prediction_correction": true,
  "search_top_k_episodes": 10,
  "search_top_k_semantic": 20
}
```

- [ ] **Step 2: Update longmemeval config.json (same pattern)**

- [ ] **Step 3: Commit**

```bash
git add evaluation/locomo/config.json evaluation/longmemeval/config.json
git commit -m "chore: update evaluation configs for OpenRouter"
```

---

### Task 3: Fix Evaluation Config Loading to Pass API Keys and Base URLs

The `load_config()` in evaluation scripts filters fields to `MemoryConfig` valid fields, but `llm_base_url`, `embedding_base_url`, `llm_api_key`, `embedding_api_key` are all valid MemoryConfig fields. However, the config JSON doesn't include API keys (they come from env vars via `_resolve_llm_key`).

The real issue: `MemoryConfig.__init__` resolves API keys from env vars via `default_factory`, so they are auto-populated. But `llm_base_url` and `embedding_base_url` default to `None` — they must be in the config JSON (done in Task 2) OR set via env vars.

**Files:**
- Modify: `evaluation/locomo/search.py:141-143` (OpenAI client for answer generation)
- Modify: `evaluation/locomo/metrics/llm_judge.py:13-16` (OpenAI client for judging)

- [ ] **Step 1: Fix search.py answer client to use config model and OpenRouter**

Replace the `AsyncOpenAI()` initialization in `LongMemEvalSearcher.__init__` and `LocomoSearcher.__init__` to use config's API key and base URL:

In `evaluation/locomo/search.py`, change `__init__`:
```python
try:
    self.openai_client = AsyncOpenAI(
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
    )
except Exception:
    self.openai_client = None
```

In `evaluation/longmemeval/search.py`, same change.

- [ ] **Step 2: Fix llm_judge.py hardcoded model name**

The client creation in `evaluation/locomo/metrics/llm_judge.py:13-16` already reads `LLM_API_KEY` and `LLM_BASE_URL` from env — that part is fine. The bug is `evaluate_llm_judge()` at line 48 hardcodes `model="gpt-4o-mini"` which fails on OpenRouter (needs `openai/` prefix). Fix by making the model configurable:

Add after the client creation (line 16):
```python
# Default judge model — reads from env or uses OpenRouter-compatible name
JUDGE_MODEL = os.getenv("LLM_JUDGE_MODEL", "openai/gpt-4.1-mini")
```

Update `evaluate_llm_judge` at line 45:
```python
def evaluate_llm_judge(question, gold_answer, generated_answer, model=None):
    response = client.chat.completions.create(
        model=model or JUDGE_MODEL,
        ...
    )
```

- [ ] **Step 3: Run unit tests to ensure no import breakage**

Run: `pytest tests/ -v --ignore=tests/test_integration_openrouter.py`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add evaluation/locomo/search.py evaluation/longmemeval/search.py evaluation/locomo/metrics/llm_judge.py
git commit -m "fix: evaluation scripts use OpenRouter-compatible client config"
```

---

### Task 4: Download LoCoMo Dataset

**Files:**
- Create: `evaluation/dataset/` directory

- [ ] **Step 1: Create dataset directory and download**

```bash
cd evaluation
mkdir -p dataset
wget https://github.com/snap-research/locomo/raw/main/data/locomo10.json -O dataset/locomo10.json
```

- [ ] **Step 2: Verify dataset structure**

```bash
python -c "import json; d=json.load(open('evaluation/dataset/locomo10.json')); print(f'{len(d)} conversations, {sum(len(x.get(\"qa\",[]))for x in d)} QA pairs')"
```

Expected: 10 conversations with ~200+ QA pairs

- [ ] **Step 3: Add dataset to .gitignore (large files)**

Append to `.gitignore`:
```
evaluation/dataset/
```

- [ ] **Step 4: Commit**

```bash
git add .gitignore
git commit -m "chore: add evaluation dataset directory to gitignore"
```

---

### Task 5: Ensure PostgreSQL Database is Ready

- [ ] **Step 1: Verify PostgreSQL is running and database exists**

```bash
psql -c "SELECT 1" postgresql://localhost/nemori
```

If the database doesn't exist:
```bash
createdb nemori
psql -c "CREATE EXTENSION IF NOT EXISTS vector" nemori
```

- [ ] **Step 2: Verify .env has correct API keys**

Ensure `.env` contains:
```
LLM_API_KEY=sk-or-v1-...          # OpenRouter API key
LLM_BASE_URL=https://openrouter.ai/api/v1
EMBEDDING_API_KEY=sk-or-v1-...     # Same or different key for embeddings
EMBEDDING_BASE_URL=https://openrouter.ai/api/v1
```

Or if using direct OpenAI for embeddings:
```
LLM_API_KEY=sk-or-v1-...
LLM_BASE_URL=https://openrouter.ai/api/v1
OPENAI_API_KEY=sk-...              # For embeddings via OpenAI directly
```

---

### Task 6: Run LoCoMo Pipeline — Add Memories

- [ ] **Step 1: Run add.py to ingest conversations**

```bash
cd evaluation
python locomo/add.py --data dataset/locomo10.json --config locomo/config.json --max_workers 5
```

Expected output: 10/10 questions processed successfully, episodes created for each.

- [ ] **Step 2: Verify data in database**

```bash
psql nemori -c "SELECT COUNT(*) FROM episodes; SELECT COUNT(*) FROM semantic_memories; SELECT COUNT(*) FROM message_buffer WHERE NOT processed;"
```

Expected: episodes > 0, semantic_memories > 0, unprocessed buffer = 0

---

### Task 7: Run LoCoMo Pipeline — Search & Answer

- [ ] **Step 1: Run search.py**

```bash
cd evaluation
python locomo/search.py --data dataset/locomo10.json --config locomo/config.json --search-method hybrid --top-k-episodes 10 --top-k-semantic 20 --output locomo/results.json
```

- [ ] **Step 2: Spot-check results**

```bash
python -c "
import json
d = json.load(open('evaluation/locomo/results.json'))
for k, items in list(d.items())[:2]:
    for item in items[:2]:
        print(f'Q: {item[\"question\"][:80]}')
        print(f'A: {item[\"answer\"]}')
        print(f'R: {item[\"response\"][:100]}')
        print('---')
"
```

---

### Task 8: Run LoCoMo Pipeline — Evaluate

- [ ] **Step 1: Run evals.py**

**Important:** `evals.py` uses `from metrics.llm_judge import ...` (relative import), so it must be run from the `evaluation/locomo/` directory:

```bash
cd evaluation/locomo
python evals.py --input_file results.json --output_file metrics.json
```

- [ ] **Step 2: Generate scores**

```bash
cd evaluation/locomo
python generate_scores.py
```

Expected output: per-category and overall mean scores for bleu_score, f1_score, llm_score.

- [ ] **Step 3: Check if llm_score >= 0.80**

If YES → proceed to Phase B.
If NO → go to Task 9 (optimization).

---

### Task 9: Optimize if Accuracy < 80% (Conditional, max 3 iterations)

If accuracy is below 80%, investigate and fix. **Maximum 3 optimization iterations** — if still below 80% after 3 rounds, accept the best result and proceed to Phase B (report the gap to the user for manual investigation). Common levers in order of impact:

**9a. Timestamp accuracy** — Already fixed in Task 1. Verify episodes have correct timestamps:
```sql
SELECT title, created_at FROM episodes ORDER BY created_at LIMIT 10;
```

**9b. Search method tuning** — Try different search methods:
```bash
# Try vector-only
python locomo/search.py --search-method vector --output locomo/results_vector.json
# Try text-only
python locomo/search.py --search-method text --output locomo/results_text.json
```

**9c. Top-K tuning** — Increase retrieval:
```bash
python locomo/search.py --top-k-episodes 20 --top-k-semantic 30 --output locomo/results_topk.json
```

**9d. Answer prompt improvement** — If retrieved memories are relevant but answers are wrong, improve the `ANSWER_PROMPT` in `evaluation/locomo/search.py`.

**9e. Episode generation quality** — Check if episode content captures key details:
```sql
SELECT title, content FROM episodes WHERE user_id LIKE '%_0' LIMIT 5;
```

**9f. Category-specific analysis** — Check which categories are failing:
```bash
python locomo/generate_scores.py  # Look at per-category breakdown
```
- Category 1: Single-hop factual → depends on retrieval quality
- Category 2: Multi-hop → depends on semantic memory quality
- Category 3: Temporal → depends on timestamp accuracy
- Category 4: Open-ended → generally easier

After each optimization, re-run Tasks 7-8 and check accuracy again. Iterate until 80%+.

- [ ] **Step: Document final configuration and accuracy**

---

### Task 10: Commit Evaluation Results

- [ ] **Step 1: Commit all evaluation fixes and config**

```bash
git add nemori/api/facade.py tests/test_facade.py \
  evaluation/locomo/config.json evaluation/longmemeval/config.json \
  evaluation/locomo/search.py evaluation/longmemeval/search.py \
  evaluation/locomo/metrics/llm_judge.py .gitignore
git commit -m "feat: LoCoMo evaluation pipeline with OpenRouter support"
```

- [ ] **Step 2: Push**

```bash
git push origin feat/hypergraph
```

---

## Phase B: Full Multimodal Support

### Current State Assessment

The codebase already has **partial multimodal support**:
- `Message.content` accepts `str | list[ContentPart]` (OpenAI content array format)
- `Message.has_images()`, `image_urls()`, `text_content()` helper methods exist
- `EpisodeGenerator._build_multimodal_prompt()` sends images to LLM
- `SemanticGenerator._extract_text()` handles content arrays
- `PromptTemplates.format_conversation()` handles content arrays with `[Image attached]` markers
- `PgMessageBufferStore.push()` stores content as `jsonb` (handles both str and list)

**What's missing:**
1. Image compression/preprocessing utility (from nemori-desktop)
2. Facade doesn't expose multimodal-aware API (no image preprocessing)
3. No tests for multimodal paths
4. README/evaluation docs reference old ChromaDB architecture

---

### Task 11: Add Image Compression Utility

Reference: `nemori-desktop/backend/utils/image.py`

**Files:**
- Create: `nemori/utils/image.py`
- Test: `tests/test_image_utils.py`

- [ ] **Step 1: Write failing tests for image compression**

```python
# tests/test_image_utils.py
import base64
import pytest
from nemori.utils.image import compress_image_for_llm, ensure_image_data_url


def _make_test_png(width: int = 200, height: int = 200) -> str:
    """Create a small test PNG as base64 data URL."""
    from PIL import Image
    import io
    img = Image.new("RGB", (width, height), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def test_compress_returns_jpeg_data_url():
    data_url = _make_test_png()
    result = compress_image_for_llm(data_url)
    assert result.startswith("data:image/jpeg;base64,")


def test_compress_respects_max_dimensions():
    data_url = _make_test_png(3000, 2000)
    result = compress_image_for_llm(data_url, max_width=1280, max_height=720)
    # Decode and check dimensions
    from PIL import Image
    import io
    b64 = result.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    assert img.width <= 1280
    assert img.height <= 720


def test_ensure_image_data_url_wraps_raw_base64():
    from PIL import Image
    import io
    img = Image.new("RGB", (10, 10), "blue")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    raw_b64 = base64.b64encode(buf.getvalue()).decode()
    result = ensure_image_data_url(raw_b64)
    assert result.startswith("data:image/jpeg;base64,")


def test_compress_handles_rgba():
    """RGBA images should be converted to RGB."""
    from PIL import Image
    import io
    img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    data_url = f"data:image/png;base64,{b64}"
    result = compress_image_for_llm(data_url)
    assert result.startswith("data:image/jpeg;base64,")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_image_utils.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement image utility**

```python
# nemori/utils/image.py
"""Image compression utilities for multimodal memory processing."""
from __future__ import annotations

import base64
import io
import logging
from typing import Sequence

from PIL import Image

logger = logging.getLogger("nemori")

_FORMAT_SIGNATURES = {
    "/9j/": "jpeg",
    "iVBORw0KGgo": "png",
    "UklGR": "webp",
    "R0lGOD": "gif",
}


def ensure_image_data_url(raw_or_url: str) -> str:
    """Ensure input is a proper data URL. Wraps raw base64 if needed."""
    if raw_or_url.startswith("data:"):
        return raw_or_url
    # Detect format from base64 header
    mime = "image/png"  # default
    for sig, fmt in _FORMAT_SIGNATURES.items():
        if raw_or_url.startswith(sig):
            mime = f"image/{fmt}"
            break
    return f"data:{mime};base64,{raw_or_url}"


def compress_image_for_llm(
    data_url: str,
    max_width: int = 1280,
    max_height: int = 720,
    quality: int = 70,
) -> str:
    """Compress an image for LLM consumption.

    - Converts RGBA/P to RGB
    - Resizes to fit within max_width x max_height
    - Outputs JPEG at specified quality
    - Returns data:image/jpeg;base64,... URL
    """
    # Extract base64 data
    if "," in data_url:
        b64_data = data_url.split(",", 1)[1]
    else:
        b64_data = data_url

    img_bytes = base64.b64decode(b64_data)
    img = Image.open(io.BytesIO(img_bytes))

    # Convert to RGB if needed
    if img.mode in ("RGBA", "LA", "P"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        background.paste(img, mask=img.split()[-1] if "A" in img.mode else None)
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # Resize if needed
    if img.width > max_width or img.height > max_height:
        ratio = min(max_width / img.width, max_height / img.height)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    # Save as JPEG
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def compress_images_for_llm(
    data_urls: Sequence[str],
    max_width: int = 1280,
    max_height: int = 720,
    quality: int = 70,
) -> list[str]:
    """Compress multiple images."""
    return [
        compress_image_for_llm(url, max_width, max_height, quality)
        for url in data_urls
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_image_utils.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add nemori/utils/image.py tests/test_image_utils.py
git commit -m "feat: add image compression utility for multimodal support"
```

---

### Task 12: Add Multimodal Message Support to Facade

Currently `add_messages` accepts dicts with `content` as string or content array — this already works. But we need a convenience method for adding messages with image attachments and optional preprocessing.

**Files:**
- Modify: `nemori/api/facade.py`
- Test: `tests/test_facade.py`

- [ ] **Step 1: Write failing test for multimodal message handling**

```python
# In tests/test_facade.py

@pytest.mark.asyncio
async def test_add_messages_handles_multimodal_content():
    """add_messages should handle OpenAI content array format."""
    with patch("nemori.api.facade.DatabaseManager") as MockDB, \
         patch("nemori.api.facade.NemoriMemory._build_system", new_callable=AsyncMock):
        MockDB.return_value = AsyncMock()
        config = MemoryConfig(dsn="postgresql://localhost/test", llm_api_key="test")
        async with NemoriMemory(config=config) as memory:
            memory._system = AsyncMock()
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Look at this"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
                    ],
                }
            ]
            await memory.add_messages("u1", messages)
            call_args = memory._system.add_messages.call_args
            msg = call_args[0][1][0]
            assert isinstance(msg.content, list)
            assert msg.has_images()
            assert len(msg.image_urls()) == 1
```

- [ ] **Step 2: Run test — should pass already since content is passed through**

Run: `pytest tests/test_facade.py::test_add_messages_handles_multimodal_content -v`
Expected: PASS (content array is already passed through by Task 1 fix)

- [ ] **Step 3: Add `add_multimodal_message` convenience method**

In `nemori/api/facade.py`, add:
```python
async def add_multimodal_message(
    self,
    user_id: str,
    text: str,
    image_urls: list[str] | None = None,
    role: str = "user",
    timestamp: str | None = None,
    compress_images: bool = True,
) -> None:
    """Add a message with optional image attachments.

    Args:
        user_id: User identifier.
        text: Text content of the message.
        image_urls: Optional list of image URLs (data URLs or http URLs).
        role: Message role (default: "user").
        timestamp: Optional ISO timestamp string.
        compress_images: Whether to compress images for LLM (default: True).
    """
    if image_urls:
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for url in image_urls:
            if compress_images:
                from nemori.utils.image import compress_image_for_llm
                url = compress_image_for_llm(url)
            content.append({"type": "image_url", "image_url": {"url": url}})
        msg_dict: dict[str, Any] = {"role": role, "content": content}
    else:
        msg_dict = {"role": role, "content": text}

    if timestamp:
        msg_dict["timestamp"] = timestamp

    await self.add_messages(user_id, [msg_dict])
```

- [ ] **Step 4: Write test for the convenience method**

```python
@pytest.mark.asyncio
async def test_add_multimodal_message_builds_content_array():
    """add_multimodal_message should build proper content array."""
    with patch("nemori.api.facade.DatabaseManager") as MockDB, \
         patch("nemori.api.facade.NemoriMemory._build_system", new_callable=AsyncMock):
        MockDB.return_value = AsyncMock()
        config = MemoryConfig(dsn="postgresql://localhost/test", llm_api_key="test")
        async with NemoriMemory(config=config) as memory:
            memory._system = AsyncMock()
            # Use a tiny real PNG for compress_images to work
            import base64, io
            from PIL import Image
            img = Image.new("RGB", (10, 10), "red")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            data_url = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

            await memory.add_multimodal_message(
                "u1", "Check this image", image_urls=[data_url], compress_images=True
            )
            call_args = memory._system.add_messages.call_args
            msg = call_args[0][1][0]
            assert isinstance(msg.content, list)
            assert msg.content[0]["type"] == "text"
            assert msg.content[1]["type"] == "image_url"
            # Compressed image should be JPEG
            assert "image/jpeg" in msg.content[1]["image_url"]["url"]
```

- [ ] **Step 5: Run all facade tests**

Run: `pytest tests/test_facade.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add nemori/api/facade.py tests/test_facade.py
git commit -m "feat: add multimodal message convenience API with image compression"
```

---

### Task 13: Add Multimodal Tests for Core Pipeline

Verify that multimodal messages flow correctly through the entire pipeline: buffer → segmenter → episode generator → semantic generator.

**Files:**
- Create: `tests/test_multimodal_pipeline.py`

- [ ] **Step 1: Write pipeline integration tests**

```python
# tests/test_multimodal_pipeline.py
"""Tests for multimodal message handling through the pipeline."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from nemori.domain.models import Message, Episode


class TestMultimodalMessage:
    def test_text_content_extracts_text_only(self):
        msg = Message(
            role="user",
            content=[
                {"type": "text", "text": "Hello world"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ],
        )
        assert msg.text_content() == "Hello world [image]"
        assert msg.text_content(include_placeholders=False) == "Hello world"

    def test_has_images_true_for_content_array(self):
        msg = Message(
            role="user",
            content=[
                {"type": "text", "text": "Look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ],
        )
        assert msg.has_images() is True

    def test_has_images_false_for_text_only(self):
        msg = Message(role="user", content="just text")
        assert msg.has_images() is False

    def test_image_urls_extraction(self):
        msg = Message(
            role="user",
            content=[
                {"type": "text", "text": "Two images"},
                {"type": "image_url", "image_url": {"url": "url1"}},
                {"type": "image_url", "image_url": {"url": "url2"}},
            ],
        )
        assert msg.image_urls() == ["url1", "url2"]

    def test_to_dict_preserves_content_array(self):
        content = [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]
        msg = Message(role="user", content=content)
        d = msg.to_dict()
        assert d["content"] == content

    def test_from_dict_restores_content_array(self):
        content = [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]
        msg = Message.from_dict({"role": "user", "content": content})
        assert isinstance(msg.content, list)
        assert msg.has_images()


class TestEpisodeGeneratorMultimodal:
    @pytest.mark.asyncio
    async def test_build_multimodal_prompt_includes_images(self):
        from nemori.llm.generators.episode import EpisodeGenerator

        gen = EpisodeGenerator(
            orchestrator=MagicMock(),
            embedding=MagicMock(),
        )
        messages = [
            Message(role="user", content=[
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ]),
        ]
        parts = gen._build_multimodal_prompt(messages, "topic_change")
        # First part is text prompt
        assert parts[0]["type"] == "text"
        # Second part is image
        assert parts[1]["type"] == "image_url"
        assert parts[1]["image_url"]["url"] == "data:image/png;base64,abc"

    def test_format_with_image_markers(self):
        from nemori.llm.generators.episode import EpisodeGenerator

        gen = EpisodeGenerator(
            orchestrator=MagicMock(),
            embedding=MagicMock(),
        )
        messages = [
            Message(role="user", content=[
                {"type": "text", "text": "Check this out"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ]),
        ]
        result = gen._format_with_image_markers(messages)
        assert "[Image attached]" in result
        assert "Check this out" in result


class TestSemanticGeneratorMultimodal:
    def test_extract_text_handles_content_array(self):
        from nemori.llm.generators.semantic import _extract_text

        msg_dict = {
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "image_url", "image_url": {"url": "data:..."}},
            ]
        }
        result = _extract_text(msg_dict)
        assert result == "Hello [image]"

    def test_extract_text_handles_plain_string(self):
        from nemori.llm.generators.semantic import _extract_text

        msg_dict = {"content": "Just a string"}
        result = _extract_text(msg_dict)
        assert result == "Just a string"
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_multimodal_pipeline.py -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_multimodal_pipeline.py
git commit -m "test: add multimodal pipeline tests for Message, Episode, Semantic"
```

---

### Task 14: Update Package Exports and Add Pillow Dependency

**Files:**
- Modify: `nemori/__init__.py` (add image utils to exports)
- Modify: `pyproject.toml` (add Pillow dependency)

- [ ] **Step 1: Update `nemori/__init__.py`**

Add to exports:
```python
from nemori.utils.image import compress_image_for_llm, compress_images_for_llm
```

- [ ] **Step 2: Add Pillow to pyproject.toml dependencies**

Add `Pillow>=10.0.0` to `[project.dependencies]`.

- [ ] **Step 3: Install and verify**

```bash
pip install -e ".[eval]"
python -c "from nemori import compress_image_for_llm; print('OK')"
```

- [ ] **Step 4: Run full test suite**

```bash
pytest tests/ -v --ignore=tests/test_integration_openrouter.py
```

Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add nemori/__init__.py pyproject.toml
git commit -m "feat: export image utilities and add Pillow dependency"
```

---

### Task 15: Update Evaluation README

**Files:**
- Modify: `evaluation/README.md`

- [ ] **Step 1: Update README to reflect current PostgreSQL + async architecture**

Remove all ChromaDB references, update config examples to show `dsn` + OpenRouter setup, update the workflow commands.

- [ ] **Step 2: Commit**

```bash
git add evaluation/README.md
git commit -m "docs: update evaluation README for PostgreSQL + OpenRouter architecture"
```

---

### Task 16: Re-run LoCoMo with Multimodal Code to Verify No Regression

- [ ] **Step 1: Clean previous evaluation data**

```bash
psql nemori -c "DELETE FROM episodes; DELETE FROM semantic_memories; DELETE FROM message_buffer;"
```

- [ ] **Step 2: Re-run the full LoCoMo pipeline (Tasks 6-8)**

```bash
cd evaluation
python locomo/add.py --data dataset/locomo10.json --config locomo/config.json
python locomo/search.py --data dataset/locomo10.json --config locomo/config.json --output locomo/results_post_multimodal.json
python locomo/evals.py --input_file locomo/results_post_multimodal.json --output_file locomo/metrics_post_multimodal.json
python locomo/generate_scores.py
```

- [ ] **Step 3: Compare accuracy with Phase A baseline**

LLM Judge score should be within ±2% of the Phase A result. If regression > 2%, investigate.

- [ ] **Step 4: Final commit and push**

```bash
git add -A
git commit -m "feat: complete multimodal support with LoCoMo regression verification"
git push origin feat/hypergraph
```

---

## Execution Summary

| Task | Description | Risk | Dependencies |
|------|-------------|------|-------------|
| 1 | Fix facade timestamp bug | **High** — directly impacts eval accuracy | None |
| 2 | Update eval config for OpenRouter | Low | None |
| 3 | Fix eval scripts OpenRouter client | Medium | Task 2 |
| 4 | Download LoCoMo dataset | Low | None |
| 5 | Verify PostgreSQL setup | Low | None |
| 6 | Run LoCoMo add | Medium | Tasks 1-5 |
| 7 | Run LoCoMo search | Medium | Task 6 |
| 8 | Run LoCoMo eval | Low | Task 7 |
| 9 | Optimize if <80% | Variable | Task 8 |
| 10 | Commit & push Phase A | Low | Task 8/9 |
| 11 | Image compression utility | Low | None |
| 12 | Multimodal facade API | Low | Task 11 |
| 13 | Multimodal pipeline tests | Low | Tasks 11-12 |
| 14 | Package exports + Pillow | Low | Task 11 |
| 15 | Update eval README | Low | None |
| 16 | Regression verification | Medium | Tasks 10-14 |

**Parallelizable groups:**
- Tasks 1, 2, 4, 5 can run in parallel
- Tasks 11, 15 can run in parallel (independent of Phase A results)
- Tasks 12, 13 depend on Task 11
