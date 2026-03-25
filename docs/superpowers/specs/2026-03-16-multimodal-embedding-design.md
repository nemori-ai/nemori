# Nemori Multimodal Support + Embedding Dimension Adaptation

**Date:** 2026-03-16
**Status:** Approved
**Scope:** Extend Nemori to support multimodal messages (images) and dynamically adapt to embedding model dimensions

---

## 1. Context & Decisions

Nemori currently only supports text messages. Users want to send conversations with images (user-attached photos, screenshots, etc.). Additionally, different embedding models produce different vector dimensions (OpenAI: 1536, Gemini: 3072, Qwen: 4096) but the DB schema hardcodes `vector(1536)`.

### Key Decisions

| Dimension | Decision |
|-----------|----------|
| Multimodal format | OpenAI content array standard: `str \| list[ContentPart]` |
| Image handling | Pass to LLM for understanding; episode output remains text narrative |
| Embedding of images | Text-only embedding (from generated narrative, not from images) |
| Embedding dimension | Probe at startup, auto-adapt schema |
| Dimension change | ALTER + SET NULL existing embeddings + warning log |
| Breaking changes | Zero — `content: str` continues working |

---

## 2. Multimodal Message Model

### Message.content type change

```python
ContentPart = dict[str, Any]  # {"type": "text", "text": "..."} or {"type": "image_url", "image_url": {"url": "..."}}
MessageContent = str | list[ContentPart]

@dataclass
class Message:
    role: str
    content: MessageContent  # backward-compatible: str still works
    ...
```

### New helper methods on Message

```python
def text_content(self) -> str:
    """Extract text parts only. Used for embedding, search, token counting."""
    if isinstance(self.content, str):
        return self.content
    parts = []
    for part in self.content:
        if part.get("type") == "text":
            parts.append(part["text"])
        elif part.get("type") == "image_url":
            parts.append("[image]")
    return " ".join(parts)

def has_images(self) -> bool:
    if isinstance(self.content, str):
        return False
    return any(p.get("type") == "image_url" for p in self.content)

def image_urls(self) -> list[str]:
    if isinstance(self.content, str):
        return []
    return [
        p["image_url"]["url"]
        for p in self.content
        if p.get("type") == "image_url"
    ]
```

### text_content() for embedding

When used for embedding generation, image placeholders should be omitted to avoid noise:
```python
def text_content(self, include_placeholders: bool = True) -> str:
    if isinstance(self.content, str):
        return self.content
    parts = []
    for part in self.content:
        if part.get("type") == "text":
            parts.append(part["text"])
        elif part.get("type") == "image_url" and include_placeholders:
            parts.append("[image]")
    return " ".join(parts)
```

Embedding uses `text_content(include_placeholders=False)`. Display/token counting uses default `True`.

### Zero-text messages

If a message contains only images and no text parts, `text_content()` returns empty string. The episode generator handles this gracefully (images still appear in the content array for LLM). Embedding is skipped for empty text.

### Type alias export

`MessageContent` and `ContentPart` are defined in `src/domain/models.py` and exported from `src/domain/__init__.py`.

### Serialization

`to_dict()` and `from_dict()` handle both `str` and `list` content transparently — `from_dict` does `content=data["content"]` which works for both types without modification. `source_messages` JSONB already stores arbitrary dicts — no schema change needed for episodes/semantics.

---

## 3. Embedding Dimension Dynamic Adaptation

### Startup probe

```python
# src/services/embedding.py
class AsyncEmbeddingClient:
    async def probe_dimension(self) -> int:
        vec = await self.embed("dimension probe")
        return len(vec)
```

### Facade auto-detection

```python
# src/api/facade.py — in __aenter__
actual_dim = await embedding.probe_dimension()
if actual_dim != self._config.embedding_dimension:
    logger.info("Embedding dimension probe: %d (config was %d)", actual_dim, self._config.embedding_dimension)
    self._config.embedding_dimension = actual_dim
```

### Schema adaptation

`migrations.py` uses the probed dimension for initial table creation. When dimension changes on an existing database:

```python
# migration v2: dimension change
async def _adapt_vector_dimension(conn, new_dim):
    """ALTER vector columns and nullify stale embeddings."""
    for table in ("episodes", "semantic_memories"):
        await conn.execute(f"ALTER TABLE {table} ALTER COLUMN embedding TYPE vector({new_dim})")
        await conn.execute(f"UPDATE {table} SET embedding = NULL WHERE embedding IS NOT NULL")
```

This runs automatically during `ensure_schema` when a dimension mismatch is detected. WARNING log emitted to inform user that embeddings need re-generation.

### Dimension adaptation must also recreate HNSW indexes

`ALTER COLUMN embedding TYPE vector(new_dim)` silently invalidates HNSW indexes. The migration must:
1. `DROP INDEX IF EXISTS idx_episodes_embedding, idx_semantic_embedding`
2. ALTER column type
3. SET embedding = NULL
4. `CREATE INDEX ... USING hnsw ...` with new dimension

### Probe failure behavior

- **Fresh database (no tables):** Probe failure raises `ConfigError` — cannot create tables without knowing dimension.
- **Existing database:** Probe failure logs WARNING, detects current dimension from existing vector column via `SELECT atttypmod FROM pg_attribute` and uses that. This prevents silent dimension mismatches.

### message_buffer.content column migration

The `message_buffer.content` column must change from `TEXT` to `JSONB` to support multimodal content arrays. This is included in migration v2:

```sql
ALTER TABLE message_buffer ALTER COLUMN content TYPE JSONB USING to_jsonb(content);
```

`PgMessageBufferStore.push()` serializes `msg.content` (str or list) via `json.dumps()`. `get_unprocessed()` deserializes back. For existing TEXT rows, the `USING to_jsonb(content)` cast wraps the string in JSON quotes automatically.

---

## 4. Episode Generation — Multimodal Prompt

### When messages contain images

1. **Build content array**: Text prompt + up to 10 image URLs
2. **Mark image positions** in conversation text: `[Image attached]`
3. **Append image guidance** to prompt:

```
If images are included in this conversation:
1. Use the images to enrich your understanding of what the user was doing or discussing.
2. Describe the visual context naturally within the narrative.
3. Do NOT reference technical details like "image_url" or "screenshot #3".
4. Integrate visual information chronologically with the text conversation.
```

### Fallback episode path

`EpisodeGenerator._create_fallback()` must use `m.text_content()` instead of `m.content` when building the conversation string.

### PromptTemplates.format_conversation

`format_conversation()` receives message dicts. It must handle `content` being either `str` or `list[ContentPart]` using the same `_extract_text()` pattern (inline text, `[Image attached]` for images).

### When messages are text-only

No change — existing behavior preserved.

### Episode output

Always pure text narrative. Images enrich LLM understanding but the generated episode `content` is text. Images are preserved in `source_messages` JSONB for traceability.

### Embedding

Based on generated text narrative only (`title + content`). No vision embedding.

---

## 5. Other Generator Adjustments

### SemanticGenerator

All code paths that read `source_messages[*]['content']` must use the `_extract_text()` helper, specifically:
- `_prediction_correction()` — builds `original` conversation text from source_messages
- `_direct_extraction()` — uses episode.content (already text, no change needed)

`_extract_text(msg_dict)` handles both `str` and `list[ContentPart]` content formats. Does NOT pass images to LLM — semantic extraction works from the episode's text narrative.

### BatchSegmenter

Uses `msg.text_content()` instead of `msg.content` when formatting messages for the segmentation prompt.

---

## 6. Files Changed

| File | Change |
|------|--------|
| `src/domain/models.py` | `Message.content` type union + `text_content()`, `has_images()`, `image_urls()` |
| `src/services/embedding.py` | `probe_dimension()` method |
| `src/api/facade.py` | Probe dimension at startup, handle mismatch |
| `src/db/migrations.py` | Add dimension adaptation logic |
| `src/llm/generators/episode.py` | Multimodal prompt building, image markers, fallback |
| `src/llm/generators/semantic.py` | `_extract_text()` helper for content arrays |
| `src/llm/generators/segmenter.py` | Use `text_content()` |
| `src/llm/prompts.py` | Append multimodal image guidance + fix `format_conversation()` for content arrays |
| `src/db/buffer_store.py` | Serialize/deserialize `msg.content` as JSON for JSONB column |
| `src/db/migrations.py` | v2: buffer content TEXT→JSONB + vector dimension adaptation + HNSW index rebuild |
| `src/domain/__init__.py` | Export `MessageContent`, `ContentPart` type aliases |

### Not changed

- `src/db/episode_store.py` — JSONB handles multimodal content transparently
- `src/db/semantic_store.py` — no change
- `src/db/connection.py` — no change
- `src/core/memory_system.py` — no change
- `src/search/unified.py` — no change
- `src/domain/interfaces.py` — no change
- `src/domain/exceptions.py` — no change
- `src/config.py` — `embedding_dimension` overwritten at runtime, no field change

---

## 7. Design Principles

- **Zero breaking changes**: `content: str` continues working everywhere
- **OpenAI standard**: Content array format is the industry standard, supported by all major LLMs
- **Text-first embedding**: Images enrich understanding but don't participate in vector search
- **Probe over config**: Actual dimension detected at runtime, not trusted from config
- **Graceful degradation**: Probe failure on fresh DB raises ConfigError; on existing DB falls back to detected column dimension
- **LLM provider assumption**: LLMProvider implementations must support OpenAI-format content arrays in message `content` fields. This is documented, not enforced at protocol level.
- **Image cap**: Max 10 images per episode generation, defined as `MAX_IMAGES_PER_EPISODE = 10` constant in `episode.py`
- **YAGNI**: No ContentBlock abstraction, no vision embedding, no audio support yet
