"""
Integration tests: verify Nemori's LLM client and embedding client
work correctly with real OpenRouter / OpenAI-compatible endpoints.

Tests real API calls — requires valid API keys in .env
"""
import asyncio
import os
import sys
import json
import time

# Load .env
from dotenv import load_dotenv
load_dotenv()

from nemori.llm.client import AsyncLLMClient
from nemori.llm.orchestrator import LLMOrchestrator, LLMRequest, LLMResponse
from nemori.services.embedding import AsyncEmbeddingClient
from nemori.domain.exceptions import LLMError

# ─── Config ───────────────────────────────────────────────────────────
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

LLM_MODELS = [
    "anthropic/claude-sonnet-4.6",
    "google/gemini-3.1-flash-lite-preview",
    "openai/gpt-5.4",
    "deepseek/deepseek-v3.2",
]

EMBEDDING_CONFIGS = [
    # (name, api_key, base_url, model)
    ("google/gemini-embedding-001 (OpenRouter)", LLM_API_KEY, LLM_BASE_URL, "google/gemini-embedding-001"),
    ("qwen/qwen3-embedding-8b (OpenRouter)", LLM_API_KEY, LLM_BASE_URL, "qwen/qwen3-embedding-8b"),
]

# If user has a direct OpenAI key, also test text-embedding-3-small
if OPENAI_API_KEY and OPENAI_API_KEY != LLM_API_KEY:
    EMBEDDING_CONFIGS.append(
        ("text-embedding-3-small (OpenAI direct)", OPENAI_API_KEY, None, "text-embedding-3-small"),
    )

TEST_PROMPT = "What is 2+2? Reply with just the number."
TEST_EMBED_TEXT = "I love hiking in the mountains on weekends."
TEST_EMBED_QUERY = "outdoor activities"


def ok(msg: str):
    print(f"  ✅ {msg}")

def fail(msg: str):
    print(f"  ❌ {msg}")

def info(msg: str):
    print(f"  ℹ️  {msg}")


# ─── LLM Tests ────────────────────────────────────────────────────────
async def test_llm_model(model: str):
    """Test a single LLM model via OpenRouter."""
    print(f"\n🔹 Testing LLM: {model}")
    client = AsyncLLMClient(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    try:
        start = time.monotonic()
        result = await client.complete(
            [{"role": "user", "content": TEST_PROMPT}],
            model=model,
            temperature=0.0,
            max_tokens=50,
        )
        latency = (time.monotonic() - start) * 1000

        if result and len(result.strip()) > 0:
            ok(f"Response: '{result.strip()[:80]}' ({latency:.0f}ms)")
            return True
        else:
            fail(f"Empty response ({latency:.0f}ms)")
            return False
    except Exception as e:
        fail(f"{type(e).__name__}: {e}")
        return False


async def test_llm_with_orchestrator(model: str):
    """Test LLM model through the orchestrator (retry, concurrency)."""
    print(f"\n🔹 Testing Orchestrator → {model}")
    client = AsyncLLMClient(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    orch = LLMOrchestrator(
        provider=client,
        default_model=model,
        max_concurrent=5,
    )

    try:
        request = LLMRequest(
            messages=(
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello in exactly 3 words."},
            ),
            model=model,
            temperature=0.0,
            max_tokens=50,
            timeout=30.0,
            retries=2,
        )
        response = await orch.execute(request)

        if isinstance(response, LLMResponse) and response.content:
            ok(f"Response: '{response.content.strip()[:80]}' ({response.latency_ms:.0f}ms, id={response.request_id})")
            return True
        else:
            fail("Invalid response object")
            return False
    except Exception as e:
        fail(f"{type(e).__name__}: {e}")
        return False


async def test_llm_json_mode(model: str):
    """Test JSON output (used by episode/semantic generators)."""
    print(f"\n🔹 Testing JSON output: {model}")
    client = AsyncLLMClient(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    prompt = """Extract facts from this text and return JSON:
Text: "Alice works at Google as a senior engineer. She likes hiking."

Return ONLY a JSON object:
{"statements": ["fact 1", "fact 2"]}"""

    try:
        result = await client.complete(
            [{"role": "user", "content": prompt}],
            model=model,
            temperature=0.0,
            max_tokens=200,
        )

        # Try to parse JSON (strip markdown fences if present)
        text = result.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        parsed = json.loads(text)
        if "statements" in parsed and isinstance(parsed["statements"], list):
            ok(f"Valid JSON with {len(parsed['statements'])} statements")
            for s in parsed["statements"][:3]:
                info(f"  → {s}")
            return True
        else:
            fail(f"JSON missing 'statements' key: {text[:100]}")
            return False
    except json.JSONDecodeError:
        fail(f"Not valid JSON: {result.strip()[:100]}")
        return False
    except Exception as e:
        fail(f"{type(e).__name__}: {e}")
        return False


# ─── Embedding Tests ──────────────────────────────────────────────────
async def test_embedding(name: str, api_key: str, base_url: str | None, model: str):
    """Test a single embedding model."""
    print(f"\n🔹 Testing Embedding: {name}")
    client = AsyncEmbeddingClient(api_key=api_key, model=model, base_url=base_url)

    try:
        # Single embed
        start = time.monotonic()
        vec = await client.embed(TEST_EMBED_TEXT)
        latency = (time.monotonic() - start) * 1000

        if not isinstance(vec, list) or len(vec) == 0:
            fail("Empty or invalid embedding")
            return False

        ok(f"Single embed: dim={len(vec)}, first3={vec[:3]}, ({latency:.0f}ms)")

        # Batch embed
        start = time.monotonic()
        vecs = await client.embed_batch([TEST_EMBED_TEXT, TEST_EMBED_QUERY])
        latency = (time.monotonic() - start) * 1000

        if len(vecs) != 2:
            fail(f"Batch returned {len(vecs)} vectors, expected 2")
            return False

        ok(f"Batch embed: 2 vectors, dim={len(vecs[0])}, ({latency:.0f}ms)")

        # Cosine similarity check
        def cosine_sim(a, b):
            dot = sum(x*y for x, y in zip(a, b))
            na = sum(x*x for x in a) ** 0.5
            nb = sum(x*x for x in b) ** 0.5
            return dot / (na * nb) if na and nb else 0

        sim = cosine_sim(vecs[0], vecs[1])
        if sim > 0.3:
            ok(f"Cosine similarity: {sim:.4f} (related texts → reasonable)")
        else:
            info(f"Cosine similarity: {sim:.4f} (low — may indicate issue)")

        return True
    except Exception as e:
        fail(f"{type(e).__name__}: {e}")
        return False


# ─── Full pipeline test ──────────────────────────────────────────────
async def test_episode_generation_pipeline():
    """Test the full episode generation prompt → parse pipeline."""
    print(f"\n🔹 Testing Episode Generation Pipeline")

    from nemori.llm.generators.episode import EpisodeGenerator
    from nemori.domain.models import Message

    client = AsyncLLMClient(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    orch = LLMOrchestrator(provider=client, default_model="anthropic/claude-sonnet-4.6")

    # Use a simple mock embedding to avoid needing embedding API
    class MockEmbedding:
        async def embed(self, text: str) -> list[float]:
            return [0.1] * 256
        async def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [[0.1] * 256 for _ in texts]

    gen = EpisodeGenerator(orchestrator=orch, embedding=MockEmbedding())

    messages = [
        Message(role="user", content="I just moved to Tokyo last month. The food is incredible!"),
        Message(role="assistant", content="That's exciting! What's your favorite dish so far?"),
        Message(role="user", content="Definitely ramen. I've been trying a different shop every day."),
    ]

    try:
        episode = await gen.generate("test_user", messages, "topic_complete")

        if episode.title and episode.content:
            ok(f"Title: {episode.title[:60]}")
            ok(f"Content length: {len(episode.content)} chars")
            ok(f"Metadata: {episode.metadata}")
            if episode.metadata.get("fallback"):
                info("Used fallback (LLM response wasn't parseable JSON)")
            else:
                ok("LLM JSON parsing successful")
            return True
        else:
            fail("Episode missing title or content")
            return False
    except Exception as e:
        fail(f"{type(e).__name__}: {e}")
        return False


# ─── Main ─────────────────────────────────────────────────────────────
async def main():
    print("=" * 60)
    print("Nemori Integration Test — OpenRouter API Compatibility")
    print("=" * 60)

    results = {"pass": 0, "fail": 0}

    # 1. LLM direct calls
    print("\n" + "─" * 40)
    print("1. LLM Direct Calls")
    print("─" * 40)
    for model in LLM_MODELS:
        if await test_llm_model(model):
            results["pass"] += 1
        else:
            results["fail"] += 1

    # 2. LLM via Orchestrator
    print("\n" + "─" * 40)
    print("2. LLM via Orchestrator")
    print("─" * 40)
    # Test with one fast model
    if await test_llm_with_orchestrator("google/gemini-3.1-flash-lite-preview"):
        results["pass"] += 1
    else:
        results["fail"] += 1

    # 3. JSON mode
    print("\n" + "─" * 40)
    print("3. JSON Output (Generator Compatibility)")
    print("─" * 40)
    for model in LLM_MODELS[:2]:  # Test first 2 models
        if await test_llm_json_mode(model):
            results["pass"] += 1
        else:
            results["fail"] += 1

    # 4. Embeddings
    print("\n" + "─" * 40)
    print("4. Embedding Models")
    print("─" * 40)
    for name, key, url, model in EMBEDDING_CONFIGS:
        if await test_embedding(name, key, url, model):
            results["pass"] += 1
        else:
            results["fail"] += 1

    # 5. Full pipeline
    print("\n" + "─" * 40)
    print("5. Episode Generation Pipeline")
    print("─" * 40)
    if await test_episode_generation_pipeline():
        results["pass"] += 1
    else:
        results["fail"] += 1

    # Summary
    print("\n" + "=" * 60)
    total = results["pass"] + results["fail"]
    print(f"Results: {results['pass']}/{total} passed, {results['fail']}/{total} failed")
    print("=" * 60)

    return results["fail"] == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
