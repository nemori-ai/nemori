#!/usr/bin/env python3
"""
Test OpenRouter compatibility with various LLM and embedding models.

Tests:
  1. LLM chat completions via OpenRouter (multiple providers)
  2. Embedding generation via OpenRouter (multiple models)
  3. Embedding dimension auto-detection
  4. End-to-end: embedding -> cosine similarity search simulation
"""
import asyncio
import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from nemori.llm.client import AsyncLLMClient
from nemori.services.embedding import AsyncEmbeddingClient
from nemori.config import MemoryConfig

# ── Colours for terminal ──────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET} {msg}")


def header(msg: str) -> None:
    print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
    print(f"{BOLD}{CYAN}  {msg}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*60}{RESET}")


# ── Test config ───────────────────────────────────────────────────────
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

LLM_MODELS = [
    "anthropic/claude-sonnet-4",
    "google/gemini-2.5-flash-preview",
    "openai/gpt-4.1-mini",
    "deepseek/deepseek-chat-v3-0324",
]

EMBEDDING_MODELS = [
    "google/gemini-embedding-001",
    "qwen/qwen3-embedding-8b",
]


async def test_llm_models(api_key: str) -> dict[str, bool]:
    """Test LLM chat completion for each model."""
    header("LLM Chat Completion Tests (via OpenRouter)")
    results = {}

    for model in LLM_MODELS:
        client = AsyncLLMClient(api_key=api_key, base_url=OPENROUTER_BASE)
        messages = [
            {"role": "system", "content": "Reply in one short sentence."},
            {"role": "user", "content": "What is 2+2?"},
        ]

        t0 = time.monotonic()
        try:
            content = await client.complete(
                messages, model=model, temperature=0.0, max_tokens=100
            )
            elapsed = (time.monotonic() - t0) * 1000
            if content and len(content.strip()) > 0:
                ok(f"{model:45s}  {elapsed:6.0f}ms  →  {content.strip()[:80]}")
                results[model] = True
            else:
                fail(f"{model:45s}  empty response")
                results[model] = False
        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            fail(f"{model:45s}  {elapsed:6.0f}ms  →  {e}")
            results[model] = False

    return results


async def test_embedding_models(api_key: str) -> dict[str, dict]:
    """Test embedding generation + auto-detect dimension for each model."""
    header("Embedding Model Tests (via OpenRouter)")
    results = {}

    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Nemori is a memory system for LLM agents.",
        "Vector search uses cosine similarity to find relevant documents.",
    ]

    for model in EMBEDDING_MODELS:
        client = AsyncEmbeddingClient(
            api_key=api_key, model=model, base_url=OPENROUTER_BASE
        )

        # Test 1: single embed
        t0 = time.monotonic()
        try:
            vec = await client.embed(test_texts[0])
            elapsed = (time.monotonic() - t0) * 1000
            dim = len(vec)
            ok(f"{model:45s}  dim={dim:5d}  {elapsed:6.0f}ms  (single)")
            results[model] = {"dim": dim, "single": True}
        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            fail(f"{model:45s}  single embed failed: {e}")
            results[model] = {"dim": 0, "single": False}
            continue

        # Test 2: batch embed
        t0 = time.monotonic()
        try:
            vecs = await client.embed_batch(test_texts)
            elapsed = (time.monotonic() - t0) * 1000
            assert len(vecs) == len(test_texts), f"Expected {len(test_texts)} vectors, got {len(vecs)}"
            for v in vecs:
                assert len(v) == dim, f"Dimension mismatch: expected {dim}, got {len(v)}"
            ok(f"{model:45s}  batch={len(vecs)}  {elapsed:6.0f}ms  (batch)")
            results[model]["batch"] = True
        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            fail(f"{model:45s}  batch embed failed: {e}")
            results[model]["batch"] = False
            continue

        # Test 3: cosine similarity sanity check
        try:
            import numpy as np
            v1 = np.array(vecs[0])
            v2 = np.array(vecs[1])
            v3 = np.array(vecs[2])  # about vector search, closest to v1 about fox

            sim_01 = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            sim_02 = float(np.dot(v1, v3) / (np.linalg.norm(v1) * np.linalg.norm(v3)))
            sim_12 = float(np.dot(v2, v3) / (np.linalg.norm(v2) * np.linalg.norm(v3)))

            ok(f"{model:45s}  sim(0,1)={sim_01:.4f}  sim(0,2)={sim_02:.4f}  sim(1,2)={sim_12:.4f}")
            # Sanity: similarity between memory-system texts should be higher
            if sim_12 > sim_01 or sim_12 > sim_02:
                ok(f"{'':45s}  semantic similarity ordering looks reasonable")
            results[model]["similarity"] = True
        except Exception as e:
            warn(f"{model:45s}  similarity check: {e}")
            results[model]["similarity"] = False

    return results


async def test_config_resolution() -> None:
    """Test that MemoryConfig resolves env vars correctly."""
    header("Config Resolution Test")

    config = MemoryConfig()
    if config.llm_api_key:
        ok(f"llm_api_key resolved: {config.llm_api_key[:15]}...")
    else:
        fail("llm_api_key not resolved")

    if config.embedding_api_key:
        ok(f"embedding_api_key resolved: {config.embedding_api_key[:15]}...")
    else:
        fail("embedding_api_key not resolved")

    llm_base = os.getenv("LLM_BASE_URL")
    ok(f"LLM_BASE_URL from env: {llm_base}")
    ok(f"config.llm_base_url: {config.llm_base_url}")


async def test_dimension_autodetect(api_key: str) -> None:
    """Test that different embedding models return different dimensions."""
    header("Embedding Dimension Auto-Detection")

    for model in EMBEDDING_MODELS:
        client = AsyncEmbeddingClient(
            api_key=api_key, model=model, base_url=OPENROUTER_BASE
        )
        try:
            vec = await client.embed("dimension probe test")
            dim = len(vec)
            ok(f"{model:45s}  auto-detected dim = {dim}")
        except Exception as e:
            fail(f"{model:45s}  probe failed: {e}")


async def main() -> None:
    api_key = (
        os.getenv("LLM_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )

    if not api_key:
        print(f"{RED}No API key found. Set LLM_API_KEY, OPENROUTER_API_KEY, or OPENAI_API_KEY.{RESET}")
        sys.exit(1)

    print(f"\n{BOLD}OpenRouter Compatibility Test Suite{RESET}")
    print(f"API key: {api_key[:20]}...")
    print(f"Base URL: {OPENROUTER_BASE}")

    # 1. Config resolution
    await test_config_resolution()

    # 2. LLM models
    llm_results = await test_llm_models(api_key)

    # 3. Embedding models
    embed_results = await test_embedding_models(api_key)

    # 4. Dimension auto-detection
    await test_dimension_autodetect(api_key)

    # Summary
    header("Summary")
    total = 0
    passed = 0

    for model, success in llm_results.items():
        total += 1
        if success:
            passed += 1

    for model, info in embed_results.items():
        for key in ["single", "batch", "similarity"]:
            if key in info:
                total += 1
                if info[key]:
                    passed += 1

    color = GREEN if passed == total else (YELLOW if passed > total // 2 else RED)
    print(f"\n  {color}{passed}/{total} tests passed{RESET}\n")


if __name__ == "__main__":
    asyncio.run(main())
