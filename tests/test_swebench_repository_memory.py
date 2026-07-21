from __future__ import annotations

import asyncio
from pathlib import Path

from evaluation.swebench.repository_memory import (
    build_native_memory,
    official_image,
    repository_messages,
    scan_repository,
)
from nemori.config import MemoryConfig


def _config() -> MemoryConfig:
    return MemoryConfig(
        dsn="postgresql://unused",
        db_pool_min=1,
        db_pool_max=1,
        llm_api_key="test",
        embedding_api_key="test",
        observation_window=2,
        batch_threshold=2,
        buffer_size_min=2,
    )


def test_scan_repository_indexes_relevant_code_and_excludes_secrets(
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "src" / "parser.py").write_text(
        "class Parser:\n    def parse_schema(self, value):\n        return value\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "test_parser.py").write_text(
        "def test_schema_parser():\n    assert True\n", encoding="utf-8"
    )
    (tmp_path / "README.md").write_text("Run pytest for parser tests", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=do-not-index", encoding="utf-8")
    (tmp_path / "node_modules" / "library.js").write_text(
        "function parser() {}", encoding="utf-8"
    )

    scan = scan_repository(
        tmp_path,
        repo="owner/repo",
        base_commit="a" * 40,
        issue="Schema parser returns an incorrect value",
        max_files=3,
        max_chunks=10,
    )

    paths = {item["path"] for item in scan.files}
    indexed = {item.path for item in scan.chunks}
    assert "src/parser.py" in indexed
    assert ".env" not in paths
    assert "node_modules/library.js" not in paths
    assert scan.manifest_sha256
    assert any("Parser" in chunk.symbols for chunk in scan.chunks)


def test_repository_messages_keep_code_provenance(tmp_path: Path) -> None:
    (tmp_path / "module.py").write_text(
        "def useful_symbol():\n    return 1\n", encoding="utf-8"
    )
    scan = scan_repository(
        tmp_path,
        repo="owner/repo",
        base_commit="b" * 40,
        issue="useful symbol is wrong",
    )

    messages = repository_messages(scan)

    assert messages[0]["metadata"]["source_type"] == "workspace_code"
    assert messages[1]["metadata"]["path"] == "module.py"
    assert "useful_symbol" in messages[1]["content"]
    assert all("patch" not in message["metadata"] for message in messages)


def test_build_native_memory_uses_facade_without_fallback(tmp_path: Path) -> None:
    (tmp_path / "module.py").write_text(
        "def solve_bug():\n    return True\n", encoding="utf-8"
    )
    scan = scan_repository(
        tmp_path,
        repo="owner/repo",
        base_commit="c" * 40,
        issue="solve bug",
    )
    calls: list[tuple[str, object]] = []

    class FakeMemory:
        def __init__(self, config: MemoryConfig) -> None:
            calls.append(("config", config.agent_id))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def add_messages(self, user_id, messages):
            calls.append(("add", (user_id, len(messages))))

        async def flush(self, user_id):
            calls.append(("flush", user_id))
            return [{"id": "episode-1"}]

        async def stats(self, user_id):
            return {"episode_count": 1, "semantic_memory_count": 1}

        async def search(self, user_id, query, **kwargs):
            calls.append(("search", user_id))
            return {
                "episodes": [
                    {
                        "id": "episode-1",
                        "title": "Repository observation",
                        "content": "The relevant module is module.py.",
                        "source_messages": [],
                    }
                ],
                "semantic_memories": [
                    {
                        "id": "semantic-1",
                        "memory_type": "code_knowledge",
                        "content": "solve_bug is defined in module.py",
                    }
                ],
            }

    result = asyncio.run(
        build_native_memory(
            scan,
            issue="solve bug",
            config=_config(),
            user_id="run::task",
            memory_factory=FakeMemory,
        )
    )

    assert result.episode_count == 1
    assert result.semantic_count == 1
    assert "module.py" in result.context
    assert any(name == "add" for name, _ in calls)
    assert any(name == "search" for name, _ in calls)


def test_official_image_uses_swebench_instance_encoding() -> None:
    assert official_image("django__django-123") == (
        "docker.io/swebench/sweb.eval.x86_64.django_1776_django-123:latest"
    )
