"""Build native Nemori memory from a SWE-bench repository at ``base_commit``.

The repository working tree is allowed Agent input.  Gold changes, evaluator
tests and future Git refs are never read by this module.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tarfile
import tempfile
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from evaluation.swebench.contracts import sanitize_agent_instance, stable_sha256
from nemori.api.facade import NemoriMemory
from nemori.config import MemoryConfig


TEXT_SUFFIXES = frozenset(
    {
        ".c",
        ".cc",
        ".cpp",
        ".cs",
        ".css",
        ".go",
        ".h",
        ".hpp",
        ".html",
        ".ini",
        ".java",
        ".js",
        ".json",
        ".jsx",
        ".kt",
        ".md",
        ".php",
        ".properties",
        ".py",
        ".rb",
        ".rs",
        ".rst",
        ".scala",
        ".sh",
        ".sql",
        ".toml",
        ".ts",
        ".tsx",
        ".txt",
        ".xml",
        ".yaml",
        ".yml",
    }
)
SPECIAL_FILES = frozenset(
    {
        "dockerfile",
        "gemfile",
        "makefile",
        "package-lock.json",
        "package.json",
        "pyproject.toml",
        "requirements.txt",
        "setup.cfg",
        "setup.py",
        "tox.ini",
    }
)
EXCLUDED_PARTS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "coverage",
        "dist",
        "node_modules",
        "target",
        "vendor",
        "venv",
    }
)
SECRET_NAMES = frozenset({".env", ".env.local", "credentials.json", "secrets.json"})
TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
SYMBOL_PATTERN = re.compile(
    r"^\s*(?:async\s+def|def|class|function|interface|type|func|fn|struct|enum)\s+([A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)


@dataclass(frozen=True)
class CodeChunk:
    path: str
    start_line: int
    end_line: int
    sha256: str
    symbols: tuple[str, ...]
    content: str
    relevance: float


@dataclass(frozen=True)
class RepositoryScan:
    repo: str
    base_commit: str
    root: Path
    manifest_sha256: str
    file_count: int
    indexed_file_count: int
    indexed_line_count: int
    excluded_file_count: int
    languages: dict[str, int]
    files: tuple[dict[str, Any], ...]
    chunks: tuple[CodeChunk, ...]

    def public_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "source_type": "workspace_code",
            "repo": self.repo,
            "base_commit": self.base_commit,
            "manifest_sha256": self.manifest_sha256,
            "file_count": self.file_count,
            "indexed_file_count": self.indexed_file_count,
            "indexed_line_count": self.indexed_line_count,
            "excluded_file_count": self.excluded_file_count,
            "languages": self.languages,
            "files": list(self.files),
            "chunks": [
                {key: value for key, value in asdict(chunk).items() if key != "content"}
                for chunk in self.chunks
            ],
        }


@dataclass(frozen=True)
class NativeMemoryResult:
    user_id: str
    episode_count: int
    semantic_count: int
    stats: dict[str, Any]
    retrieval: dict[str, Any]
    context: str
    message_count: int
    retrieval_latency_ms: float


def _language(path: Path) -> str:
    suffix = path.suffix.casefold()
    return {
        ".py": "Python",
        ".js": "JavaScript",
        ".jsx": "JavaScript",
        ".ts": "TypeScript",
        ".tsx": "TypeScript",
        ".rb": "Ruby",
        ".go": "Go",
        ".rs": "Rust",
        ".java": "Java",
        ".php": "PHP",
        ".c": "C",
        ".cc": "C++",
        ".cpp": "C++",
        ".cs": "C#",
    }.get(suffix, suffix.removeprefix(".").upper() or "text")


def _is_candidate(relative: Path, size: int, max_file_bytes: int) -> bool:
    lowered = {part.casefold() for part in relative.parts}
    name = relative.name.casefold()
    return (
        size > 0
        and size <= max_file_bytes
        and not lowered.intersection(EXCLUDED_PARTS)
        and name not in SECRET_NAMES
        and (relative.suffix.casefold() in TEXT_SUFFIXES or name in SPECIAL_FILES)
    )


def _read_text(path: Path) -> str | None:
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in raw[:8192]:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace")


def _issue_tokens(issue: str) -> set[str]:
    ignored = {
        "about",
        "after",
        "before",
        "could",
        "error",
        "issue",
        "should",
        "their",
        "there",
        "these",
        "this",
        "when",
        "where",
        "which",
        "with",
        "would",
    }
    return {
        token.casefold()
        for token in TOKEN_PATTERN.findall(issue)
        if token.casefold() not in ignored
    }


def scan_repository(
    root: Path,
    *,
    repo: str,
    base_commit: str,
    issue: str,
    max_files: int = 40,
    max_chunks: int = 80,
    max_file_bytes: int = 256_000,
    chunk_lines: int = 140,
    overlap_lines: int = 20,
) -> RepositoryScan:
    """Create a bounded, issue-seeded code view without evaluator material."""

    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Repository checkout is missing: {root}")
    tokens = _issue_tokens(issue)
    discovered: list[tuple[Path, Path, int]] = []
    excluded = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        try:
            size = path.stat().st_size
        except OSError:
            excluded += 1
            continue
        if _is_candidate(relative, size, max_file_bytes):
            discovered.append((path, relative, size))
        else:
            excluded += 1

    ranked: list[tuple[float, Path, Path, int, str]] = []
    files: list[dict[str, Any]] = []
    languages: Counter[str] = Counter()
    for path, relative, size in discovered:
        text = _read_text(path)
        if text is None:
            excluded += 1
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        relative_text = relative.as_posix()
        lowered_sample = (relative_text + "\n" + text[:64_000]).casefold()
        matches = sum(min(8, lowered_sample.count(token)) for token in tokens)
        configuration = 5 if relative.name.casefold() in SPECIAL_FILES else 0
        test_bonus = 2 if "test" in relative_text.casefold() else 0
        score = float(matches * 3 + configuration + test_bonus)
        ranked.append((score, path, relative, size, text))
        language = _language(relative)
        languages[language] += 1
        files.append(
            {
                "path": relative_text,
                "sha256": digest,
                "bytes": size,
                "lines": text.count("\n") + 1,
                "language": language,
            }
        )
    ranked.sort(key=lambda item: (-item[0], item[2].as_posix()))
    selected = ranked[: max(1, max_files)]
    chunks: list[CodeChunk] = []
    step = max(1, chunk_lines - overlap_lines)
    indexed_lines = 0
    for score, _path, relative, _size, text in selected:
        lines = text.splitlines()
        indexed_lines += len(lines)
        for start in range(0, max(1, len(lines)), step):
            body_lines = lines[start : start + chunk_lines]
            if not body_lines:
                continue
            content = "\n".join(body_lines)
            chunks.append(
                CodeChunk(
                    path=relative.as_posix(),
                    start_line=start + 1,
                    end_line=start + len(body_lines),
                    sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    symbols=tuple(dict.fromkeys(SYMBOL_PATTERN.findall(content)))[:20],
                    content=content,
                    relevance=score,
                )
            )
            if len(chunks) >= max_chunks:
                break
        if len(chunks) >= max_chunks:
            break
    manifest_identity = {
        "repo": repo,
        "base_commit": base_commit,
        "files": [{"path": item["path"], "sha256": item["sha256"]} for item in files],
    }
    return RepositoryScan(
        repo=repo,
        base_commit=base_commit,
        root=root,
        manifest_sha256=stable_sha256(manifest_identity),
        file_count=len(files),
        indexed_file_count=len({chunk.path for chunk in chunks}),
        indexed_line_count=indexed_lines,
        excluded_file_count=excluded,
        languages=dict(languages.most_common()),
        files=tuple(files),
        chunks=tuple(chunks),
    )


def repository_messages(scan: RepositoryScan) -> list[dict[str, Any]]:
    """Turn code observations into native Nemori input messages."""

    top_level = Counter(item["path"].split("/", 1)[0] for item in scan.files)
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                "Repository environment observation.\n"
                f"Repository: {scan.repo}\nBase commit: {scan.base_commit}\n"
                f"Languages: {json.dumps(scan.languages, ensure_ascii=False)}\n"
                f"Top-level layout: {json.dumps(dict(top_level.most_common()), ensure_ascii=False)}\n"
                "Treat every observation as read-only code context and verify it in the checkout."
            ),
            "metadata": {
                "source_type": "workspace_code",
                "repo": scan.repo,
                "base_commit": scan.base_commit,
                "manifest_sha256": scan.manifest_sha256,
            },
        }
    ]
    for chunk in scan.chunks:
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Repository file observation: {chunk.path}:{chunk.start_line}-{chunk.end_line}\n"
                    f"Symbols: {', '.join(chunk.symbols) or 'none detected'}\n"
                    "```\n"
                    f"{chunk.content}\n"
                    "```"
                ),
                "metadata": {
                    "source_type": "workspace_code",
                    "repo": scan.repo,
                    "base_commit": scan.base_commit,
                    "path": chunk.path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "content_sha256": chunk.sha256,
                },
            }
        )
    return messages


def render_search_context(result: Mapping[str, Any], *, max_chars: int = 60_000) -> str:
    sections: list[str] = []
    for item in result.get("episodes") or []:
        if not isinstance(item, Mapping):
            continue
        sections.append(
            f"[Episode: {item.get('title') or 'repository observation'}]\n"
            f"{item.get('content') or ''}"
        )
        for source in item.get("source_messages") or []:
            if isinstance(source, Mapping) and source.get("content"):
                sections.append(f"[Original code observation]\n{source['content']}")
    for item in result.get("semantic_memories") or []:
        if isinstance(item, Mapping):
            sections.append(
                f"[Semantic: {item.get('memory_type') or 'repository knowledge'}]\n"
                f"{item.get('content') or ''}"
            )
    rendered = "\n\n".join(section.strip() for section in sections if section.strip())
    return rendered[:max_chars]


async def build_native_memory(
    scan: RepositoryScan,
    *,
    issue: str,
    config: MemoryConfig,
    user_id: str,
    memory_factory: Callable[[MemoryConfig], Any] = NemoriMemory,
    batch_size: int = 20,
) -> NativeMemoryResult:
    """Ingest code through ``NemoriMemory`` and retrieve issue-relevant memory.

    There is intentionally no custom or generated fallback.  Failure to use
    the native memory system fails the stage instead of producing a mislabeled
    Nemori result.
    """

    messages = repository_messages(scan)
    async with memory_factory(config) as memory:
        for index in range(0, len(messages), max(1, batch_size)):
            await memory.add_messages(user_id, messages[index : index + batch_size])
        flushed = await memory.flush(user_id)
        stats = await memory.stats(user_id)
        search_started = time.perf_counter()
        retrieval = await memory.search(
            user_id,
            f"Repository: {scan.repo}\nIssue:\n{issue}",
            top_k_episodes=config.search_top_k_episodes,
            top_k_semantic=config.search_top_k_semantic,
            search_method="hybrid",
        )
        retrieval_latency_ms = (time.perf_counter() - search_started) * 1000
    episode_count = int(
        stats.get("episodes") or stats.get("episode_count") or len(flushed)
    )
    semantic_count = int(
        stats.get("semantic_memories")
        or stats.get("semantic_count")
        or stats.get("semantic_memory_count")
        or 0
    )
    return NativeMemoryResult(
        user_id=user_id,
        episode_count=episode_count,
        semantic_count=semantic_count,
        stats=dict(stats),
        retrieval=dict(retrieval),
        context=render_search_context(retrieval),
        message_count=len(messages),
        retrieval_latency_ms=round(retrieval_latency_ms, 3),
    )


def official_image(instance_id: str, *, namespace: str = "swebench") -> str:
    encoded = instance_id.replace("__", "_1776_")
    return f"docker.io/{namespace}/sweb.eval.x86_64.{encoded}:latest"


def _run(
    command: Sequence[str], *, timeout: int = 1800
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )


def materialize_official_repository(
    instance: Mapping[str, Any],
    *,
    output_dir: Path,
    namespace: str = "swebench",
    log: Callable[[str], None] | None = None,
) -> Path:
    """Export only the sanitized ``base_commit`` working tree from Docker."""

    safe = sanitize_agent_instance(instance)
    announce = log or (lambda _message: None)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        return output_dir
    image = official_image(str(safe["instance_id"]), namespace=namespace)
    announce(f"Pulling or reusing official image {image}")
    pull = _run(["docker", "pull", image], timeout=1800)
    if pull.returncode != 0:
        raise RuntimeError(f"Docker image pull failed: {pull.stdout[-2000:]}")
    container = f"nemori-repo-{uuid.uuid4().hex[:12]}"
    commit = str(safe["base_commit"])
    script = (
        "set -eu; cd /testbed; "
        f"target=$(git rev-parse '{commit}^{{commit}}'); "
        'git checkout --detach -q "$target"; git reset --hard -q "$target"; '
        "git clean -ffdx; "
        'git remote | while IFS= read -r name; do git remote remove "$name"; done; '
        "git reflog expire --expire=now --all; git gc --prune=now --quiet; "
        'test "$(git rev-parse HEAD)" = "$target"; '
        "tar --exclude='./.git' --exclude='./node_modules' --exclude='./vendor' "
        "-cf /tmp/nemori-repository.tar ."
    )
    create = _run(
        [
            "docker",
            "create",
            "--name",
            container,
            "--network",
            "none",
            image,
            "bash",
            "-lc",
            script,
        ]
    )
    if create.returncode != 0:
        raise RuntimeError(
            f"Docker repository container failed: {create.stdout[-2000:]}"
        )
    bundle: Path | None = None
    try:
        started = _run(["docker", "start", "-a", container], timeout=1800)
        if started.returncode != 0:
            raise RuntimeError(
                f"Repository checkout sanitization failed: {started.stdout[-2000:]}"
            )
        with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as handle:
            bundle = Path(handle.name)
        copied = _run(
            ["docker", "cp", f"{container}:/tmp/nemori-repository.tar", str(bundle)]
        )
        if copied.returncode != 0:
            raise RuntimeError(f"Repository export failed: {copied.stdout[-2000:]}")
        with tarfile.open(bundle) as archive:
            archive.extractall(output_dir, filter="data")
    finally:
        _run(["docker", "rm", "-f", container], timeout=60)
        if bundle is not None:
            bundle.unlink(missing_ok=True)
    announce(f"Exported sanitized {safe['repo']} at {commit[:12]}")
    return output_dir
