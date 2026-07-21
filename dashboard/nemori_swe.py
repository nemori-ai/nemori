"""Single-path Nemori repository-memory experiment for SWE-bench."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import httpx

from dashboard.run_manager import ROOT, RunManager, atomic_json
from evaluation.swebench.contracts import (
    EvaluatorInstance,
    assert_no_forbidden_fields,
    sanitize_agent_instance,
    sha256_file,
    stable_sha256,
    validate_prediction_coverage,
)
from evaluation.swebench.data import ensure_test_dataset
from evaluation.swebench.repository_memory import (
    NativeMemoryResult,
    build_native_memory,
    materialize_official_repository,
    scan_repository,
)
from nemori.config import MemoryConfig


EXPERIMENT_NAME = "Nemori SWE repository-memory experiment"
EXPERIMENT_NAMES = {
    "api": "API + SWE baseline experiment",
    "nemori": EXPERIMENT_NAME,
}
EXPERIMENT_PURPOSES = {
    "api": "Measure the API Coding Agent on SWE-bench without Nemori memory.",
    "nemori": "Test Nemori native repository-code memory on SWE-bench tasks.",
}
ARMS = frozenset(EXPERIMENT_NAMES)
RUN_ROOT = ROOT / "runs" / "nemori-swe"
ACTIVE = frozenset(
    {"preparing", "building_memory", "running_agent", "running_harness", "running_all"}
)
STAGES = ("prepare", "memory", "agent", "harness", "result")
DEFAULT_POSTGRES_PORT = 15432
DEFAULT_QDRANT_HTTP_PORT = 16333
DEFAULT_QDRANT_GRPC_PORT = 16334


def _port(env: Mapping[str, str], key: str, default: int) -> int:
    try:
        value = int(env.get(key) or default)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer TCP port") from exc
    if not 1 <= value <= 65535:
        raise ValueError(f"{key} must be between 1 and 65535")
    return value


def service_ports(env: Mapping[str, str]) -> dict[str, int]:
    """Return this dashboard's isolated host-side storage ports."""

    return {
        "postgres": _port(env, "NEMORI_POSTGRES_PORT", DEFAULT_POSTGRES_PORT),
        "qdrant_http": _port(
            env, "NEMORI_QDRANT_HTTP_PORT", DEFAULT_QDRANT_HTTP_PORT
        ),
        "qdrant_grpc": _port(
            env, "NEMORI_QDRANT_GRPC_PORT", DEFAULT_QDRANT_GRPC_PORT
        ),
    }


def _published_port(rendered: str) -> int | None:
    """Extract the host port printed by ``docker compose port``."""

    for line in rendered.splitlines():
        tail = line.strip().rsplit(":", 1)[-1]
        if tail.isdigit():
            return int(tail)
    return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _step() -> dict[str, Any]:
    return {
        "status": "pending",
        "started_at": None,
        "ended_at": None,
        "error": None,
    }


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class NemoriSWEManager:
    """Persist and execute API-baseline and Nemori-memory SWE pipelines."""

    def __init__(
        self,
        run_manager: RunManager,
        *,
        dataset_path: Path,
        root: Path = RUN_ROOT,
    ) -> None:
        self.run_manager = run_manager
        self.env_store = run_manager.env_store
        self.dataset_path = dataset_path.resolve()
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.state: dict[str, Any] | None = None
        self.task: asyncio.Task[Any] | None = None
        self.process: asyncio.subprocess.Process | None = None
        self._load_latest()

    def _load_latest(self) -> None:
        for path in sorted(self.root.glob("*/experiment.json"), reverse=True):
            try:
                state = _load_json(path)
                arm = str(
                    state.get("arm") or (state.get("config") or {}).get("arm") or "nemori"
                )
                state["arm"] = arm if arm in ARMS else "nemori"
                state.setdefault("config", {})["arm"] = state["arm"]
                if state.get("status") in ACTIVE:
                    state["status"] = "interrupted"
                    state["current_activity"] = (
                        "Dashboard restarted; resume the pending stage"
                    )
                    atomic_json(path, state)
                self.state = state
                return
            except (OSError, ValueError, json.JSONDecodeError):
                continue

    def _dir(self) -> Path:
        if not self.state:
            raise ValueError("Prepare the Nemori SWE experiment first")
        path = (self.root / str(self.state["id"])).resolve()
        if self.root not in path.parents:
            raise ValueError("Invalid Nemori SWE run id")
        return path

    def _save(self) -> None:
        assert self.state is not None
        self.state["updated_at"] = _now()
        atomic_json(self._dir() / "experiment.json", self.state)

    def _log(self, stage: str, message: str, level: str = "info") -> None:
        assert self.state is not None
        event = {
            "timestamp": _now(),
            "stage": stage,
            "level": level,
            "message": str(message),
        }
        self.state.setdefault("logs", []).append(event)
        self.state["logs"] = self.state["logs"][-500:]
        self.state["current_activity"] = str(message)
        log_path = self._dir() / "run.log"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{event['timestamp']} [{level.upper()}] [{stage}] {message}\n"
            )
        self._save()

    def get(self) -> dict[str, Any]:
        if not self.state:
            return {
                "name": EXPERIMENT_NAME,
                "arm": "nemori",
                "status": "not_started",
                "current_activity": "Click Prepare or Run All",
                "config": {"task_count": 3, "arm": "nemori"},
                "steps": {stage: _step() for stage in STAGES},
                "official_swebench_score": False,
                "purpose": "Test Nemori native repository-code memory on SWE-bench tasks.",
                "logs": [],
            }
        return json.loads(json.dumps(self.state, ensure_ascii=False))

    def history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return persisted SWE experiments without exposing protected inputs."""

        items: list[dict[str, Any]] = []
        for path in sorted(self.root.glob("*/experiment.json"), reverse=True):
            try:
                state = _load_json(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            result = state.get("result") or {}
            items.append(
                {
                    "id": state.get("id"),
                    "name": state.get("name"),
                    "arm": state.get("arm")
                    or (state.get("config") or {}).get("arm")
                    or "nemori",
                    "status": state.get("status"),
                    "created_at": state.get("created_at"),
                    "updated_at": state.get("updated_at"),
                    "sample_count": (state.get("selection") or {}).get(
                        "sample_count"
                    ),
                    "result": {
                        key: result.get(key)
                        for key in (
                            "total",
                            "resolved",
                            "failed",
                            "success_rate",
                            "agent_tokens",
                            "agent_time_seconds",
                            "memory_hits",
                        )
                    }
                    if result
                    else None,
                }
            )
            if len(items) >= max(1, min(int(limit), 100)):
                break
        return items

    def _ensure_idle(self) -> None:
        if self.task and not self.task.done():
            raise ValueError("A Nemori SWE stage is already running")

    def _new_state(self, task_count: int, arm: str = "nemori") -> None:
        arm = str(arm or "nemori").strip().lower()
        if arm not in ARMS:
            raise ValueError("arm must be api or nemori")
        run_id = (
            datetime.now().strftime("%Y%m%d-%H%M%S")
            + f"-{arm}-swe-"
            + uuid.uuid4().hex[:6]
        )
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        self.state = {
            "id": run_id,
            "name": EXPERIMENT_NAMES[arm],
            "arm": arm,
            "status": "ready",
            "created_at": _now(),
            "updated_at": _now(),
            "current_activity": "Ready",
            "config": {
                "task_count": max(1, min(20, int(task_count))),
                "arm": arm,
                "max_code_files": 32,
                "max_code_chunks": 60,
                "top_k_episodes": 8,
                "top_k_semantic": 12,
            },
            "steps": {stage: _step() for stage in STAGES},
            "selection": None,
            "memory": None,
            "agent": None,
            "harness": None,
            "result": None,
            "artifacts": {"run_dir": str(run_dir), "log": str(run_dir / "run.log")},
            "logs": [],
            "error": None,
            "official_swebench_score": False,
            "purpose": EXPERIMENT_PURPOSES[arm],
        }
        self._save()

    async def prepare(
        self, *, task_count: int = 3, arm: str = "nemori"
    ) -> dict[str, Any]:
        self._ensure_idle()
        self._new_state(task_count, arm)
        self.task = asyncio.create_task(self._run_one("prepare"))
        return self.get()

    async def build_memory(self) -> dict[str, Any]:
        self._ensure_idle()
        self._require_completed("prepare")
        self.task = asyncio.create_task(self._run_one("memory"))
        return self.get()

    async def run_agent(self) -> dict[str, Any]:
        self._ensure_idle()
        self._require_completed("memory")
        self.task = asyncio.create_task(self._run_one("agent"))
        return self.get()

    async def run_harness(self) -> dict[str, Any]:
        self._ensure_idle()
        self._require_completed("agent")
        self.task = asyncio.create_task(self._run_one("harness", include_result=True))
        return self.get()

    async def run_all(
        self, *, task_count: int = 3, arm: str = "nemori"
    ) -> dict[str, Any]:
        self._ensure_idle()
        self._new_state(task_count, arm)
        self.state["status"] = "running_all"
        self._save()
        self.task = asyncio.create_task(self._run_pipeline())
        return self.get()

    def _require_completed(self, stage: str) -> None:
        if not self.state or self.state["steps"][stage]["status"] != "completed":
            raise ValueError(f"{stage} must complete first")

    async def _run_pipeline(self) -> None:
        for stage in ("prepare", "memory", "agent", "harness", "result"):
            if not await self._execute(stage):
                return
        assert self.state is not None
        self.state["status"] = "completed"
        self.state["current_activity"] = f"{self.state['name']} completed"
        self._save()

    async def _run_one(self, stage: str, *, include_result: bool = False) -> None:
        if not await self._execute(stage):
            return
        if include_result and not await self._execute("result"):
            return
        assert self.state is not None
        self.state["status"] = (
            "completed"
            if self.state["steps"]["result"]["status"] == "completed"
            else "partial"
        )
        self._save()

    async def _execute(self, stage: str) -> bool:
        assert self.state is not None
        step = self.state["steps"][stage]
        step.update(status="running", started_at=_now(), ended_at=None, error=None)
        self.state["status"] = {
            "prepare": "preparing",
            "memory": "building_memory",
            "agent": "running_agent",
            "harness": "running_harness",
            "result": "finalizing",
        }[stage]
        self.state["error"] = None
        self._save()
        try:
            await getattr(self, f"_stage_{stage}")()
            step.update(status="completed", ended_at=_now(), error=None)
            self._log(stage, f"{stage.title()} completed")
            return True
        except asyncio.CancelledError:
            step.update(status="stopped", ended_at=_now(), error="cancelled")
            self.state["status"] = "stopped"
            self._save()
            raise
        except Exception as exc:
            rendered = f"{type(exc).__name__}: {exc}"
            step.update(status="failed", ended_at=_now(), error=rendered)
            self.state["status"] = "failed"
            self.state["error"] = rendered
            self._log(stage, rendered, "error")
            return False

    async def _stage_prepare(self) -> None:
        assert self.state is not None
        self._log("prepare", "Validating or downloading SWE-bench Lite")
        rows = await asyncio.to_thread(
            ensure_test_dataset,
            self.dataset_path,
            log=lambda message: self._log("prepare", message),
        )
        count = int(self.state["config"]["task_count"])
        selected = rows[:count]
        if len(selected) != count:
            raise RuntimeError(f"Only {len(selected)} SWE tasks are available")
        protected = [EvaluatorInstance(item).to_harness_record() for item in selected]
        safe = [sanitize_agent_instance(item) for item in selected]
        protected_path = self._dir() / "protected" / "evaluator-input.json"
        safe_path = self._dir() / "safe-tasks.json"
        atomic_json(protected_path, protected)
        atomic_json(safe_path, safe)
        self.state["selection"] = {
            "dataset": "princeton-nlp/SWE-bench_Lite",
            "dataset_sha256": sha256_file(self.dataset_path),
            "instance_ids": [item["instance_id"] for item in safe],
            "sample_count": len(safe),
        }
        self.state["artifacts"].update(
            safe_tasks=str(safe_path),
            evaluator_input=str(protected_path),
            evaluator_input_sha256=sha256_file(protected_path),
        )
        self._log("prepare", f"Prepared {len(safe)} answer-free SWE tasks")

    def _memory_config(self) -> MemoryConfig:
        assert self.state is not None
        env = self.env_store.subprocess_env()
        ports = service_ports(env)
        llm_key = (
            env.get("LLM_API_KEY")
            or env.get("AGENT_API_KEY")
            or env.get("OPENAI_API_KEY")
        )
        embedding_key = (
            env.get("EMBEDDING_API_KEY")
            or env.get("OPENAI_API_KEY")
            or env.get("AGENT_API_KEY")
        )
        if not llm_key:
            raise ValueError("Native Nemori memory requires an LLM API key")
        if not embedding_key:
            raise ValueError("Native Nemori memory requires an embedding API key")
        llm_model = env.get("NEMORI_LLM_MODEL") or env.get("AGENT_MODEL")
        embedding_model = env.get("NEMORI_EMBEDDING_MODEL") or env.get(
            "EMBEDDING_MODEL"
        )
        if not llm_model or not embedding_model:
            raise ValueError("Nemori LLM and embedding model must be configured")
        return MemoryConfig(
            dsn=env.get("DATABASE_URL")
            or env.get("DSN")
            or (
                "postgresql://nemori:nemori@localhost:"
                f"{ports['postgres']}/nemori"
            ),
            db_pool_min=1,
            db_pool_max=5,
            agent_id=f"nemori_swe_{self.state['id'].replace('-', '_')}",
            llm_model=llm_model,
            llm_api_key=llm_key,
            llm_base_url=env.get("LLM_BASE_URL") or env.get("AGENT_BASE_URL"),
            llm_timeout=float(env.get("NEMORI_SWE_MEMORY_TIMEOUT") or 120),
            embedding_model=embedding_model,
            embedding_api_key=embedding_key,
            embedding_base_url=env.get("EMBEDDING_BASE_URL")
            or env.get("OPENAI_BASE_URL")
            or env.get("AGENT_BASE_URL"),
            qdrant_url=env.get("QDRANT_URL") or "localhost",
            qdrant_port=_port(env, "QDRANT_PORT", ports["qdrant_http"]),
            qdrant_api_key=env.get("QDRANT_API_KEY"),
            qdrant_collection_prefix=f"nemori_swe_{self.state['id'].replace('-', '_')}",
            observation_window=20,
            batch_threshold=20,
            buffer_size_min=20,
            search_top_k_episodes=int(self.state["config"]["top_k_episodes"]),
            search_top_k_semantic=int(self.state["config"]["top_k_semantic"]),
        )

    async def _ensure_services(self) -> None:
        env = self.env_store.subprocess_env()
        ports = service_ports(env)

        async def capture(*command: str, timeout: float = 30) -> tuple[int, str]:
            child = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(ROOT),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            output, _ = await asyncio.wait_for(child.communicate(), timeout=timeout)
            return child.returncode or 0, output.decode("utf-8", errors="replace")

        docker_code, docker_output = await capture("docker", "info", timeout=20)
        if docker_code != 0 and os.name == "nt":
            candidates = [
                Path(os.environ.get("ProgramFiles", "C:/Program Files"))
                / "Docker"
                / "Docker"
                / "Docker Desktop.exe",
                Path(os.environ.get("LOCALAPPDATA", ""))
                / "Programs"
                / "Docker"
                / "Docker"
                / "Docker Desktop.exe",
            ]
            desktop = next((path for path in candidates if path.is_file()), None)
            if desktop:
                self._log("memory", "Docker daemon is offline; starting Docker Desktop")
                subprocess.Popen(
                    [str(desktop)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=(
                        subprocess.DETACHED_PROCESS
                        | subprocess.CREATE_NEW_PROCESS_GROUP
                    ),
                )
                for _ in range(45):
                    await asyncio.sleep(2)
                    docker_code, docker_output = await capture(
                        "docker", "info", timeout=20
                    )
                    if docker_code == 0:
                        break
        if docker_code != 0:
            raise RuntimeError("Docker daemon is unavailable: " + docker_output[-1200:])
        self._log("memory", "Starting PostgreSQL and Qdrant through Docker Compose")
        returncode, rendered = await capture(
            "docker", "compose", "up", "-d", "postgres", "qdrant", timeout=120
        )
        if returncode != 0:
            raise RuntimeError(
                "Docker Compose could not start Nemori services: " + rendered[-1500:]
            )

        expected_mappings = (
            ("postgres", 5432, ports["postgres"]),
            ("qdrant", 6333, ports["qdrant_http"]),
            ("qdrant", 6334, ports["qdrant_grpc"]),
        )
        for service, container_port, expected_host_port in expected_mappings:
            port_code, port_output = await capture(
                "docker",
                "compose",
                "port",
                service,
                str(container_port),
                timeout=15,
            )
            actual_host_port = _published_port(port_output)
            if port_code != 0 or actual_host_port != expected_host_port:
                raise RuntimeError(
                    f"Nemori {service}:{container_port} is not published on the "
                    f"isolated host port {expected_host_port}; another local stack "
                    "may own the configured port"
                )

        self._log(
            "memory",
            "Using isolated storage ports: "
            f"PostgreSQL {ports['postgres']}, Qdrant HTTP {ports['qdrant_http']}, "
            f"Qdrant gRPC {ports['qdrant_grpc']}",
        )
        for attempt in range(60):
            ready = True
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection("127.0.0.1", ports["postgres"]), timeout=1
                )
                writer.close()
                await writer.wait_closed()
                async with httpx.AsyncClient(trust_env=False, timeout=2) as client:
                    response = await client.get(
                        f"http://127.0.0.1:{ports['qdrant_http']}/healthz"
                    )
                    response.raise_for_status()
            except (OSError, asyncio.TimeoutError, httpx.HTTPError):
                ready = False
            if ready:
                self._log("memory", "PostgreSQL and Qdrant are ready")
                return
            if attempt in {9, 29}:
                self._log("memory", "Waiting for Nemori storage services")
            await asyncio.sleep(1)
        raise RuntimeError("PostgreSQL or Qdrant did not become ready in 60 seconds")

    async def _stage_memory(self) -> None:
        assert self.state is not None
        safe_rows = _load_json(Path(self.state["artifacts"]["safe_tasks"]))
        arm = str(self.state.get("arm") or self.state["config"].get("arm") or "nemori")
        if arm == "api":
            input_path = self._dir() / "agent-input.jsonl"
            _jsonl(input_path, safe_rows)
            self.state["memory"] = {
                "enabled": False,
                "source_type": "none",
                "backend": "api-baseline",
                "read_only_during_agent": True,
                "indexed_files": 0,
                "indexed_lines": 0,
                "messages": 0,
                "episodes": 0,
                "semantic_memories": 0,
                "build_time_seconds": 0,
                "tasks": [],
            }
            self.state["artifacts"]["agent_input"] = str(input_path)
            self._log(
                "memory",
                "API baseline selected; skipping Nemori memory and freezing answer-free Agent input",
            )
            return

        await self._ensure_services()
        config = self._memory_config()
        agent_rows: list[dict[str, Any]] = []
        task_metrics: list[dict[str, Any]] = []
        total_episodes = 0
        total_semantic = 0
        total_messages = 0
        total_files = 0
        total_lines = 0
        snapshot_tasks: list[dict[str, Any]] = []
        started = time.perf_counter()
        for index, item in enumerate(safe_rows, start=1):
            instance_id = str(item["instance_id"])
            task_dir = self._dir() / "repository-memory" / instance_id
            checkout = task_dir / "checkout"
            self._log(
                "memory",
                f"[{index}/{len(safe_rows)}] Exporting {item['repo']} at base_commit",
            )
            await asyncio.to_thread(
                materialize_official_repository,
                item,
                output_dir=checkout,
                log=lambda message: self._log("memory", message),
            )
            scan = await asyncio.to_thread(
                scan_repository,
                checkout,
                repo=str(item["repo"]),
                base_commit=str(item["base_commit"]),
                issue=str(item["problem_statement"]),
                max_files=int(self.state["config"]["max_code_files"]),
                max_chunks=int(self.state["config"]["max_code_chunks"]),
            )
            manifest_path = task_dir / "repository_manifest.json"
            atomic_json(manifest_path, scan.public_manifest())
            self._log(
                "memory",
                f"[{index}/{len(safe_rows)}] Ingesting {len(scan.chunks)} code observations through NemoriMemory",
            )
            memory: NativeMemoryResult = await build_native_memory(
                scan,
                issue=str(item["problem_statement"]),
                config=config,
                user_id=f"{self.state['id']}::{instance_id}",
            )
            if not memory.context.strip():
                raise RuntimeError(
                    f"Native Nemori returned no code memory for {instance_id}"
                )
            retrieval_path = task_dir / "memory_retrieval.json"
            build_path = task_dir / "memory_build.json"
            atomic_json(retrieval_path, memory.retrieval)
            atomic_json(
                build_path,
                {
                    "source_type": "workspace_code",
                    "native_api": "nemori.api.facade.NemoriMemory",
                    "search_backend": "nemori.search.UnifiedSearch",
                    "user_id": memory.user_id,
                    "repo": scan.repo,
                    "base_commit": scan.base_commit,
                    "manifest_sha256": scan.manifest_sha256,
                    "message_count": memory.message_count,
                    "episode_count": memory.episode_count,
                    "semantic_count": memory.semantic_count,
                    "stats": memory.stats,
                    "retrieval_sha256": sha256_file(retrieval_path),
                },
            )
            hits = [
                {
                    "id": value.get("id"),
                    "memory_type": kind,
                    "content": value.get("content", ""),
                    "source": "workspace_code",
                }
                for kind, values in (
                    ("episode", memory.retrieval.get("episodes") or []),
                    ("semantic", memory.retrieval.get("semantic_memories") or []),
                )
                for value in values
                if isinstance(value, Mapping)
            ]
            input_row = {
                **item,
                "memory_context": {
                    "text": memory.context,
                    "retrievals": hits,
                    "backend": "nemori.search.UnifiedSearch",
                    "search_method": "hybrid",
                    "latency_ms": memory.retrieval_latency_ms,
                    "memory_calls": 1,
                    "embedding_tokens": 0,
                },
            }
            assert_no_forbidden_fields(input_row)
            agent_rows.append(input_row)
            total_episodes += memory.episode_count
            total_semantic += memory.semantic_count
            total_messages += memory.message_count
            total_files += scan.indexed_file_count
            total_lines += scan.indexed_line_count
            task_metrics.append(
                {
                    "instance_id": instance_id,
                    "repo": scan.repo,
                    "base_commit": scan.base_commit,
                    "indexed_files": scan.indexed_file_count,
                    "indexed_lines": scan.indexed_line_count,
                    "code_chunks": len(scan.chunks),
                    "episodes": memory.episode_count,
                    "semantic_memories": memory.semantic_count,
                    "memory_hits": len(hits),
                    "manifest_sha256": scan.manifest_sha256,
                }
            )
            snapshot_tasks.append(
                {
                    "instance_id": instance_id,
                    "repo": scan.repo,
                    "base_commit": scan.base_commit,
                    "manifest_sha256": scan.manifest_sha256,
                    "user_id": memory.user_id,
                    "episode_count": memory.episode_count,
                    "semantic_count": memory.semantic_count,
                    "retrieval": memory.retrieval,
                    "context": memory.context,
                }
            )
        input_path = self._dir() / "agent-input.jsonl"
        _jsonl(input_path, agent_rows)
        frozen_at = _now()
        snapshot_body = {
            "schema_version": 1,
            "dataset": self.state["selection"]["dataset"],
            "dataset_sha256": self.state["selection"]["dataset_sha256"],
            "frozen_at": frozen_at,
            "source_type": "workspace_code",
            "source_tasks": self.state["selection"]["instance_ids"],
            "memory_count": total_episodes + total_semantic,
            "episode_count": total_episodes,
            "semantic_count": total_semantic,
            "read_only_during_agent": True,
            "tasks": snapshot_tasks,
        }
        snapshot = {
            **snapshot_body,
            "content_sha256": stable_sha256(snapshot_body),
        }
        snapshot_path = self._dir() / "memory_snapshot.json"
        atomic_json(snapshot_path, snapshot)
        self.state["memory"] = {
            "enabled": True,
            "source_type": "workspace_code",
            "native_api": "nemori.api.facade.NemoriMemory",
            "backend": "nemori.search.UnifiedSearch",
            "read_only_during_agent": True,
            "indexed_files": total_files,
            "indexed_lines": total_lines,
            "messages": total_messages,
            "episodes": total_episodes,
            "semantic_memories": total_semantic,
            "build_time_seconds": round(time.perf_counter() - started, 3),
            "tasks": task_metrics,
        }
        self.state["artifacts"]["agent_input"] = str(input_path)
        self.state["artifacts"].update(
            memory_snapshot=str(snapshot_path),
            memory_snapshot_sha256=sha256_file(snapshot_path),
        )
        self._log(
            "memory",
            f"Built native Nemori code memory: {total_episodes} Episodes, {total_semantic} Semantic Memories",
        )

    def _agent_model(self) -> str:
        env = self.env_store.subprocess_env()
        return str(env.get("AGENT_MODEL") or env.get("NEMORI_LLM_MODEL") or "").strip()

    def _agent_env(self) -> dict[str, str]:
        env = self.env_store.subprocess_env()
        api_key = (
            env.get("AGENT_API_KEY")
            or env.get("LLM_API_KEY")
            or env.get("OPENAI_API_KEY")
        )
        if not api_key:
            raise ValueError("Coding Agent requires an API key")
        env["OPENAI_API_KEY"] = api_key
        base = env.get("AGENT_BASE_URL") or env.get("LLM_BASE_URL")
        if base:
            env["OPENAI_API_BASE"] = base
        if os.name == "nt":
            inherited = [
                value for value in env.get("WSLENV", "").split(":") if value
            ]
            forwarded = [
                "OPENAI_API_KEY/u",
                "OPENAI_API_BASE/u",
                "AGENT_API_KEY/u",
                "AGENT_BASE_URL/u",
                "AGENT_PROVIDER/u",
                "MSWEA_COST_TRACKING/u",
            ]
            env["WSLENV"] = ":".join([*inherited, *forwarded])
        env["MSWEA_COST_TRACKING"] = "ignore_errors"
        return env

    @staticmethod
    def _native_uv() -> str:
        candidate = shutil.which("uv")
        if candidate:
            return candidate
        user_candidate = Path.home() / ".local" / "bin" / "uv"
        if user_candidate.is_file():
            return str(user_candidate)
        raise RuntimeError(
            "uv is required for the Linux SWE runtime; install it in PATH or ~/.local/bin"
        )

    async def _ensure_wsl_uv(self, distro: str) -> None:
        self._log("agent", f"Checking the {distro} Agent runtime")
        command = (
            'if test -x "$HOME/.local/bin/uv"; then exit 0; fi; '
            "if command -v uv >/dev/null 2>&1; then exit 0; fi; "
            "python3 -m pip install --user uv"
        )
        process = await asyncio.create_subprocess_exec(
            "wsl.exe",
            "-d",
            distro,
            "--",
            "bash",
            "-lc",
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        output, _ = await asyncio.wait_for(process.communicate(), timeout=300)
        if process.returncode != 0:
            raise RuntimeError(
                f"Could not prepare uv inside WSL {distro}: "
                + output.decode("utf-8", errors="replace")[-1500:]
            )

    async def _stage_agent(self) -> None:
        assert self.state is not None
        model = self._agent_model()
        if not model:
            raise ValueError("Configure AGENT_MODEL before running SWE")
        env = self.env_store.subprocess_env()
        run_dir = self._dir()
        input_path = Path(self.state["artifacts"]["agent_input"])
        output_dir = run_dir / "agent-output"
        arm = str(self.state.get("arm") or self.state["config"].get("arm") or "nemori")
        distro = str(os.environ.get("NEMORI_SWEBENCH_WSL_DISTRO") or "Ubuntu")
        if os.name == "nt":
            await self._ensure_wsl_uv(distro)
        args = [
            "run",
            "--python",
            "3.12",
            "--with",
            "mini-swe-agent==2.4.5",
            "--no-project",
            "python",
            "-u",
            "../../../evaluation/swebench/agent_run.py",
            "--input",
            input_path.relative_to(run_dir).as_posix(),
            "--output",
            output_dir.relative_to(run_dir).as_posix(),
            "--arm",
            arm,
            "--model",
            model,
            "--temperature",
            str(env.get("AGENT_TEMPERATURE") or 0),
            "--token-budget",
            str(env.get("NEMORI_SWE_TOKEN_BUDGET") or 120_000),
            "--time-budget",
            str(env.get("NEMORI_SWE_TIME_BUDGET") or 1_800),
            "--max-turns",
            str(env.get("NEMORI_SWE_MAX_TURNS") or 80),
            "--max-tool-calls",
            str(env.get("NEMORI_SWE_MAX_TOOL_CALLS") or 200),
        ]
        base_url = env.get("AGENT_BASE_URL") or env.get("LLM_BASE_URL")
        provider = env.get("AGENT_PROVIDER")
        if base_url:
            args.extend(["--api-base", str(base_url)])
        if provider:
            args.extend(["--provider", str(provider)])
        if os.name == "nt":
            command = [
                "wsl.exe",
                "-d",
                distro,
                "--cd",
                str(run_dir),
                "--",
                "bash",
                "-lc",
                'if test -x "$HOME/.local/bin/uv"; then '
                'exec "$HOME/.local/bin/uv" ' + shlex.join(args) + '; '
                'else exec uv ' + shlex.join(args) + '; fi',
            ]
            runtime_label = f"WSL {distro}"
        else:
            command = [self._native_uv(), *args]
            runtime_label = "native Linux"
        self._log(
            "agent",
            f"Running {arm.upper()} + SWE with mini-swe-agent 2.4.5 on {runtime_label} using {model}",
        )
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        self.process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=self._agent_env(),
            creationflags=creationflags,
        )
        log_path = run_dir / "agent.log"
        try:
            assert self.process.stdout is not None
            with log_path.open("a", encoding="utf-8") as handle:
                while line := await self.process.stdout.readline():
                    rendered = line.decode("utf-8", errors="replace")
                    handle.write(rendered)
                    handle.flush()
                    self.state["current_activity"] = rendered.strip()[-500:]
                    self._save()
            returncode = await self.process.wait()
        finally:
            self.process = None
        if returncode != 0:
            raise RuntimeError(f"Coding Agent exited with code {returncode}")
        predictions_path = output_dir / "predictions.json"
        trials_path = output_dir / "trials.json"
        if not predictions_path.is_file() or not trials_path.is_file():
            raise RuntimeError("Coding Agent did not produce predictions and trials")
        predictions = _load_json(predictions_path)
        validate_prediction_coverage(
            self.state["selection"]["instance_ids"], predictions
        )
        trials = _load_json(trials_path)
        if not isinstance(trials, list):
            raise RuntimeError("Agent trials artifact is invalid")
        self.state["agent"] = {
            "backend": "mini-swe-agent",
            "version": "2.4.5",
            "arm": arm,
            "runtime": runtime_label,
            "model": model,
            "prediction_count": len(predictions),
            "predictions_sha256": sha256_file(predictions_path),
        }
        self.state["artifacts"].update(
            predictions=str(predictions_path),
            trials=str(trials_path),
            agent_log=str(log_path),
        )

    async def _stage_harness(self) -> None:
        assert self.state is not None
        env = self.env_store.subprocess_env()
        predictions = Path(self.state["artifacts"]["predictions"])
        evaluator = Path(self.state["artifacts"]["evaluator_input"])
        count = int(self.state["selection"]["sample_count"])
        child = await self.run_manager.create_run(
            {
                "benchmark": "swebench",
                "name": f"{self.state['name']} Harness",
                "notes": f"Child Harness for {self.state['id']}",
                "start_index": 0,
                "limit": count,
                "auto_advance": True,
                "swe_predictions_path": str(predictions),
                "swe_predictions_sha256": sha256_file(predictions),
                "swe_evaluator_input_path": str(evaluator),
                "swe_evaluator_input_sha256": sha256_file(evaluator),
                "swe_expected_instance_ids": self.state["selection"]["instance_ids"],
                "swe_max_workers": max(
                    1, min(24, int(env.get("NEMORI_SWE_MAX_WORKERS") or 1))
                ),
                "swe_timeout": max(
                    60,
                    min(
                        14_400,
                        int(env.get("NEMORI_SWE_HARNESS_TIMEOUT") or 1_800),
                    ),
                ),
                "swe_cache_level": str(
                    env.get("NEMORI_SWE_CACHE_LEVEL") or "env"
                ).lower(),
                "swe_clean": str(env.get("NEMORI_SWE_CLEAN") or "true").lower()
                in {"1", "true", "yes", "on"},
            }
        )
        self.state["harness"] = {"run_id": child["id"], "status": "running"}
        self._save()
        self._log("harness", "Running official swebench==4.1.0 Docker Harness")
        await self.run_manager.start(child["id"])
        await self.run_manager.tasks[child["id"]]
        finished = self.run_manager.get_run(child["id"])
        if finished.get("status") != "completed":
            raise RuntimeError(f"Harness ended in state {finished.get('status')}")
        self.state["harness"] = {
            "run_id": child["id"],
            "status": "completed",
            "summary": finished.get("summary"),
        }
        self.state["artifacts"]["harness_results"] = finished["paths"]["results"]

    async def _stage_result(self) -> None:
        assert self.state is not None
        harness_rows = _load_json(Path(self.state["artifacts"]["harness_results"]))
        trials = _load_json(Path(self.state["artifacts"]["trials"]))
        by_trial = {
            str(item.get("instance_id")): item
            for item in trials
            if isinstance(item, Mapping)
        }
        instances: list[dict[str, Any]] = []
        resolved = 0
        tokens = 0
        duration = 0.0
        memory_hits = 0
        for row in harness_rows:
            instance_id = str(row.get("question_id") or row.get("instance_id") or "")
            trial = by_trial.get(instance_id, {})
            solved = bool(row.get("resolved"))
            resolved += int(solved)
            one_tokens = int(
                (trial.get("tokens") or {}).get("total", 0)
                if isinstance(trial.get("tokens"), Mapping)
                else trial.get("tokens") or trial.get("total_tokens") or 0
            )
            one_time = float(trial.get("duration_seconds") or 0)
            hits = len(trial.get("retrievals") or [])
            tokens += one_tokens
            duration += one_time
            memory_hits += hits
            instances.append(
                {
                    "instance_id": instance_id,
                    "resolved": solved,
                    "tokens": one_tokens,
                    "time_seconds": one_time,
                    "memory_hits": hits,
                    "exit_status": trial.get("exit_status"),
                    "model_patch": trial.get("model_patch", ""),
                }
            )
        total = len(harness_rows)
        result = {
            "name": self.state["name"],
            "arm": self.state.get("arm") or "nemori",
            "official_swebench_score": False,
            "purpose": self.state["purpose"],
            "total": total,
            "resolved": resolved,
            "failed": total - resolved,
            "success_rate": resolved / total if total else 0,
            "agent_tokens": tokens,
            "agent_time_seconds": round(duration, 3),
            "memory_hits": memory_hits,
            "memory": self.state["memory"],
            "instances": instances,
        }
        result_path = self._dir() / "result.json"
        atomic_json(result_path, result)
        self.state["result"] = result
        self.state["artifacts"]["result"] = str(result_path)

    def result(self) -> dict[str, Any]:
        if not self.state or not self.state.get("result"):
            raise ValueError("Nemori SWE result is not available yet")
        return json.loads(json.dumps(self.state["result"], ensure_ascii=False))

    async def stop(self) -> dict[str, Any]:
        if self.process and self.process.returncode is None:
            self.process.terminate()
        if self.task and not self.task.done():
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        if self.state:
            self.state["status"] = "stopped"
            self.state["current_activity"] = "Stopped"
            self._save()
        return self.get()

    async def shutdown(self) -> None:
        if self.process and self.process.returncode is None:
            self.process.terminate()
        if self.task and not self.task.done():
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
