from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import asyncpg
import httpx
import psutil
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from dashboard.env_store import ENV_FIELDS, EnvStore, normalize_openai_base_url
from dashboard.nemori_swe import (
    SWE_LITE_MAX_TASKS,
    NemoriSWEManager,
    service_ports,
)
from dashboard.run_manager import ROOT, RUNTIME_DIR, RunManager
from dashboard.swebench_data import load_dataset_rows
from evaluation.swebench.contracts import sanitize_agent_instance


STATIC_DIR = Path(__file__).resolve().parent / "static"
env_store = EnvStore(ROOT / ".env")
manager = RunManager(env_store)
SWE_DATASET_PATH = ROOT / "evaluation" / "dataset" / "swebench_lite_test.json"
nemori_swe = NemoriSWEManager(manager, dataset_path=SWE_DATASET_PATH)


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    await nemori_swe.shutdown()
    await manager.shutdown()


app = FastAPI(
    title="Nemori Experiment Lab",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url=None,
    lifespan=lifespan,
)


@app.middleware("http")
async def no_store_api(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    return response


class RunCreate(BaseModel):
    benchmark: str
    name: str | None = None
    notes: str = Field(default="", max_length=4000)
    start_index: int = Field(default=0, ge=0)
    limit: int = Field(default=1, ge=0)
    auto_advance: bool = True
    llm_model: str | None = None
    embedding_model: str | None = None
    judge_model: str | None = None
    search_method: str = "vector"
    top_k_episodes: int = Field(default=10, ge=1, le=100)
    top_k_semantic: int = Field(default=20, ge=1, le=100)
    include_original_messages_top_k: int = Field(default=2, ge=0, le=20)
    batch_size: int = Field(default=1, ge=1, le=100)
    ingest_concurrency: int = Field(default=1, ge=1, le=50)
    search_concurrency: int = Field(default=10, ge=1, le=100)
    judge_concurrency: int = Field(default=10, ge=1, le=100)
    checkpoint_every: int = Field(default=5, ge=1, le=100)
    paper_profile: bool = True
    strict_quality: bool = True
    run_mode: str = "nemori"
    create_direct_pair: bool = False
    observation_window: int = Field(default=20, ge=1, le=100)
    llm_timeout: float = Field(default=120, ge=10, le=600)
    max_process_vms_mb: int | None = Field(default=None, ge=1024, le=131072)


class RunNotesUpdate(BaseModel):
    notes: str = Field(default="", max_length=4000)


class EnvUpdate(BaseModel):
    values: dict[str, str] = Field(default_factory=dict)
    clear: list[str] = Field(default_factory=list)


class ConnectionTest(BaseModel):
    targets: list[str] = Field(default_factory=lambda: ["infrastructure"])
    llm_model: str | None = None
    agent_model: str | None = None
    judge_model: str | None = None
    embedding_model: str | None = None
    values: dict[str, str] = Field(default_factory=dict)
    clear: list[str] = Field(default_factory=list)


class SystemStart(BaseModel):
    services: list[str] = Field(
        default_factory=lambda: ["docker", "postgres", "qdrant", "ollama"]
    )


class NemoriSWEStart(BaseModel):
    task_count: int = Field(default=3, ge=1, le=SWE_LITE_MAX_TASKS)
    arm: str = Field(default="nemori", pattern="^(api|nemori)$")


def not_found_or_conflict(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail=str(exc))
    return HTTPException(status_code=409, detail=str(exc))


def local_test_http_client(base_url: str | None) -> httpx.AsyncClient | None:
    if not base_url:
        return None
    host = (urlparse(base_url).hostname or "").lower()
    if host in {"localhost", "127.0.0.1", "::1"}:
        return httpx.AsyncClient(trust_env=False, timeout=20)
    return None


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "service": "nemori-experiment-lab", "time": time.time()}


@app.get("/api/bootstrap")
async def bootstrap() -> dict[str, Any]:
    datasets, system = await asyncio.gather(manager.dataset_info(), system_status())
    return {
        "runs": manager.list_runs(),
        "nemori_swe": nemori_swe.get(),
        "nemori_swe_history": nemori_swe.history(),
        "datasets": datasets,
        "environment": env_store.public_state(),
        "defaults": env_store.defaults_for_run(),
        "system": system,
    }


@app.get("/api/datasets")
async def datasets() -> list[dict[str, Any]]:
    return await manager.dataset_info()


@app.get("/api/nemori-swe")
async def get_nemori_swe() -> dict[str, Any]:
    return nemori_swe.get()


@app.post("/api/nemori-swe/prepare", status_code=202)
async def prepare_nemori_swe(body: NemoriSWEStart | None = None) -> dict[str, Any]:
    try:
        request = body or NemoriSWEStart()
        return await nemori_swe.prepare(task_count=request.task_count, arm=request.arm)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/nemori-swe/memory", status_code=202)
async def build_nemori_swe_memory() -> dict[str, Any]:
    try:
        return await nemori_swe.build_memory()
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/nemori-swe/agent", status_code=202)
async def run_nemori_swe_agent() -> dict[str, Any]:
    try:
        return await nemori_swe.run_agent()
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/nemori-swe/harness", status_code=202)
async def run_nemori_swe_harness() -> dict[str, Any]:
    try:
        return await nemori_swe.run_harness()
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/nemori-swe/run-all", status_code=202)
async def run_all_nemori_swe(body: NemoriSWEStart | None = None) -> dict[str, Any]:
    try:
        request = body or NemoriSWEStart()
        return await nemori_swe.run_all(task_count=request.task_count, arm=request.arm)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/nemori-swe/result")
async def nemori_swe_result() -> dict[str, Any]:
    try:
        return nemori_swe.result()
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/nemori-swe/history")
async def nemori_swe_history(
    limit: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    return {"items": nemori_swe.history(limit)}


@app.get("/api/nemori-swe/history/{run_id}")
async def nemori_swe_history_detail(run_id: str) -> dict[str, Any]:
    try:
        return nemori_swe.get_run(run_id)
    except (KeyError, ValueError, OSError, json.JSONDecodeError) as exc:
        raise not_found_or_conflict(exc) from exc


@app.delete("/api/nemori-swe/history/{run_id}")
async def delete_nemori_swe_history(run_id: str) -> dict[str, Any]:
    try:
        nemori_swe.delete_run(run_id)
        return {"ok": True, "id": run_id}
    except (KeyError, ValueError, OSError) as exc:
        raise not_found_or_conflict(exc) from exc


@app.post("/api/nemori-swe/stop", status_code=202)
async def stop_nemori_swe() -> dict[str, Any]:
    return await nemori_swe.stop()


@app.get("/api/swebench/instances")
async def swebench_instances(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    """List only the strict Agent-visible projection of SWE-bench test rows."""

    try:
        rows = await asyncio.to_thread(load_dataset_rows, SWE_DATASET_PATH)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    items = [sanitize_agent_instance(item) for item in rows[offset : offset + limit]]
    return {
        "dataset": "princeton-nlp/SWE-bench_Lite",
        "split": "test",
        "total": len(rows),
        "offset": offset,
        "limit": limit,
        "items": items,
    }


@app.get("/api/swebench/instances/{instance_id}")
async def swebench_instance(instance_id: str) -> dict[str, Any]:
    """Return one safe test task without evaluator-only fields."""

    try:
        rows = await asyncio.to_thread(load_dataset_rows, SWE_DATASET_PATH)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    item = next((row for row in rows if row.get("instance_id") == instance_id), None)
    if item is None:
        raise HTTPException(status_code=404, detail="SWE-bench instance not found")
    return sanitize_agent_instance(item)


@app.get("/api/runs")
async def runs() -> list[dict[str, Any]]:
    return manager.list_runs()


@app.post("/api/runs", status_code=201)
async def create_run(body: RunCreate) -> dict[str, Any]:
    try:
        spec = body.model_dump()
        if spec.get("benchmark") == "swebench":
            raise ValueError("SWE-bench 只能通过 Nemori SWE 单一流水线运行")
        create_direct_pair = bool(spec.pop("create_direct_pair", False))
        if create_direct_pair:
            if (
                spec.get("benchmark") != "locomo"
                or spec.get("run_mode", "nemori") != "nemori"
            ):
                raise ValueError("只能为 LoCoMo 的 Nemori 运行同时创建裸上下文基线")
        run = await manager.create_run(spec)
        if create_direct_pair:
            pair_spec = {
                **spec,
                "run_mode": "direct",
                "strict_quality": False,
                "name": f"{run['name']} · 裸跑对比",
            }
            paired = await manager.create_run(pair_spec)
            manager.link_runs(run["id"], paired["id"], relation="direct_baseline")
            run = manager.get_run(run["id"])
            run["paired_run"] = manager.get_run(paired["id"])
        return run
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    try:
        return manager.get_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


async def preflight_run(run: dict[str, Any]) -> None:
    if run.get("benchmark") == "swebench":
        status = await system_status()
        registry = await asyncio.to_thread(
            _run_local_command,
            ["docker", "pull", "hello-world:latest"],
            60,
        )
        registry_check = {
            "ok": bool(registry.get("ok")),
            "version": "Docker Hub reachable" if registry.get("ok") else None,
            "latency_ms": registry.get("latency_ms"),
            **(
                {"error": registry.get("error") or "无法连接 Docker Hub"}
                if not registry.get("ok")
                else {}
            ),
        }
        sample_count = int((run.get("selection") or {}).get("sample_count", 1))
        free_gb = float((status.get("disk") or {}).get("free_gb", 0))
        resources = {
            "ok": free_gb >= 10,
            "free_gb": free_gb,
            "recommended_free_gb": 120,
            "warning": (
                "官方建议为批量运行预留约 120 GB；当前只建议运行 1 个实例"
                if free_gb < 120
                else None
            ),
        }
        checks = {
            "docker": status.get("docker", {"ok": False}),
            "registry": registry_check,
            "wsl_runner": status.get("swebench", {"ok": False}),
            "resources": resources,
        }
        failed = {
            name: result for name, result in checks.items() if not result.get("ok")
        }
        run["preflight"] = {
            "checked_at": time.time(),
            "ok": not failed,
            "checks": checks,
            "warnings": [
                message
                for message in (
                    resources.get("warning"),
                    (
                        "所选实例较多；请确认 Docker 磁盘和运行时间"
                        if sample_count > 1
                        else None
                    ),
                )
                if message
            ],
        }
        manager._save(run)
        if failed:
            summary = "；".join(
                f"{name}: {result.get('error') or '未就绪'}"
                for name, result in failed.items()
            )
            raise HTTPException(
                status_code=409, detail=f"SWE-bench 启动预检未通过：{summary}"
            )
        return

    settings = run["settings"]
    targets = ["llm"]
    if settings.get("run_mode") != "direct":
        targets.append("embedding")
    if settings.get("judge_model") != settings.get("llm_model"):
        targets.append("judge")
    checks = await connection_test(
        ConnectionTest(
            targets=targets,
            llm_model=settings.get("llm_model"),
            judge_model=settings.get("judge_model"),
            embedding_model=settings.get("embedding_model"),
        )
    )
    failed = {name: result for name, result in checks.items() if not result.get("ok")}
    run["preflight"] = {
        "checked_at": time.time(),
        "ok": not failed,
        "checks": checks,
    }
    manager._save(run)
    if failed:
        labels = {"llm": "生成模型", "judge": "Judge", "embedding": "向量模型"}
        summary = "；".join(
            f"{labels.get(name, name)} {result.get('model') or '—'} @ "
            f"{result.get('base_url') or '—'}：{result.get('error') or '连接失败'}"
            for name, result in failed.items()
        )
        raise HTTPException(status_code=409, detail=f"启动前连接检测未通过：{summary}")


@app.post("/api/runs/{run_id}/start")
async def start_run(run_id: str) -> dict[str, Any]:
    try:
        run = manager.get_run(run_id)
        if run["status"] not in {"ready", "awaiting"}:
            return await manager.start(run_id)
        await preflight_run(run)
        return await manager.start(run_id)
    except HTTPException:
        raise
    except (KeyError, ValueError) as exc:
        raise not_found_or_conflict(exc) from exc


@app.post("/api/runs/{run_id}/retry")
async def retry_run(run_id: str) -> dict[str, Any]:
    try:
        run = manager.get_run(run_id)
        if run.get("status") not in {"failed", "stopped", "interrupted"}:
            return await manager.retry_failed(run_id)
        await preflight_run(run)
        return await manager.retry_failed(run_id)
    except HTTPException:
        raise
    except (KeyError, ValueError) as exc:
        raise not_found_or_conflict(exc) from exc


@app.post("/api/runs/{run_id}/pause")
async def pause_run(run_id: str) -> dict[str, Any]:
    try:
        return await manager.pause(run_id)
    except (KeyError, ValueError, psutil.Error) as exc:
        raise not_found_or_conflict(exc) from exc


@app.post("/api/runs/{run_id}/resume")
async def resume_run(run_id: str) -> dict[str, Any]:
    try:
        return await manager.resume(run_id)
    except (KeyError, ValueError, psutil.Error) as exc:
        raise not_found_or_conflict(exc) from exc


@app.post("/api/runs/{run_id}/stop")
async def stop_run(run_id: str) -> dict[str, Any]:
    try:
        return await manager.stop(run_id)
    except (KeyError, ValueError, psutil.Error) as exc:
        raise not_found_or_conflict(exc) from exc


@app.post("/api/runs/{run_id}/skip")
async def skip_phase(run_id: str) -> dict[str, Any]:
    try:
        return await manager.skip_next(run_id)
    except (KeyError, ValueError) as exc:
        raise not_found_or_conflict(exc) from exc


@app.post("/api/runs/{run_id}/clone", status_code=201)
async def clone_run(run_id: str) -> dict[str, Any]:
    try:
        return await manager.clone_run(run_id)
    except (KeyError, ValueError, OSError) as exc:
        raise not_found_or_conflict(exc) from exc


@app.patch("/api/runs/{run_id}/notes")
async def update_run_notes(run_id: str, body: RunNotesUpdate) -> dict[str, Any]:
    try:
        return manager.update_notes(run_id, body.notes)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/api/runs/{run_id}")
async def delete_run(run_id: str) -> dict[str, Any]:
    try:
        manager.delete_run(run_id)
        return {"ok": True, "id": run_id}
    except (KeyError, ValueError, OSError, psutil.Error) as exc:
        raise not_found_or_conflict(exc) from exc


@app.get("/api/runs/{run_id}/logs")
async def logs(
    run_id: str, tail: int = Query(default=300, ge=1, le=1000)
) -> dict[str, Any]:
    try:
        return {"items": manager.logs(run_id, tail)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/runs/{run_id}/results")
async def results(
    run_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    try:
        return manager.results(run_id, limit=limit, offset=offset)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/runs/{run_id}/compare/{against_id}")
async def compare_runs(run_id: str, against_id: str) -> dict[str, Any]:
    try:
        return manager.compare(run_id, against_id)
    except (KeyError, ValueError, OSError, json.JSONDecodeError) as exc:
        raise not_found_or_conflict(exc) from exc


@app.get("/api/runs/{run_id}/workflow")
async def workflow(run_id: str) -> dict[str, Any]:
    try:
        return manager.workflow(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/runs/{run_id}/events")
async def events(run_id: str):
    try:
        manager.get_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    async def stream():
        async for event in manager.subscribe(run_id):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@app.get("/api/runs/{run_id}/artifacts/{name}")
async def artifact(run_id: str, name: str):
    try:
        path = manager.artifact_path(run_id, name)
        return FileResponse(path, filename=path.name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail="Artifact is not available yet"
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/environment")
async def get_environment() -> dict[str, Any]:
    return env_store.public_state()


@app.post("/api/environment")
async def save_environment(body: EnvUpdate) -> dict[str, Any]:
    try:
        return env_store.update(body.values, body.clear)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/environment/test")
async def test_environment(body: ConnectionTest) -> dict[str, Any]:
    return await connection_test(body)


@app.get("/api/system")
async def system_status() -> dict[str, Any]:
    env = env_store.subprocess_env()
    ports = service_ports(env)
    dsn = (
        env.get("DATABASE_URL")
        or env.get("DSN")
        or f"postgresql://nemori:nemori@localhost:{ports['postgres']}/nemori"
    )
    qdrant_host = env.get("QDRANT_URL", "localhost")
    qdrant_port = env.get("QDRANT_PORT", str(ports["qdrant_http"]))
    try:
        ollama_port = int(env.get("NEMORI_OLLAMA_PORT") or 21434)
    except (TypeError, ValueError):
        ollama_port = 21434

    async def postgres_check() -> dict[str, Any]:
        start = time.perf_counter()
        try:
            connection = await asyncio.wait_for(asyncpg.connect(dsn), timeout=3)
            version = await connection.fetchval(
                "select current_setting('server_version')"
            )
            await connection.close()
            return {
                "ok": True,
                "version": version,
                "latency_ms": round((time.perf_counter() - start) * 1000),
            }
        except Exception as exc:
            return {"ok": False, "error": type(exc).__name__}

    async def qdrant_check() -> dict[str, Any]:
        start = time.perf_counter()
        try:
            qdrant_base = str(qdrant_host).rstrip("/")
            if not qdrant_base.startswith(("http://", "https://")):
                qdrant_base = f"http://{qdrant_base}:{qdrant_port}"
            async with httpx.AsyncClient(trust_env=False, timeout=3) as client:
                response = await client.get(f"{qdrant_base}/collections")
                response.raise_for_status()
                collections = response.json().get("result", {}).get("collections", [])
            return {
                "ok": True,
                "collections": len(collections),
                "latency_ms": round((time.perf_counter() - start) * 1000),
            }
        except Exception as exc:
            return {"ok": False, "error": type(exc).__name__}

    async def ollama_check() -> dict[str, Any]:
        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(trust_env=False, timeout=4) as client:
                version_response, models_response = await asyncio.gather(
                    client.get(f"http://127.0.0.1:{ollama_port}/api/version"),
                    client.get(f"http://127.0.0.1:{ollama_port}/api/tags"),
                )
                version_response.raise_for_status()
                models_response.raise_for_status()
            models = [
                item.get("name") for item in models_response.json().get("models", [])
            ]
            return {
                "ok": True,
                "version": version_response.json().get("version"),
                "models": models,
                "qwen_embedding": next(
                    (
                        name
                        for preferred in ("qwen3-embedding:4b", "qwen3-embedding:0.6b")
                        for name in models
                        if str(name) == preferred
                    ),
                    None,
                ),
                "latency_ms": round((time.perf_counter() - start) * 1000),
            }
        except Exception as exc:
            return {"ok": False, "error": type(exc).__name__, "models": []}

    async def docker_check() -> dict[str, Any]:
        result = await asyncio.to_thread(
            _run_local_command,
            [
                "docker",
                "info",
                "--format",
                "{{.ServerVersion}}|{{.NCPU}}|{{.MemTotal}}",
            ],
            8,
        )
        if not result.get("ok"):
            return {"ok": False, "error": result.get("error") or "Docker daemon 未启动"}
        line = str(result.get("output") or "").strip().splitlines()[-1]
        parts = line.split("|")
        memory_bytes = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
        return {
            "ok": True,
            "version": parts[0] if parts else "Docker",
            "cpus": int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None,
            "memory_gb": round(memory_bytes / (1024**3), 1) if memory_bytes else None,
            "latency_ms": result.get("latency_ms"),
        }

    async def swebench_check() -> dict[str, Any]:
        if os.name == "nt":
            distro = os.environ.get("NEMORI_SWEBENCH_WSL_DISTRO", "Ubuntu")
            command = [
                "wsl.exe",
                "-d",
                distro,
                "--",
                "bash",
                "-lc",
                'if test -x "$HOME/.local/bin/uv"; then '
                '"$HOME/.local/bin/uv" --version; '
                'else command -v uv >/dev/null 2>&1 && uv --version; fi',
            ]
            runtime = distro
            missing_error = f"WSL {distro} 中未安装 uv"
        else:
            uv_path = shutil.which("uv")
            user_uv = Path.home() / ".local" / "bin" / "uv"
            if not uv_path and user_uv.is_file():
                uv_path = str(user_uv)
            command = [uv_path or "uv", "--version"]
            runtime = "Linux native"
            missing_error = "Linux PATH 或 ~/.local/bin 中未安装 uv"
        result = await asyncio.to_thread(_run_local_command, command, 12)
        output = str(result.get("output") or "").replace("\x00", "").strip()
        version = next(
            (
                line.strip()
                for line in reversed(output.splitlines())
                if line.strip().startswith("uv ")
            ),
            None,
        )
        return {
            "ok": bool(result.get("ok") and version),
            "version": f"{runtime} / {version}" if version else runtime,
            "runtime": "wsl" if os.name == "nt" else "linux",
            **({"distro": runtime} if os.name == "nt" else {}),
            "latency_ms": result.get("latency_ms"),
            **(
                {"error": result.get("error") or missing_error}
                if not (result.get("ok") and version)
                else {}
            ),
        }

    postgres, qdrant, ollama, docker, swebench = await asyncio.gather(
        postgres_check(),
        qdrant_check(),
        ollama_check(),
        docker_check(),
        swebench_check(),
    )
    disk = psutil.disk_usage(str(ROOT))
    environment = env_store.public_state()["fields"]
    return {
        "postgres": postgres,
        "qdrant": qdrant,
        "ollama": ollama,
        "docker": docker,
        "swebench": swebench,
        "environment_ready": bool(
            environment["LLM_API_KEY"]["configured"]
            or environment["OPENAI_API_KEY"]["configured"]
            or environment["AGENT_API_KEY"]["configured"]
        )
        and bool(
            environment["EMBEDDING_API_KEY"]["configured"]
            or environment["OPENAI_API_KEY"]["configured"]
        ),
        "disk": {
            "free_gb": round(disk.free / (1024**3), 1),
            "percent": disk.percent,
        },
        "runtime_dir": str(RUNTIME_DIR),
    }


def _hidden_startup_kwargs() -> dict[str, Any]:
    if os.name != "nt":
        return {}
    return {
        "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)
        | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    }


def _run_local_command(command: list[str], timeout: float = 20) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            **_hidden_startup_kwargs(),
        )
        output = "\n".join(
            part.strip()
            for part in (completed.stdout, completed.stderr)
            if part.strip()
        )
        return {
            "ok": completed.returncode == 0,
            "command": command,
            "returncode": completed.returncode,
            "latency_ms": round((time.perf_counter() - start) * 1000),
            "output": output[-2000:],
        }
    except FileNotFoundError:
        return {"ok": False, "command": command, "error": "命令不存在"}
    except subprocess.TimeoutExpired as exc:
        output = "\n".join(
            part.strip()
            for part in (exc.stdout or "", exc.stderr or "")
            if isinstance(part, str) and part.strip()
        )
        return {
            "ok": False,
            "command": command,
            "error": f"超过 {timeout:g}s 未完成",
            "output": output[-2000:],
        }


def _popen_detached(command: list[str]) -> dict[str, Any]:
    try:
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            **_hidden_startup_kwargs(),
        )
        return {"ok": True, "command": command, "pid": process.pid}
    except FileNotFoundError:
        return {"ok": False, "command": command, "error": "命令不存在"}
    except OSError as exc:
        return {"ok": False, "command": command, "error": str(exc)}


def _docker_desktop_path() -> str | None:
    candidates = [
        os.environ.get("DOCKER_DESKTOP_EXE"),
        r"C:\Program Files\Docker\Docker\Docker Desktop.exe",
    ]
    return next((path for path in candidates if path and Path(path).exists()), None)


async def _wait_for_docker(seconds: float = 45) -> dict[str, Any]:
    deadline = time.monotonic() + seconds
    last: dict[str, Any] = {"ok": False, "error": "Docker daemon is not ready"}
    while time.monotonic() < deadline:
        last = await asyncio.to_thread(_run_local_command, ["docker", "info"], 8)
        if last.get("ok"):
            return last
        await asyncio.sleep(2)
    return last


async def start_local_environment(body: SystemStart) -> dict[str, Any]:
    allowed = {"docker", "postgres", "qdrant", "ollama"}
    requested = set(body.services or allowed)
    unknown = requested - allowed
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported services: {', '.join(sorted(unknown))}",
        )

    steps: list[dict[str, Any]] = []
    if requested & {"docker", "postgres", "qdrant", "ollama"}:
        docker_info = await asyncio.to_thread(_run_local_command, ["docker", "info"], 8)
        if not docker_info.get("ok"):
            docker_desktop = _docker_desktop_path()
            if docker_desktop:
                steps.append(
                    {
                        "service": "docker",
                        "action": "start Docker Desktop",
                        **_popen_detached([docker_desktop]),
                    }
                )
                docker_info = await _wait_for_docker()
            else:
                steps.append(
                    {
                        "service": "docker",
                        "action": "find Docker Desktop" if os.name == "nt" else "docker info",
                        "ok": False,
                        "error": (
                            "未找到 Docker Desktop，可设置 DOCKER_DESKTOP_EXE"
                            if os.name == "nt"
                            else "Docker daemon 不可用，请启动系统 Docker 服务"
                        ),
                    }
                )
        else:
            steps.append({"service": "docker", "action": "docker info", **docker_info})

        if docker_info.get("ok") and requested & {"postgres", "qdrant"}:
            services = [name for name in ("postgres", "qdrant") if name in requested]
            compose = await asyncio.to_thread(
                _run_local_command,
                ["docker", "compose", "up", "-d", *services],
                90,
            )
            if not compose.get("ok") and shutil.which("docker-compose"):
                compose = await asyncio.to_thread(
                    _run_local_command,
                    ["docker-compose", "up", "-d", *services],
                    90,
                )
            steps.append(
                {
                    "service": "docker-compose",
                    "action": f"up {' '.join(services)}",
                    **compose,
                }
            )

    if "ollama" in requested:
        ollama_status = (await system_status())["ollama"]
        if ollama_status.get("ok"):
            steps.append({"service": "ollama", "action": "already running", "ok": True})
        else:
            ollama = shutil.which("ollama")
            if ollama:
                steps.append(
                    {
                        "service": "ollama",
                        "action": "ollama serve",
                        **_popen_detached([ollama, "serve"]),
                    }
                )
                await asyncio.sleep(2)
            else:
                if os.name != "nt":
                    compose = await asyncio.to_thread(
                        _run_local_command,
                        ["docker", "compose", "up", "-d", "ollama"],
                        120,
                    )
                    steps.append(
                        {
                            "service": "ollama",
                            "action": "docker compose up ollama",
                            **compose,
                        }
                    )
                    if compose.get("ok"):
                        await asyncio.sleep(3)
                else:
                    steps.append(
                        {
                            "service": "ollama",
                            "action": "find ollama",
                            "ok": False,
                            "error": "未找到 ollama 命令，请确认 Ollama 已安装并在 PATH 中",
                        }
                    )

    status = await system_status()
    qwen = status.get("ollama", {}).get("qwen_embedding")
    if status.get("ollama", {}).get("ok") and not qwen:
        steps.append(
            {
                "service": "ollama",
                "action": "check qwen3-embedding",
                "ok": False,
                "error": "Ollama 已启动，但未发现 qwen3-embedding:4b；需要时手动运行 ollama pull qwen3-embedding:4b",
            }
        )
    return {"ok": True, "steps": steps, "status": status}


@app.post("/api/system/start")
async def start_system(body: SystemStart) -> dict[str, Any]:
    return await start_local_environment(body)


async def _probe_chat_connection(
    *,
    key: str | None,
    base_url: str | None,
    model: str | None,
    source: str,
    supplied_secrets: list[str],
) -> dict[str, Any]:
    normalized_base = normalize_openai_base_url(base_url or "")
    details = {
        "source": source,
        "base_url": normalized_base or None,
        "model": model or None,
    }
    if not key:
        return {**details, "ok": False, "error": f"{source} 未配置 API Key"}
    if not normalized_base:
        return {**details, "ok": False, "error": f"{source} 未配置 Base URL"}
    if not model:
        return {**details, "ok": False, "error": f"{source} 未配置模型 ID"}

    start = time.perf_counter()
    local_http = local_test_http_client(normalized_base)
    try:
        client = AsyncOpenAI(
            api_key=key,
            base_url=normalized_base,
            timeout=20,
            **({"http_client": local_http} if local_http else {}),
        )
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Return exactly the two letters OK. Do not explain.",
                }
            ],
            temperature=0,
            max_tokens=128,
        )
        choice = response.choices[0]
        content = choice.message.content or ""
        model_extra = getattr(choice.message, "model_extra", None) or {}
        reasoning = str(model_extra.get("reasoning_content") or "")
        return {
            **details,
            "ok": bool(content.strip()),
            "latency_ms": round((time.perf_counter() - start) * 1000),
            "response": content[:40],
            "finish_reason": getattr(choice, "finish_reason", None),
            **(
                {
                    "error": f"HTTP 2xx，但模型正文为空（reasoning={len(reasoning)} 字符）"
                }
                if not content.strip()
                else {}
            ),
        }
    except Exception as exc:
        return {
            **details,
            "ok": False,
            "latency_ms": round((time.perf_counter() - start) * 1000),
            "error": safe_error(exc, supplied_secrets),
        }
    finally:
        if local_http:
            await local_http.aclose()


async def _probe_embedding_connection(
    *,
    key: str | None,
    base_url: str | None,
    model: str | None,
    supplied_secrets: list[str],
) -> dict[str, Any]:
    normalized_base = normalize_openai_base_url(base_url or "")
    details = {
        "source": "EMBEDDING_*",
        "base_url": normalized_base or None,
        "model": model or None,
    }
    if not key:
        return {**details, "ok": False, "error": "EMBEDDING_* 未配置 API Key"}
    if not normalized_base:
        return {**details, "ok": False, "error": "EMBEDDING_* 未配置 Base URL"}
    if not model:
        return {**details, "ok": False, "error": "EMBEDDING_* 未配置模型 ID"}

    start = time.perf_counter()
    local_http = local_test_http_client(normalized_base)
    try:
        client = AsyncOpenAI(
            api_key=key,
            base_url=normalized_base,
            timeout=20,
            **({"http_client": local_http} if local_http else {}),
        )
        response = await client.embeddings.create(
            model=model,
            input="Nemori connection test",
        )
        vector = response.data[0].embedding
        return {
            **details,
            "ok": bool(vector),
            "dimension": len(vector),
            "latency_ms": round((time.perf_counter() - start) * 1000),
        }
    except Exception as exc:
        return {
            **details,
            "ok": False,
            "latency_ms": round((time.perf_counter() - start) * 1000),
            "error": safe_error(exc, supplied_secrets),
        }
    finally:
        if local_http:
            await local_http.aclose()


async def connection_test(body: ConnectionTest) -> dict[str, Any]:
    allowed_targets = {"infrastructure", "llm", "agent", "judge", "embedding"}
    unknown_targets = set(body.targets) - allowed_targets
    if unknown_targets:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported test targets: {', '.join(sorted(unknown_targets))}",
        )

    env = env_store.subprocess_env()
    unknown = (set(body.values) | set(body.clear)) - set(ENV_FIELDS)
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported environment keys: {', '.join(sorted(unknown))}",
        )
    for key in body.clear:
        env.pop(key, None)
    for key, value in body.values.items():
        if value and "\n" not in value and "\r" not in value:
            rendered = value.strip()
            if key.endswith("_BASE_URL"):
                rendered = normalize_openai_base_url(rendered)
            env[key] = rendered
    if not env.get("LLM_API_KEY") and env.get("AGENT_API_KEY"):
        env["LLM_API_KEY"] = env["AGENT_API_KEY"]
    if not env.get("LLM_BASE_URL") and env.get("AGENT_BASE_URL"):
        env["LLM_BASE_URL"] = env["AGENT_BASE_URL"]
    if not env.get("NEMORI_LLM_MODEL") and env.get("AGENT_MODEL"):
        env["NEMORI_LLM_MODEL"] = env["AGENT_MODEL"]

    defaults = env_store.defaults_for_run()
    supplied_secrets = [
        value
        for key, value in body.values.items()
        if ENV_FIELDS.get(key, {}).get("secret") and len(value) >= 6
    ]
    results: dict[str, Any] = {}
    if "infrastructure" in body.targets:
        status = await system_status()
        results["postgres"] = status["postgres"]
        results["qdrant"] = status["qdrant"]
        results["ollama"] = status["ollama"]
        results["docker"] = status["docker"]
        results["swebench"] = status["swebench"]

    probe_names: list[str] = []
    probe_tasks: list[Any] = []
    if "llm" in body.targets:
        probe_names.append("llm")
        probe_tasks.append(
            _probe_chat_connection(
                key=env.get("LLM_API_KEY") or env.get("OPENAI_API_KEY"),
                base_url=env.get("LLM_BASE_URL") or env.get("OPENAI_BASE_URL"),
                model=body.llm_model
                or env.get("NEMORI_LLM_MODEL")
                or defaults["llm_model"],
                source="LLM_*",
                supplied_secrets=supplied_secrets,
            )
        )
    if "agent" in body.targets:
        probe_names.append("agent")
        probe_tasks.append(
            _probe_chat_connection(
                key=env.get("AGENT_API_KEY")
                or env.get("LLM_API_KEY")
                or env.get("OPENAI_API_KEY"),
                base_url=env.get("AGENT_BASE_URL")
                or env.get("LLM_BASE_URL")
                or env.get("OPENAI_BASE_URL"),
                model=body.agent_model or env.get("AGENT_MODEL"),
                source="AGENT_*",
                supplied_secrets=supplied_secrets,
            )
        )
    if "judge" in body.targets:
        probe_names.append("judge")
        probe_tasks.append(
            _probe_chat_connection(
                key=env.get("LLM_API_KEY") or env.get("OPENAI_API_KEY"),
                base_url=env.get("LLM_BASE_URL") or env.get("OPENAI_BASE_URL"),
                model=body.judge_model
                or env.get("NEMORI_JUDGE_MODEL")
                or env.get("LLM_JUDGE_MODEL")
                or defaults["judge_model"],
                source="LLM_* / Judge",
                supplied_secrets=supplied_secrets,
            )
        )
    if "embedding" in body.targets:
        probe_names.append("embedding")
        probe_tasks.append(
            _probe_embedding_connection(
                key=env.get("EMBEDDING_API_KEY") or env.get("OPENAI_API_KEY"),
                base_url=env.get("EMBEDDING_BASE_URL") or env.get("OPENAI_BASE_URL"),
                model=body.embedding_model
                or env.get("NEMORI_EMBEDDING_MODEL")
                or defaults["embedding_model"],
                supplied_secrets=supplied_secrets,
            )
        )
    if probe_tasks:
        probed = await asyncio.gather(*probe_tasks)
        results.update(dict(zip(probe_names, probed, strict=True)))
    return results


def safe_error(exc: Exception, extra_secrets: list[str] | None = None) -> str:
    text = str(exc)
    for value in [*env_store.secret_values(), *(extra_secrets or [])]:
        text = text.replace(value, "***")
    return f"{type(exc).__name__}: {text[:300]}"


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
