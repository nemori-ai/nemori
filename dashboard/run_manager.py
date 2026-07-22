from __future__ import annotations

import asyncio
import codecs
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Mapping

import psutil

from dashboard.env_store import EnvStore
from dashboard.workflows import workflow_for_run
from evaluation.swebench.contracts import sanitize_agent_instance, sha256_file
from evaluation.swebench.download import validate_bundle as validate_swe_bundle


ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = ROOT / "evaluation"
RUNTIME_DIR = ROOT / ".nemori-dashboard"
RUNS_DIR = ROOT / "runs"

PHASE_DEFINITIONS = {
    "locomo": [
        ("ingest", "构建记忆", "把长对话组织为情景与语义记忆"),
        ("search", "检索与回答", "检索相关记忆并生成答案"),
        ("evaluate", "裁判评分", "计算 BLEU/F1 并调用 LLM Judge"),
        ("aggregate", "汇总结果", "生成分类与总体指标"),
    ],
    "longmemeval": [
        ("ingest", "构建记忆", "逐题写入带时间戳的长历史"),
        ("search", "检索与回答", "检索情景/语义记忆并回答"),
        ("evaluate", "裁判评分", "按问题类型执行 LLM Judge"),
        ("aggregate", "汇总结果", "生成总体与分类型准确率"),
    ],
    "swebench": [
        ("evaluate", "构建并评测", "在 Docker 中应用补丁并运行官方回归测试"),
        ("aggregate", "汇总结果", "生成解决率与逐实例报告"),
    ],
}

ACTIVE_STATUSES = {"running", "paused", "stopping"}
TERMINAL_STATUSES = {"completed", "failed", "stopped"}

QUALITY_MARKERS = (
    ("segmentation", "Segmentation failed"),
    ("episode_generation", "Episode generation failed"),
    ("semantic_generation", "Failed to parse semantic statements"),
    ("semantic_generation", "Semantic generation failed"),
    ("semantic_consolidation", "Semantic consolidation failed"),
    ("episode_integration", "Episode merge check failed"),
)

DEFAULT_PROCESS_VMS_LIMIT_MB = 12_000
SWE_HARNESS_UV_DEPENDENCIES = (
    "libcst==1.7.0",
    "cbor2==5.7.1",
    "datasets==4.0.0",
    "numpy==1.26.4",
    "pandas==2.2.3",
    "pyarrow==20.0.0",
)


def _is_windows_host() -> bool:
    return os.name == "nt"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f"{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        last_error: OSError | None = None
        for attempt in range(8):
            try:
                os.replace(tmp_path, path)
                return
            except OSError as exc:
                last_error = exc
                if attempt < 7:
                    time.sleep(min(0.05 * (2**attempt), 1.0))
        raise OSError(f"Could not atomically replace {path}") from last_error
    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def flatten_records(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        records: list[dict[str, Any]] = []
        for value in data.values():
            if isinstance(value, list):
                records.extend(item for item in value if isinstance(item, dict))
            elif isinstance(value, dict):
                records.append(value)
        return records
    return []


class RunManager:
    def __init__(self, env_store: EnvStore) -> None:
        self.env_store = env_store
        self.runs: dict[str, dict[str, Any]] = {}
        self.processes: dict[str, asyncio.subprocess.Process] = {}
        self.tasks: dict[str, asyncio.Task[Any]] = {}
        self.stop_requested: set[str] = set()
        self.subscribers: dict[str, set[asyncio.Queue[dict[str, Any]]]] = defaultdict(set)
        self.log_buffers: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=1000))
        self.dataset_cache: dict[str, dict[str, Any]] = {}
        self.event_seq = 0
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        self._load_runs()

    def _load_runs(self) -> None:
        for state_path in RUNS_DIR.glob("*/run.json"):
            try:
                run = json.loads(state_path.read_text(encoding="utf-8"))
                if run.get("status") in ACTIVE_STATUSES:
                    run["status"] = "interrupted"
                    run["current_activity"] = "控制台重启，原进程状态不可恢复"
                    for phase in run.get("phases", []):
                        if phase.get("status") in ACTIVE_STATUSES:
                            phase["status"] = "interrupted"
                    atomic_json(state_path, run)
                self.runs[run["id"]] = run
                log_path = state_path.parent / "run.log"
                if log_path.exists():
                    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                    self.log_buffers[run["id"]].extend(lines[-1000:])
                    token_usage_changed = False
                    if not (run.get("token_usage") or {}).get("phases"):
                        for line in lines:
                            token_usage_changed = (
                                self._record_token_usage_line(run, line, "ingest")
                                or token_usage_changed
                            )
                    if token_usage_changed:
                        atomic_json(state_path, run)
            except Exception:
                continue

    def _run_dir(self, run_id: str) -> Path:
        path = (RUNS_DIR / run_id).resolve()
        if RUNS_DIR.resolve() not in path.parents:
            raise ValueError("Invalid run ID")
        return path

    def _state_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "run.json"

    def _save(self, run: dict[str, Any]) -> None:
        run["updated_at"] = utc_now()
        atomic_json(self._state_path(run["id"]), run)

    def _redact(self, text: str) -> str:
        redacted = text
        for value in self.env_store.secret_values():
            redacted = redacted.replace(value, "***REDACTED***")
        redacted = re.sub(r"(?i)(sk-(?:or-)?[a-z0-9_-]{10,})", "***REDACTED***", redacted)
        return redacted

    @staticmethod
    def _record_token_usage_line(
        run: dict[str, Any], line: str, phase_id: str
    ) -> bool:
        """Persist provider-reported token totals printed by evaluation workers.

        Timed-out calls do not return usage metadata and therefore remain visible
        as errors instead of being guessed into the token total.
        """

        scalar_patterns = {
            "requests": r"\bTotal LLM calls:\s*([\d,]+)",
            "total_tokens": r"\bTotal tokens:\s*([\d,]+)",
            "prompt_tokens": r"\bPrompt tokens:\s*([\d,]+)",
            "completion_tokens": r"\bCompletion tokens:\s*([\d,]+)",
            "errors": r"\bTotal errors:\s*([\d,]+)",
        }
        matched_values: dict[str, int] = {}
        for key, pattern in scalar_patterns.items():
            match = re.search(pattern, line)
            if match:
                matched_values[key] = int(match.group(1).replace(",", ""))

        operation_match = re.search(
            r"\b([A-Za-z][\w.-]*)\s+calls=\s*([\d,]+)\s+tokens=\s*([\d,]+)",
            line,
        )
        if not matched_values and operation_match is None:
            return False

        usage = run.setdefault(
            "token_usage",
            {
                "source": "provider_reported_successful_responses",
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "requests": 0,
                "errors": 0,
                "phases": {},
            },
        )
        phases = usage.setdefault("phases", {})
        phase = phases.setdefault(
            phase_id,
            {
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "requests": 0,
                "errors": 0,
                "operations": {},
            },
        )
        phase.update(matched_values)
        if operation_match is not None:
            phase.setdefault("operations", {})[operation_match.group(1)] = {
                "requests": int(operation_match.group(2).replace(",", "")),
                "tokens": int(operation_match.group(3).replace(",", "")),
            }

        for key in (
            "total_tokens",
            "prompt_tokens",
            "completion_tokens",
            "requests",
            "errors",
        ):
            usage[key] = sum(int(item.get(key, 0)) for item in phases.values())
        if not usage["total_tokens"]:
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        usage["updated_at"] = utc_now()
        return True

    async def _emit(self, run_id: str, event_type: str, data: Any) -> None:
        self.event_seq += 1
        event = {
            "seq": self.event_seq,
            "time": utc_now(),
            "type": event_type,
            "data": data,
        }
        for queue in list(self.subscribers.get(run_id, set())):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

    async def subscribe(self, run_id: str) -> AsyncIterator[dict[str, Any]]:
        run = self.get_run(run_id)
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=500)
        self.subscribers[run_id].add(queue)
        try:
            yield {"seq": 0, "time": utc_now(), "type": "state", "data": run}
            while True:
                try:
                    yield await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield {"seq": 0, "time": utc_now(), "type": "ping", "data": {}}
        finally:
            self.subscribers[run_id].discard(queue)

    def list_runs(self) -> list[dict[str, Any]]:
        return sorted(self.runs.values(), key=lambda item: item["created_at"], reverse=True)

    def get_run(self, run_id: str) -> dict[str, Any]:
        try:
            return self.runs[run_id]
        except KeyError as exc:
            raise KeyError(f"Run not found: {run_id}") from exc

    async def dataset_info(self) -> list[dict[str, Any]]:
        definitions = [
            ("locomo", EVALUATION_DIR / "dataset" / "locomo10.json"),
            ("longmemeval", EVALUATION_DIR / "dataset" / "longmemeval_s.json"),
            ("swebench", EVALUATION_DIR / "dataset" / "swebench_lite_test.json"),
        ]
        results = []
        for benchmark, path in definitions:
            manifest_path = path.with_suffix(".manifest.json")
            cache_key = (
                f"{path}:{path.stat().st_mtime_ns if path.exists() else 0}:"
                f"{manifest_path.stat().st_mtime_ns if manifest_path.exists() else 0}"
            )
            if cache_key not in self.dataset_cache:
                info = await asyncio.to_thread(self._inspect_dataset, benchmark, path)
                self.dataset_cache[cache_key] = info
            results.append(self.dataset_cache[cache_key])
        return results

    @staticmethod
    def _inspect_dataset(benchmark: str, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {"id": benchmark, "available": False, "path": str(path)}
        data = json.loads(path.read_text(encoding="utf-8"))
        info: dict[str, Any] = {
            "id": benchmark,
            "available": True,
            "path": str(path),
            "bytes": path.stat().st_size,
            "samples": len(data),
        }
        if benchmark == "locomo":
            categories = Counter(
                str(question.get("category"))
                for item in data
                for question in item.get("qa", [])
            )
            info.update(
                questions=sum(categories.values()),
                scored_questions=sum(v for k, v in categories.items() if k != "5"),
                categories=dict(sorted(categories.items())),
            )
        elif benchmark == "longmemeval":
            types = Counter(item.get("question_type", "unknown") for item in data)
            info.update(
                question_types=dict(sorted(types.items())),
                sessions=sum(len(item.get("haystack_sessions", [])) for item in data),
                abstention=sum(
                    str(item.get("question_id", "")).endswith("_abs") for item in data
                ),
            )
        else:
            manifest = validate_swe_bundle(
                path,
                expected_dataset="princeton-nlp/SWE-bench_Lite",
                expected_config="default",
                expected_split="test",
            )
            repositories = Counter(str(item.get("repo", "unknown")) for item in data)
            years = Counter(str(item.get("created_at", "unknown"))[:4] for item in data)
            dataset_hash = sha256_file(path)
            revision = manifest["resolved_revision"]
            info.update(
                repositories=dict(sorted(repositories.items())),
                repository_count=len(repositories),
                years=dict(sorted(years.items())),
                split="test",
                source="princeton-nlp/SWE-bench_Lite",
                dataset_sha256=dataset_hash,
                dataset_revision=revision,
            )
        return info

    async def create_run(self, spec: dict[str, Any]) -> dict[str, Any]:
        benchmark = spec.get("benchmark")
        if benchmark not in PHASE_DEFINITIONS:
            raise ValueError("benchmark must be locomo, longmemeval, or swebench")
        run_mode = "harness" if benchmark == "swebench" else str(
            spec.get("run_mode") or "nemori"
        )
        if benchmark != "swebench":
            if run_mode not in {"nemori", "direct"}:
                raise ValueError("run_mode must be nemori or direct")
            if run_mode == "direct" and benchmark != "locomo":
                raise ValueError("direct baseline is currently available for LoCoMo")

        datasets = {item["id"]: item for item in await self.dataset_info()}
        dataset = datasets[benchmark]
        if not dataset.get("available"):
            raise ValueError(f"Dataset is missing: {dataset['path']}")

        start_index = max(0, int(spec.get("start_index", 0)))
        limit = max(0, int(spec.get("limit", 0)))
        total = int(dataset["samples"])
        if start_index >= total:
            raise ValueError(f"start_index must be less than {total}")
        end_index = min(total, start_index + limit) if limit else total
        sample_count = end_index - start_index
        if sample_count < 1:
            raise ValueError("The selected slice is empty")

        run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        run_dir = self._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=False)
        source = Path(dataset["path"])
        swe_evaluator_source: Path | None = None
        swe_evaluator_sha256: str | None = None
        if benchmark == "swebench" and spec.get("swe_evaluator_input_path"):
            candidate = Path(str(spec["swe_evaluator_input_path"])).expanduser()
            if not candidate.is_absolute():
                candidate = ROOT / candidate
            candidate = candidate.resolve()
            if not candidate.is_file():
                raise ValueError(
                    f"Frozen SWE-bench evaluator input is missing: {candidate}"
                )
            actual_sha = sha256_file(candidate)
            expected_sha = str(spec.get("swe_evaluator_input_sha256") or "").lower()
            if not expected_sha or actual_sha != expected_sha:
                raise ValueError("Frozen SWE-bench evaluator input hash mismatch")
            evaluator_rows = json.loads(candidate.read_text(encoding="utf-8"))
            if not isinstance(evaluator_rows, list) or len(evaluator_rows) != sample_count:
                raise ValueError(
                    "Frozen SWE-bench evaluator input does not match the selected sample count"
                )
            expected_ids = [
                str(value) for value in spec.get("swe_expected_instance_ids", [])
            ]
            actual_ids = [
                str(item.get("instance_id") or "")
                for item in evaluator_rows
                if isinstance(item, Mapping)
            ]
            if len(actual_ids) != len(evaluator_rows) or actual_ids != expected_ids:
                raise ValueError(
                    "Frozen SWE-bench evaluator input instance order mismatch"
                )
            swe_evaluator_source = candidate
            swe_evaluator_sha256 = actual_sha
        input_path = source
        if benchmark == "swebench" or start_index or end_index != total:
            input_path = run_dir / "input.json"
            if swe_evaluator_source is not None:
                await asyncio.to_thread(shutil.copy2, swe_evaluator_source, input_path)
            else:
                await asyncio.to_thread(
                    self._write_subset, source, input_path, start_index, end_index
                )
        agent_input_path: Path | None = None
        if benchmark == "swebench":
            agent_input_path = run_dir / "agent-input.json"
            await asyncio.to_thread(
                self._write_swe_agent_subset,
                swe_evaluator_source or source,
                agent_input_path,
                0 if swe_evaluator_source is not None else start_index,
                sample_count if swe_evaluator_source is not None else end_index,
            )
            copied_evaluator_hash = sha256_file(input_path)
            if (
                swe_evaluator_sha256 is not None
                and copied_evaluator_hash != swe_evaluator_sha256
            ):
                raise ValueError("Copied SWE-bench evaluator input hash mismatch")
            swe_evaluator_sha256 = copied_evaluator_hash

        swe_predictions_source = str(spec.get("swe_predictions_path") or "gold").strip()
        swe_predictions_runtime = "gold"
        predictions_path: Path | None = None
        swe_cache_level = str(spec.get("swe_cache_level") or "env")
        if swe_cache_level not in {"none", "base", "env", "instance"}:
            raise ValueError("swe_cache_level must be none, base, env, or instance")
        if benchmark == "swebench" and swe_predictions_source.lower() != "gold":
            candidate = Path(swe_predictions_source).expanduser()
            if not candidate.is_absolute():
                candidate = ROOT / candidate
            candidate = candidate.resolve()
            if not candidate.is_file():
                raise ValueError(f"SWE-bench predictions file is missing: {candidate}")
            if candidate.suffix.lower() not in {".json", ".jsonl"}:
                raise ValueError("SWE-bench predictions must be a .json or .jsonl file")
            predictions_path = run_dir / f"predictions{candidate.suffix.lower()}"
            expected_predictions_sha = str(
                spec.get("swe_predictions_sha256") or ""
            ).lower()
            if expected_predictions_sha and sha256_file(candidate) != expected_predictions_sha:
                raise ValueError("Frozen SWE-bench predictions hash mismatch")
            await asyncio.to_thread(shutil.copy2, candidate, predictions_path)
            if expected_predictions_sha and sha256_file(predictions_path) != expected_predictions_sha:
                raise ValueError("Copied SWE-bench predictions hash mismatch")
            swe_predictions_source = str(candidate)
            swe_predictions_runtime = predictions_path.name

        defaults = self.env_store.defaults_for_run()
        paper_profile = benchmark != "swebench" and bool(spec.get("paper_profile", True))
        observation_window = 20 if paper_profile else max(
            1, int(spec.get("observation_window", 20))
        )
        llm_timeout = max(1.0, float(spec.get("llm_timeout", 120)))
        top_k_episodes = 10 if paper_profile else int(spec.get("top_k_episodes", 10))
        top_k_semantic = 20 if paper_profile else int(spec.get("top_k_semantic", 20))
        include_original = (
            2
            if paper_profile
            else int(spec.get("include_original_messages_top_k", 2))
        )
        search_method = "vector" if paper_profile else spec.get("search_method", "vector")
        base_config_path = EVALUATION_DIR / benchmark / "config.json"
        base_config = json.loads(base_config_path.read_text(encoding="utf-8"))
        llm_model = spec.get("llm_model") or defaults["llm_model"]
        embedding_model = spec.get("embedding_model") or defaults["embedding_model"]
        if benchmark != "swebench":
            base_config.update(
                {
                    "agent_id": f"dashboard_{run_id.replace('-', '_')}",
                    "llm_model": llm_model,
                    "embedding_model": embedding_model,
                    "qdrant_collection_prefix": f"nemori_{run_id.replace('-', '_')}",
                    "search_top_k_episodes": top_k_episodes,
                    "search_top_k_semantic": top_k_semantic,
                    "observation_window": observation_window,
                    "batch_threshold": observation_window,
                    "buffer_size_min": observation_window,
                    "llm_timeout": llm_timeout,
                    "llm_retries": 3,
                    # arXiv:2508.03341v4 native-management defaults.
                    "merge_top_k": 5,
                    "semantic_context_top_k": 10,
                    "semantic_context_similarity_threshold": 0.70,
                    "semantic_consolidation_top_k": 5,
                }
            )
        config_path = run_dir / "config.json"
        atomic_json(config_path, base_config)

        phase_source = PHASE_DEFINITIONS[benchmark]
        if run_mode == "direct":
            phase_source = [
                ("search", "直接回答", "不构建记忆，直接用原始对话上下文回答"),
                *[item for item in phase_source if item[0] in {"evaluate", "aggregate"}],
            ]
        phases = [
            {
                "id": phase_id,
                "label": label,
                "description": description,
                "status": "pending",
                "progress": 0,
                "started_at": None,
                "ended_at": None,
                "error": None,
            }
            for phase_id, label, description in phase_source
        ]
        run = {
            "id": run_id,
            "name": str(spec.get("name") or f"{benchmark}-{run_id[-6:]}")[:80],
            "benchmark": benchmark,
            "status": "ready",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "current_phase": None,
            "current_activity": "等待启动",
            "auto_advance": bool(spec.get("auto_advance", True)),
            "progress": 0,
            "phases": phases,
            "process": {
                "pid": None,
                "cpu": 0.0,
                "memory_mb": 0.0,
                "virtual_memory_mb": 0.0,
                "elapsed": 0,
            },
            "selection": {
                "start_index": start_index,
                "end_index": end_index,
                "sample_count": sample_count,
                "dataset_total": total,
            },
            "settings": {
                "llm_model": llm_model,
                "embedding_model": embedding_model,
                "judge_model": spec.get("judge_model") or defaults["judge_model"],
                "run_mode": run_mode,
                "paper_profile": paper_profile,
                "strict_quality": bool(spec.get("strict_quality", True)),
                "observation_window": observation_window,
                "llm_timeout": llm_timeout,
                "search_method": search_method,
                "top_k_episodes": top_k_episodes,
                "top_k_semantic": top_k_semantic,
                "include_original_messages_top_k": include_original,
                "batch_size": max(1, int(spec.get("batch_size", 1))),
                "ingest_concurrency": max(1, int(spec.get("ingest_concurrency", 1))),
                "search_concurrency": max(1, int(spec.get("search_concurrency", 10))),
                "judge_concurrency": max(1, int(spec.get("judge_concurrency", 10))),
                "checkpoint_every": max(1, int(spec.get("checkpoint_every", 5))),
                "swe_predictions_path": swe_predictions_source,
                "swe_predictions_runtime": swe_predictions_runtime,
                "swe_max_workers": max(1, min(24, int(spec.get("swe_max_workers", 1)))),
                "swe_timeout": max(60, min(14_400, int(spec.get("swe_timeout", 1800)))),
                "swe_cache_level": swe_cache_level,
                "swe_clean": bool(spec.get("swe_clean", False)),
                "swe_evaluator_input_sha256": swe_evaluator_sha256,
                "swe_predictions_sha256": (
                    sha256_file(predictions_path) if predictions_path else None
                ),
                "max_process_vms_mb": max(
                    1_024,
                    int(
                        spec.get("max_process_vms_mb")
                        or os.environ.get("NEMORI_DASHBOARD_MAX_VMS_MB")
                        or DEFAULT_PROCESS_VMS_LIMIT_MB
                    ),
                ),
            },
            "paths": {
                "run_dir": str(run_dir),
                "input": str(input_path),
                **(
                    {"agent_input": str(agent_input_path)}
                    if agent_input_path is not None
                    else {}
                ),
                "config": str(config_path),
                "results": str(run_dir / "results.json"),
                "metrics": str(run_dir / "metrics.json"),
                "report": str(run_dir / "report.json"),
                "summary": str(run_dir / "summary.json"),
                "log": str(run_dir / "run.log"),
                **({"predictions": str(predictions_path)} if predictions_path else {}),
            },
            "summary": None,
            "notes": str(spec.get("notes") or "")[:4000],
            "related_runs": {},
            "quality_issues": {
                "segmentation": 0,
                "episode_generation": 0,
                "semantic_generation": 0,
                "semantic_consolidation": 0,
                "episode_integration": 0,
                "details": [],
            },
            "token_usage": {
                "source": "provider_reported_successful_responses",
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "requests": 0,
                "errors": 0,
                "phases": {},
            },
            "workflow": {"active_node": None, "nodes": {}, "events": []},
            "last_log": "",
            "error": None,
        }
        if benchmark == "swebench":
            using_gold = swe_predictions_runtime == "gold"
            run["paper_alignment"] = {
                "algorithm": True,
                "score_comparable": not using_gold,
                "deviations": ["Gold patch 仅验证评测链路"] if using_gold else [],
            }
        else:
            llm_name = str(run["settings"]["llm_model"]).lower()
            embedding_name = str(run["settings"]["embedding_model"]).lower()
            judge_name = str(run["settings"]["judge_model"]).lower()
            deviations = []
            if not any(llm_name.endswith(name) for name in ("gpt-4o-mini", "gpt-4.1-mini")):
                deviations.append("生成模型替代")
            if not embedding_name.endswith("text-embedding-3-small"):
                deviations.append("向量模型替代")
            if not judge_name.endswith("gpt-4o-mini"):
                deviations.append("Judge 模型替代")
            run["paper_alignment"] = {
                "algorithm": paper_profile and run_mode == "nemori",
                "score_comparable": paper_profile and run_mode == "nemori" and not deviations,
                "deviations": [*deviations, *(["裸上下文基线"] if run_mode == "direct" else [])],
            }
        self.runs[run_id] = run
        self._save(run)
        await self._emit(run_id, "state", run)
        await self._log(run_id, "system", f"运行已创建：{sample_count}/{total} 个样本")
        return run

    def update_notes(self, run_id: str, notes: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        run["notes"] = str(notes or "")[:4000]
        self._save(run)
        return run

    def link_runs(self, run_id: str, paired_id: str, *, relation: str) -> None:
        run = self.get_run(run_id)
        paired = self.get_run(paired_id)
        run.setdefault("related_runs", {})[relation] = paired_id
        paired.setdefault("related_runs", {})["source"] = run_id
        self._save(run)
        self._save(paired)

    def delete_run(self, run_id: str) -> None:
        run = self.get_run(run_id)
        if run_id in self.tasks and not self.tasks[run_id].done():
            raise ValueError("Cannot delete an active run")
        if run_id in self.processes:
            raise ValueError("Cannot delete a run with an active process")
        if run.get("status") in ACTIVE_STATUSES:
            raise ValueError("Cannot delete an active run")
        run_dir = self._run_dir(run_id)
        del self.runs[run_id]
        self.log_buffers.pop(run_id, None)
        self.subscribers.pop(run_id, None)
        for other in self.runs.values():
            related = other.get("related_runs")
            if isinstance(related, dict):
                for key, value in list(related.items()):
                    if value == run_id:
                        related.pop(key, None)
                self._save(other)
        shutil.rmtree(run_dir)

    @staticmethod
    def _write_subset(source: Path, target: Path, start: int, end: int) -> None:
        data = json.loads(source.read_text(encoding="utf-8"))
        atomic_json(target, data[start:end])

    @staticmethod
    def _write_swe_agent_subset(
        source: Path, target: Path, start: int, end: int
    ) -> None:
        data = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("SWE-bench dataset must be a JSON array")
        safe = [sanitize_agent_instance(item) for item in data[start:end]]
        atomic_json(target, safe)

    async def _log(self, run_id: str, stream: str, message: str) -> None:
        message = self._redact(message.strip())
        if not message:
            return
        stamp = datetime.now().strftime("%H:%M:%S")
        rendered = f"[{stamp}] [{stream}] {message}"
        self.log_buffers[run_id].append(rendered)
        run = self.get_run(run_id)
        run["last_log"] = message[-500:]
        run["current_activity"] = message[-180:]
        with Path(run["paths"]["log"]).open("a", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
        await self._emit(run_id, "log", {"stream": stream, "message": rendered})

    def logs(self, run_id: str, tail: int = 300) -> list[str]:
        self.get_run(run_id)
        return list(self.log_buffers[run_id])[-max(1, min(tail, 1000)) :]

    def workflow(self, run_id: str) -> dict[str, Any]:
        return workflow_for_run(self.get_run(run_id))

    async def start(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        if run_id in self.tasks and not self.tasks[run_id].done():
            raise ValueError("Run is already active")
        if run["status"] not in {"ready", "awaiting"}:
            raise ValueError(f"Cannot start a run in state {run['status']}")
        if run.get("benchmark") == "swebench":
            evaluator_hash = str(
                run.get("settings", {}).get("swe_evaluator_input_sha256") or ""
            )
            input_path = Path(run["paths"]["input"])
            if evaluator_hash and sha256_file(input_path) != evaluator_hash:
                raise ValueError(
                    "Frozen SWE-bench evaluator input changed before Harness start"
                )
            predictions_hash = str(
                run.get("settings", {}).get("swe_predictions_sha256") or ""
            )
            predictions_value = run.get("paths", {}).get("predictions")
            if predictions_hash:
                if (
                    not predictions_value
                    or sha256_file(predictions_value) != predictions_hash
                ):
                    raise ValueError(
                        "Frozen SWE-bench predictions changed before Harness start"
                    )
        config_path = Path(run["paths"]["config"])
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if run.get("benchmark") != "swebench":
            config.setdefault(
                "qdrant_collection_prefix", f"nemori_{run_id.replace('-', '_')}"
            )
        atomic_json(config_path, config)
        task = asyncio.create_task(self._drive(run_id), name=f"nemori-run-{run_id}")
        self.tasks[run_id] = task
        return run

    async def retry_failed(self, run_id: str) -> dict[str, Any]:
        """Continue a failed non-ingest phase, preserving search checkpoints."""
        run = self.get_run(run_id)
        if run_id in self.tasks and not self.tasks[run_id].done():
            raise ValueError("Run is already active")
        if run.get("status") not in {"failed", "stopped", "interrupted"}:
            raise ValueError(f"Cannot retry a run in state {run.get('status')}")

        failed_index = next(
            (
                index
                for index, phase in enumerate(run.get("phases", []))
                if phase.get("status") in {"failed", "stopped", "interrupted"}
            ),
            None,
        )
        if failed_index is None:
            raise ValueError("No failed or interrupted phase was found")
        failed_phase = run["phases"][failed_index]
        if failed_phase["id"] == "ingest":
            raise ValueError(
                "构建记忆不能原地重试，否则会向同一记忆库重复写入；请克隆运行后重建"
            )

        for phase in run["phases"][failed_index:]:
            phase.update(
                status="pending",
                progress=0,
                started_at=None,
                ended_at=None,
                error=None,
            )
            phase.pop("completed", None)
            phase.pop("total", None)

        artifact_names: tuple[str, ...]
        if run.get("benchmark") == "swebench" and failed_phase["id"] == "evaluate":
            artifact_names = ("results", "report", "summary")
        elif failed_phase["id"] == "search":
            # results.json and recovery *.tmp files are intentionally retained.
            artifact_names = ("metrics", "report", "summary")
        elif failed_phase["id"] == "evaluate":
            # Preserve metrics.json: evals.py validates it and resumes only the
            # questions that were not checkpointed before interruption.
            artifact_names = ("report", "summary")
        else:
            artifact_names = ("summary",)
        for name in artifact_names:
            try:
                Path(run["paths"][name]).unlink(missing_ok=True)
            except OSError as exc:
                raise ValueError(f"Cannot clear stale {name} artifact: {exc}") from exc

        run["summary"] = None
        run["status"] = "ready"
        run["error"] = None
        run["current_phase"] = None
        run["current_activity"] = f"准备从检查点继续：{failed_phase['label']}"
        self._recalculate_progress(run)
        self._save(run)
        await self._log(run_id, "operator", run["current_activity"])
        task = asyncio.create_task(self._drive(run_id), name=f"nemori-retry-{run_id}")
        self.tasks[run_id] = task
        return run

    async def _drive(self, run_id: str) -> None:
        run = self.get_run(run_id)
        run["status"] = "running"
        run["error"] = None
        self._save(run)
        await self._emit(run_id, "state", run)

        try:
            while True:
                pending = next(
                    (phase for phase in run["phases"] if phase["status"] == "pending"),
                    None,
                )
                if pending is None:
                    run["status"] = "completed"
                    run["current_phase"] = None
                    run["current_activity"] = "评测完成"
                    run["progress"] = 100
                    self._save(run)
                    await self._emit(run_id, "state", run)
                    await self._log(run_id, "system", "全部阶段完成")
                    return

                if pending["id"] == "aggregate":
                    ok = await self._aggregate(run_id, pending)
                else:
                    ok = await self._run_phase(run_id, pending)
                if not ok:
                    return

                if not run["auto_advance"]:
                    remaining = any(phase["status"] == "pending" for phase in run["phases"])
                    if remaining:
                        run["status"] = "awaiting"
                        run["current_phase"] = None
                        run["current_activity"] = "等待人工确认下一阶段"
                        self._save(run)
                        await self._emit(run_id, "state", run)
                        await self._log(run_id, "system", "已暂停在阶段边界，等待人工推进")
                        return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            run["status"] = "failed"
            run["error"] = f"{type(exc).__name__}: {exc}"
            run["current_activity"] = "控制台编排失败"
            self._save(run)
            await self._log(run_id, "error", run["error"])
            await self._emit(run_id, "state", run)

    def _phase_command(self, run: dict[str, Any], phase_id: str) -> list[str]:
        python = sys.executable
        benchmark = run["benchmark"]
        paths = run["paths"]
        settings = run["settings"]
        if benchmark == "swebench":
            if phase_id != "evaluate":
                raise ValueError(f"Unsupported SWE-bench phase: {phase_id}")
            config = json.loads(Path(paths["config"]).read_text(encoding="utf-8"))
            distro = str(
                os.environ.get("NEMORI_SWEBENCH_WSL_DISTRO")
                or config.get("wsl_distribution")
                or "Ubuntu"
            )
            linux_args = [
                "run",
                "--python",
                "3.12",
                "--with",
                f"swebench=={config.get('swebench_version', '4.1.0')}",
            ]
            for dependency in SWE_HARNESS_UV_DEPENDENCIES:
                linux_args.extend(("--with", dependency))
            linux_args.extend(
                [
                    "--no-project",
                    "python",
                    "-u",
                    "../../evaluation/swebench/run.py",
                    "--input",
                    Path(paths["input"]).name,
                    "--predictions",
                    settings["swe_predictions_runtime"],
                    "--results",
                    Path(paths["results"]).name,
                    "--report",
                    Path(paths["report"]).name,
                    "--run-id",
                    run["id"],
                    "--dataset-name",
                    str(config.get("dataset_name", "princeton-nlp/SWE-bench_Lite")),
                    "--split",
                    str(config.get("split", "test")),
                    "--max-workers",
                    str(settings["swe_max_workers"]),
                    "--timeout",
                    str(settings["swe_timeout"]),
                    "--cache-level",
                    settings["swe_cache_level"],
                    "--namespace",
                    str(config.get("docker_namespace", "swebench")),
                ]
            )
            if settings.get("swe_clean"):
                linux_args.append("--clean")
            if _is_windows_host():
                return [
                    "wsl.exe",
                    "-d",
                    distro,
                    "--cd",
                    paths["run_dir"],
                    "--",
                    "bash",
                    "-lc",
                    'if test -x "$HOME/.local/bin/uv"; then '
                    'exec "$HOME/.local/bin/uv" ' + shlex.join(linux_args) + '; '
                    'else exec uv ' + shlex.join(linux_args) + '; fi',
                ]
            uv_path = shutil.which("uv")
            user_uv = Path.home() / ".local" / "bin" / "uv"
            if not uv_path and user_uv.is_file():
                uv_path = str(user_uv)
            if not uv_path:
                raise ValueError(
                    "uv is required for the Linux SWE Harness runtime"
                )
            return [uv_path, *linux_args]
        if benchmark == "locomo":
            if phase_id == "ingest":
                return [
                    python,
                    "-u",
                    str(EVALUATION_DIR / "locomo" / "add.py"),
                    "--data",
                    paths["input"],
                    "--config",
                    paths["config"],
                    "--batch-size",
                    str(settings["batch_size"]),
                    "--max-workers",
                    str(settings["ingest_concurrency"]),
                ]
            if phase_id == "search":
                if settings.get("run_mode") == "direct":
                    command = [
                        python,
                        "-u",
                        str(EVALUATION_DIR / "locomo" / "baseline_direct.py"),
                        "--data",
                        paths["input"],
                        "--config",
                        paths["config"],
                        "--output",
                        paths["results"],
                        "--max-concurrent-answer",
                        str(settings["search_concurrency"]),
                        "--checkpoint-every",
                        str(settings["checkpoint_every"]),
                    ]
                    if settings.get("paper_profile"):
                        command.append("--scored-only")
                    return command
                command = [
                    python,
                    "-u",
                    str(EVALUATION_DIR / "locomo" / "search.py"),
                    "--data",
                    paths["input"],
                    "--config",
                    paths["config"],
                    "--output",
                    paths["results"],
                    "--top-k-episodes",
                    str(settings["top_k_episodes"]),
                    "--top-k-semantic",
                    str(settings["top_k_semantic"]),
                    "--include-original-messages-top-k",
                    str(settings["include_original_messages_top_k"]),
                    "--search-method",
                    settings["search_method"],
                    "--max-workers",
                    str(settings["search_concurrency"]),
                    "--max-concurrent-answer",
                    str(settings["search_concurrency"]),
                    "--checkpoint-every",
                    str(settings["checkpoint_every"]),
                ]
                if settings.get("paper_profile"):
                    command.append("--scored-only")
                return command
            return [
                python,
                "-u",
                str(EVALUATION_DIR / "locomo" / "evals.py"),
                "--input_file",
                paths["results"],
                "--output_file",
                paths["metrics"],
                "--max_workers",
                str(settings["judge_concurrency"]),
                "--checkpoint_every",
                str(settings["checkpoint_every"]),
            ]

        if phase_id == "ingest":
            return [
                python,
                "-u",
                str(EVALUATION_DIR / "longmemeval" / "add.py"),
                "--data_path",
                paths["input"],
                "--config",
                paths["config"],
                "--batch_size",
                str(settings["batch_size"]),
                "--max_workers",
                str(settings["ingest_concurrency"]),
            ]
        if phase_id == "search":
            return [
                python,
                "-u",
                str(EVALUATION_DIR / "longmemeval" / "search.py"),
                "--data",
                paths["input"],
                "--config",
                paths["config"],
                "--output",
                paths["results"],
                "--top-k-episodes",
                str(settings["top_k_episodes"]),
                "--top-k-semantic",
                str(settings["top_k_semantic"]),
                "--include-original-messages-top-k",
                str(settings["include_original_messages_top_k"]),
                "--search-method",
                settings["search_method"],
                "--max-concurrent-answer",
                str(settings["search_concurrency"]),
                "--checkpoint-every",
                str(settings["checkpoint_every"]),
            ]
        return [
            python,
            "-u",
            str(EVALUATION_DIR / "longmemeval" / "evals.py"),
            paths["results"],
            "--model",
            settings["judge_model"],
            "--output",
            paths["report"],
            "--max-concurrent",
            str(settings["judge_concurrency"]),
        ]

    @staticmethod
    def _phase_working_directory(run: Mapping[str, Any]) -> Path:
        if run.get("benchmark") == "swebench":
            return Path(run["paths"]["run_dir"])
        return ROOT

    async def _run_phase(self, run_id: str, phase: dict[str, Any]) -> bool:
        run = self.get_run(run_id)
        phase["status"] = "running"
        phase["started_at"] = utc_now()
        phase["progress"] = 0
        run["status"] = "running"
        run["current_phase"] = phase["id"]
        run["current_activity"] = phase["description"]
        self._recalculate_progress(run)
        self._save(run)
        await self._emit(run_id, "state", run)

        command = self._phase_command(run, phase["id"])
        phase["command"] = command
        await self._log(run_id, "system", f"开始阶段：{phase['label']}")
        await self._log(run_id, "command", subprocess.list2cmdline(command))

        env = self.env_store.subprocess_env()
        env["PYTHONPATH"] = str(ROOT)
        env["LLM_JUDGE_MODEL"] = run["settings"]["judge_model"]
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(self._phase_working_directory(run)),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            creationflags=creationflags,
        )
        self.processes[run_id] = process
        run["process"].update(
            pid=process.pid,
            cpu=0.0,
            memory_mb=0.0,
            virtual_memory_mb=0.0,
            elapsed=0,
        )
        self._save(run)
        await self._emit(run_id, "state", run)

        reader = asyncio.create_task(self._read_output(run_id, phase, process))
        resource = asyncio.create_task(self._resource_loop(run_id, process, phase["started_at"]))
        return_code = await process.wait()
        await reader
        resource.cancel()
        try:
            await resource
        except asyncio.CancelledError:
            pass
        self.processes.pop(run_id, None)
        run["process"].update(pid=None, cpu=0.0, memory_mb=0.0, virtual_memory_mb=0.0)

        if run_id in self.stop_requested:
            self.stop_requested.discard(run_id)
            phase["status"] = "stopped"
            phase["ended_at"] = utc_now()
            stop_reason = run.pop("_stop_reason", None)
            run["status"] = "failed" if stop_reason else "stopped"
            if stop_reason:
                phase["error"] = stop_reason
                run["error"] = stop_reason
                run["current_activity"] = "内存保护已停止阶段"
            else:
                run["current_activity"] = "已由用户停止"
            self._save(run)
            await self._log(run_id, "system", stop_reason or f"阶段已停止：{phase['label']}")
            await self._emit(run_id, "state", run)
            return False

        if return_code != 0:
            phase["status"] = "failed"
            phase["error"] = f"Process exited with code {return_code}"
            phase["ended_at"] = utc_now()
            run["status"] = "failed"
            run["error"] = phase["error"]
            run["current_activity"] = f"{phase['label']}失败"
            self._save(run)
            await self._log(run_id, "error", phase["error"])
            await self._emit(run_id, "state", run)
            return False

        ingest_quality = run.get("ingest_quality") or {}
        quality_issues = run.get("quality_issues") or {}
        issue_count = sum(
            int(quality_issues.get(key, 0))
            for key in {
                "segmentation",
                "episode_generation",
                "semantic_generation",
                "semantic_consolidation",
                "episode_integration",
            }
        )
        strict_failure = bool(run["settings"].get("strict_quality")) and (
            issue_count > 0 or int(ingest_quality.get("fallback_episodes", 0)) > 0
        )
        if phase["id"] == "ingest" and (
            ingest_quality.get("all_fallback") or strict_failure
        ):
            phase["status"] = "failed"
            if strict_failure:
                phase["error"] = (
                    "论文严格模式质量门禁未通过："
                    f"检测到 {issue_count} 次生成/解析异常，"
                    f"{int(ingest_quality.get('fallback_episodes', 0))} 个 fallback 情景。"
                    "请先修复模型超时或结构化输出，再克隆运行重建记忆。"
                )
            else:
                phase["error"] = (
                    "所有情景记忆均由 fallback 生成；LLM 叙事与语义抽取未成功"
                )
            phase["ended_at"] = utc_now()
            run["status"] = "failed"
            run["error"] = phase["error"]
            run["current_activity"] = "构建记忆质量门禁未通过"
            self._save(run)
            await self._log(run_id, "error", phase["error"])
            await self._emit(run_id, "state", run)
            return False

        phase["status"] = "completed"
        phase["progress"] = 100
        phase["ended_at"] = utc_now()
        run["current_activity"] = f"{phase['label']}完成"
        self._recalculate_progress(run)
        self._save(run)
        await self._log(run_id, "system", f"完成阶段：{phase['label']}")
        await self._emit(run_id, "state", run)
        await self._emit(run_id, "results", self.results(run_id, limit=20))
        return True

    async def _read_output(
        self,
        run_id: str,
        phase: dict[str, Any],
        process: asyncio.subprocess.Process,
    ) -> None:
        assert process.stdout is not None
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        buffer = ""
        while True:
            chunk = await process.stdout.read(1024)
            if not chunk:
                break
            buffer += decoder.decode(chunk)
            pieces = re.split(r"[\r\n]+", buffer)
            buffer = pieces.pop() if pieces else ""
            for line in pieces:
                await self._handle_output_line(run_id, phase, line)
        buffer += decoder.decode(b"", final=True)
        if buffer.strip():
            await self._handle_output_line(run_id, phase, buffer)

    async def _handle_output_line(
        self, run_id: str, phase: dict[str, Any], line: str
    ) -> None:
        clean = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line).strip()
        if not clean or "\x00" in clean:
            return
        if clean.startswith("NEMORI_TRACE "):
            try:
                payload = json.loads(clean.removeprefix("NEMORI_TRACE "))
                await self._record_trace(run_id, payload)
                return
            except Exception:
                pass
        if clean.startswith("NEMORI_PROGRESS "):
            try:
                payload = json.loads(clean.removeprefix("NEMORI_PROGRESS "))
                completed = int(payload.get("completed", 0))
                total = max(1, int(payload.get("total", 1)))
                phase["completed"] = completed
                phase["total"] = total
                await self._set_phase_progress(run_id, phase, completed * 100 / total)
                await self._emit(run_id, "progress", payload)
                return
            except Exception:
                pass
        for issue_type, marker in QUALITY_MARKERS:
            if marker in clean:
                await self._record_quality_issue(run_id, issue_type, clean)
                break
        run = self.get_run(run_id)
        if self._record_token_usage_line(run, clean, str(phase.get("id") or "unknown")):
            self._save(run)
            await self._emit(run_id, "token_usage", run["token_usage"])
        matches = re.findall(r"(\d{1,3})%[^\r\n]*?(\d+)\s*/\s*(\d+)", clean)
        if matches:
            percent, completed, total = matches[-1]
            await self._set_phase_progress(run_id, phase, float(percent))
            await self._emit(
                run_id,
                "progress",
                {"phase": phase["id"], "completed": int(completed), "total": int(total)},
            )
        await self._log(run_id, "process", clean)

    async def _record_quality_issue(
        self, run_id: str, issue_type: str, message: str
    ) -> None:
        run = self.get_run(run_id)
        issues = run.setdefault(
            "quality_issues",
            {
                "segmentation": 0,
                "episode_generation": 0,
                "semantic_generation": 0,
                "semantic_consolidation": 0,
                "episode_integration": 0,
                "details": [],
            },
        )
        issues[issue_type] = int(issues.get(issue_type, 0)) + 1
        details = issues.setdefault("details", [])
        details.append(
            {
                "time": utc_now(),
                "type": issue_type,
                "message": self._redact(message)[:600],
            }
        )
        issues["details"] = details[-20:]
        self._save(run)
        await self._emit(run_id, "quality", issues)

    async def _record_trace(self, run_id: str, payload: dict[str, Any]) -> None:
        run = self.get_run(run_id)
        workflow = run.setdefault(
            "workflow", {"active_node": None, "nodes": {}, "events": []}
        )
        payload = {
            **payload,
            "time": payload.get("time") or utc_now(),
            "trace_id": payload.get("trace_id") or uuid.uuid4().hex[:12],
        }
        node_ids = payload.get("nodes") or [payload.get("node")]
        node_ids = [str(node_id) for node_id in node_ids if node_id]
        for node_id in node_ids:
            previous = workflow["nodes"].get(node_id, {})
            workflow["nodes"][node_id] = {
                "count": int(previous.get("count", 0)) + 1,
                "updated_at": payload["time"],
                "last_event_id": payload["trace_id"],
            }
        workflow["active_node"] = node_ids[-1] if node_ids else None
        workflow["events"].append(payload)
        workflow["events"] = workflow["events"][-24:]
        if payload.get("event") == "ingest_complete":
            episodes = payload.get("episodes") or []
            fallback_count = sum(
                1 for episode in episodes if episode.get("fallback") is True
            )
            run["ingest_quality"] = {
                "episodes": len(episodes),
                "fallback_episodes": fallback_count,
                "semantic_memories": len(payload.get("semantic_memories") or []),
                "compression_ratio": payload.get("compression_ratio"),
                "all_fallback": bool(episodes) and fallback_count == len(episodes),
                "quality_issues": {
                    key: int((run.get("quality_issues") or {}).get(key, 0))
                    for key in (
                        "segmentation",
                        "episode_generation",
                        "semantic_generation",
                        "semantic_consolidation",
                        "episode_integration",
                    )
                },
            }
        self._save(run)
        summary = payload.get("summary") or payload.get("event") or "workflow trace"
        await self._log(run_id, "trace", str(summary))
        await self._emit(run_id, "trace", payload)

    async def _set_phase_progress(
        self, run_id: str, phase: dict[str, Any], progress: float
    ) -> None:
        value = max(0, min(100, int(progress)))
        if value == phase.get("progress"):
            return
        phase["progress"] = value
        run = self.get_run(run_id)
        self._recalculate_progress(run)
        self._save(run)
        await self._emit(run_id, "state", run)

    @staticmethod
    def _recalculate_progress(run: dict[str, Any]) -> None:
        phases = run.get("phases", [])
        run["progress"] = int(sum(phase.get("progress", 0) for phase in phases) / max(1, len(phases)))

    async def _resource_loop(
        self, run_id: str, process: asyncio.subprocess.Process, started_at: str
    ) -> None:
        started = time.monotonic()
        try:
            parent = psutil.Process(process.pid)
            parent.cpu_percent(None)
            while process.returncode is None:
                await asyncio.sleep(1)
                try:
                    processes = [parent, *parent.children(recursive=True)]
                    cpu = sum(item.cpu_percent(None) for item in processes if item.is_running())
                    rss_bytes = 0
                    vms_bytes = 0
                    for item in processes:
                        if not item.is_running():
                            continue
                        info = item.memory_info()
                        rss_bytes += info.rss
                        vms_bytes += info.vms
                    memory = rss_bytes / (1024 * 1024)
                    virtual_memory = vms_bytes / (1024 * 1024)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
                run = self.get_run(run_id)
                run["process"].update(
                    cpu=round(cpu, 1),
                    memory_mb=round(memory, 1),
                    virtual_memory_mb=round(virtual_memory, 1),
                    elapsed=int(time.monotonic() - started),
                )
                await self._emit(run_id, "resource", run["process"])
                if self._process_exceeds_memory_guard(run, virtual_memory):
                    await self._trip_memory_guard(run_id, process, virtual_memory)
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return

    def _process_exceeds_memory_guard(self, run: dict[str, Any], process_vms_mb: float) -> bool:
        limit_mb = float(
            run.get("settings", {}).get("max_process_vms_mb")
            or DEFAULT_PROCESS_VMS_LIMIT_MB
        )
        return process_vms_mb >= limit_mb

    async def _trip_memory_guard(
        self,
        run_id: str,
        process: asyncio.subprocess.Process,
        process_vms_mb: float,
    ) -> None:
        run = self.get_run(run_id)
        limit_mb = int(
            run.get("settings", {}).get("max_process_vms_mb")
            or DEFAULT_PROCESS_VMS_LIMIT_MB
        )
        reason = (
            "内存保护已停止子进程："
            f"Python 进程树虚拟内存 {process_vms_mb:.0f} MB，"
            f"限制 {limit_mb} MB。请降低样本量/并发，或增大页面文件后重试。"
        )
        run["_stop_reason"] = reason
        self.stop_requested.add(run_id)
        await self._log(run_id, "error", reason)
        try:
            await asyncio.to_thread(self._terminate_tree, process.pid)
        except psutil.Error:
            process.terminate()

    async def pause(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        process = self.processes.get(run_id)
        if not process or run["status"] != "running":
            raise ValueError("Run is not currently running")
        await asyncio.to_thread(self._suspend_tree, process.pid)
        run["status"] = "paused"
        current = next(
            (p for p in run["phases"] if p["id"] == run["current_phase"]), None
        )
        if current:
            current["status"] = "paused"
        run["current_activity"] = "进程树已暂停，等待人工继续"
        self._save(run)
        await self._log(run_id, "system", "运行已暂停")
        await self._emit(run_id, "state", run)
        return run

    async def resume(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        process = self.processes.get(run_id)
        if not process or run["status"] != "paused":
            raise ValueError("Run is not paused")
        await asyncio.to_thread(self._resume_tree, process.pid)
        run["status"] = "running"
        current = next(
            (p for p in run["phases"] if p["id"] == run["current_phase"]), None
        )
        if current:
            current["status"] = "running"
        run["current_activity"] = current["description"] if current else "继续运行"
        self._save(run)
        await self._log(run_id, "system", "运行已继续")
        await self._emit(run_id, "state", run)
        return run

    async def stop(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        process = self.processes.get(run_id)
        if not process or run["status"] not in {"running", "paused"}:
            raise ValueError("Run has no active process")
        run["status"] = "stopping"
        run["current_activity"] = "正在停止进程树"
        self.stop_requested.add(run_id)
        self._save(run)
        await self._emit(run_id, "state", run)
        await self._log(run_id, "system", "收到停止请求")
        await asyncio.to_thread(self._terminate_tree, process.pid)
        return run

    @staticmethod
    def _suspend_tree(pid: int) -> None:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for item in reversed(children):
            try:
                item.suspend()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        parent.suspend()

    @staticmethod
    def _resume_tree(pid: int) -> None:
        parent = psutil.Process(pid)
        try:
            parent.resume()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return
        for item in parent.children(recursive=True):
            try:
                item.resume()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    @staticmethod
    def _terminate_tree(pid: int) -> None:
        try:
            parent = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return
        try:
            parent.resume()
        except (psutil.Error, AttributeError):
            pass
        processes = parent.children(recursive=True)
        for item in processes:
            try:
                item.terminate()
            except psutil.Error:
                pass
        try:
            parent.terminate()
        except psutil.Error:
            pass
        _, alive = psutil.wait_procs([*processes, parent], timeout=5)
        for item in alive:
            try:
                item.kill()
            except psutil.Error:
                pass

    async def skip_next(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        if run["status"] not in {"ready", "awaiting"}:
            raise ValueError("A phase can only be skipped at a phase boundary")
        phase = next((p for p in run["phases"] if p["status"] == "pending"), None)
        if not phase:
            raise ValueError("No pending phase")
        phase["status"] = "skipped"
        phase["progress"] = 100
        phase["ended_at"] = utc_now()
        run["current_activity"] = f"已人工跳过：{phase['label']}"
        self._recalculate_progress(run)
        self._save(run)
        await self._log(run_id, "operator", run["current_activity"])
        await self._emit(run_id, "state", run)
        return run

    async def _aggregate(self, run_id: str, phase: dict[str, Any]) -> bool:
        run = self.get_run(run_id)
        phase.update(status="running", progress=10, started_at=utc_now())
        run["current_phase"] = "aggregate"
        run["current_activity"] = "正在汇总指标"
        self._recalculate_progress(run)
        self._save(run)
        await self._emit(run_id, "state", run)
        try:
            summary = await asyncio.to_thread(self._build_summary, run)
            atomic_json(Path(run["paths"]["summary"]), summary)
            run["summary"] = summary
            phase.update(status="completed", progress=100, ended_at=utc_now())
            run["current_activity"] = "指标汇总完成"
            self._recalculate_progress(run)
            self._save(run)
            await self._log(run_id, "system", "汇总结果已生成")
            await self._emit(run_id, "results", self.results(run_id, limit=50))
            await self._emit(run_id, "state", run)
            return True
        except Exception as exc:
            phase.update(status="failed", error=str(exc), ended_at=utc_now())
            run["status"] = "failed"
            run["error"] = f"Summary failed: {exc}"
            self._save(run)
            await self._log(run_id, "error", run["error"])
            await self._emit(run_id, "state", run)
            return False

    @staticmethod
    def _build_summary(run: dict[str, Any]) -> dict[str, Any]:
        if run["benchmark"] == "locomo":
            metrics_path = Path(run["paths"]["metrics"])
            records = flatten_records(json.loads(metrics_path.read_text(encoding="utf-8")))
            if not records:
                raise ValueError("No metric records were produced")
            category_names = {
                "1": "Multi-Hop",
                "2": "Temporal",
                "3": "Open-Domain",
                "4": "Single-Hop",
            }
            buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for item in records:
                buckets[str(item.get("category"))].append(item)

            def means(items: list[dict[str, Any]]) -> dict[str, Any]:
                return {
                    "count": len(items),
                    "bleu": sum(float(x.get("bleu_score", 0)) for x in items) / len(items),
                    "f1": sum(float(x.get("f1_score", 0)) for x in items) / len(items),
                    "accuracy": sum(float(x.get("llm_score", 0)) for x in items) / len(items),
                }

            return {
                "benchmark": "locomo",
                "overall": means(records),
                "by_category": {
                    category_names.get(key, key): means(items)
                    for key, items in sorted(buckets.items())
                },
            }

        report_path = Path(run["paths"]["report"])
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if run["benchmark"] == "swebench":
            total = int(report.get("total_instances", 0))
            completed = int(report.get("completed_instances", 0))
            resolved = int(report.get("resolved_instances", 0))
            results_path = Path(run["paths"]["results"])
            records = (
                flatten_records(json.loads(results_path.read_text(encoding="utf-8")))
                if results_path.exists()
                else []
            )
            repo_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for item in records:
                repo_buckets[str(item.get("category") or "unknown")].append(item)
            return {
                "benchmark": "swebench",
                "overall": {
                    "count": total,
                    "submitted": int(report.get("submitted_instances", 0)),
                    "completed": completed,
                    "resolved": resolved,
                    "accuracy": resolved / total if total else 0,
                },
                "by_category": {
                    repo: {
                        "count": len(items),
                        "resolved": sum(int(bool(item.get("resolved"))) for item in items),
                    }
                    for repo, items in sorted(repo_buckets.items())
                },
                "errors": int(report.get("error_instances", 0)),
            }
        return {
            "benchmark": "longmemeval",
            "overall": {
                "count": report.get("total_questions", 0),
                "correct": report.get("correct_answers", 0),
                "accuracy": report.get("overall_accuracy", 0),
            },
            "by_category": report.get("accuracy_by_type", {}),
            "errors": report.get("error_cases_count", 0),
        }

    def results(self, run_id: str, limit: int = 100, offset: int = 0) -> dict[str, Any]:
        run = self.get_run(run_id)
        if run.get("current_phase") == "search" and run.get("status") in ACTIVE_STATUSES:
            live_events = [
                event
                for event in (run.get("workflow") or {}).get("events", [])
                if event.get("event") == "search_complete"
            ]
            live_items = [
                {
                    "question_id": event.get("question_id"),
                    "question": event.get("question"),
                    "answer": None,
                    "response": event.get("answer"),
                    "category": None,
                    "llm_score": None,
                    "f1_score": None,
                    "bleu_score": None,
                    "memory_count": len(event.get("episodes") or [])
                    + len(event.get("semantic_memories") or []),
                }
                for event in live_events
            ]
            phase = next(
                (item for item in run.get("phases", []) if item.get("id") == "search"),
                {},
            )
            return {
                "total": int(phase.get("completed", len(live_items))),
                "offset": max(0, int(phase.get("completed", 0)) - len(live_items)),
                "items": live_items[-max(1, min(limit, 500)) :],
                "summary": run.get("summary"),
                "source": "live_trace",
            }
        paths = run["paths"]
        source = None
        if Path(paths["metrics"]).exists():
            source = Path(paths["metrics"])
        elif Path(paths["results"]).exists():
            source = Path(paths["results"])
        records: list[dict[str, Any]] = []
        if source:
            try:
                records = flatten_records(json.loads(source.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                records = []
        compact = []
        for item in records[offset : offset + max(1, min(limit, 500))]:
            rendered = {
                "question_id": item.get("question_id"),
                "question": item.get("question"),
                "answer": item.get("answer"),
                "response": item.get("response"),
                "category": item.get("category") or item.get("question_type"),
                "llm_score": item.get("llm_score"),
                "f1_score": item.get("f1_score"),
                "bleu_score": item.get("bleu_score"),
                "memory_count": len(item.get("memories", [])),
            }
            if run.get("benchmark") == "swebench":
                rendered.update(
                    resolved=bool(item.get("resolved")),
                    patch_successfully_applied=item.get(
                        "patch_successfully_applied"
                    ),
                    model_name_or_path=item.get("model_name_or_path"),
                )
            compact.append(rendered)
        return {
            "total": len(records),
            "offset": offset,
            "items": compact,
            "summary": run.get("summary"),
            "source": source.name if source else None,
        }

    def compare(self, run_id: str, against_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        against = self.get_run(against_id)
        left = run.get("summary") or self._load_summary(run)
        right = against.get("summary") or self._load_summary(against)
        left_overall = left.get("overall") or {}
        right_overall = right.get("overall") or {}
        delta: dict[str, float] = {}
        for key in sorted(set(left_overall) | set(right_overall)):
            try:
                delta[key] = float(left_overall.get(key, 0)) - float(
                    right_overall.get(key, 0)
                )
            except (TypeError, ValueError):
                continue
        return {
            "run": {
                "id": run_id,
                "name": run.get("name"),
                "mode": run.get("settings", {}).get("run_mode"),
            },
            "against": {
                "id": against_id,
                "name": against.get("name"),
                "mode": against.get("settings", {}).get("run_mode"),
            },
            "overall": {
                "run": left_overall,
                "against": right_overall,
                "delta": delta,
            },
            "by_category": {
                "run": left.get("by_category"),
                "against": right.get("by_category"),
            },
        }

    @staticmethod
    def _load_summary(run: dict[str, Any]) -> dict[str, Any]:
        path = Path(run["paths"]["summary"])
        if not path.exists():
            raise ValueError(f"Summary is not available for run {run.get('id')}")
        return json.loads(path.read_text(encoding="utf-8"))

    def artifact_path(self, run_id: str, name: str) -> Path:
        run = self.get_run(run_id)
        allowed = {
            "results",
            "metrics",
            "report",
            "summary",
            "log",
            "config",
            "input",
            "agent_input",
            "predictions",
        }
        if name not in allowed:
            raise ValueError("Unknown artifact")
        if run.get("benchmark") == "swebench" and name == "input":
            raise ValueError(
                "Raw SWE-bench evaluator input is protected; use agent_input instead"
            )
        if name not in run["paths"]:
            raise FileNotFoundError(name)
        path = Path(run["paths"][name]).resolve()
        run_dir = self._run_dir(run_id).resolve()
        if path != run_dir and run_dir not in path.parents:
            if name != "input":
                raise ValueError("Artifact escaped run directory")
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    async def clone_run(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        selection = run["selection"]
        spec = {
            "benchmark": run["benchmark"],
            "name": run["name"] + " · retry",
            "start_index": selection["start_index"],
            "limit": selection["sample_count"],
            "auto_advance": run["auto_advance"],
            **run["settings"],
        }
        return await self.create_run(spec)

    async def shutdown(self) -> None:
        for run_id, process in list(self.processes.items()):
            self.stop_requested.add(run_id)
            await asyncio.to_thread(self._terminate_tree, process.pid)
        for task in self.tasks.values():
            if not task.done():
                try:
                    await asyncio.wait_for(task, timeout=6)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    task.cancel()
