from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import dashboard.run_manager as run_manager_module
import dashboard.app as dashboard_app
from dashboard.app import ConnectionTest, app, connection_test, start_run
from dashboard.env_store import EnvStore, normalize_openai_base_url
from dashboard.nemori_swe import (
    AGENT_UV_DEPENDENCIES,
    SWE_LITE_MAX_TASKS,
    NemoriSWEManager,
    _published_port,
    service_ports,
)
from dashboard.run_manager import RunManager, flatten_records
from evaluation.swebench.download import write_dataset_bundle


def test_swe_agent_dependencies_keep_centos_compatible_tiktoken() -> None:
    assert AGENT_UV_DEPENDENCIES == (
        "mini-swe-agent==2.4.5",
        "litellm==1.80.10",
        "tiktoken==0.11.0",
        "datasets==4.0.0",
        "numpy==1.26.4",
        "pandas==2.2.3",
        "pyarrow==20.0.0",
    )


def test_environment_store_never_returns_secret_values(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    store = EnvStore(tmp_path / ".env")
    state = store.update(
        {
            "LLM_API_KEY": "sk-local-secret-value",
            "LLM_BASE_URL": "https://example.test/v1",
        },
        [],
    )

    rendered = json.dumps(state)
    assert "sk-local-secret-value" not in rendered
    assert state["fields"]["LLM_API_KEY"] == {
        "configured": True,
        "secret": True,
        "local_placeholder": False,
    }
    assert state["fields"]["LLM_BASE_URL"]["value"] == "https://example.test/v1"
    assert store.subprocess_env()["LLM_API_KEY"] == "sk-local-secret-value"
    os.environ.pop("LLM_API_KEY", None)
    os.environ.pop("LLM_BASE_URL", None)


def test_agent_aliases_feed_nemori_llm_without_exposing_key(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("AGENT_API_KEY", raising=False)
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    store = EnvStore(tmp_path / ".env")
    state = store.update(
        {
            "AGENT_API_KEY": "new-agent-secret",
            "AGENT_BASE_URL": "https://agent.example/v1",
            "AGENT_MODEL": "agent-model",
        },
        [],
    )

    env = store.subprocess_env()
    assert env["LLM_API_KEY"] == "new-agent-secret"
    assert env["LLM_BASE_URL"] == "https://agent.example/v1"
    assert store.defaults_for_run()["llm_model"] == "agent-model"
    assert "new-agent-secret" not in json.dumps(state)
    for key in ("AGENT_API_KEY", "AGENT_BASE_URL", "AGENT_MODEL"):
        os.environ.pop(key, None)


def test_unconfigured_defaults_are_placeholders_not_saved_values(
    tmp_path: Path, monkeypatch
) -> None:
    for key in ("AGENT_API_KEY", "AGENT_BASE_URL", "AGENT_MODEL"):
        monkeypatch.delenv(key, raising=False)
    store = EnvStore(tmp_path / ".env")
    initial = store.public_state()

    assert initial["fields"]["LLM_BASE_URL"]["value"] == ""
    assert (
        initial["fields"]["LLM_BASE_URL"]["default"] == "https://openrouter.ai/api/v1"
    )

    store.update(
        {
            "AGENT_API_KEY": "agent-only-secret",
            "AGENT_BASE_URL": "https://agent.example/v1/chat/completions",
            "AGENT_MODEL": "agent/model",
        },
        [],
    )
    saved = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "LLM_BASE_URL" not in saved
    assert "EMBEDDING_BASE_URL" not in saved
    assert "AGENT_BASE_URL='https://agent.example/v1'" in saved
    for key in ("AGENT_API_KEY", "AGENT_BASE_URL", "AGENT_MODEL"):
        os.environ.pop(key, None)


def test_openai_sdk_base_urls_are_normalized() -> None:
    assert (
        normalize_openai_base_url("https://api.example/v1/chat/completions/")
        == "https://api.example/v1"
    )
    assert (
        normalize_openai_base_url("https://api.example/v1/embeddings")
        == "https://api.example/v1"
    )
    assert (
        normalize_openai_base_url("http://127.0.0.1:11434/v1/")
        == "http://127.0.0.1:11434/v1"
    )


def test_dashboard_entrypoint_loads_project_env_before_binding(monkeypatch) -> None:
    from dashboard import __main__ as dashboard_main

    captured = {}

    def fake_load_dotenv(path, *, override):
        captured["env_path"] = path
        captured["override"] = override
        monkeypatch.setenv("NEMORI_DASHBOARD_HOST", "127.0.0.1")
        monkeypatch.setenv("NEMORI_DASHBOARD_PORT", "18765")

    def fake_run(app_name, **kwargs):
        captured["app"] = app_name
        captured.update(kwargs)

    monkeypatch.setattr(dashboard_main, "load_dotenv", fake_load_dotenv)
    monkeypatch.setattr(dashboard_main.uvicorn, "run", fake_run)

    dashboard_main.main()

    assert captured["env_path"].name == ".env"
    assert captured["override"] is False
    assert captured["app"] == "dashboard.app:app"
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 18765


def test_llm_and_agent_connections_are_probed_independently(monkeypatch) -> None:
    captured: list[dict[str, str]] = []

    class FakeCompletions:
        def __init__(self, client: dict[str, str]) -> None:
            self.client = client

        async def create(self, **kwargs: object) -> SimpleNamespace:
            captured.append({**self.client, "model": str(kwargs["model"])})
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            client = {
                "api_key": str(kwargs["api_key"]),
                "base_url": str(kwargs["base_url"]),
            }
            self.chat = SimpleNamespace(completions=FakeCompletions(client))

    monkeypatch.setattr(dashboard_app, "AsyncOpenAI", FakeOpenAI)
    result = asyncio.run(
        connection_test(
            ConnectionTest(
                targets=["llm", "agent"],
                llm_model="llm/model",
                agent_model="agent/model",
                values={
                    "LLM_API_KEY": "llm-secret",
                    "LLM_BASE_URL": "https://llm.example/v1/chat/completions",
                    "AGENT_API_KEY": "agent-secret",
                    "AGENT_BASE_URL": "https://agent.example/v1/chat/completions",
                },
            )
        )
    )

    assert result["llm"]["base_url"] == "https://llm.example/v1"
    assert result["agent"]["base_url"] == "https://agent.example/v1"
    assert {item["model"]: item["base_url"] for item in captured} == {
        "llm/model": "https://llm.example/v1",
        "agent/model": "https://agent.example/v1",
    }


def test_standard_openai_credentials_are_valid_agent_and_embedding_fallbacks(
    monkeypatch,
) -> None:
    captured: list[tuple[str, str, str]] = []

    class FakeCompletions:
        async def create(self, **kwargs: object) -> SimpleNamespace:
            captured.append(("agent", "openai-secret", str(kwargs["model"])))
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))]
            )

    class FakeEmbeddings:
        async def create(self, **kwargs: object) -> SimpleNamespace:
            captured.append(("embedding", "openai-secret", str(kwargs["model"])))
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2])])

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["api_key"] == "openai-secret"
            assert str(kwargs["base_url"]) == "https://api.openai.test/v1"
            self.chat = SimpleNamespace(completions=FakeCompletions())
            self.embeddings = FakeEmbeddings()

    monkeypatch.setattr(dashboard_app, "AsyncOpenAI", FakeOpenAI)
    result = asyncio.run(
        connection_test(
            ConnectionTest(
                targets=["agent", "embedding"],
                agent_model="agent/model",
                embedding_model="embedding/model",
                values={
                    "OPENAI_API_KEY": "openai-secret",
                    "OPENAI_BASE_URL": "https://api.openai.test/v1",
                },
                clear=[
                    "AGENT_API_KEY",
                    "AGENT_BASE_URL",
                    "LLM_API_KEY",
                    "LLM_BASE_URL",
                    "EMBEDDING_API_KEY",
                    "EMBEDDING_BASE_URL",
                ],
            )
        )
    )

    assert result["agent"]["ok"] is True
    assert result["embedding"]["ok"] is True
    assert captured == [
        ("agent", "openai-secret", "agent/model"),
        ("embedding", "openai-secret", "embedding/model"),
    ]


def test_nemori_storage_ports_are_isolated_and_configurable() -> None:
    assert service_ports({}) == {
        "postgres": 15432,
        "qdrant_http": 16333,
        "qdrant_grpc": 16334,
    }
    assert service_ports(
        {
            "NEMORI_POSTGRES_PORT": "25432",
            "NEMORI_QDRANT_HTTP_PORT": "26333",
            "NEMORI_QDRANT_GRPC_PORT": "26334",
        }
    ) == {"postgres": 25432, "qdrant_http": 26333, "qdrant_grpc": 26334}


def test_compose_published_port_parser_handles_ipv4_and_ipv6() -> None:
    assert _published_port("0.0.0.0:16333\n[::]:16333\n") == 16333
    assert _published_port("") is None


def test_old_swe_experiment_routes_are_removed() -> None:
    with TestClient(app) as client:
        listed = client.get("/api/swebench/experiments")
        paired = client.post("/api/swebench/experiments/old/start")

    assert listed.status_code == 404
    assert paired.status_code == 404


def test_nemori_swe_has_one_public_experiment_path() -> None:
    with TestClient(app) as client:
        response = client.get("/api/nemori-swe")

    assert response.status_code == 200
    assert response.json()["name"] == "Nemori SWE repository-memory experiment"
    assert response.json()["official_swebench_score"] is False


def test_swe_public_path_accepts_api_and_nemori_arms(monkeypatch) -> None:
    calls: list[tuple[int, str]] = []

    class FakeSWE:
        async def shutdown(self) -> None:
            return None

        async def prepare(self, *, task_count: int, arm: str) -> dict[str, object]:
            calls.append((task_count, arm))
            return {"status": "preparing", "arm": arm}

        async def run_all(self, *, task_count: int, arm: str) -> dict[str, object]:
            calls.append((task_count, arm))
            return {"status": "running_all", "arm": arm}

    monkeypatch.setattr(dashboard_app, "nemori_swe", FakeSWE())
    with TestClient(app) as client:
        prepared = client.post(
            "/api/nemori-swe/prepare",
            json={"task_count": SWE_LITE_MAX_TASKS, "arm": "api"},
        )
        started = client.post(
            "/api/nemori-swe/run-all", json={"task_count": 1, "arm": "nemori"}
        )
        invalid = client.post(
            "/api/nemori-swe/run-all", json={"task_count": 1, "arm": "other"}
        )
        too_many = client.post(
            "/api/nemori-swe/run-all",
            json={"task_count": SWE_LITE_MAX_TASKS + 1, "arm": "api"},
        )

    assert prepared.json() == {"status": "preparing", "arm": "api"}
    assert started.json() == {"status": "running_all", "arm": "nemori"}
    assert invalid.status_code == 422
    assert too_many.status_code == 422
    assert calls == [(SWE_LITE_MAX_TASKS, "api"), (1, "nemori")]


def test_nemori_swe_history_can_be_inspected_and_deleted(tmp_path: Path) -> None:
    run_manager = RunManager(EnvStore(tmp_path / ".env"))
    swe = NemoriSWEManager(
        run_manager,
        dataset_path=tmp_path / "dataset.json",
        root=tmp_path / "swe-runs",
    )
    swe._new_state(SWE_LITE_MAX_TASKS, "api")
    assert swe.state is not None
    run_id = str(swe.state["id"])
    run_dir = swe._dir()

    detail = swe.get_run(run_id)
    assert detail["id"] == run_id
    assert detail["config"]["task_count"] == SWE_LITE_MAX_TASKS
    assert swe.history()[0]["id"] == run_id
    assert swe.history()[0]["sample_count"] == SWE_LITE_MAX_TASKS

    swe.delete_run(run_id)
    assert not run_dir.exists()
    assert swe.get()["status"] == "not_started"
    with pytest.raises(KeyError):
        swe.get_run(run_id)


def test_nemori_swe_active_history_cannot_be_deleted(tmp_path: Path) -> None:
    run_manager = RunManager(EnvStore(tmp_path / ".env"))
    swe = NemoriSWEManager(
        run_manager,
        dataset_path=tmp_path / "dataset.json",
        root=tmp_path / "swe-runs",
    )
    swe._new_state(1, "nemori")
    assert swe.state is not None
    run_id = str(swe.state["id"])
    swe.state["status"] = "running_all"
    swe._save()

    with pytest.raises(ValueError, match="Cannot delete an active"):
        swe.delete_run(run_id)
    assert swe._dir().is_dir()


def test_nemori_swe_history_detail_and_delete_routes(monkeypatch) -> None:
    deleted: list[str] = []

    class FakeSWE:
        async def shutdown(self) -> None:
            return None

        def get_run(self, run_id: str) -> dict[str, object]:
            if run_id == "missing":
                raise KeyError(run_id)
            return {"id": run_id, "status": "completed", "config": {"task_count": 8}}

        def delete_run(self, run_id: str) -> None:
            deleted.append(run_id)

    monkeypatch.setattr(dashboard_app, "nemori_swe", FakeSWE())
    with TestClient(app) as client:
        detail = client.get("/api/nemori-swe/history/swe-123")
        removed = client.delete("/api/nemori-swe/history/swe-123")
        missing = client.get("/api/nemori-swe/history/missing")

    assert detail.status_code == 200
    assert detail.json()["config"]["task_count"] == 8
    assert removed.json() == {"ok": True, "id": "swe-123"}
    assert deleted == ["swe-123"]
    assert missing.status_code == 404


def test_api_swe_arm_freezes_memoryless_agent_input(tmp_path: Path) -> None:
    run_manager = RunManager(EnvStore(tmp_path / ".env"))
    swe = NemoriSWEManager(
        run_manager,
        dataset_path=tmp_path / "dataset.json",
        root=tmp_path / "swe-runs",
    )
    swe._new_state(1, "api")
    safe_tasks = swe._dir() / "safe-tasks.json"
    rows = [
        {
            "instance_id": "owner__repo-1",
            "repo": "owner/repo",
            "base_commit": "abcdef1",
            "problem_statement": "Fix the bug.",
        }
    ]
    safe_tasks.write_text(json.dumps(rows), encoding="utf-8")
    assert swe.state is not None
    swe.state["artifacts"]["safe_tasks"] = str(safe_tasks)

    asyncio.run(swe._stage_memory())

    agent_rows = [
        json.loads(line)
        for line in Path(swe.state["artifacts"]["agent_input"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert agent_rows == rows
    assert "memory_context" not in agent_rows[0]
    assert swe.state["memory"]["enabled"] is False
    assert swe.history()[0]["arm"] == "api"


def test_start_preflight_blocks_process_when_exact_run_model_fails(monkeypatch) -> None:
    run = {
        "id": "failed-preflight",
        "status": "ready",
        "settings": {
            "llm_model": "run/llm",
            "embedding_model": "run/embedding",
            "judge_model": "run/llm",
        },
    }

    class FakeManager:
        started = False

        def get_run(self, _: str) -> dict[str, object]:
            return run

        def _save(self, _: dict[str, object]) -> None:
            return None

        async def start(self, _: str) -> dict[str, object]:
            self.started = True
            return run

    fake_manager = FakeManager()

    async def fake_connection_test(body: ConnectionTest) -> dict[str, object]:
        assert body.llm_model == "run/llm"
        assert body.embedding_model == "run/embedding"
        return {
            "llm": {
                "ok": False,
                "model": "run/llm",
                "base_url": "https://bad/v1",
                "error": "401",
            },
            "embedding": {
                "ok": True,
                "model": "run/embedding",
                "base_url": "http://local/v1",
            },
        }

    monkeypatch.setattr(dashboard_app, "manager", fake_manager)
    monkeypatch.setattr(dashboard_app, "connection_test", fake_connection_test)
    with pytest.raises(HTTPException, match="启动前连接检测未通过"):
        asyncio.run(start_run("failed-preflight"))

    assert fake_manager.started is False
    assert run["preflight"]["ok"] is False


def test_flatten_records_handles_grouped_evaluation_output() -> None:
    assert flatten_records(
        {"1": [{"question_id": "a"}], "2": [{"question_id": "b"}]}
    ) == [
        {"question_id": "a"},
        {"question_id": "b"},
    ]


def test_process_memory_guard_trips_only_on_process_virtual_memory(
    tmp_path: Path,
) -> None:
    manager = RunManager(EnvStore(tmp_path / ".env"))
    run = {"settings": {"max_process_vms_mb": 4096}}

    assert manager._process_exceeds_memory_guard(run, 4097) is True
    assert manager._process_exceeds_memory_guard(run, 512) is False


def test_create_run_uses_isolated_subset_and_config(
    tmp_path: Path, monkeypatch
) -> None:
    evaluation = tmp_path / "evaluation"
    dataset_dir = evaluation / "dataset"
    locomo_dir = evaluation / "locomo"
    longmem_dir = evaluation / "longmemeval"
    dataset_dir.mkdir(parents=True)
    locomo_dir.mkdir()
    longmem_dir.mkdir()
    (dataset_dir / "locomo10.json").write_text(
        json.dumps(
            [
                {"sample_id": "first", "qa": [{"category": 1}]},
                {"sample_id": "second", "qa": [{"category": 2}]},
            ]
        ),
        encoding="utf-8",
    )
    (locomo_dir / "config.json").write_text("{}", encoding="utf-8")
    (longmem_dir / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(run_manager_module, "EVALUATION_DIR", evaluation)
    monkeypatch.setattr(run_manager_module, "RUNS_DIR", tmp_path / "runtime" / "runs")
    manager = RunManager(EnvStore(tmp_path / ".env"))

    run = asyncio.run(
        manager.create_run(
            {
                "benchmark": "locomo",
                "start_index": 1,
                "limit": 1,
                "llm_model": "test-llm",
                "embedding_model": "test-embedding",
            }
        )
    )

    assert run["status"] == "ready"
    assert run["selection"]["sample_count"] == 1
    assert run["settings"]["llm_model"] == "test-llm"
    saved_config = json.loads(Path(run["paths"]["config"]).read_text(encoding="utf-8"))
    assert saved_config["agent_id"].startswith("dashboard_")
    assert saved_config["qdrant_collection_prefix"].startswith("nemori_")
    assert saved_config["observation_window"] == 20
    assert saved_config["llm_timeout"] == 120
    assert saved_config["semantic_context_top_k"] == 10
    assert saved_config["semantic_context_similarity_threshold"] == 0.70
    assert saved_config["semantic_consolidation_top_k"] == 5
    assert run["settings"]["paper_profile"] is True
    subset = json.loads(Path(run["paths"]["input"]).read_text(encoding="utf-8"))
    assert subset == [{"sample_id": "second", "qa": [{"category": 2}]}]
    assert Path(run["paths"]["log"]).exists()
    workflow = manager.workflow(run["id"])
    assert (
        workflow["definitions"]["ingest"]["parameters"]["embedding_model"]
        == "test-embedding"
    )
    assert any(
        node["id"] == "narrative" for node in workflow["definitions"]["ingest"]["nodes"]
    )

    asyncio.run(
        manager._handle_output_line(
            run["id"],
            {"id": "ingest"},
            'NEMORI_TRACE {"flow":"ingest","nodes":["source","narrative"],"summary":"trace ok"}',
        )
    )
    assert manager.get_run(run["id"])["workflow"]["nodes"]["narrative"]["count"] == 1

    for line in (
        "Total LLM calls:        353",
        "Total tokens:           357,314",
        "  Prompt tokens:        322267",
        "  Completion tokens:    35047",
        "Total errors:           29",
        "  semantic_consolidation     calls= 161  tokens=126,028",
    ):
        asyncio.run(manager._handle_output_line(run["id"], {"id": "ingest"}, line))
    usage = manager.get_run(run["id"])["token_usage"]
    assert usage["total_tokens"] == 357314
    assert usage["prompt_tokens"] == 322267
    assert usage["completion_tokens"] == 35047
    assert usage["requests"] == 353
    assert usage["errors"] == 29
    assert usage["phases"]["ingest"]["operations"]["semantic_consolidation"] == {
        "requests": 161,
        "tokens": 126028,
    }

    asyncio.run(
        manager._record_trace(
            run["id"],
            {
                "event": "ingest_complete",
                "nodes": ["semantic_store"],
                "episodes": [{"fallback": True}, {"fallback": True}],
                "semantic_memories": [],
                "compression_ratio": 1.2,
            },
        )
    )
    assert manager.get_run(run["id"])["ingest_quality"] == {
        "episodes": 2,
        "fallback_episodes": 2,
        "semantic_memories": 0,
        "compression_ratio": 1.2,
        "all_fallback": True,
        "quality_issues": {
            "segmentation": 0,
            "episode_generation": 0,
            "semantic_generation": 0,
            "semantic_consolidation": 0,
            "episode_integration": 0,
        },
    }


def test_create_direct_baseline_run_skips_ingest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluation = tmp_path / "evaluation"
    dataset_dir = evaluation / "dataset"
    locomo_dir = evaluation / "locomo"
    longmem_dir = evaluation / "longmemeval"
    dataset_dir.mkdir(parents=True)
    locomo_dir.mkdir()
    longmem_dir.mkdir()
    (dataset_dir / "locomo10.json").write_text(
        json.dumps([{"sample_id": "first", "qa": [{"category": 1}]}]),
        encoding="utf-8",
    )
    (locomo_dir / "config.json").write_text("{}", encoding="utf-8")
    (longmem_dir / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(run_manager_module, "EVALUATION_DIR", evaluation)
    monkeypatch.setattr(run_manager_module, "RUNS_DIR", tmp_path / "runtime" / "runs")
    manager = RunManager(EnvStore(tmp_path / ".env"))

    run = asyncio.run(
        manager.create_run(
            {
                "benchmark": "locomo",
                "limit": 1,
                "run_mode": "direct",
            }
        )
    )

    assert run["settings"]["run_mode"] == "direct"
    assert [phase["id"] for phase in run["phases"]] == [
        "search",
        "evaluate",
        "aggregate",
    ]
    command = manager._phase_command(run, "search")
    assert "baseline_direct.py" in " ".join(command)
    assert "--scored-only" in command


def test_create_swebench_run_uses_wsl_harness_and_isolated_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluation = tmp_path / "evaluation"
    dataset_dir = evaluation / "dataset"
    swebench_dir = evaluation / "swebench"
    dataset_dir.mkdir(parents=True)
    swebench_dir.mkdir()
    write_dataset_bundle(
        dataset_dir / "swebench_lite_test.json",
        [
            {
                "repo": "owner/first",
                "instance_id": "owner__first-1",
                "base_commit": "abc",
                "problem_statement": "first issue",
                "patch": "diff --git a/a b/a",
                "test_patch": "diff --git a/test_a b/test_a",
                "FAIL_TO_PASS": "[]",
                "PASS_TO_PASS": "[]",
                "created_at": "2024-01-01T00:00:00Z",
            },
            {
                "repo": "owner/second",
                "instance_id": "owner__second-2",
                "base_commit": "def",
                "problem_statement": "second issue",
                "patch": "diff --git a/b b/b",
                "test_patch": "diff --git a/test_b b/test_b",
                "FAIL_TO_PASS": "[]",
                "PASS_TO_PASS": "[]",
                "created_at": "2024-01-02T00:00:00Z",
            },
        ],
        dataset="princeton-nlp/SWE-bench_Lite",
        config="default",
        split="test",
        requested_revision="main",
        resolved_revision="a" * 40,
    )
    (swebench_dir / "config.json").write_text(
        json.dumps(
            {
                "dataset_name": "princeton-nlp/SWE-bench_Lite",
                "split": "test",
                "swebench_version": "4.1.0",
                "wsl_distribution": "Ubuntu",
                "docker_namespace": "swebench",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(run_manager_module, "EVALUATION_DIR", evaluation)
    monkeypatch.setattr(run_manager_module, "RUNS_DIR", tmp_path / "runtime" / "runs")
    monkeypatch.setattr(run_manager_module, "_is_windows_host", lambda: True)
    manager = RunManager(EnvStore(tmp_path / ".env"))

    run = asyncio.run(
        manager.create_run(
            {
                "benchmark": "swebench",
                "start_index": 1,
                "limit": 1,
                "swe_predictions_path": "gold",
                "swe_max_workers": 1,
                "swe_timeout": 900,
            }
        )
    )

    assert run["settings"]["run_mode"] == "harness"
    assert run["settings"]["paper_profile"] is False
    assert [phase["id"] for phase in run["phases"]] == ["evaluate", "aggregate"]
    subset = json.loads(Path(run["paths"]["input"]).read_text(encoding="utf-8"))
    assert [item["instance_id"] for item in subset] == ["owner__second-2"]
    safe_subset = json.loads(
        Path(run["paths"]["agent_input"]).read_text(encoding="utf-8")
    )
    assert [item["instance_id"] for item in safe_subset] == ["owner__second-2"]
    assert "patch" not in safe_subset[0]
    assert manager.artifact_path(run["id"], "agent_input") == Path(
        run["paths"]["agent_input"]
    )
    with pytest.raises(ValueError, match="protected"):
        manager.artifact_path(run["id"], "input")
    command = manager._phase_command(run, "evaluate")
    assert command[:3] == ["wsl.exe", "-d", "Ubuntu"]
    assert "swebench==4.1.0" in " ".join(command)
    assert "../../../evaluation/swebench/run.py" in " ".join(command)
    assert run["paper_alignment"]["score_comparable"] is False
    workflow = manager.workflow(run["id"])
    assert workflow["definitions"]["ingest"]["tab_title"] == "环境构建"
    assert any(
        node["id"] == "tests" for node in workflow["definitions"]["search"]["nodes"]
    )
    Path(run["paths"]["input"]).write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="evaluator input changed"):
        asyncio.run(manager.start(run["id"]))


def test_swebench_summary_reports_resolution_rate(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    results = tmp_path / "results.json"
    report.write_text(
        json.dumps(
            {
                "total_instances": 2,
                "submitted_instances": 2,
                "completed_instances": 2,
                "resolved_instances": 1,
                "error_instances": 0,
            }
        ),
        encoding="utf-8",
    )
    results.write_text(
        json.dumps(
            [
                {"category": "owner/repo", "resolved": True},
                {"category": "owner/repo", "resolved": False},
            ]
        ),
        encoding="utf-8",
    )
    summary = RunManager._build_summary(
        {
            "benchmark": "swebench",
            "paths": {"report": str(report), "results": str(results)},
        }
    )

    assert summary["overall"] == {
        "count": 2,
        "submitted": 2,
        "completed": 2,
        "resolved": 1,
        "accuracy": 0.5,
    }
    assert summary["by_category"]["owner/repo"] == {"count": 2, "resolved": 1}


def test_api_can_create_paired_direct_run_update_notes_and_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluation = tmp_path / "evaluation"
    dataset_dir = evaluation / "dataset"
    locomo_dir = evaluation / "locomo"
    longmem_dir = evaluation / "longmemeval"
    dataset_dir.mkdir(parents=True)
    locomo_dir.mkdir()
    longmem_dir.mkdir()
    (dataset_dir / "locomo10.json").write_text(
        json.dumps([{"sample_id": "first", "qa": [{"category": 1}]}]),
        encoding="utf-8",
    )
    (locomo_dir / "config.json").write_text("{}", encoding="utf-8")
    (longmem_dir / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(run_manager_module, "EVALUATION_DIR", evaluation)
    monkeypatch.setattr(run_manager_module, "RUNS_DIR", tmp_path / "runtime" / "runs")
    manager = RunManager(EnvStore(tmp_path / ".env"))
    monkeypatch.setattr(dashboard_app, "manager", manager)

    with TestClient(app) as client:
        created = client.post(
            "/api/runs",
            json={
                "benchmark": "locomo",
                "limit": 1,
                "notes": "compare semantic failures",
                "create_direct_pair": True,
            },
        )
        assert created.status_code == 201
        body = created.json()
        paired = body["paired_run"]
        assert body["notes"] == "compare semantic failures"
        assert body["related_runs"]["direct_baseline"] == paired["id"]
        assert paired["settings"]["run_mode"] == "direct"
        assert paired["related_runs"]["source"] == body["id"]

        updated = client.patch(
            f"/api/runs/{body['id']}/notes",
            json={"notes": "parser fixed, retry next"},
        )
        assert updated.status_code == 200
        assert updated.json()["notes"] == "parser fixed, retry next"

        paired_dir = Path(paired["paths"]["run_dir"])
        deleted = client.delete(f"/api/runs/{paired['id']}")
        assert deleted.status_code == 200
        assert not paired_dir.exists()
        assert "direct_baseline" not in manager.get_run(body["id"])["related_runs"]


def test_compare_run_summaries(tmp_path: Path) -> None:
    manager = RunManager(EnvStore(tmp_path / ".env"))
    manager.runs["nemori"] = {
        "id": "nemori",
        "name": "Nemori",
        "settings": {"run_mode": "nemori"},
        "summary": {"overall": {"accuracy": 0.75, "f1": 0.5}, "by_category": {}},
        "paths": {"summary": str(tmp_path / "missing-a.json")},
    }
    manager.runs["direct"] = {
        "id": "direct",
        "name": "Direct",
        "settings": {"run_mode": "direct"},
        "summary": {"overall": {"accuracy": 0.5, "f1": 0.4}, "by_category": {}},
        "paths": {"summary": str(tmp_path / "missing-b.json")},
    }

    comparison = manager.compare("nemori", "direct")

    assert comparison["overall"]["delta"]["accuracy"] == pytest.approx(0.25)
    assert comparison["run"]["mode"] == "nemori"
    assert comparison["against"]["mode"] == "direct"


def test_unsaved_ollama_style_embedding_values_can_be_probed(monkeypatch) -> None:
    class FakeEmbeddings:
        async def create(self, **_: object) -> SimpleNamespace:
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])])

    class FakeOpenAI:
        def __init__(self, **_: object) -> None:
            self.embeddings = FakeEmbeddings()

    monkeypatch.setattr(dashboard_app, "AsyncOpenAI", FakeOpenAI)
    result = asyncio.run(
        connection_test(
            ConnectionTest(
                targets=["embedding"],
                embedding_model="qwen3-embedding:0.6b",
                values={
                    "EMBEDDING_API_KEY": "ollama",
                    "EMBEDDING_BASE_URL": "http://127.0.0.1:11434/v1",
                },
            )
        )
    )
    assert result["embedding"] == {
        "source": "EMBEDDING_*",
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "qwen3-embedding:0.6b",
        "ok": True,
        "dimension": 3,
        "latency_ms": result["embedding"]["latency_ms"],
    }


def test_dashboard_health_and_static_page() -> None:
    with TestClient(app) as client:
        health = client.get("/api/health")
        page = client.get("/")
        script = client.get("/static/app.js")

    assert health.status_code == 200
    assert health.json()["service"] == "nemori-experiment-lab"
    assert page.status_code == 200
    assert "Nemori Experiment Lab" in page.text
    assert "运行完整 Nemori + SWE" in page.text
    assert "API + SWE" in page.text
    assert "NEMORI_SWE_CACHE_LEVEL" in page.text
    assert "LoCoMo" in page.text
    assert "环境与连接" in page.text
    assert "内部工作流" in page.text
    assert "Qdrant HTTP" in page.text
    assert "16333" in page.text
    assert "本地 Qdrant 502 与代理隔离" in page.text
    assert "完整实验控制室重新接回" in page.text
    assert "Gold Validation" not in page.text
    assert script.status_code == 200
    assert "/api/nemori-swe/run-all" in script.text
    assert "/api/system/start" in script.text
    assert "/api/environment/test" in script.text
    assert "/workflow" in script.text
    assert "/compare/" in script.text


def test_start_local_environment_endpoint(monkeypatch) -> None:
    async def fake_start(body: dashboard_app.SystemStart) -> dict[str, object]:
        assert body.services == ["ollama"]
        return {
            "ok": True,
            "steps": [{"service": "ollama", "action": "already running", "ok": True}],
            "status": {"ollama": {"ok": True}},
        }

    monkeypatch.setattr(dashboard_app, "start_local_environment", fake_start)
    with TestClient(app) as client:
        response = client.post("/api/system/start", json={"services": ["ollama"]})

    assert response.status_code == 200
    assert response.json()["steps"][0]["service"] == "ollama"
