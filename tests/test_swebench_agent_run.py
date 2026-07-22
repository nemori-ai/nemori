from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from evaluation.swebench import agent_run


SAFE = {
    "instance_id": "django__django-12345",
    "repo": "django/django",
    "base_commit": "abcdef1234567890",
    "problem_statement": "Fix the regression without changing tests.",
    "created_at": "2020-01-01T00:00:00Z",
    "version": "3.0",
}


def settings(**overrides: Any) -> agent_run.RunnerSettings:
    values = {"model": "test-model", "provider": "openai-compatible"}
    values.update(overrides)
    return agent_run.RunnerSettings(**values)


@pytest.mark.parametrize(
    "field", ["patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS"]
)
def test_safe_input_rejects_evaluator_fields_recursively(field: str) -> None:
    value = dict(SAFE)
    value["memory_context"] = {
        "text": "historical note",
        "retrievals": [{field: "secret"}],
    }

    with pytest.raises(agent_run.UnsafeAgentInput, match="evaluator-only"):
        agent_run.validate_safe_instance(value)


def test_safe_input_rejects_unknown_fields_and_duplicates(tmp_path: Path) -> None:
    with pytest.raises(agent_run.UnsafeAgentInput, match="unsupported fields"):
        agent_run.validate_safe_instance({**SAFE, "hints_text": "not allowlisted"})

    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        "\n".join(json.dumps(SAFE) for _ in range(2)), encoding="utf-8"
    )
    with pytest.raises(agent_run.UnsafeAgentInput, match="duplicate"):
        agent_run.load_safe_instances(input_path)


def test_jsonl_load_and_repository_memory_prompt(tmp_path: Path) -> None:
    first = dict(SAFE)
    first["memory_context"] = {
        "text": "Django migrations usually live under django/db/migrations.",
        "retrievals": [{"memory_id": "m-1", "score": 0.8}],
    }
    input_path = tmp_path / "safe.jsonl"
    input_path.write_text(json.dumps(first) + "\n", encoding="utf-8")

    loaded = agent_run.load_safe_instances(input_path)

    assert loaded == [first]
    task = agent_run.build_task(loaded[0])
    assert task.startswith(SAFE["problem_statement"])
    assert "<nemori_repository_memory>" in task
    assert "migrations usually live" in task
    assert agent_run.build_task(loaded[0], arm="api") == SAFE["problem_statement"]
    assert agent_run._empty_trial(first, arm="nemori")["retrievals"] == [
        {"memory_id": "m-1", "score": 0.8}
    ]
    agent_run.validate_arm_input(first, settings(arm="nemori"))
    agent_run.validate_arm_input(SAFE, settings(arm="api"))
    with pytest.raises(agent_run.UnsafeAgentInput, match="cannot receive"):
        agent_run.validate_arm_input(first, settings(arm="api"))
    with pytest.raises(ValueError, match="arm must be api or nemori"):
        settings(arm="baseline")


def test_docker_policy_is_fixed_to_official_image_without_mounts() -> None:
    run_args = agent_run.docker_run_args()

    assert run_args == ["--rm", "--network", "none"]
    assert not any(arg in {"-v", "--volume", "--mount"} for arg in run_args)
    assert "docker.sock" not in " ".join(run_args)
    assert agent_run.docker_image(SAFE["instance_id"], settings()) == (
        "docker.io/swebench/sweb.eval.x86_64.django_1776_django-12345:latest"
    )


class FakeInterrupt(Exception):
    def __init__(self, *messages: dict[str, Any]) -> None:
        super().__init__()
        self.messages = messages


class FakeDefaultAgent:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.messages: list[dict[str, Any]] = []
        self.n_calls = 0

    def query(self) -> dict[str, Any]:
        self.n_calls += 1
        message = {
            "role": "assistant",
            "extra": {
                "actions": [],
                "response": {
                    "usage": {
                        "prompt_tokens": 4,
                        "completion_tokens": 2,
                        "total_tokens": 6,
                    }
                },
            },
        }
        self.messages.append(message)
        return message

    def execute_actions(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        return [message]


def test_budgeted_agent_enforces_token_and_tool_call_budgets() -> None:
    budgeted = agent_run._make_budgeted_agent(FakeDefaultAgent, FakeInterrupt)
    token_agent = budgeted(None, None, token_budget=5, tool_budget=10)

    with pytest.raises(FakeInterrupt) as token_error:
        token_agent.query()
    assert (
        token_error.value.messages[0]["extra"]["exit_status"] == "TokenBudgetExceeded"
    )
    assert token_agent.n_calls == 1

    tool_agent = budgeted(None, None, token_budget=100, tool_budget=1)
    with pytest.raises(FakeInterrupt) as tool_error:
        tool_agent.execute_actions({"extra": {"actions": [{}, {}]}})
    assert (
        tool_error.value.messages[0]["extra"]["exit_status"] == "ToolCallBudgetExceeded"
    )
    assert tool_agent.nemori_tool_calls == 0


def test_run_all_keeps_an_empty_prediction_for_every_instance(tmp_path: Path) -> None:
    second = {
        **SAFE,
        "instance_id": "sympy__sympy-99999",
        "repo": "sympy/sympy",
    }

    def fake_runner(
        instance: dict[str, Any],
        _settings: agent_run.RunnerSettings,
        _trajectory: Path,
    ) -> dict[str, Any]:
        if instance["instance_id"].startswith("sympy"):
            raise RuntimeError("agent failed")
        trial = agent_run._empty_trial(instance, exit_status="Submitted")
        trial["status"] = "empty_patch"
        return trial

    predictions, trials = agent_run.run_all(
        [SAFE, second], tmp_path / "output", settings(), instance_runner=fake_runner
    )

    assert [item["instance_id"] for item in predictions] == [
        SAFE["instance_id"],
        second["instance_id"],
    ]
    assert [item["model_patch"] for item in predictions] == ["", ""]
    assert trials[1]["status"] == "agent_error"
    assert (
        json.loads((tmp_path / "output" / "predictions.json").read_text())
        == predictions
    )
    assert len(json.loads((tmp_path / "output" / "trials.json").read_text())) == 2


def test_run_instance_passes_security_and_budget_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    class FakeEnvironment:
        def __init__(self, **kwargs: Any) -> None:
            captured["environment"] = kwargs
            self.config = SimpleNamespace(executable=kwargs["executable"])
            self.container_id = "container-id"

        def execute(
            self, action: dict[str, Any], cwd: str = "", *, timeout: int | None = None
        ) -> dict[str, Any]:
            captured.setdefault("commands", []).append(action["command"])
            if action["command"].startswith("git add -N"):
                return {
                    "returncode": 0,
                    "output": "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n",
                }
            return {"returncode": 0, "output": "prepared"}

    class FakeModel:
        def __init__(self, **kwargs: Any) -> None:
            captured["model"] = kwargs

    class FakeRunnableAgent:
        def __init__(self, model: Any, env: Any, **kwargs: Any) -> None:
            captured["agent"] = kwargs
            self.messages = [
                {
                    "role": "assistant",
                    "extra": {
                        "response": {
                            "usage": {
                                "prompt_tokens": 10,
                                "completion_tokens": 5,
                                "total_tokens": 15,
                            }
                        }
                    },
                }
            ]
            self.n_calls = 1

        def run(self, task: str) -> dict[str, str]:
            captured["task"] = task
            captured["agent_instance"] = self
            return {"exit_status": "Submitted", "submission": "untrusted patch"}

    def config(_name: str) -> dict[str, Any]:
        return {
            "agent": {"system_template": "system", "instance_template": "{{task}}"},
            "environment": {"environment_class": "docker", "env": {}},
            "model": {"model_name": "old", "model_kwargs": {}},
        }

    runtime = agent_run.RuntimeBindings(
        default_agent=FakeRunnableAgent,
        docker_environment=FakeEnvironment,
        litellm_model=FakeModel,
        interrupt_agent_flow=FakeInterrupt,
        get_config_from_spec=config,
    )
    removed: list[list[str]] = []

    def fake_subprocess(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        removed.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(agent_run.subprocess, "run", fake_subprocess)
    trial = agent_run.run_instance(
        SAFE,
        settings(step_budget=7, tool_budget=8, token_budget=900, wall_seconds=30),
        tmp_path / "trajectory.json",
        runtime=runtime,
    )

    environment = captured["environment"]
    assert environment["run_args"] == ["--rm", "--network", "none"]
    assert environment["cwd"] == "/testbed"
    assert environment["forward_env"] == []
    assert not any("mount" in str(value).lower() for value in environment.values())
    assert captured["agent"]["step_limit"] == 7
    assert captured["agent_instance"].nemori_token_budget == 900
    assert captured["agent_instance"].nemori_tool_budget == 8
    assert captured["agent"]["wall_time_limit_seconds"] == 30
    assert captured["model"]["model_kwargs"]["temperature"] == 0.0
    assert "git checkout --detach" in captured["commands"][0]
    assert 'test "$(git rev-parse HEAD)" = "$target"' in captured["commands"][0]
    assert "git fsck --unreachable --no-reflogs" in captured["commands"][0]
    assert captured["commands"][-1].startswith("git add -N")
    assert "git diff --binary --full-index" in captured["commands"][-1]
    assert "':(exclude)patch.txt'" in captured["commands"][-1]
    assert trial["model_patch"].startswith("diff --git")
    assert trial["arm"] == "nemori"
    assert "untrusted patch" not in trial["model_patch"]
    assert trial["tokens"] == {"prompt": 10, "completion": 5, "total": 15}
    assert removed == [["docker", "rm", "-f", "container-id"]]


def test_cli_accepts_experiment_manager_budget_aliases() -> None:
    args = agent_run.build_parser().parse_args(
        [
            "--input",
            "safe.json",
            "--output",
            "out",
            "--model",
            "model",
            "--arm",
            "nemori",
            "--time-budget",
            "90",
            "--max-turns",
            "12",
            "--max-tool-calls",
            "18",
            "--command-timeout",
            "20",
        ]
    )

    assert args.arm == "nemori"
    assert args.wall_seconds == 90
    assert args.step_budget == 12
    assert args.tool_budget == 18
    assert args.command_seconds == 20

    api_args = agent_run.build_parser().parse_args(
        ["--input", "safe.json", "--output", "out", "--model", "model", "--arm", "api"]
    )
    assert api_args.arm == "api"
