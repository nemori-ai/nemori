"""Run leak-safe API and Nemori SWE-bench arms with mini-swe-agent 2.4.5.

This module is intentionally not part of Nemori's normal dependency set.  Run
it in an isolated ``uv`` environment, for example::

    uv run --python 3.12 --with mini-swe-agent==2.4.5 --no-project \
      python evaluation/swebench/agent_run.py \
      --input agent-input.jsonl --output output --model openai/my-model

The input must already be the sanitized Agent view.  Evaluator-only patches
and tests are rejected recursively before Docker or a model client is started.
The Docker container never receives a host mount or the Docker socket.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


MINISWEAGENT_VERSION = "2.4.5"
SAFE_INSTANCE_FIELDS = frozenset(
    {
        "instance_id",
        "repo",
        "base_commit",
        "problem_statement",
        "created_at",
        "version",
        "memory_context",
    }
)
FORBIDDEN_FIELDS = frozenset({"patch", "test_patch", "fail_to_pass", "pass_to_pass"})
REQUIRED_INSTANCE_FIELDS = frozenset(
    {"instance_id", "repo", "base_commit", "problem_statement"}
)
INSTANCE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+-[0-9]+$")
REPO_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
COMMIT_PATTERN = re.compile(r"^[0-9a-fA-F]{7,40}$")


class UnsafeAgentInput(ValueError):
    """Raised before execution when an agent input can contain answer data."""


class WallClockExceeded(TimeoutError):
    """Raised by the outer per-instance wall-clock guard."""


@dataclass(frozen=True)
class RunnerSettings:
    """Immutable settings for one leak-safe SWE Agent arm."""

    model: str
    arm: str = "nemori"
    api_base: str | None = None
    provider: str | None = None
    temperature: float = 0.0
    step_budget: int = 100
    tool_budget: int = 100
    token_budget: int = 100_000
    wall_seconds: int = 1_800
    command_seconds: int = 60
    model_timeout_seconds: int = 300
    max_output_tokens: int = 4_096
    cost_budget: float = 0.0
    docker_executable: str = "docker"
    image_namespace: str = "swebench"
    image_arch: str = "x86_64"

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must be non-empty")
        if self.arm not in {"api", "nemori"}:
            raise ValueError("arm must be api or nemori")
        for name in (
            "step_budget",
            "tool_budget",
            "token_budget",
            "wall_seconds",
            "command_seconds",
            "model_timeout_seconds",
            "max_output_tokens",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if self.cost_budget < 0:
            raise ValueError("cost_budget cannot be negative")


@dataclass(frozen=True)
class RuntimeBindings:
    """Lazily imported mini-swe-agent symbols used by the production runner."""

    default_agent: type
    docker_environment: type
    litellm_model: type
    interrupt_agent_flow: type[Exception]
    get_config_from_spec: Callable[[str], dict[str, Any]]


def atomic_json(path: Path, value: Any) -> None:
    """Atomically persist a JSON checkpoint."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f"{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink(missing_ok=True)


def _load_runtime() -> RuntimeBindings:
    """Import mini-swe-agent only when an actual Agent run begins."""

    try:
        from minisweagent import __version__ as minisweagent_version
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.config import get_config_from_spec
        from minisweagent.environments.docker import DockerEnvironment
        from minisweagent.exceptions import InterruptAgentFlow
        from minisweagent.models.litellm_model import LitellmModel
    except ImportError as exc:  # pragma: no cover - exercised by the CLI host
        raise RuntimeError(
            "mini-swe-agent is not installed. Run this module with "
            f"`uv run --with mini-swe-agent=={MINISWEAGENT_VERSION} --no-project`."
        ) from exc
    if minisweagent_version != MINISWEAGENT_VERSION:
        raise RuntimeError(
            "The SWE Agent runtime is version-pinned for reproducibility: "
            f"expected {MINISWEAGENT_VERSION}, got {minisweagent_version}."
        )
    return RuntimeBindings(
        default_agent=DefaultAgent,
        docker_environment=DockerEnvironment,
        litellm_model=LitellmModel,
        interrupt_agent_flow=InterruptAgentFlow,
        get_config_from_spec=get_config_from_spec,
    )


def _find_forbidden_keys(value: Any, path: str = "$") -> list[str]:
    findings: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            child_path = f"{path}.{key}"
            if key.casefold() in FORBIDDEN_FIELDS:
                findings.append(child_path)
            findings.extend(_find_forbidden_keys(child, child_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            findings.extend(_find_forbidden_keys(child, f"{path}[{index}]"))
    return findings


def _validate_memory_context(value: Any, *, instance_id: str) -> None:
    if isinstance(value, str):
        return
    if not isinstance(value, Mapping):
        raise UnsafeAgentInput(
            f"{instance_id}: memory_context must be a string or an object"
        )
    extra = set(value) - {
        "text",
        "retrievals",
        "backend",
        "search_method",
        "latency_ms",
        "memory_calls",
        "embedding_tokens",
    }
    if extra:
        raise UnsafeAgentInput(
            f"{instance_id}: unsupported memory_context keys: {sorted(extra)}"
        )
    if not isinstance(value.get("text", ""), str):
        raise UnsafeAgentInput(f"{instance_id}: memory_context.text must be a string")
    if not isinstance(value.get("retrievals", []), list):
        raise UnsafeAgentInput(
            f"{instance_id}: memory_context.retrievals must be a list"
        )
    if not isinstance(value.get("backend", ""), str) or not isinstance(
        value.get("search_method", ""), str
    ):
        raise UnsafeAgentInput(
            f"{instance_id}: memory retrieval backend metadata must be strings"
        )
    latency = value.get("latency_ms", 0)
    calls = value.get("memory_calls", 0)
    embedding_tokens = value.get("embedding_tokens", 0)
    if (
        isinstance(latency, bool)
        or not isinstance(latency, (int, float))
        or latency < 0
    ):
        raise UnsafeAgentInput(f"{instance_id}: latency_ms must be non-negative")
    if isinstance(calls, bool) or not isinstance(calls, int) or calls < 0:
        raise UnsafeAgentInput(f"{instance_id}: memory_calls must be non-negative")
    if (
        isinstance(embedding_tokens, bool)
        or not isinstance(embedding_tokens, int)
        or embedding_tokens < 0
    ):
        raise UnsafeAgentInput(f"{instance_id}: embedding_tokens must be non-negative")


def validate_safe_instance(raw: Any, *, index: int = 0) -> dict[str, Any]:
    """Validate one strictly allowlisted, answer-free Agent record."""

    if not isinstance(raw, Mapping):
        raise UnsafeAgentInput(f"record {index} must be a JSON object")
    findings = _find_forbidden_keys(raw)
    if findings:
        raise UnsafeAgentInput(
            "evaluator-only fields crossed the Agent boundary: " + ", ".join(findings)
        )
    keys = set(raw)
    extra = keys - SAFE_INSTANCE_FIELDS
    missing = REQUIRED_INSTANCE_FIELDS - keys
    if extra:
        raise UnsafeAgentInput(
            f"record {index} has unsupported fields: {sorted(extra)}"
        )
    if missing:
        raise UnsafeAgentInput(f"record {index} is missing fields: {sorted(missing)}")

    item = dict(raw)
    instance_id = item["instance_id"]
    repo = item["repo"]
    base_commit = item["base_commit"]
    problem = item["problem_statement"]
    if not isinstance(instance_id, str) or not INSTANCE_ID_PATTERN.fullmatch(
        instance_id
    ):
        raise UnsafeAgentInput(f"record {index} has an invalid instance_id")
    if not isinstance(repo, str) or not REPO_PATTERN.fullmatch(repo):
        raise UnsafeAgentInput(f"{instance_id}: repo must be owner/name")
    if not isinstance(base_commit, str) or not COMMIT_PATTERN.fullmatch(base_commit):
        raise UnsafeAgentInput(f"{instance_id}: base_commit must be a git hex id")
    if not isinstance(problem, str) or not problem.strip():
        raise UnsafeAgentInput(f"{instance_id}: problem_statement must be non-empty")
    for optional in ("created_at", "version"):
        if (
            optional in item
            and item[optional] is not None
            and not isinstance(item[optional], str)
        ):
            raise UnsafeAgentInput(
                f"{instance_id}: {optional} must be a string or null"
            )
    if "memory_context" in item:
        _validate_memory_context(item["memory_context"], instance_id=instance_id)

    # Prove the record is ordinary JSON, so trajectories cannot trigger custom
    # serializers or carry opaque handles across the boundary.
    try:
        return json.loads(json.dumps(item, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise UnsafeAgentInput(f"{instance_id}: record is not canonical JSON") from exc


def load_safe_instances(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSON list or JSONL file and reject duplicate/unsafe records."""

    input_path = Path(path)
    text = input_path.read_text(encoding="utf-8")
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        decoded = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                decoded.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise UnsafeAgentInput(
                    f"invalid JSONL on line {line_number}: {exc.msg}"
                ) from exc
    # A single JSON object is also one valid JSONL record.  Accept it while
    # still requiring arrays for multi-record ordinary JSON documents.
    if isinstance(decoded, Mapping):
        decoded = [decoded]
    if not isinstance(decoded, list) or not decoded:
        raise UnsafeAgentInput("agent input must be a non-empty JSON list or JSONL")
    instances = [
        validate_safe_instance(item, index=index) for index, item in enumerate(decoded)
    ]
    ids = [item["instance_id"] for item in instances]
    if len(ids) != len(set(ids)):
        raise UnsafeAgentInput("agent input contains duplicate instance_id values")
    return instances


def docker_image(instance_id: str, settings: RunnerSettings) -> str:
    """Return the official SWE-bench instance image name."""

    if not INSTANCE_ID_PATTERN.fullmatch(instance_id):
        raise UnsafeAgentInput("cannot build an image name from an invalid instance_id")
    encoded_id = instance_id.replace("__", "_1776_")
    return (
        f"docker.io/{settings.image_namespace}/sweb.eval."
        f"{settings.image_arch}.{encoded_id}:latest"
    )


def docker_run_args() -> list[str]:
    """Fixed security arguments; callers cannot add mounts or host networking."""

    return ["--rm", "--network", "none"]


def normalize_model_name(model: str, provider: str | None) -> str:
    """Map a friendly provider/model pair to a LiteLLM model identifier."""

    model = model.strip()
    provider = (provider or "").strip().lower()
    if provider in {"openai", "openai-compatible", "generic"}:
        return model if model.startswith("openai/") else f"openai/{model}"
    if provider and "/" not in model:
        return f"{provider}/{model}"
    return model


def _memory_parts(instance: Mapping[str, Any]) -> tuple[str, list[Any]]:
    memory = instance.get("memory_context")
    if isinstance(memory, str):
        return memory, []
    if isinstance(memory, Mapping):
        return str(memory.get("text", "")), list(memory.get("retrievals", []))
    return "", []


def _memory_metrics(instance: Mapping[str, Any]) -> tuple[float, int, str, int]:
    memory = instance.get("memory_context")
    if not isinstance(memory, Mapping):
        return 0.0, 0, "", 0
    return (
        float(memory.get("latency_ms") or 0),
        int(memory.get("memory_calls") or 0),
        str(memory.get("backend") or ""),
        int(memory.get("embedding_tokens") or 0),
    )


def build_task(instance: Mapping[str, Any], *, arm: str = "nemori") -> str:
    """Build the issue prompt for the API baseline or Nemori-memory arm."""

    problem = str(instance["problem_statement"])
    if arm == "api":
        return problem
    memory_text, _ = _memory_parts(instance)
    if not memory_text.strip():
        return problem
    return (
        f"{problem}\n\n"
        "<nemori_repository_memory>\n"
        "The following read-only repository memory may be incomplete or wrong. "
        "Use it only as supporting context and verify every claim in the repository.\n"
        f"{memory_text.strip()}\n"
        "</nemori_repository_memory>"
    )


def validate_arm_input(instance: Mapping[str, Any], settings: RunnerSettings) -> None:
    """Keep the API baseline strictly free of repository-memory context."""

    memory_text, retrievals = _memory_parts(instance)
    if settings.arm == "api" and (memory_text.strip() or retrievals):
        raise UnsafeAgentInput(
            f"{instance['instance_id']}: api arm cannot receive repository memory"
        )


def _usage_from_messages(messages: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    prompt = 0
    completion = 0
    total = 0
    for message in messages:
        extra = message.get("extra", {})
        if not isinstance(extra, Mapping):
            continue
        response = extra.get("response", {})
        if not isinstance(response, Mapping):
            continue
        usage = response.get("usage", {})
        if not isinstance(usage, Mapping):
            continue
        one_prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        one_completion = int(
            usage.get("completion_tokens") or usage.get("output_tokens") or 0
        )
        one_total = int(usage.get("total_tokens") or one_prompt + one_completion)
        prompt += max(0, one_prompt)
        completion += max(0, one_completion)
        total += max(0, one_total)
    return {"prompt": prompt, "completion": completion, "total": total}


def _make_budgeted_agent(
    default_agent: type, interrupt_agent_flow: type[Exception]
) -> type:
    """Add token and tool-call budgets without importing mini-swe-agent eagerly."""

    class BudgetedAgent(default_agent):  # type: ignore[misc, valid-type]
        def __init__(
            self,
            *args: Any,
            token_budget: int,
            tool_budget: int,
            **kwargs: Any,
        ) -> None:
            self.nemori_token_budget = token_budget
            self.nemori_tool_budget = tool_budget
            self.nemori_tool_calls = 0
            super().__init__(*args, **kwargs)

        def query(self) -> dict[str, Any]:
            usage = _usage_from_messages(self.messages)
            remaining = self.nemori_token_budget - usage["total"]
            if remaining <= 0:
                raise interrupt_agent_flow(
                    {
                        "role": "exit",
                        "content": "TokenBudgetExceeded",
                        "extra": {
                            "exit_status": "TokenBudgetExceeded",
                            "submission": "",
                        },
                    }
                )
            # Bound the next completion by the remaining experiment budget.
            # Provider-reported prompt usage is only known after the request,
            # so the post-call guard below remains authoritative.
            model_config = getattr(getattr(self, "model", None), "config", None)
            model_kwargs = getattr(model_config, "model_kwargs", None)
            if isinstance(model_kwargs, dict):
                configured = int(model_kwargs.get("max_tokens") or remaining)
                model_kwargs["max_tokens"] = max(1, min(configured, remaining))
            message = super().query()
            updated_usage = _usage_from_messages(self.messages)
            if updated_usage["total"] <= usage["total"]:
                raise interrupt_agent_flow(
                    {
                        "role": "exit",
                        "content": "TokenUsageUnavailable",
                        "extra": {
                            "exit_status": "TokenUsageUnavailable",
                            "submission": "",
                        },
                    }
                )
            if updated_usage["total"] > self.nemori_token_budget:
                raise interrupt_agent_flow(
                    {
                        "role": "exit",
                        "content": "TokenBudgetExceeded",
                        "extra": {
                            "exit_status": "TokenBudgetExceeded",
                            "submission": "",
                        },
                    }
                )
            return message

        def execute_actions(self, message: dict[str, Any]) -> list[dict[str, Any]]:
            actions = message.get("extra", {}).get("actions", [])
            if self.nemori_tool_calls + len(actions) > self.nemori_tool_budget:
                raise interrupt_agent_flow(
                    {
                        "role": "exit",
                        "content": "ToolCallBudgetExceeded",
                        "extra": {
                            "exit_status": "ToolCallBudgetExceeded",
                            "submission": "",
                        },
                    }
                )
            self.nemori_tool_calls += len(actions)
            return super().execute_actions(message)

    BudgetedAgent.__name__ = "NemoriBudgetedDefaultAgent"
    return BudgetedAgent


@contextmanager
def _wall_clock_limit(seconds: int):
    """Enforce a hard outer wall limit on Unix; mini also checks between calls."""

    alarm_supported = (
        hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer") and seconds > 0
    )
    previous_handler: Any = None
    if alarm_supported:
        previous_handler = signal.getsignal(signal.SIGALRM)

        def raise_timeout(_signum: int, _frame: Any) -> None:
            raise WallClockExceeded(f"instance exceeded {seconds} wall seconds")

        signal.signal(signal.SIGALRM, raise_timeout)
        signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        if alarm_supported:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)


def _force_remove_container(environment: Any) -> None:
    """Synchronously remove the exact container created for this instance."""

    container_id = getattr(environment, "container_id", None)
    config = getattr(environment, "config", None)
    executable = getattr(config, "executable", "docker")
    if container_id:
        try:
            subprocess.run(
                [executable, "rm", "-f", str(container_id)],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        finally:
            # Prevent mini-swe-agent's asynchronous destructor cleanup from
            # racing with a later container that happens to reuse a name/id.
            environment.container_id = None


def _empty_trial(
    instance: Mapping[str, Any], *, arm: str = "nemori", exit_status: str = "pending"
) -> dict[str, Any]:
    _, retrievals = _memory_parts(instance)
    (
        retrieval_latency_ms,
        memory_calls,
        retrieval_backend,
        embedding_tokens,
    ) = _memory_metrics(instance)
    patch = ""
    return {
        "instance_id": instance["instance_id"],
        "arm": arm,
        "status": "pending",
        "exit_status": exit_status,
        "duration_seconds": 0.0,
        "tokens": {"prompt": 0, "completion": 0, "total": 0},
        "model_calls": 0,
        "tool_calls": 0,
        "patch_sha256": hashlib.sha256(patch.encode("utf-8")).hexdigest(),
        "model_patch": patch,
        "retrievals": retrievals,
        "retrieval_latency_ms": retrieval_latency_ms,
        "memory_calls": memory_calls,
        "retrieval_backend": retrieval_backend,
        "embedding_tokens": embedding_tokens,
    }


def run_instance(
    instance: Mapping[str, Any],
    settings: RunnerSettings,
    trajectory_path: Path,
    *,
    runtime: RuntimeBindings | None = None,
) -> dict[str, Any]:
    """Run one Agent in one official image and always return a trial record."""

    instance = validate_safe_instance(instance)
    validate_arm_input(instance, settings)
    started = time.monotonic()
    environment: Any = None
    agent: Any = None
    patch = ""
    exit_status = "RunnerError"
    status = "agent_error"
    _, retrievals = _memory_parts(instance)
    (
        retrieval_latency_ms,
        memory_calls,
        retrieval_backend,
        embedding_tokens,
    ) = _memory_metrics(instance)
    # The experiment budget covers repository preparation, Agent work and
    # final patch collection—not only the model loop.  Container cleanup is
    # deliberately allowed to finish after the alarm so resources are not
    # leaked on timeout.
    wall_limit = _wall_clock_limit(settings.wall_seconds)
    wall_limit.__enter__()

    try:
        runtime = runtime or _load_runtime()
        official = runtime.get_config_from_spec("swebench")
        agent_config = dict(official["agent"])
        environment_config = dict(official["environment"])
        model_config = dict(official["model"])

        environment_config.pop("environment_class", None)
        environment_config.update(
            {
                "image": docker_image(str(instance["instance_id"]), settings),
                "cwd": "/testbed",
                "timeout": settings.command_seconds,
                "executable": settings.docker_executable,
                "run_args": docker_run_args(),
                "container_timeout": f"{settings.wall_seconds + 120}s",
                "forward_env": [],
            }
        )
        model_kwargs = dict(model_config.get("model_kwargs", {}))
        model_kwargs.update(
            {
                "temperature": settings.temperature,
                "timeout": min(settings.model_timeout_seconds, settings.wall_seconds),
                "max_tokens": min(settings.max_output_tokens, settings.token_budget),
                "drop_params": True,
                "parallel_tool_calls": False,
            }
        )
        if settings.api_base:
            model_kwargs["api_base"] = settings.api_base
        model_config.update(
            {
                "model_name": normalize_model_name(settings.model, settings.provider),
                "model_kwargs": model_kwargs,
                "cost_tracking": "ignore_errors",
            }
        )
        agent_config.update(
            {
                "step_limit": settings.step_budget,
                "cost_limit": settings.cost_budget,
                "wall_time_limit_seconds": settings.wall_seconds,
                "output_path": trajectory_path,
            }
        )

        environment = runtime.docker_environment(**environment_config)
        commit = shlex.quote(str(instance["base_commit"]))
        prepared = environment.execute(
            {
                "command": (
                    f'target="$(git rev-parse {commit}^{{commit}})" && '
                    'git checkout --detach -q "$target" && '
                    'git reset --hard -q "$target" && git clean -ffdx && '
                    "git remote | while IFS= read -r remote_name; do "
                    'git remote remove "$remote_name"; done && '
                    "git for-each-ref --format='%(refname)' | "
                    "while IFS= read -r ref; do "
                    'if case "$ref" in refs/tags/*) true;; *) false;; esac && '
                    'tag_commit="$(git rev-parse "$ref^{commit}" 2>/dev/null)" && '
                    'git merge-base --is-ancestor "$tag_commit" "$target"; then '
                    'continue; fi; git update-ref -d "$ref"; done && '
                    "git reflog expire --expire=now --all && "
                    "git gc --prune=now --quiet && "
                    'test "$(git rev-parse HEAD)" = "$target" && '
                    'test -z "$(git remote)" && '
                    "test -z \"$(git for-each-ref --format='%(refname)' | "
                    "grep -v '^refs/tags/' || true)\" && "
                    "git for-each-ref --format='%(refname)' refs/tags | "
                    "while IFS= read -r tag; do "
                    'git merge-base --is-ancestor "$tag^{commit}" "$target"; done && '
                    'test -z "$(git status --porcelain)" && '
                    'test -z "$(git fsck --unreachable --no-reflogs)"'
                )
            },
            timeout=settings.command_seconds,
        )
        if int(prepared.get("returncode", 1)) != 0:
            raise RuntimeError(
                "failed to reset the official image to base_commit: "
                + str(prepared.get("output", ""))[-1_000:]
            )

        model = runtime.litellm_model(**model_config)
        agent_class = _make_budgeted_agent(
            runtime.default_agent, runtime.interrupt_agent_flow
        )
        agent = agent_class(
            model,
            environment,
            token_budget=settings.token_budget,
            tool_budget=settings.tool_budget,
            **agent_config,
        )
        result = agent.run(build_task(instance, arm=settings.arm))
        exit_status = str(result.get("exit_status") or "Completed")

        # The only submitted patch is recomputed from the final workspace.
        # Model-authored submission/patch text is intentionally ignored.
        diff = environment.execute(
            {
                "command": (
                    # Intent-to-add makes untracked source files visible to
                    # git diff without staging their contents.  Submission
                    # artifacts are excluded and the candidate patch is
                    # always derived from the actual final workspace.
                    "git add -N -- . ':(exclude)patch.txt' && "
                    f"git diff --binary --full-index {commit} -- . "
                    "':(exclude)patch.txt'"
                )
            },
            timeout=settings.command_seconds,
        )
        if int(diff.get("returncode", 1)) != 0:
            raise RuntimeError(
                "failed to collect final workspace diff: "
                + str(diff.get("output", ""))[-1_000:]
            )
        patch = str(diff.get("output", ""))
        status = "generated" if patch.strip() else "empty_patch"
    except Exception as exc:
        exit_status = type(exc).__name__
        status = "agent_error"
        # Preserve the mini trajectory when available and add a small runner
        # error artifact otherwise.  Never include API secrets or raw input.
        if not trajectory_path.exists():
            atomic_json(
                trajectory_path,
                {
                    "instance_id": instance["instance_id"],
                    "exit_status": exit_status,
                    "error": str(exc),
                },
            )
    finally:
        wall_limit.__exit__(None, None, None)
        if environment is not None:
            _force_remove_container(environment)

    usage = _usage_from_messages(getattr(agent, "messages", []))
    duration = round(max(0.0, time.monotonic() - started), 6)
    return {
        "instance_id": instance["instance_id"],
        "arm": settings.arm,
        "status": status,
        "exit_status": exit_status,
        "duration_seconds": duration,
        "tokens": usage,
        "model_calls": int(getattr(agent, "n_calls", 0)),
        "tool_calls": int(getattr(agent, "nemori_tool_calls", 0)),
        "patch_sha256": hashlib.sha256(patch.encode("utf-8")).hexdigest(),
        "model_patch": patch,
        "retrievals": retrievals,
        "retrieval_latency_ms": retrieval_latency_ms,
        "memory_calls": memory_calls,
        "retrieval_backend": retrieval_backend,
        "embedding_tokens": embedding_tokens,
    }


def run_all(
    instances: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    settings: RunnerSettings,
    *,
    instance_runner: Callable[
        [Mapping[str, Any], RunnerSettings, Path], dict[str, Any]
    ] = run_instance,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run all inputs with checkpointed, exact prediction coverage."""

    instances = [
        validate_safe_instance(instance, index=index)
        for index, instance in enumerate(instances)
    ]
    for instance in instances:
        validate_arm_input(instance, settings)
    output_path = Path(output_dir)
    prediction_path = output_path / "predictions.json"
    trials_path = output_path / "trials.json"
    trajectory_dir = output_path / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)

    trials = [_empty_trial(instance, arm=settings.arm) for instance in instances]
    predictions = [
        {
            "instance_id": instance["instance_id"],
            "model_name_or_path": settings.model,
            "model_patch": "",
        }
        for instance in instances
    ]
    # Establish full denominator coverage before the first expensive action.
    atomic_json(prediction_path, predictions)
    atomic_json(trials_path, trials)

    for index, instance in enumerate(instances):
        safe_name = str(instance["instance_id"])
        trajectory = trajectory_dir / f"{safe_name}.json"
        try:
            trial = instance_runner(instance, settings, trajectory)
        except Exception as exc:
            trial = _empty_trial(
                instance, arm=settings.arm, exit_status=type(exc).__name__
            )
            trial.update(status="agent_error")
            atomic_json(
                trajectory,
                {
                    "instance_id": instance["instance_id"],
                    "exit_status": type(exc).__name__,
                    "error": str(exc),
                },
            )
        trials[index] = trial
        predictions[index] = {
            "instance_id": instance["instance_id"],
            "model_name_or_path": settings.model,
            "model_patch": str(trial.get("model_patch", "")),
        }
        atomic_json(prediction_path, predictions)
        atomic_json(trials_path, trials)

    return predictions, trials


def _environment_default(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("api", "nemori"), default="nemori")
    parser.add_argument(
        "--model",
        default=_environment_default("SWE_AGENT_MODEL", "AGENT_MODEL"),
        help="LiteLLM model id; defaults to SWE_AGENT_MODEL or AGENT_MODEL",
    )
    parser.add_argument(
        "--api-base",
        default=_environment_default("SWE_AGENT_API_BASE", "AGENT_BASE_URL"),
    )
    parser.add_argument(
        "--provider",
        default=_environment_default("SWE_AGENT_PROVIDER", "AGENT_PROVIDER"),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--step-budget", "--max-turns", dest="step_budget", type=int, default=100
    )
    parser.add_argument(
        "--tool-budget", "--max-tool-calls", dest="tool_budget", type=int, default=100
    )
    parser.add_argument("--token-budget", type=int, default=100_000)
    parser.add_argument(
        "--wall-seconds", "--time-budget", dest="wall_seconds", type=int, default=1_800
    )
    parser.add_argument(
        "--command-seconds",
        "--command-timeout",
        dest="command_seconds",
        type=int,
        default=60,
    )
    parser.add_argument("--model-timeout-seconds", type=int, default=300)
    parser.add_argument("--max-output-tokens", type=int, default=4_096)
    parser.add_argument("--cost-budget", type=float, default=0.0)
    parser.add_argument("--docker-executable", default="docker")
    parser.add_argument("--image-namespace", default="swebench")
    parser.add_argument("--image-arch", default="x86_64")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.model:
        raise SystemExit("--model is required (or set SWE_AGENT_MODEL / AGENT_MODEL)")
    # LiteLLM reads provider credentials from environment variables.  Copy the
    # existing dashboard alias without ever serializing the secret in config.
    if not os.getenv("OPENAI_API_KEY") and os.getenv("AGENT_API_KEY"):
        os.environ["OPENAI_API_KEY"] = str(os.environ["AGENT_API_KEY"])
    if args.provider and os.getenv("AGENT_API_KEY"):
        provider_key = re.sub(r"[^A-Za-z0-9]", "_", args.provider).upper()
        os.environ.setdefault(
            f"{provider_key}_API_KEY", str(os.environ["AGENT_API_KEY"])
        )
    settings = RunnerSettings(
        model=args.model,
        arm=args.arm,
        api_base=args.api_base,
        provider=args.provider,
        temperature=args.temperature,
        step_budget=args.step_budget,
        tool_budget=args.tool_budget,
        token_budget=args.token_budget,
        wall_seconds=args.wall_seconds,
        command_seconds=args.command_seconds,
        model_timeout_seconds=args.model_timeout_seconds,
        max_output_tokens=args.max_output_tokens,
        cost_budget=args.cost_budget,
        docker_executable=args.docker_executable,
        image_namespace=args.image_namespace,
        image_arch=args.image_arch,
    )
    instances = load_safe_instances(args.input)
    run_all(instances, args.output, settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
