from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import dotenv_values, load_dotenv, set_key, unset_key


BASE_URL_FIELDS = {
    "AGENT_BASE_URL",
    "LLM_BASE_URL",
    "EMBEDDING_BASE_URL",
    "OPENAI_BASE_URL",
}


def normalize_openai_base_url(value: str) -> str:
    """Turn a raw OpenAI endpoint into an SDK base URL.

    Raw HTTP examples commonly show ``/chat/completions`` or ``/embeddings``.
    The OpenAI SDK appends those resource paths itself, so accepting the raw
    endpoint verbatim would produce duplicated paths.
    """

    normalized = str(value or "").strip().rstrip("/")
    lowered = normalized.lower()
    for suffix in ("/chat/completions", "/embeddings", "/responses"):
        if lowered.endswith(suffix):
            return normalized[: -len(suffix)].rstrip("/")
    return normalized


ENV_FIELDS: dict[str, dict[str, Any]] = {
    "AGENT_API_KEY": {"secret": True, "group": "agent"},
    "AGENT_BASE_URL": {"secret": False, "group": "agent"},
    "AGENT_MODEL": {"secret": False, "group": "agent"},
    "AGENT_WORK_DIR": {"secret": False, "group": "agent"},
    "AGENT_PROVIDER": {"secret": False, "group": "agent"},
    "WEB_LLM_MODE": {"secret": False, "group": "agent"},
    "AGENT_TEMPERATURE": {"secret": False, "group": "agent"},
    "LLM_API_KEY": {"secret": True, "group": "model"},
    "LLM_BASE_URL": {"secret": False, "group": "model"},
    "EMBEDDING_API_KEY": {"secret": True, "group": "embedding"},
    "EMBEDDING_BASE_URL": {"secret": False, "group": "embedding"},
    "OPENAI_API_KEY": {"secret": True, "group": "model"},
    "OPENAI_BASE_URL": {"secret": False, "group": "model"},
    "DATABASE_URL": {"secret": True, "group": "infrastructure"},
    "DSN": {"secret": True, "group": "infrastructure"},
    "QDRANT_URL": {"secret": False, "group": "infrastructure"},
    "QDRANT_PORT": {"secret": False, "group": "infrastructure"},
    "QDRANT_API_KEY": {"secret": True, "group": "infrastructure"},
    "NEMORI_POSTGRES_PORT": {"secret": False, "group": "infrastructure"},
    "NEMORI_QDRANT_HTTP_PORT": {"secret": False, "group": "infrastructure"},
    "NEMORI_QDRANT_GRPC_PORT": {"secret": False, "group": "infrastructure"},
    "NEMORI_OLLAMA_PORT": {"secret": False, "group": "infrastructure"},
    "NEMORI_BIND_ADDRESS": {"secret": False, "group": "infrastructure"},
    "NEMORI_DOCKER_SUBNET": {"secret": False, "group": "infrastructure"},
    "NO_PROXY": {"secret": False, "group": "infrastructure"},
    "NEMORI_SWE_TOKEN_BUDGET": {"secret": False, "group": "swe"},
    "NEMORI_SWE_TIME_BUDGET": {"secret": False, "group": "swe"},
    "NEMORI_SWE_MAX_TURNS": {"secret": False, "group": "swe"},
    "NEMORI_SWE_MAX_TOOL_CALLS": {"secret": False, "group": "swe"},
    "NEMORI_SWE_MEMORY_TIMEOUT": {"secret": False, "group": "swe"},
    "NEMORI_SWE_MAX_WORKERS": {"secret": False, "group": "swe"},
    "NEMORI_SWE_HARNESS_TIMEOUT": {"secret": False, "group": "swe"},
    "NEMORI_SWE_CACHE_LEVEL": {"secret": False, "group": "swe"},
    "NEMORI_SWE_CLEAN": {"secret": False, "group": "swe"},
    "LLM_JUDGE_MODEL": {"secret": False, "group": "defaults"},
    "NEMORI_LLM_MODEL": {"secret": False, "group": "defaults"},
    "NEMORI_EMBEDDING_MODEL": {"secret": False, "group": "defaults"},
    "NEMORI_JUDGE_MODEL": {"secret": False, "group": "defaults"},
}

DEFAULTS = {
    "AGENT_BASE_URL": "https://api.minimaxi.com/v1",
    "AGENT_MODEL": "MiniMax-M2.7",
    "AGENT_WORK_DIR": ".",
    "AGENT_PROVIDER": "minimax",
    "WEB_LLM_MODE": "llm",
    "AGENT_TEMPERATURE": "0.7",
    "LLM_BASE_URL": "https://openrouter.ai/api/v1",
    "EMBEDDING_BASE_URL": "https://openrouter.ai/api/v1",
    "QDRANT_URL": "localhost",
    "QDRANT_PORT": "16333",
    "NEMORI_POSTGRES_PORT": "15432",
    "NEMORI_QDRANT_HTTP_PORT": "16333",
    "NEMORI_QDRANT_GRPC_PORT": "16334",
    "NEMORI_OLLAMA_PORT": "21434",
    "NEMORI_BIND_ADDRESS": "127.0.0.1",
    "NEMORI_DOCKER_SUBNET": "10.231.0.0/24",
    "NO_PROXY": "localhost,127.0.0.1,::1",
    "NEMORI_SWE_TOKEN_BUDGET": "120000",
    "NEMORI_SWE_TIME_BUDGET": "1800",
    "NEMORI_SWE_MAX_TURNS": "80",
    "NEMORI_SWE_MAX_TOOL_CALLS": "200",
    "NEMORI_SWE_MEMORY_TIMEOUT": "120",
    "NEMORI_SWE_MAX_WORKERS": "1",
    "NEMORI_SWE_HARNESS_TIMEOUT": "1800",
    "NEMORI_SWE_CACHE_LEVEL": "env",
    "NEMORI_SWE_CLEAN": "true",
    "LLM_JUDGE_MODEL": "openai/gpt-4o-mini",
    "NEMORI_LLM_MODEL": "openai/gpt-4.1-mini",
    "NEMORI_EMBEDDING_MODEL": "google/gemini-embedding-001",
    "NEMORI_JUDGE_MODEL": "openai/gpt-4o-mini",
}


class EnvStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def _values(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        return {
            key: value
            for key, value in dotenv_values(self.path).items()
            if value is not None
        }

    def public_state(self) -> dict[str, Any]:
        values = self._values()
        fields: dict[str, Any] = {}
        for key, metadata in ENV_FIELDS.items():
            value = values.get(key, "")
            if metadata["secret"]:
                fields[key] = {
                    "configured": bool(value),
                    "secret": True,
                    "local_placeholder": value == "ollama",
                }
            else:
                rendered = value
                default = DEFAULTS.get(key, "")
                if key in BASE_URL_FIELDS:
                    rendered = normalize_openai_base_url(rendered)
                    default = normalize_openai_base_url(default)
                fields[key] = {
                    "configured": bool(value),
                    "secret": False,
                    "value": rendered,
                    "default": default,
                }
        return {
            "exists": self.path.exists(),
            "path": str(self.path),
            "fields": fields,
        }

    def update(self, values: dict[str, str], clear: list[str]) -> dict[str, Any]:
        unknown = (set(values) | set(clear)) - set(ENV_FIELDS)
        if unknown:
            raise ValueError(f"Unsupported environment keys: {', '.join(sorted(unknown))}")

        self.path.touch(exist_ok=True)
        for key in clear:
            unset_key(str(self.path), key)
            os.environ.pop(key, None)

        for key, raw_value in values.items():
            if raw_value is None or raw_value == "":
                continue
            value = str(raw_value).strip()
            if "\n" in value or "\r" in value:
                raise ValueError(f"{key} cannot contain line breaks")
            if key in BASE_URL_FIELDS:
                value = normalize_openai_base_url(value)
            set_key(str(self.path), key, value, quote_mode="always")
            os.environ[key] = value

        load_dotenv(self.path, override=True)
        return self.public_state()

    def subprocess_env(self) -> dict[str, str]:
        env = os.environ.copy()
        for key, value in self._values().items():
            env[key] = (
                normalize_openai_base_url(value)
                if key in BASE_URL_FIELDS
                else value
            )
        env.setdefault("NO_PROXY", DEFAULTS["NO_PROXY"])
        if not env.get("LLM_API_KEY") and env.get("AGENT_API_KEY"):
            env["LLM_API_KEY"] = env["AGENT_API_KEY"]
        if not env.get("LLM_BASE_URL") and env.get("AGENT_BASE_URL"):
            env["LLM_BASE_URL"] = env["AGENT_BASE_URL"]
        if not env.get("NEMORI_LLM_MODEL") and env.get("AGENT_MODEL"):
            env["NEMORI_LLM_MODEL"] = env["AGENT_MODEL"]
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        return env

    def defaults_for_run(self) -> dict[str, str]:
        values = self._values()
        return {
            "llm_model": values.get(
                "NEMORI_LLM_MODEL",
                values.get("AGENT_MODEL", DEFAULTS["NEMORI_LLM_MODEL"]),
            ),
            "embedding_model": values.get(
                "NEMORI_EMBEDDING_MODEL", DEFAULTS["NEMORI_EMBEDDING_MODEL"]
            ),
            "judge_model": values.get(
                "NEMORI_JUDGE_MODEL",
                values.get("LLM_JUDGE_MODEL", DEFAULTS["NEMORI_JUDGE_MODEL"]),
            ),
        }

    def secret_values(self) -> list[str]:
        values = self._values()
        return [
            value
            for key, value in values.items()
            if ENV_FIELDS.get(key, {}).get("secret") and len(value) >= 6
        ]
