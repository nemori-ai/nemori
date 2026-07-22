"""Safety and data-boundary contracts for the Nemori SWE experiment.

This module deliberately separates the data an agent may see from the raw
record required by the official evaluator.  The separation is structural: an
``AgentInstance`` cannot store evaluator-only answer or test fields.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


FORBIDDEN_AGENT_FIELDS = frozenset(
    {"patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS"}
)
FORBIDDEN_API_FIELDS = frozenset({"patch", "test_patch"})
AGENT_INSTANCE_FIELDS = frozenset(
    {
        "instance_id",
        "repo",
        "base_commit",
        "problem_statement",
        "created_at",
        "version",
    }
)
EVALUATOR_REQUIRED_FIELDS = frozenset(
    {
        "instance_id",
        "repo",
        "base_commit",
        "problem_statement",
        *FORBIDDEN_AGENT_FIELDS,
    }
)


class ForbiddenFieldError(ValueError):
    """Raised when evaluator-only fields cross an agent or API boundary."""


class PredictionCoverageError(ValueError):
    """Raised when predictions do not cover the selected instances exactly."""


def _canonical_json(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("The value must be canonical JSON data") from exc
    return rendered.encode("utf-8")


def stable_sha256(value: Any) -> str:
    """Hash JSON-compatible data independently of dictionary key order."""

    return hashlib.sha256(_canonical_json(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a dataset artifact without loading it all."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_forbidden_fields(
    value: Any,
    *,
    forbidden_fields: Iterable[str] = FORBIDDEN_AGENT_FIELDS,
) -> tuple[str, ...]:
    """Return JSON-style paths of forbidden keys found recursively.

    Matching is case-insensitive but exact, so the official ``model_patch``
    prediction field and the guide's ``agent_output_diff`` remain valid.
    """

    forbidden = {str(item).casefold() for item in forbidden_fields}
    findings: list[str] = []

    def walk(item: Any, path: str) -> None:
        if isinstance(item, BaseModel):
            walk(item.model_dump(mode="python"), path)
            return
        if isinstance(item, Mapping):
            for key, child in item.items():
                rendered_key = str(key)
                child_path = f"{path}.{rendered_key}"
                if rendered_key.casefold() in forbidden:
                    findings.append(child_path)
                walk(child, child_path)
            return
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for index, child in enumerate(item):
                walk(child, f"{path}[{index}]")

    walk(value, "$")
    return tuple(findings)


def assert_no_forbidden_fields(
    value: Any,
    *,
    forbidden_fields: Iterable[str] = FORBIDDEN_AGENT_FIELDS,
) -> None:
    """Fail closed if evaluator-only keys appear in a public payload."""

    findings = find_forbidden_fields(value, forbidden_fields=forbidden_fields)
    if findings:
        raise ForbiddenFieldError(
            "Evaluator-only fields crossed a safe boundary: " + ", ".join(findings)
        )


class AgentInstance(BaseModel):
    """Allowlisted SWE-bench input visible to a Coding Agent."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    instance_id: str = Field(min_length=1)
    repo: str = Field(min_length=1)
    base_commit: str = Field(min_length=1)
    problem_statement: str = Field(min_length=1)
    created_at: str | None = None
    version: str | None = None

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any]) -> "AgentInstance":
        """Copy only allowlisted fields from a raw evaluator record."""

        selected = {key: raw[key] for key in AGENT_INSTANCE_FIELDS if key in raw}
        instance = cls.model_validate(selected)
        assert_no_forbidden_fields(instance)
        return instance

    def to_agent_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json", exclude_none=True)
        assert_no_forbidden_fields(payload)
        return payload


def sanitize_agent_instance(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Return the JSON-ready, strictly allowlisted view used by an agent."""

    return AgentInstance.from_raw(raw).to_agent_payload()


class EvaluatorInstance:
    """Opaque holder for a full record used only by the official Harness.

    There is intentionally no generic serialization method.  Callers must
    explicitly request either the safe agent view or the evaluator record.
    The latter must never be mounted in an agent sandbox or returned by an API.
    """

    __slots__ = ("_record",)

    def __init__(self, raw: Mapping[str, Any]) -> None:
        missing = sorted(EVALUATOR_REQUIRED_FIELDS - set(raw))
        if missing:
            raise ValueError(
                "SWE-bench evaluator record is missing fields: " + ", ".join(missing)
            )
        instance_id = raw.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id.strip():
            raise ValueError("Evaluator instance_id must be a non-empty string")
        self._record = copy.deepcopy(dict(raw))

    @property
    def instance_id(self) -> str:
        return str(self._record["instance_id"])

    def to_agent_instance(self) -> AgentInstance:
        return AgentInstance.from_raw(self._record)

    def to_harness_record(self) -> dict[str, Any]:
        """Return a defensive copy for the isolated evaluator process only."""

        return copy.deepcopy(self._record)

    def __repr__(self) -> str:
        return (
            f"EvaluatorInstance(instance_id={self.instance_id!r}, record=<restricted>)"
        )


class SelectionManifest(BaseModel):
    """Immutable identity of a fair SWE-bench experiment selection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    dataset_name: str = Field(min_length=1)
    source_split: str = Field(min_length=1)
    dataset_revision: str | None = None
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    instance_ids: tuple[str, ...]
    fairness_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("instance_ids")
    @classmethod
    def validate_instance_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("Selection manifest must contain at least one instance")
        if any(not item.strip() for item in value):
            raise ValueError("Selection manifest contains an empty instance_id")
        if len(value) != len(set(value)):
            raise ValueError("Selection manifest contains duplicate instance_ids")
        return value


def create_selection_manifest(
    *,
    dataset_path: str | Path,
    instance_ids: Iterable[str],
    fairness_config: Mapping[str, Any],
    dataset_name: str = "princeton-nlp/SWE-bench_Lite",
    source_split: str = "test",
    dataset_revision: str | None = None,
) -> SelectionManifest:
    """Create a manifest pinning data bytes, exact IDs and fair-arm settings."""

    return SelectionManifest(
        dataset_name=dataset_name,
        source_split=source_split,
        dataset_revision=dataset_revision,
        dataset_sha256=sha256_file(dataset_path),
        instance_ids=tuple(str(item) for item in instance_ids),
        fairness_hash=stable_sha256(dict(fairness_config)),
    )


def build_selection_manifest(
    rows: Iterable[Mapping[str, Any]],
    source_path: str | Path,
    *,
    fairness_config: Mapping[str, Any],
    dataset_name: str = "princeton-nlp/SWE-bench_Lite",
    source_split: str = "test",
    dataset_revision: str | None = None,
) -> SelectionManifest:
    """Build a manifest from explicit rows while hashing the source artifact."""

    instance_ids: list[str] = []
    for index, row in enumerate(rows):
        instance_id = row.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id.strip():
            raise ValueError(f"Dataset row {index} has no non-empty instance_id")
        instance_ids.append(instance_id)
    return create_selection_manifest(
        dataset_path=source_path,
        instance_ids=instance_ids,
        fairness_config=fairness_config,
        dataset_name=dataset_name,
        source_split=source_split,
        dataset_revision=dataset_revision,
    )


class EvaluatorSelection:
    """Pair an immutable selection manifest with restricted evaluator data."""

    __slots__ = ("manifest", "_instances")

    def __init__(
        self,
        manifest: SelectionManifest,
        instances: Iterable[EvaluatorInstance],
    ) -> None:
        materialized = tuple(instances)
        actual_ids = tuple(item.instance_id for item in materialized)
        if actual_ids != manifest.instance_ids:
            raise ValueError(
                "Evaluator records must match manifest instance_ids in the same order"
            )
        self.manifest = manifest
        self._instances = materialized

    def agent_instances(self) -> tuple[AgentInstance, ...]:
        return tuple(item.to_agent_instance() for item in self._instances)

    def harness_records(self) -> list[dict[str, Any]]:
        return [item.to_harness_record() for item in self._instances]


class PredictionCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    expected_count: int = Field(ge=0)
    prediction_count: int = Field(ge=0)
    covered_count: int = Field(ge=0)
    missing_ids: tuple[str, ...] = ()
    unexpected_ids: tuple[str, ...] = ()
    duplicate_ids: tuple[str, ...] = ()
    complete: bool


def inspect_prediction_coverage(
    instance_ids: Iterable[str],
    predictions: Iterable[Mapping[str, Any]],
) -> PredictionCoverage:
    """Inspect exact prediction coverage without accepting evaluator-only keys."""

    expected = tuple(str(item) for item in instance_ids)
    if len(expected) != len(set(expected)):
        raise ValueError("Expected instance_ids contain duplicates")
    materialized = [dict(item) for item in predictions]
    assert_no_forbidden_fields(materialized)

    submitted: list[str] = []
    duplicate_ids: list[str] = []
    seen: set[str] = set()
    for index, record in enumerate(materialized):
        instance_id = record.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id.strip():
            raise PredictionCoverageError(
                f"Prediction {index} has no non-empty instance_id"
            )
        if "model_patch" not in record or not isinstance(record["model_patch"], str):
            raise PredictionCoverageError(
                f"Prediction {instance_id} must contain a string model_patch"
            )
        submitted.append(instance_id)
        if instance_id in seen and instance_id not in duplicate_ids:
            duplicate_ids.append(instance_id)
        seen.add(instance_id)

    expected_set = set(expected)
    submitted_set = set(submitted)
    missing = tuple(item for item in expected if item not in submitted_set)
    unexpected = tuple(item for item in submitted if item not in expected_set)
    unexpected = tuple(dict.fromkeys(unexpected))
    duplicates = tuple(duplicate_ids)
    complete = not missing and not unexpected and not duplicates
    return PredictionCoverage(
        expected_count=len(expected),
        prediction_count=len(materialized),
        covered_count=len(expected_set & submitted_set),
        missing_ids=missing,
        unexpected_ids=unexpected,
        duplicate_ids=duplicates,
        complete=complete,
    )


def validate_prediction_coverage(
    instance_ids: Iterable[str],
    predictions: Iterable[Mapping[str, Any]],
) -> PredictionCoverage:
    """Require one prediction record for every selected instance and no others."""

    coverage = inspect_prediction_coverage(instance_ids, predictions)
    if not coverage.complete:
        raise PredictionCoverageError(
            "Predictions do not exactly cover the selection: "
            f"missing={list(coverage.missing_ids)}, "
            f"unexpected={list(coverage.unexpected_ids)}, "
            f"duplicates={list(coverage.duplicate_ids)}"
        )
    return coverage
