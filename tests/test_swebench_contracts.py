from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from dashboard.swebench_data import load_dataset_rows, swebench_guide_payload
from evaluation.swebench.contracts import (
    FORBIDDEN_API_FIELDS,
    AgentInstance,
    EvaluatorInstance,
    EvaluatorSelection,
    ForbiddenFieldError,
    PredictionCoverageError,
    SelectionManifest,
    assert_no_forbidden_fields,
    build_selection_manifest,
    create_selection_manifest,
    find_forbidden_fields,
    inspect_prediction_coverage,
    sanitize_agent_instance,
    validate_prediction_coverage,
)


def raw_instance(instance_id: str = "owner__repo-1") -> dict[str, object]:
    return {
        "instance_id": instance_id,
        "repo": "owner/repo",
        "base_commit": "abc123",
        "problem_statement": "A real issue",
        "created_at": "2024-01-01T00:00:00Z",
        "version": "1.0",
        "hints_text": "not part of the strict agent protocol",
        "environment_setup_commit": "def456",
        "patch": "answer diff",
        "test_patch": "private tests",
        "FAIL_TO_PASS": '["test_target"]',
        "PASS_TO_PASS": '["test_regression"]',
    }


def test_agent_instance_is_a_physical_allowlist() -> None:
    agent = AgentInstance.from_raw(raw_instance())
    payload = agent.to_agent_payload()

    assert set(payload) == {
        "instance_id",
        "repo",
        "base_commit",
        "problem_statement",
        "created_at",
        "version",
    }
    assert "hints_text" not in payload
    assert_no_forbidden_fields(payload)
    serialized = json.dumps(payload)
    assert "answer diff" not in serialized
    assert "private tests" not in serialized


def test_agent_instance_rejects_extra_fields_when_constructed_directly() -> None:
    with pytest.raises(ValidationError):
        AgentInstance(
            instance_id="owner__repo-1",
            repo="owner/repo",
            base_commit="abc",
            problem_statement="issue",
            patch="leak",  # type: ignore[call-arg]
        )


def test_evaluator_instance_is_opaque_and_returns_defensive_copies() -> None:
    raw = raw_instance()
    evaluator = EvaluatorInstance(raw)
    harness_record = evaluator.to_harness_record()
    harness_record["patch"] = "mutated"

    assert evaluator.instance_id == "owner__repo-1"
    assert evaluator.to_harness_record()["patch"] == "answer diff"
    assert "answer diff" not in repr(evaluator)
    assert_no_forbidden_fields(evaluator.to_agent_instance().to_agent_payload())


def test_recursive_forbidden_field_scan_is_exact_and_case_insensitive() -> None:
    payload = {
        "safe": [{"model_patch": "allowed"}],
        "nested": {"TeSt_PaTcH": "leak"},
    }

    assert find_forbidden_fields(payload) == ("$.nested.TeSt_PaTcH",)
    with pytest.raises(ForbiddenFieldError, match="TeSt_PaTcH"):
        assert_no_forbidden_fields(payload)


def test_selection_manifest_pins_dataset_ids_and_fairness(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.json"
    dataset.write_text('[{"instance_id":"a"}]\n', encoding="utf-8")
    fairness_a = {
        "model": "same-model",
        "token_budget": 10_000,
        "tool_policy": {"network": False, "shell": True},
    }
    fairness_b = {
        "tool_policy": {"shell": True, "network": False},
        "token_budget": 10_000,
        "model": "same-model",
    }

    first = create_selection_manifest(
        dataset_path=dataset,
        instance_ids=["a", "b"],
        fairness_config=fairness_a,
    )
    second = create_selection_manifest(
        dataset_path=dataset,
        instance_ids=["a", "b"],
        fairness_config=fairness_b,
    )

    assert first.dataset_sha256 == hashlib.sha256(dataset.read_bytes()).hexdigest()
    assert first.fairness_hash == second.fairness_hash
    assert first.instance_ids == ("a", "b")
    with pytest.raises(ValidationError, match="duplicate"):
        SelectionManifest(
            dataset_name="dataset",
            source_split="test",
            dataset_sha256="0" * 64,
            instance_ids=("a", "a"),
            fairness_hash="1" * 64,
        )


def test_evaluator_selection_requires_exact_manifest_order(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.json"
    dataset.write_text("[]", encoding="utf-8")
    manifest = create_selection_manifest(
        dataset_path=dataset,
        instance_ids=["a", "b"],
        fairness_config={"model": "m"},
    )
    selection = EvaluatorSelection(
        manifest,
        [EvaluatorInstance(raw_instance("a")), EvaluatorInstance(raw_instance("b"))],
    )

    assert [item.instance_id for item in selection.agent_instances()] == ["a", "b"]
    with pytest.raises(ValueError, match="same order"):
        EvaluatorSelection(
            manifest,
            [
                EvaluatorInstance(raw_instance("b")),
                EvaluatorInstance(raw_instance("a")),
            ],
        )


def test_dataset_helpers_and_selection_contract_names(tmp_path: Path) -> None:
    source = tmp_path / "dataset.json"
    source.write_text(json.dumps([raw_instance("a")]), encoding="utf-8")
    rows = load_dataset_rows(source)
    manifest = build_selection_manifest(
        rows,
        source,
        fairness_config={"model": "m", "token_budget": 100},
    )

    assert manifest.instance_ids == ("a",)
    assert sanitize_agent_instance(rows[0])["instance_id"] == "a"


def test_prediction_coverage_requires_exact_ids_but_allows_empty_model_diff() -> None:
    predictions = [
        {
            "instance_id": "a",
            "model_name_or_path": "agent",
            "model_patch": "",
        },
        {
            "instance_id": "b",
            "model_name_or_path": "agent",
            "model_patch": "diff --git ...",
        },
    ]

    coverage = validate_prediction_coverage(["a", "b"], predictions)
    assert coverage.complete is True
    assert coverage.covered_count == 2

    incomplete = inspect_prediction_coverage(
        ["a", "b"],
        [*predictions, {"instance_id": "b", "model_patch": "duplicate"}],
    )
    assert incomplete.duplicate_ids == ("b",)
    with pytest.raises(PredictionCoverageError, match="duplicates"):
        validate_prediction_coverage(
            ["a", "b"],
            [*predictions, {"instance_id": "b", "model_patch": "duplicate"}],
        )


def test_prediction_coverage_rejects_forbidden_raw_fields() -> None:
    with pytest.raises(ForbiddenFieldError):
        validate_prediction_coverage(
            ["a"],
            [{"instance_id": "a", "model_patch": "safe", "test_patch": "leak"}],
        )


def test_guide_uses_real_dev_demo_and_has_no_evaluator_only_keys() -> None:
    guide = swebench_guide_payload()
    demo = guide["demo"]

    assert demo["instance_id"] == "marshmallow-code__marshmallow-1359"
    assert demo["repo"] == "marshmallow-code/marshmallow"
    assert demo["base_commit"] == "b40a0f4e33823e6d0f341f7e8684e359a99060d1"
    assert demo["source_split"] == "dev"
    assert demo["excluded_from_scoring"] is True
    assert "getattr(self.root.opts" in demo["agent_output_diff"]
    assert demo["harness"]["resolved"] is True
    assert demo["harness"]["fail_to_pass"]["after"] == "passed"
    assert demo["harness"]["pass_to_pass"]["count"] == 76
    assert_no_forbidden_fields(guide, forbidden_fields=FORBIDDEN_API_FIELDS)

    # Callers cannot mutate the module-level template returned to later clients.
    demo["repo"] = "mutated"
    assert swebench_guide_payload()["demo"]["repo"] == "marshmallow-code/marshmallow"
