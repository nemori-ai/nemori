from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluation.swebench import download


def complete_row() -> dict[str, Any]:
    return {
        "repo": "owner/repo",
        "instance_id": "owner__repo-1",
        "base_commit": "abcdef1",
        "problem_statement": "issue",
        "patch": "gold",
        "test_patch": "tests",
        "FAIL_TO_PASS": "[]",
        "PASS_TO_PASS": "[]",
    }


def test_download_rows_passes_immutable_revision_to_every_viewer_call(
    monkeypatch,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_fetch(endpoint: str, **params: Any) -> dict[str, Any]:
        calls.append((endpoint, params))
        if endpoint == "splits":
            return {"splits": [{"config": "default", "split": "test"}]}
        return {"num_rows_total": 1, "rows": [{"row": complete_row()}]}

    monkeypatch.setattr(download, "fetch_json", fake_fetch)
    revision = "a" * 40

    rows = download.download_rows(revision=revision)

    assert rows == [complete_row()]
    assert calls
    assert all(params["revision"] == revision for _, params in calls)


def test_download_manifest_records_revision_and_exact_file_hash(tmp_path: Path) -> None:
    target = tmp_path / "swe.json"
    download.atomic_write(target, [complete_row()])

    manifest_path = download.write_manifest(
        target,
        dataset=download.DATASET,
        config="default",
        split="test",
        requested_revision="main",
        resolved_revision="b" * 40,
        rows=1,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["resolved_revision"] == "b" * 40
    assert manifest["rows"] == 1
    assert len(manifest["dataset_sha256"]) == 64
    assert download.validate_bundle(target) == manifest


def test_dataset_bundle_requires_its_manifest(tmp_path: Path) -> None:
    target = tmp_path / "swe.json"
    download.atomic_write(target, [complete_row()])

    try:
        download.validate_bundle(target)
    except FileNotFoundError as exc:
        assert "manifest is required but missing" in str(exc)
    else:  # pragma: no cover - documents the fail-closed security boundary
        raise AssertionError("A dataset without its manifest must be rejected")


def test_dataset_bundle_detects_dataset_or_manifest_tampering(tmp_path: Path) -> None:
    target = tmp_path / "swe.json"
    manifest_path = download.write_dataset_bundle(
        target,
        [complete_row()],
        dataset=download.DATASET,
        config="default",
        split="test",
        requested_revision="main",
        resolved_revision="c" * 40,
    )

    changed = complete_row()
    changed["problem_statement"] = "tampered"
    download.atomic_write(target, [changed])
    try:
        download.validate_bundle(target)
    except RuntimeError as exc:
        assert "bytes do not match" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Changed dataset bytes must be rejected")

    download.atomic_write(target, [complete_row()])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["rows"] = 2
    download.atomic_write(manifest_path, manifest)
    try:
        download.validate_bundle(target)
    except RuntimeError as exc:
        assert "row count mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Changed manifest metadata must be rejected")


def test_manifest_writer_rejects_declared_row_count_mismatch(tmp_path: Path) -> None:
    target = tmp_path / "swe.json"
    download.atomic_write(target, [complete_row()])

    try:
        download.write_manifest(
            target,
            dataset=download.DATASET,
            config="default",
            split="test",
            requested_revision="main",
            resolved_revision="d" * 40,
            rows=2,
        )
    except RuntimeError as exc:
        assert "row count mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Declared rows must match the dataset")
