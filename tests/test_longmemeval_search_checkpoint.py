import json
from pathlib import Path

import pytest

from evaluation.longmemeval import search as search_module
from evaluation.longmemeval.search import LongMemEvalSearcher
from nemori import MemoryConfig


def make_searcher(output_path: Path, **overrides) -> LongMemEvalSearcher:
    options = {
        "config": MemoryConfig(llm_api_key="test-key"),
        "output_path": output_path,
        "top_k_episodes": 10,
        "top_k_semantic": 10,
        "search_method": "vector",
        "include_original_messages_top_k": 0,
        "checkpoint_every": 0,
    }
    options.update(overrides)
    return LongMemEvalSearcher(**options)


def result(question_id: str, question: str) -> dict:
    return {
        "question_id": question_id,
        "question": question,
        "question_date": "2024-01-01",
        "answer": "gold",
        "question_type": "default",
        "response": "done",
        "memories": [],
        "search_method": "vector",
    }


def dataset_item(question_id: str, question: str) -> dict:
    return {
        "question_id": question_id,
        "question": question,
        "question_date": "2024-01-01",
        "answer": "gold",
        "question_type": "default",
    }


def test_save_retries_atomic_replace_with_unique_temp(tmp_path, monkeypatch):
    output = tmp_path / "results.json"
    output.write_text(json.dumps({"old": [result("old", "old")]}), encoding="utf-8")
    searcher = make_searcher(output)
    searcher.results["q1"].append(result("q1", "new"))

    original_replace = Path.replace
    attempted_paths = []

    def intermittently_locked(path, target):
        attempted_paths.append(path)
        if len(attempted_paths) < 3:
            raise PermissionError(5, "file is temporarily locked")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", intermittently_locked)
    monkeypatch.setattr(search_module.time, "sleep", lambda _delay: None)

    searcher.save()

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["q1"][0]["question"] == "new"
    assert len(attempted_paths) == 3
    assert len(set(attempted_paths)) == 1
    assert attempted_paths[0].name.startswith("results.json.")
    assert attempted_paths[0].name.endswith(".tmp")
    assert not list(tmp_path.glob("results.json*.tmp"))


def test_failed_replace_preserves_old_checkpoint_and_recovery_temp(tmp_path, monkeypatch):
    output = tmp_path / "results.json"
    old_checkpoint = {"old": [result("old", "old")]}
    output.write_text(json.dumps(old_checkpoint), encoding="utf-8")
    searcher = make_searcher(output)
    searcher.results["q1"].append(result("q1", "new"))

    monkeypatch.setattr(
        Path,
        "replace",
        lambda _path, _target: (_ for _ in ()).throw(
            PermissionError(5, "file is persistently locked")
        ),
    )
    monkeypatch.setattr(search_module.time, "sleep", lambda _delay: None)

    with pytest.raises(OSError, match="recovery data was kept"):
        searcher.save()

    assert json.loads(output.read_text(encoding="utf-8")) == old_checkpoint
    recovery_files = list(tmp_path.glob("results.json.*.tmp"))
    assert len(recovery_files) == 1
    recovered = json.loads(recovery_files[0].read_text(encoding="utf-8"))
    assert recovered["q1"][0]["question"] == "new"


@pytest.mark.asyncio
async def test_process_resumes_from_most_complete_valid_temp(tmp_path, monkeypatch, capsys):
    output = tmp_path / "results.json"
    output.write_text(
        json.dumps({"id1": [result("id1", "q1")]}),
        encoding="utf-8",
    )
    recovery = tmp_path / "results.json.interrupted.tmp"
    recovery.write_text(
        json.dumps({
            "id1": [result("id1", "q1")],
            "id2": [result("id2", "q2")],
        }),
        encoding="utf-8",
    )
    dataset = [
        dataset_item("id1", "q1"),
        dataset_item("id2", "q2"),
        dataset_item("id3", "q3"),
    ]
    searched_questions = []

    class FakeMemory:
        def __init__(self, _config):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def search(self, _user_id, question, **_kwargs):
            searched_questions.append(question)
            return {"episodes": [], "semantic_memories": []}

    async def fake_answer(_question, _question_date, _memories):
        return "new answer"

    monkeypatch.setattr(search_module, "NemoriMemory", FakeMemory)
    monkeypatch.setattr(search_module, "emit_trace", lambda _event: None)
    searcher = make_searcher(output, checkpoint_every=1)
    searcher.answer = fake_answer

    await searcher.process(dataset)

    assert searched_questions == ["q3"]
    assert searcher.completed_questions == 3
    assert set(searcher.results) == {"id1", "id2", "id3"}
    assert set(json.loads(output.read_text(encoding="utf-8"))) == {"id1", "id2", "id3"}
    assert not recovery.exists()
    progress_lines = [
        line for line in capsys.readouterr().out.splitlines() if line.startswith("NEMORI_PROGRESS ")
    ]
    progress = [json.loads(line.removeprefix("NEMORI_PROGRESS ")) for line in progress_lines]
    assert [event["completed"] for event in progress] == [2, 3]
    assert all(event["total"] == 3 for event in progress)


@pytest.mark.asyncio
async def test_no_resume_ignores_existing_checkpoint(tmp_path, monkeypatch):
    output = tmp_path / "results.json"
    output.write_text(
        json.dumps({"id1": [result("id1", "q1")]}),
        encoding="utf-8",
    )
    dataset = [dataset_item("id1", "q1"), dataset_item("id2", "q2")]
    searched_questions = []

    class FakeMemory:
        def __init__(self, _config):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def search(self, _user_id, question, **_kwargs):
            searched_questions.append(question)
            return {"episodes": [], "semantic_memories": []}

    async def fake_answer(_question, _question_date, _memories):
        return "new answer"

    monkeypatch.setattr(search_module, "NemoriMemory", FakeMemory)
    monkeypatch.setattr(search_module, "emit_trace", lambda _event: None)
    searcher = make_searcher(output, resume=False)
    searcher.answer = fake_answer

    await searcher.process(dataset)

    assert sorted(searched_questions) == ["q1", "q2"]
    assert searcher.completed_questions == 2
