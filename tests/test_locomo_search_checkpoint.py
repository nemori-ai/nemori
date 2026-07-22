import json
from pathlib import Path

import pytest

from evaluation.locomo import search as search_module
from evaluation.locomo.search import LocomoSearcher
from nemori import MemoryConfig


def make_searcher(output_path: Path, **overrides) -> LocomoSearcher:
    options = {
        "config": MemoryConfig(llm_api_key="test-key"),
        "output_path": output_path,
        "top_k_episodes": 10,
        "top_k_semantic": 20,
        "search_method": "vector",
        "include_original_messages_top_k": 2,
        "checkpoint_every": 0,
    }
    options.update(overrides)
    return LocomoSearcher(**options)


def result(question: str, response: str = "done") -> dict:
    return {
        "question": question,
        "answer": response,
        "category": 1,
        "evidence": [],
        "response": response,
        "memories": [],
        "search_method": "vector",
    }


def test_save_retries_atomic_replace_with_unique_temp(tmp_path, monkeypatch):
    output = tmp_path / "results.json"
    output.write_text(json.dumps({"0": [result("old")]}), encoding="utf-8")
    searcher = make_searcher(output)
    searcher.results["0"].append(result("new"))

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

    assert json.loads(output.read_text(encoding="utf-8"))["0"][0]["question"] == "new"
    assert len(attempted_paths) == 3
    assert len(set(attempted_paths)) == 1
    assert attempted_paths[0].name.startswith("results.json.")
    assert attempted_paths[0].name.endswith(".tmp")
    assert not list(tmp_path.glob("results.json*.tmp"))


def test_failed_replace_preserves_old_checkpoint_and_recovery_temp(tmp_path, monkeypatch):
    output = tmp_path / "results.json"
    old_checkpoint = {"0": [result("old")]}
    output.write_text(json.dumps(old_checkpoint), encoding="utf-8")
    searcher = make_searcher(output)
    searcher.results["0"].append(result("new"))

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
    assert recovered["0"][0]["question"] == "new"


@pytest.mark.asyncio
async def test_process_resumes_from_most_complete_valid_temp(tmp_path, monkeypatch, capsys):
    output = tmp_path / "results.json"
    output.write_text(json.dumps({"0": [result("q1")]}), encoding="utf-8")
    recovery = tmp_path / "results.json.interrupted.tmp"
    recovery.write_text(
        json.dumps({"0": [result("q1"), result("q2")]}),
        encoding="utf-8",
    )
    dataset = [
        {
            "conversation": {"speaker_a": "Alice"},
            "qa": [
                {"question": "q1", "answer": "a1"},
                {"question": "q2", "answer": "a2"},
                {"question": "q3", "answer": "a3"},
            ],
        }
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

    async def fake_answer(_question, _memories):
        return "new answer"

    monkeypatch.setattr(search_module, "NemoriMemory", FakeMemory)
    monkeypatch.setattr(search_module, "emit_trace", lambda _event: None)
    searcher = make_searcher(output, checkpoint_every=1)
    searcher.answer = fake_answer

    await searcher.process(dataset, max_workers=1)

    assert searched_questions == ["q3"]
    assert searcher.completed_questions == 3
    assert [item["question"] for item in searcher.results["0"]] == ["q1", "q2", "q3"]
    assert [item["question"] for item in json.loads(output.read_text())["0"]] == [
        "q1",
        "q2",
        "q3",
    ]
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
    output.write_text(json.dumps({"0": [result("q1")]}), encoding="utf-8")
    dataset = [
        {
            "conversation": {"speaker_a": "Alice"},
            "qa": [
                {"question": "q1", "answer": "a1"},
                {"question": "q2", "answer": "a2"},
            ],
        }
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

    async def fake_answer(_question, _memories):
        return "new answer"

    monkeypatch.setattr(search_module, "NemoriMemory", FakeMemory)
    monkeypatch.setattr(search_module, "emit_trace", lambda _event: None)
    searcher = make_searcher(output, resume=False)
    searcher.answer = fake_answer

    await searcher.process(dataset, max_workers=1)

    assert searched_questions == ["q1", "q2"]
    assert searcher.completed_questions == 2


@pytest.mark.asyncio
async def test_scored_only_filters_category_5_and_resumes_filtered_sequence(
    tmp_path,
    monkeypatch,
    capsys,
):
    output = tmp_path / "results.json"
    output.write_text(json.dumps({"0": [result("q1")]}), encoding="utf-8")
    dataset = [
        {
            "conversation": {"speaker_a": "Alice"},
            "qa": [
                {"question": "q1", "answer": "a1", "category": 1},
                {"question": "excluded-int", "answer": "a2", "category": 5},
                {"question": "excluded-string", "answer": "a3", "category": "5"},
                {"question": "q4", "answer": "a4", "category": 2},
            ],
        }
    ]
    searched_questions = []
    traces = []

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

    async def fake_answer(_question, _memories):
        return "new answer"

    monkeypatch.setattr(search_module, "NemoriMemory", FakeMemory)
    monkeypatch.setattr(search_module, "retrieval_snapshot", lambda **kwargs: kwargs)
    monkeypatch.setattr(search_module, "emit_trace", traces.append)
    searcher = make_searcher(output, checkpoint_every=1, scored_only=True)
    searcher.answer = fake_answer

    await searcher.process(dataset, max_workers=1)

    assert searched_questions == ["q4"]
    assert searcher.total_questions == 2
    assert searcher.completed_questions == 2
    assert [item["question"] for item in searcher.results["0"]] == ["q1", "q4"]
    assert traces[0]["question_id"] == "0:1"
    progress_lines = [
        line for line in capsys.readouterr().out.splitlines() if line.startswith("NEMORI_PROGRESS ")
    ]
    progress = [json.loads(line.removeprefix("NEMORI_PROGRESS ")) for line in progress_lines]
    assert [event["completed"] for event in progress] == [1, 2]
    assert all(event["total"] == 2 for event in progress)


def test_scored_only_checkpoint_validation_uses_filtered_order(tmp_path):
    dataset = [
        {
            "qa": [
                {"question": "q1", "category": 1},
                {"question": "excluded", "category": 5},
                {"question": "q3", "category": 2},
            ]
        }
    ]
    scored_checkpoint = {"0": [result("q1"), result("q3")]}
    full_checkpoint = {"0": [result("q1"), result("excluded")]}

    assert LocomoSearcher._matching_checkpoint(
        scored_checkpoint,
        dataset,
        scored_only=True,
    ) is not None
    assert LocomoSearcher._matching_checkpoint(
        full_checkpoint,
        dataset,
        scored_only=True,
    ) is None
    assert LocomoSearcher._matching_checkpoint(
        full_checkpoint,
        dataset,
        scored_only=False,
    ) is not None
