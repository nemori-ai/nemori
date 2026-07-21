from __future__ import annotations

import json

from evaluation.locomo.evals import flatten_scored_items, resume_checkpoint


def test_flatten_scored_items_skips_category_five() -> None:
    data = {
        "0": [
            {"question": "q1", "category": 1},
            {"question": "q5", "category": 5},
        ],
        "1": [{"question": "q2", "category": "2"}],
    }

    items = flatten_scored_items(data)

    assert [(key, index, item["question"]) for key, index, item in items] == [
        ("0", 0, "q1"),
        ("1", 0, "q2"),
    ]


def test_resume_checkpoint_skips_only_matching_scored_records(tmp_path) -> None:
    work_items = [
        (
            "0",
            0,
            {"question": "q1", "answer": "a1", "response": "r1", "category": 1},
        ),
        (
            "0",
            1,
            {"question": "q2", "answer": "a2", "response": "r2", "category": 2},
        ),
    ]
    output = tmp_path / "metrics.json"
    output.write_text(
        json.dumps(
            {
                "0": [
                    {
                        "question": "q1",
                        "answer": "a1",
                        "response": "r1",
                        "category": "1",
                        "llm_score": 1,
                    },
                    {
                        "question": "stale",
                        "answer": "old",
                        "response": "old",
                        "category": "1",
                        "llm_score": 0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    restored, pending = resume_checkpoint(output, work_items)

    assert [item["question"] for item in restored["0"]] == ["q1"]
    assert [(key, index, item["question"]) for key, index, item in pending] == [
        ("0", 1, "q2")
    ]
