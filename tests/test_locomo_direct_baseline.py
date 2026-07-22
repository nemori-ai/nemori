from __future__ import annotations

from evaluation.locomo.baseline_direct import DirectLocomoBaseline, transcript_for_item
from evaluation.locomo.search import questions_for_item
from nemori import MemoryConfig


def test_direct_transcript_flattens_locomo_sessions() -> None:
    item = {
        "conversation": {
            "speaker_a": "Ada",
            "speaker_b": "Ben",
            "session_1_date_time": "1:56 pm on 8 May, 2023",
            "session_1": [
                {"speaker": "Ada", "text": "I adopted a cat."},
                {"speaker": "Ben", "text": "What is its name?"},
            ],
        }
    }

    transcript = transcript_for_item(item)

    assert "2023-05-08T13:56:00" in transcript
    assert "Ada: I adopted a cat." in transcript
    assert "Ben: What is its name?" in transcript


def test_direct_checkpoint_accepts_scored_question_prefix(tmp_path) -> None:
    dataset = [
        {
            "conversation": {},
            "qa": [
                {"question": "q1", "category": 1},
                {"question": "q5", "category": 5},
            ],
        }
    ]
    baseline = DirectLocomoBaseline(
        MemoryConfig(llm_api_key="x", llm_base_url="http://127.0.0.1:9/v1"),
        output_path=tmp_path / "results.json",
        checkpoint_every=1,
        max_concurrent_answer=1,
        scored_only=True,
        resume=True,
        max_transcript_chars=0,
    )
    matched = baseline._matching_checkpoint(
        {"0": [{"question": "q1", "response": "a"}]},
        dataset,
    )

    assert questions_for_item(dataset[0], scored_only=True) == [
        {"question": "q1", "category": 1}
    ]
    assert matched == {"0": [{"question": "q1", "response": "a"}]}
