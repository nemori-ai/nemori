import pytest

from evaluation.longmemeval.evals import normalize_results


def test_normalize_results_flattens_search_output():
    record = {"question_id": "q1", "response": "answer"}

    assert normalize_results({"q1": [record]}) == [record]


def test_normalize_results_keeps_flat_output():
    records = [{"question_id": "q1"}, {"question_id": "q2"}]

    assert normalize_results(records) == records


def test_normalize_results_rejects_unknown_shape():
    with pytest.raises(ValueError):
        normalize_results("not-results")
