import pytest

from prompt_methods.mctsv8.utils_call import (
    extract_explicit_answer,
    select_explicit_answer_vote,
)


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (
            "Reasoning complete. THE ANSWER IS 42.",
            {
                "raw_response": "Reasoning complete. THE ANSWER IS 42.",
                "extracted_answer": "42",
                "answer_extract_status": "ok",
                "matches": ["42"],
            },
        ),
        (
            "The final answer is: D (Tokyo).",
            {
                "raw_response": "The final answer is: D (Tokyo).",
                "extracted_answer": "D (Tokyo)",
                "answer_extract_status": "ok",
                "matches": ["D (Tokyo)"],
            },
        ),
        (
            "There is no explicit marker here.",
            {
                "raw_response": "There is no explicit marker here.",
                "extracted_answer": None,
                "answer_extract_status": "missing_explicit_answer",
                "matches": [],
            },
        ),
        (
            "The answer is:   ",
            {
                "raw_response": "The answer is:   ",
                "extracted_answer": None,
                "answer_extract_status": "empty_after_marker",
                "matches": [],
            },
        ),
    ],
)
def test_extract_explicit_answer_statuses(response, expected):
    assert extract_explicit_answer(response) == expected


def test_repeated_identical_answer_markers_remain_valid():
    result = extract_explicit_answer("The answer is 7. Double-check: answer is 7.")

    assert result["answer_extract_status"] == "ok"
    assert result["extracted_answer"] == "7"
    assert result["matches"] == ["7", "7"]


def test_conflicting_answer_markers_are_ambiguous():
    result = extract_explicit_answer("The answer is 7. Correction: answer is 8.")

    assert result["answer_extract_status"] == "ambiguous_multiple_markers"
    assert result["extracted_answer"] is None
    assert result["matches"] == ["7", "8"]


def test_vote_excludes_candidates_without_valid_explicit_answers():
    responses = [
        "First path. The answer is 42.",
        "Second path concludes 7 without the marker.",
        "Third path. ANSWER IS 42",
    ]

    result = select_explicit_answer_vote(responses)

    assert result["answer_extract_status"] == "ok"
    assert result["response"] == "42"
    assert result["raw_response"] == responses[0]
    assert result["valid_vote_count"] == 2
    assert result["excluded_vote_count"] == 1
    assert result["excluded_candidate_indices"] == [1]
    assert [item["answer_extract_status"] for item in result["candidates"]] == [
        "ok",
        "missing_explicit_answer",
        "ok",
    ]


def test_vote_marks_parse_failed_when_every_candidate_is_excluded():
    responses = ["No marker in this response.", "The answer is:   "]

    result = select_explicit_answer_vote(responses)

    assert result["answer_extract_status"] == "parse_failed"
    assert result["response"] == ""
    assert result["raw_response"] is None
    assert result["valid_vote_count"] == 0
    assert result["excluded_vote_count"] == 2
    assert result["excluded_candidate_indices"] == [0, 1]
