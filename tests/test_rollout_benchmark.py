import json

import pytest

from utils.generate import is_out_of_memory_exception

from scripts.benchmark_mctsv8_rollouts import (
    add_scaling_ratios,
    build_benchmark_subset,
    is_oom_failure,
    mark_first_unfinished_as_skipped,
    prepare_empty_run_directory,
    summarize_generation,
    validate_rollouts,
)


def test_subset_selection_is_deterministic_and_reserves_warmup():
    data = {str(index): {"value": index} for index in range(30)}

    subset, manifest = build_benchmark_subset(data, sample_size=5, seed=42)

    assert list(subset) == ["20", "3", "0", "23", "8", "7"]
    assert manifest == {
        "seed": 42,
        "warmup_id": "20",
        "measured_ids": ["3", "0", "23", "8", "7"],
    }


def test_subset_selection_accepts_a_known_good_warmup_id():
    data = {str(index): {"value": index} for index in range(30)}

    subset, manifest = build_benchmark_subset(
        data, sample_size=5, seed=42, warmup_id="2"
    )

    assert list(subset) == ["2", "21", "4", "0", "24", "9"]
    assert manifest["warmup_id"] == "2"
    assert manifest["measured_ids"] == ["21", "4", "0", "24", "9"]


def test_prepare_run_directory_refuses_existing_artifacts(tmp_path):
    run_dir = tmp_path / "rollout_2"
    run_dir.mkdir()
    (run_dir / "generation.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        prepare_empty_run_directory(run_dir)


def test_generation_summary_excludes_warmup_and_counts_parse_failures(tmp_path):
    generation = {
        "warmup": {"time": 100.0, "answer_extract_status": "ok"},
        "a": {"time": 1.0, "answer_extract_status": "ok"},
        "b": {"time": 2.0, "answer_extract_status": "parse_failed"},
        "c": {"time": 3.0, "answer_extract_status": "ok"},
        "d": {"time": 4.0, "answer_extract_status": "missing_explicit_answer"},
        "e": {"time": 10.0, "answer_extract_status": "ok"},
    }
    path = tmp_path / "generation.json"
    path.write_text(json.dumps(generation), encoding="utf-8")

    summary = summarize_generation(path, rollout=4, warmup_id="warmup")

    assert summary == {
        "rollout": 4,
        "sample_count": 5,
        "total_seconds": 20.0,
        "mean_seconds": 4.0,
        "median_seconds": 3.0,
        "p90_seconds": 10.0,
        "questions_per_hour": 900.0,
        "parse_failure_count": 2,
        "parse_failure_rate": 0.4,
        "skipped_count": 0,
        "skipped_ids": [],
    }


def test_generation_summary_rejects_a_failed_warmup(tmp_path):
    path = tmp_path / "generation.json"
    path.write_text(
        json.dumps(
            {
                "warmup": {"error": "model input exceeded context"},
                "a": {"time": 1.0, "answer_extract_status": "ok"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="warm-up item failed"):
        summarize_generation(path, rollout=2, warmup_id="warmup")


def test_oom_detection_accepts_cuda_oom_and_sigkill():
    assert is_oom_failure(1, "torch.OutOfMemoryError: CUDA out of memory")
    assert is_oom_failure(-9, "")
    assert not is_oom_failure(1, "invalid model configuration")


def test_generation_re_raises_only_out_of_memory_errors():
    assert is_out_of_memory_exception(
        RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    )
    assert not is_out_of_memory_exception(RuntimeError("context length exceeded"))


def test_oom_recovery_marks_first_unfinished_item_and_preserves_completed(tmp_path):
    path = tmp_path / "generation.json"
    path.write_text(
        json.dumps(
            {
                "warmup": {"time": 3.0, "response": "A"},
                "a": {"time": 4.0, "response": "B"},
                "b": {"question": "unfinished"},
                "c": {"question": "not attempted"},
            }
        ),
        encoding="utf-8",
    )

    skipped_id = mark_first_unfinished_as_skipped(
        path, ["warmup", "a", "b", "c"], reason="CUDA out of memory"
    )
    saved = json.loads(path.read_text(encoding="utf-8"))

    assert skipped_id == "b"
    assert saved["a"] == {"time": 4.0, "response": "B"}
    assert saved["b"]["response"] == "Skipped after runtime failure"
    assert saved["b"]["answer_extract_status"] == "skipped_runtime_failure"
    assert saved["b"]["runtime_failure"] == "CUDA out of memory"


def test_generation_summary_excludes_skipped_runtime_failures(tmp_path):
    path = tmp_path / "generation.json"
    path.write_text(
        json.dumps(
            {
                "warmup": {"time": 3.0, "answer_extract_status": "ok"},
                "a": {"time": 2.0, "answer_extract_status": "ok"},
                "b": {
                    "response": "Skipped after runtime failure",
                    "answer_extract_status": "skipped_runtime_failure",
                    "runtime_failure": "CUDA out of memory",
                },
                "c": {"time": 4.0, "answer_extract_status": "ok"},
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_generation(path, rollout=2, warmup_id="warmup")

    assert summary["sample_count"] == 2
    assert summary["mean_seconds"] == 3.0
    assert summary["skipped_count"] == 1
    assert summary["skipped_ids"] == ["b"]


def test_scaling_ratios_use_rollout_two_as_baseline():
    summaries = [
        {"rollout": 2, "mean_seconds": 10.0},
        {"rollout": 4, "mean_seconds": 18.0},
        {"rollout": 10, "mean_seconds": 40.0},
    ]

    assert add_scaling_ratios(summaries) == [
        {"rollout": 2, "mean_seconds": 10.0, "time_vs_rollout_2": 1.0},
        {"rollout": 4, "mean_seconds": 18.0, "time_vs_rollout_2": 1.8},
        {"rollout": 10, "mean_seconds": 40.0, "time_vs_rollout_2": 4.0},
    ]


def test_rollout_validation_allows_a_rollout_two_smoke_run():
    assert validate_rollouts([2]) == [2]


@pytest.mark.parametrize("rollouts", [[], [4], [2, 2], [0, 2]])
def test_rollout_validation_rejects_invalid_benchmark_sets(rollouts):
    with pytest.raises(ValueError):
        validate_rollouts(rollouts)
