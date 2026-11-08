import json
import pytest

from scripts import run_resumable_inference as runner

build_inference_command = runner.build_inference_command
load_dataset_ids = runner.load_dataset_ids


def test_load_dataset_ids_supports_json_mapping_and_jsonl(tmp_path):
    json_path = tmp_path / "mathvista.json"
    json_path.write_text(json.dumps({"2": {}, "7": {}}), encoding="utf-8")
    jsonl_path = tmp_path / "mathvision.jsonl"
    jsonl_path.write_text(
        '\n'.join([json.dumps({"id": "11"}), json.dumps({"id": 12})]) + '\n',
        encoding="utf-8",
    )

    assert load_dataset_ids(json_path) == ["2", "7"]
    assert load_dataset_ids(jsonl_path) == ["11", "12"]


def test_build_command_uses_requested_dataset_rollout_and_output(tmp_path):
    command = build_inference_command(
        python_executable="/env/bin/python",
        dataset="mathvision",
        input_file=tmp_path / "testmini.jsonl",
        method_name="baseline",
        prompt_method="mctsv8_vpw",
        rollout=8,
        output_dir=tmp_path / "baseline" / "mathvision",
    )

    assert command[:2] == ["/env/bin/python", "inference.py"]
    assert "data=mathvision" in command
    assert "desc=baseline" in command
    assert "prompt_method=mctsv8_vpw" in command
    assert "prompt_method.mcts_rollouts=8" in command
    assert f"hydra.run.dir={tmp_path / 'baseline' / 'mathvision'}" in command


def test_write_filtered_jsonl_skips_question_ids_and_preserves_rows(tmp_path):
    source = tmp_path / "testmini.jsonl"
    rows = [
        {"id": 4, "question": "first"},
        {"id": "32", "question": "keep"},
        {"id": 46, "question": "last"},
    ]
    source.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    destination = tmp_path / "run_input.jsonl"

    kept_ids = runner.write_filtered_jsonl(source, destination, {"4", "46"})

    assert kept_ids == ["32"]
    assert destination.read_text(encoding="utf-8") == json.dumps(rows[1]) + "\n"


def test_prepare_output_directory_requires_resume_for_existing_run(tmp_path):
    output_dir = tmp_path / "mctsv8_vpw" / "mathvision"
    output_dir.mkdir(parents=True)
    (output_dir / "generation.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError):
        runner.prepare_output_directory(output_dir, resume=False)

    runner.prepare_output_directory(output_dir, resume=True)


def test_next_attempt_number_does_not_overwrite_existing_logs(tmp_path):
    (tmp_path / "inference_attempt_1.log").write_text("old", encoding="utf-8")
    (tmp_path / "inference_attempt_3.log").write_text("old", encoding="utf-8")

    assert runner.next_attempt_number(tmp_path) == 4
