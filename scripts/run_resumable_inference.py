#!/usr/bin/env python3
"""Run one VReST dataset with OOM restart-and-skip recovery."""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.benchmark_mctsv8_rollouts import (
    is_oom_failure,
    mark_first_unfinished_as_skipped,
    mark_remaining_failures_as_skipped,
)


REPO_ROOT = Path("/home/jncsnlp4/SSD2/ZJJ/VReST")
DEFAULT_INPUTS = {
    "mathvision": REPO_ROOT / "data/MATH-V/data/testmini.jsonl",
    "mathvista": REPO_ROOT / "data/MathVista/data/testmini.json",
}


def load_dataset_ids(path):
    path = Path(path)
    if path.suffix.casefold() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [str(json.loads(line)["id"]) for line in handle if line.strip()]
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        return [str(item_id) for item_id in data]
    return [str(item["id"]) for item in data]


def write_filtered_jsonl(source_path, destination_path, skip_question_ids):
    """Write a JSONL run input without the requested question IDs."""
    source_path = Path(source_path)
    destination_path = Path(destination_path)
    skip_ids = {str(item_id) for item_id in skip_question_ids}
    kept_ids = []
    with source_path.open("r", encoding="utf-8") as source, destination_path.open(
        "w", encoding="utf-8"
    ) as destination:
        for line in source:
            if not line.strip():
                continue
            item = json.loads(line)
            item_id = str(item["id"])
            if item_id in skip_ids:
                continue
            destination.write(json.dumps(item) + "\n")
            kept_ids.append(item_id)
    return kept_ids


def prepare_output_directory(output_dir, resume):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def next_attempt_number(output_dir):
    attempts = []
    for path in Path(output_dir).glob("inference_attempt_*.log"):
        match = re.fullmatch(r"inference_attempt_(\d+)\.log", path.name)
        if match:
            attempts.append(int(match.group(1)))
    return max(attempts, default=0) + 1


def build_inference_command(
    python_executable,
    dataset,
    input_file,
    method_name,
    prompt_method,
    rollout,
    output_dir,
):
    return [
        str(python_executable),
        "inference.py",
        f"data={dataset}",
        "llm=qwen25vl_7b",
        f"prompt_method={prompt_method}",
        f"data.input_file={input_file}",
        f"desc={method_name}",
        f"prompt_method.mcts_rollouts={rollout}",
        f"hydra.run.dir={output_dir}",
    ]


def write_status(path, payload):
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--dataset", choices=sorted(DEFAULT_INPUTS), required=True)
    parser.add_argument("--input-file", type=Path)
    parser.add_argument("--method-name", default="baseline")
    parser.add_argument("--prompt-method", default="mctsv8_a")
    parser.add_argument("--rollout", type=int, default=8)
    parser.add_argument("--skip-question-ids", nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-oom-restarts", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.rollout < 1:
        raise ValueError("rollout must be positive")
    source_input_file = args.input_file or DEFAULT_INPUTS[args.dataset]
    skip_question_ids = list(dict.fromkeys(str(item) for item in args.skip_question_ids))
    prepare_output_directory(args.output_dir, args.resume)

    input_file = source_input_file
    if skip_question_ids:
        if Path(source_input_file).suffix.casefold() != ".jsonl":
            raise ValueError("--skip-question-ids currently requires a JSONL input")
        input_file = args.output_dir / "run_input.filtered.jsonl"
        ordered_ids = write_filtered_jsonl(
            source_input_file, input_file, skip_question_ids
        )
    else:
        ordered_ids = load_dataset_ids(input_file)

    run_config = {
        "method_name": args.method_name,
        "prompt_method": args.prompt_method,
        "dataset": args.dataset,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "rollout": args.rollout,
        "input_file": str(source_input_file),
        "effective_input_file": str(input_file),
        "skip_type": "question_id",
        "skip_question_ids": skip_question_ids,
        "resume": args.resume,
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    status_path = args.output_dir / "run_status.json"
    previous_status = {}
    if args.resume and status_path.exists():
        previous_status = json.loads(status_path.read_text(encoding="utf-8"))
    status = {
        "dataset": args.dataset,
        "method_name": args.method_name,
        "prompt_method": args.prompt_method,
        "rollout": args.rollout,
        "input_file": str(source_input_file),
        "effective_input_file": str(input_file),
        "skip_question_ids": skip_question_ids,
        "state": "running",
        "started_at": previous_status.get(
            "started_at", datetime.now().astimezone().isoformat()
        ),
        "oom_restart_ids": previous_status.get("oom_restart_ids", []),
    }
    if args.resume:
        status["resumed_at"] = datetime.now().astimezone().isoformat()
    write_status(status_path, status)

    command = build_inference_command(
        sys.executable,
        args.dataset,
        input_file,
        args.method_name,
        args.prompt_method,
        args.rollout,
        args.output_dir,
    )
    generation_path = args.output_dir / "generation.json"
    attempt = next_attempt_number(args.output_dir) - 1
    while True:
        attempt += 1
        log_path = args.output_dir / f"inference_attempt_{attempt}.log"
        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=args.repo_root,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            returncode = process.wait()
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if returncode == 0:
            break
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if not is_oom_failure(returncode, log_text):
            status.update(
                state="failed",
                returncode=returncode,
                finished_at=datetime.now().astimezone().isoformat(),
                last_log=str(log_path),
            )
            write_status(status_path, status)
            raise subprocess.CalledProcessError(returncode, command)
        if len(status["oom_restart_ids"]) >= args.max_oom_restarts:
            raise RuntimeError("maximum OOM restart count exceeded")
        skipped_id = mark_first_unfinished_as_skipped(
            generation_path,
            ordered_ids,
            reason="CUDA out of memory; vLLM process restarted",
        )
        status["oom_restart_ids"].append(skipped_id)
        status["attempts"] = attempt
        write_status(status_path, status)

    residual = mark_remaining_failures_as_skipped(generation_path, ordered_ids)
    skipped = {
        "configured_skip_ids": skip_question_ids,
        "oom_restart_ids": status["oom_restart_ids"],
        "other_failed_ids": residual,
    }
    (args.output_dir / "skipped_items.json").write_text(
        json.dumps(skipped, indent=2), encoding="utf-8"
    )
    status.update(
        state="complete",
        attempts=attempt,
        other_failed_ids=residual,
        finished_at=datetime.now().astimezone().isoformat(),
    )
    write_status(status_path, status)
    print(args.output_dir)


if __name__ == "__main__":
    main()
