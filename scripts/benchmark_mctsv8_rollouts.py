#!/usr/bin/env python3
"""Run deterministic MCTSv8 rollout-scaling benchmarks."""

import argparse
import csv
import json
import math
import os
import random
import signal
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_REPO_ROOT = Path("/home/jncsnlp4/SSD2/ZJJ/VReST")
DEFAULT_INPUT_FILE = DEFAULT_REPO_ROOT / "data/Test/mathvista/generated_mathvista_qwen2_correct100_incorrect100.json"
DEFAULT_OUTPUT_ROOT = DEFAULT_REPO_ROOT / "newoutput"
DEFAULT_ROLLOUTS = [2, 4, 6, 8, 10]
SKIPPED_RESPONSE = "Skipped after runtime failure"
SKIPPED_STATUS = "skipped_runtime_failure"


def build_benchmark_subset(data, sample_size, seed, warmup_id=None):
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1")
    if len(data) < sample_size + 1:
        raise ValueError("input data must include one warm-up item plus measured items")

    if warmup_id is None:
        selected_ids = random.Random(seed).sample(list(data), sample_size + 1)
    else:
        warmup_id = str(warmup_id)
        if warmup_id not in data:
            raise ValueError(f"warmup_id is not present in input data: {warmup_id}")
        measured_pool = [item_id for item_id in data if item_id != warmup_id]
        selected_ids = [warmup_id] + random.Random(seed).sample(
            measured_pool, sample_size
        )
    subset = {item_id: data[item_id] for item_id in selected_ids}
    manifest = {
        "seed": seed,
        "warmup_id": selected_ids[0],
        "measured_ids": selected_ids[1:],
    }
    return subset, manifest


def prepare_empty_run_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"benchmark run directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _nearest_rank_percentile(values, percentile):
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def is_oom_failure(returncode, log_text):
    normalized = log_text.casefold()
    oom_markers = (
        "cuda out of memory",
        "outofmemoryerror",
        "cublas_status_alloc_failed",
        "cuda error: out of memory",
    )
    return returncode in (-9, 137) or any(marker in normalized for marker in oom_markers)


def _load_generation(path):
    path = Path(path)
    if not path.exists():
        raise RuntimeError("inference failed before generation.json was created")
    with path.open("r", encoding="utf-8") as handle:
        return path, json.load(handle)


def _write_generation(path, generation):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(generation, indent=4), encoding="utf-8")
    temporary_path.replace(path)


def _is_completed_or_skipped(item):
    return isinstance(item, dict) and (
        isinstance(item.get("time"), (int, float))
        or item.get("answer_extract_status") == SKIPPED_STATUS
    )


def mark_first_unfinished_as_skipped(path, ordered_ids, reason):
    path, generation = _load_generation(path)
    for item_id in ordered_ids:
        item = generation.get(item_id)
        if _is_completed_or_skipped(item):
            continue
        if not isinstance(item, dict):
            item = {}
            generation[item_id] = item
        item.update(
            {
                "response": SKIPPED_RESPONSE,
                "raw_response": None,
                "answer_extract_status": SKIPPED_STATUS,
                "runtime_failure": reason,
            }
        )
        _write_generation(path, generation)
        return item_id
    raise RuntimeError("inference failed but no unfinished item could be identified")


def mark_remaining_failures_as_skipped(path, ordered_ids):
    path, generation = _load_generation(path)
    skipped_ids = []
    for item_id in ordered_ids:
        item = generation.get(item_id)
        if _is_completed_or_skipped(item):
            continue
        if not isinstance(item, dict):
            item = {}
            generation[item_id] = item
        reason = item.get("error") or "missing numeric time after inference completed"
        item.update(
            {
                "response": SKIPPED_RESPONSE,
                "raw_response": None,
                "answer_extract_status": SKIPPED_STATUS,
                "runtime_failure": reason,
            }
        )
        skipped_ids.append(item_id)
    if skipped_ids:
        _write_generation(path, generation)
    return skipped_ids


def summarize_generation(path, rollout, warmup_id):
    with Path(path).open("r", encoding="utf-8") as handle:
        generation = json.load(handle)

    warmup = generation.get(warmup_id)
    if not isinstance(warmup, dict) or not isinstance(warmup.get("time"), (int, float)):
        detail = warmup.get("error") if isinstance(warmup, dict) else "missing item"
        raise ValueError(f"warm-up item failed: {detail}")

    measured = [
        (item_id, item)
        for item_id, item in generation.items()
        if item_id != warmup_id
    ]
    skipped_ids = [
        item_id
        for item_id, item in measured
        if item.get("answer_extract_status") == SKIPPED_STATUS
    ]
    completed = [
        item
        for item_id, item in measured
        if item_id not in skipped_ids
    ]
    times = [
        float(item["time"])
        for item in completed
        if isinstance(item.get("time"), (int, float))
    ]
    if len(times) != len(completed):
        raise ValueError(
            f"missing numeric time values: measured={len(completed)}, timed={len(times)}"
        )
    if not times:
        raise ValueError("no completed measured items remain after runtime skips")

    parse_failures = sum(
        item.get("answer_extract_status") != "ok" for item in completed
    )
    total_seconds = sum(times)
    sample_count = len(times)
    return {
        "rollout": rollout,
        "sample_count": sample_count,
        "total_seconds": total_seconds,
        "mean_seconds": statistics.fmean(times),
        "median_seconds": statistics.median(times),
        "p90_seconds": _nearest_rank_percentile(times, 0.90),
        "questions_per_hour": sample_count * 3600.0 / total_seconds,
        "parse_failure_count": parse_failures,
        "parse_failure_rate": parse_failures / sample_count,
        "skipped_count": len(skipped_ids),
        "skipped_ids": skipped_ids,
    }


def add_scaling_ratios(summaries):
    baseline = next(
        (summary["mean_seconds"] for summary in summaries if summary["rollout"] == 2),
        None,
    )
    if baseline is None or baseline <= 0:
        raise ValueError("rollout 2 with a positive mean is required as the scaling baseline")
    return [
        {**summary, "time_vs_rollout_2": summary["mean_seconds"] / baseline}
        for summary in summaries
    ]


def validate_rollouts(rollouts):
    if not rollouts or 2 not in rollouts:
        raise ValueError("rollouts must include 2 as the scaling baseline")
    if any(not isinstance(value, int) or value <= 0 for value in rollouts):
        raise ValueError("rollouts must be positive integers")
    if len(set(rollouts)) != len(rollouts):
        raise ValueError("rollouts must not contain duplicates")
    return list(rollouts)


def _write_reports(root, summaries, manifest):
    (root / "rollout_scaling.json").write_text(
        json.dumps({"manifest": manifest, "results": summaries}, indent=2),
        encoding="utf-8",
    )

    with (root / "rollout_scaling.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    lines = [
        "# MCTSv8 Rollout Test-Time Scaling",
        "",
        "| Rollouts | Samples | Mean (s) | Median (s) | P90 (s) | Questions/hour | vs rollout=2 | Parse failures | Skipped |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            "| {rollout} | {sample_count} | {mean_seconds:.3f} | {median_seconds:.3f} | "
            "{p90_seconds:.3f} | {questions_per_hour:.3f} | {time_vs_rollout_2:.3f}x | "
            "{parse_failure_count} ({parse_failure_rate:.1%}) | {skipped_count} |".format(**item)
        )
    (root / "rollout_scaling.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-id", default="2")
    parser.add_argument("--rollouts", type=int, nargs="+", default=DEFAULT_ROLLOUTS)
    parser.add_argument("--max-oom-restarts", type=int, default=20)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    args.rollouts = validate_rollouts(args.rollouts)

    with args.input_file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    subset, manifest = build_benchmark_subset(
        data, args.sample_size, args.seed, warmup_id=args.warmup_id
    )

    run_name = args.run_name or datetime.now().strftime("rollout_scaling_%Y%m%d_%H%M%S")
    benchmark_root = prepare_empty_run_directory(args.output_root / run_name)
    subset_path = benchmark_root / "benchmark_subset.json"
    subset_path.write_text(json.dumps(subset, indent=2), encoding="utf-8")
    (benchmark_root / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    summaries = []
    for rollout in args.rollouts:
        run_dir = prepare_empty_run_directory(benchmark_root / f"rollout_{rollout}")
        command = [
            sys.executable,
            "inference.py",
            "data=mathvista",
            "llm=qwen25vl_7b",
            "prompt_method=mctsv8_a",
            f"data.input_file={subset_path}",
            f"desc=batched_rollout_scaling_r{rollout}",
            f"prompt_method.mcts_rollouts={rollout}",
            f"hydra.run.dir={run_dir}",
        ]
        generation_path = run_dir / "generation.json"
        ordered_ids = list(subset)
        oom_skips = []
        attempt = 0
        while True:
            attempt += 1
            log_path = run_dir / f"benchmark_attempt_{attempt}.log"
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
                raise subprocess.CalledProcessError(returncode, command)
            if len(oom_skips) >= args.max_oom_restarts:
                raise RuntimeError(
                    f"rollout {rollout} exceeded {args.max_oom_restarts} OOM restarts"
                )
            reason = "CUDA out of memory; vLLM process restarted"
            skipped_id = mark_first_unfinished_as_skipped(
                generation_path, ordered_ids, reason=reason
            )
            oom_skips.append(skipped_id)

        residual_skips = mark_remaining_failures_as_skipped(
            generation_path, ordered_ids
        )
        skipped_records = {
            "oom_restart_ids": oom_skips,
            "other_failed_ids": residual_skips,
        }
        (run_dir / "skipped_items.json").write_text(
            json.dumps(skipped_records, indent=2), encoding="utf-8"
        )
        summaries.append(
            summarize_generation(
                generation_path,
                rollout=rollout,
                warmup_id=manifest["warmup_id"],
            )
        )
        _write_reports(benchmark_root, add_scaling_ratios(summaries), manifest)

    print(benchmark_root)


if __name__ == "__main__":
    main()
