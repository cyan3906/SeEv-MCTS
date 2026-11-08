#!/usr/bin/env python3
"""Build deterministic correct/incorrect Qwen2 baseline subsets."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = REPO_ROOT / "outputs" / "desc=baseline"


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return data


def load_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["id"])] = row
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def id_sort_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def choose_ids(
    scores: dict[str, dict[str, Any]],
    correct_count: int,
    incorrect_count: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    correct = sorted(
        (key for key, value in scores.items() if value.get("llm_score") is True),
        key=id_sort_key,
    )
    incorrect = sorted(
        (key for key, value in scores.items() if value.get("llm_score") is False),
        key=id_sort_key,
    )
    if len(correct) < correct_count or len(incorrect) < incorrect_count:
        raise ValueError(
            f"Insufficient scored examples: correct={len(correct)}, incorrect={len(incorrect)}"
        )
    rng = random.Random(seed)
    return (
        sorted(rng.sample(correct, correct_count), key=id_sort_key),
        sorted(rng.sample(incorrect, incorrect_count), key=id_sort_key),
    )


def validate_scores_match_generation(
    generation: dict[str, Any], scores: dict[str, Any], name: str
) -> None:
    if set(generation) != set(scores):
        raise ValueError(f"{name}: generation and score IDs differ")
    mismatched = [
        key
        for key in generation
        if generation[key].get("response") != scores[key].get("response")
    ]
    if mismatched:
        raise ValueError(
            f"{name}: {len(mismatched)} score records do not match baseline responses"
        )
    missing_labels = [
        key for key, value in scores.items() if value.get("llm_score") not in (True, False)
    ]
    if missing_labels:
        raise ValueError(f"{name}: {len(missing_labels)} records lack boolean llm_score")


def copy_images(
    ids: list[str],
    records: dict[str, dict[str, Any]],
    source_for: Callable[[str, dict[str, Any]], Path],
    target_for: Callable[[str, dict[str, Any]], Path],
) -> int:
    copied: set[Path] = set()
    for key in ids:
        source = source_for(key, records[key])
        target = target_for(key, records[key])
        if not source.is_file():
            raise FileNotFoundError(f"Missing image for {key}: {source}")
        if target in copied:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.add(target)
    return len(copied)


def baseline_annotations(
    ids: list[str], scores: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "split": "correct" if scores[key]["llm_score"] else "incorrect",
            "answer": scores[key].get("answer"),
            "response": scores[key].get("response"),
            "extracted_answer": scores[key].get("extracted_answer"),
            "llm_score": scores[key]["llm_score"],
        }
        for key in ids
    }


def build_dataset(
    *,
    name: str,
    records: dict[str, dict[str, Any]],
    generation_path: Path,
    score_path: Path,
    output_dir: Path,
    correct_count: int,
    incorrect_count: int,
    seed: int,
    combined_filename: str,
    correct_filename: str,
    incorrect_filename: str,
    write_records: Callable[[Path, list[dict[str, Any]], list[str]], None],
    source_image_for: Callable[[str, dict[str, Any]], Path],
    target_image_for: Callable[[str, dict[str, Any]], Path],
) -> dict[str, Any]:
    generation = load_json(generation_path)
    scores = load_json(score_path)
    validate_scores_match_generation(generation, scores, name)
    if set(records) != set(generation):
        raise ValueError(f"{name}: source dataset IDs differ from baseline IDs")

    correct_ids, incorrect_ids = choose_ids(
        scores, correct_count, incorrect_count, seed
    )
    selected_ids = sorted(correct_ids + incorrect_ids, key=id_sort_key)

    write_records(output_dir / "data" / combined_filename, [records[k] for k in selected_ids], selected_ids)
    write_records(output_dir / "data" / correct_filename, [records[k] for k in correct_ids], correct_ids)
    write_records(output_dir / "data" / incorrect_filename, [records[k] for k in incorrect_ids], incorrect_ids)
    write_json(
        output_dir / "baseline_results.json",
        baseline_annotations(selected_ids, scores),
    )

    image_count = copy_images(
        selected_ids, records, source_image_for, target_image_for
    )
    manifest = {
        "dataset": name,
        "seed": seed,
        "baseline_model": "Qwen2-VL-7B-Instruct",
        "prompt_method": "mctsv8",
        "extract_field": "max_vote_terminal",
        "reward_method": "self_consistency",
        "generation_source": str(generation_path.relative_to(REPO_ROOT)),
        "score_source": str(score_path.relative_to(REPO_ROOT)),
        "available": {
            "total": len(scores),
            "correct": sum(v["llm_score"] is True for v in scores.values()),
            "incorrect": sum(v["llm_score"] is False for v in scores.values()),
        },
        "selected": {
            "total": len(selected_ids),
            "correct": len(correct_ids),
            "incorrect": len(incorrect_ids),
            "correct_ids": correct_ids,
            "incorrect_ids": incorrect_ids,
            "unique_images": image_count,
        },
    }
    write_json(output_dir / "selection_manifest.json", manifest)
    return manifest


def write_dict_records(path: Path, rows: list[dict[str, Any]], ids: list[str]) -> None:
    write_json(path, dict(zip(ids, rows)))


def write_list_records(path: Path, rows: list[dict[str, Any]], ids: list[str]) -> None:
    del ids
    write_jsonl(path, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "data" / "Test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force", action="store_true", help="replace an existing non-empty output root"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        if not args.force:
            raise FileExistsError(
                f"Output root is not empty: {output_root}; pass --force to replace it"
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    mathvista_generation = BASELINE_ROOT / (
        "data=mathvista/llm=Qwen2-VL-7B-Instruct/prompt_method=mctsv8/generation.json"
    )
    mathvista_scores = BASELINE_ROOT / (
        "data=mathvista/llm=Qwen2-VL-7B-Instruct/prompt_method=mctsv8/"
        "ext_llm=qwen3/method=ans_and_score/field=max_vote_terminal/n=10/"
        "reward_method=self_consistency/extraction.json"
    )
    mathvision_generation = BASELINE_ROOT / (
        "data=mathvision/llm=Qwen2-VL-7B-Instruct/prompt_method=mctsv8/generation.json"
    )
    mathvision_scores = BASELINE_ROOT / (
        "data=mathvision/llm=Qwen2-VL-7B-Instruct/prompt_method=mctsv8/"
        "ext_llm=qwen3/method=ans_and_score/field=max_vote_terminal/n=10/"
        "reward_method=self_consistency/scores.json"
    )
    charxiv_generation = BASELINE_ROOT / (
        "data=charxiv/llm=Qwen2-VL-7B-Instruct/prompt_method=mctsv8/"
        "gen-reasoning_val.json"
    )
    charxiv_scores = BASELINE_ROOT / (
        "data=charxiv/llm=Qwen2-VL-7B-Instruct/prompt_method=mctsv8/"
        "ext_llm=qwen3/method=ans_and_score/field=max_vote_terminal/n=10/"
        "reward_method=self_consistency/nothink-scores-reasoning_val.json"
    )

    mathvista_records = load_json(REPO_ROOT / "data/MathVista/data/testmini.json")
    mathvision_records = load_jsonl_by_id(REPO_ROOT / "data/MATH-V/data/testmini.jsonl")
    charxiv_records = load_json(REPO_ROOT / "data/CharXiv/data/reasoning_val.json")

    manifests = {}
    manifests["mathvista"] = build_dataset(
        name="mathvista",
        records=mathvista_records,
        generation_path=mathvista_generation,
        score_path=mathvista_scores,
        output_dir=output_root / "MathVista",
        correct_count=100,
        incorrect_count=100,
        seed=args.seed,
        combined_filename="testmini.json",
        correct_filename="testmini_correct.json",
        incorrect_filename="testmini_incorrect.json",
        write_records=write_dict_records,
        source_image_for=lambda key, row: REPO_ROOT
        / "data/MathVista/data/images/testmini/ALL"
        / Path(row["image"]).name,
        target_image_for=lambda key, row: output_root
        / "MathVista/data/images/testmini/ALL"
        / Path(row["image"]).name,
    )
    manifests["mathvision"] = build_dataset(
        name="mathvision",
        records=mathvision_records,
        generation_path=mathvision_generation,
        score_path=mathvision_scores,
        output_dir=output_root / "MATH-V",
        correct_count=50,
        incorrect_count=50,
        seed=args.seed,
        combined_filename="testmini.jsonl",
        correct_filename="testmini_correct.jsonl",
        incorrect_filename="testmini_incorrect.jsonl",
        write_records=write_list_records,
        source_image_for=lambda key, row: REPO_ROOT / "data/MATH-V" / row["image"],
        target_image_for=lambda key, row: output_root / "MATH-V" / row["image"],
    )
    manifests["charxiv"] = build_dataset(
        name="charxiv",
        records=charxiv_records,
        generation_path=charxiv_generation,
        score_path=charxiv_scores,
        output_dir=output_root / "CharXiv",
        correct_count=100,
        incorrect_count=100,
        seed=args.seed,
        combined_filename="reasoning_val.json",
        correct_filename="reasoning_val_correct.json",
        incorrect_filename="reasoning_val_incorrect.json",
        write_records=write_dict_records,
        source_image_for=lambda key, row: REPO_ROOT
        / "data/CharXiv/images"
        / f"{row['figure_id']}.jpg",
        target_image_for=lambda key, row: output_root
        / "CharXiv/images"
        / f"{row['figure_id']}.jpg",
    )

    write_json(output_root / "manifest.json", manifests)
    (output_root / "README.md").write_text(
        """# Qwen2 baseline test subsets

These subsets were sampled with seed 42 from the Qwen2-VL-7B-Instruct
`mctsv8` baseline using the saved `max_vote_terminal` self-consistency
evaluation labels.

- MathVista: 100 correct + 100 incorrect
- MathVision: 50 correct + 50 incorrect
- CharXiv: 100 correct + 100 incorrect

Each dataset contains a combined source file, separate correct/incorrect source
files, copied images, `baseline_results.json`, and `selection_manifest.json`.
Run `python scripts/build_test_subsets.py` from the repository root to rebuild
the exact same selection.
""",
        encoding="utf-8",
    )
    print(json.dumps(manifests, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
