#!/usr/bin/env python3
"""Run baseline MathVision then MathVista sequentially on one visible GPU."""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path("/home/jncsnlp4/SSD2/ZJJ/VReST")


def write_status(path, payload):
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "newoutput")
    parser.add_argument("--method-name", default="baseline")
    parser.add_argument("--rollout", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    method_root = args.output_root / args.method_name
    method_root.mkdir(parents=True, exist_ok=True)
    status_path = method_root / "scheduler_status.json"
    status = {
        "state": "running",
        "method_name": args.method_name,
        "rollout": args.rollout,
        "datasets": ["mathvision", "mathvista"],
        "started_at": datetime.now().astimezone().isoformat(),
        "runs": {},
    }
    write_status(status_path, status)

    for dataset in status["datasets"]:
        output_dir = method_root / dataset
        status["current_dataset"] = dataset
        status["runs"][dataset] = {
            "state": "running",
            "started_at": datetime.now().astimezone().isoformat(),
            "output_dir": str(output_dir),
        }
        write_status(status_path, status)
        command = [
            sys.executable,
            "scripts/run_resumable_inference.py",
            f"--repo-root={args.repo_root}",
            f"--dataset={dataset}",
            f"--method-name={args.method_name}",
            f"--rollout={args.rollout}",
            f"--output-dir={output_dir}",
        ]
        returncode = subprocess.run(command, cwd=args.repo_root).returncode
        status["runs"][dataset].update(
            state="complete" if returncode == 0 else "failed",
            returncode=returncode,
            finished_at=datetime.now().astimezone().isoformat(),
        )
        write_status(status_path, status)
        if returncode != 0:
            status.update(
                state="failed",
                finished_at=datetime.now().astimezone().isoformat(),
            )
            write_status(status_path, status)
            raise SystemExit(returncode)

    status.update(
        state="complete",
        current_dataset=None,
        finished_at=datetime.now().astimezone().isoformat(),
    )
    write_status(status_path, status)


if __name__ == "__main__":
    main()
