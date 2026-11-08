#!/usr/bin/env python3

import argparse
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class Task:
    name: str
    data_config: str
    prompt_method: str
    input_override: str
    desc: str
    log_name: str
    llm_config: str = "qwen25vl_7b"


@dataclass(frozen=True)
class GpuState:
    index: int
    memory_used_mb: int
    utilization: int


@dataclass
class RunningTask:
    task: Task
    gpu_index: int
    process: subprocess.Popen
    log_handle: object


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXECUTABLE = Path("/home/jncsnlp4/anaconda3/envs/rstar_zjj/bin/python")
TARGET_GPU_INDEX = 0


def default_tasks(project_root=PROJECT_ROOT):
    return [
        Task(
            name="mathvision",
            data_config="custom_mathvision",
            prompt_method="mctsv8_a",
            input_override=(
                "data.run_file="
                f"{project_root}/data/Test/mathvision/"
                "generated_mathvision_qwen2_correct50_incorrect50.json"
            ),
            desc="mathvision_qwen2_correct50_incorrect50",
            log_name="mathvision_mctsv8_a_qwen25vl_100.log",
        ),
        Task(
            name="charxiv",
            data_config="custom_charxiv",
            prompt_method="mctsv8_a",
            input_override=(
                "data.input_file="
                f"{project_root}/data/Test/charxiv/"
                "generated_charxiv_qwen2_correct100_incorrect100.json"
            ),
            desc="charxiv_qwen2_correct100_incorrect100",
            log_name="charxiv_mctsv8_a_qwen25vl_200.log",
        ),
    ]


def mathvista_experiment_tasks(project_root=PROJECT_ROOT):
    input_override = (
        "data.input_file="
        f"{project_root}/data/Test/mathvista/"
        "generated_mathvista_qwen2_correct100_incorrect100.json"
    )
    return [
        Task(
            name="mathvista_prune_t08",
            data_config="mathvista",
            prompt_method="mctsv8_b",
            input_override=input_override,
            desc="mathvista_qwen2_correct100_incorrect100_prune_t08",
            log_name="mathvista_mctsv8_b_prune_t08_qwen25vl_200.log",
        ),
        Task(
            name="mathvista_prune_p3pro_t08",
            data_config="mathvista",
            prompt_method="mctsv8_c",
            input_override=input_override,
            desc="mathvista_qwen2_correct100_incorrect100_prune_p3pro_t08",
            log_name="mathvista_mctsv8_c_prune_p3pro_t08_qwen25vl_200.log",
        ),
    ]


def math500_vpw_tasks(project_root=PROJECT_ROOT):
    return [
        Task(
            name="math500_vpw",
            data_config="math500",
            prompt_method="mctsv8_vpw",
            input_override=(
                "data.input_file="
                "/home/jncsnlp4/SSD2/ZJJ/rStar/data/MATH/test_all.json"
            ),
            desc="math500_qwen25_7b_vpw_rollout8",
            log_name="math500_qwen25_7b_mctsv8_vpw_rollout8.log",
            llm_config="qwen25_7b",
        )
    ]


def math500_vpw_smoke_tasks(project_root=PROJECT_ROOT):
    return [
        Task(
            name="math500_vpw_smoke",
            data_config="math500",
            prompt_method="mctsv8_vpw",
            input_override="data.max_samples=1",
            desc="math500_qwen25_7b_vpw_rollout8_smoke",
            log_name="math500_qwen25_7b_mctsv8_vpw_rollout8_smoke.log",
            llm_config="qwen25_7b",
        )
    ]


def parse_gpu_states(output):
    states = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            raise ValueError(f"Unexpected nvidia-smi row: {line!r}")
        values = [int(field.split()[0]) for field in fields]
        states.append(GpuState(*values))
    return states


def idle_gpu_indices(
    states,
    reserved,
    max_memory_used_mb,
    max_utilization,
):
    return [
        state.index
        for state in states
        if state.index not in reserved
        and state.memory_used_mb <= max_memory_used_mb
        and state.utilization <= max_utilization
    ]


def build_inference_command(task, python_executable, project_root):
    return [
        str(python_executable),
        "-u",
        str(project_root / "inference.py"),
        f"data={task.data_config}",
        f"llm={task.llm_config}",
        f"prompt_method={task.prompt_method}",
        task.input_override,
        f"desc={task.desc}",
    ]


def query_gpu_states():
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={TARGET_GPU_INDEX}",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_gpu_states(result.stdout)


def log(message):
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    print(f"[{timestamp}] {message}", flush=True)


def launch_task(task, gpu_index, run_logs_dir):
    command = build_inference_command(task, PYTHON_EXECUTABLE, PROJECT_ROOT)
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu_index),
            "PYTHONPATH": str(PROJECT_ROOT),
            "TOKENIZERS_PARALLELISM": "true",
        }
    )
    run_logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_logs_dir / task.log_name
    log_handle = log_path.open("a", encoding="utf-8")
    log_handle.write(
        f"\n[{datetime.now().astimezone().isoformat()}] "
        f"launch GPU {gpu_index}: {' '.join(command)}\n"
    )
    log_handle.flush()
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log(
        f"started {task.name} on GPU {gpu_index}: "
        f"pid={process.pid}, log={log_path}"
    )
    return RunningTask(task, gpu_index, process, log_handle)


def run_queue(poll_seconds, max_memory_used_mb, max_utilization, tasks=None):
    pending = list(default_tasks() if tasks is None else tasks)
    running = {}
    failures = []
    run_logs_dir = PROJECT_ROOT / "run_logs"

    log(
        f"queue started with {len(pending)} tasks; scanning GPU "
        f"{TARGET_GPU_INDEX} only; idle means memory <= "
        f"{max_memory_used_mb} MiB and utilization <= {max_utilization}%"
    )
    while pending or running:
        for gpu_index, running_task in list(running.items()):
            return_code = running_task.process.poll()
            if return_code is None:
                continue
            running_task.log_handle.close()
            del running[gpu_index]
            if return_code == 0:
                log(f"completed {running_task.task.name} on GPU {gpu_index}")
            else:
                failures.append((running_task.task.name, return_code))
                log(
                    f"failed {running_task.task.name} on GPU {gpu_index}: "
                    f"exit={return_code}"
                )

        if pending:
            try:
                states = query_gpu_states()
            except (OSError, subprocess.CalledProcessError, ValueError) as error:
                log(f"GPU scan failed: {error}")
            else:
                available = idle_gpu_indices(
                    states,
                    reserved=set(running),
                    max_memory_used_mb=max_memory_used_mb,
                    max_utilization=max_utilization,
                )
                summary = ", ".join(
                    f"GPU {state.index}: {state.memory_used_mb} MiB, "
                    f"{state.utilization}%"
                    for state in states
                )
                log(f"scan: {summary}; available={available}")
                for gpu_index in available:
                    if not pending:
                        break
                    running[gpu_index] = launch_task(
                        pending.pop(0),
                        gpu_index,
                        run_logs_dir,
                    )

        if pending or running:
            time.sleep(poll_seconds)

    if failures:
        log(f"queue finished with failures: {failures}")
        return 1
    log("all queued tasks completed; scheduler stopping")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Run queued VReST jobs when a GPU becomes idle."
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-memory-used-mb", type=int, default=1024)
    parser.add_argument("--max-utilization", type=int, default=10)
    parser.add_argument(
        "--task-set",
        choices=(
            "default",
            "mathvista_prune_p3pro",
            "math500_vpw",
            "math500_vpw_smoke",
        ),
        default="default",
    )
    args = parser.parse_args()
    task_sets = {
        "default": default_tasks,
        "mathvista_prune_p3pro": mathvista_experiment_tasks,
        "math500_vpw": math500_vpw_tasks,
        "math500_vpw_smoke": math500_vpw_smoke_tasks,
    }
    tasks = task_sets[args.task_set]()
    raise SystemExit(
        run_queue(
            poll_seconds=args.poll_seconds,
            max_memory_used_mb=args.max_memory_used_mb,
            max_utilization=args.max_utilization,
            tasks=tasks,
        )
    )


if __name__ == "__main__":
    main()
