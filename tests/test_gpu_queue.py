from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

import scripts.gpu_queue as gpu_queue
from scripts.gpu_queue import (
    GpuState,
    RunningTask,
    Task,
    build_inference_command,
    idle_gpu_indices,
    parse_gpu_states,
)


def test_parse_gpu_states_reads_nvidia_smi_csv():
    states = parse_gpu_states("0, 512, 2\n1, 6047, 100\n")

    assert [(state.index, state.memory_used_mb, state.utilization) for state in states] == [
        (0, 512, 2),
        (1, 6047, 100),
    ]


def test_idle_gpu_indices_excludes_busy_and_reserved_gpus():
    states = parse_gpu_states("0, 512, 2\n1, 128, 0\n2, 6047, 100\n")

    assert idle_gpu_indices(
        states,
        reserved={1},
        max_memory_used_mb=1024,
        max_utilization=10,
    ) == [0]


def test_query_gpu_states_only_scans_gpu_zero(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        return SimpleNamespace(stdout="0, 512, 2\n")

    monkeypatch.setattr(gpu_queue.subprocess, "run", fake_run)

    states = gpu_queue.query_gpu_states()

    assert captured["command"] == [
        "nvidia-smi",
        "--id=0",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    assert [state.index for state in states] == [0]


def test_build_mathvision_command_uses_custom_subset_file():
    task = Task(
        name="mathvision",
        data_config="custom_mathvision",
        prompt_method="mctsv8_a",
        input_override="data.run_file=/datasets/mathvision.json",
        desc="mathvision_subset",
        log_name="mathvision.log",
    )

    command = build_inference_command(
        task,
        python_executable=Path("/env/bin/python"),
        project_root=Path("/repo"),
    )

    assert command == [
        "/env/bin/python",
        "-u",
        "/repo/inference.py",
        "data=custom_mathvision",
        "llm=qwen25vl_7b",
        "prompt_method=mctsv8_a",
        "data.run_file=/datasets/mathvision.json",
        "desc=mathvision_subset",
    ]


def test_build_charxiv_command_uses_custom_input_file():
    task = Task(
        name="charxiv",
        data_config="custom_charxiv",
        prompt_method="mctsv8_a",
        input_override="data.input_file=/datasets/charxiv.json",
        desc="charxiv_subset",
        log_name="charxiv.log",
    )

    command = build_inference_command(
        task,
        python_executable=Path("/env/bin/python"),
        project_root=Path("/repo"),
    )

    assert command[3:] == [
        "data=custom_charxiv",
        "llm=qwen25vl_7b",
        "prompt_method=mctsv8_a",
        "data.input_file=/datasets/charxiv.json",
        "desc=charxiv_subset",
    ]


def test_mathvista_experiment_tasks_run_pruning_then_pruning_with_p3pro():
    tasks = gpu_queue.mathvista_experiment_tasks(Path("/repo"))

    assert [
        (
            task.name,
            task.data_config,
            task.prompt_method,
            task.input_override,
            task.desc,
        )
        for task in tasks
    ] == [
        (
            "mathvista_prune_t08",
            "mathvista",
            "mctsv8_b",
            "data.input_file=/repo/data/Test/mathvista/"
            "generated_mathvista_qwen2_correct100_incorrect100.json",
            "mathvista_qwen2_correct100_incorrect100_prune_t08",
        ),
        (
            "mathvista_prune_p3pro_t08",
            "mathvista",
            "mctsv8_c",
            "data.input_file=/repo/data/Test/mathvista/"
            "generated_mathvista_qwen2_correct100_incorrect100.json",
            "mathvista_qwen2_correct100_incorrect100_prune_p3pro_t08",
        ),
    ]


def test_math500_vpw_task_builds_requested_text_model_command():
    tasks = gpu_queue.math500_vpw_tasks(Path("/repo"))

    assert len(tasks) == 1
    task = tasks[0]
    assert (
        task.name,
        task.data_config,
        task.llm_config,
        task.prompt_method,
        task.input_override,
        task.desc,
        task.log_name,
    ) == (
        "math500_vpw",
        "math500",
        "qwen25_7b",
        "mctsv8_vpw",
        "data.input_file=/home/jncsnlp4/SSD2/ZJJ/rStar/data/MATH/test_all.json",
        "math500_qwen25_7b_vpw_rollout8",
        "math500_qwen25_7b_mctsv8_vpw_rollout8.log",
    )
    assert build_inference_command(
        task,
        python_executable=Path("/env/bin/python"),
        project_root=Path("/repo"),
    ) == [
        "/env/bin/python",
        "-u",
        "/repo/inference.py",
        "data=math500",
        "llm=qwen25_7b",
        "prompt_method=mctsv8_vpw",
        "data.input_file=/home/jncsnlp4/SSD2/ZJJ/rStar/data/MATH/test_all.json",
        "desc=math500_qwen25_7b_vpw_rollout8",
    ]


def test_math500_vpw_smoke_task_limits_run_to_one_record():
    task = gpu_queue.math500_vpw_smoke_tasks(Path("/repo"))[0]

    assert task.input_override == "data.max_samples=1"
    assert task.desc == "math500_qwen25_7b_vpw_rollout8_smoke"
    assert build_inference_command(
        task,
        python_executable=Path("/env/bin/python"),
        project_root=Path("/repo"),
    )[3:] == [
        "data=math500",
        "llm=qwen25_7b",
        "prompt_method=mctsv8_vpw",
        "data.max_samples=1",
        "desc=math500_qwen25_7b_vpw_rollout8_smoke",
    ]


def test_mathvista_experiment_configs_use_point_eight_thresholds():
    config_dir = Path(__file__).parents[1] / "configs" / "prompt_method"
    prune = OmegaConf.load(config_dir / "mctsv8_b.yaml")
    prune_p3pro = OmegaConf.load(config_dir / "mctsv8_c.yaml")

    assert prune.prune_low_reward_nodes is True
    assert prune.use_p3pro_for_low_answer is False
    assert prune.prune_reward_threshold == 0.8

    assert prune_p3pro.prune_low_reward_nodes is True
    assert prune_p3pro.use_p3pro_for_low_answer is True
    assert prune_p3pro.prune_reward_threshold == 0.8
    assert prune_p3pro.p3pro_question_threshold == 0.8
    assert prune_p3pro.p3pro_answer_threshold == 0.8


def test_run_queue_stops_after_all_tasks_complete(monkeypatch, capsys):
    task = Task(
        name="mathvision",
        data_config="custom_mathvision",
        prompt_method="mctsv8_a",
        input_override="data.run_file=/datasets/mathvision.json",
        desc="mathvision_subset",
        log_name="mathvision.log",
    )

    class CompletedProcess:
        def poll(self):
            return 0

    class LogHandle:
        closed = False

        def close(self):
            self.closed = True

    log_handle = LogHandle()

    monkeypatch.setattr(gpu_queue, "default_tasks", lambda: [task])
    monkeypatch.setattr(
        gpu_queue,
        "query_gpu_states",
        lambda: [GpuState(index=0, memory_used_mb=0, utilization=0)],
    )
    monkeypatch.setattr(
        gpu_queue,
        "launch_task",
        lambda queued_task, gpu_index, run_logs_dir: RunningTask(
            queued_task,
            gpu_index,
            CompletedProcess(),
            log_handle,
        ),
    )
    monkeypatch.setattr(gpu_queue.time, "sleep", lambda _: None)

    assert gpu_queue.run_queue(30, 1024, 10) == 0
    assert log_handle.closed
    assert "all queued tasks completed; scheduler stopping" in capsys.readouterr().out
