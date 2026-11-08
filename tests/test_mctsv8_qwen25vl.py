from types import SimpleNamespace
from unittest.mock import mock_open

from hydra import compose, initialize_config_dir
import mllms.base_models as base_models
from prompt_methods.mctsv8.utils_call import generate_message


CONFIG_DIR = "/home/jncsnlp4/SSD2/ZJJ/VReST/configs"


def test_qwen25vl_builds_multimodal_vrest_messages():
    cfg = SimpleNamespace(
        llm=SimpleNamespace(name="Qwen2.5-VL-7B-Instruct")
    )

    messages = generate_message(
        cfg,
        query="Question: what is shown?",
        image_url="/tmp/example.png",
        assistant_prompt="Sub-question 1:",
        step=1,
    )

    assert messages[0]["role"] == "system"
    assert messages[1] == {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": "/tmp/example.png"},
            {"type": "text", "text": "Question: what is shown?"},
        ],
    }
    assert messages[2] == {
        "role": "assistant",
        "content": "Sub-question 1:",
    }


def test_qwen25vl_honors_configured_gpu_memory_utilization(monkeypatch):
    captured = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def get_tokenizer(self):
            return object()

    class FakeProcessor:
        @staticmethod
        def from_pretrained(path):
            return object()

    monkeypatch.setattr(base_models, "LLM", FakeLLM)
    monkeypatch.setattr(base_models, "AutoProcessor", FakeProcessor)

    llm_cfg = SimpleNamespace(
        path="/models/qwen25vl",
        max_input_len=16384,
        max_num_seqs=32,
        gpu_memory_utilization=0.80,
    )
    cfg = SimpleNamespace(
        prompt_method=SimpleNamespace(is_kv_cache=True, attempt_num=3)
    )

    base_models.Qwen2_VL_Model(llm_cfg, cfg)

    assert captured["gpu_memory_utilization"] == 0.80
    assert captured["max_num_seqs"] == 32


def test_qwen25vl_chat_template_continues_assistant_prefix(monkeypatch):
    captured = {}

    class FakeProcessor:
        image_processor = SimpleNamespace(patch_size=14)

        def apply_chat_template(self, messages, **kwargs):
            captured.update(kwargs)
            return "rendered prompt"

    monkeypatch.setattr(
        base_models,
        "process_vision_info",
        lambda *args, **kwargs: (None, None, {}),
    )

    result = base_models.prepare_inputs_for_vllm(
        [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "Sub-question 1:"},
        ],
        FakeProcessor(),
        continue_final_message=True,
    )

    assert result["prompt"] == "rendered prompt"
    assert captured["add_generation_prompt"] is False
    assert captured["continue_final_message"] is True


def test_qwen25vl_get_response_uses_assistant_prefix_continuation(monkeypatch):
    continuation_flags = []

    def fake_prepare(messages, processor, continue_final_message=False):
        continuation_flags.append(continue_final_message)
        return {"prompt": "rendered", "multi_modal_data": {}}

    class FakeLLM:
        def generate(self, inputs, sampling_params):
            output = SimpleNamespace(text="answer")
            return [
                SimpleNamespace(
                    outputs=[output],
                    prompt_token_ids=[1],
                    num_cached_tokens=0,
                )
            ]

    monkeypatch.setattr(base_models, "prepare_inputs_for_vllm", fake_prepare)
    monkeypatch.setattr("builtins.open", mock_open(read_data="-1"))

    model = object.__new__(base_models.Qwen2_VL_Model)
    model.llm_cfg = SimpleNamespace(
        name="Qwen2.5-VL-7B-Instruct",
        temperature=0.7,
        max_output_len=64,
        top_p=0.95,
    )
    model.processor = object()
    model.llm = FakeLLM()

    response = model.get_response(
        messages=[
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "Sub-question 1:"},
        ],
        continue_final_message=True,
    )

    assert response == {"response": ["answer"]}
    assert continuation_flags == [True]


def test_qwen25vl_get_reward_batch_submits_all_prompts_once(monkeypatch):
    prepared_messages = []

    def fake_prepare(messages, processor, continue_final_message=False):
        prepared_messages.append(messages)
        return {"prompt": f"rendered-{len(prepared_messages)}", "multi_modal_data": {}}

    class FakeLLM:
        def __init__(self):
            self.calls = []

        def generate(self, inputs, sampling_params, use_tqdm=False):
            self.calls.append((inputs, sampling_params.n, use_tqdm))
            groups = [
                ["Yes"] * 7 + ["No"] * 3,
                ["Yes"] * 2 + ["No"] * 8,
            ]
            return [
                SimpleNamespace(
                    outputs=[SimpleNamespace(text=text) for text in group]
                )
                for group in groups
            ]

    monkeypatch.setattr(base_models, "prepare_inputs_for_vllm", fake_prepare)

    model = object.__new__(base_models.Qwen2_VL_Model)
    model.cfg = SimpleNamespace(
        reward_method="self_consistency",
        prompt_method=SimpleNamespace(attempt_num=3),
    )
    model.llm_cfg = SimpleNamespace(
        temperature=0.7,
        top_p=0.95,
        max_output_len=64,
    )
    model.processor = object()
    model.llm = FakeLLM()

    messages_batch = [
        [{"role": "user", "content": "first"}],
        [{"role": "user", "content": "second"}],
    ]
    rewards = model.get_reward_batch(messages_batch, n=10, node_depth=1)

    assert len(model.llm.calls) == 1
    assert model.llm.calls[0][0] == [
        {"prompt": "rendered-1", "multi_modal_data": {}},
        {"prompt": "rendered-2", "multi_modal_data": {}},
    ]
    assert model.llm.calls[0][1:] == (10, False)
    assert rewards == [
        {"yes_prob": 0.7, "no_prob": 0.3},
        {"yes_prob": 0.2, "no_prob": 0.8},
    ]


def test_qwen25vl_get_response_batch_submits_distinct_prompts_once(monkeypatch):
    prepared_messages = []

    def fake_prepare(messages, processor, continue_final_message=False):
        prepared_messages.append((messages, continue_final_message))
        return {"prompt": f"rendered-{len(prepared_messages)}", "multi_modal_data": {}}

    class FakeLLM:
        def __init__(self):
            self.calls = []

        def generate(self, inputs, sampling_params, use_tqdm=False):
            self.calls.append((inputs, sampling_params.n, use_tqdm))
            return [
                SimpleNamespace(outputs=[SimpleNamespace(text=f"answer-{index}")])
                for index in range(1, len(inputs) + 1)
            ]

    monkeypatch.setattr(base_models, "prepare_inputs_for_vllm", fake_prepare)

    model = object.__new__(base_models.Qwen2_VL_Model)
    model.llm_cfg = SimpleNamespace(
        name="Qwen2.5-VL-7B-Instruct",
        temperature=0.7,
        max_output_len=64,
        top_p=0.95,
    )
    model.processor = object()
    model.llm = FakeLLM()

    messages_batch = [
        [{"role": "user", "content": "first"}],
        [{"role": "user", "content": "second"}],
        [{"role": "user", "content": "third"}],
    ]
    responses = model.get_response_batch(
        messages_batch,
        n=1,
        continue_final_message=True,
    )

    assert len(model.llm.calls) == 1
    assert model.llm.calls[0] == (
        [
            {"prompt": "rendered-1", "multi_modal_data": {}},
            {"prompt": "rendered-2", "multi_modal_data": {}},
            {"prompt": "rendered-3", "multi_modal_data": {}},
        ],
        1,
        False,
    )
    assert prepared_messages == [
        (messages_batch[0], True),
        (messages_batch[1], True),
        (messages_batch[2], True),
    ]
    assert responses == [["answer-1"], ["answer-2"], ["answer-3"]]


def test_three_mcts_experiment_configs_form_the_expected_ablation():
    expected = {
        "mctsv8_a": (False, False, 0.7),
        "mctsv8_b": (True, False, 0.8),
        "mctsv8_c": (True, True, 0.8),
    }

    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        for config_name, experiment in expected.items():
            cfg = compose(
                config_name="config",
                overrides=[f"prompt_method={config_name}", "llm=qwen25vl_7b"],
            )
            assert cfg.prompt_method.name == config_name
            assert (
                cfg.prompt_method.prune_low_reward_nodes,
                cfg.prompt_method.use_p3pro_for_low_answer,
            ) == experiment[:2]
            threshold = experiment[2]
            assert cfg.prompt_method.prune_reward_threshold == threshold
            assert cfg.prompt_method.p3pro_question_threshold == threshold
            assert cfg.prompt_method.p3pro_answer_threshold == threshold
            assert cfg.prompt_method.embedding_batch_size == 32
            assert cfg.prompt_method.is_p3 is False
            assert cfg.prompt_method.is_p3pro is False
            assert cfg.llm.gpu_memory_utilization == 0.9
