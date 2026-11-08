from types import SimpleNamespace

import pytest

import mllms.base_models as base_models
from prompt_methods.mctsv8.utils_call import generate_message


def _llm_cfg(**overrides):
    values = {
        "name": "Qwen2.5-7B-Instruct",
        "path": "/models/qwen25",
        "call_method": "local",
        "max_input_len": 8192,
        "max_num_seqs": 32,
        "max_output_len": 128,
        "temperature": 0.7,
        "top_p": 0.95,
        "gpu_memory_utilization": 0.9,
        "text_only": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _cfg():
    return SimpleNamespace(
        llm=_llm_cfg(),
        reward_method="self_consistency",
        prompt_method=SimpleNamespace(is_kv_cache=True, attempt_num=1),
    )


class RecordingTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return messages[-1]["content"]


class FakeLLM:
    def __init__(self, responses, tokenizer=None):
        self.responses = responses
        self.tokenizer = tokenizer or RecordingTokenizer()
        self.calls = []

    def get_tokenizer(self):
        return self.tokenizer

    def generate(self, inputs, sampling_params, use_tqdm=False):
        self.calls.append((inputs, sampling_params, use_tqdm))
        return [
            SimpleNamespace(
                outputs=[SimpleNamespace(text=text) for text in response_group]
            )
            for response_group in self.responses
        ]


def test_generate_message_omits_image_for_qwen25_text_and_keeps_continuation():
    cfg = _cfg()

    messages = generate_message(
        cfg,
        "Solve the problem",
        "/must/not/be/used.png",
        assistant_prompt="Sub-question 1:",
        step=1,
    )

    assert messages == [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Solve the problem"},
        {"role": "assistant", "content": "Sub-question 1:"},
    ]


def test_qwen25_local_constructor_accepts_main_cfg_and_keeps_vpw_cache_setting(
    monkeypatch,
):
    captured = {}
    fake_llm = FakeLLM([])

    def build_llm(**kwargs):
        captured.update(kwargs)
        return fake_llm

    monkeypatch.setattr(base_models, "LLM", build_llm)
    cfg = _cfg()

    model = base_models.Qwen25_Model(cfg.llm, cfg)

    assert model.cfg is cfg
    assert model.tokenizer is fake_llm.tokenizer
    assert captured == {
        "model": "/models/qwen25",
        "max_model_len": 8192,
        "max_num_seqs": 32,
        "gpu_memory_utilization": 0.9,
        "enable_prefix_caching": True,
    }


def test_qwen25_response_batch_preserves_order_n_stop_and_continuation():
    tokenizer = RecordingTokenizer()
    fake_llm = FakeLLM(
        [["first-a", "first-b"], ["second-a", "second-b"]],
        tokenizer=tokenizer,
    )
    model = base_models.Qwen25_Model.__new__(base_models.Qwen25_Model)
    model.llm_cfg = _llm_cfg()
    model.cfg = _cfg()
    model.llm = fake_llm
    model.tokenizer = tokenizer
    messages_batch = [
        [{"role": "assistant", "content": "first-prefix"}],
        [{"role": "assistant", "content": "second-prefix"}],
    ]

    responses = model.get_response_batch(
        messages_batch,
        n=2,
        continue_final_message=True,
        stop_token="Sub-question 2:",
    )

    assert responses == [
        ["first-a", "first-b"],
        ["second-a", "second-b"],
    ]
    assert [call[1]["continue_final_message"] for call in tokenizer.calls] == [
        True,
        True,
    ]
    inputs, sampling_params, use_tqdm = fake_llm.calls[0]
    assert inputs == ["first-prefix", "second-prefix"]
    assert sampling_params.n == 2
    assert sampling_params.stop == ["Sub-question 2:"]
    assert use_tqdm is False


def test_qwen25_reward_batch_returns_ordered_self_consistency_rates():
    tokenizer = RecordingTokenizer()
    fake_llm = FakeLLM(
        [
            ["Yes", "yes, useful", "No", "Yes"],
            ["No", "No", "yes", "No"],
        ],
        tokenizer=tokenizer,
    )
    model = base_models.Qwen25_Model.__new__(base_models.Qwen25_Model)
    model.llm_cfg = _llm_cfg()
    model.cfg = _cfg()
    model.llm = fake_llm
    model.tokenizer = tokenizer
    messages_batch = [
        [{"role": "user", "content": "reward one"}],
        [{"role": "user", "content": "reward two"}],
    ]

    rewards = model.get_reward_batch(messages_batch, n=4, node_depth=3)
    single_reward = model.get_reward(messages=messages_batch[0], n=4, node_depth=3)

    assert rewards == [
        {"yes_prob": 0.75, "no_prob": 0.25},
        {"yes_prob": 0.25, "no_prob": 0.75},
    ]
    assert single_reward == {"yes_prob": 0.75, "no_prob": 0.25}
