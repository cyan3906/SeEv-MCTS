import json
from types import SimpleNamespace

from prompt_methods.mctsv8_vpw.generation import (
    build_diversity_messages,
    generate_diverse_candidates,
    score_candidates_batch,
)


def make_cfg(reward_prompt_batch_size=32):
    return SimpleNamespace(
        llm=SimpleNamespace(name="Qwen2.5-VL-7B-Instruct"),
        prompt_method=SimpleNamespace(
            return_num=10,
            reward_prompt_batch_size=reward_prompt_batch_size,
        ),
    )


def test_three_diversity_prompts_include_all_normal_candidates_and_roles():
    normal = [(f"normal question {i}", f"normal answer {i}") for i in range(5)]

    messages_batch = build_diversity_messages(
        make_cfg(),
        image_url="/tmp/image.png",
        question="main problem",
        sub_questions=["history question"],
        sub_answers=["history answer"],
        normal_candidates=normal,
        depth=2,
    )
    rendered = [json.dumps(messages, ensure_ascii=False) for messages in messages_batch]

    assert len(rendered) == 3
    assert all("main problem" in prompt for prompt in rendered)
    assert all("normal question 0" in prompt for prompt in rendered)
    assert all("normal question 4" in prompt for prompt in rendered)
    assert len(set(rendered)) == 3


def test_diversity_generation_uses_one_batch_and_drops_malformed_output():
    class FakeModel:
        def __init__(self):
            self.calls = []

        def get_response_batch(self, messages_batch, **kwargs):
            self.calls.append((messages_batch, kwargs))
            return [
                ["Sub-question 2: diverse one?\nAnswer 2: first"],
                ["Sub-question 2: missing answer?"],
                ["Sub-question 2: diverse three?\nAnswer 2: third"],
            ]

    model = FakeModel()
    candidates = generate_diverse_candidates(
        make_cfg(),
        image_url="/tmp/image.png",
        model=model,
        question="main problem",
        sub_questions=["history question"],
        sub_answers=["history answer"],
        normal_candidates=[("normal", "answer")],
        depth=2,
    )

    assert len(model.calls) == 1
    assert model.calls[0][1] == {"n": 1, "continue_final_message": False}
    assert candidates == [
        ("diverse one?", "first"),
        ("diverse three?", "third"),
    ]


def test_candidate_scoring_flattens_question_and_answer_prompts_in_one_phase():
    class FakeModel:
        def __init__(self):
            self.calls = []

        def get_reward_batch(self, messages_batch, **kwargs):
            self.calls.append((messages_batch, kwargs))
            yes = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
            return [
                {"yes_prob": value, "no_prob": 1.0 - value}
                for value in yes
            ]

    model = FakeModel()
    rewards = score_candidates_batch(
        make_cfg(),
        image_url="/tmp/image.png",
        model=model,
        question="main problem",
        sub_questions=["history question"],
        sub_answers=["history answer"],
        candidates=[("q1", "a1"), ("q2", "a2"), ("q3", "a3")],
    )

    assert len(model.calls) == 1
    assert len(model.calls[0][0]) == 6
    assert rewards == [
        {
            "sub_question_reward": 0.9,
            "sub_answer_reward": 0.6,
            "base_sub_answer_reward": 0.6,
            "p3pro_applied": False,
        },
        {
            "sub_question_reward": 0.8,
            "sub_answer_reward": 0.5,
            "base_sub_answer_reward": 0.5,
            "p3pro_applied": False,
        },
        {
            "sub_question_reward": 0.7,
            "sub_answer_reward": 0.4,
            "base_sub_answer_reward": 0.4,
            "p3pro_applied": False,
        },
    ]
