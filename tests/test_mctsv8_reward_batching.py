from types import SimpleNamespace

from prompt_methods.mctsv8.utils_call import get_reward_parallel


def make_cfg(*, batch_size=5, conditional_p3pro=False):
    return SimpleNamespace(
        llm=SimpleNamespace(name="Qwen2.5-VL-7B-Instruct"),
        prompt_method=SimpleNamespace(
            attempt_num=3,
            return_num=10,
            reward_prompt_batch_size=batch_size,
            use_p3pro_for_low_answer=conditional_p3pro,
            p3pro_question_threshold=0.7,
            p3pro_answer_threshold=0.7,
        ),
    )


class RecordingBatchModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def get_reward_batch(
        self,
        messages_batch,
        n,
        node_depth,
        is_p3pro=False,
        is_p3=False,
    ):
        self.calls.append(
            {
                "messages_batch": messages_batch,
                "n": n,
                "node_depth": node_depth,
                "is_p3pro": is_p3pro,
                "is_p3": is_p3,
            }
        )
        values = next(self.responses)
        assert len(values) == len(messages_batch)
        return [{"yes_prob": value, "no_prob": 1.0 - value} for value in values]


def run_batch(model, *, count, batch_size=5, conditional_p3pro=False):
    return get_reward_parallel(
        make_cfg(batch_size=batch_size, conditional_p3pro=conditional_p3pro),
        image_url="image.png",
        model=model,
        question="main question",
        sub_questions=["existing question"],
        sub_answers=["existing answer"],
        candidate_sub_questions=[f"candidate question {index}" for index in range(count)],
        candidate_sub_answers=[f"candidate answer {index}" for index in range(count)],
        is_p3pro=False,
        is_p3=False,
    )


def test_reward_candidates_use_two_vllm_batches_and_keep_order():
    model = RecordingBatchModel(
        responses=[
            [0.91, 0.82, 0.73, 0.64, 0.55],
            [0.15, 0.25, 0.35, 0.45, 0.55],
        ]
    )

    rewards = run_batch(model, count=5)

    assert len(model.calls) == 2
    assert [len(call["messages_batch"]) for call in model.calls] == [5, 5]
    assert [call["n"] for call in model.calls] == [10, 10]
    assert [reward["sub_question_reward"] for reward in rewards] == [
        0.91,
        0.82,
        0.73,
        0.64,
        0.55,
    ]
    assert [reward["sub_answer_reward"] for reward in rewards] == [
        0.15,
        0.25,
        0.35,
        0.45,
        0.55,
    ]


def test_reward_microbatches_preserve_candidate_order():
    model = RecordingBatchModel(
        responses=[
            [0.91, 0.82],
            [0.73, 0.64],
            [0.55],
            [0.15, 0.25],
            [0.35, 0.45],
            [0.55],
        ]
    )

    rewards = run_batch(model, count=5, batch_size=2)

    assert [len(call["messages_batch"]) for call in model.calls] == [2, 2, 1, 2, 2, 1]
    assert [reward["sub_question_reward"] for reward in rewards] == [
        0.91,
        0.82,
        0.73,
        0.64,
        0.55,
    ]
    assert [reward["sub_answer_reward"] for reward in rewards] == [
        0.15,
        0.25,
        0.35,
        0.45,
        0.55,
    ]


def test_conditional_p3pro_batches_only_eligible_answers():
    model = RecordingBatchModel(
        responses=[
            [0.90, 0.60, 0.95],
            [0.20, 0.10, 0.80],
            [0.88],
        ]
    )

    rewards = run_batch(model, count=3, conditional_p3pro=True)

    assert len(model.calls) == 3
    assert model.calls[-1]["is_p3pro"] is True
    assert len(model.calls[-1]["messages_batch"]) == 1
    assert rewards == [
        {
            "sub_question_reward": 0.90,
            "sub_answer_reward": 0.88,
            "base_sub_answer_reward": 0.20,
            "p3pro_applied": True,
        },
        {
            "sub_question_reward": 0.60,
            "sub_answer_reward": 0.10,
            "base_sub_answer_reward": 0.10,
            "p3pro_applied": False,
        },
        {
            "sub_question_reward": 0.95,
            "sub_answer_reward": 0.80,
            "base_sub_answer_reward": 0.80,
            "p3pro_applied": False,
        },
    ]
