from types import SimpleNamespace

import numpy as np
import pytest

import prompt_methods.mctsv8.utils_call as utils_call


def make_cfg(*, enabled):
    return SimpleNamespace(
        prompt_method=SimpleNamespace(
            use_p3pro_for_low_answer=enabled,
            p3pro_question_threshold=0.7,
            p3pro_answer_threshold=0.7,
        )
    )


def run_rewards(monkeypatch, *, enabled, base_rewards, p3pro_reward=0.91):
    remaining = iter(base_rewards)
    p3pro_calls = []

    def fake_get_reward(*args, **kwargs):
        return next(remaining)

    def fake_get_reward_answer(
        cfg,
        image_url,
        model,
        question,
        sub_questions,
        sub_answers,
        curr_sub_question,
        curr_sub_answer,
        is_p3pro=False,
        is_p3=False,
    ):
        p3pro_calls.append((curr_sub_question, curr_sub_answer, is_p3pro, is_p3))
        return p3pro_reward

    monkeypatch.setattr(utils_call, "get_reward", fake_get_reward)
    monkeypatch.setattr(utils_call, "get_reward_answer", fake_get_reward_answer)

    rewards = utils_call.get_reward_parallel(
        make_cfg(enabled=enabled),
        image_url="image.png",
        model=object(),
        question="question",
        sub_questions=[],
        sub_answers=[],
        candidate_sub_questions=[f"sub-question-{i}" for i in range(len(base_rewards))],
        candidate_sub_answers=[f"sub-answer-{i}" for i in range(len(base_rewards))],
        is_p3pro=False,
        is_p3=False,
    )
    return rewards, p3pro_calls


def test_disabled_switch_keeps_original_answer_rewards(monkeypatch):
    rewards, calls = run_rewards(
        monkeypatch,
        enabled=False,
        base_rewards=[(0.9, 0.2)],
    )

    assert calls == []
    assert rewards == [
        {
            "sub_question_reward": 0.9,
            "sub_answer_reward": 0.2,
            "base_sub_answer_reward": 0.2,
            "p3pro_applied": False,
        }
    ]


@pytest.mark.parametrize(
    ("rq", "ra", "should_apply"),
    [
        (0.70, 0.69, True),
        (0.95, 0.20, True),
        (0.69, 0.20, False),
        (0.95, 0.70, False),
        (0.69, 0.69, False),
    ],
)
def test_p3pro_only_replaces_low_answer_reward_for_high_question(
    monkeypatch, rq, ra, should_apply
):
    rewards, calls = run_rewards(
        monkeypatch,
        enabled=True,
        base_rewards=[(rq, ra)],
        p3pro_reward=0.91,
    )

    assert bool(calls) is should_apply
    assert rewards[0]["sub_question_reward"] == rq
    assert rewards[0]["base_sub_answer_reward"] == ra
    assert rewards[0]["p3pro_applied"] is should_apply
    assert rewards[0]["sub_answer_reward"] == (0.91 if should_apply else ra)
    if should_apply:
        assert calls == [("sub-question-0", "sub-answer-0", True, False)]


def test_invalid_p3pro_reward_falls_back_to_original_reward(monkeypatch):
    rewards, calls = run_rewards(
        monkeypatch,
        enabled=True,
        base_rewards=[(0.9, 0.2)],
        p3pro_reward="",
    )

    assert calls
    assert rewards[0]["sub_answer_reward"] == 0.2
    assert rewards[0]["p3pro_applied"] is False


def test_embedding_deduplication_uses_configured_batch_size():
    captured = {}

    class FakeEmbeddingModel:
        def encode(self, texts, **kwargs):
            captured["texts"] = texts
            captured.update(kwargs)
            return np.array(
                [
                    [1.0, 0.0],
                    [0.99, 0.01],
                    [0.0, 1.0],
                ]
            )

    indices = utils_call.select_deduplicated_indices(
        ["first", "near-duplicate", "different"],
        embedding_model=FakeEmbeddingModel(),
        batch_size=2,
        similarity_threshold=0.9,
    )

    assert indices == [0, 2]
    assert captured == {
        "texts": ["first", "near-duplicate", "different"],
        "batch_size": 2,
        "convert_to_numpy": True,
        "normalize_embeddings": True,
    }
