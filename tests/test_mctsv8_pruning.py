from types import MethodType, SimpleNamespace

import pytest

from prompt_methods.mctsv8.chartqa_mcts import ReasoningMCTSNode, mcts_forward_v8
from prompt_methods.mctsv8.mcts import MCTS


def make_node(*, enabled=True, threshold=0.7):
    prompt_method = SimpleNamespace(
        is_deduplicate=False,
        prune_low_reward_nodes=enabled,
        prune_reward_threshold=threshold,
    )
    cfg = SimpleNamespace(prompt_method=prompt_method)
    return ReasoningMCTSNode(
        question="question",
        sub_questions=[],
        sub_answers=[],
        depth=1,
        cfg=cfg,
        model=None,
        image_url=None,
    )


def install_candidates(node, rewards):
    def gen_fn(self):
        questions = [f"candidate-{i}?" for i in range(len(rewards))]
        answers = [f"answer-{i}" for i in range(len(rewards))]
        values = [
            {"sub_question_reward": rq, "sub_answer_reward": ra}
            for rq, ra in rewards
        ]
        return questions, answers, values

    node.gen_fn = MethodType(gen_fn, node)


@pytest.mark.parametrize(
    ("rewards", "expected_questions"),
    [
        ([(0.69, 0.69)], []),
        ([(0.70, 0.69)], ["candidate-0?"]),
        ([(0.69, 0.70)], ["candidate-0?"]),
        ([(0.95, 0.20)], ["candidate-0?"]),
        ([(0.20, 0.95)], ["candidate-0?"]),
    ],
)
def test_prunes_only_when_both_rewards_are_below_threshold(
    rewards, expected_questions
):
    node = make_node(enabled=True, threshold=0.7)
    install_candidates(node, rewards)

    children = node.find_children()

    assert [child.sub_questions[-1] for child in children] == expected_questions


def test_disabled_pruning_keeps_low_reward_candidate():
    node = make_node(enabled=False, threshold=0.7)
    install_candidates(node, [(0.10, 0.10)])

    children = node.find_children()

    assert [child.sub_questions[-1] for child in children] == ["candidate-0?"]


def test_pruning_uses_configured_threshold():
    node = make_node(enabled=True, threshold=0.5)
    install_candidates(node, [(0.49, 0.49), (0.50, 0.49)])

    children = node.find_children()

    assert [child.sub_questions[-1] for child in children] == ["candidate-1?"]


def test_rollout_stops_when_every_candidate_is_pruned_without_regenerating():
    node = make_node(enabled=True, threshold=0.7)
    calls = 0

    def gen_fn(self):
        nonlocal calls
        calls += 1
        return (
            ["candidate-0?", "candidate-1?"],
            ["answer-0", "answer-1"],
            [
                {"sub_question_reward": 0.20, "sub_answer_reward": 0.30},
                {"sub_question_reward": 0.60, "sub_answer_reward": 0.50},
            ],
        )

    node.gen_fn = MethodType(gen_fn, node)

    MCTS(aggr_reward="mean").rollout(node)

    assert calls == 1
    assert node.find_children() == []
    assert node.pruned_fallback == {
        "sub_question": "candidate-1?",
        "sub_answer": "answer-1",
        "sub_question_reward": 0.60,
        "sub_answer_reward": 0.50,
        "reward": pytest.approx((0.60 * 0.50) ** 0.5),
    }


def test_repeated_rollout_stops_when_expanded_node_has_no_children():
    node = make_node(enabled=True, threshold=0.7)
    install_candidates(node, [(0.20, 0.30)])
    mcts = MCTS(aggr_reward="mean")

    mcts.rollout(node)
    mcts.rollout(node)

    assert mcts.children[node] == []


def test_max_terminal_handles_expanded_nonterminal_leaf():
    node = make_node(enabled=True, threshold=0.7)
    mcts = MCTS(aggr_reward="mean")
    mcts.children[node] = []

    best_node, reward = mcts.max_terminal(node)

    assert best_node is node
    assert reward == -float("inf")


def test_mcts_returns_best_already_generated_answer_when_all_are_pruned(monkeypatch):
    def gen_fn(self):
        return (
            ["candidate-0?", "candidate-1?"],
            ["answer-0", "answer-1"],
            [
                {"sub_question_reward": 0.20, "sub_answer_reward": 0.30},
                {"sub_question_reward": 0.60, "sub_answer_reward": 0.50},
            ],
        )

    monkeypatch.setattr(ReasoningMCTSNode, "gen_fn", gen_fn)
    cfg = SimpleNamespace(
        prompt_method=SimpleNamespace(
            mcts_rollouts=1,
            is_deduplicate=False,
            prune_low_reward_nodes=True,
            prune_reward_threshold=0.7,
        )
    )

    result = mcts_forward_v8(
        query="question",
        img_path=None,
        is_pkl=False,
        tree_save_path=None,
        model=None,
        cfg=cfg,
    )

    assert result["response"] == "answer-1"
    assert result["root"].find_children() == []
    assert result["memory"]["pruned_fallback"]["reward"] == pytest.approx(
        (0.60 * 0.50) ** 0.5
    )
