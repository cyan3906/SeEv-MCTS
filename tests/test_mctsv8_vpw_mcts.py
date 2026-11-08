import math
import pickle
from types import SimpleNamespace

import pytest

import prompt_methods.mctsv8_vpw.chartqa_mcts as vpw_chart
from prompt_methods.mctsv8_vpw.chartqa_mcts import VPWReasoningNode, mcts_forward_vpw
from prompt_methods.mctsv8_vpw.mcts import VPWMCTS


class SelectionNode:
    def __init__(self, reward, *, original=None, terminal=False):
        self._reward = reward
        self.activation_original_score = reward if original is None else original
        self.children = []
        self.reserve_pool = []
        self.reserve_opened_count = 0
        self.latest_backprop_mean = None
        self.latest_activation_score = None
        self._terminal = terminal

    @property
    def reward(self):
        return self._reward

    @property
    def is_terminal(self):
        return self._terminal

    @property
    def visited(self):
        return False

    def find_children(self):
        return self.children

    def set_reward(self, value):
        self._reward = value


def make_cfg():
    return SimpleNamespace(
        llm=SimpleNamespace(name="Qwen2.5-VL-7B-Instruct"),
        prompt_method=SimpleNamespace(
            decompose_question_num=8,
            tree_width=5,
            return_num=10,
            reward_prompt_batch_size=32,
            reserve_activation_threshold=0.8,
        ),
    )


def reward_rows(count):
    return [
        {
            "sub_question_reward": 0.9 - index * 0.01,
            "sub_answer_reward": 0.8,
            "base_sub_answer_reward": 0.8,
            "p3pro_applied": False,
        }
        for index in range(count)
    ]


def test_shallow_expansion_keeps_five_visible_and_three_hidden(monkeypatch):
    score_sizes = []

    monkeypatch.setattr(
        vpw_chart,
        "generate_normal_candidates",
        lambda *args, width, **kwargs: [
            (f"normal question {index}", f"normal answer {index}")
            for index in range(width)
        ],
    )
    monkeypatch.setattr(
        vpw_chart,
        "generate_diverse_candidates",
        lambda *args, **kwargs: [
            (f"diverse question {index}", f"diverse answer {index}")
            for index in range(3)
        ],
    )

    def fake_score(*args, candidates, **kwargs):
        score_sizes.append(len(candidates))
        return reward_rows(len(candidates))

    monkeypatch.setattr(vpw_chart, "score_candidates_batch", fake_score)

    node = VPWReasoningNode(
        question="main",
        sub_questions=[],
        sub_answers=[],
        depth=1,
        cfg=make_cfg(),
        model=object(),
        image_url="/tmp/image.png",
    )

    visible = node.find_children()

    assert len(visible) == 5
    assert len(node.reserve_pool) == 3
    assert score_sizes == [8]
    assert {child.generation_source for child in visible} == {"normal"}
    assert {child.generation_source for child in node.reserve_pool} == {"divgen"}
    assert node.activation_original_score == 1.0


@pytest.mark.parametrize(
    ("depth", "expected_width"),
    [(4, 5), (7, 3)],
)
def test_middle_and_deep_expansion_do_not_generate_reserves(
    monkeypatch, depth, expected_width
):
    diverse_calls = []
    monkeypatch.setattr(
        vpw_chart,
        "generate_normal_candidates",
        lambda *args, width, **kwargs: [
            (f"question {index}", f"answer {index}") for index in range(width)
        ],
    )
    monkeypatch.setattr(
        vpw_chart,
        "generate_diverse_candidates",
        lambda *args, **kwargs: diverse_calls.append(True),
    )
    monkeypatch.setattr(
        vpw_chart,
        "score_candidates_batch",
        lambda *args, candidates, **kwargs: reward_rows(len(candidates)),
    )
    parent = SelectionNode(0.7)
    node = VPWReasoningNode(
        question="main",
        sub_questions=["history"],
        sub_answers=["answer"],
        depth=depth,
        parent=parent,
        r0=0.8,
        r1=0.9,
        cfg=make_cfg(),
        model=object(),
        image_url="/tmp/image.png",
    )

    assert len(node.find_children()) == expected_width
    assert node.reserve_pool == []
    assert diverse_calls == []
    assert node.activation_original_score == pytest.approx(math.sqrt(0.72))


def test_node_rejects_nonpositive_max_depth_before_generation(monkeypatch):
    cfg = make_cfg()
    cfg.prompt_method.decompose_question_num = 0
    monkeypatch.setattr(
        vpw_chart,
        "generate_normal_candidates",
        lambda *args, **kwargs: pytest.fail("generation must not start"),
    )
    node = VPWReasoningNode(
        question="main",
        sub_questions=[],
        sub_answers=[],
        depth=1,
        cfg=cfg,
        model=object(),
        image_url="/tmp/image.png",
    )

    with pytest.raises(ValueError, match="max_depth must be positive"):
        node.find_children()


def test_activation_opens_one_reserve_into_same_uct_comparison():
    parent = SelectionNode(0.1, original=1.0)
    normal_low = SelectionNode(0.4)
    normal_high = SelectionNode(0.6)
    reserve_high = SelectionNode(0.9)
    reserve_second = SelectionNode(0.8)
    parent.children = [normal_low, normal_high]
    parent.reserve_pool = [reserve_high, reserve_second]

    mcts = VPWMCTS(w_exp=1, aggr_reward="mean", aggr_child="max")
    mcts.children[parent] = parent.children
    mcts.N[parent] = 1
    mcts.Q[parent] = 0.7

    selected = mcts.select_child(parent)

    assert selected is reserve_high
    assert parent.children == [normal_low, normal_high, reserve_high]
    assert parent.reserve_pool == [reserve_second]
    assert parent.reserve_opened_count == 1
    assert parent.latest_backprop_mean == 0.7
    assert parent.latest_activation_score == 0.85


def test_activation_requires_a_visit_and_is_strictly_above_threshold():
    parent = SelectionNode(0.1, original=1.0)
    parent.children = [SelectionNode(0.4)]
    parent.reserve_pool = [SelectionNode(0.9)]
    mcts = VPWMCTS()
    mcts.children[parent] = parent.children

    assert mcts.maybe_open_reserve(parent) is None
    assert len(parent.reserve_pool) == 1

    mcts.N[parent] = 1
    mcts.Q[parent] = 0.6
    assert mcts.maybe_open_reserve(parent) is None
    assert len(parent.reserve_pool) == 1


def test_empty_reserve_pool_is_idempotent():
    parent = SelectionNode(0.1, original=1.0)
    parent.children = [SelectionNode(0.4)]
    mcts = VPWMCTS()
    mcts.children[parent] = parent.children
    mcts.N[parent] = 2
    mcts.Q[parent] = 1.8

    assert mcts.maybe_open_reserve(parent) is None
    assert mcts.maybe_open_reserve(parent) is None
    assert parent.children == mcts.children[parent]


def test_hidden_terminal_reserve_is_excluded_until_opened():
    root = SelectionNode(0.0, original=1.0)
    visible_terminal = SelectionNode(0.6, terminal=True)
    hidden_terminal = SelectionNode(0.99, terminal=True)
    root.children = [visible_terminal]
    root.reserve_pool = [hidden_terminal]
    mcts = VPWMCTS()
    mcts.children[root] = root.children

    assert [node for node, _ in mcts.max_vote_terminal(root)] == [visible_terminal]


def test_node_pickle_preserves_activation_and_reserve_membership():
    node = VPWReasoningNode(
        question="main",
        sub_questions=[],
        sub_answers=[],
        depth=1,
        cfg=make_cfg(),
        model=object(),
        image_url="/tmp/image.png",
    )
    visible = VPWReasoningNode(
        "main", ["q"], ["a"], 2, parent=node, r0=0.8, r1=0.7,
        cfg=make_cfg(), model=object(), image_url="/tmp/image.png",
        generation_source="normal",
    )
    hidden = VPWReasoningNode(
        "main", ["dq"], ["da"], 2, parent=node, r0=0.9, r1=0.9,
        cfg=make_cfg(), model=object(), image_url="/tmp/image.png",
        generation_source="divgen",
    )
    node.children = [visible]
    node.reserve_pool = [hidden]

    restored = pickle.loads(pickle.dumps(node))

    assert restored.activation_original_score == 1.0
    assert [child.generation_source for child in restored.children] == ["normal"]
    assert [child.generation_source for child in restored.reserve_pool] == ["divgen"]
    assert restored.model is None
    assert restored.cfg is None


def test_backprop_mutation_preserves_original_question_and_answer_rewards():
    parent = SelectionNode(0.5)
    node = VPWReasoningNode(
        "main",
        ["q"],
        ["a"],
        2,
        parent=parent,
        r0=0.8,
        r1=0.9,
        cfg=make_cfg(),
        model=object(),
        image_url="/tmp/image.png",
    )

    node.set_reward(0.95)
    restored = pickle.loads(pickle.dumps(node))

    assert node.reward == 0.95
    assert restored.original_question_reward == 0.8
    assert restored.original_answer_reward == 0.9
    assert restored.activation_original_score == pytest.approx(math.sqrt(0.72))


def test_full_mcts_checkpoint_is_pickleable_and_keeps_canonical_root(tmp_path):
    cfg = make_cfg()
    root = VPWReasoningNode(
        "main", [], [], 1, cfg=cfg, model=object(), image_url="/tmp/image.png"
    )
    terminal = VPWReasoningNode(
        "main",
        ["Now we can answer the main question"],
        ["The answer is A"],
        2,
        parent=root,
        r0=0.9,
        r1=0.9,
        cfg=cfg,
        model=object(),
        image_url="/tmp/image.png",
    )
    root.children = [terminal]
    mcts = VPWMCTS()
    mcts.root = root
    mcts.children[root] = root.children
    mcts.N[root] = 1
    mcts.Q[root] = 0.9

    mcts_path = tmp_path / "samplemcts.pkl"
    root_path = tmp_path / "sample.pkl"
    mcts_path.write_bytes(pickle.dumps(mcts))
    root_path.write_bytes(pickle.dumps(root))

    restored_mcts = pickle.loads(mcts_path.read_bytes())
    assert restored_mcts.root in restored_mcts.children

    result = mcts_forward_vpw(
        "ignored",
        "/tmp/image.png",
        True,
        str(root_path),
        model=None,
        cfg=None,
    )
    assert result["response"] == "A"
