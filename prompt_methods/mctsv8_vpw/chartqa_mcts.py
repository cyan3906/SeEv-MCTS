import math
import os
import pickle

from tqdm import trange

from prompt_methods.mctsv8.chartqa_mcts import ReasoningMCTSNode
from prompt_methods.mctsv8.utils_call import select_explicit_answer_vote

from .generation import (
    generate_diverse_candidates,
    generate_normal_candidates,
    score_candidates_batch,
)
from .mcts import VPWMCTS
from .policy import get_stage_policy


class VPWReasoningNode(ReasoningMCTSNode):
    def __init__(
        self,
        question,
        sub_questions,
        sub_answers,
        depth,
        parent=None,
        r0=0.0,
        r1=0.0,
        cfg=None,
        model=None,
        embedding_model=None,
        image_url=None,
        generation_source="normal",
    ):
        super().__init__(
            question,
            sub_questions,
            sub_answers,
            depth,
            parent=parent,
            r0=r0,
            r1=r1,
            cfg=cfg,
            model=model,
            embedding_model=embedding_model,
            image_url=image_url,
        )
        self.generation_source = generation_source
        self.original_question_reward = r0
        self.original_answer_reward = r1
        self._search_reward = math.sqrt(r0 * r1)
        self.activation_original_score = (
            1.0 if parent is None else self._search_reward
        )
        self.reserve_pool = []
        self.reserve_generated_count = 0
        self.reserve_opened_count = 0
        self.latest_backprop_mean = None
        self.latest_activation_score = None
        self.progress = None
        self.search_stage = None

    @property
    def reward(self):
        return self._search_reward

    def set_reward(self, value):
        self._search_reward = value

    def _child_node(
        self,
        question,
        sub_questions,
        sub_answers,
        r0,
        r1,
        generation_source="normal",
    ):
        return VPWReasoningNode(
            question,
            sub_questions,
            sub_answers,
            self.depth + 1,
            parent=self,
            r0=r0,
            r1=r1,
            cfg=self.cfg,
            model=self.model,
            embedding_model=self.embedding_model,
            image_url=self.image_url,
            generation_source=generation_source,
        )

    def _build_child(self, candidate, reward, generation_source):
        sub_question, sub_answer = candidate
        return self._child_node(
            self.question,
            self.sub_questions + [sub_question],
            self.sub_answers + [sub_answer],
            reward["sub_question_reward"],
            reward["sub_answer_reward"],
            generation_source=generation_source,
        )

    def _get_children(self):
        self._visited = True
        if self.is_terminal:
            return self.children

        policy = get_stage_policy(
            self.depth,
            self.cfg.prompt_method.decompose_question_num,
        )
        self.progress = policy.progress
        self.search_stage = policy.stage
        normal_candidates = generate_normal_candidates(
            self.cfg,
            self.image_url,
            self.model,
            self.question,
            self.sub_questions,
            self.sub_answers,
            depth=self.depth,
            width=policy.normal_width,
        )
        diverse_candidates = []
        if policy.reserve_width:
            diverse_candidates = generate_diverse_candidates(
                self.cfg,
                self.image_url,
                self.model,
                self.question,
                self.sub_questions,
                self.sub_answers,
                normal_candidates,
                depth=self.depth,
            )[: policy.reserve_width]

        candidates = normal_candidates + diverse_candidates
        rewards = score_candidates_batch(
            self.cfg,
            self.image_url,
            self.model,
            self.question,
            self.sub_questions,
            self.sub_answers,
            candidates=candidates,
        )
        normal_count = len(normal_candidates)
        self.children = [
            self._build_child(candidate, reward, "normal")
            for candidate, reward in zip(
                normal_candidates, rewards[:normal_count]
            )
        ]
        self.reserve_pool = [
            self._build_child(candidate, reward, "divgen")
            for candidate, reward in zip(
                diverse_candidates, rewards[normal_count:]
            )
        ]
        self.reserve_pool.sort(key=lambda child: child.reward, reverse=True)
        self.reserve_generated_count = len(self.reserve_pool)
        return self.children

    def __getstate__(self):
        state = super().__getstate__()
        state.update(
            {
                "generation_source": self.generation_source,
                "original_question_reward": self.original_question_reward,
                "original_answer_reward": self.original_answer_reward,
                "_search_reward": self._search_reward,
                "activation_original_score": self.activation_original_score,
                "reserve_pool": self.reserve_pool,
                "reserve_generated_count": self.reserve_generated_count,
                "reserve_opened_count": self.reserve_opened_count,
                "latest_backprop_mean": self.latest_backprop_mean,
                "latest_activation_score": self.latest_activation_score,
                "progress": self.progress,
                "search_stage": self.search_stage,
            }
        )
        return state


def _tree_audit(root, mcts):
    audit = []
    stack = [root]
    seen = set()
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        audit.append(
            {
                "depth": node.depth,
                "progress": getattr(node, "progress", None),
                "search_stage": getattr(node, "search_stage", None),
                "generation_source": getattr(node, "generation_source", "normal"),
                "activation_original_score": getattr(
                    node, "activation_original_score", None
                ),
                "original_question_reward": getattr(
                    node, "original_question_reward", None
                ),
                "original_answer_reward": getattr(
                    node, "original_answer_reward", None
                ),
                "backprop_mean": getattr(node, "latest_backprop_mean", None),
                "activation_score": getattr(node, "latest_activation_score", None),
                "reserve_generated_count": getattr(
                    node, "reserve_generated_count", 0
                ),
                "reserve_opened_count": getattr(node, "reserve_opened_count", 0),
                "reserve_remaining_count": len(getattr(node, "reserve_pool", [])),
            }
        )
        stack.extend(mcts.children.get(node, []))
    return audit


def mcts_forward_vpw(
    query,
    img_path,
    is_pkl,
    tree_save_path,
    model,
    cfg=None,
    embedding_model=None,
):
    if is_pkl:
        mcts_path = os.path.join(
            os.path.dirname(tree_save_path),
            f"{os.path.basename(tree_save_path).split('.')[0]}mcts.pkl",
        )
        with open(mcts_path, "rb") as handle:
            mcts = pickle.load(handle)
        root = mcts.root
        if root is None:
            raise RuntimeError("VPW MCTS checkpoint is missing its canonical root")
    else:
        mcts = VPWMCTS(w_exp=1, aggr_reward="mean", aggr_child="max")
        root = VPWReasoningNode(
            question=query,
            sub_questions=[],
            sub_answers=[],
            depth=1,
            cfg=cfg,
            model=model,
            embedding_model=embedding_model,
            image_url=img_path,
        )
        mcts.root = root
        for _ in trange(cfg.prompt_method.mcts_rollouts, desc="VPW MCTS rollouts"):
            mcts.rollout(root)

    terminal_nodes = mcts.max_vote_terminal(root)
    if not terminal_nodes:
        raise RuntimeError("VPW MCTS produced no terminal answer")

    final_answers = [node.sub_answers[-1] for node, _ in terminal_nodes]
    vote_result = select_explicit_answer_vote(final_answers)
    memory = {
        "max_vote_terminal": {
            "sub_questions": [node.sub_questions[-1] for node, _ in terminal_nodes],
            "sub_answers": final_answers,
            "answer_extractions": vote_result["candidates"],
            "valid_vote_count": vote_result["valid_vote_count"],
            "excluded_vote_count": vote_result["excluded_vote_count"],
            "excluded_candidate_indices": vote_result[
                "excluded_candidate_indices"
            ],
        },
        "vpw_tree_audit": _tree_audit(root, mcts),
    }
    return {
        "memory": memory,
        "response": vote_result["response"],
        "raw_response": vote_result["raw_response"],
        "answer_extract_status": vote_result["answer_extract_status"],
        "root": root,
        "mcts": mcts,
    }
