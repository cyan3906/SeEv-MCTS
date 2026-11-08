import math
from collections import defaultdict

from prompt_methods.mctsv8.mcts import MCTS


def _negative_infinity():
    return -math.inf


class VPWMCTS(MCTS):
    def __init__(
        self,
        w_exp=1,
        discount=1,
        aggr_reward="mean",
        aggr_child="max",
    ):
        super().__init__(
            w_exp=w_exp,
            discount=discount,
            prior=True,
            aggr_reward=aggr_reward,
            aggr_child=aggr_child,
        )
        self.Q = defaultdict(float)
        self.N = defaultdict(int)
        self.M = defaultdict(_negative_infinity)
        self.root = None

    def maybe_open_reserve(self, parent):
        reserve_pool = getattr(parent, "reserve_pool", None)
        if not reserve_pool or self.N[parent] <= 0:
            return None

        backprop_mean = self.Q[parent] / self.N[parent]
        activation_score = (
            0.5 * parent.activation_original_score + 0.5 * backprop_mean
        )
        parent.latest_backprop_mean = backprop_mean
        parent.latest_activation_score = activation_score
        threshold = getattr(
            getattr(parent, "cfg", None),
            "prompt_method",
            None,
        )
        threshold = getattr(threshold, "reserve_activation_threshold", 0.8)
        if activation_score <= threshold:
            return None

        opened = reserve_pool.pop(0)
        parent.children.append(opened)
        visible = self.children.get(parent)
        if visible is not None and visible is not parent.children:
            visible.append(opened)
        parent.reserve_opened_count += 1
        return opened

    def select_child(self, parent):
        self.maybe_open_reserve(parent)
        return self._uct_select(parent)

    def _select_prior(self, node):
        path = [node]
        while not node.is_terminal:
            self._expand(node)
            if not self.children[node]:
                return path
            self.maybe_open_reserve(node)
            node = self._uct_select(node)
            path.append(node)
        self._expand(node)
        return path
