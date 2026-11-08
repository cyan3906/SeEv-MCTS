import math
import random
from collections import defaultdict
from abc import ABC, abstractmethod
from typing import Optional

from .color_print import printred, printgreen, printblue, printcyan, printpurple, printyellow

class MCTSNode(ABC):
    @abstractmethod
    def find_children(self):
        return set()

    @abstractmethod
    def find_one_child(self) -> 'MCTSNode':
        raise NotImplementedError

    @property
    @abstractmethod
    def is_terminal(self):
        return True

    @property
    @abstractmethod
    def reward(self):
        return 0

    @property
    @abstractmethod
    def visited(self):
        return 0


class MCTS:
    def __init__(self, w_exp=1, discount=1, prior=False, aggr_reward='sum', aggr_child='max'):
        self.Q: dict[MCTSNode, float] = defaultdict(lambda : 0.)
        self.N: dict[MCTSNode, int] = defaultdict(lambda : 0)
        self.M: dict[MCTSNode, float] = defaultdict(lambda : -math.inf)
        self.children = dict()
        self.w_exp = w_exp
        self.discount = discount
        self.prior = prior # 是否使用已存在的路径 节约时间
        self.aggr_reward = aggr_reward
        self.aggr_child = aggr_child

    def rollout(self, node: MCTSNode):
        if self.prior:
            path = self._select_prior(node)
        else:
            path = self._select(node)
            self._expand(path[-1]) # 不要从结果的角度考虑 只是为了拓展节点
            self._simulate(path)
            # import sys
            # sys.exit(0)
            
        # 这里可以对叶子节点做答案的判断
        
        self._back_propagate(path)

    def _select_prior(self, node: MCTSNode):
        path = [node]
        while not node.is_terminal:   
            self._expand(node)
            if len(self.children[node]) == 0:
                return path
            node = self._uct_select(node)
            path.append(node)
        self._expand(node)
        return path
    
    def _select(self, node: MCTSNode):
        path = []
        while True:
            path.append(node)
            if node not in self.children or node.is_terminal:
                return path
            # 套娃的思想，先找一个没有被访问过的子节点
            for child in self.children[node]:
                if child not in self.children.keys():
                    path.append(child)
                    return path
            node = self._uct_select(node)

    def _expand(self, node: MCTSNode):
        if node not in self.children:
            self.children[node] = node.find_children()

    # @staticmethod
    def _simulate(self, path: list[MCTSNode]):
        # path 只需要在本函数中调用就行了
        node = path[-1]
        while not node.is_terminal:
            # node = node.find_one_child()
            list_node = node.find_children()
            # print(3, list_node)
            self.children[node] = list_node # 新加入的东西
            node = random.choice(list_node)
            # printred(f"{node.is_terminal}")
            
            if node:
                path.append(node)
            else:
                break
            
        # 加入这个 必杀技
        if node.is_terminal:
            node._visited = True
        
    # 贪心选择最大的路径上所有的奖励值
    def max_terminal(self, cur: MCTSNode):
        if cur.is_terminal:
            if cur.visited:
                # printyellow(f"{cur.depth}")
                return cur, cur.reward
            else:
                return cur, -math.inf
        if cur not in self.children:
            return cur, -math.inf

        max_n, max_r = max((self.max_terminal(child) for child in self.children[cur]), key=lambda x: x[1])
        return max_n, max_r + cur.reward

    # 返回某一个路径上的最大平均奖励节点的平均值 说白了就是按照平均值来选择
    def max_mean_terminal(self, cur: MCTSNode, sum=0., cnt=0):
        if cur.is_terminal:
            if cur.visited:
                # printred("有终端点")
                return cur, (sum + cur.reward) / (cnt + 1)
            else:
                return cur, -math.inf
        if cur not in self.children or not self.children[cur]:
            # printred("cur not in self.children or not self.children[cur]")
            # cur.print()
            # printred("打印结束")
            return cur, -math.inf
        
        if len(self.children[cur]):
            return max((self.max_mean_terminal(child, sum + cur.reward, cnt + 1) for child in self.children[cur]), key=lambda x: x[1])
    
    # 投票最后的决策是按照平均奖励来的
    def max_vote_terminal(self, root: MCTSNode, k: int = 10):
        # 从根节点出发，收集所有is_terminal的节点，然后取平均reward前k个
        # 广度优先遍历
        def visit_path(node: MCTSNode, sum=0., cnt=0):
            terminal_nodes = []
            if node.is_terminal:
                terminal_nodes.append((node, (sum + node.reward) / (cnt + 1)))
                
            if node not in self.children or not self.children[node]:
                return terminal_nodes
            
            for child in self.children[node]:
                terminal_nodes.extend(visit_path(child, sum + node.reward, cnt + 1))
            return terminal_nodes
        terminal_nodes = visit_path(root)
        terminal_nodes.sort(key=lambda x: x[1], reverse=True)
        
        # 取平均reward前k个
        return terminal_nodes[:k]

    def _back_propagate(self, path: list[MCTSNode], reward=0.):
        try:
            coeff = 0
            for node in reversed(path):
                try:
                    # printyellow(f"{node._r0,node._r1,node._r2}")
                    reward = reward * self.discount + node.reward # 计算当前路径的奖励 开根号
                except Exception as e:
                    printred(f"_back_propagate中reward计算报错 {reward}")
                    
                coeff = coeff * self.discount + 1
                if self.aggr_reward == 'mean':
                    c_reward = reward / coeff
                else:
                    c_reward = reward # 保持不变
                if node not in self.N:
                    self.Q[node] = c_reward
                else:
                    self.Q[node] += c_reward
                    
                self.N[node] += 1
                self.M[node] = max(self.M[node], c_reward)
                node.set_reward(self.M[node])
                
        except Exception as e:
            printred("_back_propagate")
    def _uct(self, node: MCTSNode, log_n_f: float):
        # print("# in _uct (reward, uct)")
        if self.prior and self.N[node] == 0:
            # print("## unexplored: ", node.reward, node.reward + self.w_exp * math.sqrt(log_n_f))
            return node.reward + self.w_exp * math.sqrt(log_n_f)
        if self.aggr_child == 'max':
            # print("## explored: ", self.N[node], self.M[node], self.w_exp * math.sqrt(log_n_f / self.N[node]))
            return self.M[node] + self.w_exp * math.sqrt(log_n_f / self.N[node])
        elif self.aggr_child == 'mean':
            return self.Q[node] / self.N[node] + self.w_exp * math.sqrt(log_n_f / self.N[node])

    def _uct_select(self, node: MCTSNode):
        if self.prior and self.N[node] == 0:
            log_n = math.log(1)
        else:
            log_n = math.log(self.N[node])
        return max(self.children[node], key=lambda n: self._uct(n, log_n))
