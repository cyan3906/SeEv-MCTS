import io
import os
import random
import math
import re
import warnings
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm, trange
import time
import json
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

from .color_print import printred, printgreen, printblue, printcyan, printpurple, printyellow

from .mcts import MCTS, MCTSNode

from .utils_call import (
    Node_Type,
    generate_message,
    default_call,
    reward_call,
    decompose_question_call,
    extract_final_answer,
    extract_final_answer_word,
    majority_vote,
    extract_sub_questions_and_answers,
    get_reward_answer,
    get_reward_subquestion,
    get_reward,
    get_reward_parallel,
    is_terminal_question,
)

class ReasoningMCTSNode(MCTSNode):
    @property
    def visited(self):
        return self._visited

    def __init__(self, node_type = Node_Type.USER_QUESTION, question = "", sub_questions = [], sub_answers = [], depth = 0, parent: 'ReasoningMCTSNode' = None, ancestor_node: 'ReasoningMCTSNode' = None, r0=0.0, r1=0.0, r2=0.0, cfg=None, model=None, image_url=None):
        self.node_type = node_type
        self._conf = None
        self.idx = None # 每一个变量的下标 主要是给最终图的节点编号用的
        self.children = []
        self.depth = depth
        self._r0 = r0
        self._r1 = r1
        self._r2 = r2
        self._visited = False
        self.question = question
        self.sub_questions = sub_questions
        self.sub_answers = sub_answers
        self.cfg = cfg
        self.model = model
        self.image_url = image_url
        self.ancestor_node = ancestor_node
        self.parent = parent
        if self.node_type == Node_Type.USER_QUESTION:
            # self.best_answer = ""
            self.ancestor_node = self
            self.parent = self
            self.answer_tree = []
            self.solution_tree = {}
    
    def build_subquestion_node(self, node_type , question, sub_questions , sub_answers , r0, r1 ,r2):
        try:
            return ReasoningMCTSNode(
                node_type = Node_Type.SUBQUESTION, 
                question = question, 
                sub_questions = sub_questions, 
                sub_answers = sub_answers, 
                depth = self.depth + 1, 
                r0=r0,
                r1=r1,
                r2 = r2,
                cfg=self.cfg, 
                model=self.model, 
                image_url=self.image_url,
                ancestor_node=self.ancestor_node,
                parent=self, 
            )
        except Exception as e:
            printred(f"build_subquestion_node error: {e}")
            return None
    
    def gen_fn(self):
        try:
            # pattern = r' The answer is .*$'
            # for i,j in enumerate(self.sub_questions):
            #     self.sub_questions[i] = re.sub(pattern, '', j).strip() # 防止格式不对
            #     if self.sub_questions[i][-1] == ".":
            #         self.sub_questions[i] = self.sub_questions[i][:-1] + "?" # 字符串不可变类型
                    
            # printblue(f"当前子问题: {self.sub_questions}")
            
            if self.depth == self.cfg.prompt_method.decompose_question_num: # 如果是最后一个问题，则加上"Now we can answer"
                last_sub_question = True
            else:
                last_sub_question = False
            candidate_paths = decompose_question_call(self.cfg, self.image_url, self.model, self.question, self.sub_questions, self.sub_answers, last_sub_question)

            # print("路径", candidate_paths)
            candidate_sub_questions = []
            candidate_sub_answers = []
            for j, path in enumerate(candidate_paths):
                # printpurple(f"第{j}条路径: {path}")
                # printgreen(f"{self.depth}")
                sub_question, sub_answer = extract_sub_questions_and_answers(path, self.depth)
                
                if sub_question == "" or sub_answer == "" or extract_final_answer_word(sub_answer) == "":
                    continue
                candidate_sub_questions.append(sub_question)
                candidate_sub_answers.append(sub_answer)
                
            # printgreen(f"{candidate_sub_questions}")
            # printblue(f"{candidate_sub_answers}")
            
            candidate_rewards,answer_tree,solution_tree = get_reward_parallel(self.cfg, self.image_url, self.model, self.question, self.sub_questions, self.sub_answers, candidate_sub_questions, candidate_sub_answers,self.ancestor_node.answer_tree,self.ancestor_node.solution_tree)
            # print(1,self.parent)
            self.ancestor_node.answer_tree = answer_tree
            self.ancestor_node.solution_tree = solution_tree
            
            return candidate_sub_questions, candidate_sub_answers, candidate_rewards
        
        except Exception as e:
            printred(f"gen_fn error: {e}")
            return [], [], []
        
    def _get_children(self):
        try:
            self._visited = True # 已访问
            if self.is_terminal:
                return self.children
            candidate_sub_questions, candidate_sub_answers, candidate_rewards = self.gen_fn()
            for i, (sub_question, sub_answer, reward) in enumerate(zip(candidate_sub_questions, candidate_sub_answers, candidate_rewards)):
                sub_questions = self.sub_questions + [sub_question]
                sub_answers = self.sub_answers + [sub_answer]
                r0 = reward["sub_question_reward"]
                r1 = reward["sub_answer_reward"]
                r2 = reward["pk_reward"]
                # 添加孩子的时候判断一下 有没有问题重复
                new_node = self.build_subquestion_node(Node_Type.SUBQUESTION, self.question, sub_questions, sub_answers, r0, r1,r2)
                
                self.children.append(new_node)
                            
            return self.children
        except Exception as e:
            printred(f"_get_children error: {e}")
            return []
    def find_children(self):
        self.children = self.children or self._get_children()
        return self.children

    def find_one_child(self) -> MCTSNode:
        return random.choice(self.find_children())

    def _static_terminal(self):
        if self.sub_answers:
            return is_terminal_question(self.question, self.sub_questions[-1])
        else:
            return False

    @property
    def is_terminal(self):
        return self._static_terminal()

    @property
    def reward(self):
        return 0.3 * math.sqrt(self._r0 * self._r1) + 0.7 * self._r2 
    
    def set_reward(self, r):
        self._r0 = r
        self._r1 = 1.0 # 说白了就是保留最大的设置
        self._r2 = 1.0

    def __setstate__(self, state):
        # 恢复状态时，将不可序列化的对象设置为None
        self.__dict__.update(state)
        self.parent = None
        self.cfg = None
        self.model = None
        self.image_url = None

    def __getstate__(self):
        # 创建一个新的状态字典，只包含需要序列化的属性
        state = {
            'question': self.question,
            'sub_questions': self.sub_questions,
            'sub_answers': self.sub_answers,
            'depth': self.depth,
            '_r0': self._r0,
            '_r1': self._r1,
            "_r2": self._r2,
            '_visited': self._visited,
            'children': self.children,
            "answer_tree": self.ancestor_node.answer_tree,
            "solution_tree": self.ancestor_node.solution_tree,
            "ancestor_node": self.ancestor_node,
            # "mcts":""
            # 不保存model和其他不可序列化的对象
        }
        return state
    def print(self, print_children=True):
        print("question: ", self.question)
        print("sub_questions: ", self.sub_questions)
        print("sub_answers: ", self.sub_answers)
        print("depth: ", self.depth)
        print("children: ", self.children)
        print("is_visited: ", self._visited)
        print("reward: ", self.reward)
        
        # all children
        if print_children:
            for child in self.children:
                printgreen(f"child: {child}")
                child.print()


def mcts_forward_v8v3(query, img_path, is_pkl, tree_save_path, model, cfg=None):
    
    if is_pkl:
        print(f"存在:MCTS树")
        mcts_path = os.path.join(os.path.dirname(tree_save_path), f"{os.path.basename(tree_save_path).split('.')[0]}mcts.pkl")
        print(f"mcts_path: {mcts_path}")
        
        with open(mcts_path, "rb") as f:
            mcts = pickle.load(f)
        
        with open(tree_save_path, "rb") as f:
            root = pickle.load(f)
            
        print("结果提取完毕")
        
        # print(dir(mcts))
        
    else:
        mcts = MCTS(w_exp=1, prior=False, aggr_reward='mean', aggr_child='max') # 这里prior修改为false了
        
        image_url = img_path # 直接用路径
        
        root = ReasoningMCTSNode(node_type = Node_Type.USER_QUESTION , question = query , depth=1, cfg=cfg, model=model, image_url=image_url) # 加入step by step action
        
        print("MCTS初始化完毕")
        for _ in trange(cfg.prompt_method.mcts_rollouts,desc="MCTS rollouts"):
            mcts.rollout(root)
            print("一次 MCTS rollout完毕")
        
        print("MCTS执行完毕")
    
    printblue(f"{root.ancestor_node.answer_tree}")
    printgreen(f"{root.ancestor_node.solution_tree}")
    
    memory = {}
    
    # 所有的max_n都是代表的终端节点的意思
    
    # max_mean_terminal
    max_n, max_r = mcts.max_mean_terminal(root)
    final_answer = max_n.sub_answers[-1]
    
    memory["max_mean_terminal"] = {}
    memory["max_mean_terminal"]["sub_questions"] = max_n.sub_questions
    memory["max_mean_terminal"]["sub_answers"] = max_n.sub_answers
    
    # max_terminal
    max_n, max_r = mcts.max_terminal(root)
    final_answer = max_n.sub_answers[-1]
    
    memory["max_terminal"] = {}
    memory["max_terminal"]["sub_questions"] = max_n.sub_questions
    memory["max_terminal"]["sub_answers"] = max_n.sub_answers
    
    terminal_nodes = mcts.max_vote_terminal(root)
    # 这里的node是元组的格式 两个元素 1.节点 2.奖励值
    final_answers = [node[0].sub_answers[-1] for node in terminal_nodes]
    final_answers_word = [extract_final_answer_word(a) for a in final_answers]
    final_answer, index = majority_vote(final_answers_word) # 抽取后的答案，所在的下标
    final_answer = final_answers[index] # 原始答案
    printblue(f"{final_answer}")
    memory["max_vote_terminal"] = {}
    memory["max_vote_terminal"]["sub_questions"] = [node[0].sub_questions for node in terminal_nodes]
    memory["max_vote_terminal"]["sub_answers"] = [node[0].sub_answers for node in terminal_nodes]
    
    res = {
        "memory": memory,
        "response": final_answer,
        "root": root, # 以上都是基于root来做的筛选
        # "mcts": mcts,
    }
    
    return res


