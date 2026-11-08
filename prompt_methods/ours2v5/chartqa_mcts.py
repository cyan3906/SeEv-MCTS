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
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from typing import Dict, Tuple

from .color_print import printred, printgreen, printblue, printcyan, printpurple, printyellow,printpink

from .mcts import MCTS, MCTSNode

from .utils_call import (
    Node_Type,
    generate_message,
    default_call,
    reward_call,
    ost_step_call, # step by step
    extract_final_answer,
    extract_final_answer_word,
    majority_vote,
    extract_ost_step,
    reach_terminal_ost_step,
    ost_get_reward_parallel,
    direct_QA,
    is_terminal_question,
)


class ReasoningMCTSNode(MCTSNode):
    @property
    def visited(self):
        return self._visited

    def __init__(self, node_type = Node_Type.USER_QUESTION, solution_trace = [] , question = "", depth = 0, r2 = 0.0,parent: 'ReasoningMCTSNode' = None ,ancestor_node: 'ReasoningMCTSNode' = None, cfg=None, model=None, image_url=None):
        self._conf = None
        self.children = []
        # self.mcts_children = {}
        self.depth = depth
        self._visited = False
        self.cfg = cfg
        self.model = model
        self.image_url = image_url
        self._r2 = r2
        self.solution_trace = solution_trace
        self.node_type = node_type
        self.question = question
        self.parent = parent
        self.ancestor_node = ancestor_node
        if self.node_type == Node_Type.USER_QUESTION:
            self.ancestor_node = self
            self.parent = self
            # self.best_answer = ""
            self.answer_tree = []
            self.solution_tree = []
            
    def build_ost_step_node(self, node_type , question, solution_trace , r2, parent: 'ReasoningMCTSNode' = None, ancestor_node: 'ReasoningMCTSNode' = None):
        return ReasoningMCTSNode(
            node_type = Node_Type.OST_STEP, 
            question = question, 
            solution_trace = solution_trace, 
            depth = self.depth + 1, 
            parent = parent, 
            ancestor_node = ancestor_node,
            r2=r2,
            cfg=self.cfg, 
            model=self.model, 
            image_url=self.image_url
        )
    
    def ost_gen_fn(self):
        try:
            last_ost_step = False
            if self.depth == self.cfg.prompt_method.ost_step_num: # 处理最后一次分解
                last_ost_step = True

            candidate_paths = ost_step_call(self.cfg, self.image_url, self.model, self.question, self.solution_trace, last_ost_step)
            candidate_ost_steps = []
            
            for j, path in enumerate(candidate_paths):
                step = extract_ost_step(path)
                candidate_ost_steps.append(step)
            
            candidate_rewards,answer_tree,solution_tree,list_new_curr_ost_steps = ost_get_reward_parallel(self.cfg, self.image_url, self.model, self.question, self.solution_trace, candidate_ost_steps,self.ancestor_node.answer_tree,self.ancestor_node.solution_tree)
            
            # printblue(f"candidate_rewards:{candidate_rewards}")
            # printpurple(f"best_answer:{best_answer}")
            # printblue(f"list_new_curr_ost_steps:{list_new_curr_ost_steps}")
            
            self.ancestor_node.answer_tree = answer_tree
            self.ancestor_node.solution_tree = solution_tree
            
            candidate_rewards = {
                "ost_step_reward": candidate_rewards,
            }
            
            return candidate_ost_steps , candidate_rewards,list_new_curr_ost_steps # return candidate_ost_steps, candidate_rewards # 给出下一个step和该step分数
        
        except Exception as e:
            printred(f"Error: ost_gen_fn->{e}")
            return [], {}
    
    def _get_children(self):
        self._visited = True # 已访问
        if self.is_terminal:
            return self.children

        try:
            candidate_ost_steps , candidate_rewards,list_new_curr_ost_steps = self.ost_gen_fn()
            
            for i, (ost_step, reward) in enumerate(zip(candidate_ost_steps, candidate_rewards["ost_step_reward"])):
                # printyellow(f"{i} {ost_step} {reward}")
                if extract_final_answer_word(ost_step).strip(". , ! \n") == "": # 答案为空的情况直接跳过
                    continue
                ost_step = list_new_curr_ost_steps[i] # 可能重写过
                # printyellow(f"{i} {ost_step} {reward}")
                solution_trace = self.solution_trace + [ost_step]
                new_node = self.build_ost_step_node(
                    node_type = Node_Type.OST_STEP, 
                    question = self.question, 
                    solution_trace = solution_trace, 
                    r2 = reward["ost_step_reward"],
                    parent=self.parent,
                    ancestor_node=self.ancestor_node,
                )
                self.children.append(new_node) 
            
            
            # direct_answer_list = direct_QA(self.cfg, self.image_url, self.model, self.question)
            # for direct_answer in direct_answer_list:
            #     if direct_answer == "":
            #         continue

            #     new_node = self.build_ost_step_node(
            #         node_type = Node_Type.DIRECT_ANSWER, 
            #         question = self.question, 
            #         solution_trace = [], 
            #         r2 = 0.5,
            #         parent=self.parent,
            #         ancestor_node=self.ancestor_node,
            #     )
                
            #     self.children.append(new_node) 
                
        except Exception as e:
            printred(f"Error: _get_children->{e}")
            
        return self.children

    def find_children(self):
        self.children = self.children or self._get_children()
        return self.children

    def find_one_child(self) -> MCTSNode:
        return random.choice(self.find_children())

    def _static_terminal(self):
        if self.node_type == Node_Type.OST_STEP:
            if self.solution_trace:
                # return reach_terminal_ost_step(list(list(self.solution_trace.items())[-1][1].items())[-1][1]) # 返回最后一个step
                return reach_terminal_ost_step(self.solution_trace[-1])
            return False
        elif self.node_type == Node_Type.DIRECT_ANSWER:
            return True
        else:
            return False
        
    @property
    def is_terminal(self):
        return self._static_terminal()

    @property
    def reward(self):
        if self.node_type == Node_Type.OST_STEP:
            return self._r2
        elif self.node_type == Node_Type.USER_QUESTION:
            return 1
        
    def set_reward(self, r):
        if self.node_type == Node_Type.OST_STEP:
            self._r2 = r
        
    def __setstate__(self, state):
        # 恢复状态时，将不可序列化的对象设置为None
        self.__dict__.update(state)
        self.parent = None
        self.cfg = None
        self.model = None
        self.image_url = None

    def __getstate__(self):
        # 创建一个新的状态字典，只包含需要序列化的属性 用于保存pkl文件
        state = {
            'question': self.question,
            "solution_trace": self.solution_trace,
            'depth': self.depth,
            '_r2': self._r2,
            'node_type': self.node_type,
            '_visited': self._visited,
            'children': self.children,
            "parent": self.parent,
            "ancestor_node": self.ancestor_node,
            'answer_tree': self.ancestor_node.answer_tree,
            "solution_tree": self.ancestor_node.solution_tree,
            # 不保存model和其他不可序列化的对象
        }
        return state
    
    def print(self, print_children=True):
        # print("question: ", self.question)
        # print("sub_questions: ", self.sub_questions)
        # print("sub_answers: ", self.sub_answers)
        printblue(f"depth: {self.depth}")
        printgreen(f"node_type: {self.node_type}")
        # print("children: ", self.children)
        # print("is_visited: ", self._visited)
        print("reward: ", self.reward)
        print("is_terminal: ", self.is_terminal)
        
        # all children
        if print_children:
            for child in self.children:
                printgreen(f"child: {child}")
                child.print()


def our2v5_forward(query, img_path, is_pkl, tree_save_path, model, cfg=None):
    
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
        
    # print(f"mcts_children: {mcts.mcts_children}")

    # print(root)
    
    root = next(iter(mcts.mcts_children))
    if root.node_type == Node_Type.USER_QUESTION:
        printblue("你找对了根节点了!")
        
    memory = {}
    # best_answer
    # memory["best_answer"] = get_best_answer(root.ancestor_node.answer_tree)
    
    # printyellow(f"best_answer: {memory['best_answer']}")
    # max_mean_terminal
    max_n, max_r = mcts.max_mean_terminal(root)
    
    if max_n.node_type == Node_Type.OST_STEP:
        final_answer = max_n.solution_trace[-1]
    elif max_n.node_type == Node_Type.SUBQUESTION:
        final_answer = max_n.sub_answers[-1]
    
    memory["max_mean_terminal"] = {}
    if max_n.node_type == Node_Type.OST_STEP:
        memory["max_mean_terminal"]["solution_trace"] = max_n.solution_trace
    elif max_n.node_type == Node_Type.SUBQUESTION:   
        memory["max_mean_terminal"]["sub_questions"] = max_n.sub_questions
        memory["max_mean_terminal"]["sub_answers"] = max_n.sub_answers
    print("max_mean_terminal ok !")
    
    # max_terminal
    max_n, max_r = mcts.max_terminal(root)
    memory["max_terminal"] = {}
    if max_n.node_type == Node_Type.OST_STEP:
        memory["max_terminal"]["solution_trace"] = max_n.solution_trace
    elif max_n.node_type == Node_Type.SUBQUESTION:
        final_answer = max_n.sub_answers[-1]
        memory["max_terminal"]["sub_questions"] = max_n.sub_questions
        memory["max_terminal"]["sub_answers"] = max_n.sub_answers
    
    print(memory)
    print("max_terminal ok !")
    
    # max_vote_terminal
    terminal_nodes = mcts.max_vote_terminal(root)
    final_answers = []
    sub_questions = []
    sub_answers = []
    solution_trace = []
    for node in terminal_nodes:
        if node[0].node_type == Node_Type.OST_STEP:
            final_answers.append(node[0].solution_trace[-1])
            solution_trace.append(node[0].solution_trace[-1])
        elif node[0].node_type == Node_Type.SUBQUESTION:
            final_answers.append(node[0].sub_answers[-1])
            sub_questions.append(node[0].sub_questions[-1])
            sub_answers.append(node[0].sub_answers[-1]) 
    
    final_answers_word = [extract_final_answer_word(a) for a in final_answers]
    print(final_answers_word)
    final_answer, index = majority_vote(final_answers_word) # 抽取后的答案，所在的下标
    final_answer = final_answers[index] # 原始答案
    printblue(f"final_answer: {final_answer}")  
    
    printpurple(f"answer_tree: {root.ancestor_node.answer_tree}")
    
    memory["max_vote_terminal"] = {}
    
    memory["max_vote_terminal"]["sub_questions"] = sub_questions
    memory["max_vote_terminal"]["sub_answers"] = sub_answers
    memory["max_vote_terminal"]["solution_trace"] = solution_trace
    
    res = {
        "memory": memory,
        "response": final_answer, # memory["best_answer"] final_answer
        "root": root, # 以上都是基于root来做的筛选
        "mcts": mcts,
    }
    
    return res