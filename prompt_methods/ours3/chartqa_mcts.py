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
    decompose_question_call,
    ost_step_call, # step by step
    subanswers_ost_step_call,
    extract_final_answer,
    extract_final_answer_word,
    majority_vote,
    extract_sub_questions_and_answers, 
    is_terminal_question,
    extract_ost_step,
    reach_terminal_ost_step,
    ost_get_reward_parallel,
)


class ReasoningMCTSNode(MCTSNode):
    @property
    def visited(self):
        return self._visited

    def __init__(self, node_type = Node_Type.USER_QUESTION, is_sub_questions_completed = False, solution_trace = [] , question = "", sub_questions = [], sub_answers = [], depth = 0, r0 = 0.0, r1 = 0.0, r2 = 0.0,parent: 'ReasoningMCTSNode' = None ,cfg=None, model=None, embedding_model=None, image_url=None):
        self._conf = None
        self.idx = None # 每一个变量的下标 主要是给最终图的节点编号用的
        self.children = []
        self.depth = depth
        self._visited = False
        self.parent = parent
        self.temp_children = [] # 用来判断两个点相似度是不是一样的
        self.question = question
        self.cfg = cfg
        self.model = model
        self.embedding_model = embedding_model # 添加embedding_model
        self.image_url = image_url
        self._r2 = r2
        self.sub_questions = sub_questions
        self.sub_answers = sub_answers
        self.solution_trace = solution_trace
        self.node_type = node_type
        self.is_sub_questions_completed = is_sub_questions_completed

    def build_ost_step_node(self, node_type , is_sub_questions_completed, question, sub_questions , sub_answers , solution_trace , r2):
        return ReasoningMCTSNode(
            node_type = Node_Type.OST_STEP, 
            is_sub_questions_completed = is_sub_questions_completed,
            question = question, 
            sub_questions = sub_questions, 
            sub_answers = sub_answers, 
            solution_trace = solution_trace, 
            depth = self.depth + 1, 
            parent=self, 
            r2=r2,
            cfg=self.cfg, 
            model=self.model, 
            image_url=self.image_url
        )
    # gen_fn 对子问题预处理
    def subquestion_gen_fn(self):
        # printgreen(f"一开始子问题: {self.sub_questions}")
        # pattern = r' The answer is .*$'
        # for i,j in enumerate(self.sub_questions):
        #     self.sub_questions[i] = re.sub(pattern, '', j).strip() # 防止格式不对
        #     if self.sub_questions[i][-1] == ".":
        #         self.sub_questions[i][-1] = "?"
                
        # printblue(f"当前子问题: {self.sub_questions}")
        
        
        if self.depth == self.cfg.prompt_method.decompose_question_num: # 如果是最后一个问题，则加上"Now we can answer"
            last_sub_question = True
        else:
            last_sub_question = False
        
        candidate_paths = decompose_question_call(self.cfg, self.image_url, self.model, self.question, self.sub_questions, self.sub_answers, last_sub_question)
        candidate_sub_questions = []
        candidate_sub_answers = []
    
        for j, path in enumerate(candidate_paths):
            sub_question, sub_answer = extract_sub_questions_and_answers(path, self.depth)
            if sub_question == "" or sub_answer == "":
                # printred("不要啊")
                continue
            candidate_sub_questions.append(sub_question)
            candidate_sub_answers.append(sub_answer)
        
        # print("子问题", candidate_sub_questions)
        # print("子答案", candidate_sub_answers)
        # 对每个子问题进行评估，得到分数。
        # candidate_rewards = subquestions_get_reward_parallel(self.cfg, self.image_url, self.model, self.question, self.sub_questions, self.sub_answers, candidate_sub_questions, candidate_sub_answers)
    
        return candidate_sub_questions, candidate_sub_answers
    
    def ost_gen_fn(self):
        last_ost_step = False
        if self.depth == self.cfg.prompt_method.ost_step_num: # 处理最后一次分解
            last_ost_step = True

        candidate_paths = ost_step_call(self.cfg, self.image_url, self.model, self.question, self.sub_answers, self.solution_trace, last_ost_step)
        
        candidate_ost_steps = []
        
        for j, path in enumerate(candidate_paths):
            step = extract_ost_step(path)
            candidate_ost_steps.append(step)

        candidate_rewards = ost_get_reward_parallel(self.cfg, self.image_url, self.model, self.question, self.sub_answers, self.solution_trace, candidate_ost_steps)
        
        return candidate_ost_steps , candidate_rewards
    
    def subanswers_ost_gen_fn(self):
        last_ost_step = False
        if self.depth == self.cfg.prompt_method.ost_step_num: # 处理最后一次分解
            last_ost_step = True
        
        candidate_paths = subanswers_ost_step_call(self.cfg, self.image_url, self.model, self.question, self.sub_answers, last_ost_step)
        
        candidate_ost_steps = []
        
        for j, path in enumerate(candidate_paths):
            step = extract_ost_step(path)
            candidate_ost_steps.append(step)

        candidate_rewards = ost_get_reward_parallel(self.cfg, self.image_url, self.model, self.question, self.sub_answers, self.solution_trace, candidate_ost_steps , is_answers_gen = True)
        
        return candidate_ost_steps , candidate_rewards
    # 返回一个列表
    def _get_children(self):
        self._visited = True # 已访问
        if self.is_terminal:
            return self.children
        
        if not self.is_sub_questions_completed:
            candidate_sub_questions, candidate_sub_answers = self.subquestion_gen_fn() # 这里就可能存在分解完成了
            # printgreen(f"{candidate_sub_questions}")
            # printblue(f"{candidate_sub_answers}")
            candidate_ost_steps , candidate_rewards = self.ost_gen_fn()
            # printyellow(f"{candidate_ost_steps}")
            # printblue(f"{candidate_rewards}")
            try:
                for i, (ost_step, reward) in enumerate(zip(candidate_ost_steps, candidate_rewards)):
                    if i >= len(candidate_sub_questions): # 可能生成子问题的数量不足，跳过
                        continue
                    solution_trace = self.solution_trace + [ost_step]
                    is_sub_questions_completed = is_terminal_question(self.question, candidate_sub_questions[i])
                    new_node = self.build_ost_step_node(
                        node_type = Node_Type.OST_STEP, 
                        is_sub_questions_completed = is_sub_questions_completed,
                        question = self.question,
                        sub_questions = self.sub_questions + [candidate_sub_questions[i]] if not is_sub_questions_completed else self.sub_questions, 
                        sub_answers = self.sub_answers + [candidate_sub_answers[i]] if not is_sub_questions_completed else self.sub_answers, 
                        solution_trace = solution_trace, 
                        r2 = reward["ost_step_reward"]  # 0.86 0.8 0.74 self.cfg.prompt_method.ost_reward_score
                    )
                    self.children.append(new_node) # 这种是ost 生 ost
                
            except Exception as e:
                printblue(f"{candidate_sub_questions}")
                printyellow(f"{candidate_ost_steps}")
                printred(f"Error: add_ost_step_node->{e}")
        
            # 将子答案组合成ost
            if self.node_type != Node_Type.SUBQUESTION:
                candidate_ost_steps , candidate_rewards = self.subanswers_ost_gen_fn()
                try:
                    for i, (ost_step, reward) in enumerate(zip(candidate_ost_steps, candidate_rewards)):
                        if i >= len(candidate_sub_questions):
                            continue
                        solution_trace = self.sub_answers + [ost_step] # 这里修改了
                        new_node = self.build_ost_step_node(
                            node_type = Node_Type.OST_STEP, 
                            is_sub_questions_completed = is_sub_questions_completed,
                            question = self.question,
                            sub_questions = self.sub_questions + [candidate_sub_questions[i]], 
                            sub_answers = self.sub_answers + [candidate_sub_answers[i]], 
                            solution_trace = solution_trace, 
                            r2 = reward["ost_step_reward"]  # 0.86 0.8 0.74 self.cfg.prompt_method.ost_reward_score
                        )
                        self.children.append(new_node) # 这种是ost 生 ost
                
                except Exception as e:
                    # printblue(f"{candidate_sub_questions}")
                    printyellow(f"{candidate_ost_steps}")
                    printred(f"Error: add_sub_answers_ost_step_node->{e}")
        else:
            candidate_ost_steps , candidate_rewards = self.ost_gen_fn()
            # printyellow(f"{candidate_ost_steps}")
            # printblue(f"{candidate_rewards}")
            try:
                for i, (ost_step, reward) in enumerate(zip(candidate_ost_steps, candidate_rewards)):
                    solution_trace = self.solution_trace + [ost_step]
                    new_node = self.build_ost_step_node(
                        node_type = Node_Type.OST_STEP, 
                        is_sub_questions_completed = True,
                        question = self.question,
                        sub_questions = self.sub_questions, 
                        sub_answers = self.sub_answers, 
                        solution_trace = solution_trace, 
                        r2 = reward["ost_step_reward"]  # 0.86 0.8 0.74 self.cfg.prompt_method.ost_reward_score
                    )
                    self.children.append(new_node) # 这种是ost 生 ost
                
            except Exception as e:
                printred(f"Error: add_ost_step_node->{e}")
                
        return self.children

    def find_children(self):
        self.children = self.children or self._get_children()
        return self.children

    def find_one_child(self) -> MCTSNode:
        return random.choice(self.find_children())

    def _static_terminal(self):
        if self.node_type == Node_Type.SUBQUESTION:
            if self.sub_answers:
                return is_terminal_question(self.question, self.sub_questions[-1])
            else:
                return False
        elif self.node_type == Node_Type.OST_STEP:
            if self.solution_trace:
                # return reach_terminal_ost_step(list(list(self.solution_trace.items())[-1][1].items())[-1][1]) # 返回最后一个step
                return reach_terminal_ost_step(self.solution_trace[-1])
            return False
        else:
            return False
        
    @property
    def is_terminal(self):
        return self._static_terminal()
    @property
    def reward(self):
        if self.node_type == Node_Type.SUBQUESTION:
            return math.sqrt(self._r0 * self._r1) # [0, 1] # 开根号的计算方式 论文中提到过
        elif self.node_type == Node_Type.OST_STEP:
            return self._r2
        elif self.node_type == Node_Type.USER_QUESTION:
            return 1
    def set_reward(self, r):
        if self.node_type == Node_Type.SUBQUESTION:
            self._r0 = r
            self._r1 = 1.0 # 说白了就是保留最大的设置
        elif self.node_type == Node_Type.OST_STEP:
            self._r2 = r
        
    def __setstate__(self, state):
        # 恢复状态时，将不可序列化的对象设置为None
        self.__dict__.update(state)
        self.parent = None
        self.cfg = None
        self.model = None
        self.image_url = None
        self.embedding_model = None

    def __getstate__(self):
        # 创建一个新的状态字典，只包含需要序列化的属性
        state = {
            'question': self.question,
            'sub_questions': self.sub_questions,
            'sub_answers': self.sub_answers,
            "solution_trace": self.solution_trace,
            'depth': self.depth,
            '_r2': self._r2,
            'node_type': self.node_type,
            '_visited': self._visited,
            'children': self.children,
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


def our3_forward(query, img_path, is_pkl, tree_save_path, model, embedding_model=None, cfg=None):
    
    # print("我进来了")
    
    # with open(img_path, "rb") as image_file:
    #     image_data = image_file.read()
    # image_base64 = base64.b64encode(image_data).decode('utf-8')
    # image_url = f"data:image/jpeg;base64,{image_base64}"
    
    mcts = MCTS(w_exp=1, prior=False, aggr_reward='mean', aggr_child='max') # 这里prior修改为false了
    
    if is_pkl:
        print(tree_save_path)
        with open(tree_save_path, "rb") as f:
            root = pickle.load(f)
    else:
        image_url = img_path # 直接用路径
        
        root = ReasoningMCTSNode(node_type = Node_Type.USER_QUESTION , question = query , depth=1, cfg=cfg, model=model, embedding_model=embedding_model, image_url=image_url) # 加入step by step action
        
        print("MCTS初始化完毕")
        for _ in trange(cfg.prompt_method.mcts_rollouts,desc="MCTS rollouts"):
            mcts.rollout(root)
            print("一次 MCTS rollout完毕")
        
        print("MCTS执行完毕")
    
        # list1 = [root]
        # while list1:
        #     node = list1.pop(0)
        #     node.print(print_children=False)
        #     list1.extend(node.children)
    
    # 所有的max_n都是代表的终端节点的意思
    
    # print(type(root))
    # print(root.children)
    # print(root.children[0].solution_trace, root.children[0].sub_questions, root.children[0].sub_answers) 
    # print(root.node_type)
    # import sys
    # sys.exit(0)
    
    memory = {}
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
    memory["max_vote_terminal"] = {}
    
    memory["max_vote_terminal"]["sub_questions"] = sub_questions
    memory["max_vote_terminal"]["sub_answers"] = sub_answers
    memory["max_vote_terminal"]["solution_trace"] = solution_trace
    
    res = {
        "memory": memory,
        "response": final_answer,
        "root": root, # 以上都是基于root来做的筛选
    }
    
    return res