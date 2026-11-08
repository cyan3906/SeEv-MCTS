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
    generate_message,
    default_call,
    get_reward,
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

    def __init__(self, question, sub_questions, sub_answers, depth, parent: 'ReasoningMCTSNode' = None, r0=0.0, r1=0.0, cfg=None, model=None, image_url=None):
        self._conf = None
        self.idx = None # 每一个变量的下标 主要是给最终图的节点编号用的
        self.children = []
        self.depth = depth
        self._r0 = r0
        self._r1 = r1
        self._visited = False
        self.parent = parent
        self.temp_children = [] # 用来判断两个点相似度是不是一样的
        self.question = question
        self.sub_questions = sub_questions
        self.sub_answers = sub_answers
        self.cfg = cfg
        self.model = model
        self.image_url = image_url
        
        
        
    def _child_node(self, question, sub_questions, sub_answers, r0, r1):
        return ReasoningMCTSNode(question, sub_questions, sub_answers, self.depth + 1, parent=self, r0=r0, r1=r1,
                                 cfg=self.cfg, model=self.model, image_url=self.image_url)
    
    def gen_fn(self):
        # 对子问题预处理
        # printgreen(f"一开始子问题: {self.sub_questions}")
        pattern = r' The answer is .*$'
        for i,j in enumerate(self.sub_questions):
            self.sub_questions[i] = re.sub(pattern, '', j).strip() # 防止格式不对
            if self.sub_questions[i][-1] == ".":
                self.sub_questions[i] = self.sub_questions[i][:-1] + "?" # 字符串不可变类型
                
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
            
            if sub_question == "" or sub_answer == "":
                continue
            candidate_sub_questions.append(sub_question)
            candidate_sub_answers.append(sub_answer)
        # print("子问题", candidate_sub_questions)
        # print("子答案", candidate_sub_answers)
        # 对每个子问题进行评估，得到分数。
        candidate_rewards = get_reward_parallel(self.cfg, self.image_url, self.model, self.question, self.sub_questions, self.sub_answers, candidate_sub_questions, candidate_sub_answers, is_p3pro=self.cfg.prompt_method.is_p3pro, is_p3=self.cfg.prompt_method.is_p3)
        # print(1233444)
        # print(candidate_rewards)
        return candidate_sub_questions, candidate_sub_answers, candidate_rewards
    
    # 返回一个列表
    def _get_children(self):
        self._visited = True # 已访问
        if self.is_terminal:
            return self.children
        candidate_sub_questions, candidate_sub_answers, candidate_rewards = self.gen_fn()
        
        # 这里可以对子问题进行去重
        if self.cfg.prompt_method.is_deduplicate:
            # print("\033[92m采用了去重\033[0m")
            deduplicate_sub_questions = [candidate_sub_questions[0]]
            deduplicate_sub_answers = [candidate_sub_answers[0]]
            deduplicate_rewards = [candidate_rewards[0]]
            
            for _ in range(1, len(candidate_sub_questions)):
                if candidate_sub_questions[_] not in deduplicate_sub_questions:
                    deduplicate_sub_questions.append(candidate_sub_questions[_])
                    deduplicate_sub_answers.append(candidate_sub_answers[_])
                    deduplicate_rewards.append(candidate_rewards[_])
            
            candidate_sub_questions = deduplicate_sub_questions[:5]
            candidate_sub_answers = deduplicate_sub_answers[:5]
            candidate_rewards = deduplicate_rewards[:5]
        
        # else:
        #     print("\033[93m没有采用了去重\033[0m")
            
        # for z in deduplicate_sub_questions:
        #     print(z)
        
        # candidate_rewards 每一个子问题子答案的分数 是一个字典
        for i, (sub_question, sub_answer, reward) in enumerate(zip(candidate_sub_questions, candidate_sub_answers, candidate_rewards)):
            sub_questions = self.sub_questions + [sub_question]
            sub_answers = self.sub_answers + [sub_answer]
            r0 = reward["sub_question_reward"]
            r1 = reward["sub_answer_reward"]
            # 添加孩子的时候判断一下 有没有问题重复
            new_node = self._child_node(self.question, sub_questions, sub_answers, r0, r1)
            
            # print("我来了!!!!!")
            self.children.append(new_node)
            
        return self.children

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
        return math.sqrt(self._r0 * self._r1) # [0, 1] # 开根号的计算方式 论文中提到过
    
    def set_reward(self, r):
        self._r0 = r
        self._r1 = 1.0 # 说白了就是保留最大的设置

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
            '_visited': self._visited,
            'children': self.children,
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


def mcts_forward_v8(query, img_path, is_pkl, tree_save_path, model, cfg=None):
    
    # print("我进来了")
    
    # with open(img_path, "rb") as image_file:
    #     image_data = image_file.read()
    # image_base64 = base64.b64encode(image_data).decode('utf-8')
    # image_url = f"data:image/jpeg;base64,{image_base64}"
    
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
        
        root = ReasoningMCTSNode(question = query, sub_questions=[], sub_answers=[], depth=1, cfg=cfg, model=model, image_url=image_url) # 加入step by step action
        
        print("MCTS初始化完毕")
        for _ in trange(cfg.prompt_method.mcts_rollouts,desc="MCTS rollouts"):
            mcts.rollout(root)
            print("一次 MCTS rollout完毕")
        
        print("MCTS执行完毕")
        
    # print(f"mcts_children: {mcts.mcts_children}")

    # print(root)
    
    # root = next(iter(mcts.mcts_children))
    # if root.node_type == Node_Type.USER_QUESTION:
    #     printblue("你找对了根节点了!")
        
    memory = {}
    # best_answer
    # memory["best_answer"] = get_best_answer(root.ancestor_node.answer_tree)
    
    # printyellow(f"best_answer: {memory['best_answer']}")
    # max_mean_terminal
    max_n, max_r = mcts.max_mean_terminal(root)
    
    # if max_n.node_type == Node_Type.OST_STEP:
    #     final_answer = max_n.solution_trace[-1]
    # elif max_n.node_type == Node_Type.SUBQUESTION:
    final_answer = max_n.sub_answers[-1]
    
    memory["max_mean_terminal"] = {}
    # if max_n.node_type == Node_Type.OST_STEP:
    #     memory["max_mean_terminal"]["solution_trace"] = max_n.solution_trace
    # elif max_n.node_type == Node_Type.SUBQUESTION:   
    memory["max_mean_terminal"]["sub_questions"] = max_n.sub_questions
    memory["max_mean_terminal"]["sub_answers"] = max_n.sub_answers
    print("max_mean_terminal ok !")
    
    # max_terminal
    max_n, max_r = mcts.max_terminal(root)
    memory["max_terminal"] = {}
    
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
        final_answers.append(node[0].sub_answers[-1])
        sub_questions.append(node[0].sub_questions[-1])
        sub_answers.append(node[0].sub_answers[-1]) 
    
    final_answers_word = [extract_final_answer_word(a) for a in final_answers]
    print(final_answers_word)
    final_answer, index = majority_vote(final_answers_word) # 抽取后的答案，所在的下标
    final_answer = final_answers[index] # 原始答案
    printblue(f"final_answer: {final_answer}")  
    
    # printpurple(f"answer_tree: {root.ancestor_node.answer_tree}")
    
    memory["max_vote_terminal"] = {}
    
    memory["max_vote_terminal"]["sub_questions"] = sub_questions
    memory["max_vote_terminal"]["sub_answers"] = sub_answers
    memory["max_vote_terminal"]["solution_trace"] = solution_trace
    
    res = {
        "memory": memory,
        "response": final_answer, # memory["best_answer"] final_answer
        "root": root, # 以上都是基于root来做的筛选
        # "mcts": mcts,
    }
    
    return res