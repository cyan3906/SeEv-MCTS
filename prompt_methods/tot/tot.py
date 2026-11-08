
from .utils_call import (
    generate_message,
    default_call,
    decompose_question_call,
    extract_final_answer,
    extract_final_answer_word,
    majority_vote,
    extract_sub_questions_and_answers,
    is_terminal_question,
)

from .color_print import *
import queue

class ReasoningMCTSNode:
    @property
    def visited(self):
        return self._visited

    def __init__(self, question, sub_questions, sub_answers, depth, parent: 'ReasoningMCTSNode' = None,
                 cfg=None, model=None, image_url=None):
        self._conf = None
        self.children = []
        self.depth = depth
        self._visited = False
        self.parent = parent
        self.question = question
        self.sub_questions = sub_questions
        self.sub_answers = sub_answers
        self.cfg = cfg
        self.model = model
        self.image_url = image_url

    def _child_node(self, question, sub_questions, sub_answers):
        return ReasoningMCTSNode(question, sub_questions, sub_answers, self.depth + 1, parent=self, 
                                 cfg=self.cfg, model=self.model, image_url=self.image_url)
    
    def gen_fn(self):
        # 生成候选路径
        if self.depth == self.cfg.prompt_method.decompose_question_num:  # 如果是最后一个问题，则加上"Now we can answer"
            last_sub_question = True
        else:
            last_sub_question = False
        
        # 生成候选子问题路径
        candidate_paths = decompose_question_call(self.cfg, self.image_url, self.model, self.question, 
                                                  self.sub_questions, self.sub_answers, last_sub_question)
        
        # printyellow(f"{candidate_paths}")
        # print(self.depth)
        candidate_sub_questions = []
        candidate_sub_answers = []
        for j, path in enumerate(candidate_paths):
            sub_question, sub_answer = extract_sub_questions_and_answers(path, self.depth + 1)
            if sub_question == "" or sub_answer == "" or "answer is" in sub_question.lower():
                continue
            # printblue(f"{sub_question}")
            # printgreen(f"{sub_answer}")
            candidate_sub_questions.append(sub_question)
            candidate_sub_answers.append(sub_answer)
        
        return candidate_sub_questions, candidate_sub_answers
    
    def _get_children(self):
        self._visited = True
        
        if self.is_terminal:
            return self.children
        
        candidate_sub_questions, candidate_sub_answers = self.gen_fn()
        
        for i, (sub_question, sub_answer) in enumerate(zip(candidate_sub_questions, candidate_sub_answers)):
            
            sub_questions = self.sub_questions + [sub_question]
            sub_answers = self.sub_answers + [sub_answer]
            self.children.append(self._child_node(self.question, sub_questions, sub_answers))
        
        return self.children

    def __setstate__(self, state):
        # 恢复状态时，将不可序列化的对象设置为None
        self.__dict__.update(state)
        # self.parent = None
        self.cfg = None
        self.model = None
        self.image_url = None
    
    def __getstate__(self):
        # 创建一个新的状态字典，只包含需要序列化的属性 用于保存pkl文件
        state = {
            'question': self.question,
            "sub_questions": self.sub_questions,
            "sub_answers": self.sub_answers,
            'depth': self.depth,
            '_visited': self._visited,
            'children': self.children,
            "parent": self.parent,
        }
        return state
    
    def _static_terminal(self):
        if self.sub_answers:
            return is_terminal_question(self.question, self.sub_questions[-1])
        else:
            return False

    @property
    def is_terminal(self):
        return self._static_terminal()
    
def tot_forward(cfg, question,image_url, model):
    
    q = queue.Queue()
    root = ReasoningMCTSNode(question, [], [], depth=0, cfg=cfg, model=model, image_url=image_url)
    
    q.put(root)
    num = 1
    list_answer = []
    
    while not q.empty():
        node = q.get()
        node._get_children()
        for child in node.children:
            if "now we can answer" in child.sub_questions[-1].lower():
                list_answer.append(child.sub_answers[-1])
                
            printyellow(f"{child.sub_questions}")
            printblue(f"{child.sub_answers}")
            printgreen(f"{child.depth}")
            # input("点击继续")
            num += 1
            q.put(child)
    
    printpurple(f"节点总数: {num}")
    
    answers = []
    for candidata_answer in list_answer:
        if "answer is" in candidata_answer.lower():
            # filtered_answer = re.sub(r'[^\w\s]', '', candidata_answer.split("answer is")[-1].strip(". \n ,"))
            filtered_answer = candidata_answer.split("answer is")[-1].strip(". \n ,")
            answers.append(filtered_answer)
    print(answers)
    final_answer = majority_vote(answers)[0]
    
    printgreen(f"最终答案: {final_answer}")
    
    response = {
        "response": final_answer,
        # "root": root,
    }
    
    return response
    

