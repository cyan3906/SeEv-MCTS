import pickle
import json,os,time
from tools.color_print import *
# 读取 .pkl 文件


# file = f'/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=baseline/data=charxiv/llm=Qwen2-VL-7B-Instruct/prompt_method=mctsv8/tree_results/{dot}.pkl'
dir_file = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=baseline/data=charxiv/llm=Qwen2-VL-7B-Instruct/prompt_method=mctsv8/tree_results"

data_file = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=baseline/data=charxiv/llm=Qwen2-VL-7B-Instruct/prompt_method=mctsv8/gen-reasoning_val.json"
answer_file = "/home/jncsnlp4/SSD2/ZJJ/VReST/data/CharXiv/data/reasoning_val.json"
with open(data_file, 'r') as f:
    data = json.load(f)
with open(answer_file, 'r') as f:
    answer_data = json.load(f)
    
alls = 0
mmax = 0

is_can_answer = {
    
}

for dot in os.listdir(dir_file):
    file = f"{dir_file}/{dot}"
    idx = int(dot.split(".")[0])
    is_can_answer[idx] = {
        "pre_answer": "",
        # "ground_answer": data[str(idx)]["answer"],
        "ground_answer": answer_data[str(idx)]["answer"],
        "is_can_answer_flag": False,
        "depth":{
            
        }
    }
    
    answer = answer_data[str(idx)]["answer"]
    
    with open(file, 'rb') as f:  # 'rb' 表示二进制读取模式
        root = pickle.load(f)  # data 即为反序列化后的对象（可能是列表、字典、类实例等）

    # data_file = "/home/jncsnlp4/SSD2/ZJJ/VReST/data/MathVista/data/testmini.json"
    # data_file = "/home/jncsnlp4/SSD2/ZJJ/VReST/data/CharXiv/data/reasoning_val.json"

    # with open(data_file, 'r') as f:
    #     data = json.load(f)

    # answer = data[str(dot)]["answer"]

    # 查看读取结果
    # print(root)
    # print(root.node_type)
    # print(root.children)
    # print(root.answer_tree)
    # print(dir(root))
    # answer_tree = root.answer_tree
    import queue
    queue = queue.Queue()
    queue.put(root)

    deep_dict = {}

    num = 1

    is_can_answer_flag = False
    
    while not queue.empty():
        root = queue.get()
        
        # if root.node_type == "SUBQUESTION":

        last_question = root.sub_questions[-1] if len(root.sub_questions) > 0 else ""
        last_answer = root.sub_answers[-1] if len(root.sub_answers) > 0 else ""
        depth = root.depth
        
        if "Now we can answer the question" in last_question and "answer is" in last_answer:
            xiu_last_answer = last_answer.split("answer is")[-1].strip()
            if depth not in is_can_answer[idx]["depth"]:
                is_can_answer[idx]["depth"][depth] = []
            is_can_answer[idx]["depth"][depth].append(xiu_last_answer)
                
            if answer in xiu_last_answer:
                # printgreen(f"{last_answer},{answer}")
                is_can_answer[idx]["pre_answer"] = xiu_last_answer
                is_can_answer_flag = True
            
        for child in root.children:
            # print(child.node_type)
            num += 1
            queue.put(child)
        
    is_can_answer[idx]["is_can_answer_flag"] = is_can_answer_flag
            
    alls += num
    mmax = max(num, mmax)
    

    
print(alls)
print(mmax)

print(len(is_can_answer))

save_file = "charxiv_select_0426.json"
with open(save_file, 'w') as f:
    json.dump(is_can_answer, f, indent=4)