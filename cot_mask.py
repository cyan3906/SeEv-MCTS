

# 掩码策略 bfs思维树

"""
测试能不能使用stop token


"""

import json,os,time,re,sys,random
from copy import deepcopy
from tqdm import tqdm
from tools.user_model import Qwen2
from tools.color_print import *
from tools.get_all_data import data, data_qa_score, data_cot_score, data_baseline_score,IMAGE_DIR_CONFIG,get_messages

from tools.get_all_data import cot_prompt_template,cot_system_prompt,is_answer_consistent_prompt,cot_output_template,cot_output_system_promot

from tools.get_all_data import is_answer_consistent
from tools.get_all_data import (
    mathvision_cot_true_baseline_false,
    mathvision_cot_false_baseline_true,
    mathvista_cot_true_baseline_false,
    mathvista_cot_false_baseline_true,
    charxiv_cot_true_baseline_false,
    charxiv_cot_false_baseline_true,
    charxiv_cot_data
)

llm_path = "/home/jncsnlp4/SSD2/model/qwen2-vl-intruct"

llm = Qwen2(llm_path)

while 1:
    pass

task = "charxiv" # mathvision mathvista charxiv

# if task == "charxiv":
#     data_cot_score[task] = charxiv_cot_data
    

acc = loss = acc1 = 0
error_list = []
acc_list = []
mask_win = []
save_json = {}

time_start = time.time()

for i,j in tqdm(data[task].items(),desc=f"\033[91m mask {task}\033[0m",total=len(data[task])):
    
    if i not in charxiv_cot_false_baseline_true: #mathvision_cot_true_baseline_false
        continue
    
    printred("-----------")
    printgreen(f"目前掩码成功的数量为{len(mask_win)}")
    printred("-----------")
    
    printred(f"正在处理{i}")
    save_json[i] = {}
    
    try:
        if task == "charxiv":
            question = j["query"]
        else:
            question = j["question"]
        
        image_path = os.path.join(IMAGE_DIR_CONFIG[task], f"{i}.jpg")
        
        if task == "charxiv":
            reasoning = charxiv_cot_data[i]["decision"]["cot"]
            answer = charxiv_cot_data[i]["decision"]["response"]
        else:
            reasoning = data_cot_score[task][i]["decision"]["cot"]
            answer = data_cot_score[task][i]["decision"]["response"]
        
        pattern = r"Step \d+: (.*?)(?= Step \d+:|$)"
        steps = re.findall(pattern, reasoning)
        
        ori_steps = deepcopy(steps)
        printgreen(f"step的长度为{len(steps)}")
        
        if len(steps) == 1:
            continue
        
        # 尝试多次掩码
        win = True
        
        for __ in range(3):
            steps = deepcopy(ori_steps)
            mask_id = random.randint(0, len(steps) // 3)
            mask_id = 1
            printyellow(f"掩码第{__}次，掩码位置{mask_id},总长度{len(steps)}")
            if len(steps) == 1:
                steps = reasoning
            else:
                steps = steps[:mask_id] # 掩掉最后一步
                
            step_answer = ""
            for idx, content in enumerate(steps, 1):
                step_answer += f"Step {idx}: {content}\n"
            print(step_answer)
            # input("点击继续")
            yes = 0
            turn = 11
            
            for _ in range(turn):
                user_prompt = cot_output_template.format(question=question, answer=answer)
                messages = get_messages(cot_output_system_promot,user_prompt, image_path)
                predict_response = ""
                for __ in range(3):
                    predict_response = llm.get_response(messages,n = 1)["response"][0]
                    if "answer is" in predict_response:
                        predict_response = predict_response.split("answer is")[-1].strip()
                        break
                print(f"mask后答案：{predict_response},cot原答案：{answer}")
                
                if task == "charxivss":
                    if predict_response != "" and is_answer_consistent(llm, answer, predict_response):
                        yes += 1
                else:
                    if predict_response != "" and (answer in predict_response or predict_response in answer):
                        yes += 1
                # print(predict_response,yes)
                # input("点击继续")
                
            if yes > 9:
                if "llm_score" in data_cot_score[task][i] and data_cot_score[task][i]["llm_score"]:
                    acc += 1
                    acc_list.append(i)
                    save_json[i].update({"cot":True})
                    
                else:
                    loss += 1
                    save_json[i].update({"cot":False})
            else:
                win = False
                # go mcts
                if data_baseline_score[task][i]["llm_score"]:
                    acc1 += 1
                    save_json[i].update({"mcts":True})
                else:
                    save_json[i].update({"mcts":False})
            
            if not win:
                break
            
            with open(f"{task}_cot_mask.json", "w") as f:
                json.dump(save_json, f, ensure_ascii=False, indent=4)
            
            
            printblue(f"acc / loss = {acc} / {loss}")
        
        if win:
            printred("恭喜你，掩码成功")
            mask_win.append(i)
            
    except Exception as e:
        printred(f"{e}")
        error_list.append(i)
        continue


printblue(f"acc / loss = {acc} / {loss} ")
    
printyellow(f"acc: {acc + acc1} / {len(data[task])} = {(acc + acc1) / (len(data[task]))}")

printgreen(f"acc_list: {acc_list}")

time_end = time.time()

printpurple(f"所花的分钟数量: {(time_end - time_start) / 60}")


print(f"mask_win: {mask_win}")

print(len(mask_win))

print(error_list)

