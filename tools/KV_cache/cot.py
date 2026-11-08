
import json
import sys
sys.path.insert(0,"/home/jncsnlp4/SSD2/ZJJ/VReST")
import time,os
from tools.user_model import Qwen2
from prompt import plan_prompt, question_prefix, subquestion_prefix, overall_question_prefix, answer_prefix
import re

def get_message(question, image_file):
    messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            
            {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": image_file,
                },
                {"type": "text", "text": question},
            ],
            },
        ]

    return messages

def extract_sub_questions_and_answers(response, step):
    sub_question_pattern = subquestion_prefix.format(step=step, subquestion="").strip()
    sub_answer_pattern = answer_prefix.format(step=step, answer="").strip()
    # printcyan(f"{sub_question_pattern}")
    # printblue(f"{sub_answer_pattern}")
    # 找到pattern后的那个subquestion
    response = response.strip()
    # printyellow(f"{response}")
    try:
        # printgreen(f"{response}")
        sub_question = response.split(sub_question_pattern)[1].split("\n")[0].strip()
        sub_answer = response.split(sub_question_pattern)[1].split("\n")[-1].strip()
        # 将"Answer %d:"替换为空
        sub_answer = re.sub(r"Answer \d+:", "", sub_answer).strip()
    except:
        printred(f"Errorextract_sub_questions_and_answers: {response}")
        sub_question, sub_answer = "", ""
    return sub_question, sub_answer
    
llm_path = "/home/jncsnlp4/SSD2/model/qwen2-vl-intruct"

llm = Qwen2(llm_path, enable_thinking = False)

text_data = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=baseline/data=mathvista/llm=Qwen2-VL-7B-Instruct/prompt_method=mctsv8/generation.json"
image_data = "/home/jncsnlp4/SSD2/ZJJ/VReST/data/MathVista/data/images/testmini/ALL"

with open(text_data, "r") as f:
    text_data = json.load(f)

save_data = {}
from tqdm import tqdm

for i,j in tqdm(text_data.items()):
    time1 = time.time()
    question = j["question"]
    image_file = os.path.join(image_data, str(i)+ ".jpg")
    save_data[i] = {
        "response": [
            
        ]
    }
    
    step = 1
    while step <= 6:
        prompt = plan_prompt + question_prefix.format(question = question,subquestion = "") + subquestion_prefix.format(step = step)
            
        messages = get_message(prompt, image_file)
        
        response = llm.generate(messages,n = 5,stop_token = [subquestion_prefix.format(step = step + 1)])["response"][0]
        sub_question, sub_answer = extract_sub_questions_and_answers(response, step)
        
        if "Now we can answer the question:" in sub_question:
            res = overall_question_prefix.format(step = step, overall_question = sub_question) + "\n" + answer_prefix.format(step = step, answer = sub_answer)
        else:
            res = subquestion_prefix.format(step = step, subquestion = sub_question) + "\n" + answer_prefix.format(step = step, answer = sub_answer)
        
        save_data[i]["response"].append(res)
        
        prompt += res + "\n"
        
        step += 1
        
    save_data[i]["time"] = time.time() - time1


with open("no_kv.json", "w") as f:
    json.dump(save_data, f, indent = 4)

