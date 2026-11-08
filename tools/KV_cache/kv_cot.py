import json
import sys
sys.path.insert(0,"/home/jncsnlp4/SSD2/ZJJ/VReST")
import time, os
from tools.user_model import Qwen2
from prompt import plan_prompt, question_prefix, subquestion_prefix, overall_question_prefix, answer_prefix
import re
from tqdm import tqdm


# def get_message(question, image_file=None):
#     content = []

#     # ✅ 只在第一步带 image（避免重复计算）
#     if image_file is not None:
#         content.append({
#             "type": "image",
#             "image": image_file,
#         })

#     content.append({"type": "text", "text": question})

#     return [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": content},
#     ]

def get_message(query, image_url=None):
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        
        {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": image_url,
            },
            {"type": "text", "text": query},
        ],
        },
    ]
    
    return messages


def extract_sub_questions_and_answers(response, step):
    sub_question_pattern = subquestion_prefix.format(step=step, subquestion="").strip()
    sub_answer_pattern = answer_prefix.format(step=step, answer="").strip()
    response = response.strip()
    try:
        # printgreen(f"{response}")
        sub_question = response.split(sub_question_pattern)[1].split("\n")[0].strip()
        sub_answer = response.split(sub_question_pattern)[1].split("\n")[-1].strip()
        # 将"Answer %d:"替换为空
        sub_answer = re.sub(r"Answer \d+:", "", sub_answer).strip()
    except:
        # printred(f"Errorextract_sub_questions_and_answers: {response}")
        sub_question, sub_answer = "", ""
    return sub_question, sub_answer


llm_path = "/home/jncsnlp4/SSD2/model/qwen2-vl-intruct"
llm = Qwen2(llm_path, enable_thinking=False) # True False

text_data_path = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=baseline/data=mathvista/llm=Qwen2-VL-7B-Instruct/prompt_method=mctsv8/generation.json"
image_data = "/home/jncsnlp4/SSD2/ZJJ/VReST/data/MathVista/data/images/testmini/ALL"

with open(text_data_path, "r") as f:
    text_data = json.load(f)

save_data = {}

for i, j in tqdm(text_data.items(),desc="Processing"):
    time1 = time.time()

    question = j["question"]
    image_file = os.path.join(image_data, str(i) + ".jpg")

    save_data[i] = {"response": []}

    # ✅ history 作为 prefix（KV cache 关键）
    history = plan_prompt + question_prefix.format(question=question) + "\n"

    for step in tqdm(range(1, 7)):

        prompt = history + subquestion_prefix.format(step=step, subquestion="")

        # ✅ 只有第一步带 image
        # if step == 1:
        #     messages = get_message(prompt, image_file)
        # else:
        #     messages = get_message(prompt, None)

        messages = get_message(history, image_file)
        
        # 🚀 核心：n=5 → KV cache 共享 prefix
        responses = llm.get_response(
            messages,
            n=5,
            stop_token=[subquestion_prefix.format(step=step + 1, subquestion="")]
        )["response"]

        # ✅ 选第一条（你也可以做 rerank）
        if subquestion_prefix.format(step=step, subquestion="") in responses[0]:
            response = responses[0]
        else:
            response = subquestion_prefix.format(step=step, subquestion="") + responses[0]
        print(f"\033[92m{response}\033[0m")
        
        sub_q, sub_a = extract_sub_questions_and_answers(response, step)
        print(f"\033[93m{sub_q}，{sub_a}\033[0m")
        
        input("点击继续")
        
        is_break = False
        
        if "Now we can answer the question:" in sub_q:
            res = overall_question_prefix.format(step=step, overall_question=sub_q) + "\n" + \
                  answer_prefix.format(step=step, answer=sub_a)
            is_break = True
        else:
            res = subquestion_prefix.format(step=step, subquestion=sub_q) + "\n" + \
                  answer_prefix.format(step=step, answer=sub_a)

        save_data[i]["response"].append(res)

        # ✅ 累积 history（形成长 prefix → KV cache 命中）
        history += res + "\n"
        print(f"\033[94m{history}\033[0m")
        if is_break:
            break
        
    save_data[i]["time"] = time.time() - time1


    with open("kv_cache.json", "w") as f:
        json.dump(save_data, f, indent=4)