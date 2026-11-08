

import json
import os

json_file = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=ours2v3.2.mathvista.GPS/data=custom_mathvista/llm=Qwen2-VL-7B-Instruct/prompt_method=our2v3/ext_llm=Qwen2.5-7B-Instruct/method=ans_and_score/field=max_vote_terminal/n=10/reward_method=self_consistency/scores.json"

with open(json_file, "r") as f:
    data = json.load(f)

new_file = os.path.join("/home/jncsnlp4/SSD2/ZJJ/VReST/tools", "generaton_eye.json")

print(new_file)

dict1 = {}

num = 0

for i,j in data.items():
    if j["true_false"]:
        dict2 = {}
        dict2["choices"] = j.get("choices")
        dict2["answer"] = j.get("answer")
        dict2["response"] = j.get("response")
        dict2["time"] = j.get("time")
        dict1[i] = dict2
        num += 1

print(num)

with open(new_file, "w") as f:
    json.dump(dict1, f, ensure_ascii=False, indent=4)













