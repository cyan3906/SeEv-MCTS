

import json,re,os


mathvista_dict = {
    "geometry problem solving" : "GPS",
    "figure question answering" : "FQA",
    "math word problem" : "MWP",
    "visual question answering" : "VQA",
    "textbook question answering" : "TQA",
    
}

score_data = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=baseline/data=mathvista/llm=Qwen2-VL-7B-Instruct/prompt_method=our1/ext_llm=Qwen3-8B/method=ans_and_score/field=max_vote_terminal/n=10/reward_method=self_consistency/think-scores.json"

with open(score_data, "r") as f:
    data = json.load(f)

score_dict = {
    "GPS" : [0,0],
    "FQA" : [0,0],
    "MWP" : [0,0],
    "VQA" : [0,0],
    "TQA" : [0,0],
}

for i,j in data.items():
    task = mathvista_dict[j["metadata"]["task"]]
    score_dict[task][1] += 1
    if j["llm_score"]:
        score_dict[task][0] += 1

for i,j in score_dict.items():
    print(f"{i}:{j[0]}/{j[1]}, acc:{j[0]/j[1]:.4f}")
    