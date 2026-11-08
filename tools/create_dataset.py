
import json,os,time

json_data = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=baseline/data=charxiv/llm=Qwen2-VL-7B-Instruct/prompt_method=mctsv8/ext_llm=Qwen3-8B/method=ans_and_score/field=max_vote_terminal/n=10/reward_method=self_consistency/think-scores-reasoning_val.json"

with open(json_data, "r") as f:
    data = json.load(f)
    
list1 = []
list2 = []

for i,j in data.items():
    if j["llm_score"]:
        list1.append(i)
    else:
        list2.append(i)

print(list1[:100])

print(list1[100:200])

print(list1[200:])

print(len(list1))

print(len(list2))

print(list2[:50])

print(list2[50:250])

print(list2[250:400])

print(list2[400:])



