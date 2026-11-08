

# import json


# file = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=baseline/data=charxiv/llm=Qwen2-VL-7B-Instruct/prompt_method=mctsv8/ext_llm=qwen2/method=ans_and_score/field=max_vote_terminal/n=10/reward_method=self_consistency/scores-reasoning_val.json"
# score_file = "/home/jncsnlp4/SSD2/ZJJ/VReST/data/CharXiv/data/reasoning_val.json"

# with open(file, "r") as f:
#     data = json.load(f)

# with open(score_file, "r") as f:
#     score_data = json.load(f)
    
# file2 = "charxiv-look-qwen2-false.json"

# dict1 = {}

# for i,j in data.items():
#     if "response" not in j or "extracted_answer" not in j:
#         continue
#     if j["llm_score"]:
#         continue
#     dict1[i] = {}
#     # dict1[i]["choices"] = j["choices"]
#     dict1[i]["answer"] = score_data[i]["answer"]
#     # dict1[i]["answer"] = j["answer"]
#     # dict1[i]["response"] = j["response"]
#     dict1[i]["extracted_answer"] = j["extracted_answer"]
#     dict1[i]["llm_score"] = j["llm_score"]
#     dict1[i]["my_check"] = 3

# with open(file2, "w") as f:
#     json.dump(dict1, f, ensure_ascii=False, indent=4)


import json

file = "charxiv_select_0426.json"

with open(file, "r") as f:
    data = json.load(f)
    
dict1 = {
    
}

mm = 0

for i,j in data.items():
    depth = j["depth"]
    for k,v in depth.items():
        s = set(list(v))
        
        mm = max(mm,len(s))
        
print(mm)