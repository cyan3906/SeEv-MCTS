

import json

file = "/home/jncsnlp4/SSD2/ZJJ/VReST/mllms/evaluation/scoring温度为0.7的第一遍.json"

with open(file, "r") as f:
    data = json.load(f)

good = bad = 0
save_file = "qwen2_is_numeric=false_llm_score=true.json"

dict1 = {}
for i,j in data.items():
    if "llm_score" not in j or j["is_numeric"] == "true":
        continue
    
    if not j["llm_score"]:
        continue
    
    dict1[i] = {}
    dict1[i]["extracted_answer"] = j["extracted_answer"]
    dict1[i]["answer"] = j["answer"]
    dict1[i]["llm_score"] = j["llm_score"]
    

with open(save_file, "w") as f:
    json.dump(dict1, f, ensure_ascii=False, indent=4)





