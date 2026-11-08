

import json

file = "/home/jncsnlp4/SSD2/ZJJ/VReST/mllms/evaluation/mathvista/scoring.json"

save_file = "look_mathvista.json"
with open(file, "r") as f:
    data = json.load(f)

dict1 = {}
for i,j in data.items():
    if "llm_score" not in j:
        continue
    
    if j["is_numeric"] == "false" and j["llm_score"]:
        dict1[i] = {}
        dict1[i]["extracted_answer"] = j["extracted_answer"]
        dict1[i]["answer"] = j["answer"]
        dict1[i]["llm_score"] = j["llm_score"]
    
with open(save_file, "w") as f:
    json.dump(dict1, f, ensure_ascii=False, indent=4)
    




