

import json

input_file = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=ours1/data=charxiv/llm=Qwen2-VL-7B-Instruct/prompt_method=our1/gen-reasoning_val.json"

with open(file) as f:
    data = json.load(f)

dict1 = {}

for i,j in data.items():
    if "response" in j:
        dict1[i] = j

des = ""

output_file = "/home/jncsnlp4/SSD2/ZJJ/VReST/tools/random_output.json"

with open(output_file, "w") as f:
    json.dump(dict1, f, indent=4)













