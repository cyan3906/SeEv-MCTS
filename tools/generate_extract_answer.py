

import json,os,tqdm

dataset = "charxiv"
your_file = "pk_charxiv_true" # pk_charxiv_false pk_charxiv_true

if dataset == "charxiv":
    data_json = "/home/jncsnlp4/SSD2/ZJJ/VReST/data/CharXiv/data/reasoning_val.json"
    with open(data_json, 'r') as f:
        data = json.load(f)
    
    base_json = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=baseline/data=charxiv/llm=Qwen2-VL-7B-Instruct/prompt_method=mctsv8/gen-reasoning_val.json"
    with open(base_json, 'r') as f:
        base_data = json.load(f)

    if your_file == "pk_charxiv_true":
        target_json = "/home/jncsnlp4/SSD2/ZJJ/VReST/charxiv_pk_best_answer.json"
    else:
        target_json = "/home/jncsnlp4/SSD2/ZJJ/VReST/charxiv_pk_best_answer_false.json"
        
    with open(target_json, 'r') as f:
        target_json = json.load(f)
    
    dict2 = {}
    for i,j in target_json.items():
        dict1 = {}
        dict1["figure_id"] = base_data[i]["figure_id"]
        dict1["inst_category"] = base_data[i]["inst_category"]
        dict1["raw_question"] = base_data[i]["raw_question"]
        dict1["answer"] = data[i]["answer"]
        dict1["response"] = j["answer"]
        
        dict2[i] = dict1

    
    with open(your_file+".json", 'w') as f:
        json.dump(dict2, f, ensure_ascii=False, indent=4)
    
    print(f"已写入的数据集有{len(dict2)}个")
    
    







