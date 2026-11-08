import json

dataset = "mathvista"

# data_score = "/home/jncsnlp4/SSD2/ZJJ/VReST/data/CharXiv/data/reasoning_val.json"
data_score = "/home/jncsnlp4/SSD2/ZJJ/VReST/data/MathVista/data/testmini.json"

with open(data_score, "r") as f:
    data = json.load(f)

# qwen2_score = f"/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=baseline/data={dataset}/llm=Qwen2-VL-7B-Instruct/prompt_method=mctsv8/ext_llm=Qwen2-7B-Instruct/method=ans_and_score/field=max_vote_terminal/n=10/reward_method=self_consistency/scores-reasoning_val.json"

# qwen25_score = f"/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=baseline/data={dataset}/llm=Qwen2-VL-7B-Instruct/prompt_method=mctsv8/ext_llm=Qwen2.5-7B-Instruct/method=ans_and_score/field=max_vote_terminal/n=10/reward_method=self_consistency/scores-reasoning_val.json"
# qwen3_nothink_score = f"/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=baseline/data={dataset}/llm=Qwen2-VL-7B-Instruct/prompt_method=mctsv8/ext_llm=Qwen3-8B/method=ans_and_score/field=max_vote_terminal/n=10/reward_method=self_consistency/scores-reasoning_val.json"
# qwen3_think_score = f"/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=baseline/data={dataset}/llm=Qwen2-VL-7B-Instruct/prompt_method=mctsv8/ext_llm=Qwen3-8B/method=ans_and_score/field=max_vote_terminal/n=10/reward_method=self_consistency/think-scores-reasoning_val.json"

# llama31_score = f"/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=baseline/data={dataset}/llm=Qwen2-VL-7B-Instruct/prompt_method=mctsv8/ext_llm=Meta-Llama-3.1-8B-Instruct/method=ans_and_score/field=max_vote_terminal/n=10/reward_method=self_consistency/scores-reasoning_val.json"

# mistral_score = f"/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=baseline/data={dataset}/llm=Qwen2-VL-7B-Instruct/prompt_method=mctsv8/ext_llm=mistral/method=ans_and_score/field=max_vote_terminal/n=10/reward_method=self_consistency/scores-reasoning_val.json"

qwen2_score = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=baseline/data=mathvista/llm=Qwen2-VL-7B-Instruct/prompt_method=mctsv8/ext_llm=Qwen2-7B-Instruct/method=ans_and_score/field=max_vote_terminal/n=10/reward_method=self_consistency/scores.json"
qwen25_score = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=baseline/data=mathvista/llm=Qwen2-VL-7B-Instruct/prompt_method=mctsv8/ext_llm=qwen25/method=ans_and_score/field=max_vote_terminal/n=10/reward_method=self_consistency/scores.json"
qwen3_nothink_score = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=baseline/data=mathvista/llm=Qwen2-VL-7B-Instruct/prompt_method=mctsv8/ext_llm=Qwen3-8B/method=ans_and_score/field=max_vote_terminal/n=10/reward_method=self_consistency/scores.json"
qwen3_think_score = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=baseline/data=mathvista/llm=Qwen2-VL-7B-Instruct/prompt_method=mctsv8/ext_llm=Qwen3-8B/method=ans_and_score/field=max_vote_terminal/n=10/reward_method=self_consistency/think-scores.json"
llama31_score = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=baseline/data=mathvista/llm=Qwen2-VL-7B-Instruct/prompt_method=mctsv8/ext_llm=Meta-Llama-3.1-8B-Instruct/method=ans_and_score/field=max_vote_terminal/n=10/reward_method=self_consistency/scores.json"
mistral_score = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=baseline/data=mathvista/llm=Qwen2-VL-7B-Instruct/prompt_method=mctsv8/ext_llm=mistral/method=ans_and_score/field=max_vote_terminal/n=10/reward_method=self_consistency/scores.json"

with open(qwen2_score, "r") as f:
    qwen2_data = json.load(f)

with open(qwen25_score, "r") as f:
    qwen25_data = json.load(f)

with open(qwen3_nothink_score, "r") as f:
    qwen3_nothink_data = json.load(f)

with open(qwen3_think_score, "r") as f:
    qwen3_think_data = json.load(f)

with open(llama31_score, "r") as f:
    llama31_data = json.load(f)

with open(mistral_score, "r") as f:
    mistral_data = json.load(f)

# 创建结果列表来存储符合条件的记录
results = []
num = 0
for i,j in qwen3_think_data.items():
    ok = qwen2_data[i]["llm_score"] and qwen25_data[i]["llm_score"] and qwen3_nothink_data[i]["llm_score"] and llama31_data[i]["llm_score"] and mistral_data[i]["llm_score"]
    
    if not j["llm_score"] and ok:
        num += 1
        
        # 将符合条件的记录添加到结果列表中
        result_record = {
            "id": i,
            "response": j['response'],
            "choices": data[i]["choices"],
            "ground_answer": data[i]['answer'],
            "qwen2_extracted_answer": qwen2_data[i]['extracted_answer'],
            "qwen2_llm_score": qwen2_data[i]["llm_score"],
            "qwen25_extracted_answer": qwen25_data[i]['extracted_answer'],
            "qwen25_llm_score": qwen25_data[i]["llm_score"],
            "qwen3_nothink_extracted_answer": qwen3_nothink_data[i]['extracted_answer'],
            "qwen3_nothink_llm_score": qwen3_nothink_data[i]["llm_score"],
            "llama31_extracted_answer": llama31_data[i]['extracted_answer'],
            "llama31_llm_score": llama31_data[i]["llm_score"],
            "mistral_extracted_answer": mistral_data[i]['extracted_answer'],
            "mistral_llm_score": mistral_data[i]["llm_score"],
            "qwen3_think_extracted_answer": j['extracted_answer'],
            "qwen3_think_llm_score": j["llm_score"],
        }
        results.append(result_record)

print(f"符合条件的记录总数: {num}")

# 将结果写入JSON文件

output_file = f"base2_filtered_results_{dataset}.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"结果已保存到 {output_file} 文件")

print(num)