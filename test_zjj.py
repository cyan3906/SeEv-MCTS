# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

# # ===================== 1. 加载Qwen2-VL模型（使用transformers） =====================
# # 使用本地的Qwen2-VL模型路径
# model_path = "/home/jncsnlp4/SSD2/model/qwen2-vl-intruct"
# print(f"正在使用transformers加载Qwen2-VL模型：{model_path}")

# # 加载模型和分词器
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     trust_remote_code=True,
#     torch_dtype="auto",
#     device_map="auto"
# )

# # ===================== 2. 自定义你的前缀文本【核心修改这里！】 =====================
# prefix_text = "我喜欢编程"  # 可修改成：我喜欢春天、我喜欢旅行、我喜欢读书 等等任意内容

# # ===================== 3. 核心：逐Token生成 + 每次都选择第二个概率最高的token =====================
# # 第一步：使用聊天模板处理输入
# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": prefix_text}
# ]

# # 使用Qwen2-VL的聊天模板
# text = tokenizer.apply_chat_template(
#     messages, 
#     tokenize=False, 
#     add_generation_prompt=True
# )

# # 将文本转换为token
# input_ids = tokenizer(text, return_tensors="pt").to(model.device)

# # 存储完整生成的文本和所有预测的token
# generated_tokens = []  # 存放【所有预测出来的后续token】
# complete_text = prefix_text  # 拼接后的完整文本，初始是前缀

# print(f"===== 前缀文本：{prefix_text} =====")
# print(f"===== 开始逐一生成后续Token，每次都选择第二个概率最高的token ====")
# print(f"使用模型：Qwen2-VL + transformers ({model_path})\n")

# # 控制最大生成Token数，防止无限生成，可自行调整
# MAX_NEW_TOKENS = 50
# token_count = 0  # 预测的Token计数

# # 逐Token生成循环
# while token_count < MAX_NEW_TOKENS:
#     # 获取模型输出
#     with torch.no_grad():
#         outputs = model(**input_ids)
#         logits = outputs.logits
    
#     # 获取最后一个位置的logits
#     next_token_logits = logits[0, -1, :]
    
#     # 应用温度参数
#     temperature = 0.7
#     next_token_logits = next_token_logits / temperature
    
#     # 获取概率分布
#     probs = torch.softmax(next_token_logits, dim=-1)
    
#     # 获取前2个概率最高的token
#     top_2_probs, top_2_indices = torch.topk(probs, 2)
    
#     # 选择第二个概率最高的token（索引1）
#     second_token_id = top_2_indices[1]
#     second_token_prob = top_2_probs[1]
    
#     # 解码token
#     second_token = tokenizer.decode(second_token_id.item())
    
#     # 获取第一个概率最高的token用于对比
#     first_token = tokenizer.decode(top_2_indices[0].item())
#     first_token_prob = top_2_probs[0]
    
#     # 打印当前生成的token和对比信息
#     print(f"第{token_count + 1}个Token: {second_token}")
#     print(f"  第一个token: {first_token} (概率: {first_token_prob:.4f})")
#     print(f"  第二个token: {second_token} (概率: {second_token_prob:.4f})")
    
#     # 检查是否为结束符
#     if second_token in ["</s>", "

# from tabulate import tabulate  # 添加表格显示
# import time

# def print_result_table(data, headers):
#     """打印漂亮的表格"""
#     print(tabulate(data, headers=headers, tablefmt="grid", numalign="center", stralign="center"))

# import json,re,time
# import os

# def print_section(title, width=50):
#     """打印章节标题"""
#     print("\n" + "=" * width)
#     print(f" {title} ".center(width, "="))
#     print("=" * width)

# print_section("模型评估结果")

# incorrect_samples = [1,2,3,4]
# acc = 107
# accuracy = acc / 189

# detail_data = [
#         ["正确率", f"{accuracy:.2%}"],
#         ["错误率", f"{(1-accuracy):.2%}"],
#         ["正确/错误比例", f"{accuracy}:{len(incorrect_samples)}"],
#         ["评估完成时间", time.strftime("%Y-%m-%d %H:%M:%S")]
#     ]
    
# print_result_table(detail_data, ["详细信息", "数值"])

def generate():
        
    file1 = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=baseline/data=charxiv/llm=Qwen2-VL-7B-Instruct/prompt_method=mctsv8/ori-gen-reasoning_val.json"

    file2 = "/home/jncsnlp4/SSD2/ZJJ/VReST/charxiv_pk_best_answer_true_soft_0.5.json"

    file3 = "/home/jncsnlp4/SSD2/ZJJ/VReST/charxiv_pk_best_answer_false_soft_0.5.json"

    file4 = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=baseline/data=charxiv/llm=Qwen2-VL-7B-Instruct/prompt_method=mctsv8/gen-reasoning_val.json"

    with open(file1, "r") as f:
        data1 = json.load(f)

    with open(file2, "r") as f:
        data2 = json.load(f)

    with open(file3, "r") as f:
        data3 = json.load(f)



# import json,os,time

# mathvision_qa_score_file = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=qa.mathvision.ALL/data=mathvision/llm=Qwen2-VL-7B-Instruct/prompt_method=qa/ext_llm=qwen25/method=ans_and_score/field=max_vote_terminal/n=10/reward_method=self_consistency/scores.json"
# mathvision_baseline_score_file = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=baseline/data=mathvision/llm=Qwen2-VL-7B-Instruct/prompt_method=mctsv8/old-outcomes/ext_llm=Qwen2.5-7B-Instruct/method=ans_and_score/field=max_vote_terminal/n=10/reward_method=self_consistency/scores.json"
# mathvista_qa_score_file = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=qa.mathvista.ALL/data=mathvista/llm=Qwen2-VL-7B-Instruct/prompt_method=qa/ext_llm=qwen25/method=ans_and_score/field=max_vote_terminal/n=10/reward_method=self_consistency/extraction.json"
# mathvista_baseline_score_file = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=baseline/data=mathvista/llm=Qwen2-VL-7B-Instruct/prompt_method=mctsv8/old-comes/ext_llm=qwen25/method=ans_and_score/field=max_vote_terminal/n=10/reward_method=self_consistency/scores.json"
# charxiv_qa_score_file = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=qa.charxiv.ALL/data=charxiv/llm=Qwen2-VL-7B-Instruct/prompt_method=qa/ext_llm=qwen25/method=ans_and_score/field=max_vote_terminal/n=10/reward_method=self_consistency/scores-reasoning_val.json"
# charxiv_baseline_score_file = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=baseline/data=charxiv/llm=Qwen2-VL-7B-Instruct/prompt_method=mctsv8/old-outcomes/ext_llm=Qwen2.5-7B-Instruct/method=ans_and_score/field=max_vote_terminal/n=10/reward_method=self_consistency/scores-reasoning_val.json"

# with open(mathvision_qa_score_file, "r") as f:
#     mathvision_qa_score = json.load(f)
# with open(mathvision_baseline_score_file, "r") as f:
#     mathvision_baseline_score = json.load(f)

# with open(mathvista_qa_score_file, "r") as f:
#     mathvista_qa_score = json.load(f)
# with open(mathvista_baseline_score_file, "r") as f:
#     mathvista_baseline_score = json.load(f)

# with open(charxiv_qa_score_file, "r") as f:
#     charxiv_qa_score = json.load(f)
# with open(charxiv_baseline_score_file, "r") as f:
#     charxiv_baseline_score = json.load(f)


# mathvision_list = []
# mathvista_list = []
# charxiv_list = []



# for i,j in mathvision_qa_score.items():
#     if j["llm_score"] and not mathvision_baseline_score[i]["llm_score"]:
#         mathvision_list.append(j)

# for i,j in mathvista_qa_score.items():
#     if j["llm_score"] and not mathvista_baseline_score[i]["llm_score"]:
#         mathvista_list.append(j)


# for i,j in charxiv_qa_score.items():
#     if j["llm_score"] and not charxiv_baseline_score[i]["llm_score"]:
#         charxiv_list.append(j)


# print(f"mathvision_list: {len(mathvision_list)}")
# print(f"mathvista_list: {len(mathvista_list)}")
# print(f"charxiv_list: {len(charxiv_list)}")

import json
file1 = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=ours2v2.mathvision.ALL/data=mathvision/llm=Qwen2-VL-7B-Instruct/prompt_method=our2v2/ext_llm=qwen25/method=ans_and_score/field=max_vote_terminal/n=10/reward_method=self_consistency/scores.json"
with open(file1, "r") as f:
    data1 = json.load(f)
    
list1 = []
list2 = []
list3 = ['45', '60', '66', '79', '130', '159', '254', '259', '325', '330', '442', '473', '526', '559', '675', '761', '813', '893', '913', '1167', '1451', '1781', '1889', '2588']

for i,j in data1.items():
    if "llm_score" not in j:
        continue
    if j["llm_score"]:
        list1.append(i)
    else:
        list2.append(i)

print(len(list1))
print(len(list2))
print(len(set(list1) & set(list3)))

print(len(set(list2) & set(list3)))



