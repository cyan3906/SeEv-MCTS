
from vllm import LLM, SamplingParams
import json,re,os
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from tools import (
    SYSTEM_SCORING_INST , 
    Mistral_SYSTEM_SCORING_INST,
    Mistral_SCORING_INST,
    SCORING_INST,
    system_judge_numeric_inst,
    judge_numeric_inst,
    USER_SCORING_INST
)
from collections import Counter 
from pydantic import BaseModel
from tqdm import tqdm

class Extract_Answer(BaseModel):
    extracted_answer: str
    score: int

def get_majority_answer(answers):
    if "score" in answers[0]:
        answers_extracted = [answer["extracted_answer"] for answer in answers]
        answers_score = [answer["score"] for answer in answers]
        # print(answers_extracted, answers_score)
        majority_score = Counter(answers_score).most_common(1)[0][0]
        # print(answers_extracted, answers_score, majority_score)
        remained_answers = [answers_extracted[i] for i in range(len(answers_extracted)) if answers_score[i] == majority_score]
        majority_answer = Counter(remained_answers).most_common(1)[0][0] # 对可能有重复的答案做一个取一个
        return majority_answer, majority_score
    elif "extracted_answer" in answers[0]:
        answers_extracted = [answer["extracted_answer"] for answer in answers]
        majority_answer = Counter(answers_extracted).most_common(1)[0][0] # 对可能有重复的答案做一个取一个
        return majority_answer, None
    
    else:
        majority_answer = Counter(answers).most_common(1)[0][0] # 对可能有重复的答案做一个取一个
        return majority_answer
    
def ext_ans_and_score(ground_truth, model_response, model,llm_name = "mistral"):
    
    if llm_name == "mistral":
        user_prompt = Mistral_SCORING_INST.format(model_response=model_response)
        messages = [
            {"role": "system", "content": Mistral_SYSTEM_SCORING_INST},
            {"role": "user", "content": user_prompt}
        ]
    else:
        user_prompt = SYSTEM_SCORING_INST + "model_response: " + model_response + "\n\nExtracted Answer:"
        messages = [
            {"role": "user", "content": user_prompt}
        ]
    max_patience = 10
    patience = 0
    while patience < max_patience:
        try:
            ori_responses = model.get_response(messages,json_format = Extract_Answer)["response"] # 调用大模型
            print(f"\033[92m{ori_responses}\033[0m")
            # responses = []
            if isinstance(ori_responses, list):
                # for response in ori_responses:
                #     if type(response) == str:
                #         response = response.replace("\\", "/")
                #         responses.append(json.loads(str(response)))
                #     else:
                #         responses.append(response)
                        
                # ori_responses = responses
                
                extracted_answer = get_majority_answer(ori_responses)
                
                # if score == None:
                #     response_json = {"extracted_answer": extracted_answer, "score": None}
                # else:
                #     response_json = {"extracted_answer": extracted_answer, "score": score == 1}
            else:
                extracted_answer = ori_responses
                # if type(ori_responses) == str:
                #     ori_responses = json.loads(ori_responses)
                # else:
                #     response_json = ori_responses
                # response_json["score"] = response_json["score"] == 1
                
            break
        
        except Exception as e:
            print(f"\033[91mError: {str(e)}\033[0m")
        patience += 1
    
    if patience == max_patience:
        print(f"Failed to get extracted_answer for prompt")
        # response_json = {"extracted_answer": "Extract Error!", "score": False}
        # ori_responses = [response_json]
        extracted_answer = "Error to get extracted_answer"
        
    # return response_json, ori_responses
    return extracted_answer

def pre_evl_ans_and_score(ground_truth, model_response, model):
    prompt = judge_numeric_inst.format(ground_truth=ground_truth, model_response=model_response)
    
    messages = [
        {"role": "system", "content": system_judge_numeric_inst}, 
        {"role": "user", "content": prompt}
    ]

    max_patience = 10
    patience = 0
    while patience < max_patience:
        try:
            ori_responses = model.get_response(messages)["response"] # 调用大模型
            print(f"\033[94m{ori_responses}\033[0m")
            # responses = []
            if isinstance(ori_responses, list):
                # for response in ori_responses:
                #     if type(response) == str:
                #         response = response.replace("\\", "/")
                #         responses.append(json.loads(str(response)))
                #     else:
                #         responses.append(response)
                        
                # ori_responses = responses
                
                is_numeric = get_majority_answer(ori_responses)
                
            else:
                is_numeric = ori_responses
                
                # if type(ori_responses) == str:
                #     ori_responses = json.loads(ori_responses)
                # else:
                #     response_json = ori_responses
                # response_json["score"] = response_json["score"] == 1
                
            break
        except Exception as e:
            print(f"\033[91mError: {str(e)}\033[0m")
        patience += 1
        
    if patience == max_patience:
        print(f"Failed to get is_numeric for prompt: {prompt}")
        # response_json = {"extracted_answer": "Extract Error!", "score": False}
        # ori_responses = [response_json]
        is_numeric = "Error to get is_numeric"
    # return response_json, ori_responses
    return is_numeric

def evl_ans_and_score(question, ground_truth, model_response, model):
    prompt = USER_SCORING_INST.format(question=question, ground_truth=ground_truth, model_response=model_response)
    messages = [
        {"role": "user", "content": prompt}
    ]
    max_patience = 10
    patience = 0
    while patience < max_patience:
        try:
            # 这里修改了将原来的get_yes_or_no替换为get_response
            # responses = model.get_yes_or_no(messages)["response"]
            
            responses = model.get_response(messages, json_format=Extract_Answer)["response"] 
            # print(responses)
            # input("点击输入")
            if isinstance(responses, list):
                yes_or_no = Counter(responses).most_common(1)[0][0]
                # print(f"\033[93m{yes_or_no}\033[0m")
                # response_json = {"extracted_answer": yes_or_no, "score": "yes" in yes_or_no.lower().strip()}
                
            else:
                # response_json = {"extracted_answer": responses, "score": "yes" in responses.lower().strip()}
                yes_or_no = responses
                
            break
        except Exception as e:
            print(f"Error: {str(e)}")
        patience += 1
    if patience == max_patience:
        print(f"Failed to get evaluation for prompt: {prompt}")
        # response_json = {"extracted_answer": "Extract Error!", "score": False}
        # responses = [response_json]
        yes_or_no = "Error to get evaluation"
        
    # return response_json, responses
    return yes_or_no

class Qwen25_Model(): # 适配qwen2 和 qwen25
    def __init__(self,llm_path):
        self.llm = LLM(
            model=llm_path,
            max_model_len = 4096,
            max_num_seqs = 512,
            gpu_memory_utilization = 0.85
        )
        
        self.processor = AutoProcessor.from_pretrained(llm_path)
    
    def get_response(self, messages=None, json_format=None):
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            # repetition_penalty=1.05,
            max_tokens=713,
            stop_token_ids=[],
            n=10,
        )
        
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        llm_inputs = {
            "prompt": prompt
        }
                
        outputs = self.llm.generate([llm_inputs], sampling_params=sampling_params, use_tqdm=False)  
        
        generated_text = []
        for output in outputs[0].outputs:
            generated_text.append(output.text)
        
        response = {
            "response": generated_text
        }
        
        return response
            
class Llama_Model(): # 适配llama3.1
    def __init__(self,llm_path):
        self.llm = LLM(
            model=llm_path,
            max_model_len = 4096,
            max_num_seqs = 512,
            gpu_memory_utilization = 0.85
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)

    def get_response(self, messages=None, json_format=None):
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            # repetition_penalty=1.05,
            max_tokens=713,
            stop_token_ids=[],
            n=10,
        )
        
        inputs = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        output = self.llm.generate(inputs, sampling_params, use_tqdm=False)
        generated_text = []
        
        # pattern = r'"extracted_answer": "([^"]+)",\n\s*"score": (\d+)'
        # for output in output[0].outputs:
        #     res = re.search(pattern, output.text)
        #     if res:
        #         extracted_answer = res.group(1)  # 提取 答案内容
        #         score = int(res.group(2))
        #         print(extracted_answer, score)
        #         generated_text.append({"extracted_answer": extracted_answer, "score": score})
        # pattern = r'"extracted_answer": "([^"]+)"'
        pattern = r"Extracted Answer:\s*(.*?)(?=\n|$|['\"])"
        print(2222)
        for output in output[0].outputs:
            res = re.search(pattern, output.text)
            
            if res:
                extracted_answer = res.group(1)  # 提取 答案内容
                print(1111,extracted_answer)
                generated_text.append(extracted_answer)
            else:
                generated_text.append(output.text)
                
        response = {
            "response": generated_text
        }
        
        return response

class Mistral_Model():
    def __init__(self,llm_path):
        self.llm = LLM(
            model=llm_path,
            max_model_len = 4096,
            max_num_seqs = 512,
            gpu_memory_utilization = 0.85
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)

    def get_response(self, messages=None, json_format=None):
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            # repetition_penalty=1.05,
            max_tokens=713,
            stop_token_ids=[],
            n=10,
        )
        
        inputs = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        output = self.llm.generate(inputs, sampling_params, use_tqdm=False)
        generated_text = []
        
        pattern = r'"extracted_answer": "([^"]+)"'
        for output in output[0].outputs:
            res = re.search(pattern, output.text)
            if res:
                extracted_answer = res.group(1)  # 提取 答案内容
                generated_text.append(extracted_answer)
            else:
                generated_text.append(output.text)
                
        response = {
            "response": generated_text
        }
        
        return response

class Qwen3_Model():
    def __init__(self,llm_path):
        self.llm = LLM(
            model=llm_path,
            max_model_len = 4096,
            max_num_seqs = 512,
            gpu_memory_utilization = 0.85
        )
        
        self.processor = AutoProcessor.from_pretrained(llm_path)

    def get_response(self, messages=None, json_format=None):
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            # repetition_penalty=1.05,
            max_tokens=713,
            stop_token_ids=[],
            n=10,
        )

        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            # enable_thinking = self.llm_cfg.enable_thinking,
            enable_thinking = False # 是否打开qwen3的思考模式
        )

        llm_inputs = {
            "prompt": prompt
        }
        
        outputs = self.llm.generate([llm_inputs], sampling_params=sampling_params, use_tqdm=False)
        generated_text = []
        
        for output in outputs[0].outputs:
            generated_text.append(output.text)
        
        # pattern = r'"extracted_answer": "([^"]+)",\n\s*"score": (\d+)'
        # generated_text = []
        # for output in outputs[0].outputs:
        #     # print(output.text)
        #     # print(output.text)
        #     res = re.search(pattern, output.text)
        #     if res:
        #         extracted_answer = res.group(1)  # 提取 答案内容
        #         score = int(res.group(2))
        #         # print(extracted_answer, score)
        #         generated_text.append({"extracted_answer": extracted_answer, "score": score})
        
        response = {
            "response": generated_text
        }
        
        return response
    
if __name__ == "__main__":
    
    llm_name = "mistral" # qwen2 qwen25 qwen3 qwen35 llama3.1 mistral
    
    if llm_name == "qwen2":
        llm_path = "/home/jncsnlp4/SSD2/model/Qwen2-7b-Instruct"
        model = Qwen2_Model(llm_path)
    elif llm_name == "qwen25":
        llm_path = "/home/jncsnlp4/SSD2/model/Qwen2.5-7b-Instruct"
        model = Qwen25_Model(llm_path)
    elif llm_name == "qwen3":
        llm_path = "/home/jncsnlp4/SSD2/model/Qwen3-8b"
        model = Qwen3_Model(llm_path)
    elif llm_name == "llama3.1":
        llm_path = "/home/jncsnlp4/SSD2/model/Meta-Llama-3.1-8B-Instruct"
        model = Llama_Model(llm_path)
    elif llm_name == "mistral":
        llm_path = "/home/jncsnlp4/SSD2/model/Mistral-7b-instruct"
        model = Mistral_Model(llm_path)
    else:
        raise ValueError(f"LLM name {llm_name} not supported!")
    
    data_file = "generation.json"
    is_use_numeric = False # True False
    if is_use_numeric:
        desc = "评估带有问题的数字验证"
    else:
        desc = "评估带有问题的无数字验证"
    save_file = llm_path.split("/")[-1]+f"_scoring_{desc}.json"
        
    with open(data_file, "r") as f:
        data = json.load(f)
    
    if os.path.exists(save_file):
        print("\033[93mScoring file exists, loading...\033[0m")
        with open(save_file, "r") as f:
            save_dict = json.load(f)
    else:
        save_dict = {}
        
    # 先抽取答案
    for i,j in tqdm(data.items(), desc="Extracting Answers"):
        if i in save_dict:
            continue
        
        save_dict[i] = {}
        
        if "response" not in j or j["response"] == "Failed to generate response!":
            save_dict[i]["response"] = "null"
            continue

        extracted_answer = None
        x = 0
        y = 10
        while (extracted_answer is None and x < y) or (extracted_answer == "null" and x < y):
            extracted_answer = ext_ans_and_score(j["answer"], j["response"], model,llm_name = llm_name)
            x += 1
        
        save_dict[i]["question"] = j["question"]
        save_dict[i]["choices"] = j["choices"]
        save_dict[i]["response"] = j["response"]
        
        if extracted_answer in ["A","B","C","D","E","F","G","H"]:
            extracted_answer = j["choices"][ord(extracted_answer)-65]
        
        if extracted_answer in ["(A)","(B)","(C)","(D)","(E)","(F)","(G)","(H)"]:
            extracted_answer = j["choices"][ord(extracted_answer[1])-65]
            
        save_dict[i]["extracted_answer"] = extracted_answer
        save_dict[i]["answer"] = j["answer"]

        with open(save_file, "w") as f:
            json.dump(save_dict, f, ensure_ascii=False, indent=4)
    
    num = 0
    
    # 再做评估
    for i,j in tqdm(save_dict.items(), desc="Evaluating Answers"):
        if "extracted_answer" not in j or "llm_score" in j:
            continue
        
        is_numeric = pre_evl_ans_and_score(j["answer"], j["extracted_answer"], model)
        
        if is_use_numeric and ("yes" in is_numeric.lower().strip() or "true" in is_numeric.lower().strip()):
            if str(j["answer"]).strip() == str(j["extracted_answer"]).strip():
                save_dict[i]["llm_score"] = True
            else:
                save_dict[i]["llm_score"] = False
            
        else:
            num += 1
            # 进行评估
            question = save_dict[i]["question"]
            if save_dict[i]["choices"]:
                for _,option in enumerate(save_dict[i]["choices"]):
                    question += f"\n{chr(65+_)}: {option}\n"
            
            yes_or_no = evl_ans_and_score(question, j["answer"], j["extracted_answer"], model)
            # yes_or_no = evl_ans_and_score(j["answer"], j["extracted_answer"], model)
            save_dict[i]["llm_score"] = "yes" in yes_or_no.lower().strip()
            
            
        save_dict[i]["is_numeric"] = is_numeric
            
        with open(save_file, "w") as f:
            json.dump(save_dict, f, ensure_ascii=False, indent=4)
    
    with open(save_file, "r") as f:
        save_dict = json.load(f)
    
    acc = 0
    for i,j in save_dict.items():
        if "llm_score" in j and j["llm_score"]:
            acc += 1
        
    print(f"\033[93m{llm_name}_{desc} 准确率: {acc} / {len(save_dict)} = {acc/len(save_dict):.4f}\033[0m")
    print(f"直接跑的次数{num}")
