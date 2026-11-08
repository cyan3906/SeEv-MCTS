import time
import os
import json
import random
import re
from openai import OpenAI
from pydantic import BaseModel
import base64
from PIL import Image
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from .color_print import get_logger,printyellow,printred,printgreen,printblue,printpurple,printpink,printwhite,printbold,printcyan

from .prompts import init_system_prompt
from .prompts import new_ost_prefix, ost_reward_prompt, ost_reward_prefix,question_prefix
from .answer_verify_prompt import direct_answer_verify_prefix, double_answer_verify_prompt,rewrite_answer_prefix
from enum import Enum, unique
from numpy import unique

import sys

sys.path.insert(0, "/home/jncsnlp4/SSD2/ZJJ/VReST/tools")
from create_logger import get_logger

class Node_Type:
    USER_QUESTION = "USER_QUESTION" # 用户原始问题节点
    # REPHRASED_USER_QUESTION = "REPHRASED_USER_QUESTION" # 重述后的用户问题节点
    DIRECT_ANSWER = "DIRECT_ANSWER" # 直接答案节点
    SUBQUESTION = "SUBQUESTION" # 子问题节点
    # RE_SUBANSWER = "RE_SUBANSWER" # 子问题的答案节点
    OST_STEP = "OST_STEP" # 步步思考节点

def save_json(file_name,data):
    with open(file_name, "w",encoding="utf-8") as f:
        json.dump(data, f, indent=4,ensure_ascii=False)

def generate_message(cfg,query, image_url, text_only=False, assistant_prompt=None):
    try:
        if cfg.llm.name == "Qwen2-VL-7B-Instruct":
            if text_only:
                messages=[
                    {"role": "system", "content": init_system_prompt},
                    
                    {
                        "role": "user",
                        "content": query,
                    },
                ]
            else:
                messages=[
                    {"role": "system", "content": init_system_prompt},
                    
                    {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": image_url,
                        },
                        {"type": "text", "text": query},
                    ],
                    },
                ]
        elif cfg.llm.name == "Qwen3-VL-8B-Instruct":
            if text_only:
                messages=[
                    {
                        "role": "user",
                        "content": query + assistant_prompt,
                    },
                ]
            else:
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image_url,
                            },
                            {"type": "text", "text": query + assistant_prompt if assistant_prompt is not None and assistant_prompt != "" else query},
                        ],
                    },
                ]
        else:
            assert False, f"generate_message: {cfg.llm.name} is not Exist!"
            
        if cfg.llm.name == "Qwen2-VL-7B-Instruct" and assistant_prompt is not None and assistant_prompt != "":
            messages.append({"role": "assistant", "content": assistant_prompt})
    except Exception as e:
        printred(f"generate_message: {str(e)}")
        return None
    # print(messages)
    return messages

# 没有问题
def default_call(cfg, messages, model, default_response, task_name="get response", n=1, continue_final_message=False, stop_token=None):
    for attempt in range(cfg.prompt_method.attempt_num):
        try:
            response = model.get_response(messages=messages, json_format=None, n=n, continue_final_message=continue_final_message, stop_token=stop_token)
            # printcyan(f"{response}")
            if "response" in response:
                response = response["response"]
            return response
        except Exception as e:
            printred(f"Errordefault_call: {str(e)}")
            print(f"The model response was:\n {response}")
            # if attempt == cfg.prompt_method.attempt_num-1:
            #     print(f"Failed to {task_name} after {cfg.prompt_method.return_num} attempts. Error: {str(e)}")
            #     return default_response
            
def reward_call(cfg, messages, model, node_depth,is_PK = False,is_answer_verify = False,answer1 = None , answer2 = None):
    for attempt in range(cfg.prompt_method.attempt_num):
        try:
            if is_PK:
                print(1)
                response = model.get_reward(messages=messages,n = cfg.prompt_method.return_num, node_depth=node_depth,is_PK = is_PK,answer1 = answer1 , answer2 = answer2)
                print(2)
            elif is_answer_verify:
                print(3)
                response = model.get_reward(messages=messages,n = 3, node_depth=node_depth,is_answer_verify = is_answer_verify)
                print(4)
            else:
                print(5)
                response = model.get_reward(messages=messages,n = cfg.prompt_method.return_num, node_depth=node_depth)
                print(6)
            # printpink(f"{response}")
            if "yes_prob" in response:
                response = response["yes_prob"]
            return response
        except Exception as e:
            printred(f"Errorreward_call: {str(e)}")
            # print(f"The model response was:\n {response}")
            # if attempt == cfg.prompt_method.decompose_question_num-1:
            #     print(f"Failed to get reward after {cfg.prompt_method.return_num} attempts. Error: {str(e)}")
            #     return 0.0

def concat_ost_steps(solution_trace):
    try:
        if type(solution_trace) == list:
            if len(solution_trace) == 0:
                return "", 1
            
            solution_trace_str = ""
            for step_id, step_text in enumerate(solution_trace): # step_id 从0开始的
                solution_trace_str += f"Step {step_id}: " + step_text + "\n"
                
            return solution_trace_str, step_id + 1
            
        else:
            alert(f"Error:concat_ost_steps: solution_trace is not a list!")
            return solution_trace, 1
    except Exception as e:
        printred(f"Error:concat_ost_steps: {str(e)}")
        return "", 1

def ost_step_call(cfg, image_url, model, question, solution_trace, last_ost_step = False):
    try:
        existing_ost_steps, next_ost_step_id = concat_ost_steps(solution_trace)
        
        if last_ost_step: # 达到最大深度需要直接回答了
            io_input = cfg.fewshot_cot_config["prompt_template"].format(
                examples=cfg.fewshot_cot_prompt,
                instruction=question,
            ) + existing_ost_steps + f"Step {next_ost_step_id}:"
        else:
            io_input = cfg.fewshot_ost_config["prompt_template"].format(
                examples=cfg.fewshot_ost_prompt,
                instruction=question,
            ) + existing_ost_steps + f"Step {next_ost_step_id}:"
        
        #printblue(f"{io_input}")
        
        messages = generate_message(cfg , io_input, image_url, assistant_prompt="")
        
        stop_token = [f"Step {next_ost_step_id + 1}:"]

        responses = default_call(cfg, messages, model, "", "step-by-step", n=cfg.prompt_method.tree_width, continue_final_message=True, stop_token=stop_token) # list
        
        responses = [response for response in responses] # qwen2
        
        # printgreen(f"{responses}")
        return responses
    
    except Exception as e:
        printred(f"Error:ost_step_call: {e}")
        return ""

def simple_cot_call(cfg, image_url, model, question, n=1):
    try:
        user_prompt = plan_prompt + question_prefix.format(question=question)
        assistant_prompt = ""
        assistant_prompt += subquestion_prefix.format(step=1, subquestion='').strip()
        messages = generate_message(cfg,user_prompt, image_url, assistant_prompt=assistant_prompt)
        return default_call(cfg, messages, model, "", "simple cot", n=n, continue_final_message=True)
    except Exception as e:
        printred(f"Error:simple_cot_call: {e}")
        return ""
def extract_final_answer(response):
    try:
        final_answer = re.findall(r"Answer \d+: (.*)", response)[-1]
    except:
        print(f"Error: Failed to extract final answer!")
        final_answer = response
    return final_answer

# 抽取answer is 后面的部分
def extract_final_answer_word(response):
    if "answer is:" in response.lower():
        final_answer = response.split("answer is:")[-1].strip(". \n")
    elif "answer is：" in response.lower():
        final_answer = response.split("answer is：")[-1].strip(". \n")
    elif "answer is" in response.lower():
        final_answer = response.split("answer is")[-1].strip(". \n")
    else:
        final_answer = response
        
    return final_answer

def majority_vote(answers):
    try:
        answer = Counter(answers).most_common(1)[0][0]
        index = answers.index(answer)
        return answer, index
    
    except Exception as e:
        printred(f"Error:majority_vote: {e}")
        return "", -1
def extract_ost_step(step):
    try:
        pattern = r"(Step|step|步骤)\s*\d+[:：]\s*"
        result = re.sub(pattern, '', step)
        return result.strip()
    except:
        printred(f"Error:extract_ost_step: {step}")
        result = ""
        return ""

def get_reward_ost_prompt(question,solution_trace,curr_ost_step):
    try:
        user_prompt = ost_reward_prompt + question_prefix.format(question=question) + "\n"
            
        for step_id, step_text in enumerate(solution_trace):
            user_prompt += f"Step {step_id + 1}: " + step_text + "\n"
        
        user_prompt += f"{new_ost_prefix.format(idx=len(solution_trace) + 1, step=curr_ost_step)}\n"
        user_prompt += ost_reward_prefix
        
        return user_prompt
    
    except Exception as e:
        printred(f"Error:get_reward_ost_prompt: {e}")
        return ""

def get_ost_prompt(solution_trace,curr_ost_step):
    try:
        user_prompt = ""
        for step_id, step_text in enumerate(solution_trace):
            user_prompt += f"Step {step_id + 1}: " + step_text + "\n"
        
        user_prompt += f"Step {len(solution_trace) + 1}: {curr_ost_step}\n"
        return user_prompt
    
    except Exception as e:
        printred(f"Error:get_ost_prompt: {e}")
        return ""
def get_rewrite_answer(list_answer):
    try:
        rewrite_answer_list = []
        for answer in list_answer:
            rewrite_answer = answer.split("answer is")
            if rewrite_answer == []:
                rewrite_answer_list.append(answer)
            else:
                rewrite_answer = rewrite_answer[-1].strip()
                if rewrite_answer == "":
                    rewrite_answer_list.append(answer)
                else:
                    if rewrite_answer[-1].strip() in ["?", "！", "?", "！", ".", "。"]:
                        rewrite_answer_list.append(rewrite_answer[:-1])
                    else:
                        rewrite_answer_list.append(rewrite_answer)
        
        if rewrite_answer_list == []:
            return ""
        return majority_vote(rewrite_answer_list)[0]
        
    except Exception as e:
        printred(f"Error:get_rewrite_answer: {e}")
        return ""

def get_best_answer(answer_tree):
    try:
        if len(answer_tree) == 0:
            return ""

        for i,j in answer_tree.items():
            if j == -1e18:
                return i
            
    except Exception as e:
        printred(f"Error:get_best_answer: {e}")
        return ""

def exist_in_answer_tree(answer_tree,answer):
    try:
        if answer in answer_tree:
            return True
        else:
            return False
    except Exception as e:
        printred(f"Error:exist_in_answer_tree: {e}")
        return False

def update_answer_tree(answer_tree,answer,reward):
    try:
        best_answer = get_best_answer(answer_tree)
        if reward:
            answer_tree[best_answer] = answer
            answer_tree[answer] = -1e18
        else:
            answer_tree[answer] = best_answer
        return answer_tree
    
    except Exception as e:
        printred(f"Error:update_answer_tree: {e}")
        return answer_tree
def get_reward_ost(cfg, image_url, model, question,solution_trace,curr_ost_step,answer_tree={}): # 这里的step_id已经是加了 1 的
    try:
        
        logger_path = os.path.join(cfg.output_dir, "logger")
        os.makedirs(logger_path, exist_ok=True)
        
        logger_json = os.path.join(logger_path, "PK_logger.json")
        if os.path.exists(logger_json):
            with open(logger_json, "r") as f:
                PK_data = json.load(f)
        else:
            PK_data = {}
        
        question_id = str(cfg.prompt_method.data_id)
        
        if cfg.prompt_method.answer_verify == 2:
            new_curr_ost_step = curr_ost_step
            # printblue(f"{curr_ost_step}")
            # printgreen(f"{reach_terminal_ost_step(curr_ost_step)}")
            if reach_terminal_ost_step(curr_ost_step):
                answer = extract_final_answer_word(curr_ost_step).strip()
            
                best_answer = get_best_answer(answer_tree) # 获取根节点
                
                if answer.strip() == "" or exist_in_answer_tree(answer_tree,answer):
                    if answer.strip() == "":
                        PK_data[question_id].append(f"answer为空，跳过")
                    elif best_answer == "":
                        PK_data[question_id].append(f"best_answer为空，跳过")
                    else:
                        PK_data[question_id].append(f"answer:{answer} 已存在与answer_tree中，跳过")
                    save_json(logger_json,PK_data)
                    return 0.5 , answer_tree , new_curr_ost_step 
                
                if len(answer_tree) == 0:
                    user_prompt = direct_answer_verify_prefix.format(question=question, answer=answer)
                    reward1 = reward_call(cfg, generate_message(cfg,user_prompt, image_url), model, node_depth=len(solution_trace),is_answer_verify=True)
                    rewrite_answer = answer # 默认初始值
                    if reward1 < 0.3: # 进行纠正
                        # printred(f"当前{curr_ost_step}的奖励为{reward1}，小于0.3，进行纠正")
                        rewrite_answer_prompt = rewrite_answer_prefix.format(question=question, false_answer=answer)
                        list_rewrite_answer = default_call(cfg, generate_message(cfg,rewrite_answer_prompt, image_url), model, "", "step-by-step", n=cfg.prompt_method.return_num, continue_final_message=True, stop_token=None) # list
                        rewrite_answer = get_rewrite_answer(list_rewrite_answer) # 投票
                        # rewrite_answer_step = get_ost_prompt(solution_trace,rewrite_answer) # 整个step过程
                        new_curr_ost_step = "The answer is " + rewrite_answer
                        # printred(f"纠正后的答案为{new_curr_ost_step}")
                        PK_data[question_id].append(f"answer_tree为空-纠正-分数{reward1}，前答案:{answer} ,后:{rewrite_answer}")
                        # PK_data[question_id].append(f"answer_tree为空-纠正-不PK，加入{rewrite_answer}")
                    else:
                        PK_data[question_id].append(f"answer_tree为空-不纠正，加入答案:{rewrite_answer}")
                        
                    save_json(logger_json,PK_data)
                    answer_tree[rewrite_answer] = -1e18
                    
                else:
                    # printgreen(f"当前{curr_ost_step}的奖励为{reward1}，大于等于0.3，进行PK")
                    user_prompt = double_answer_verify_prompt.format(question=question, answer1=answer, answer2=best_answer)
                    reward1 = reward_call(cfg, generate_message(cfg,user_prompt, image_url), model, node_depth=len(solution_trace),is_PK = True , answer1 = answer , answer2 = best_answer)
                    # printpurple(f"PK之后{reward1}")
                    answer_tree = update_answer_tree(answer_tree,answer,reward1)
                    if reward1:
                        PK_data[question_id].append(f"更新answer_tree-更新根节点:{reward1}，原根节点：{best_answer} 现：{answer}")
                    else:
                        PK_data[question_id].append(f"更新answer_tree-不更新根节点:{reward1}，原根节点：{best_answer} 现：{answer}")
                        
                    save_json(logger_json,PK_data)
                    
                return 0.5 , answer_tree , new_curr_ost_step
                
            else:
                # user_prompt = get_reward_ost_prompt(question,solution_trace,curr_ost_step)
                # return reward_call(cfg, generate_message(cfg,user_prompt, image_url), model, node_depth=len(solution_trace)) , answer_tree , new_curr_ost_step
                return 0.5 , answer_tree , new_curr_ost_step

    except Exception as e:
        printred(f"Error:get_reward_ost: {e}")
        return -1, -1, -1

def ost_get_reward(cfg, image_url, model, question, solution_trace, curr_ost_step,answer_tree = {}): # 这里设置ost_step的价值
    try:
        #if reach_terminal_ost_step(curr_ost_step):
        reward_ost,answer_tree,new_curr_ost_step = get_reward_ost(cfg, image_url, model, question, solution_trace, curr_ost_step,answer_tree)
        return reward_ost,answer_tree,new_curr_ost_step
        #else:
           # return 0.5, 0, 0
        
    except Exception as e:
        printred(f"Error:ost_get_reward: {e}")
        return -1, -1, -1
def ost_get_reward_parallel(cfg, image_url, model, question, solution_trace, candidate_ost_steps,answer_tree = {}):
    try:
        candidate_ost_rewards = []
        list_new_curr_ost_steps = []
        # best_answer = get_best_answer(answer_tree)
        answer_tree = answer_tree
        for i, ost_step in enumerate(candidate_ost_steps):
            reward_ost,candidate_answer_tree,new_curr_ost_step = ost_get_reward(cfg, image_url, model, question, solution_trace, ost_step,answer_tree)
            answer_tree = candidate_answer_tree
            list_new_curr_ost_steps.append(new_curr_ost_step)
            candidate_ost_rewards.append({
                "ost_step_reward": reward_ost,
            })
        
        # printred(f"{candidate_ost_rewards}")
        
        return candidate_ost_rewards , answer_tree , list_new_curr_ost_steps
    except Exception as e:
        printred(f"Error:ost_get_reward_parallel: {e}")
        return [], {}
    
def reach_terminal_ost_step(ost_step: str):
    assert ost_step is not None

    return "answer is" in str(ost_step).lower()


def direct_QA(cfg, image_url, model, question):
    try:
        direct_answer_prompt = f"""
            Please answer directly and only output the result. For multiple-choice questions, output the option.Do not generate any other irrelevant content.\n
            **Question**
            {question}\n
            **Answer**
        """
        direct_answer_list = default_call(cfg, generate_message(cfg,direct_answer_prompt, image_url), model, "", "step-by-step", n=cfg.prompt_method.tree_width, continue_final_message=True, stop_token=None) # list
        
        if direct_answer_list == []:
            return [""]
        
        return random.choices(direct_answer_list,k=1)
        # return majority_vote(direct_answer_list)[0]
        
    except Exception as e:
        printred(f"Error:direct_QA: {e}")
        # return -1, -1, -1


