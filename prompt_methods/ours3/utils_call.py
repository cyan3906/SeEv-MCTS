import time
import os
import json
import re
from openai import OpenAI
from pydantic import BaseModel
import base64
from PIL import Image
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from .color_print import printyellow,printred,printgreen,printblue,printpurple,printpink,printwhite,printbold,printcyan

from .prompts import init_system_prompt, plan_prompt, question_prefix, subquestion_prefix, overall_question_prefix, answer_prefix, reward_prompt, new_subquestion_prefix, reward_prefix, reward_prompt_for_answer, reward_prefix_for_answer
from .prompts import new_ost_prefix, ost_reward_prompt, ost_reward_prefix
from enum import Enum, unique
from numpy import unique


class Node_Type:
    USER_QUESTION = "USER_QUESTION" # 用户原始问题节点
    # REPHRASED_USER_QUESTION = "REPHRASED_USER_QUESTION" # 重述后的用户问题节点
    # DIRECT_ANSWER = "DIRECT_ANSWER" # 直接答案节点
    SUBQUESTION = "SUBQUESTION" # 子问题节点
    # RE_SUBANSWER = "RE_SUBANSWER" # 子问题的答案节点
    OST_STEP = "OST_STEP" # 步步思考节点
    # OST_SUBANSWER = "OST_SUBANSWER" # 步步思考与子问题答案结合的节点
    
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
            if attempt == cfg.prompt_method.attempt_num-1:
                print(f"Failed to {task_name} after {cfg.prompt_method.return_num} attempts. Error: {str(e)}")
                return default_response
            
def reward_call(cfg, messages, model, node_depth):
    for attempt in range(cfg.prompt_method.attempt_num):
        try:
            response = model.get_reward(messages=messages,n = cfg.prompt_method.return_num, node_depth=node_depth)
            # printred("9999999999999999")
            # printpink(f"{response}")
            if "yes_prob" in response:
                response = response["yes_prob"]
            return response
        except Exception as e:
            printred(f"Errorreward_call: {str(e)}")
            print(f"The model response was:\n {response}")
            if attempt == cfg.prompt_method.decompose_question_num-1:
                print(f"Failed to get reward after {cfg.prompt_method.return_num} attempts. Error: {str(e)}")
                return 0.0

def decompose_question_call(cfg, image_url, model, question, sub_questions, sub_answers, last_sub_question=False):
    user_prompt = plan_prompt + question_prefix.format(question=question) # 构造题目
    stop_token = subquestion_prefix.format(step=len(sub_questions)+2, subquestion='').strip() # 结束的前缀
    assistant_prompt = ""
    for i, sub_question in enumerate(sub_questions):
        assistant_prompt += f"{subquestion_prefix.format(step=i+1, subquestion=sub_question)}\n"
        assistant_prompt += f"{answer_prefix.format(step=i+1, answer=sub_answers[i])}\n\n"
    
    if cfg.llm.name == "Qwen2-VL-7B-Instruct":
        if last_sub_question:
            assistant_prompt += overall_question_prefix.format(step=len(sub_questions)+1, overall_question='') # 加上结束的前缀
        else:
            assistant_prompt += subquestion_prefix.format(step=len(sub_questions)+1, subquestion='').strip() # 加上新问题的前缀
    else:
        if last_sub_question:
            assistant_prompt += overall_question_prefix.format(step=len(sub_questions)+1, overall_question='') # 加上结束的前缀
        else:
            assistant_prompt += subquestion_prefix.format(step=len(sub_questions)+1, subquestion='').strip() # 加上新问题的前缀
            
    messages = generate_message(cfg,user_prompt, image_url, assistant_prompt=assistant_prompt)
    responses = default_call(cfg, messages, model, "", "decompose question", n=cfg.prompt_method.tree_width, continue_final_message=True, stop_token=stop_token) # list
    # printcyan(f"{assistant_prompt}")
    # printgreen(f"{messages}")
    # printblue(f"{responses}")
    
    responses = [assistant_prompt + response for response in responses] # qwen2
    # printyellow(f"{responses}")
    
    return responses

def concat_ost_steps(solution_trace):
    """
    solution_trace = {
        0:{
            "1": "dsdsd",
        },
    }
    """
    # n = input()
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

def ost_step_call(cfg, image_url, model, question , sub_answers, solution_trace, last_ost_step = False):
    
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

    messages = generate_message(cfg , io_input, image_url, assistant_prompt="")
    stop_token = [f"Step {next_ost_step_id + 1}:","\n\n","\n"]
    responses = default_call(cfg, messages, model, "", "step-by-step", n=cfg.prompt_method.tree_width, continue_final_message=True, stop_token=stop_token) # list
    
    responses = [response for response in responses] # qwen2
    
    # printgreen(f"{responses}")
    return responses
            
def concat_subqs_subas_as_ost_steps(cfg, question, sub_answers):
    subqs_subas_as_ost_steps_str = "" # 返回输出的内容
    
    if len(sub_answers) == 0:
        return "", 1
    
    for step_id , answer in enumerate(sub_answers):
        match = re.search(r"(.+?\.) The answer is", answer) # 抽取答案前面的内容
        if match:
            step_text = match.group(1).strip()
        else: # 答案前面啥也没有 保底 正确的格式
            step_text = answer.strip()
        subqs_subas_as_ost_steps_str += f"Step {step_id}: " + step_text + "\n"
        
    return subqs_subas_as_ost_steps_str, step_id + 1

def subanswers_ost_step_call(cfg, image_url, model, question, sub_answers, last_ost_step = False):
    existing_ost_steps, next_ost_step_id = concat_subqs_subas_as_ost_steps(cfg, question, sub_answers)
    
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
    
    messages = generate_message(cfg , io_input, image_url, assistant_prompt="")
    stop_token = [f"Step {next_ost_step_id + 1}:","\n\n","\n"]
    responses = default_call(cfg, messages, model, "", "step-by-step", n=cfg.prompt_method.tree_width, continue_final_message=True, stop_token=stop_token) # list
    
    responses = [response for response in responses] # qwen2
    
    return responses
    
def simple_cot_call(cfg, image_url, model, question, n=1):
    user_prompt = plan_prompt + question_prefix.format(question=question)
    assistant_prompt = ""
    assistant_prompt += subquestion_prefix.format(step=1, subquestion='').strip()
    messages = generate_message(cfg,user_prompt, image_url, assistant_prompt=assistant_prompt)
    return default_call(cfg, messages, model, "", "simple cot", n=n, continue_final_message=True)

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
    answer = Counter(answers).most_common(1)[0][0]
    index = answers.index(answer)
    return answer, index

def extract_sub_questions_and_answers(response, step):
    sub_question_pattern = subquestion_prefix.format(step=step, subquestion="").strip()
    sub_answer_pattern = answer_prefix.format(step=step, answer="").strip()
    # printcyan(f"{sub_question_pattern}")
    # printblue(f"{sub_answer_pattern}")
    # 找到pattern后的那个subquestion
    response = response.strip()
    # printyellow(f"{response}")
    try:
        # printgreen(f"{response}")
        sub_question = response.split(sub_question_pattern)[1].split("\n")[0].strip()
        sub_answer = response.split(sub_question_pattern)[1].split("\n")[-1].strip()
        # 将"Answer %d:"替换为空
        sub_answer = re.sub(r"Answer \d+:", "", sub_answer).strip()
    except:
        printred(f"Errorextract_sub_questions_and_answers: {response}")
        sub_question, sub_answer = "", ""
        
    return sub_question, sub_answer

def extract_ost_step(step):
    try:
        pattern = r"(Step|step|步骤)\s*\d+[:：]\s*"
        result = re.sub(pattern, '', step)
        return result.strip()
    except:
        printred(f"Error:extract_ost_step: {step}")
        result = ""
        return ""
# 求最后的子问题的答案价值

# 拼接构建奖励提示内容 求子问题的价值
def get_reward_ost(cfg, image_url, model, question, sub_answers, solution_trace, curr_ost_step, is_answers_gen = False): # 这里的steo_id已经是加了 1 的
    user_prompt = ost_reward_prompt + question_prefix.format(question=question) + "\n"
    if is_answers_gen:
        for step_id, step_text in enumerate(sub_answers):
            user_prompt += f"Step {step_id + 1}: " + step_text + "\n"
    else:
        for step_id, step_text in enumerate(solution_trace):
            user_prompt += f"Step {step_id + 1}: " + step_text + "\n"
    user_prompt += f"{new_ost_prefix.format(idx=len(solution_trace) + 1, step=curr_ost_step)}\n"
    user_prompt += ost_reward_prefix
    return reward_call(cfg, generate_message(cfg,user_prompt, image_url), model, node_depth=len(solution_trace))

def ost_get_reward(cfg, image_url, model, question, sub_answers, solution_trace, curr_ost_step , is_answers_gen = False): # 这里设置ost_step的价值
    reward_ost = get_reward_ost(cfg, image_url, model, question, sub_answers, solution_trace, curr_ost_step , is_answers_gen)
    return reward_ost

def ost_get_reward_parallel(cfg, image_url, model, question, sub_answers, solution_trace, candidate_ost_steps , is_answers_gen = False):
    candidate_ost_rewards = []
    for i, ost_step in enumerate(candidate_ost_steps):
        reward_ost = ost_get_reward(cfg, image_url, model, question, sub_answers, solution_trace, ost_step , is_answers_gen)

        candidate_ost_rewards.append({
            "ost_step_reward": reward_ost
        })
    
    return candidate_ost_rewards
# 处理很多很多的子问题和子答案 做expand的时候

def is_terminal_question(question, curr_sub_question):
    assert curr_sub_question is not None
    
    return 'Now we can answer' in curr_sub_question.strip()

def reach_terminal_ost_step(ost_step: str):
    assert ost_step is not None

    return "answer is" in ost_step.lower()





