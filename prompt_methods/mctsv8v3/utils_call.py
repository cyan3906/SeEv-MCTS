import time
import os
import json
import re,sys
from openai import OpenAI
from pydantic import BaseModel
import base64
from PIL import Image
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from .color_print import printyellow,printred,printgreen,printblue,printpurple,printpink,printwhite,printbold,printcyan

from .prompts import (
    init_system_prompt, 
    plan_prompt, question_prefix, 
    subquestion_prefix, 
    overall_question_prefix, 
    answer_prefix, 
    reward_prompt, 
    new_subquestion_prefix, 
    reward_prefix, 
    reward_prompt_for_answer, 
    reward_prefix_for_answer
)

from .answer_verify_prompt import (
    double_existing_reasoning_verify_prompt,
    double_answer_verify_prompt,
    direct_answer_verify_prompt,
    rewrite_answer_prompt,
    existing_reasoning_verify_prompt,
    is_answer_consistent_prompt
)

class Node_Type:
    USER_QUESTION = "USER_QUESTION" # 用户原始问题节点
    # REPHRASED_USER_QUESTION = "REPHRASED_USER_QUESTION" # 重述后的用户问题节点
    DIRECT_ANSWER = "DIRECT_ANSWER" # 直接答案节点
    SUBQUESTION = "SUBQUESTION" # 子问题节点
    # RE_SUBANSWER = "RE_SUBANSWER" # 子问题的答案节点
    # OST_STEP = "OST_STEP" # 步步思考节点


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

def default_call(cfg, messages, model, default_response = "错误了", task_name="get response", n=1, continue_final_message=False, stop_token=None):
    for attempt in range(cfg.prompt_method.attempt_num):
        try:
            response = model.get_response(messages=messages, json_format=None, n=n, continue_final_message=continue_final_message, stop_token=stop_token)
            # printcyan(f"{response}")
            if "response" in response:
                response = response["response"]
            return response
        except Exception as e:
            printred(f"Error default_call: {str(e)}")
            print(f"The model response was:\n {response}")
            if attempt == cfg.prompt_method.attempt_num-1:
                print(f"Failed to {task_name} after {cfg.prompt_method.return_num} attempts. Error: {str(e)}")
                return default_response
            
def reward_call(cfg, messages, model, node_depth,
    is_double_existing_reasoning_verify = False,
    is_double_answer_verify = False,
    is_existing_reasoning_verify = False,
    is_rewrite_answer_verify = False,
    is_direct_answer_verify = False,
    answer1 = None, 
    answer2 = None,
    answer3 = None,
    answer4 = None
):
    
    for attempt in range(cfg.prompt_method.attempt_num):
        try:
            if is_double_existing_reasoning_verify:
                response = model.get_reward(messages=messages,n = cfg.prompt_method.return_num + 1, node_depth=node_depth,is_double_existing_reasoning_verify = is_double_existing_reasoning_verify,answer1 = answer1 , answer2 = answer2,answer3=answer3,answer4=answer4)
            elif is_double_answer_verify:
                response = model.get_reward(messages=messages,n = cfg.prompt_method.return_num + 1, node_depth=node_depth,is_double_answer_verify = is_double_answer_verify,answer1 = answer1 , answer2 = answer2)
            elif is_existing_reasoning_verify:
                response = model.get_reward(messages=messages,n = cfg.prompt_method.return_num + 1, node_depth=node_depth,is_existing_reasoning_verify = is_existing_reasoning_verify,answer1 = answer1,answer2 = answer2)
            elif is_rewrite_answer_verify:
                response = model.get_reward(messages=messages,n = cfg.prompt_method.return_num + 1, node_depth=node_depth,is_rewrite_answer_verify = is_rewrite_answer_verify,answer1 = answer1,answer2 = answer2)
            elif is_direct_answer_verify:
                response = model.get_reward(messages=messages,n = cfg.prompt_method.return_num + 1, node_depth=node_depth,is_direct_answer_verify = is_direct_answer_verify,answer1 = answer1,answer2 = answer2)
            else:
                response = model.get_reward(messages=messages,n = cfg.prompt_method.return_num, node_depth=node_depth)
                
            # printpink(f"{response}")
            if "yes_prob" in response:
                response = response["yes_prob"]
                return response
            
            if "extract_answer" in response:
                extract_answer = response["extract_answer"]
                return response,extract_answer
        except Exception as e:
            printred(f"Errorreward_call: {str(e)}")
            print(f"The model response was:\n {response}")
            if attempt == cfg.prompt_method.decompose_question_num-1:
                print(f"Failed to get reward after {cfg.prompt_method.return_num} attempts. Error: {str(e)}")
                return 0.0


def decompose_question_call(cfg, image_url, model, question, sub_questions, sub_answers, last_sub_question=False):
    try:
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
        
        return responses
    except Exception as e:
        printred(f"Errordecompose_question_call: {str(e)}")
        return []
    
def simple_cot_call(cfg, image_url, model, question, n=1):
    try:
        user_prompt = plan_prompt + question_prefix.format(question=question)
        assistant_prompt = ""
        assistant_prompt += subquestion_prefix.format(step=1, subquestion='').strip()
        messages = generate_message(cfg,user_prompt, image_url, assistant_prompt=assistant_prompt)
        return default_call(cfg, messages, model, "", "simple cot", n=n, continue_final_message=True)
    except Exception as e:
        printred(f"Error simple_cot_call: {str(e)}")
        return []
def extract_final_answer(response):
    try:
        final_answer = re.findall(r"Answer \d+: (.*)", response)[-1]
    except:
        print(f"Error: Failed to extract final answer!")
        final_answer = response
    return final_answer

# 抽取answer is 后面的部分
def extract_final_answer_word(response):
    try:
        if "answer is" in response:
            final_answer = response.split("answer is")[-1].strip(". \n")
        elif "answer is:" in response:
            final_answer = response.split("answer is:")[-1].strip(". \n")
        elif "answer is：" in response:
            final_answer = response.split("answer is：")[-1].strip(". \n")
        else:
            final_answer = response
        return final_answer
    except Exception as e:
        printred(f"Error extract_final_answer_word: {str(e)}")
        return response
def majority_vote(answers):
    try:
        answer = Counter(answers).most_common(1)[0][0]
        index = answers.index(answer)
        return answer, index
    except Exception as e:
        printred(f"Error majority_vote: {str(e)}")
        return "", -1
def extract_sub_questions_and_answers(response, step):
    try:
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
    except Exception as e:
        printred(f"Error extract_sub_questions_and_answers: {str(e)}")
        return "", ""
def get_reward_answer(cfg, image_url, model, question, sub_questions, sub_answers, curr_sub_question, curr_sub_answer):
    try:
        user_prompt = reward_prompt_for_answer + question_prefix.format(question=question)
        for i, sub_question in enumerate(sub_questions):
            user_prompt += f"{subquestion_prefix.format(step=i+1, subquestion=sub_question)}\n"
            user_prompt += f"{answer_prefix.format(step=i+1, answer=sub_answers[i])}"
        user_prompt += f"{subquestion_prefix.format(step=len(sub_questions)+1, subquestion=curr_sub_question)}\n"
        user_prompt += f"{answer_prefix.format(step=len(sub_questions)+1, answer=curr_sub_answer)}"
        user_prompt += reward_prefix_for_answer
        return reward_call(cfg, generate_message(cfg,user_prompt, image_url), model, node_depth=len(sub_questions))
    except Exception as e:
        printred(f"Error get_reward_answer: {str(e)}")
        return -1
# 拼接构建奖励提示内容 求子问题的价值
def get_reward_subquestion(cfg, image_url, model, question, sub_questions, curr_sub_question):
    try:
        user_prompt = reward_prompt + question_prefix.format(question=question) + "\n"
        for i, sub_question in enumerate(sub_questions):
            user_prompt += f"{subquestion_prefix.format(step=i+1, subquestion=sub_question)}\n"
        user_prompt += f"{new_subquestion_prefix.format(step=len(sub_questions)+1, subquestion=curr_sub_question)}"
        user_prompt += f"{reward_prefix}"
        return reward_call(cfg, generate_message(cfg,user_prompt, image_url), model, node_depth=len(sub_questions))
    except Exception as e:
        printred(f"Error get_reward_subquestion: {str(e)}")
        return -1
def get_solution_subquestions_subanswers(sub_questions, sub_answers,is_mask = False):
    try:
        
        solution = ""
        if is_mask:
            if len(sub_questions) >= 2: # 进行掩码
                sub_questions = sub_questions[:-1]
                sub_answers = sub_answers[:-1]
        
        for i, (sub_question, sub_answer) in enumerate(zip(sub_questions, sub_answers)):
            solution += f"{subquestion_prefix.format(step=i+1, subquestion=sub_question)}\n"
            solution += f"{answer_prefix.format(step=i+1, answer=sub_answer)}\n"
        
        return solution
    except Exception as e:
        printred(f"Error get_solution_subquestions_subanswers: {str(e)}")
        return ""
def get_pk_answer_reward(cfg, image_url, model, question, sub_questions, sub_answers, curr_sub_question, curr_sub_answer,answer_tree = []):
    try:
        if len(answer_tree) == 0:
            return 1
        
        win = loss = 0
        for candidate_answer in answer_tree:
            if candidate_answer == curr_sub_answer:
                continue
            user_prompt = double_answer_verify_prompt.format(question=question, answer1=curr_sub_answer, answer2=candidate_answer)
            reward = reward_call(cfg, generate_message(cfg,user_prompt, image_url), model, node_depth=len(sub_questions),is_double_answer_verify=True,answer1=curr_sub_answer,answer2=candidate_answer)
            if reward > 0.5:
                win += 1
            else:
                loss += 1
        
        if win + loss == 0:
            return 1
        
        return win / (win + loss)
    except Exception as e:
        printred(f"Error get_pk_answer_reward: {str(e)}")
        return -1
    
def get_pk_solution_reward(cfg, image_url, model, question, sub_questions, sub_answers, curr_sub_question, curr_sub_answer,solution,solution_tree = {}):
    try:
        if len(solution_tree) == 0:
            return 1
        
        win = loss = 0
        for candidate_answer,candidate_solution in solution_tree.items():
            if candidate_answer == curr_sub_answer:
                continue
            user_prompt = double_existing_reasoning_verify_prompt.format(question=question, answer1=solution, answer2=candidate_solution)
            reward = reward_call(cfg, generate_message(cfg,user_prompt, image_url), model, node_depth=len(sub_questions),is_double_existing_reasoning_verify=True,answer1=solution,answer2=candidate_solution,answer3 = curr_sub_answer,answer4=candidate_answer)
            if reward > 0.5:
                win += 1
            else:
                loss += 1
        
        print(f"win: {win}, loss: {loss}")
        
        if win + loss == 0:
            return 1
        return win / (win + loss)
    
    except Exception as e:
        printred(f"Error get_pk_solution_reward: {str(e)}")
        return -1

def is_answer_consistent(cfg, image_url, model, question, answer1, answer2):
    printgreen("我进来了")
    user_prompt = is_answer_consistent_prompt.format(answer1=answer1, answer2=answer2)
    messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                
                "content": [
                    # {
                    #     "type": "image_url",
                    #     "image_url": "",
                    # },
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
    list_response = default_call(cfg, messages, model,n = cfg.prompt_method.return_num + 1)
    
    win = loss = 0
    
    bool_list = []
    for response in list_response:
        match = re.search(r'"is_consistent"\s*:\s*(true|false)', response)
        if match:
            is_consistent = bool(match.group(1))  # 提取匹配到的值
            bool_list.append(is_consistent)
            if is_consistent:
                win += 1
            else:
                loss += 1
    
    # print(f"list_response: {list_response}")
    printyellow(f"bool_list: {bool_list}")
    if win + loss == 0:
        return 1
    
    return win > (win + loss) / 2

def get_reward(cfg, image_url, model, question, sub_questions, sub_answers, curr_sub_question, curr_sub_answer,answer_tree = [],solution_tree = {}):
    try:
        # logger_path = os.path.join(cfg.output_dir, "logger")
        # os.makedirs(logger_path, exist_ok=True)
        
        # logger_json = os.path.join(logger_path, "PK_logger.json")
        # if os.path.exists(logger_json):
        #     with open(logger_json, "r") as f:
        #         PK_data = json.load(f)
        # else:
        #     PK_data = {}
        
        # question_id = str(cfg.prompt_method.data_id)
        printblue(f"answer_tree: {answer_tree}")
        printpurple(f"solution_tree: {solution_tree}")
        pk_reward = 0
        reward_subquestion = get_reward_subquestion(cfg, image_url, model, question, sub_questions, curr_sub_question) # 求子问题的价值
        reward_answer = get_reward_answer(cfg, image_url, model, question, sub_questions, sub_answers, curr_sub_question, curr_sub_answer)
        final_answer = extract_final_answer_word(curr_sub_answer)
        
        if is_terminal_question(question, curr_sub_question):
            user_prompt = existing_reasoning_verify_prompt.format(question=question, answer=get_solution_subquestions_subanswers(sub_questions, sub_answers,is_mask=True))
            # printpurple(f"{user_prompt}")
            classify_reward = reward_call(cfg, generate_message(cfg,user_prompt, image_url), model, node_depth=len(sub_questions),is_existing_reasoning_verify=True,answer1=final_answer)
            printyellow(f"classify_reward: {classify_reward}")
            # PK_data[question_id].append(f"classify_reward: {classify_reward}")
            if classify_reward <= 0.7:
                # 进行推理答案的PK
                pk_reward = get_pk_answer_reward(cfg, image_url, model, question, sub_questions, sub_answers, curr_sub_question, final_answer,answer_tree)
                printblue(f"answer_pk_reward: {pk_reward}")
                
                # 拓展answer_tree
                is_answer_consistent_flag = True
                for candidate_answer in answer_tree:
                    if is_answer_consistent(cfg, image_url, model, question, final_answer, candidate_answer):
                        is_answer_consistent_flag = False
                        break
                
                if is_answer_consistent_flag:
                    answer_tree.append(final_answer)

            else:
                # 进行推理过程的PK
                solution = get_solution_subquestions_subanswers(sub_questions+[curr_sub_question], sub_answers+[curr_sub_answer])
                pk_reward = get_pk_solution_reward(cfg, image_url, model, question, sub_questions, sub_answers, curr_sub_question, final_answer,solution,solution_tree)
                printpurple(f"solution_pk_reward: {pk_reward}")
                
                # 拓展solution_tree
                is_answer_consistent_flag = True
                for candidate_answer in solution_tree:
                    if is_answer_consistent(cfg, image_url, model, question, final_answer, candidate_answer):
                        is_answer_consistent_flag = False
                        break
                
                if is_answer_consistent_flag:
                    solution_tree[final_answer] = solution
                
        
        return reward_subquestion, reward_answer , pk_reward , answer_tree , solution_tree
    
    except Exception as e:
        printred(f"Error utils_call-get_reward: {str(e)}!!!!")
        return -1
    
def get_reward_parallel(cfg, image_url, model, question, sub_questions, sub_answers, candidate_sub_questions, candidate_sub_answers,answer_tree = [],solution_tree = {}):
    try:
        candidate_sub_questions_rewards = [] # 列表套字典
        # list_new_question_answer = []
        # answer_tree = answer_tree
        # solution_tree = solution_tree
        for i, sub_question in enumerate(candidate_sub_questions):
            reward_subquestion, reward_answer,reward_pk,answer_tree,solution_tree = get_reward(cfg, image_url, model, question, sub_questions, sub_answers, sub_question, candidate_sub_answers[i],answer_tree,solution_tree)
            answer_tree = answer_tree
            solution_tree = solution_tree
            # list_new_question_answer.append(new_question_answer)
            candidate_sub_questions_rewards.append({
                "sub_question_reward": reward_subquestion,
                "sub_answer_reward": reward_answer,
                "pk_reward": reward_pk,
            })
            
        return candidate_sub_questions_rewards,answer_tree,solution_tree
    except Exception as e:
        printred(f"Error get_reward_parallel: {str(e)}")
        return -1
def is_terminal_question(question, curr_sub_question):
    try:
        if 'Now we can answer' in curr_sub_question:
            return True
        # if curr_sub_question.lower() in question.lower():
        #     return True
        return False
    except Exception as e:
        printred(f"Error is_terminal_question: {str(e)}")
        return -1
