

from .prompts import init_system_prompt, plan_prompt, question_prefix, subquestion_prefix, overall_question_prefix, answer_prefix
from .color_print import *
import re
from collections import Counter

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

def extract_final_answer(response):
    try:
        final_answer = re.findall(r"Answer \d+: (.*)", response)[-1]
    except:
        print(f"Error: Failed to extract final answer!")
        final_answer = response
    return final_answer

def extract_final_answer_word(response):
    try:
        if "answer is" in response:
            final_answer = response.split("answer is")[-1].strip(". \n")
        elif "answer is：" in response:
            final_answer = response.split("answer is")[-1].strip(". \n")
        elif "answer is:" in response:
            final_answer = response.split("answer is")[-1].strip(". \n")
        else:
            final_answer = response
        return final_answer
    except Exception as e:
        printred(f"Erronextract_final_answer_word: {str(e)}")
        return ""
def majority_vote(answers):
    try:
        answer = Counter(answers).most_common(1)[0][0]
        index = answers.index(answer)
        return answer, index
    except Exception as e:
        printred(f"Errormajority_vote: {str(e)}")
        return "", -1

def decompose_question_call(cfg, image_url, model, question, sub_questions, sub_answers, last_sub_question):
    # 这里将返回一组候选子问题路径。根据实际需求实现。
    try:
        user_prompt = plan_prompt + question_prefix.format(question=question) # 构造题目
        stop_token = subquestion_prefix.format(step=len(sub_questions)+2, subquestion='').strip() # 结束的前缀
        assistant_prompt = ""
        for i, sub_question in enumerate(sub_questions):
            assistant_prompt += f"{subquestion_prefix.format(step=i+1, subquestion=sub_question)}\n"
            assistant_prompt += f"{answer_prefix.format(step=i+1, answer=sub_answers[i])}\n\n"
        
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
        printred(f"Errorextrct_sub_questions_and_answers: {str(e)}")
        return "", ""
    
def extract_sub_questions_and_answers(response, step):
    sub_question_pattern = subquestion_prefix.format(step=step, subquestion="").strip()
    sub_answer_pattern = answer_prefix.format(step=step, answer="").strip()
    
    # 清理响应
    response = response.strip()
    
    try:
        # 尝试提取子问题和答案
        sub_question = response.split(sub_question_pattern)[1].split("\n")[0].strip()
        sub_answer = response.split(sub_question_pattern)[1].split("\n")[-1].strip()
        
        # 进一步清理答案
        sub_answer = re.sub(r"Answer \d+:", "", sub_answer).strip()
    
    except Exception as e:
        # 如果捕获到异常，打印错误信息并设置空字符串
        print(f"Error in extract_sub_questions_and_answers: {str(e)}")
        sub_question, sub_answer = "", ""
    
    # 返回提取的子问题和子答案
    return sub_question, sub_answer



def is_terminal_question(question, curr_sub_question):
    if 'Now we can answer' in curr_sub_question:
        return True
    # if curr_sub_question.lower() in question.lower():
    #     return True
    return False