import time
import os
import json
import re
from openai import OpenAI
from pydantic import BaseModel
import base64
from PIL import Image
import io
import math
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from .color_print import printyellow,printred,printgreen,printblue,printpurple,printpink,printwhite,printbold,printcyan

from .prompts import init_system_prompt, plan_prompt, question_prefix, subquestion_prefix, overall_question_prefix, answer_prefix, reward_prompt, new_subquestion_prefix, reward_prefix, reward_prompt_for_answer, reward_prefix_for_answer


def select_deduplicated_indices(
    texts,
    embedding_model=None,
    batch_size=32,
    similarity_threshold=0.9,
):
    if not texts:
        return []
    if embedding_model is None:
        seen = set()
        indices = []
        for index, text in enumerate(texts):
            if text not in seen:
                seen.add(text)
                indices.append(index)
        return indices

    embeddings = embedding_model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings)
    indices = []
    for index, embedding in enumerate(embeddings):
        if not indices:
            indices.append(index)
            continue
        similarities = embeddings[indices] @ embedding
        if float(np.max(similarities)) < similarity_threshold:
            indices.append(index)
    return indices

# 计算问题与问题的相似度 
# def similarity_compute(embedding_model,question, temp_children):
#     def get_detailed_instruct(task_description: str, query: str) -> str:
#         return f'Instruct: {task_description}\nQuery: {query}'

#     # Each query must come with a one-sentence instruction that describes the task
#     task = 'Given a web search query, retrieve relevant passages that answer the query'
#     queries = [
#         get_detailed_instruct(task, question),
#         # get_detailed_instruct(task, '南瓜的家常做法')
#     ]
#     # No need to add instruction for retrieval documents
#     # documents = [
#     #     "you are so smart",
#     #     'Which number is bigger, 4 or 5?',
#     #     'What is biggest number that they could make?',
#     #     'Which two numbers are being added together?',
#     #     'What are the two numbers that will be added together?'
#     # ]
    
#     # print(queries)
#     documents = [child.sub_questions[-1] for child in temp_children]
#     # print(documents)
#     input_texts = queries + documents
    
#     # print(input_texts)
#     # model = SentenceTransformer('/home/jncsnlp4/SSD2/model/multilingual-e5-large-instruct',device='cuda:0')
    
#     embeddings = embedding_model.encode(input_texts, convert_to_tensor=True, normalize_embeddings=True)
#     scores = (embeddings[:1] @ embeddings[1:].T) * 100
    
#     return scores.tolist()[0] # 二维矩阵

def generate_message(cfg,query, image_url, text_only=False, assistant_prompt=None,step=None):
    try:
        is_text_only = bool(getattr(cfg.llm, "text_only", False))
        is_qwen2_vl_family = cfg.llm.name.startswith(("Qwen2-VL-", "Qwen2.5-VL-"))
        if is_text_only:
            messages = [
                {"role": "system", "content": init_system_prompt},
                {"role": "user", "content": query},
            ]
        elif is_qwen2_vl_family:
            if text_only:
                messages=[
                    {"role": "system", "content": init_system_prompt},
                    
                    {
                        "role": "user",
                        "content": query,
                    },
                ]
            else:
                # if step % 2:
                if step:
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
                else:
                    messages=[
                        {"role": "system", "content": init_system_prompt},
                        
                        {
                            "role": "user",
                            "content": {"type": "text", "text": query},
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
            
        if (is_text_only or is_qwen2_vl_family) and assistant_prompt is not None and assistant_prompt != "":
            messages.append({"role": "assistant", "content": assistant_prompt})
    except Exception as e:
        printred(f"generate_message: {str(e)}")
        return None
    # print(messages)
    return messages

def default_call(cfg, messages, model, default_response, task_name="get response", n=1, continue_final_message=False, stop_token=None,
                
                 ):
    for attempt in range(cfg.prompt_method.attempt_num):
        try:
            
            if cfg.prompt_method.is_deduplicate:
                # printblue("我进来了")
                response = model.get_response(messages=messages, json_format=None, n = cfg.prompt_method.deduplicate_multiple, continue_final_message=continue_final_message, stop_token=stop_token)
            
            elif cfg.prompt_method.is_deduplicatev2:
                response = model.get_response(messages=messages, json_format=None, n = cfg.prompt_method.deduplicate_multiple * n, continue_final_message=continue_final_message, stop_token=stop_token)
            else:
                response = model.get_response(messages=messages, json_format=None, n = n, continue_final_message=continue_final_message, stop_token=stop_token)
            # printcyan(f"{response}")
            if "response" in response:
                response = response["response"]
                # cache_hit_ratio = response["cache_hit_ratio"]
            return response
        
        except Exception as e:
            printred(f"Errordefault_call: {str(e)}")
            print(f"The model response was:\n {response}")
            if attempt == cfg.prompt_method.attempt_num-1:
                print(f"Failed to {task_name} after {cfg.prompt_method.return_num} attempts. Error: {str(e)}")
                return default_response
            
def reward_call(cfg, messages, model, node_depth, is_p3pro=False, is_p3=False):
    for attempt in range(cfg.prompt_method.attempt_num):
        try:
            response = model.get_reward(messages=messages,n = cfg.prompt_method.return_num, node_depth=node_depth,is_p3pro = is_p3pro,is_p3 = is_p3)
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
                
        messages = generate_message(cfg,user_prompt, image_url, assistant_prompt=assistant_prompt,step = len(sub_questions)+1)
        
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
        printred(f"Errosimple_cot_call: {str(e)}")
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


_EXPLICIT_ANSWER_MARKER = re.compile(r"\banswer\s+is\b\s*[:：\-]?\s*", re.IGNORECASE)


def _clean_explicit_answer_segment(segment):
    """Return the answer immediately following an ``answer is`` marker."""
    segment = segment.strip()
    if not segment:
        return ""

    # Keep decimal points, but stop before a new prose sentence such as
    # ``42. Therefore ...``.  Most VReST answers place the marker at the end.
    segment = re.split(r"(?<=[.!?])\s+(?=[A-Z])", segment, maxsplit=1)[0]
    return segment.strip(" \t\r\n.,;:：")


def extract_explicit_answer(response):
    """Extract case-insensitive ``answer is`` clauses without guessing.

    The structured return value is intentionally JSON-serializable so it can
    be stored beside every terminal candidate for later auditing.
    """
    raw_response = "" if response is None else str(response)
    markers = list(_EXPLICIT_ANSWER_MARKER.finditer(raw_response))
    if not markers:
        return {
            "raw_response": raw_response,
            "extracted_answer": None,
            "answer_extract_status": "missing_explicit_answer",
            "matches": [],
        }

    matches = []
    saw_empty = False
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(raw_response)
        answer = _clean_explicit_answer_segment(raw_response[marker.end():end])
        if answer:
            matches.append(answer)
        else:
            saw_empty = True

    if not matches:
        status = "empty_after_marker" if saw_empty else "missing_explicit_answer"
        return {
            "raw_response": raw_response,
            "extracted_answer": None,
            "answer_extract_status": status,
            "matches": [],
        }

    normalized = [answer.casefold() for answer in matches]
    if len(set(normalized)) > 1:
        return {
            "raw_response": raw_response,
            "extracted_answer": None,
            "answer_extract_status": "ambiguous_multiple_markers",
            "matches": matches,
        }

    return {
        "raw_response": raw_response,
        "extracted_answer": matches[-1],
        "answer_extract_status": "ok",
        "matches": matches,
    }


def select_explicit_answer_vote(responses):
    """Majority-vote only candidates with an explicit, unambiguous answer."""
    candidates = [extract_explicit_answer(response) for response in responses]
    valid = [
        (index, candidate)
        for index, candidate in enumerate(candidates)
        if candidate["answer_extract_status"] == "ok"
    ]
    excluded_indices = [
        index
        for index, candidate in enumerate(candidates)
        if candidate["answer_extract_status"] != "ok"
    ]

    if not valid:
        return {
            "response": "",
            "raw_response": None,
            "answer_extract_status": "parse_failed",
            "valid_vote_count": 0,
            "excluded_vote_count": len(candidates),
            "excluded_candidate_indices": excluded_indices,
            "candidates": candidates,
        }

    normalized_answers = [candidate["extracted_answer"].casefold() for _, candidate in valid]
    winning_normalized = Counter(normalized_answers).most_common(1)[0][0]
    winning_valid_index = normalized_answers.index(winning_normalized)
    response_index, winner = valid[winning_valid_index]

    return {
        "response": winner["extracted_answer"],
        "raw_response": responses[response_index],
        "answer_extract_status": "ok",
        "valid_vote_count": len(valid),
        "excluded_vote_count": len(excluded_indices),
        "excluded_candidate_indices": excluded_indices,
        "candidates": candidates,
    }


def majority_vote(answers):
    try:
        answer = Counter(answers).most_common(1)[0][0]
        index = answers.index(answer)
        return answer, index
    except Exception as e:
        printred(f"Errormajority_vote: {str(e)}")
        return "", -1

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

# 求最后的子问题的答案价值
def build_reward_answer_messages(cfg, image_url, question, sub_questions, sub_answers, curr_sub_question, curr_sub_answer):
    user_prompt = reward_prompt_for_answer + question_prefix.format(question=question)
    for i, sub_question in enumerate(sub_questions):
        user_prompt += f"{subquestion_prefix.format(step=i+1, subquestion=sub_question)}\n"
        user_prompt += f"{answer_prefix.format(step=i+1, answer=sub_answers[i])}"
    user_prompt += f"{subquestion_prefix.format(step=len(sub_questions)+1, subquestion=curr_sub_question)}\n"
    user_prompt += f"{answer_prefix.format(step=len(sub_questions)+1, answer=curr_sub_answer)}"
    user_prompt += reward_prefix_for_answer
    return generate_message(cfg, user_prompt, image_url, step=len(sub_questions)+1)


def get_reward_answer(cfg, image_url, model, question, sub_questions, sub_answers, curr_sub_question, curr_sub_answer, is_p3pro=False, is_p3=False):
    try:
        messages = build_reward_answer_messages(
            cfg, image_url, question, sub_questions, sub_answers,
            curr_sub_question, curr_sub_answer
        )
        return reward_call(cfg, messages, model, node_depth=len(sub_questions), is_p3pro=is_p3pro, is_p3=is_p3)
    except Exception as e:
        printred(f"Errorget_reward_answer: {str(e)}")
        return ""
# 拼接构建奖励提示内容 求子问题的价值
def build_reward_subquestion_messages(cfg, image_url, question, sub_questions, curr_sub_question):
    user_prompt = reward_prompt + question_prefix.format(question=question) + "\n"
    for i, sub_question in enumerate(sub_questions):
        user_prompt += f"{subquestion_prefix.format(step=i+1, subquestion=sub_question)}\n"
    user_prompt += f"{new_subquestion_prefix.format(step=len(sub_questions)+1, subquestion=curr_sub_question)}"
    user_prompt += f"{reward_prefix}"
    return generate_message(cfg, user_prompt, image_url, step=len(sub_questions)+1)


def get_reward_subquestion(cfg, image_url, model, question, sub_questions, curr_sub_question):
    try:
        messages = build_reward_subquestion_messages(
            cfg, image_url, question, sub_questions, curr_sub_question
        )
        return reward_call(cfg, messages, model, node_depth=len(sub_questions))
    except Exception as e:
        printred(f"Errorget_reward_subquestion: {str(e)}")
        return ""
# 求子问题和子答案的价值
def get_reward(cfg, image_url, model, question, sub_questions, sub_answers, curr_sub_question, curr_sub_answer, is_p3pro=False, is_p3=False):
    try:
        reward_subquestion = get_reward_subquestion(cfg, image_url, model, question, sub_questions, curr_sub_question) # 求子问题的价值
        
        # Outcome Reward Model
        # if is_terminal_question(question, curr_sub_question):
        #     # 子答案的价值
        #     reward_answer = get_reward_answer(cfg, image_url, model, question, sub_questions, sub_answers, curr_sub_question, curr_sub_answer)
        # else:
        #     reward_answer = 0.5
        
        reward_answer = get_reward_answer(cfg, image_url, model, question, sub_questions, sub_answers, curr_sub_question, curr_sub_answer, is_p3pro=is_p3pro, is_p3=is_p3)
        # if is_terminal_question(question, curr_sub_question):
        #     if len(sub_questions) <= 3: # 只进行分解一次就可以回答了 分数设置很低
        #         reward_answer *= 0.5
        
        return reward_subquestion, reward_answer
    
    except Exception as e:
        printred(f"Errorget_reward: {str(e)}")
        return "", ""
# 处理很多很多的子问题和子答案 做expand的时候
def _get_reward_parallel_serial(cfg, image_url, model, question, sub_questions, sub_answers, candidate_sub_questions, candidate_sub_answers, is_p3pro=False, is_p3=False):
    try:
        candidate_sub_questions_rewards = [] # 列表套字典
        for i, sub_question in enumerate(candidate_sub_questions):
            reward_subquestion, reward_answer = get_reward(cfg, image_url, model, question, sub_questions, sub_answers, sub_question, candidate_sub_answers[i], is_p3pro=is_p3pro, is_p3=is_p3)

            base_reward_answer = reward_answer
            conditional_p3pro = (
                getattr(cfg.prompt_method, "use_p3pro_for_low_answer", False)
                and not is_p3pro
                and reward_subquestion
                >= getattr(cfg.prompt_method, "p3pro_question_threshold", 0.7)
                and reward_answer
                < getattr(cfg.prompt_method, "p3pro_answer_threshold", 0.7)
            )
            p3pro_applied = False
            if conditional_p3pro:
                p3pro_reward = get_reward_answer(
                    cfg,
                    image_url,
                    model,
                    question,
                    sub_questions,
                    sub_answers,
                    sub_question,
                    candidate_sub_answers[i],
                    is_p3pro=True,
                    is_p3=False,
                )
                try:
                    p3pro_reward = float(p3pro_reward)
                    if math.isfinite(p3pro_reward):
                        reward_answer = p3pro_reward
                        p3pro_applied = True
                except (TypeError, ValueError):
                    pass

            candidate_sub_questions_rewards.append({
                "sub_question_reward": reward_subquestion,
                "sub_answer_reward": reward_answer,
                "base_sub_answer_reward": base_reward_answer,
                "p3pro_applied": p3pro_applied,
            })
            
        return candidate_sub_questions_rewards
    except Exception as e:
        printred(f"Errorget_reward_parallel: {str(e)}")
        return []


def reward_call_batch(cfg, messages_batch, model, node_depth, is_p3pro=False, is_p3=False):
    if not messages_batch:
        return []

    batch_size = max(
        1,
        int(getattr(cfg.prompt_method, "reward_prompt_batch_size", len(messages_batch))),
    )
    rewards = []
    for start in range(0, len(messages_batch), batch_size):
        chunk = messages_batch[start:start + batch_size]
        response = model.get_reward_batch(
            chunk,
            n=cfg.prompt_method.return_num,
            node_depth=node_depth,
            is_p3pro=is_p3pro,
            is_p3=is_p3,
        )
        if len(response) != len(chunk):
            raise ValueError(
                f"Batched reward count mismatch: expected {len(chunk)}, got {len(response)}"
            )
        rewards.extend(item["yes_prob"] for item in response)
    return rewards


def get_reward_parallel(cfg, image_url, model, question, sub_questions, sub_answers, candidate_sub_questions, candidate_sub_answers, is_p3pro=False, is_p3=False):
    if not hasattr(model, "get_reward_batch"):
        return _get_reward_parallel_serial(
            cfg, image_url, model, question, sub_questions, sub_answers,
            candidate_sub_questions, candidate_sub_answers,
            is_p3pro=is_p3pro, is_p3=is_p3,
        )

    question_messages = [
        build_reward_subquestion_messages(
            cfg, image_url, question, sub_questions, candidate_sub_question
        )
        for candidate_sub_question in candidate_sub_questions
    ]
    answer_messages = [
        build_reward_answer_messages(
            cfg, image_url, question, sub_questions, sub_answers,
            candidate_sub_question, candidate_sub_answer
        )
        for candidate_sub_question, candidate_sub_answer in zip(
            candidate_sub_questions, candidate_sub_answers
        )
    ]

    question_rewards = reward_call_batch(
        cfg, question_messages, model, node_depth=len(sub_questions)
    )
    answer_rewards = reward_call_batch(
        cfg, answer_messages, model, node_depth=len(sub_questions),
        is_p3pro=is_p3pro, is_p3=is_p3,
    )
    base_answer_rewards = list(answer_rewards)
    p3pro_applied = [False] * len(answer_rewards)

    eligible_indices = [
        index
        for index, (question_reward, answer_reward) in enumerate(
            zip(question_rewards, answer_rewards)
        )
        if (
            getattr(cfg.prompt_method, "use_p3pro_for_low_answer", False)
            and not is_p3pro
            and question_reward >= getattr(cfg.prompt_method, "p3pro_question_threshold", 0.7)
            and answer_reward < getattr(cfg.prompt_method, "p3pro_answer_threshold", 0.7)
        )
    ]
    if eligible_indices:
        p3pro_rewards = reward_call_batch(
            cfg,
            [answer_messages[index] for index in eligible_indices],
            model,
            node_depth=len(sub_questions),
            is_p3pro=True,
            is_p3=False,
        )
        for index, p3pro_reward in zip(eligible_indices, p3pro_rewards):
            try:
                p3pro_reward = float(p3pro_reward)
                if math.isfinite(p3pro_reward):
                    answer_rewards[index] = p3pro_reward
                    p3pro_applied[index] = True
            except (TypeError, ValueError):
                pass

    return [
        {
            "sub_question_reward": question_reward,
            "sub_answer_reward": answer_reward,
            "base_sub_answer_reward": base_answer_reward,
            "p3pro_applied": applied,
        }
        for question_reward, answer_reward, base_answer_reward, applied in zip(
            question_rewards, answer_rewards, base_answer_rewards, p3pro_applied
        )
    ]
def is_terminal_question(question, curr_sub_question):
    try:
        if 'Now we can answer' in curr_sub_question:
            return True
        # if curr_sub_question.lower() in question.lower():
        #     return True
        return False
    
    except Exception as e:
        printred(f"Erroris_terminal_question: {str(e)}")
        return False
        
