import os
import json
import time
import re
from tqdm import tqdm, trange

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from tools.color_print import *

from .prompts.scoring_inst import SCORING_INST, SYSTEM_SCORING_INST, USER_SCORING_INST,system_judge_numeric_inst,judge_numeric_inst
from pydantic import BaseModel

class Extract_Answer(BaseModel):
    extracted_answer: str
    score: int

def get_majority_answer(answers):
    # print(answers)
    if "score" in answers[0]:
        answers_extracted = [answer["extracted_answer"] for answer in answers]
        answers_score = [answer["score"] for answer in answers]
        # print(answers_extracted, answers_score)
        majority_score = Counter(answers_score).most_common(1)[0][0]
        # print(answers_extracted, answers_score, majority_score)
        remained_answers = [answers_extracted[i] for i in range(len(answers_extracted)) if answers_score[i] == majority_score]
        majority_answer = Counter(remained_answers).most_common(1)[0][0] # 对可能有重复的答案做一个取一个
        return majority_answer, majority_score
    else:
        answers_extracted = [answer["extracted_answer"] for answer in answers]
        majority_answer = Counter(answers_extracted).most_common(1)[0][0] # 对可能有重复的答案做一个取一个
        return majority_answer, None
    
def ext_yes_or_no(question, ground_truth, model_response, model, is_use_check=False):
    
    if is_use_check:
        # 在评估前先进行判断是否为数值型答案
        judge_prompt = judge_numeric_inst.format(ground_truth=ground_truth, model_response=model_response)
        judge_messages = [
            {"role": "system", "content": system_judge_numeric_inst}, 
            {"role": "user", "content": judge_prompt}
        ]
        
        judge_response = model.get_response(judge_messages, json_format=Extract_Answer)["response"]
        yes_or_no = Counter(responses).most_common(1)[0][0]
        if yes_or_no == "yes":
            if ground_truth.lower().strip() == model_response.lower().strip():
                return {"extracted_answer": model_response, "score": True}, []
            else:
                return {"extracted_answer": model_response, "score": False}, []
            
    prompt = USER_SCORING_INST.format(question=question, ground_truth=ground_truth, model_response=model_response)
    # print(prompt)
    messages = [
        {"role": "user", "content": prompt}
    ]
    max_patience = 10
    patience = 0
    while patience < max_patience:
        try:
            # 这里修改了将原来的get_yes_or_no替换为get_response
            # responses = model.get_yes_or_no(messages)["response"]
            
            responses = model.get_response(messages, json_format=Extract_Answer, is_evaluation=True)["response"] 
            # print(responses)
            # input("点击输入")
            if isinstance(responses, list):
                yes_or_no = Counter(responses).most_common(1)[0][0]
                # print(f"\033[93m{yes_or_no}\033[0m")
                response_json = {"extracted_answer": yes_or_no, "score": "yes" in yes_or_no.lower().strip()}
            else:
                response_json = {"extracted_answer": responses, "score": "yes" in responses.lower().strip()}
            break
        except Exception as e:
            print(f"Error: {str(e)}")
        patience += 1
    if patience == max_patience:
        print(f"Failed to get response for prompt: {prompt}")
        response_json = {"extracted_answer": "Extract Error!", "score": False}
        responses = [response_json]
    return response_json, responses

def ext_ans_and_score(question, ground_truth, model_response, model , choices = None):
    
    # prompt = SCORING_INST.format(question=question, ground_truth=ground_truth, model_response=model_response)
    prompt = SCORING_INST.format(question=question, model_response=model_response)
    
    messages = [
        {"role": "system", "content": SYSTEM_SCORING_INST}, 
        {"role": "user", "content": prompt}
    ]
    max_patience = 10
    patience = 0
    while patience < max_patience:
        try:
            # print(f"\033[94m{messages}\033[0m")
            ori_responses = model.get_response(messages, json_format=Extract_Answer)["response"] # 调用大模型
            print(f"\033[92m{ori_responses}\033[0m")
            responses = []
            if isinstance(ori_responses, list):
                for response in ori_responses:
                    if type(response) == str:
                        response = response.replace("\\", "/")
                        responses.append(json.loads(str(response)))
                    else:
                        responses.append(response)
                        
                ori_responses = responses
                # print(responses)
                # 将所有选项转为具体答案
                # process_responses = []
                # if choices:
                    
                #     for res in responses:
                #         if "A" in res or "B" in res or "C" in res or "D" in res or "E" in res:
                #             # 提取所有字母（a-z、A-Z），返回列表后拼接
                #             result = ''.join(re.findall(r'[a-zA-Z]', res))
                #             if result in choices:
                #                 process_responses.append(ground_truth)
                # if choices:
                #     responses = process_responses
                
                # print(ori_responses)
                extracted_answer, score = get_majority_answer(ori_responses)
                # print(extracted_answer, score)
                
                if score == None:
                    response_json = {"extracted_answer": extracted_answer, "score": None}
                else:
                    response_json = {"extracted_answer": extracted_answer, "score": score == 1}
            else:
                if type(ori_responses) == str:
                    ori_responses = json.loads(ori_responses)
                else:
                    response_json = ori_responses
                response_json["score"] = response_json["score"] == 1
                
            break
        except Exception as e:
            print(f"\033[91mError: {str(e)}\033[0m")
        patience += 1
        
    if patience == max_patience:
        print(f"Failed to get response for prompt: {prompt}")
        response_json = {"extracted_answer": "Extract Error!", "score": False}
        ori_responses = [response_json]
        
    return response_json, ori_responses

def ext_ans_and_score_with_llm(cfg, question, ground_truth, model_response, model , choices = None,is_get_score = False,is_use_check=False):
    if cfg.extract_method == "yes_or_no" or is_get_score:
        return ext_yes_or_no(question, ground_truth, model_response, model, is_use_check = is_use_check)
    else:
        return ext_ans_and_score(question, ground_truth, model_response, model , choices)
    
def ext_ans_charxiv(cfg, model):
    data_cfg = cfg.data

    # input_file
    input_file = os.path.join(data_cfg.data_dir, f"{data_cfg.mode}_{data_cfg.split}.json")
    print(input_file)
    # print(f"Reading {input_file}...")
    with open(input_file) as f:
        data = json.load(f)
    
    resp_file = os.path.join(cfg.output_dir, 
            f'gen-{data_cfg.mode}_{data_cfg.split}.json')

    print(resp_file)
    # output file
    output_dir = cfg.extract_result_dir
    os.makedirs(output_dir, exist_ok=True)
    if cfg.ext_llm.name == "qwen3":
        if cfg.ext_llm.enable_thinking:
            output_file = os.path.join(output_dir, f'think-scores-{data_cfg.mode}_{data_cfg.split}.json')
        else:
            output_file = os.path.join(output_dir, f'nothink-scores-{data_cfg.mode}_{data_cfg.split}.json')
    else:
        output_file = os.path.join(output_dir, f'scores-{data_cfg.mode}_{data_cfg.split}.json')
    print(output_file)
    data, response = json.load(open(input_file)), json.load(open(resp_file))
    
    mode = 'descriptive' if 'descriptive' in resp_file.split('-')[-1] else 'reasoning'

    complete_ids = []
    saved_queries = {}
    if os.path.exists(output_file):
        print(f"{output_file} exists. Skipping generation.")
        with open(output_file) as f:
            queries = json.load(f)
        
        complete_ids = [k for k in queries if "extracted_answer" in queries[k] and queries[k]["extracted_answer"] != "Extract Error!"]
        print(f"{len(complete_ids)} problems skipped.")
        saved_queries = queries
        for i,j in saved_queries.items():
            if "figure_id" in j:
                saved_queries[i]["figure_id"] = str(j["figure_id"])
    
    
    if mode == 'descriptive':
        pass
    
    elif mode == 'reasoning':
        from data.CharXiv.src.reasoning_utils import build_reasoning_grading_queries, get_reasoning_result_gpt
        # dict of figure_id -> {figure_id, grading_query}
        queries = build_reasoning_grading_queries(data, response) 
        
        # merge
        for k in queries:
            # print(k, queries[k])
            # input("点击输入")
            if k in saved_queries and k in complete_ids:
                queries[k] = saved_queries[k]

        # for i, j in queries.items():
        #     print(i, j)
        #     input("请点击输入")
        # print(next(iter(queries)), next(iter(queries.values())))
        # for i,j in queries.items():
        #     print(j)
        #     input("点击继续")
        
        # 及时展示结果
        global acc
        acc = 0
        
        # 写成多线程
        lock = threading.Lock()

        def process_query(figure_id):
            
            global acc
            figure_id = str(figure_id)
            
            if figure_id in complete_ids:
                return None
            
            # print(queries)
            # input("点击继续")
            query = queries[figure_id]
            
            # print(query)
            if hasattr(cfg, "extract_field") and 'decision' in query:
                if cfg.extract_field == "max_mean_terminal":
                    model_response = query['decision']['memory']['max_mean_terminal']['sub_answers'][-1]
                elif cfg.extract_field == "max_terminal":
                    model_response = query['decision']['memory']['max_terminal']['sub_answers'][-1]
                else:
                    model_response = query['response']
            else:
                model_response = query['response']
            response, responses = ext_ans_and_score_with_llm(cfg, data[figure_id]['query'], data[figure_id]['answer'], model_response, model) # 这里的query是 question
            # print(response)
            # 这里的response有抽取的结果
            second_score = False
            if response['score'] == None:
                # 说明采用了二次抽取评估答案
                response_score, _ = ext_ans_and_score_with_llm(cfg, data[figure_id]['query'], data[figure_id]['answer'], response['extracted_answer'], model, is_get_score=True) 
                if response_score['score']:
                    second_score = True
                    acc += 1
            else:
                if response['score']:
                    acc += 1
            printyellow(f"当前准确率：{acc} / 1000 = {(acc / 1000):.2%}")
            with lock:
                queries[figure_id]["answer"] = data[figure_id]['answer']
                queries[figure_id]['extracted_answer'] = response['extracted_answer']
                if response['score'] == None: # 二次评估
                    queries[figure_id]['llm_score'] = second_score
                else:
                    queries[figure_id]['llm_score'] = response['score']
                queries[figure_id]['extracted_votes'] = responses
                queries[figure_id].pop('grading_query')
            return figure_id
        
        def save_results():
            try:
                print(f"Saving results to {output_file}...")
                with lock:
                    with open(output_file, "w+") as f:
                        json.dump(queries, f, indent=4)
                print(f"Results saved.")
            except Exception as e:
                print(e)
                print(f"Error in saving {output_file}")   

        # print(response["0"])
        # 执行程序
        for pid in tqdm(response.keys(),desc="Extracting answers"):
            if "response" not in response[pid]:
                continue
            
            process_query(pid)
            save_results()
            # return
            
        return 1
        
        # with ThreadPoolExecutor(max_workers=cfg.worker_num) as executor:
        #     futures = {executor.submit(process_query, figure_id): figure_id for figure_id in queries}
        #     for i, future in enumerate(tqdm(as_completed(futures), total=len(queries))):
        #         print(f"Completed {i} queries.")
        #         if future.result() is not None and i % 10 == 0:
        #             save_results()

        # save_results()

    else:
        raise ValueError("Mode not supported")

from data.MathVista.evaluation.extract_answer import extract_answer
def ext_ans_mathvista(cfg, model):
    data_cfg = cfg.data

    # input_file
    result_file = os.path.join(cfg.output_dir, f'generation.json')
    print(f"Reading {result_file}...")
    with open(result_file) as f:
        results = json.load(f)
    
    output_dir = cfg.extract_result_dir
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f'extraction.json')
    if os.path.exists(output_file):
        print(f"{output_file} exists. Skipping extraction.")
        saved_results = json.load(open(output_file))
        results.update(saved_results)
    
    # full pids
    full_pids = list(results.keys())
    print("Number of testing problems:", len(full_pids))

    test_pids = []
    for pid in full_pids:
        if "extracted_answer" not in results[pid] or results[pid]['extracted_answer'] == 'Extract Error!': # 出错了
            test_pids.append(pid)
            
        # if "response" in results[pid] and 'extracted_answer' not in results[pid] and results[pid]["response"] != "Failed to generate response!":
        #     test_pids.append(pid)
    
    test_num = len(test_pids)
    print("Number of problems to run:", test_num)
    
    # 同步显示预测对的数量
    global acc
    acc = 0
    
    # 写成多线程
    lock = threading.Lock()

    def process_query(pid):
        global acc
        # print(pid)
        if hasattr(cfg, "extract_field") and 'decision' in results[pid]:
            if cfg.extract_field == "max_mean_terminal":
                model_response = results[pid]['decision']['memory']['max_mean_terminal']['sub_answers'][-1]
            elif cfg.extract_field == "max_terminal":
                model_response = results[pid]['decision']['memory']['max_terminal']['sub_answers'][-1]
            else:
                model_response = results[pid]['response']
        else:
            model_response = results[pid]['response']
        
        # 自己写一个补充的
        choices = [results[pid]['answer']]
        if results[pid]["question_type"] == "multiple_choice":
            all_choice = results[pid]['choices']
            for z,j in enumerate(all_choice):
                if j == results[pid]['answer']:
                    choices.append(str(65 + z))

        # print(results[pid]['answer'])
        # print(model_response)
        response, responses = ext_ans_and_score_with_llm(cfg, results[pid]['query'], results[pid]['answer'], model_response, model)
        second_score = False
        if response["score"] == None:
            response_score, _ = ext_ans_and_score_with_llm(cfg, results[pid]['query'], results[pid]['answer'], model_response, model,is_get_score=True)
            if response_score["score"]:
                acc += 1
                second_score = True
        else:
            if response["score"]:
                acc += 1
        printyellow(f"当前准确率：{acc} / 1000 = {(acc / 1000):.2%}")
        # print(response)
        with lock:
            results[pid]['extracted_answer'] = response['extracted_answer']
            if response['score'] == None: # 二次评估
                results[pid]['llm_score'] = second_score
            else:
                results[pid]['llm_score'] = response['score']
            
            results[pid]['extracted_votes'] = responses
        return pid
    
    def save_results():
        try:
            print(f"Saving results to {output_file}...")
            with lock:
                with open(output_file, "w+") as f:
                    json.dump(results, f, indent=4)
            print(f"Results saved.")
        except Exception as e:
            print(e)
            print(f"Error in saving {output_file}")

    
    for pid in tqdm(test_pids,desc="Extracting answers"):
        if "response" not in results[pid]:
            continue
        
        process_query(pid)
        save_results()
        # return
        
    return 1
    
    # with ThreadPoolExecutor(max_workers=cfg.worker_num) as executor:
    #     futures = {executor.submit(process_query, pid): pid for pid in test_pids}
    #     for i, future in enumerate(tqdm(as_completed(futures), total=len(test_pids))):
    #         print(f"Completed {i} queries.")
    #         if future.result() is not None and i % 10 == 0:
    #             save_results()

    # save_results()

from .mathvision_utils import find_math_answer, is_number, is_equal
def ext_ans_mathvision(cfg, model):
    data_cfg = cfg.data

    # input_file
    result_file = os.path.join(cfg.output_dir, f'generation.json')
    print(f"Reading {result_file}...")
    with open(result_file) as f:
        results = json.load(f)
    
    output_dir = cfg.extract_result_dir
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'scores.json')
    
    if os.path.exists(output_file):
        print(f"{output_file} exists. Skipping extraction.")
        saved_results = json.load(open(output_file))
        results.update(saved_results) # 更新了
        
    # full pids
    full_pids = list(results.keys())
    print("Number of testing problems:", len(full_pids))

    test_pids = [] 
    for pid in full_pids:
        if "extracted_answer" not in results[pid] or results[pid]['extracted_answer'] == 'Extract Error!': # 出错了
            test_pids.append(pid)
    
    test_num = len(test_pids)
    print("Number of problems to run:", test_num) # 才是关键
    
    # return 1

    # 多线程
    lock = threading.Lock()

    global acc
    acc = 0
    
    def process_query(pid):
        
        start_time = time.time()
        
        raw_exampe = results[pid]
        gt_answer = str(raw_exampe['answer'])
        if len(raw_exampe['options']) > 0:
            gt_answer_value = raw_exampe['options'][ord(gt_answer)-ord('A')] # 就是获取哪个的标准答案 用字母的下标来
        else:
            gt_answer_value = ''
        
        if hasattr(cfg, "extract_field") and 'decision' in results[pid]:
            if cfg.extract_field == "max_mean_terminal":
                model_response = results[pid]['decision']['memory']['max_mean_terminal']['sub_answers'][-1]
            elif cfg.extract_field == "max_terminal":
                model_response = results[pid]['decision']['memory']['max_terminal']['sub_answers'][-1]
            else:
                model_response = results[pid]['response']
        else:
            model_response = results[pid]['response']

        # 开始抽取 最终结果就一个 正确答案和一个对还是错的判断 response就是答案和分数的json格式,responses没有处理后的
        response, responses = ext_ans_and_score_with_llm(cfg, results[pid]['question'], results[pid]['answer'], model_response, model)
        # print(f"{pid}: {response}")
        # print(f"{pid}: {responses}")
        
        
        global acc
        # 二次评估
        second_score = False
        if response['score'] == None: # 分离式评估
            response_score, _ = ext_ans_and_score_with_llm(cfg, results[pid]['question'], results[pid]['answer'], model_response, model,is_get_score=True)
            
            if response_score["score"]:
                acc += 1
                second_score = True
                
        else:
            if response["score"]:
                acc += 1
                
        printyellow(f"当前准确率：{acc} / 304 = {(acc / 304):.2%}")
        results[pid]['extracted_answer'] = response['extracted_answer']
        if response['score'] == None: # 二次评估
            results[pid]['llm_score'] = second_score
        else:
            results[pid]['llm_score'] = response['score']
                
        
        extraction = response['extracted_answer']
        
        # results[pid]["rule_score"] = is_equal(gt_answer, extraction) or is_equal(gt_answer_value, extraction) # 两种都可以
        results[pid]['extracted_votes'] = responses
        end_time = time.time()
        results[pid]['extraction_time'] = end_time - start_time
        
        # with lock:
        #     results[pid]['extracted_answer'] = response['extracted_answer']
        #     results[pid]['llm_score'] = response['score']
        #     extraction = response['extracted_answer']
        #     results[pid]["rule_score"] = is_equal(gt_answer, extraction) or is_equal(gt_answer_value, extraction) # 两种都可以
        #     results[pid]['extracted_votes'] = responses
        #     end_time = time.time()
        #     results[pid]['extraction_time'] = end_time - start_time
        
        return pid
    
    def save_results():
        try:
            print(f"Saving results to {output_file}...")
            with lock:
                with open(output_file, "w+") as f:
                    json.dump(results, f, indent=4)
            print(f"Results saved.")
        except Exception as e:
            print(e)
            print(f"Error in saving {output_file}")
    
    # 操作 test_pids
    
    for pid in tqdm(test_pids,desc="Extracting answers"):
        if "response" not in results[pid]:
            continue
        
        process_query(pid)
        save_results()
        # break
    
    return 1
    
    # with ThreadPoolExecutor(max_workers=cfg.worker_num) as executor:
    #     futures = {executor.submit(process_query, pid): pid for pid in test_pids}
    #     for i, future in enumerate(tqdm(as_completed(futures), total=len(test_pids))):
    #         print(f"Completed {i} queries.")
    #         if future.result() is not None and i % 10 == 0:
    #             save_results()
    # save_results()

def extract(cfg, model):
    if cfg.data.name == "charxiv" or cfg.data.name == "custom_charxiv":
        ext_ans_charxiv(cfg, model)
    elif cfg.data.name == "mathvista" or cfg.data.name == "custom_mathvista":
        ext_ans_mathvista(cfg, model)
    elif cfg.data.name == "mathvision" or cfg.data.name == "custom_mathvision":
        ext_ans_mathvision(cfg, model)
    else:
        raise ValueError("Dataset not supported")
    
    input("\033[31m评估完毕！！！点击放弃GPU\033[0m")
    