import os
import json
import asyncio
import pickle
import time
from tqdm import tqdm, trange
import aiofiles
import os
import threading
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from tools.color_print import *


def atomic_json_dump(data, output_file):
    output_file = os.fspath(output_file)
    output_dir = os.path.dirname(output_file) or "."
    descriptor, temporary_file = tempfile.mkstemp(
        prefix=f".{os.path.basename(output_file)}.",
        suffix=".tmp",
        dir=output_dir,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=4, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_file, output_file)
    except Exception:
        if os.path.exists(temporary_file):
            os.unlink(temporary_file)
        raise


def atomic_pickle_dump(data, output_file):
    output_file = os.fspath(output_file)
    output_dir = os.path.dirname(output_file) or "."
    descriptor, temporary_file = tempfile.mkstemp(
        prefix=f".{os.path.basename(output_file)}.",
        suffix=".tmp",
        dir=output_dir,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            pickle.dump(data, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_file, output_file)
    except Exception:
        if os.path.exists(temporary_file):
            os.unlink(temporary_file)
        raise


def _has_usable_mcts_checkpoint(checkpoint_file):
    if not os.path.exists(checkpoint_file):
        return False
    try:
        with open(checkpoint_file, "rb") as handle:
            checkpoint = pickle.load(handle)
        return getattr(checkpoint, "root", None) is not None
    except Exception:
        return False


def is_out_of_memory_exception(error):
    message = str(error).casefold()
    return any(
        marker in message
        for marker in (
            "cuda out of memory",
            "outofmemoryerror",
            "cublas_status_alloc_failed",
            "cuda error: out of memory",
        )
    )

def gen_charxiv(cfg, model ,is_custom=False):
    data_cfg = cfg.data

    # input_file
    if is_custom:
        input_file = data_cfg.input_file
    else:
        input_file = os.path.join(data_cfg.data_dir, f"{data_cfg.mode}_{data_cfg.split}.json")
    
    with open(input_file, "r") as f:
        answer_data = json.load(f)
        
    # output file
    output_file = os.path.join(cfg.output_dir, 
            f'gen-{data_cfg.mode}_{data_cfg.split}.json')
    
    if os.path.exists(output_file):
        with open(output_file) as f:
            data = json.load(f)
    else:
        print(f"Reading {input_file}...")
        data = answer_data

    # if "mcts" in cfg.prompt_method.name or cfg.prompt_method.name.startswith("our"):
    tree_save_dir = os.path.join(cfg.output_dir, f'tree_results')
    os.makedirs(tree_save_dir, exist_ok=True)

    complete_ids = []
    saved_queries = {}
    if os.path.exists(output_file):
        print(f"{output_file} exists. Skipping generation.")
        with open(output_file) as f:
            queries = json.load(f)
        complete_ids = [k for k in queries if "response" in queries[k] and queries[k]['response'] != "Failed to generate response!"]
        print(f"{len(complete_ids)} problems skipped.")
        saved_queries = queries
    else:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # print(456)
    if data_cfg.mode == 'descriptive':
        from data.CharXiv.src.descriptive_utils import build_descriptive_quries
        queries = build_descriptive_quries(data, data_cfg.image_dir)
    elif data_cfg.mode == 'reasoning':
        from data.CharXiv.src.reasoning_utils import build_reasoning_queries
        queries = build_reasoning_queries(data, data_cfg.image_dir)
    else: 
        raise ValueError("Mode not supported")
    
    # merge queries
    for k in queries:
        if k in saved_queries and k in complete_ids:
            queries[k] = saved_queries[k]
    # print(queries["1"])
    print("Number of test problems to run:", len(queries))
    print("Evaluation mode:", data_cfg.mode)
    print("Output file:", output_file)
    
    # 创建一个线程安全的字典和锁
    lock = threading.Lock()

    def process_query(k):
        if k in complete_ids:
            return None
        query = queries[k]['raw_question']
        print(query)
        image = queries[k]["figure_path"]
        
        is_pkl = False
        tree_save_path = None
        if tree_save_dir:
            tree_save_path = os.path.join(tree_save_dir, f"{k}.pkl")
            if os.path.exists(tree_save_path):
                is_pkl = True
                
        try:
            time_start = time.time()
            response = model.get_response(query, image) # 模型真正的输出结果的地方
            
        except Exception as e:
            print(e)
            response = {"response": "Failed to generate response!"}
            
        if "root" in response:
            tree_save_path = os.path.join(tree_save_dir, f"{k}.pkl")
            with open(tree_save_path, "wb") as f:
                pickle.dump(response["root"], f)
            del response["root"]

        if "mcts" in response:
            tree_save_path = os.path.join(tree_save_dir, f"{k}mcts.pkl")
            with open(tree_save_path, "wb") as f:
                pickle.dump(response["mcts"], f)
            del response["mcts"]

        with lock:
            queries[k]['decision'] = response
            
            queries[k]["answer"] = answer_data[k]['answer']
            if cfg.prompt_method.name == "directQ":
                queries[k]['response'] = response["response"][0]
            else:
                queries[k]['response'] = response["response"]
            queries[k]['time'] = time.time() - time_start
            queries[k].pop("figure_path", None)
            queries[k].pop("question", None)
            
        
        return k

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


    if cfg.prompt_method.name == "mctsv8" and cfg.prompt_method.is_p3:
        skip_list = ["38", "130"]
    else:
        skip_list = []
        
    # skip_list = ['196']
    
    for id in tqdm(data.keys(),desc="Processing queries"):
        # if int(id) < 1170:
        if id in skip_list:
            continue
        
        if str(id) == "1791" or str(id) == "247":
            continue
        
        print(f"\033[95m正在处理第{int(id)}条样本\033[0m")
        
        cfg.prompt_method.data_id = id
        # 处理logger
        logger_path = os.path.join(cfg.output_dir, "logger")
        os.makedirs(logger_path, exist_ok=True)
        
        logger_json = os.path.join(logger_path, "PK_logger.json")
        if os.path.exists(logger_json):
            with open(logger_json, "r") as f:
                PK_data = json.load(f)
        else:
            PK_data = {}
        PK_data[str(id)] = [] # 重置
        # 保存PK_json
        with open(logger_json, "w",encoding="utf-8") as f:
            json.dump(PK_data, f, indent=4 , ensure_ascii=False)
            
        try_num = 1
        
        for _ in range(try_num):
            if _ > 0:
                with open(output_file, "r", encoding="utf-8") as f:
                    results = json.load(f)
                    if id in results and "response" in results[id] and results[id]["response"] != "Failed to generate response!":
                        break
            process_query(id)
            save_results()
            # return 1
            
    # 使用线程池来并发处理查询
    # with ThreadPoolExecutor(max_workers=cfg.worker_num) as executor:
    #     futures = {executor.submit(process_query, k): k for k in queries}
        
    #     for i, future in enumerate(tqdm(as_completed(futures), total=len(queries))):
    #         if future.result() is not None:
    #             print(f"Completed {i} queries.")
    #             if i % cfg.save_steps == 0:
    #                 save_results()
    
    # save_results()

import sys
# sys.path.insert(0,"/home/jncsnlp4/SSD2/ZJJ/VReST")
from data.MathVista.evaluation.build_query import create_query_data
def gen_mathvista(cfg, model, is_custom=False):
    data_cfg = cfg.data
    
    if is_custom:
        input_file = data_cfg.input_file
    else:
        input_file = os.path.join(data_cfg.data_dir, data_cfg.input_file)
    
    
    # data_cfg.data_dir:data/MATH-V/data
    # data_cfg.input_file:testmini.jsonl
    
    print(f"Reading {input_file}...")
    with open(input_file) as f:
        data = json.load(f)
    
    # 是否进行问题重塑
    if cfg.prompt_method.is_rephrasing_question:
        from utils.rephrasing_question_prompt import rephrasing_question_prompt
        
        for i,j in data.items():
            messages=[
                {
                    "role": "system", "content": "You are a helpful assistant."
                },
                {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": os.path.join("/home/jncsnlp4/SSD2/ZJJ/VReST/data/MathVista/data/images/testmini/ALL", j["image"].split("/")[-1]),
                    },
                    {"type": "text", "text": rephrasing_question_prompt.format(question=j["question"])},
                    ],
                },
            ]
            
            data[i]["original_question"] = j["question"]
            
            print("原来的问题:",j["question"])
            data[i]["question"] = model.get_response(cfg,messages)
            print("重塑后的问题:",data[i]["question"])
        
    query_data = create_query_data(data)

    # print(query_data["1"])
    # import sys
    # sys.exit(0)
    
    # output file
    output_file = os.path.join(cfg.output_dir, f'generation.json')
    
    tree_save_dir = None
    if "mcts" in cfg.prompt_method.name or cfg.prompt_method.name.startswith("our"):
        tree_save_dir = os.path.join(cfg.output_dir, f'tree_results')
        os.makedirs(tree_save_dir, exist_ok=True)

    # load results
    if os.path.exists(output_file):
        print("\nResults already exist.")
        print(f"Reading {output_file}...")
        with open(output_file) as f:
            results = json.load(f)
    else:
        results = {}
    
    data.update(results)
    results = data
    
    complited_ids = [] # 我需要测试的列表
    
    # build final test pid list
    test_pids = list(results.keys())
    print("\nNumber of test problems in total:", len(test_pids))

    print("\nRemoving problems with existing valid response...")
    for pid in test_pids:
        if pid in results and 'response' in results[pid] and results[pid]['response'] != "Failed to generate response!":
            complited_ids.append(pid)
    
    print(f"已经完成的{len(complited_ids)}条样本")
    print(complited_ids)
    
    # 写成多线程
    lock = threading.Lock()

    def process_query(pid):
        if pid in complited_ids:
            return None
        
        problem = data[pid]
        query = query_data[pid]
        image = problem['image'].split("/")[-1]
        image_path = os.path.join(data_cfg.image_dir, image)
        
        # options = ''
        # if line['choices'] and len(line['choices']) > 0:
        #     if len(line['choices']) == 5:
        #         options = f"(A) {line['choices'][0]}\n(B) {line['choices'][1]}\n(C) {line['choices'][2]}\n(D) {line['choices'][3]}\n(E) {line['choices'][4]}\n"
        #     elif len(line['choices']) == 4:
        #         options = f"(A) {line['choices'][0]}\n(B) {line['choices'][1]}\n(C) {line['choices'][2]}\n(D) {line['choices'][3]}\n"
        #     elif len(line['choices']) == 3:
        #         options = f"(A) {line['choices'][0]}\n(B) {line['choices'][1]}\n(C) {line['choices'][2]}\n"
        #     elif len(line['choices']) == 2:
        #         options = f"(A) {line['choices'][0]}\n(B) {line['choices'][1]}\n"
        # question = f"{question}\n{options}"
        
        # query = query_data[pid]
        
        is_pkl = False
        tree_save_path = None
        if tree_save_dir:
            tree_save_path = os.path.join(tree_save_dir, f"{pid}.pkl")
            if os.path.exists(tree_save_path):
                is_pkl = True
            
        try:
            time_start = time.time()
            printblue(f"{query}")
            response = model.get_response(query, image_path, is_pkl, tree_save_path)
            # print(f"\033[92mResponse: {response}\033[0m")
            if "root" in response:
                tree_save_path = os.path.join(tree_save_dir, f"{pid}.pkl")
                # print(dir(response["root"]))
                with open(tree_save_path, "wb") as f:
                    pickle.dump(response["root"], f)
                del response["root"]

            if "mcts" in response:
                tree_save_path = os.path.join(tree_save_dir, f"{pid}mcts.pkl")
                with open(tree_save_path, "wb") as f:
                    pickle.dump(response["mcts"], f)
                del response["mcts"]
                
            with lock:
                # results = data
                results[pid]['query'] = query
                results[pid]['decision'] = response
                results[pid]['response'] = response["response"]
                results[pid]['raw_response'] = response.get("raw_response")
                results[pid]['answer_extract_status'] = response.get(
                    "answer_extract_status", "not_applicable"
                )
                results[pid]["time"] = time.time() - time_start
                # if results[pid]["question_type"] == "multi_choice":
                #     for i in range(len(line['choices'])):
                #         if results[pid]["answer"] == line['choices'][i]:
                #             results[pid]["answer"] = chr(65 + i)
                #             break
                        
        except Exception as e:
            print(str(e))
            with lock:
                results[pid]['query'] = query
                results[pid]['error'] = str(e)
                results[pid]['response'] = "Failed to generate response!"
            if is_out_of_memory_exception(e):
                raise
        
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
    
    
    if cfg.prompt_method.name == "mctsv8" and cfg.prompt_method.is_p3:
        skip_list = ["27", "70","86","167","178","229","231","240","241","264","299","311","329","340","349","405",
                     "420","473","518","525","542","569","583","588","603","661","664","670","688","702","710","715","745","792","793","796","813","832","836","838","847","864","883","902","923","932","941","960","973","","","",""]
    else:
        skip_list = []
    
    # skip_list = []
    
    print(f"\033[93mskip_list: {skip_list}\033[0m")
    
    last_id = -1
    for id in tqdm(test_pids,desc="Processing queries"):
        
        if id in skip_list:
            continue
        
        print(f"\033[95m正在处理第{int(id)}条样本\033[0m")
        
        cfg.prompt_method.data_id = id
        # 处理logger
        logger_path = os.path.join(cfg.output_dir, "logger")
        os.makedirs(logger_path, exist_ok=True)
        
        logger_json = os.path.join(logger_path, "PK_logger.json")
        if os.path.exists(logger_json):
            with open(logger_json, "r") as f:
                PK_data = json.load(f)
        else:
            PK_data = {}
        PK_data[str(id)] = [] # 重置
        # 保存PK_json
        with open(logger_json, "w",encoding="utf-8") as f:
            json.dump(PK_data, f, indent=4 , ensure_ascii=False)
        
        try_num = 1
        
        for _ in range(try_num):
            if _ > 0:
                with open(output_file, "r", encoding="utf-8") as f:
                    results = json.load(f)
                    if id in results and "response" in results[id] and results[id]["response"] != "Failed to generate response!":
                        break
            process_query(id)
            save_results()
            # return 1
        
    return 1

    # with ThreadPoolExecutor(max_workers=cfg.worker_num) as executor:
    #     futures = {executor.submit(process_query, pid): pid for pid in results}
        
    #     for i, future in enumerate(tqdm(as_completed(futures), total=len(results))):
    #         print(f"Completed {i} queries.")
    #         if future.result() is not None and i % cfg.save_steps == 0:
    #             save_results()

    # save_results()

def load_jsonl(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data

def gen_mathvision(cfg, model, is_custom = False):
    output_file = os.path.join(cfg.output_dir, f'generation.json') # 注意格式问题 !!!!!!
    data_cfg = cfg.data
    if os.path.exists(output_file):
        # 已经存在就加载
        with open(output_file, "r") as f:
            data = json.load(f)
            
        questions = {}
        for idx,line in data.items():
            questions[idx] = line
            
    else:
        
        test_list = []
        
        if is_custom:
            run_file = data_cfg.run_file
            with open(run_file, "r") as f:
                run_data = json.load(f)
            for i,j in run_data.items():
                test_list.append(i)
        
        input_file = os.path.join(data_cfg.data_dir, data_cfg.input_file)
            
        print(f"Reading {input_file}...")
        
        data = load_jsonl(input_file)
        
        questions = {}
        for line in data:
            if is_custom:
                if line['id'] in test_list:
                    questions[line['id']] = line
                    
            else:
                questions[line['id']] = line
    
    print(output_file)
    
    tree_save_dir = None
    if "mcts" in cfg.prompt_method.name or cfg.prompt_method.name.startswith("our"):
        tree_save_dir = os.path.join(cfg.output_dir, f'tree_results')
        os.makedirs(tree_save_dir, exist_ok=True)
    
    old_id = []
    if os.path.exists(output_file):
        with open(output_file) as f:
            results = json.load(f)
            for k in questions:
                if k in results and 'response' in results[k] and results[k]['response'] != "Failed to generate response!":
                    questions[k] = results[k]
                    old_id.append(k)
                    
    print(f"old_id:{old_id}")
    
    # old_id.remove("39")
    
    # for j in [44, 45, 60, 78, 79, 86, 90, 91]:
    #     old_id.remove(str(j))
        
    completed_ids = old_id
    # return 1
    # benchmark_prompt = "Please first solve the problem step by step, then put your final answer or a single letter (if it is a multiple choice question) in one \"\\boxed{}\". \n"

    # 写成多线程
    lock = threading.Lock()

    def process_query(idx):
        if idx in old_id:
            return None
        
        line = questions[idx]
        
        question = line['question']
        # if "query" in line:
        #     question = line['query']
        # else:
        #     question = line['question']
        
        print(question)
        
        options = ''
        if len(line['options']) > 0:
            assert len(line['options']) == 5, f"len(line['options']) == {len(line['options'])} != 5"
            options = f"(A) {line['options'][0]}\n(B) {line['options'][1]}\n(C) {line['options'][2]}\n(D) {line['options'][3]}\n(E) {line['options'][4]}\n"
        question = f"{question}\n{options}"
        image_path = os.path.join(data_cfg.image_dir, line['image'])

        is_pkl = False
        tree_save_path = None
        if tree_save_dir:
            tree_save_path = os.path.join(tree_save_dir, f"{idx}.pkl")
            if os.path.exists(tree_save_path):
                is_pkl = True
        
        try:
            time_start = time.time()
            response = model.get_response(question, image_path) # 这里的get_response 方法是指整体模型的调用里面再调 mcts_forward_v8
            # print(f"\033[92m输出了response: {response}\033[0m")
            if "root" in response:
                tree_save_path = os.path.join(tree_save_dir, f"{idx}.pkl")
                with open(tree_save_path, "wb") as f:
                    pickle.dump(response["root"], f)
                del response["root"] # 释放内存

            if "mcts" in response:
                tree_save_path = os.path.join(tree_save_dir, f"{idx}mcts.pkl")
                # print(f"tree_save_path: {tree_save_path}")
                with open(tree_save_path, "wb") as f:
                    pickle.dump(response["mcts"], f)
                del response["mcts"]

            with lock:
                questions[idx]["question"] = question
                # questions[idx]["answer"] = line['answer'] # 2026-1-14
                questions[idx]["decision"] = response
                if cfg.prompt_method.name == "directQ":
                    questions[idx]["response"] = response["response"][0]
                else:
                    questions[idx]["response"] = response["response"]
                questions[idx]["time"] = time.time() - time_start
                
        except Exception as e:
            print(str(e))
            with lock:
                questions[idx]["error"] = str(e)
                questions[idx]["response"] = "Failed to generate response!"

        return idx
    
    # 这个是更新generation.json用的
    def save_results():
        try:
            print(f"Saving results to {output_file}...")
            with lock:
                with open(output_file, "w+") as f:
                    json.dump(questions, f, indent=4)
            print(f"Results saved.")
        except Exception as e:
            print(e)
            print(f"Error in saving {output_file}")

    # 随机试试
    # process_query('5')
    
    # skip_list = ['131', '357', '893', '1399', '2955', '2984', '3033']
    
    if cfg.prompt_method.name == "mctsv8" and cfg.prompt_method.is_p3:
        skip_list = ["33", "1168"]
    else:
        skip_list = []
    
    print(f"\033[93mskip_list: {skip_list}\033[0m")
    
    for id in tqdm(questions.keys(),desc="Processing queries"):
        if id in skip_list:
            continue
        #if str(id) not in ['20', '32', '55', '64', '65', '78', '86', '92', '95', '114', '123', '152', '181', '206', '242', '253', '254', '318', '336', '403', '441', '522', '537', '628', '739', '766', '767', '812', '819', '855', '923', '946', '1060', '1137', '1203', '1215', '1222', '1343', '1389', '1451', '1634', '1680', '1813', '1969', '2443', '2565', '2688', '3033']:
            #continue
        print(f"\033[95m正在处理第{int(id)}条样本\033[0m")
        
        
        cfg.prompt_method.data_id = id
    
        # 处理logger
        logger_path = os.path.join(cfg.output_dir, "logger")
        os.makedirs(logger_path, exist_ok=True)
        
        logger_json = os.path.join(logger_path, "PK_logger.json")
        if os.path.exists(logger_json):
            with open(logger_json, "r") as f:
                PK_data = json.load(f)
        else:
            PK_data = {}
        PK_data[str(id)] = [] # 重置
        # 保存PK_json
        with open(logger_json, "w",encoding="utf-8") as f:
            json.dump(PK_data, f, indent=4 , ensure_ascii=False)
            
        try_num = 1
        
        for _ in range(try_num):
            if _ > 0:
                with open(output_file, "r", encoding="utf-8") as f:
                    results = json.load(f)
                    if id in results and "response" in results[id] and results[id]["response"] != "Failed to generate response!":
                        break
            process_query(id)
            save_results()
            # return 1
    
    return 1

# 自定义数据集 可以跑部分的意思就是
def custom_mathvison(cfg,model):
    
    data_cfg = cfg.data
    
    # 导入自定义的数据
    with open(cfg.data.input_file) as f:
        questions = json.load(f)
    
    output_file = os.path.join(cfg.output_dir, f'generation.json')
    print(output_file)
    
    if "mcts" in cfg.prompt_method.name or cfg.data.name.startswith("our"):
        tree_save_dir = os.path.join(cfg.output_dir, f'tree_results')
        os.makedirs(tree_save_dir, exist_ok=True)
    
    output_file = os.path.join(cfg.output_dir, f'generation.json')
    print(output_file)
    
    old_id = []
    if os.path.exists(output_file):
        with open(output_file) as f:
            results = json.load(f)
            for k in questions:
                if k in results and 'response' in results[k] and results[k]['response'] != "Failed to generate response!":
                    questions[k] = results[k]
                    old_id.append(k)
                    
    print(f"old_id:{old_id}")

    lock = threading.Lock()

    def process_query(idx):
        if idx in old_id:
            return None
        
        line = questions[idx]
        question = line['question']
        options = ''
        if len(line['options']) > 0:
            assert len(line['options']) == 5, f"len(line['options']) == {len(line['options'])} != 5"
            options = f"(A) {line['options'][0]}\n(B) {line['options'][1]}\n(C) {line['options'][2]}\n(D) {line['options'][3]}\n(E) {line['options'][4]}\n"
        question = f"{question}\n{options}"
        image_path = os.path.join(data_cfg.image_dir, line['image'])

        try:
            time_start = time.time()
            response = model.get_response(question, image_path) # 这里的get_response 方法是指整体模型的调用里面再调 mcts_forward_v8
            print("输出了response")
            if "root" in response:
                tree_save_path = os.path.join(tree_save_dir, f"{idx}.pkl")
                with open(tree_save_path, "wb") as f:
                    pickle.dump(response["root"], f)
                del response["root"] # 释放内存

            with lock:
                questions[idx]["question"] = question
                questions[idx]["decision"] = response
                questions[idx]["response"] = response["response"]
                questions[idx]["time"] = time.time() - time_start
                
        except Exception as e:
            print(str(e))
            with lock:
                questions[idx]["error"] = str(e)
                questions[idx]["response"] = "Failed to generate response!"

        return idx
    
    # 这个是更新generation.json用的
    def save_results():
        try:
            print(f"Saving results to {output_file}...")
            with lock:
                with open(output_file, "w+") as f:
                    json.dump(questions, f, indent=4)
            print(f"Results saved.")
        except Exception as e:
            print(e)
            print(f"Error in saving {output_file}")

    # 随机试试
    # process_query('5')
    
    for idx in tqdm(questions.keys(),desc="Processing queries"):
        
        print(f"\033[95m正在处理第{int(idx)}条样本\033[0m")
        process_query(idx)
        save_results()
        # return 1
    
    return 1


def load_math500_records(input_file, max_samples=None):
    with open(input_file, "r", encoding="utf-8") as handle:
        raw_records = json.load(handle)
    if not isinstance(raw_records, list):
        raise ValueError("MATH500 input must be a JSON list")
    if max_samples is not None:
        raw_records = raw_records[: int(max_samples)]

    records = {}
    for record in raw_records:
        record_id = str(record["id"])
        if record_id in records:
            raise ValueError(f"Duplicate MATH500 id: {record_id}")
        if not record.get("problem"):
            raise ValueError(f"MATH500 record {record_id} has no problem")
        records[record_id] = record
    return records


def gen_math500(cfg, model):
    records = load_math500_records(
        cfg.data.input_file,
        max_samples=getattr(cfg.data, "max_samples", None),
    )
    output_file = os.path.join(cfg.output_dir, "generation.json")
    tree_save_dir = os.path.join(cfg.output_dir, "tree_results")
    os.makedirs(tree_save_dir, exist_ok=True)

    completed_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
        for record_id in records:
            saved = existing.get(record_id)
            if (
                saved
                and saved.get("response")
                and saved["response"] != "Failed to generate response!"
            ):
                records[record_id] = saved
                completed_ids.add(record_id)

    def save_results():
        atomic_json_dump(records, output_file)

    for record_id, record in tqdm(records.items(), desc="Processing MATH500"):
        if record_id in completed_ids:
            continue

        question = record["problem"]
        tree_save_path = os.path.join(tree_save_dir, f"{record_id}.pkl")
        mcts_path = os.path.join(tree_save_dir, f"{record_id}mcts.pkl")
        is_pkl = _has_usable_mcts_checkpoint(mcts_path)
        cfg.prompt_method.data_id = record_id
        try:
            time_start = time.time()
            response = model.get_response(
                question,
                None,
                is_pkl,
                tree_save_path,
            )
            root = response.pop("root", None)
            if root is not None:
                atomic_pickle_dump(root, tree_save_path)
            mcts = response.pop("mcts", None)
            if mcts is not None:
                atomic_pickle_dump(mcts, mcts_path)

            record["query"] = question
            record["decision"] = response
            record["response"] = response["response"]
            record["raw_response"] = response.get("raw_response")
            record["answer_extract_status"] = response.get(
                "answer_extract_status", "not_applicable"
            )
            record["time"] = time.time() - time_start
        except Exception as error:
            record["query"] = question
            record["error"] = str(error)
            record["response"] = "Failed to generate response!"
            save_results()
            if is_out_of_memory_exception(error):
                raise
        save_results()
    return 1


def generate(cfg, model):
    if cfg.data.name == "charxiv":
        gen_charxiv(cfg, model)
        
    elif cfg.data.name == "mathvista":
        gen_mathvista(cfg, model)
        
    elif cfg.data.name == "mathvision":
        gen_mathvision(cfg, model)
        
    elif cfg.data.name == "custom_mathvision":
        gen_mathvision(cfg, model, is_custom = True)
    
    elif cfg.data.name == "custom_mathvista":
        gen_mathvista(cfg, model ,is_custom=True)
    
    elif cfg.data.name == "custom_charxiv":
        gen_charxiv(cfg, model, is_custom=True)

    elif cfg.data.name == "math500":
        gen_math500(cfg, model)
        
    else:
        raise ValueError("Dataset not supported")
    
    
    # input("\033[31m跑完了！！！点击放弃GPU\033[0m")
