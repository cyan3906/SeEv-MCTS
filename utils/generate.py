import os
import json
import asyncio
import pickle
from tqdm import tqdm, trange
import aiofiles
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

def gen_charxiv(cfg, model):
    data_cfg = cfg.data

    # input_file
    input_file = os.path.join(data_cfg.data_dir, f"{data_cfg.mode}_{data_cfg.split}.json")
    print(f"Reading {input_file}...")
    with open(input_file) as f:
        data = json.load(f)

    # output file
    output_file = os.path.join(cfg.output_dir, 
            f'gen-{data_cfg.mode}_{data_cfg.split}.json')
    
    if "mcts" in cfg.prompt_method.name:
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
    
    print("Number of test problems to run:", len(queries))
    print("Evaluation mode:", data_cfg.mode)
    print("Output file:", output_file)
    
    # 创建一个线程安全的字典和锁
    lock = threading.Lock()

    def process_query(k):
        if k in complete_ids:
            return None
        
        query = queries[k]['raw_question']
        image = queries[k]["figure_path"]
        try:
            response = model.get_response(query, image)
        except Exception as e:
            print(e)
            response = {"response": "Failed to generate response!"}

        if "root" in response:
            tree_save_path = os.path.join(tree_save_dir, f"{k}.pkl")
            with open(tree_save_path, "wb") as f:
                pickle.dump(response["root"], f)
            del response["root"]

        with lock:
            queries[k]['decision'] = response
            queries[k]['response'] = response["response"]

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

    # 使用线程池来并发处理查询
    with ThreadPoolExecutor(max_workers=cfg.worker_num) as executor:
        futures = {executor.submit(process_query, k): k for k in queries}
        
        for i, future in enumerate(tqdm(as_completed(futures), total=len(queries))):
            if future.result() is not None:
                print(f"Completed {i} queries.")
                if i % cfg.save_steps == 0:
                    save_results()
    
    save_results()


from data.MathVista.evaluation.build_query import create_query_data
def gen_mathvista(cfg, model):
    data_cfg = cfg.data
    input_file = os.path.join(data_cfg.data_dir, data_cfg.input_file)
    print(f"Reading {input_file}...")
    with open(input_file) as f:
        data = json.load(f)
    query_data = create_query_data(data)

    # output file
    output_file = os.path.join(cfg.output_dir, f'generation.json')
    
    if "mcts" in cfg.prompt_method.name:
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
    
    complited_ids = []
    
    # build final test pid list
    test_pids = list(results.keys())
    print("\nNumber of test problems in total:", len(test_pids))

    print("\nRemoving problems with existing valid response...")
    for pid in test_pids:
        if pid in results and 'response' in results[pid] and results[pid]['response'] != "Failed to generate response!":
            complited_ids.append(pid)
    
    # 写成多线程
    lock = threading.Lock()

    def process_query(pid):
        if pid in complited_ids:
            return None
        problem = data[pid]
        query = query_data[pid]
        image = problem['image']
        image_path = os.path.join(data_cfg.data_dir, image)

        try:
            response = model.get_response(query, image_path)
            
            if "root" in response:
                tree_save_path = os.path.join(tree_save_dir, f"{pid}.pkl")
                with open(tree_save_path, "wb") as f:
                    pickle.dump(response["root"], f)
                del response["root"]

            with lock:
                results[pid]['query'] = query

                results[pid]['decision'] = response
                results[pid]['response'] = response["response"]

        except Exception as e:
            print(str(e))
            with lock:
                results[pid]['query'] = query
                results[pid]['error'] = str(e)
                results[pid]['response'] = "Failed to generate response!"

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
    
    with ThreadPoolExecutor(max_workers=cfg.worker_num) as executor:
        futures = {executor.submit(process_query, pid): pid for pid in results}
        
        for i, future in enumerate(tqdm(as_completed(futures), total=len(results))):
            print(f"Completed {i} queries.")
            if future.result() is not None and i % cfg.save_steps == 0:
                save_results()

    save_results()


def load_jsonl(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data
def gen_mathvision(cfg, model):
    data_cfg = cfg.data
    input_file = os.path.join(data_cfg.data_dir, data_cfg.input_file)
    print(f"Reading {input_file}...")
    data = load_jsonl(input_file)
    questions = {}
    for line in data:
        questions[line['id']] = line
    output_file = os.path.join(cfg.output_dir, f'generation.json')
    
    if "mcts" in cfg.prompt_method.name:
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
    completed_ids = old_id

    # benchmark_prompt = "Please first solve the problem step by step, then put your final answer or a single letter (if it is a multiple choice question) in one \"\\boxed{}\". \n"

    # 写成多线程
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
            response = model.get_response(question, image_path)

            if "root" in response:
                tree_save_path = os.path.join(tree_save_dir, f"{idx}.pkl")
                with open(tree_save_path, "wb") as f:
                    pickle.dump(response["root"], f)
                del response["root"]

            with lock:
                questions[idx]["question"] = question
                questions[idx]["decision"] = response
                questions[idx]["response"] = response["response"]

        except Exception as e:
            print(str(e))
            with lock:
                questions[idx]["error"] = str(e)
                questions[idx]["response"] = "Failed to generate response!"

        return idx
    
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

    with ThreadPoolExecutor(max_workers=cfg.worker_num) as executor:
        futures = {executor.submit(process_query, idx): idx for idx in questions}
        
        for i, future in enumerate(tqdm(as_completed(futures), total=len(questions))):
            print(f"Completed {i} queries.")
            if future.result() is not None and i % cfg.save_steps == 0:
                save_results()

    save_results()
    

def generate(cfg, model):
    if cfg.data.name == "charxiv":
        gen_charxiv(cfg, model)
    elif cfg.data.name == "mathvista":
        gen_mathvista(cfg, model)
    elif cfg.data.name == "mathvision":
        gen_mathvision(cfg, model)
    else:
        raise ValueError("Dataset not supported")