import os,json,csv,re,time

from tools.color_print import *
import pickle
from vllm import LLM, SamplingParams
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

from tqdm import tqdm
import queue as queue_module

def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # qwen_vl_utils 0.0.14+ reqired
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )
    # print(f"video_kwargs: {video_kwargs}")

    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs

    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs
    }
    
def pk_vllm():
    model_path = "/home/jncsnlp4/SSD2/model/qwen2-vl-intruct"
    # model_path = "/home/jncsnlp4/SSD2/model/Qwen3-VL-8B-Instruct"
    llm = LLM(
        model=model_path,
        limit_mm_per_prompt = {"image": 10, "video": 10}, # 每个提示最多包含的图片和视频数量
        max_model_len = 6096,
        max_num_seqs = 713,
        gpu_memory_utilization = 0.85
    )
    
    printpink("模型加载完毕!")
    
    # 优化采样参数，获取更合理的概率分布
    sampling_params = SamplingParams(
        temperature=0.7,  # 稍微增加随机性，有助于获取更合理的概率
        top_p=0.95,
        repetition_penalty=1.05,
        max_tokens=713,  # 允许生成更多token
        # logprobs=5     # 获取前5个可能的token及其概率
        n = 11
    )
    
def vllm():
    try:
        
        model_path = "/home/jncsnlp4/SSD2/model/qwen2-vl-intruct"
        # model_path = "/home/jncsnlp4/SSD2/model/Qwen3-VL-8B-Instruct"
        llm = LLM(
            model=model_path,
            limit_mm_per_prompt = {"image": 10, "video": 10}, # 每个提示最多包含的图片和视频数量
            max_model_len = 6096,
            max_num_seqs = 713,
            gpu_memory_utilization = 0.85
        )
        
        printpink("模型加载完毕!")
        
        # 优化采样参数，获取更合理的概率分布
        sampling_params = SamplingParams(
            temperature=0.7,  # 稍微增加随机性，有助于获取更合理的概率
            top_p=0.95,
            repetition_penalty=1.05,
            max_tokens=713,  # 允许生成更多token
            # logprobs=5     # 获取前5个可能的token及其概率
            n = 11
        )
        
        # image_url = "/home/jncsnlp4/SSD2/ZJJ/VReST/data/MATH-V/images/26.jpg"
        # list1 = ["20"]
        
        image_dir = "/home/jncsnlp4/SSD2/ZJJ/VReST/data/MathVista/data/images/testmini/ALL"
        
        score_file = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=ours2v3.2.mathvista.MWP2/data=custom_mathvista/llm=Qwen2-VL-7B-Instruct/prompt_method=our2v3/ext_llm=Qwen2.5-7B-Instruct/method=ans_and_score/field=max_vote_terminal/n=10/reward_method=self_consistency/scores.json"
        # score_file = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=ours2v3.2.mathvista.FQA/data=custom_mathvista/llm=Qwen2-VL-7B-Instruct/prompt_method=our2v3/ext_llm=Qwen2.5-7B-Instruct/method=ans_and_score/field=max_vote_terminal/n=10/reward_method=self_consistency/scores.json"
        # score_file = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=ours2v3.2.mathvista.TQA/data=custom_mathvista/llm=Qwen2-VL-7B-Instruct/prompt_method=our2v3/ext_llm=Qwen2.5-7B-Instruct/method=ans_and_score/field=max_vote_terminal/n=10/reward_method=self_consistency/scores.json"
        # score_file = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=ours2v3.2.mathvista.GPS/data=custom_mathvista/llm=Qwen2-VL-7B-Instruct/prompt_method=our2v3/ext_llm=Qwen2.5-7B-Instruct/method=ans_and_score/field=max_vote_terminal/n=10/reward_method=self_consistency/scores.json"
        # score_file = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=ours2v3.2.mathvista.VQA/data=custom_mathvista/llm=Qwen2-VL-7B-Instruct/prompt_method=our2v3/ext_llm=Qwen2.5-7B-Instruct/method=ans_and_score/field=max_vote_terminal/n=10/reward_method=self_consistency/scores.json"
        
        with open(score_file, "r") as f:
            data = json.load(f)
        
        
        list2 = []
        
        pk_file = "pk.json"
        if not os.path.exists(pk_file):
            with open(pk_file, "w") as f:
                json.dump({}, f, ensure_ascii=False, indent=4)
            with open(pk_file, "r") as f:
                pk_data = json.load(f)
            
        else:
            with open(pk_file, "r") as f:
                pk_data = json.load(f)
        
        test_list1 = []
        for i,j in data.items():
            if i not in pk_data:
                with open(f'/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=ours2v4.1.mathvista.MWP/data=custom_mathvista/llm=Qwen2-VL-7B-Instruct/prompt_method=our2v4/tree_results/{i}.pkl', 'rb') as f:  # 'rb' 表示二进制读取模式
                    root = pickle.load(f)  # data 即为反序列化后的对象（可能是列表、字典、类实例等）
                test_list1.append((i,j["answer"],sorted(list(root.answer_tree.keys()))))
        
        printyellow(f"测试数量:{len(test_list1)}")
        
        for i in tqdm(test_list1):
            
            # printblue(f"{i}")
            printblue(f"正在测试idx:{i[0]}")
            printblue(f"{i}")
            idx,answer,response = i
            
            win_num = 0
            # extracted_answer = response.split("answer is")[-1].strip()
            # if extracted_answer[-1] in [".","?","!"]:
            #     extracted_answer = extracted_answer[:-1]
            
            for extracted_answer in response:
                if extracted_answer == answer:
                    continue
                
                # i = input("请输入idx：")
                win = 0
                for j in range(5):
                    
                    image_file = os.path.join(image_dir,f"{int(idx)}.jpg")
                    # print(idx)
                    # question = data[idx]["question"]
                    
                    false_answer = 10
                    
                    if data[idx]["choices"]:
                        direct_answer = f"""
                            According to the options and please answer directly and only output the result. For multiple-choice questions, output the option.Do not generate any other irrelevant content.\n
                            **Question**
                            {question}\n
                            **Options**
                            {data[idx]["choices"]}\n
                            **Answer**
                        """
                    else:
                        direct_answer = f"""
                            Please answer directly and only output the result. For multiple-choice questions, output the option.Do not generate any other irrelevant content.\n
                            **Question**
                            {question}\n
                            **Answer**
                        """
                    rewrite_answer_prefix = f"""
                        Provide you one question and False Answer. Please output the correct answer.
                        **Output Example:**

                        **Question**
                        What is 2+3?\n
                        False Answer: 6

                        **Output**
                        The correct answer is 5.

                        **Question**
                        What is sum of 12 and 3?\n
                        False Answer: 14

                        **Output**
                        The correct answer is 15.

                        **Question**
                        What is 2+3+2?\n
                        False Answer: 6

                        **Output**
                        The correct answer is 7.

                        **Question**
                        What is sum of 22 and 3?\n
                        False Answer: 24

                        **Output**
                        The correct answer is 25.
                        
                        **Question**
                        {question}\n
                        False Answer: {false_answer}

                        **Output**

                    """
                    
                    # answer = 5
                    
                    direct_answer_verify_prefix = f"""
                        Provide you one question and answer. Please judge whether the answer is correct.
                        **Output Example:**
                        
                        **Question**
                        What is 2+3?\n
                        Answer: 6
                        
                        **Output**
                        No.The answer is not correct.
                        
                        **Question**
                        What is sum of 2 and 3?\n
                        Answer: 5
                        
                        **Output**
                        Yes.The answer is correct.
                        
                        **Question**
                        {question}\n
                        Answer: {answer}
                        
                        **Output**
                    
                    """
                    
                    answer2 = f"""
                    {answer}
                    """
                    answer1 = f"""
                    {extracted_answer}
                    """
                    
                    direct_answer_verify_prefix2 = f"""
                        Provide you with two answers to this question.Please output the answer that is more accurate.
                        **Output Example:**
                        
                        **Question:**
                        What is 2+3?\n
                        Answer 1: 5
                        Answer 2: 6
                        
                        **Output**
                        The more accurate answer is 5.
                        
                        **Question:**
                        What is sum of 2 and 3?\n
                        Answer 1: 5
                        Answer 2: 6
                        
                        **Output**
                        The more accurate answer is 5.
                        
                        **Question:**
                        {question} (strictly greater than 2, excluding exactly 2 times)\n
                        Answer 1: {answer1}
                        Answer 2: {answer2}
                        
                        **Output**
                    """
                    
                    # printred(f"{direct_answer_verify_prefix2}")
                    # 准备消息
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": image_file,
                                },
                                {"type": "text", "text": direct_answer_verify_prefix2},
                            ],
                        },
                    ]
                    
                    processor = AutoProcessor.from_pretrained(model_path)
                    prompt = processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    image_inputs, video_inputs = process_vision_info(messages)
                    
                    mm_data = {}
                    if image_inputs is not None:
                        mm_data["image"] = image_inputs
                    if video_inputs is not None:
                        mm_data["video"] = video_inputs

                    llm_inputs = {
                        "prompt": prompt,
                        "multi_modal_data": mm_data,
                    }

                    # imilarity()
                    import time
                    time_start = time.time()
                    outputs = llm.generate([llm_inputs], sampling_params=sampling_params, use_tqdm=False)
                    # print(outputs)
                    # 处理输出并计算概率
                    a = b = 0
                    for _,output in enumerate(outputs[0].outputs):
                        ans = output.text.strip().split("answer is")[-1].strip()
                        # if _ % 2: printgreen(f"{ans},{answer}")
                        # else: printblue(f"{ans},{answer}")
                        
                        if answer in ans:
                            a += 1
                        else:
                            b += 1
                    
                    if a > b:
                        win += 1
                        # if idx % 2:
                        #     printgreen(f"{output.text}")
                        # else:
                        #     printblue(f"{output.text}")
                            
                    time_end = time.time()
                    print(f"vllm time: {time_end - time_start}")
                
                if win >= 3:
                    # list2.append(idx)
                    win_num += 1
            
            if win_num >= (len(response) - 1) // 2 + 1:
                printgreen(f"存在: {idx} 是PK冠军!!!")
                list2.append(idx)
                pk_data.update({idx:{"win":f"{win}","total":f"{len(response) - 1}","is_win":True}})
            else:
                printred(f"存在: {idx} 不是PK冠军!!!")
                pk_data.update({idx:{"win":f"{win}","total":f"{len(response) - 1}","is_win":False}})
                
            with open(pk_file, "w") as f:
                json.dump(pk_data, f, ensure_ascii=False, indent=4)
            # input("点击继续")
        print(len(list2))
        print(list2)
    except Exception as e:
        print(f"vllm error: {e}")
        
def qvllm():
    def get_ost_prompt(solution_trace):
        try:
            user_prompt = "Let us think step by step.\n"
            for step_id, step_text in enumerate(solution_trace):
                user_prompt += f"Step {step_id + 1}: " + step_text + "\n"
            
            return user_prompt
        
        except Exception as e:
            printred(f"Error:get_ost_prompt: {e}")
            return ""
        
    model_path = "/home/jncsnlp4/SSD2/model/qwen2-vl-intruct"
    # model_path = "/home/jncsnlp4/SSD2/model/Qwen3-VL-8B-Instruct"
    llm = LLM(
        model=model_path,
        limit_mm_per_prompt = {"image": 10, "video": 10}, # 每个提示最多包含的图片和视频数量
        max_model_len = 6096,
        max_num_seqs = 713,
        gpu_memory_utilization = 0.85
    )
    
    printpink("模型加载完毕!")
    
    return_num = 11
    
    # 优化采样参数，获取更合理的概率分布
    sampling_params = SamplingParams(
        temperature=0.7,  # 稍微增加随机性，有助于获取更合理的概率
        top_p=0.95,
        repetition_penalty=1.05,
        max_tokens=713,  # 允许生成更多token
        # logprobs=5     # 获取前5个可能的token及其概率
        n = return_num
    )
    
    # image_url = "/home/jncsnlp4/SSD2/ZJJ/VReST/data/MATH-V/images/26.jpg"
    # list1 = ["20"]
    
    image_dir = "/home/jncsnlp4/SSD2/ZJJ/VReST/data/MathVista/data/images/testmini/ALL"
    
    data_json = "/home/jncsnlp4/SSD2/ZJJ/VReST/data/MathVista/data/testmini.json"
    with open(data_json, "r") as f:
        data = json.load(f)
    
    # test_list = list(data.keys())
    test_list = []
    
    for i,j in data.items():
        if j["metadata"]["task"] == "math word problem":
            test_list.append(i)
    
    
    if not os.path.exists("test.json"):
        with open("test.json", "w") as f:
            json.dump({}, f, ensure_ascii=False, indent=4)
    
    with open("test.json", "r") as f:
        save_data = json.load(f)
    
    for i in tqdm(test_list):
        if i in save_data:
            continue
        
        save_data.update({i: {}})
        
        run_list = []
        
        with open(f'/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=ours2v4.1.mathvista.MWP/data=custom_mathvista/llm=Qwen2-VL-7B-Instruct/prompt_method=our2v4/tree_results/{i}.pkl', 'rb') as f:  # 'rb' 表示二进制读取模式
            root = pickle.load(f)  # data 即为反序列化后的对象（可能是列表、字典、类实例等）

        queue = queue_module.Queue()
        queue.put(root)

        while not queue.empty():
            root = queue.get()
            if root.node_type == "OST_STEP":
                # printcyan(f"当前节点深度：{root.depth},r2的值{root._r2}")
                # printgreen(f"{root.question}")
                # printyellow(f"{answer}")
                # printred(f"parent: {root.parent},ancestor_node: {root.ancestor_node}")
                run_list.append(root.solution_trace)
                
            for child in root.children:
                # print(child.node_type)
                queue.put(child)

        print(len(run_list))
        
        for j,listx in tqdm(enumerate(run_list)):
            
            question = data[i]["question"]
            answer = get_ost_prompt(listx)
            # plan_prompt = data[i]["question"]
            # plan_prompt = "Haley went to the store. She bought 3+9/10 pounds of pumpernickel bread crumbs. How much did she spend? (Unit: $)"
            plan_prompt = f"""
            Judge whether the given answer solves the problem.output yes or no.
            **Question**
            {question}
            
            **Answer**
            {answer}
            """
            
            image_url = os.path.join(image_dir,f"{int(i)}.jpg")
            
            # for i in plan_prompt:
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": image_url,
                        },
                        {"type": "text", "text": plan_prompt},
                    ],
                },
            ]
            
            processor = AutoProcessor.from_pretrained(model_path,use_fast=False)
            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            
            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs
            if video_inputs is not None:
                mm_data["video"] = video_inputs

            llm_inputs = {
                "prompt": prompt,
                "multi_modal_data": mm_data,
            }
            
            outputs = llm.generate([llm_inputs], sampling_params=sampling_params, use_tqdm=False)
            
            win = no = 0
            is_win = False
            
            for idx_,output in enumerate(outputs[0].outputs):
                if "yes" in output.text.lower():
                    win += 1
                else:
                    no += 1
            
            if win > no:
                is_win = True
            
            save_data[i].update({
                f"{j + 1}" : {
                    "is_win" : is_win,
                    "solution" : answer,
                }
            })
            
            with open("test.json", "w") as f:
                json.dump(save_data, f, ensure_ascii=False, indent=4)
                
            
                # if idx_ % 2:
                #     printgreen(f"{output.text}")
                # else:
                #     printblue(f"{output.text}")


def onevllm():
    model_path = "/home/jncsnlp4/SSD2/model/qwen2-vl-intruct"
    # model_path = "/home/jncsnlp4/SSD2/model/Qwen3-VL-8B-Instruct"
    llm = LLM(
        model=model_path,
        limit_mm_per_prompt = {"image": 3, "video": 1}, # 每个提示最多包含的图片和视频数量
        max_model_len = 16384,
        max_num_seqs = 64,
        gpu_memory_utilization = 0.85
    )
    
    processor = AutoProcessor.from_pretrained(model_path,use_fast=False)
    
    printpink("模型加载完毕!")

    return_num = 11
    
    sampling_params = SamplingParams(
        temperature=0.7,  # 稍微增加随机性，有助于获取更合理的概率
        top_p=0.95,
        repetition_penalty=1.05,
        max_tokens=1280,  # 允许生成更多token
        # logprobs=5     # 获取前5个可能的token及其概率
        n = return_num
    )
    
    IMAGE_DIR_CONFIG = {
        "mathvision": "/home/jncsnlp4/SSD2/ZJJ/VReST/data/MATH-V/images",
        "mathvista": "/home/jncsnlp4/SSD2/ZJJ/VReST/data/MathVista/data/images/testmini/ALL", 
        "charxiv": "/home/jncsnlp4/SSD2/ZJJ/VReST/data/CharXiv/images"
    }
    

    mathvision_qa_score_file = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=qa.mathvision.ALL/data=mathvision/llm=Qwen2-VL-7B-Instruct/prompt_method=qa/ext_llm=qwen25/method=ans_and_score/field=max_vote_terminal/n=10/reward_method=self_consistency/scores.json"
    mathvision_baseline_score_file = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=baseline/data=mathvision/llm=Qwen2-VL-7B-Instruct/prompt_method=mctsv8/old-outcomes/ext_llm=Qwen2.5-7B-Instruct/method=ans_and_score/field=max_vote_terminal/n=10/reward_method=self_consistency/scores.json"
    mathvista_qa_score_file = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=qa.mathvista.ALL/data=mathvista/llm=Qwen2-VL-7B-Instruct/prompt_method=qa/ext_llm=qwen25/method=ans_and_score/field=max_vote_terminal/n=10/reward_method=self_consistency/extraction.json"
    mathvista_baseline_score_file = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=baseline/data=mathvista/llm=Qwen2-VL-7B-Instruct/prompt_method=mctsv8/ext_llm=qwen25/method=ans_and_score/field=max_vote_terminal/n=10/reward_method=self_consistency/extraction.json"
    charxiv_qa_score_file = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=qa.charxiv.ALL/data=charxiv/llm=Qwen2-VL-7B-Instruct/prompt_method=qa/ext_llm=qwen25/method=ans_and_score/field=max_vote_terminal/n=10/reward_method=self_consistency/scores-reasoning_val.json"
    charxiv_baseline_score_file = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=baseline/data=charxiv/llm=Qwen2-VL-7B-Instruct/prompt_method=mctsv8/old-outcomes/ext_llm=Qwen2.5-7B-Instruct/method=ans_and_score/field=max_vote_terminal/n=10/reward_method=self_consistency/scores-reasoning_val.json"
    
    with open(mathvision_qa_score_file, "r") as f:
        mathvision_qa_score = json.load(f)
    with open(mathvision_baseline_score_file, "r") as f:
        mathvision_baseline_score = json.load(f)
    
    with open(mathvista_qa_score_file, "r") as f:
        mathvista_qa_score = json.load(f)
    with open(mathvista_baseline_score_file, "r") as f:
        mathvista_baseline_score = json.load(f)
    
    with open(charxiv_qa_score_file, "r") as f:
        charxiv_qa_score = json.load(f)
    with open(charxiv_baseline_score_file, "r") as f:
        charxiv_baseline_score = json.load(f)
    
    mathvision_data = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=baseline/data=mathvision/llm=Qwen2-VL-7B-Instruct/prompt_method=mctsv8/ori_generation.json"
    mathvista_data = "/home/jncsnlp4/SSD2/ZJJ/VReST/data/MathVista/data/testmini.json"
    charxiv_data = "/home/jncsnlp4/SSD2/ZJJ/VReST/data/CharXiv/data/reasoning_val.json"
    with open(mathvision_data, "r") as f:
        mathvision_data = json.load(f)
    with open(mathvista_data, "r") as f:
        mathvista_data = json.load(f)
    with open(charxiv_data, "r") as f:
        charxiv_data = json.load(f)
    
    mathvision_image_dir = "/home/jncsnlp4/SSD2/ZJJ/VReST/data/MATH-V/images"
    mathvista_image_dir = "/home/jncsnlp4/SSD2/ZJJ/VReST/data/MathVista/data/images/testmini/ALL"
    charxiv_image_dir = "/home/jncsnlp4/SSD2/ZJJ/VReST/data/CharXiv/images"
    
    data = {
        "mathvision": mathvision_data,
        "mathvista": mathvista_data,
        "charxiv": charxiv_data
    }
    
    data_socore = {
        "mathvision": mathvision_qa_score,
        "mathvista": mathvista_qa_score,
        "charxiv": charxiv_qa_score
    }
        
    data_baseline_score = {
        "mathvision": mathvision_baseline_score,
        "mathvista": mathvista_baseline_score,
        "charxiv": charxiv_baseline_score
    }
    
    task = "charxiv" # mathvision mathvista charxiv 
    # prompt = """
    # If you think this problem is very simple and does not require a long time of reasoning, output yes or no.
    
    # """
    
    prompt = "If this QA pair is difficult and requires a long time of reasoning to solve, output no; otherwise, output yes." 
    
    qa_list = []
    qa_true_list = []
    qa_false_list = []
    
    error_list = []
    
    acc = loss = tol = 0
    acc2 = 0
    
    time_start = time.time()
    
    for i,j in tqdm(data[task].items(), desc="\033[31m判断简单问题\033[0m", total=len(data[task])):
        
        try:
            if task == "charxiv":
                question = j["query"]
            else:
                question = j["question"]
            
            image_url = os.path.join(IMAGE_DIR_CONFIG[task],f"{int(i)}.jpg")
            
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": image_url,
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            
            inputs = [prepare_inputs_for_vllm(message, processor) for message in [messages]]
            outputs = llm.generate(inputs, sampling_params)
            yes = no = 0
            for output in outputs[0].outputs:
                if "yes" in output.text.lower():
                    yes += 1
                else:
                    no += 1
            # printgreen(f"yes: {yes}, no: {no}")
            if yes > no and yes > 8:
                tol += 1
                if data_socore[task][i]["llm_score"]:
                    acc += 1
                    qa_true_list.append(i)
                else:
                    loss += 1
                    qa_false_list.append(i)
                    
                qa_list.append(i)
                printyellow(f"acc: {acc} / {loss} = {acc / (loss)}")
                
            else:
                if data_baseline_score[task][i]["llm_score"]:
                    acc2 += 1
                
        except Exception as e:
            error_list.append(i)
            if "llm_score" in data_baseline_score[task][i] and data_baseline_score[task][i]["llm_score"]:
                acc2 += 1
            printred(f"{e}")
    
    print(len(qa_list))
    
    print(f"qa_list: {qa_list}, len: {len(qa_list)}")
    print(f"qa_true_list: {qa_true_list}, len: {len(qa_true_list)}")
    print(f"qa_false_list: {qa_false_list}, len: {len(qa_false_list)}")
    
    print(f"tol: {tol}")
    print(f"error: {error_list}, len: {len(error_list)}")
    
    printblue(f"acc / loss = {acc} / {loss} = {acc / (loss)}")
    
    printyellow(f"acc: {acc + acc2} / {len(data[task])} = {(acc + acc2) / (len(data[task]))}")
    
    time_end = time.time()
    printpurple(f"所花的分钟数量: {(time_end - time_start) / 60}")
    
    while 1:
        pass
    
# vllm()
# qvllm()
onevllm()