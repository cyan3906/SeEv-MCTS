import math
import numpy as np
import itertools
import threading
import collections
import omegaconf
from openai import OpenAI
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.utils import FlexibleArgumentParser
from PIL import Image
import json,re
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from openai import AsyncOpenAI

# from prompt_methods.cantor.cantor import cantor_forward
# from prompt_methods.ours.our import our_forward
# from prompt_methods.ours1.chartqa_mcts import our1_forward
# from prompt_methods.ours2.chartqa_mcts import our2_forward
from prompt_methods.ours2v2.chartqa_mcts import our2v2_forward
# from prompt_methods.ours2v3.chartqa_mcts import our2v3_forward
# from prompt_methods.ours2v4.chartqa_mcts import our2v4_forward
# from prompt_methods.ours2v5.chartqa_mcts import our2v5_forward
# from prompt_methods.ours3.chartqa_mcts import our3_forward
# from prompt_methods.ours4.chartqa_mcts import our4_forward
# from prompt_methods.ours.our5 import our5_forward
# from prompt_methods.ours.our6 import our6_forward
from prompt_methods.cot.cot import cot_forward
from prompt_methods.qa.qa import qa_forward
from prompt_methods.tot.tot import tot_forward

# from prompt_methods.mcts.chartqa_mcts import mcts_forward
# from prompt_methods.mctsv2.chartqa_mcts import mcts_forward_v2
# from prompt_methods.mctsv3.chartqa_mcts import mcts_forward_v3
# from prompt_methods.mctsv4.chartqa_mcts import mcts_forward_v4
# from prompt_methods.mctsv5.chartqa_mcts import mcts_forward_v5
# from prompt_methods.mctsv6.chartqa_mcts import mcts_forward_v6
# from prompt_methods.mctsv7.chartqa_mcts import mcts_forward_v7
from prompt_methods.mctsv8.chartqa_mcts import mcts_forward_v8
from prompt_methods.mctsv8_vpw.chartqa_mcts import mcts_forward_vpw
# from prompt_methods.mctsv8v2.chartqa_mcts import mcts_forward_v8v2
# from prompt_methods.mctsv8v3.chartqa_mcts import mcts_forward_v8v3

from prompt_methods.directQ.run import directQ

from sentence_transformers import SentenceTransformer # 加载嵌入模型用的

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor,AutoModelForCausalLM
from vllm import LLM, SamplingParams
from tools.color_print import printred,printgreen,printyellow,printblue,printcyan,printpurple
from utils.prompts.scoring_inst import USER_SCORING_INST,SYSTEM_SCORING_INST,SCORING_INST
from utils.extract_answer import get_majority_answer
# from is_answer_consistent import is_answer_consistent_prompt

# class Extract_Answer(BaseModel):
#     extracted_answer: str
#     score: int

import json
from typing import Optional
from pydantic import BaseModel  # 如果使用 Pydantic 模型

NUM_PLACEHOLDERS = 8  # 建议设为 64 或 32，128 可能太长导致漂移
PLACEHOLDER_TOKEN = " ." # 使用 " ." 作为占位符，这对视觉模型更自然
        
def prepare_inputs_for_vllm(messages, processor, continue_final_message=False):
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=not continue_final_message,
        continue_final_message=continue_final_message,
    )
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

def is_extract_answer_yes_no(outputs,ground_truth,model_response,is_double_existing_reasoning_verify=False,is_double_answer_verify=False,is_existing_reasoning_verify=False):
    try:
        pattern = r'"extracted_answer": "([^"]+)",\n\s*"score": (\d+)'
        generated_text = []
        yes = no = 0
        for output in outputs[0].outputs:
            # print(output.text)
            res = re.search(pattern, output.text)
            if res:
                extracted_answer = res.group(1)  # 提取 答案内容
                score = int(res.group(2))
                if score == 1:
                    yes += 1
                else:
                    no += 1
                # print(extracted_answer, score)
                save_logger(model_response,ground_truth,extracted_answer,score,is_double_existing_reasoning_verify,is_double_answer_verify,is_existing_reasoning_verify) # 存入日子
        
                generated_text.append({"extracted_answer": extracted_answer, "score": score})
            
        # print(generated_text)
        # print(f"yes:{yes},no:{no}")
        majority_answer , score = get_majority_answer(generated_text)
        # print(f"majority_answer: {majority_answer}, score: {score}")
        return majority_answer , yes/(yes+no) , generated_text
    except Exception as e:
        printred(f"Error in is_extract_answer_yes_no: {e}")
        return None, 0, []
    
def save_logger(text,target_answer,extract_answer,score,
                 is_double_existing_reasoning_verify = False,
                 is_double_answer_verify = False,
                 is_existing_reasoning_verify = False
                 ):
    try:
        with open("mllms/extract_answer.log", "a") as f:
            if is_double_existing_reasoning_verify:
                f.write("double existing reasoning verify\n")
            elif is_double_answer_verify:
                f.write("double answer verify\n")
            elif is_existing_reasoning_verify:
                f.write("existing reasoning verify\n")
            
            f.write(f"初始答案：{text}\n")
            f.write(f"提取答案：{extract_answer}\n")
            f.write(f"目标答案：{target_answer}\n")
            if score == 1:
                f.write("提取答案正确\n")
            else:
                f.write("提取答案错误\n")
                
            f.write("\n")
            
    except Exception as e:
        printred(f"Error in save_logger: {e}")
# text only model

class Llama_Model():
    def __init__(self, llm_cfg):

        self.llm_cfg = llm_cfg
        
        if llm_cfg.call_method == "api":
            self.client = OpenAI(
                api_key=llm_cfg.api_key,
                base_url=llm_cfg.base_url,
            )
        
        elif llm_cfg.call_method == "local":
            self.llm = LLM(
                model=llm_cfg.path,
                max_model_len = llm_cfg.max_input_len,
                max_num_seqs = 512,
                gpu_memory_utilization = 0.85
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(llm_cfg.path)
            
        else:
            raise ValueError("Invalid call method")
            
    def get_response(self, messages=None, json_format=None,is_evaluation=False):

        if self.llm_cfg.call_method == "api":

            if json_format is not None:
            
                chat_response = self.client.beta.chat.completions.parse(
                    model=self.llm_cfg.name,
                    messages=messages,
                    response_format=json_format if json_format else None,
                    temperature=self.llm_cfg.temperature,
                    max_tokens=self.llm_cfg.max_output_len,
                    extra_body={
                        "repetition_penalty": 1.05,
                    },
                )

            else:
                    
                chat_response = self.client.chat.completions.create(
                    model=self.llm_cfg.name,
                    messages=messages,
                    temperature=self.llm_cfg.temperature,
                    max_tokens=self.llm_cfg.max_output_len,
                    extra_body={
                        "repetition_penalty": 1.05,
                    },
                )

            response = {
                "response": chat_response.choices[0].message.content
            }

            return response
        
        elif self.llm_cfg.call_method == "local":
            sampling_params = SamplingParams(
                temperature=self.llm_cfg.temperature,
                top_p=self.llm_cfg.top_p,
                # repetition_penalty=1.05,
                max_tokens=self.llm_cfg.max_output_len,
                stop_token_ids=[],
                n=self.llm_cfg.n,
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
            pattern = r'"extracted_answer": "([^"]+)"'
            for output in output[0].outputs:
                if is_evaluation: # 如果是评估模式的话就直接把输出文本添加到列表中 不需要re进行操作
                    generated_text.append(output.text)
                    continue
                res = re.search(pattern, output.text)
                # print(res)
                if res:
                    extracted_answer = res.group(1)  # 提取 答案内容
                    # score = int(res.group(2))
                    # print(extracted_answer, score)
                    generated_text.append({"extracted_answer": extracted_answer})
            

            response = {
                "response": generated_text
            }
            
            return response
        
        else:
            raise ValueError("Invalid call method")
    
    
class Qwen25_Model():
    def __init__(self, llm_cfg, cfg=None):

        self.llm_cfg = llm_cfg
        self.cfg = cfg
        
        if llm_cfg.call_method == "api":
            if isinstance(llm_cfg.base_url, omegaconf.listconfig.ListConfig):
                clients = []
                for base_url in llm_cfg.base_url:
                    clients.append(OpenAI(
                        api_key=llm_cfg.api_key,
                        base_url=base_url,
                    ))
                self.all_clients = clients
                # self.client = itertools.cycle(self.all_clients)
            else:
                self.client = OpenAI(
                    api_key=llm_cfg.api_key,
                    base_url=llm_cfg.base_url,
                )
                self.all_clients = [self.client]
                # self.client = itertools.cycle(self.all_clients)
            # 定义一个锁
            self.lock = threading.Lock()
            # 定义一个字典记录每个client的请求次数
            self.request_count = collections.Counter()
            
            self.attempt_num = 2*len(self.all_clients)
        elif llm_cfg.call_method == "local":
            self.llm = LLM(
                model=llm_cfg.path,
                max_model_len=llm_cfg.max_input_len,
                max_num_seqs=getattr(llm_cfg, "max_num_seqs", 512),
                gpu_memory_utilization=getattr(
                    llm_cfg, "gpu_memory_utilization", 0.85
                ),
                enable_prefix_caching=getattr(
                    getattr(cfg, "prompt_method", None),
                    "is_kv_cache",
                    False,
                ),
            )
            self.tokenizer = self.llm.get_tokenizer()
        else:
            raise ValueError("Invalid call method")

    def _prepare_text_prompts(
        self,
        messages_batch,
        continue_final_message=False,
    ):
        return [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=not continue_final_message,
                continue_final_message=continue_final_message,
            )
            for messages in messages_batch
        ]

    def get_response_batch(
        self,
        messages_batch,
        n=1,
        continue_final_message=False,
        stop_token=None,
    ):
        if not messages_batch:
            return []
        if self.llm_cfg.call_method != "local":
            return [
                self.get_response(
                    messages=messages,
                    n=n,
                    continue_final_message=continue_final_message,
                    stop_token=stop_token,
                )["response"]
                for messages in messages_batch
            ]

        sampling_params = SamplingParams(
            temperature=self.llm_cfg.temperature,
            top_p=self.llm_cfg.top_p,
            max_tokens=self.llm_cfg.max_output_len,
            stop_token_ids=[],
            stop=stop_token,
            n=n,
        )
        prompts = self._prepare_text_prompts(
            messages_batch,
            continue_final_message=continue_final_message,
        )
        outputs = self.llm.generate(
            prompts,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        return [
            [candidate.text.strip() for candidate in output.outputs]
            for output in outputs
        ]

    def get_reward_batch(
        self,
        messages_batch,
        n=1,
        node_depth=8,
        is_p3pro=False,
        is_p3=False,
    ):
        if not messages_batch:
            return []
        if is_p3 or is_p3pro:
            raise ValueError("Qwen2.5 text VPW does not enable P3/P3Pro")
        if self.cfg.reward_method != "self_consistency":
            raise ValueError(
                f"Unsupported batched reward method: {self.cfg.reward_method}"
            )

        responses = self.get_response_batch(messages_batch, n=n)
        rewards = []
        for response_group in responses:
            num_yes = sum("yes" in text.lower() for text in response_group)
            num_no = sum("no" in text.lower() for text in response_group)
            rewards.append(
                {
                    "yes_prob": num_yes / n,
                    "no_prob": num_no / n,
                }
            )
        return rewards

    def get_reward(
        self,
        messages=None,
        json_format=None,
        n=1,
        continue_final_message=False,
        top_logprobs=5,
        node_depth=8,
        is_p3=False,
        is_p3pro=False,
        **kwargs,
    ):
        return self.get_reward_batch(
            [messages],
            n=n,
            node_depth=node_depth,
            is_p3=is_p3,
            is_p3pro=is_p3pro,
        )[0]

    def get_least_request_client(self):
        for client in self.all_clients:
            if self.request_count[client] == 0:
                return client
        return min(self.request_count, key=self.request_count.get)
    
    def get_yes_or_no(self, messages=None):
        
        for attempt in range(self.attempt_num):
            try:
                with self.lock:
                    client = self.get_least_request_client()
                    self.request_count[client] += 1
                
                chat_response = client.chat.completions.create(
                    n=self.llm_cfg.n,
                    model=self.llm_cfg.name,
                    messages=messages,
                    temperature=self.llm_cfg.temperature,
                    top_p=self.llm_cfg.top_p,
                    max_tokens=self.llm_cfg.max_output_len,
                    extra_body={
                        "guided_choice": ["Yes", "No"],
                    },
                )
            
                response = {
                    "response": [choice.message.content for choice in chat_response.choices] if self.llm_cfg.n > 1 else chat_response.choices[0].message.content
                }
                
                with self.lock:
                    self.request_count[client] -= 1

                return response
            
            except Exception as e:
                print(f"Error: {str(e)}")
                if attempt == self.attempt_num-1:
                    print(f"Failed to get response after {self.attempt_num} attempts. Error: {str(e)}")
                    raise ValueError(f"Failed to get response after {self.attempt_num} attempts. Error: {str(e)}")

    def get_response(
        self,
        messages=None,
        json_format=None,
        is_evaluation=False,
        n=1,
        continue_final_message=False,
        top_logprobs=0,
        stop_token=None,
        **kwargs,
    ):

        if self.llm_cfg.call_method == "api":
            for attempt in range(self.attempt_num):
                try:
                    with self.lock:
                        client = self.get_least_request_client()
                        self.request_count[client] += 1

                    if json_format is not None:
                        
                        json_schema = json_format.model_json_schema()
                        
                        chat_response = client.chat.completions.create(
                            n=self.llm_cfg.n,
                            model=self.llm_cfg.name,
                            messages=messages,
                            temperature=self.llm_cfg.temperature,
                            top_p=self.llm_cfg.top_p,
                            max_tokens=self.llm_cfg.max_output_len,
                            extra_body={
                                "repetition_penalty": 1.05,
                                # "guided_decoding_backend": "outlines",
                                "guided_json": json_schema,
                            },
                        )
                        
                        # chat_response = client.beta.chat.completions.parse(
                        #     n=self.llm_cfg.n,
                        #     model=self.llm_cfg.name,
                        #     messages=messages,
                        #     response_format=json_format if json_format else None,
                        #     temperature=self.llm_cfg.temperature,
                        #     top_p=self.llm_cfg.top_p,
                        #     max_tokens=self.llm_cfg.max_output_len,
                        #     extra_body={
                        #         "repetition_penalty": 1.05,
                        #     },
                        # )

                    else:
                    
                        chat_response = client.chat.completions.create(
                            n=self.llm_cfg.n,
                            model=self.llm_cfg.name,
                            messages=messages,
                            temperature=self.llm_cfg.temperature,
                            top_p=self.llm_cfg.top_p,
                            max_tokens=self.llm_cfg.max_output_len,
                            extra_body={
                                "repetition_penalty": 1.05,
                            },
                        )
            
                    response = {
                        "response": [choice.message.content for choice in chat_response.choices] if self.llm_cfg.n > 1 else chat_response.choices[0].message.content
                    }

                    with self.lock:
                        self.request_count[client] -= 1

                    return response
                
                except Exception as e:
                    print(f"Error: {str(e)}")
                    if attempt == self.attempt_num-1:
                        print(f"Failed to get response after {self.attempt_num} attempts. Error: {str(e)}")
                        raise ValueError(f"Failed to get response after {self.attempt_num} attempts. Error: {str(e)}")
        elif self.llm_cfg.call_method == "local":
            return {
                "response": self.get_response_batch(
                    [messages],
                    n=n,
                    continue_final_message=continue_final_message,
                    stop_token=stop_token,
                )[0]
            }
            
        else:
            raise ValueError("Invalid call method")

class Qwen3_Model():
    def __init__(self, llm_cfg):

        self.llm_cfg = llm_cfg
        
        if llm_cfg.call_method == "api":
            if isinstance(llm_cfg.base_url, omegaconf.listconfig.ListConfig):
                clients = []
                for base_url in llm_cfg.base_url:
                    clients.append(OpenAI(
                        api_key=llm_cfg.api_key,
                        base_url=base_url,
                    ))
                self.all_clients = clients
                # self.client = itertools.cycle(self.all_clients)
            else:
                self.client = OpenAI(
                    api_key=llm_cfg.api_key,
                    base_url=llm_cfg.base_url,
                )
                self.all_clients = [self.client]
                # self.client = itertools.cycle(self.all_clients)
            # 定义一个锁
            self.lock = threading.Lock()
            # 定义一个字典记录每个client的请求次数
            self.request_count = collections.Counter()
            
            self.attempt_num = 2*len(self.all_clients)
        
        elif llm_cfg.call_method == "local":
            self.llm = LLM(
                model=llm_cfg.path,
                max_model_len = llm_cfg.max_input_len,
                max_num_seqs = 32,
                gpu_memory_utilization =0.9,
            )
            
            self.processor = AutoProcessor.from_pretrained(llm_cfg.path)

        else:
            raise ValueError("Invalid call method")
    def get_least_request_client(self):
        for client in self.all_clients:
            if self.request_count[client] == 0:
                return client
        return min(self.request_count, key=self.request_count.get)
    
    def get_yes_or_no(self, messages=None):
        
        for attempt in range(self.attempt_num):
            try:
                with self.lock:
                    client = self.get_least_request_client()
                    self.request_count[client] += 1
                
                chat_response = client.chat.completions.create(
                    n=self.llm_cfg.n,
                    model=self.llm_cfg.name,
                    messages=messages,
                    temperature=self.llm_cfg.temperature,
                    top_p=self.llm_cfg.top_p,
                    max_tokens=self.llm_cfg.max_output_len,
                    extra_body={
                        "guided_choice": ["Yes", "No"],
                    },
                )
            
                response = {
                    "response": [choice.message.content for choice in chat_response.choices] if self.llm_cfg.n > 1 else chat_response.choices[0].message.content
                }
                
                with self.lock:
                    self.request_count[client] -= 1

                return response
            
            except Exception as e:
                print(f"Error: {str(e)}")
                if attempt == self.attempt_num-1:
                    print(f"Failed to get response after {self.attempt_num} attempts. Error: {str(e)}")
                    raise ValueError(f"Failed to get response after {self.attempt_num} attempts. Error: {str(e)}")

    def get_response(self, messages=None, json_format=None,is_evaluation = False):

        if self.llm_cfg.call_method == "api":
            for attempt in range(self.attempt_num):
                try:
                    with self.lock:
                        client = self.get_least_request_client()
                        self.request_count[client] += 1

                    if json_format is not None:
                        
                        json_schema = json_format.model_json_schema()
                        
                        chat_response = client.chat.completions.create(
                            n=self.llm_cfg.n,
                            model=self.llm_cfg.name,
                            messages=messages,
                            temperature=self.llm_cfg.temperature,
                            top_p=self.llm_cfg.top_p,
                            max_tokens=self.llm_cfg.max_output_len,
                            extra_body={
                                "repetition_penalty": 1.05,
                                # "guided_decoding_backend": "outlines",
                                "guided_json": json_schema,
                            },
                        )
                        
                        # chat_response = client.beta.chat.completions.parse(
                        #     n=self.llm_cfg.n,
                        #     model=self.llm_cfg.name,
                        #     messages=messages,
                        #     response_format=json_format if json_format else None,
                        #     temperature=self.llm_cfg.temperature,
                        #     top_p=self.llm_cfg.top_p,
                        #     max_tokens=self.llm_cfg.max_output_len,
                        #     extra_body={
                        #         "repetition_penalty": 1.05,
                        #     },
                        # )

                    else:
                    
                        chat_response = client.chat.completions.create(
                            n=self.llm_cfg.n,
                            model=self.llm_cfg.name,
                            messages=messages,
                            temperature=self.llm_cfg.temperature,
                            top_p=self.llm_cfg.top_p,
                            max_tokens=self.llm_cfg.max_output_len,
                            extra_body={
                                "repetition_penalty": 1.05,
                            },
                        )
            
                    response = {
                        "response": [choice.message.content for choice in chat_response.choices] if self.llm_cfg.n > 1 else chat_response.choices[0].message.content
                    }

                    with self.lock:
                        self.request_count[client] -= 1

                    return response
                
                except Exception as e:
                    print(f"Error: {str(e)}")
                    if attempt == self.attempt_num-1:
                        print(f"Failed to get response after {self.attempt_num} attempts. Error: {str(e)}")
                        raise ValueError(f"Failed to get response after {self.attempt_num} attempts. Error: {str(e)}")
        elif self.llm_cfg.call_method == "local":
            
            sampling_params = SamplingParams(
                temperature=self.llm_cfg.temperature,
                top_p=self.llm_cfg.top_p,
                # repetition_penalty=1.05,
                max_tokens=1024,
                stop_token_ids=[],
                n=self.llm_cfg.n,
            )

            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking = self.llm_cfg.enable_thinking,
                # enable_thinking = False # 是否打开qwen3的思考模式
            )

            llm_inputs = {
                "prompt": prompt
            }
            
            outputs = self.llm.generate([llm_inputs], sampling_params=sampling_params, use_tqdm=False)
            pattern = r'"extracted_answer": "([^"]+)",\n\s*"score": (\d+)'
            generated_text = []
            for output in outputs[0].outputs:
                # print(output.text)
                print(output.text)
                res = re.search(pattern, output.text)
                if res:
                    extracted_answer = res.group(1)  # 提取 答案内容
                    # score = int(res.group(2))
                    # print(extracted_answer, score)
                    # generated_text.append({"extracted_answer": extracted_answer, "score": score})
                    generated_text.append(extracted_answer)
                else:
                    generated_text.append(output.text)
                    
            response = {
                "response": generated_text
            }
            
            # text = self.tokenizer.apply_chat_template(
            #     messages,
            #     tokenize=False,
            #     add_generation_prompt=True,
            # )
            # model_inputs = self.tokenizer([text], return_tensors="pt").to(self.llm.device)

            # # conduct text completion
            # generated_ids = self.llm.generate(
            #     **model_inputs,
            #     max_new_tokens=16384,
            #     temperature=self.llm_cfg.temperature,
            # )
            # output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

            # content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            # # print(content)
            # pattern = r'"extracted_answer": "([^"]+)",\n\s*"score": (\d+)'
            # res = re.search(pattern, content)
            # if res:
            #     extracted_answer = res.group(1)  # 提取 答案内容
            #     score = res.group(2)             # 提取 分数
            #     response = {
            #         "response": {"extracted_answer": extracted_answer, "score": score}
            #     }
            
            return response
            
        else:
            raise ValueError("Invalid call method")

# vistul model

class Qwen2_VL_Model():
    def __init__(self, llm_cfg, cfg):
        self.llm_cfg = llm_cfg
        self.cfg = cfg
        self.llm = LLM(
            model=llm_cfg.path,
            limit_mm_per_prompt={"image": 3, "video": 1}, # 每个提示最多包含的图片和视频数量
            max_model_len= llm_cfg.max_input_len,
            max_num_seqs=getattr(llm_cfg, "max_num_seqs", 512),
            gpu_memory_utilization=getattr(llm_cfg, "gpu_memory_utilization", 0.85),
            enable_prefix_caching = self.cfg.prompt_method.is_kv_cache, # 是否使用kv cache
            # disable_log_stats = False,
        )
        self.processor = AutoProcessor.from_pretrained(llm_cfg.path)
        self.tokenizer = self.llm.get_tokenizer()
    def get_extract_score(self,question, ground_truth, model_response,
                          is_double_existing_reasoning_verify = False,
                          is_double_answer_verify = False,
                          is_existing_reasoning_verify = False
                          ):
        try:
            prompt = SCORING_INST.format(question=question, ground_truth=ground_truth, model_response=model_response)
            messages = [
                {"role": "system", "content": SYSTEM_SCORING_INST}, 
                {"role": "user", "content": prompt}
            ]
            inputs = [prepare_inputs_for_vllm(message, self.processor) for message in [messages]]
            
            outputs = self.llm.generate(inputs, sampling_params=self.sampling_params, use_tqdm=False,)
            
            extract_answer,extract_score,generated_text = is_extract_answer_yes_no(outputs,ground_truth,model_response,is_double_existing_reasoning_verify,is_double_answer_verify,is_existing_reasoning_verify)
            
            return extract_score

        except Exception as e:
            printred(f"Error in get_extract_score: {e}")
            return 0

    def get_reward_batch(
        self,
        messages_batch,
        n=1,
        node_depth=8,
        is_p3pro=False,
        is_p3=False,
    ):
        """Score multiple reward prompts in one vLLM scheduler call."""
        if not messages_batch:
            return []

        inputs = [
            prepare_inputs_for_vllm(messages, self.processor)
            for messages in messages_batch
        ]

        for attempt in range(self.cfg.prompt_method.attempt_num):
            try:
                if is_p3:
                    sampling_params = SamplingParams(
                        max_tokens=1,
                        temperature=0.7,
                        prompt_logprobs=20,
                    )
                    outputs = self.llm.generate(
                        inputs, sampling_params=sampling_params, use_tqdm=False
                    )
                    rewards = []
                    candidates_yes = ["Yes", " Yes", "yes", " yes"]
                    candidates_no = ["No", " No", "no", " no"]
                    yes_ids = [
                        ids[0]
                        for candidate in candidates_yes
                        if len(ids := self.processor.tokenizer.encode(candidate, add_special_tokens=False)) == 1
                    ]
                    no_ids = [
                        ids[0]
                        for candidate in candidates_no
                        if len(ids := self.processor.tokenizer.encode(candidate, add_special_tokens=False)) == 1
                    ]
                    for output in outputs:
                        yes_scores = []
                        no_scores = []
                        for logprob_dict in output.prompt_logprobs[-NUM_PLACEHOLDERS:]:
                            if not logprob_dict:
                                continue
                            p_yes = max(
                                (
                                    value.logprob if hasattr(value, "logprob") else value
                                    for token_id, value in logprob_dict.items()
                                    if token_id in yes_ids
                                ),
                                default=-10000.0,
                            )
                            p_no = max(
                                (
                                    value.logprob if hasattr(value, "logprob") else value
                                    for token_id, value in logprob_dict.items()
                                    if token_id in no_ids
                                ),
                                default=-10000.0,
                            )
                            if p_yes == -10000.0 and p_no == -10000.0:
                                continue
                            prob_yes = np.exp(p_yes)
                            prob_no = np.exp(p_no)
                            total = prob_yes + prob_no + 1e-12
                            yes_scores.append(prob_yes / total)
                            no_scores.append(prob_no / total)
                        rewards.append(
                            {
                                "yes_prob": float(np.mean(yes_scores)) if yes_scores else 0.5,
                                "no_prob": float(np.mean(no_scores)) if no_scores else 0.5,
                            }
                        )
                    return rewards

                if is_p3pro:
                    sampling_params = SamplingParams(
                        max_tokens=1,
                        temperature=0.7,
                        logprobs=20,
                    )
                    outputs = self.llm.generate(
                        inputs, sampling_params=sampling_params, use_tqdm=False
                    )
                    rewards = []
                    for output in outputs:
                        token_logprobs = output.outputs[0].logprobs[0]
                        yes_p = no_p = 0.5
                        for token_id, logprob in sorted(
                            token_logprobs.items(),
                            key=lambda item: item[1].logprob,
                            reverse=True,
                        ):
                            token = self.tokenizer.decode([token_id]).strip().lower()
                            probability = math.exp(logprob.logprob)
                            if "yes" in token and yes_p == 0.5:
                                yes_p = probability
                            if "no" in token and no_p == 0.5:
                                no_p = probability
                        probabilities = torch.softmax(
                            torch.tensor([yes_p, no_p], dtype=torch.float32), dim=0
                        )
                        rewards.append(
                            {
                                "yes_prob": float(probabilities[0].item()),
                                "no_prob": float(probabilities[1].item()),
                            }
                        )
                    return rewards

                if self.cfg.reward_method != "self_consistency":
                    raise ValueError(
                        f"Unsupported batched reward method: {self.cfg.reward_method}"
                    )

                sampling_params = SamplingParams(
                    temperature=self.llm_cfg.temperature,
                    top_p=self.llm_cfg.top_p,
                    max_tokens=self.llm_cfg.max_output_len,
                    stop_token_ids=[],
                    n=n,
                )
                outputs = self.llm.generate(
                    inputs, sampling_params=sampling_params, use_tqdm=False
                )
                rewards = []
                for output in outputs:
                    response_text = [candidate.text.strip() for candidate in output.outputs]
                    num_yes = sum("yes" in text.lower() for text in response_text)
                    num_no = sum("no" in text.lower() for text in response_text)
                    rewards.append(
                        {
                            "yes_prob": num_yes / n,
                            "no_prob": num_no / n,
                        }
                    )
                return rewards
            except Exception as exc:
                print(f"Qwen2_VL_Model_get_reward_batch_Error: {str(exc)}")
                if attempt == self.cfg.prompt_method.attempt_num - 1:
                    raise

        raise RuntimeError("batched reward generation exhausted retries")
    
    def get_reward(
        self, messages=None, json_format=None, n=1, continue_final_message=False, top_logprobs=5,node_depth=8,
        is_double_existing_reasoning_verify = False,
        is_double_answer_verify = False,
        is_existing_reasoning_verify = False,
        is_rewrite_answer_verify = False,
        is_direct_answer_verify = False,
        question = None,
        question_id = None,
        answer1 = None , answer2 = None,answer3 = None,answer4 = None,
        is_p3 = False,
        is_p3pro = False
        ):
        
        # if answer1:
        #     answer1 = answer1.split("answer is")[-1].strip(". \n ").lower()
        # if answer2:
        #     answer2 = answer2.split("answer is")[-1].strip(". \n ").lower()
        if answer1: answer1 = answer1.strip(". , ! \n")
        if answer2: answer2 = answer2.strip(". , ! \n")
        if answer3: answer3 = answer3.strip(". , ! \n")
        if answer4: answer4 = answer4.strip(". , ! \n")
        
        for attempt in range(self.cfg.prompt_method.attempt_num):
            try:
                if is_p3:
                    text_prompt = messages[1]["content"][1]["text"]
                    final_prompt = text_prompt + PLACEHOLDER_TOKEN * NUM_PLACEHOLDERS
                    inputs = [prepare_inputs_for_vllm(messages, self.processor)]
                    sampling_params = SamplingParams(
                        max_tokens=1,
                        temperature=0.7,
                        # logprobs=20,
                        prompt_logprobs=20,
                    )
                    outputs = self.llm.generate(inputs, sampling_params=sampling_params, use_tqdm=False)

                    # print("raw_output:", outputs[0].outputs[0].text)

                    prompt_logprobs = outputs[0].prompt_logprobs

                    yes_scores = []
                    no_scores = []

                    # 3.5 分析最后 N 个位置的 Logprobs
                    valid_logprobs = prompt_logprobs[-NUM_PLACEHOLDERS:]
                    # print(f"len(valid_logprobs): {len(valid_logprobs)}")
                    # print(f"valid_logprobs: {valid_logprobs}")
                    yes_ids = []
                    no_ids = []
                    candidates_yes = ["Yes", " Yes", "yes", " yes"]
                    candidates_no = ["No", " No", "no", " no"]

                    for c in candidates_yes:
                        ids = self.processor.tokenizer.encode(c, add_special_tokens=False)
                        if len(ids) == 1: yes_ids.append(ids[0])
                            
                    for c in candidates_no:
                        ids = self.processor.tokenizer.encode(c, add_special_tokens=False)
                        if len(ids) == 1: no_ids.append(ids[0])
                            
                    for logprob_dict in valid_logprobs:
                        if not logprob_dict: continue
                        
                        p_yes = -10000.0
                        for vid in yes_ids:
                            if vid in logprob_dict:
                                # 【修复】兼容 Logprob 对象和 float
                                obj = logprob_dict[vid]
                                val = obj.logprob if hasattr(obj, 'logprob') else obj
                                p_yes = max(p_yes, val)
                                
                        p_no = -10000.0
                        for vid in no_ids:
                            if vid in logprob_dict:
                                # 【修复】兼容 Logprob 对象和 float
                                obj = logprob_dict[vid]
                                val = obj.logprob if hasattr(obj, 'logprob') else obj
                                p_no = max(p_no, val)

                        if p_yes == -10000.0 and p_no == -10000.0:
                            continue
                            
                        try:
                            prob_yes = np.exp(p_yes)
                            prob_no = np.exp(p_no)
                            total = prob_yes + prob_no + 1e-12
                            yes_scores.append(prob_yes / total)
                            no_scores.append(prob_no / total)
                        except:
                            pass
                        
                    avg_yes = np.mean(yes_scores)
                    avg_no = np.mean(no_scores)
                    
                    response = {
                        "yes_prob": avg_yes,
                        "no_prob": avg_no,
                    }
                    
                    return response 
                    
                elif is_p3pro:
                    inputs = [prepare_inputs_for_vllm(messages, self.processor)]
                    
                    sampling_params = SamplingParams(
                        max_tokens=1,
                        temperature=0.7,
                        logprobs=20,
                    )

                    yes_probs = []
                    no_probs = []

                    # print("我进来了")
                    for _ in range(1): # 10 1
                        outputs = self.llm.generate(
                            inputs,
                            sampling_params=sampling_params,
                            use_tqdm=False,
                        )
                        out = outputs[0].outputs[0]
                        
                        th_token_logprobs = out.logprobs[0] # 这里下标为0很关键
                        sorted_items = sorted(
                            th_token_logprobs.items(),
                            key=lambda x: x[1].logprob,
                            reverse=True
                        )
                        # print(sorted_items)
                        # 提前存入
                        listx_token_str = []
                        listx_logprob = []
                        for z in sorted_items:
                            token_id, logprob = z
                            token_str = self.tokenizer.decode([token_id])
                            listx_token_str.append(token_str)
                            listx_logprob.append(logprob)
                        
                        yes_exist = no_exist = False
                        yes_p = 0.5
                        no_p = 0.5
                        for _idx in range(len(sorted_items)):
                            if yes_exist and no_exist: # yes和no都已经获取了 跳出
                                break
                
                            logprob = listx_logprob[_idx].logprob
                            token_str = listx_token_str[_idx]
                            # print(logprob, token_str)
                            # input("点击继续")
                            if "yes" in token_str.strip().lower():
                                if not yes_exist:
                                    yes_p = math.exp(logprob)
                                    yes_exist = True
                                    
                            if "no" in token_str.strip().lower():
                                if not no_exist:
                                    no_p = math.exp(logprob)
                                    no_exist = True
                        
                        yes_p = float(yes_p)
                        no_p  = float(no_p)
                        logits = torch.tensor([yes_p, no_p], dtype=torch.float32)
                        probs = torch.softmax(logits, dim=0)
                        yes_probs.append(probs[0].item())
                        no_probs.append(probs[1].item())
                        # print(yes_probs, no_probs)
                        # input("点击输入")
                    final_yes_prob = np.mean(yes_probs)
                    fianl_no_prob = np.mean(no_probs)
                    
                    response = {
                        "yes_prob": final_yes_prob,
                        "no_prob": fianl_no_prob,
                    }
                    
                    # printblue(f"进入p3pro-{response}")
                    return response 
                    
                else:
                
                    inputs = [prepare_inputs_for_vllm(message, self.processor) for message in [messages]]
                    
                    if self.cfg.reward_method == "self_consistency":
                
                        self.sampling_params = SamplingParams(
                            temperature=self.llm_cfg.temperature,
                            top_p=self.llm_cfg.top_p,
                            # repetition_penalty=1.05,
                            max_tokens=self.llm_cfg.max_output_len,
                            stop_token_ids=[],
                            n = n,
                        )
                        outputs = self.llm.generate(inputs, sampling_params=self.sampling_params, use_tqdm=False)
                        response_text = [output.text.strip() for output in outputs[0].outputs]
                        # print(response_text)
                        num_yes = num_no = 0
                        
                        if is_double_existing_reasoning_verify or is_double_answer_verify:
                            win = 0
                            game = 5
                            for _ in range(game): # 三局两胜制度
                                num_yes = num_no = 0
                                for text in response_text:
                                    text = str(text).strip(". , ! \n")
                                    if answer3 and answer4: # solution PK
                                        extract_score = self.get_extract_score(question, answer3, text,is_double_existing_reasoning_verify = True)
                                        if extract_score:
                                            num_yes += 1
                                        extract_score = self.get_extract_score(question, answer4, text,is_double_existing_reasoning_verify = True)
                                        if extract_score:
                                            num_no += 1
                                        
                                    else: # answer pk
                                        extract_score = self.get_extract_score(question, answer1, text,is_double_answer_verify = True)
                                        if extract_score:
                                            num_yes += 1
                                        extract_score = self.get_extract_score(question, answer2, text,is_double_answer_verify = True)
                                        if extract_score:
                                            num_no += 1
                                        
                                if num_yes > num_no: win += 1
                                    
                            if win >= game / 2 + 1: return {"yes_prob": True, "no_prob": 0.0}
                            else: return {"yes_prob": False, "no_prob": 1.0}
                                
                        
                        elif is_existing_reasoning_verify: # 掩码生成后续文本，判断是否与已有的推理一致
                            num_yes = num_no = 0
                            for text in response_text:
                                text = str(text).strip(". , ! \n")
                                if text == "": continue
                                
                                if text == answer1:
                                    num_yes += 1
                                    continue
                                
                                extract_score = self.get_extract_score(question, answer1, text,is_existing_reasoning_verify=True)
                                
                                if extract_score: num_yes += 1
                                else: num_no += 1
                                
                            response = {
                                "yes_prob": num_yes / (num_yes + num_no),
                            }
                            
                            return response
                        
                        else: # 单纯的PRM
                            for text in response_text:
                                if "yes" in text.lower():
                                    num_yes += 1
                                elif "no" in text.lower():
                                    num_no += 1
                            # print(num_yes, num_no)
                        
                            yes_prob = num_yes / n
                            no_prob = num_no / n
                            
                            response = {
                                "yes_prob": yes_prob,
                                "no_prob": no_prob,
                            }
                            
                            return response  
                    
                    
                
            except Exception as e:
                print(f"Qwen2_VL_Model_get_reward_Error: {str(e)}")
                if attempt == self.cfg.prompt_method.attempt_num-1:
                    print(f"Qwen2_VL_Model_get_reward_Failed to get reward after {self.cfg.prompt_method.attempt_num} attempts. Error: {str(e)}")
                    raise ValueError(f"Qwen2_VL_Model_get_reward_Failed to get reward after {self.cfg.prompt_method.attempt_num} attempts. Error: {str(e)}")

                return {"yes_prob": 0.5, "no_prob": 0.5}
            
    def get_response_batch(
        self,
        messages_batch,
        n=1,
        continue_final_message=False,
        stop_token=None,
    ):
        if not messages_batch:
            return []

        sampling_params = SamplingParams(
            temperature=self.llm_cfg.temperature,
            max_tokens=self.llm_cfg.max_output_len,
            top_p=self.llm_cfg.top_p,
            stop_token_ids=[],
            stop=stop_token,
            n=n,
        )
        continue_qwen25vl_message = (
            continue_final_message
            and getattr(self.llm_cfg, "name", "").startswith("Qwen2.5-VL-")
        )
        inputs = [
            prepare_inputs_for_vllm(
                messages,
                self.processor,
                continue_final_message=continue_qwen25vl_message,
            )
            for messages in messages_batch
        ]
        outputs = self.llm.generate(
            inputs,
            sampling_params,
            use_tqdm=False,
        )
        return [
            [candidate.text.strip() for candidate in output.outputs]
            for output in outputs
        ]

    def get_response(self, messages=None, json_format: Optional[BaseModel] = None, n=1, continue_final_message=False, top_logprobs=0, stop_token=None,
                     is_deduplicatev2 = False
                     ):
        
        if is_deduplicatev2:
            sampling_params = SamplingParams(
                temperature=1.0,
                max_tokens=self.llm_cfg.max_output_len,
                top_p=self.llm_cfg.top_p,
                stop_token_ids=[],
                stop=stop_token,
                n=n,
            )
        
        else:
            sampling_params = SamplingParams(
                temperature=self.llm_cfg.temperature,
                max_tokens=self.llm_cfg.max_output_len,
                top_p=self.llm_cfg.top_p,
                stop_token_ids=[],
                stop=stop_token,
                n=n,
            )
        
        continue_qwen25vl_message = (
            continue_final_message
            and getattr(self.llm_cfg, "name", "").startswith("Qwen2.5-VL-")
        )
        inputs = [
            prepare_inputs_for_vllm(
                message,
                self.processor,
                continue_final_message=continue_qwen25vl_message,
            )
            for message in [messages]
        ]
        outputs = self.llm.generate(inputs, sampling_params)
        
        # prefix cache 命中
        prompt_tokens = len(outputs[0].prompt_token_ids)
        cached_tokens = outputs[0].num_cached_tokens

        cache_hit_ratio = f"{cached_tokens / prompt_tokens:.04f}"
        
        with open("/home/jncsnlp4/SSD2/ZJJ/VReST/mllms/cache_hit_ratio.txt", "r") as f:
            old_cache_hit_ratio = float(f.readline().strip())
        
        if old_cache_hit_ratio != -1:
            new_cache_hit_ratio = float(f"{(float(cache_hit_ratio) + float(old_cache_hit_ratio)) / 2:.04f}")
        else:
            print(111111)
            new_cache_hit_ratio = float(cache_hit_ratio)
        
        with open("/home/jncsnlp4/SSD2/ZJJ/VReST/mllms/cache_hit_ratio.txt", "w") as f:
            f.write(str(new_cache_hit_ratio))
        
        print(f"\033[94mprompt_tokens:{prompt_tokens}\033[0m")
        print(f"\033[95mcached_tokens:{cached_tokens}\033[0m")
        print(f"\033[96mcache_hit_ratio:{cache_hit_ratio}\033[0m")
        
        # print(dir(self.llm.llm_engine))
        # print(1111111111,self.llm.llm_engine.get_metrics())
        # core = self.llm.llm_engine.engine_core

        # for name in dir(core):
        #     if "cache" in name.lower():
        #         print(name)
        
        # cache_stats = self.llm.llm_engine.cache_stats
        # print(f"\033[92m✅ 总缓存块: {cache_stats.num_gpu_blocks}\033[0m")
        # print(f"\033[93m✅ 已使用缓存: {cache_stats.num_used_gpu_blocks}\033[0m")
        # print(f"\033[91m✅ 空闲缓存: {cache_stats.num_free_gpu_blocks}\033[0m")
        
        # 处理json_format参数
        
        response = {
            "response": [output.text.strip() for output in outputs[0].outputs],
            # "cache_hit_ratio": cache_hit_ratio,
        }
        
        # print(f'生成了{len(response["response"])}个')
        return response
       

model_name_map = {
    "Qwen2-VL-7B-Instruct": Qwen2_VL_Model,
    "Qwen2-VL-2B-Instruct": Qwen2_VL_Model,
    "Qwen2.5-VL-3B-Instruct": Qwen2_VL_Model,
    "Qwen2.5-VL-7B-Instruct": Qwen2_VL_Model,
    "Qwen2.5-7B-Instruct": Qwen25_Model,
    "Qwen2.5-14B-Instruct": Qwen25_Model,
    "qwen25": Qwen25_Model,
    "Qwen3-8B": Qwen3_Model,
    "qwen3": Qwen3_Model,
    "Meta-Llama-3-8B-Instruct": Llama_Model,
}

class Model():
    def __init__(self, cfg):
        self.cfg = cfg
        self.prompt_cfg = cfg.prompt_method
        self.model = model_name_map[cfg.llm.name](cfg.llm,cfg)
        self.embedding_model = None
        if cfg.embedding_model_path != "none":
            self.embedding_model = SentenceTransformer(
                cfg.embedding_model_path,
                device=getattr(cfg, "embedding_device", "cpu"),
            )
        
    def get_response(self, input_text, image_path, is_pkl = False, tree_save_path=None):
        try:
            if self.prompt_cfg.name == "cantor":
                return cantor_forward(input_text, image_path, self.model)
            elif self.prompt_cfg.name == "cot":
                return cot_forward(input_text, image_path, self.model)
            elif self.prompt_cfg.name == "qa":
                return qa_forward(input_text, image_path, self.model)
            elif self.prompt_cfg.name == "our":
                return our_forward(input_text, image_path, self.model)
            
            elif self.prompt_cfg.name == "tot":
                return tot_forward(self.cfg, input_text, image_path, self.model)
            
            elif "our1" in self.prompt_cfg.name: # 这个是自己的方法 固定ost奖励分数为0.8
                return our1_forward(input_text, image_path, is_pkl, tree_save_path, self.model , self.embedding_model , self.cfg)
            
            elif "our2v2" in self.prompt_cfg.name: # 这个是自己的方法 让大模型去批判ost的分数
                return our2v2_forward(input_text, image_path, is_pkl, tree_save_path, self.model , self.cfg)
            
            elif "our2v3" in self.prompt_cfg.name: # 这个是自己的方法 让大模型去批判ost的分数 加入对答案的双重验证判断
                return our2v3_forward(input_text, image_path, is_pkl, tree_save_path, self.model , self.cfg)
            
            elif "our2v4" in self.prompt_cfg.name: # 这个是自己的方法 让大模型去批判ost的分数 只取best_answer 并查集
                return our2v4_forward(input_text, image_path, is_pkl, tree_save_path, self.model , self.cfg)
            
            elif "our2v5" in self.prompt_cfg.name: # 这个是自己的方法 让大模型去批判ost的分数 分类进行PK ansewr_tree
                return our2v5_forward(input_text, image_path, is_pkl, tree_save_path, self.model , self.cfg)
            
            elif "our2" in self.prompt_cfg.name: # 这个是自己的方法 让大模型去批判ost的分数
                return our2_forward(input_text, image_path, is_pkl, tree_save_path, self.model , self.embedding_model , self.cfg)
            
            
            elif "our3" in self.prompt_cfg.name: # 这个是自己的方法 子问题与ost的组合
                return our3_forward(input_text, image_path, is_pkl, tree_save_path, self.model , self.embedding_model , self.cfg)

            
            # elif "our4" in self.prompt_cfg.name:
            #     return our4_forward(input_text, image_path, self.model, self.cfg)
            # elif "our5" in self.prompt_cfg.name:
            #     return our5_forward(input_text, image_path, self.model, self.cfg)
            # elif "our6" in self.prompt_cfg.name:
            #     return our6_forward(input_text, image_path, self.model, self.cfg)
            # elif self.prompt_cfg.name in ["mcts", "mctsv1"]:
            #     return mcts_forward(input_text, image_path, self.model, self.cfg)
            # elif self.prompt_cfg.name in ["mctsv2"]:
            #     return mcts_forward_v2(input_text, image_path, self.model, self.cfg)
            # elif self.prompt_cfg.name in ["mctsv3"]:
            #     return mcts_forward_v3(input_text, image_path, self.model, self.cfg)
            # elif self.prompt_cfg.name in ["mctsv4"]:
            #     return mcts_forward_v4(input_text, image_path, self.model, self.cfg)
            # elif self.prompt_cfg.name in ["mctsv5"]:
            #     return mcts_forward_v5(input_text, image_path, self.model, self.cfg)
            # elif self.prompt_cfg.name in ["mctsv6"]:
            #     return mcts_forward_v6(input_text, image_path, self.model, self.cfg)
            # elif self.prompt_cfg.name in ["mctsv7"]:
            #     return mcts_forward_v7(input_text, image_path, self.model, self.cfg)
            elif self.prompt_cfg.name == "mctsv8_vpw":
                return mcts_forward_vpw(
                    input_text,
                    image_path,
                    is_pkl,
                    tree_save_path,
                    self.model,
                    self.cfg,
                    embedding_model=self.embedding_model,
                )

            elif self.prompt_cfg.name.startswith("mctsv8") and self.prompt_cfg.name not in ["mctsv8v2", "mctsv8v3"]:
                return mcts_forward_v8(
                    input_text,
                    image_path,
                    is_pkl,
                    tree_save_path,
                    self.model,
                    self.cfg,
                    embedding_model=self.embedding_model,
                )
            
            elif self.prompt_cfg.name in ["mctsv8v2"]:
                return mcts_forward_v8v2(input_text, image_path, self.model, self.cfg)
            
            elif self.prompt_cfg.name in ["mctsv8v3"]:
                return mcts_forward_v8v3(input_text, image_path, is_pkl, tree_save_path, self.model , self.cfg)
            
            elif self.prompt_cfg.name in ["directQ"]:
                return directQ(input_text, image_path, self.model, self.cfg)
            else:
                raise ValueError("Invalid prompt method")
        except Exception as e:
            print(e)
            raise e




