import math
import itertools
import threading
import collections
import omegaconf
from openai import OpenAI
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.utils import FlexibleArgumentParser
from PIL import Image
import json
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from openai import AsyncOpenAI

# from prompt_methods.cantor.cantor import cantor_forward
# from prompt_methods.ours.our import our_forward
# from prompt_methods.ours.our1 import our1_forward
# from prompt_methods.ours.our2 import our2_forward
# from prompt_methods.ours.our3 import our3_forward
# from prompt_methods.ours.our4 import our4_forward
# from prompt_methods.ours.our5 import our5_forward
# from prompt_methods.ours.our6 import our6_forward
# from prompt_methods.cot.cot import cot_forward
# from prompt_methods.qa.qa import qa_forward

from prompt_methods.mcts.chartqa_mcts import mcts_forward
from prompt_methods.mctsv2.chartqa_mcts import mcts_forward_v2
from prompt_methods.mctsv3.chartqa_mcts import mcts_forward_v3
from prompt_methods.mctsv4.chartqa_mcts import mcts_forward_v4
from prompt_methods.mctsv5.chartqa_mcts import mcts_forward_v5
from prompt_methods.mctsv6.chartqa_mcts import mcts_forward_v6
from prompt_methods.mctsv7.chartqa_mcts import mcts_forward_v7
from prompt_methods.mctsv8.chartqa_mcts import mcts_forward_v8

from prompt_methods.directQ.run import directQ

from sentence_transformers import SentenceTransformer # 加载嵌入模型用的


import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

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


# text only

class Llama_Model():
    def __init__(self, llm_cfg):

        self.llm_cfg = llm_cfg
        
        if llm_cfg.call_method == "api":
            self.client = OpenAI(
                api_key=llm_cfg.api_key,
                base_url=llm_cfg.base_url,
            )

    def get_response(self, messages=None, json_format=None):

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
        else:
            raise ValueError("Invalid call method")


class Multi_Agents_Model():
    def __init__(self, cfg):
        self.cfg = cfg

        self.planner = self.init_agent("planner")
        self.reasoner = self.init_agent("reasoner")
        self.verifier = self.init_agent("verifier")
        self.judger = self.init_agent("judger")
        self.summarizer = self.init_agent("summarizer")
    
    def init_agent(self, agent_name):
        prompt_cfg = self.cfg.prompt_method
        agents_cfg = self.cfg.llm.agents
        
        llm_cfg = agents_cfg[prompt_cfg[agent_name]]
        base_model_name = llm_cfg.name
        return model_name_map[base_model_name](llm_cfg)


class Qwen3_VL_Model2():
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor,StoppingCriteria,StoppingCriteriaList
    
    class NewlineStoppingCriteria(StoppingCriteria):
        def __init__(self, input_length,stop_token):
            # 保存输入的长度，用于后续判断新生成的部分
            self.input_length = input_length
            self.stop_token = stop_token
        
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            # 只解码新生成的部分（跳过输入部分）
            new_generated_ids = input_ids[0][self.input_length:]
            generated_text = processor.decode(new_generated_ids, skip_special_tokens=True)
            
            # 检查是否包含目标字符或字符串
            # print("新生成的文本:", generated_text)
            return self.stop_token in generated_text
    
    def __init__(self,llm_cfg,cfg):
        self.llm_cfg = llm_cfg
        self.cfg = cfg
        self.llm = Qwen3VLForConditionalGeneration.from_pretrained(
            self.llm_cfg.path, dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.llm_cfg.path)

    def get_response(self, messages=None, json_format=None, n=1, continue_final_message=False, top_logprobs=0, stop_token=None):
    
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        stopping_criteria = StoppingCriteriaList([NewlineStoppingCriteria(input_length=inputs.input_ids.shape[1],stop_token=stop_token)])
        
        response = []
        
        for _ in range(n):
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=self.llm_cfg.max_output_len,
                do_sample=True,
                temperature=self.llm_cfg.temperature,
                top_p=self.llm_cfg.top_p,
                stopping_criteria = stopping_criteria,
            )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_texts = processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )

            # # 打印所有生成的结果
            # for idx, text in enumerate(output_texts):
            #     if idx % 2 == 0:
            #         printgreen(f"{text}")
            #     else:
            #         printpurple(f"{text}")
            response.append(text)
        
        return response
    def get_reward(self, messages=None, json_format=None, n=1, continue_final_message=False, top_logprobs=5):
        
        response = self.get_response(messages, json_format, n, continue_final_message, top_logprobs)
        response_text = response["response"]
        
        num_yes = num_no = 0
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


class Qwen25_Model():
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
                max_model_len = 4096,
                max_num_seqs = 512,
                gpu_memory_utilization = 0.85
            )
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

    def get_response(self, messages=None, json_format=None):

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
                max_tokens=self.llm_cfg.max_output_len,
                stop_token_ids=[],
                n=self.llm_cfg.n,
            )
            
            processor = AutoProcessor.from_pretrained(self.llm_cfg.path)
            
            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            

            llm_inputs = {
                "prompt": prompt
            }
                    
            outputs = self.llm.generate([llm_inputs], sampling_params=sampling_params, use_tqdm=False)  
            
            generated_text = []
            for output in outputs[0].outputs:
                generated_text.append(output.text)
            
            response = {
                    "response": generated_text
                }
            
            return response
            
        else:
            raise ValueError("Invalid call method")

class Phi_Model():
    def __init__(self, cfg):
        # Note: The default setting of max_num_seqs (256) and
        # max_model_len (128k) for this model may cause OOM.
        # You may lower either to run this example on lower-end GPUs.

        # In this example, we override max_num_seqs to 5 while
        # keeping the original context length of 128k.
        llm_cfg = cfg.llm
        self.llm = LLM(
            model=llm_cfg.path,
            trust_remote_code=True,
            max_model_len=llm_cfg.max_input_len,
            max_num_seqs=llm_cfg.max_output_len,
        )
        stop_token_ids = None
        self.sampling_params = SamplingParams(temperature=llm_cfg.temperature,
                                     max_tokens=1024,
                                     stop_token_ids=stop_token_ids)

    def get_response(self, input_text, image_path):
        
        prompt = f"<|user|>\n<|image_1|>\n{input_text}<|end|>\n<|assistant|>\n"  # noqa: E501
        image = Image.open(image_path)
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {
                "image": image
            },
        }
        outputs = self.llm.generate(inputs, sampling_params=self.sampling_params, use_tqdm=False)
        response = {
            "response": outputs[0].outputs[0].text
        }
        return response


# 两种模型

class Qwen2_VL_Model():
    def __init__(self, llm_cfg, cfg):
        self.llm_cfg = llm_cfg
        self.cfg = cfg
        self.llm = LLM(
            model=llm_cfg.path,
            limit_mm_per_prompt={"image": 10, "video": 10}, # 每个提示最多包含的图片和视频数量
            max_model_len= llm_cfg.max_input_len,
            max_num_seqs=512,
            gpu_memory_utilization=0.85
        )
        self.processor = AutoProcessor.from_pretrained(llm_cfg.path)
       
    def get_reward(self, messages=None, json_format=None, n=1, continue_final_message=False, top_logprobs=5,node_depth=8):
        for attempt in range(self.cfg.prompt_method.attempt_num):
            try:
                inputs = [prepare_inputs_for_vllm(message, self.processor) for message in [messages]]
                if self.cfg.reward_method == "self_consistency":
                    sampling_params = SamplingParams(
                        temperature=self.llm_cfg.temperature,
                        top_p=self.llm_cfg.top_p,
                        # repetition_penalty=1.05,
                        max_tokens=self.llm_cfg.max_output_len,
                        stop_token_ids=[],
                        n = n,
                    )
                    # 处理多个输入
                    outputs = self.llm.generate(inputs, sampling_params=sampling_params, use_tqdm=False)
                    response_text = [output.text.strip() for output in outputs[0].outputs]
                    
                    num_yes = num_no = 0
                    for text in response_text:
                        if "yes" in text.lower():
                            num_yes += 1
                        elif "no" in text.lower():
                            num_no += 1
                    # print(num_yes, num_no)
                    
                    yes_prob = num_yes / n
                    no_prob = num_no / n
                
                    # response = {
                    #     "yes_prob": 1 * node_depth * yes_prob / 8,
                    #     "no_prob": 1 * node_depth * no_prob / 8,
                    # }
                    
                    response = {
                        "yes_prob": yes_prob,
                        "no_prob": no_prob,
                    }
                    
                    return response  
                else:
                    sampling_params = SamplingParams(
                        temperature=self.llm_cfg.temperature,
                        top_p=self.llm_cfg.top_p,
                        # repetition_penalty=1.05,
                        max_tokens=self.llm_cfg.max_output_len,
                        stop_token_ids=[],
                        logprobs=top_logprobs,
                        # n = 1,
                    )

                    outputs = self.llm.generate(inputs, sampling_params=sampling_params, use_tqdm=False)
                    
                    # 处理输出并计算概率
                    for output in outputs[0].outputs:
                        prob_dict = {"yes": 0.0, "no": 0.0}
                        # 获取token对数概率
                        token_logprobs = output.logprobs
                        # 检查是否存在logprobs
                        if token_logprobs:
                            # 检查logprobs的类型并相应处理
                            if isinstance(token_logprobs, dict):
                                # 遍历所有可能的token及其概率
                                for _, logprob_obj in token_logprobs.items():
                                    # 确保是有效的Logprob对象
                                    if hasattr(logprob_obj, 'decoded_token') and hasattr(logprob_obj, 'logprob'):
                                        token = logprob_obj.decoded_token.lower()  # 转为小写以增加匹配成功率
                                        logprob = logprob_obj.logprob
                                        
                                        # 计算概率（使用自然指数）
                                        prob = float(math.exp(logprob))
                                        
                                        # 更灵活的匹配逻辑
                                        if "yes" in token:
                                            prob_dict["yes"] = max(prob_dict["yes"], prob)  # 取最大概率
                                            # print(f"匹配到yes相关token: '{token}', 概率: {prob:.4f}")
                                        elif "no" in token:
                                            prob_dict["no"] = max(prob_dict["no"], prob)  # 取最大概率
                                            # print(f"匹配到no相关token: '{token}', 概率: {prob:.4f}")
                            else:
                                # 处理非字典类型的logprobs
                                # print(f"logprobs类型: {type(token_logprobs)}")
                                # 如果是列表，尝试遍历每个位置的候选token
                                if hasattr(token_logprobs, '__iter__') and not isinstance(token_logprobs, (dict, str)):
                                    for position_probs in token_logprobs:
                                        if isinstance(position_probs, dict):
                                            for _, logprob_obj in position_probs.items():
                                                if hasattr(logprob_obj, 'decoded_token') and hasattr(logprob_obj, 'logprob'):
                                                    token = logprob_obj.decoded_token.lower()
                                                    logprob = logprob_obj.logprob
                                                    prob = float(math.exp(logprob))
                                                    if "yes" in token:
                                                        prob_dict["yes"] = max(prob_dict["yes"], prob)
                                                    elif "no" in token:
                                                        prob_dict["no"] = max(prob_dict["no"], prob)
                        else:
                            print("警告: logprobs为空或None")
                        # 归一化概率（确保yes和no的概率和为1）
                        total_prob = prob_dict["yes"] + prob_dict["no"]
                        if total_prob > 0:
                            prob_dict["yes"] /= total_prob
                            prob_dict["no"] /= total_prob
                        else:
                            # 如果没有匹配到任何yes/no相关token，设置默认概率
                            # print("未匹配到任何yes/no相关token，使用默认概率分布")
                            prob_dict["yes"] = 0.5
                            prob_dict["no"] = 0.5
                        
                        response = {
                            "yes_prob": prob_dict["yes"],
                            "no_prob": prob_dict["no"],
                        }
                        
                    return response
            except Exception as e:
                print(f"Error: {str(e)}")
                if attempt == self.attempt_num-1:
                    print(f"Failed to get reward after {self.attempt_num} attempts. Error: {str(e)}")
                    raise ValueError(f"Failed to get reward after {self.attempt_num} attempts. Error: {str(e)}")

    def get_response(self, messages=None, json_format=None, n=1, continue_final_message=False, top_logprobs=0, stop_token=None):
        sampling_params = SamplingParams(
            temperature=self.llm_cfg.temperature,
            max_tokens=self.llm_cfg.max_output_len,
            top_p=self.llm_cfg.top_p,
            stop_token_ids=[],
            stop = stop_token,
            n = n,
        )
        
        inputs = [prepare_inputs_for_vllm(message, self.processor) for message in [messages]]
        
        outputs = self.llm.generate(inputs, sampling_params)
        
        response = {
            "response": [output.text.strip() for output in outputs[0].outputs],
        }
        
        return response
    
class Qwen3_VL_Model():
    def __init__(self, llm_cfg, cfg):
        self.llm_cfg = llm_cfg
        self.cfg = cfg
        self.llm = LLM(
            model=self.llm_cfg.path,
            limit_mm_per_prompt={"image": 2, "video": 1},  # 限制每个提示的多模态数量
            max_model_len= self.llm_cfg.max_input_len,
            max_num_seqs=llm_cfg.max_num_seqs,
            gpu_memory_utilization=self.llm_cfg.gpu_memory_utilization,
            max_num_batched_tokens = llm_cfg.max_num_batched_tokens, # 8196
        )
        self.processor = AutoProcessor.from_pretrained(llm_cfg.path)
        
    
    def get_response(self, messages=None, json_format=None, n=1, continue_final_message=False, top_logprobs=0, stop_token=None):
        sampling_params = SamplingParams(
            temperature=self.llm_cfg.temperature,
            max_tokens=self.llm_cfg.max_output_len,
            top_p=self.llm_cfg.top_p,
            stop_token_ids=[],
            stop = stop_token,
            n = n,
        )
        
        inputs = [prepare_inputs_for_vllm(message, self.processor) for message in [messages]]
        
        outputs = self.llm.generate(inputs, sampling_params)
        
        ans = []
        
        for output in outputs[0].outputs:
            content = output.text.strip()
            
            if content.startswith("Sub-question"):
                idx = content.find(":")
                content = content[idx+1:].strip()
            
            # print(content)
            
            ans.append(content)
            
        response = {
            "response": ans,
        }
        
        return response
        
    def get_reward(self, messages=None, json_format=None, n=1, continue_final_message=False, top_logprobs=5,node_depth=8):
        for attempt in range(self.cfg.prompt_method.attempt_num):
            try:
                inputs = [prepare_inputs_for_vllm(message, self.processor) for message in [messages]]
                
                if self.cfg.reward_method == "self_consistency":
                    sampling_params = SamplingParams(
                        temperature=self.llm_cfg.temperature,
                        top_p=self.llm_cfg.top_p,
                        # repetition_penalty=1.05,
                        max_tokens=self.llm_cfg.max_output_len,
                        stop_token_ids=[],
                        n = n,
                    )
                    # 处理多个输入
                    outputs = self.llm.generate(inputs, sampling_params=sampling_params, use_tqdm=False)
                    response_text = [output.text.strip() for output in outputs[0].outputs]
                    
                    num_yes = num_no = 0
                    for text in response_text:
                        if "yes" in text.lower():
                            num_yes += 1
                        elif "no" in text.lower():
                            num_no += 1
                    # print(num_yes, num_no)
                    
                    yes_prob = num_yes / n
                    no_prob = num_no / n
                
                    # response = {
                    #     "yes_prob": 1 * node_depth * yes_prob / 8,
                    #     "no_prob": 1 * node_depth * no_prob / 8,
                    # }
                    
                    response = {
                        "yes_prob": yes_prob,
                        "no_prob": no_prob,
                    }
                    
                    return response  
                else:
                    sampling_params = SamplingParams(
                        temperature=self.llm_cfg.temperature,
                        top_p=self.llm_cfg.top_p,
                        # repetition_penalty=1.05,
                        max_tokens=self.llm_cfg.max_output_len,
                        stop_token_ids=[],
                        logprobs=top_logprobs,
                        # n = 1,
                    )

                    outputs = self.llm.generate(inputs, sampling_params=sampling_params, use_tqdm=False)
                    
                    # 处理输出并计算概率
                    for output in outputs[0].outputs:
                        prob_dict = {"yes": 0.0, "no": 0.0}
                        # 获取token对数概率
                        token_logprobs = output.logprobs
                        # 检查是否存在logprobs
                        if token_logprobs:
                            # 检查logprobs的类型并相应处理
                            if isinstance(token_logprobs, dict):
                                # 遍历所有可能的token及其概率
                                for _, logprob_obj in token_logprobs.items():
                                    # 确保是有效的Logprob对象
                                    if hasattr(logprob_obj, 'decoded_token') and hasattr(logprob_obj, 'logprob'):
                                        token = logprob_obj.decoded_token.lower()  # 转为小写以增加匹配成功率
                                        logprob = logprob_obj.logprob
                                        
                                        # 计算概率（使用自然指数）
                                        prob = float(math.exp(logprob))
                                        
                                        # 更灵活的匹配逻辑
                                        if "yes" in token:
                                            prob_dict["yes"] = max(prob_dict["yes"], prob)  # 取最大概率
                                            # print(f"匹配到yes相关token: '{token}', 概率: {prob:.4f}")
                                        elif "no" in token:
                                            prob_dict["no"] = max(prob_dict["no"], prob)  # 取最大概率
                                            # print(f"匹配到no相关token: '{token}', 概率: {prob:.4f}")
                            else:
                                # 处理非字典类型的logprobs
                                # print(f"logprobs类型: {type(token_logprobs)}")
                                # 如果是列表，尝试遍历每个位置的候选token
                                if hasattr(token_logprobs, '__iter__') and not isinstance(token_logprobs, (dict, str)):
                                    for position_probs in token_logprobs:
                                        if isinstance(position_probs, dict):
                                            for _, logprob_obj in position_probs.items():
                                                if hasattr(logprob_obj, 'decoded_token') and hasattr(logprob_obj, 'logprob'):
                                                    token = logprob_obj.decoded_token.lower()
                                                    logprob = logprob_obj.logprob
                                                    prob = float(math.exp(logprob))
                                                    if "yes" in token:
                                                        prob_dict["yes"] = max(prob_dict["yes"], prob)
                                                    elif "no" in token:
                                                        prob_dict["no"] = max(prob_dict["no"], prob)
                        else:
                            print("警告: logprobs为空或None")
                        # 归一化概率（确保yes和no的概率和为1）
                        total_prob = prob_dict["yes"] + prob_dict["no"]
                        if total_prob > 0:
                            prob_dict["yes"] /= total_prob
                            prob_dict["no"] /= total_prob
                        else:
                            # 如果没有匹配到任何yes/no相关token，设置默认概率
                            # print("未匹配到任何yes/no相关token，使用默认概率分布")
                            prob_dict["yes"] = 0.5
                            prob_dict["no"] = 0.5
                        
                        response = {
                            "yes_prob": prob_dict["yes"],
                            "no_prob": prob_dict["no"],
                        }
                        
                    return response
                        
            except Exception as e:
                print(f"第{attempt}次尝试失败: {e}")
                continue
        

model_name_map = {
    "Qwen2-VL-7B-Instruct": Qwen2_VL_Model,
    "Qwen2-VL-2B-Instruct": Qwen2_VL_Model,
    "Qwen2.5-VL-3B-Instruct": Qwen2_VL_Model,
    "Qwen2.5-VL-7B-Instruct": Qwen2_VL_Model,
    "Qwen2.5-7B-Instruct": Qwen25_Model,
    "Qwen2.5-14B-Instruct": Qwen25_Model,
    "Qwen3-VL-8B-Instruct": Qwen3_VL_Model,
    "multi_agents": Multi_Agents_Model,
    
}

class Model():
    def __init__(self, cfg):
        self.cfg = cfg
        self.prompt_cfg = cfg.prompt_method
        if cfg.llm.name == "multi_agents":
            self.model = model_name_map[cfg.llm.name](cfg)
        else:
            self.model = model_name_map[cfg.llm.name](cfg.llm,cfg)

        self.embedding_model = None
        if cfg.embedding_model_path != "none":
            self.embedding_model = SentenceTransformer(cfg.embedding_model_path, device="cuda:0")
            
    def get_response(self, input_text, image_path):
        try:
            if self.prompt_cfg.name == "cantor":
                return cantor_forward(input_text, image_path, self.model)
            elif self.prompt_cfg.name == "cot":
                return cot_forward(input_text, image_path, self.model)
            elif self.prompt_cfg.name == "qa":
                return qa_forward(input_text, image_path, self.model)
            elif self.prompt_cfg.name == "our":
                return our_forward(input_text, image_path, self.model)
            elif "our1" in self.prompt_cfg.name:
                return our1_forward(input_text, image_path, self.model)
            elif "our2" in self.prompt_cfg.name:
                return our2_forward(input_text, image_path, self.model)
            elif "our3" in self.prompt_cfg.name:
                return our3_forward(input_text, image_path, self.model, self.cfg)
            elif "our4" in self.prompt_cfg.name:
                return our4_forward(input_text, image_path, self.model, self.cfg)
            elif "our5" in self.prompt_cfg.name:
                return our5_forward(input_text, image_path, self.model, self.cfg)
            elif "our6" in self.prompt_cfg.name:
                return our6_forward(input_text, image_path, self.model, self.cfg)
            elif self.prompt_cfg.name in ["mcts", "mctsv1"]:
                return mcts_forward(input_text, image_path, self.model, self.cfg)
            elif self.prompt_cfg.name in ["mctsv2"]:
                return mcts_forward_v2(input_text, image_path, self.model, self.cfg)
            elif self.prompt_cfg.name in ["mctsv3"]:
                return mcts_forward_v3(input_text, image_path, self.model, self.cfg)
            elif self.prompt_cfg.name in ["mctsv4"]:
                return mcts_forward_v4(input_text, image_path, self.model, self.cfg)
            elif self.prompt_cfg.name in ["mctsv5"]:
                return mcts_forward_v5(input_text, image_path, self.model, self.cfg)
            elif self.prompt_cfg.name in ["mctsv6"]:
                return mcts_forward_v6(input_text, image_path, self.model, self.cfg)
            elif self.prompt_cfg.name in ["mctsv7"]:
                return mcts_forward_v7(input_text, image_path, self.model, self.cfg)
            elif self.prompt_cfg.name in ["mctsv8"]:
                return mcts_forward_v8(input_text, image_path, self.model, self.embedding_model, self.cfg)
            elif self.prompt_cfg.name in ["directQ"]:
                return directQ(input_text, image_path, self.model, self.cfg)
            else:
                raise ValueError("Invalid prompt method")
        except Exception as e:
            print(e)
            raise e




