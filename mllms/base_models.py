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

from prompt_methods.cantor.cantor import cantor_forward
from prompt_methods.ours.our import our_forward
from prompt_methods.ours.our1 import our1_forward
from prompt_methods.ours.our2 import our2_forward
from prompt_methods.ours.our3 import our3_forward
from prompt_methods.ours.our4 import our4_forward
from prompt_methods.ours.our5 import our5_forward
from prompt_methods.ours.our6 import our6_forward
from prompt_methods.cot.cot import cot_forward
from prompt_methods.qa.qa import qa_forward

from prompt_methods.mcts.chartqa_mcts import mcts_forward
from prompt_methods.mctsv2.chartqa_mcts import mcts_forward_v2
from prompt_methods.mctsv3.chartqa_mcts import mcts_forward_v3
from prompt_methods.mctsv4.chartqa_mcts import mcts_forward_v4
from prompt_methods.mctsv5.chartqa_mcts import mcts_forward_v5
from prompt_methods.mctsv6.chartqa_mcts import mcts_forward_v6
from prompt_methods.mctsv7.chartqa_mcts import mcts_forward_v7
from prompt_methods.mctsv8.chartqa_mcts import mcts_forward_v8

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
        else:
            raise ValueError("Invalid call method")

# vision and text

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

class Qwen2_VL_Model():
    def __init__(self, llm_cfg):

        self.llm_cfg = llm_cfg

        if llm_cfg.call_method == "local":
            self.llm = LLM(
                model=llm_cfg.path,
                limit_mm_per_prompt={"image": 10, "video": 10},
                max_model_len=llm_cfg.max_input_len,
                max_num_seqs=llm_cfg.max_output_len,
            )
            self.sampling_params = SamplingParams(
                temperature=0.1,
                top_p=0.001,
                repetition_penalty=1.05,
                max_tokens=llm_cfg.max_output_len,
                stop_token_ids=[],
            )
        else:
            self.llm = None
            self.sampling_params = None

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
    
    def get_least_request_client(self):
        for client in self.all_clients:
            if self.request_count[client] == 0:
                return client
        return min(self.request_count, key=self.request_count.get)
    
    def get_reward(self, messages=None, json_format=None, n=1, continue_final_message=False, top_logprobs=5):
        response = None
        for attempt in range(self.attempt_num):
            try:
                with self.lock:
                    client = self.get_least_request_client()
                    self.request_count[client] += 1
                    
                if json_format is not None:
                
                    chat_response = client.beta.chat.completions.parse(
                        n=n,
                        model=self.llm_cfg.name,
                        messages=messages,
                        response_format=json_format if json_format else None,
                        top_p=self.llm_cfg.top_p,
                        temperature=self.llm_cfg.temperature,
                        max_tokens=self.llm_cfg.max_output_len,
                        extra_body={
                            "repetition_penalty": self.llm_cfg.repetition_penalty,
                            "guided_choice": ["Yes", "No"],
                            "add_generation_prompt": False if continue_final_message else True,
                            "continue_final_message": continue_final_message,
                        },
                        logprobs=True if top_logprobs > 0 else False,
                        top_logprobs=top_logprobs,
                    )

                else:

                    chat_response = client.chat.completions.create(
                        n=n,
                        model=self.llm_cfg.name,
                        messages=messages,
                        temperature=0.0,
                        top_p=1.0,
                        max_tokens=self.llm_cfg.max_output_len,
                        extra_body={
                            "repetition_penalty": self.llm_cfg.repetition_penalty,
                            "guided_choice": ["Yes", "No"],
                            "add_generation_prompt": False if continue_final_message else True,
                            "continue_final_message": continue_final_message,
                        },
                        logprobs=True if top_logprobs > 0 else False,
                        top_logprobs=top_logprobs,
                    )
                # 提取Yes和No的logprobs
                logprobs = chat_response.choices[0].logprobs.content[0].top_logprobs if chat_response.choices[0].logprobs else None
                logits = {}
                for token in logprobs:
                    logits[token.token] = token.logprob
                        
                yes_logprob = logits['Yes'] if 'Yes' in logits else 0.0
                no_logprob = logits['No'] if 'No' in logits else 0.0
                
                # 转换为0-1的概率
                yes_prob = math.exp(yes_logprob) / (math.exp(yes_logprob) + math.exp(no_logprob))
                no_prob = math.exp(no_logprob) / (math.exp(yes_logprob) + math.exp(no_logprob))
                
                response = {
                    "yes_prob": yes_prob,
                    "no_prob": no_prob,
                }
                
                with self.lock:
                    self.request_count[client] -= 1
                    
                return response
            
            except Exception as e:
                print(f"Error: {str(e)}")
                if attempt == self.attempt_num-1:
                    print(f"Failed to get reward after {self.attempt_num} attempts. Error: {str(e)}")
                    raise ValueError(f"Failed to get reward after {self.attempt_num} attempts. Error: {str(e)}")

    def get_response(self, messages=None, json_format=None, n=1, continue_final_message=False, top_logprobs=0, stop_token=None):

        if self.llm_cfg.call_method == "api":
            response = None
            for attempt in range(self.attempt_num):
                try:
                    with self.lock:
                        client = self.get_least_request_client()
                        self.request_count[client] += 1

                    if json_format is not None:
                    
                        chat_response = client.beta.chat.completions.parse(
                            n=n,
                            model=self.llm_cfg.name,
                            messages=messages,
                            response_format=json_format if json_format else None,
                            top_p=self.llm_cfg.top_p,
                            temperature=self.llm_cfg.temperature,
                            max_tokens=self.llm_cfg.max_output_len,
                            extra_body={
                                "repetition_penalty": self.llm_cfg.repetition_penalty,
                                "add_generation_prompt": False if continue_final_message else True,
                                "continue_final_message": continue_final_message,
                            },
                            stop=stop_token,
                        )

                    else:

                        chat_response = client.chat.completions.create(
                            n=n,
                            model=self.llm_cfg.name,
                            messages=messages,
                            temperature=self.llm_cfg.temperature,
                            top_p=self.llm_cfg.top_p,
                            max_tokens=self.llm_cfg.max_output_len,
                            extra_body={
                                "repetition_penalty": self.llm_cfg.repetition_penalty,
                                "add_generation_prompt": False if continue_final_message else True,
                                "continue_final_message": continue_final_message,
                                # "echo": True,
                            },
                            stop=stop_token,
                            # seed=self.llm_cfg.seed,
                        )

                    response = {
                        "response": [choice.message.content for choice in chat_response.choices] if n > 1 else chat_response.choices[0].message.content
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

            processor = AutoProcessor.from_pretrained(self.llm_cfg.path)
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

            json_shape = '''{
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string"
                            },
                            "content": {
                                "type": "string"
                            },
                            "next_action": {
                                "type": "string"
                            }
                        },
                        "required": ["title", "content", "next_action"]
                    }'''

            outputs = self.llm.generate([llm_inputs], sampling_params=self.sampling_params, guided_options_request={
                    "guided_json": json_shape,
                    # "guided_json_object": json_format
                } if json_format else None, 
                use_tqdm=False)
            generated_text = outputs[0].outputs[0].text

            response = {
                "response": generated_text
            }
            return response
        
        else:
            raise ValueError("Invalid call method")


# multi agents

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

model_name_map = {
    "Qwen2-VL-7B-Instruct": Qwen2_VL_Model,
    "Qwen2-VL-2B-Instruct": Qwen2_VL_Model,
    "Qwen2.5-VL-3B-Instruct": Qwen2_VL_Model,
    "Qwen2.5-VL-7B-Instruct": Qwen2_VL_Model,
    "Qwen2.5-7B-Instruct": Qwen25_Model,
    "Qwen2.5-14B-Instruct": Qwen25_Model,
    "multi_agents": Multi_Agents_Model,
}

class Model():
    def __init__(self, cfg):
        self.cfg = cfg
        self.prompt_cfg = cfg.prompt_method
        if cfg.llm.name == "multi_agents":
            self.model = model_name_map[cfg.llm.name](cfg)
        else:
            self.model = model_name_map[cfg.llm.name](cfg.llm)

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
                return mcts_forward_v8(input_text, image_path, self.model, self.cfg)
            else:
                raise ValueError("Invalid prompt method")
        except Exception as e:
            print(e)
            raise e

