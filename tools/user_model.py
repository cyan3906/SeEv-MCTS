
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
import re
from tools.color_print import *
import queue
import pickle
import json,os,time

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

class Qwen2: # 支持qwen2, qwen25 qwen3
    def __init__(self, llm_path, enable_thinking = False):
        self.llm = LLM(
            model=llm_path,
            limit_mm_per_prompt={"image": 10, "video": 10}, # 每个提示最多包含的图片和视频数量
            max_model_len= 7300,
            max_num_seqs=512,
            gpu_memory_utilization=0.85,
            enable_prefix_caching = enable_thinking
        )
        
        self.processor = AutoProcessor.from_pretrained(llm_path)
        
    def get_response(self, messages, stop_token = None, n = 11,temperature = 0.7):
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=1288,
            top_p=0.95,
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


class Qwen3:
    def __init__(self, llm_path, enable_thinking = False):
        self.llm = LLM(
            model=llm_path,
            max_model_len = 3500,
            max_num_seqs = 16,
            gpu_memory_utilization = 0.95,
            max_num_batched_tokens = 512
        )
        self.enable_thinking = enable_thinking
        
        self.processor = AutoProcessor.from_pretrained(llm_path)

    def get_response(self, messages, stop_token = None, n = 11,temperature = 0.7):
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=648,
            top_p=0.95,
            stop_token_ids=[],
            stop = stop_token,
            n = n,
        )
        
        prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking = self.enable_thinking,
                # enable_thinking = False # 是否打开qwen3的思考模式
        )
        
        llm_inputs = {
            "prompt": prompt
        }
        
        outputs = self.llm.generate([llm_inputs], sampling_params=sampling_params, use_tqdm=False)
        
        response = {
            "response": [output.text.strip() for output in outputs[0].outputs],
        }
        
        return response

class Llama3: # 支持llama,mistral
    def __init__(self, llm_path):
        self.llm = LLM(
            model=llm_path,
            max_model_len= 7300,
            max_num_seqs=512,
            gpu_memory_utilization=0.85
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
    
    def get_response(self, messages, stop_token = None, n = 11):
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=1288,
            top_p=0.95,
            stop_token_ids=[],
            stop = stop_token,
            n = n,
        )
        
        inputs = [self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in [messages]]
        outputs = self.llm.generate(inputs, sampling_params)
        response = {
            "response": [output.text.strip() for output in outputs[0].outputs],
        }
        
        return response