
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.utils import FlexibleArgumentParser


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




class Qwen2_VL_Model():
    def __init__(self):
        self.llm = LLM(
            model="/home/jncsnlp4/SSD2/model/qwen2-vl-intruct",
            limit_mm_per_prompt={"image": 10, "video": 10}, # 每个提示最多包含的图片和视频数量
            max_model_len= 7300,
            max_num_seqs=512,
            gpu_memory_utilization=0.85
        )
        self.processor = AutoProcessor.from_pretrained("/home/jncsnlp4/SSD2/model/qwen2-vl-intruct")
       
    def get_response(self, messages=None, json_format=None, n=1, continue_final_message=False, top_logprobs=0, stop_token=None):
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=512,
            top_p=0.9,
            stop_token_ids=[],
            stop = stop_token,
            n = n,
        )
        
        inputs = [prepare_inputs_for_vllm(message, self.processor) for message in [messages]]
        
        generated = self.llm.generate(inputs, sampling_params,output_logits=True,return_dict_in_generate=True)
        
        logits = generated.logits
        probs = [torch.softmax(log, dim=-1) for log in logits]
            # print(probs)
        generated_ids = generated.sequences
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        logit_pro = 0
        # for i, token_id in enumerate(generated_ids[0][len(inputs.input_ids[0]):]):
        #     token_prob = probs[i][0, token_id].item()
        #     word = processor.decode(token_id)
        #     # if token_id == 7951 or token_id == 30570:
        #     print(f"Token ID: {token_id}, word:{word},Probability: {token_prob}")
        #         # logit_pro = token_prob
        real_prob = 0
        fake_prob = 0
        certain_prob = 0
        for i, token_id in enumerate(generated_ids[0][len(inputs.input_ids[0]):]):
            if processor.decode(token_id) == "sure":
                certain_prob = probs[i][0, token_id].item()
                # print("prob:",certain_prob,token_id)
            if token_id == 7951 or token_id == 30570:
                # index_prob = probs[i][0]
                top_value,top_index = torch.topk(probs[i][0],5)
                for x,y in zip(top_value,top_index):
                    # print(x.item(),y.item())  
                    word = processor.decode(y)
                    if word == "fake":
                        fake_prob = x
                    if word == "real":
                        real_prob = x 
                break        
        
        # response = {
        #     "response": [output.text.strip() for output in outputs[0].outputs],
        # }
        
        return output_text,real_prob,fake_prob


llm = Qwen2_VL_Model()

query = "describe the image"

init_system_prompt = "You are a helpful assistant."

messages=[
    {"role": "system", "content": init_system_prompt},
    
    {
    "role": "user",
    "content": [
        {
            "type": "image_url",
            "image_url": "/home/jncsnlp4/SSD2/ZJJ/data/images/ScienceQA_TEST/1.jpg",
        },
        {"type": "text", "text": query},
    ],
    },
]

response,real_prob,fake_prob = llm.get_response(messages,n = 5)

print(response ,real_prob,fake_prob)

