import time
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info


class MultiModalKVCacheDemo:

    def __init__(self, model_path):
        # ✅ 初始化 processor（多模态必须）
        self.processor = AutoProcessor.from_pretrained(model_path)

        # ✅ 两个模型：对比用
        # self.llm_no_cache = LLM(
        #     model=model_path,
        #     enable_prefix_caching=False,   # ❌ 关闭
        #     gpu_memory_utilization=0.85,
        #     max_model_len=4096
        # )

        self.llm_cache = LLM(
            model=model_path,
            enable_prefix_caching=True,    # ✅ 开启 KV cache
            gpu_memory_utilization=0.85,
            max_model_len=4096
        )
        

    # =========================
    # 多模态输入处理
    # =========================
    def prepare_inputs_for_vllm(self, messages):
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=self.processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True
        )

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

    # =========================
    # KV cache batch推理
    # =========================
    def batch_infer(self, llm, messages_list, n=5):

        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=256,
            n=n
        )

        inputs = [
            self.prepare_inputs_for_vllm(messages)
            for messages in messages_list
        ]

        outputs = llm.generate(inputs, sampling_params)

        results = []
        for output in outputs:
            results.append([o.text.strip() for o in output.outputs])

        return results

    # =========================
    # 构造共享 prefix（关键！）
    # =========================
    def build_messages(self):
        common_prefix = [
            {"role": "system", "content": "你是一个多模态AI助手"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请阅读图片并回答问题"},
                    {
                        "type": "image",
                        "image": "/home/jncsnlp4/SSD2/ZJJ/data/images/ScienceQA_VAL/1.jpg"
                    }
                ]
            }
        ]

        messages_list = []

        for i in range(100000):
            messages = common_prefix + [
                {"role": "user", "content": f"这张图片里有什么？变体{i}"}
            ]
            messages_list.append(messages)

        return messages_list

    # =========================
    # 性能测试
    # =========================
    def run_test(self):

        messages_list = self.build_messages()

        # 🔥 预热
        # self.batch_infer(self.llm_no_cache, [messages_list[0]])

        # ❌ 无 KV cache
        # start = time.time()
        # self.batch_infer(self.llm_no_cache, messages_list)
        # t1 = time.time() - start

        # # ✅ 有 KV cache
        start = time.time()
        self.batch_infer(self.llm_cache, messages_list)
        t2 = time.time() - start

        print("\n===== 📊 KV Cache 对比 =====")
        # print(f"无缓存: {t1:.4f}s")
        print(f"有缓存: {t2:.4f}s")
        # print(f"加速比: {t1 / t2:.2f}x")


if __name__ == "__main__":
    model_path = "/home/jncsnlp4/SSD2/model/qwen2-vl-intruct"

    # demo = MultiModalKVCacheDemo(model_path)
    # demo.run_test()
    
    import json
    
    file = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=baseline_KV/data=charxiv/llm=Qwen2-VL-7B-Instruct/prompt_method=mctsv8/gen-reasoning_val.json"
    
    list_file = ["/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=baseline_KV/data=charxiv/llm=Qwen2-VL-7B-Instruct/prompt_method=mctsv8/gen-reasoning_val.json"]
    data_num = 0
    all_time = 0
    acc = 0
    
    for file in list_file:
        with open(file, 'r') as f:
            data = json.load(f)
        for i,j in data.items():
            data_num += 1
            if data_num <= 356 or data_num > 493:
                continue
            if "time" not in j:
                continue
            acc += 1
            all_time += j['time']
            
    print(data_num)
    print(acc)
    print(f"平均时间: {all_time / acc:.4f}s")
