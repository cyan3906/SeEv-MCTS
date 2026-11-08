from vllm import LLM, SamplingParams
import re


# 全局只初始化一次，避免每次都重新加载模型
llm = LLM(
    model="/home/jncsnlp4/SSD2/model/qwen2-vl-intruct",
    max_model_len=2048,
    gpu_memory_utilization=0.85,
    enable_prefix_caching=True,
)


sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=16,
)


def text_similarity_score(
    text1: str,
    text2: str,
    threshold: float = 0.8,
) -> tuple[bool, float]:
    """
    使用 qwen2-vl 通过生成方式判断两个文本的语义相似度
    返回: (是否相似, 分数)
    """

    prompt = f"""你是一个文本语义相似度评估器。

请比较下面两段文本的语义是否一致。

文本1：
{text1}

文本2：
{text2}

请只输出一个 0 到 1 之间的小数：
- 1 表示完全相同
- 0 表示完全不同
不要输出其它内容。
"""

    outputs = llm.generate(prompt, sampling_params)
    response = outputs[0].outputs[0].text.strip()

    # 提取模型输出中的数字
    match = re.search(r"0(?:\.\d+)?|1(?:\.0+)?", response)

    if match:
        score = float(match.group())
    else:
        # 防止模型输出异常
        score = 0.0

    return score >= threshold, score


if __name__ == "__main__":
    text1 = "The answer is 1."
    text2 = "窗台上有一只橘色的猫，正在享受阳光。"

    is_similar, score = text_similarity_score(text1, text2)
    print(f"score={score:.3f}, similar={is_similar}")

    text3 = "一只黑狗在草地上奔跑。"
    text4 = "画面中有两只白色的猫在睡觉。"

    is_similar, score = text_similarity_score(text3, text4)
    print(f"score={score:.3f}, similar={is_similar}")