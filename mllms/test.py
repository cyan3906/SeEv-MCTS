import json
import re
from typing import Any, Optional

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


class Llama_Model:
    def __init__(
        self,
        model_path: str = ("/home/jncsnlp4/SSD2/model/" "Meta-Llama-3-8B-Instruct"),
    ):
        self.model_path = model_path

        self.llm = LLM(
            model=self.model_path,
            # 输入提示词和输出内容共用这个上下文长度
            max_model_len=4096,
            # 512 通常已经很高，显存不足时可以降低
            max_num_seqs=512,
            gpu_memory_utilization=0.85,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=True,
        )

        # Llama 3 通常没有单独的 pad token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @staticmethod
    def _extract_answer(text: str) -> Optional[Any]:
        """
        从模型输出中提取 extracted_answer。

        支持以下形式：
        1. {"extracted_answer": "42"}
        2. ```json
           {"extracted_answer": "42"}
           ```
        3. 模型在 JSON 前后附加解释文字
        """

        cleaned_text = text.strip()

        # 删除 Markdown JSON 代码块标记
        cleaned_text = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned_text,
            flags=re.IGNORECASE,
        )
        cleaned_text = re.sub(
            r"\s*```$",
            "",
            cleaned_text,
        )

        # 第一种情况：整个输出就是 JSON
        try:
            result = json.loads(cleaned_text)

            if isinstance(result, dict) and "extracted_answer" in result:
                return result["extracted_answer"]

        except json.JSONDecodeError:
            pass

        # 第二种情况：输出中包含一个 JSON 对象
        json_pattern = r"\{[\s\S]*?\}"

        for json_text in re.findall(
            json_pattern,
            cleaned_text,
        ):
            try:
                result = json.loads(json_text)

                if isinstance(result, dict) and "extracted_answer" in result:
                    return result["extracted_answer"]

            except json.JSONDecodeError:
                continue

        # 第三种情况：JSON 不规范，使用正则兜底
        answer_pattern = (
            r'["\']?extracted_answer["\']?'
            r"\s*:\s*"
            r'(?:"([^"]*)"|\'([^\']*)\'|([^,\}\n]+))'
        )

        match = re.search(
            answer_pattern,
            cleaned_text,
            flags=re.IGNORECASE,
        )

        if match:
            answer = next(
                (group for group in match.groups() if group is not None),
                None,
            )

            if answer is not None:
                return answer.strip()

        return None

    def get_response(
        self,
        messages: list[dict[str, str]],
        json_format: bool = True,
        is_evaluation: bool = False,
        n: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 512,
    ) -> dict[str, list[Any]]:
        """
        调用 Llama 模型生成回答。

        参数：
            messages:
                OpenAI 格式的消息列表。

            json_format:
                是否要求模型输出 JSON。
                该参数主要用于控制解析方式。

            is_evaluation:
                True：直接返回模型原始输出。
                False：尝试提取 extracted_answer。

            n:
                一次生成的候选回答数量。

            temperature:
                采样温度。

            top_p:
                nucleus sampling 参数。

            max_tokens:
                最大输出 token 数。

        返回：
            {
                "response": [...]
            }
        """

        if not messages:
            raise ValueError("messages 不能为空")

        if n < 1:
            raise ValueError("n 必须大于等于 1")

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=n,
            # 使用模型自身的 EOS token 停止生成
            stop_token_ids=[self.tokenizer.eos_token_id],
        )

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        request_outputs = self.llm.generate(
            [prompt],
            sampling_params,
            use_tqdm=False,
        )

        if not request_outputs:
            return {
                "response": [],
            }

        generated_results = []

        for candidate in request_outputs[0].outputs:
            raw_text = candidate.text.strip()

            # 评估模式直接返回模型原始文本
            if is_evaluation:
                generated_results.append(raw_text)
                continue

            if json_format:
                extracted_answer = self._extract_answer(raw_text)

                generated_results.append(
                    {
                        "extracted_answer": extracted_answer,
                        "raw_text": raw_text,
                    }
                )
            else:
                generated_results.append(raw_text)

        return {
            "response": generated_results,
        }


def main():
    llm = Llama_Model()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a mathematical reasoning assistant. "
                "Solve the problem carefully. "
                "Your final output must be valid JSON in the "
                "following format: "
                '{"extracted_answer": "final answer"}'
            ),
        },
        {
            "role": "user",
            "content": (
                """
                Solve the following mathematics problem step by step.

Problem:

A sequence ((a_n)) is defined recursively by

[
a_{i+1}=\frac{1}{1-a_i}
]

for (i\geq 1). If (a_3=a_1), compute

[
(a_9)^9.
]

Requirements:

1. Express (a_2) in terms of (a_1).
2. Express and simplify (a_3) in terms of (a_1).
3. Use the condition (a_3=a_1) to derive an algebraic equation for (a_1).
4. Explain why (a_3=a_1) causes the subsequent terms of the sequence to repeat.
5. Determine the relationship between (a_9) and (a_1).
6. Compute ((a_9)^9).
7. Do not skip algebraic transformations.
8. Return valid JSON only. Do not include Markdown code fences or any text outside the JSON object.

Use exactly the following output format:

{
"a2_expression": "",
"a3_expression": "",
"equation_from_condition": "",
"periodicity_reasoning": [
"",
""
],
"a9_expression": "",
"power_reasoning": [
"",
""
],
"extracted_answer": ""
}

                
                """
            ),
        },
    ]

    result = llm.get_response(
        messages=messages,
        json_format=True,
        is_evaluation=False,
        # 一次生成一个回答
        n=1,
        temperature=0.7,
        top_p=0.95,
        max_tokens=512,
    )

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    # 待执行的字符串代码
    code_str = """
    nums = [1, 1, 6, 9]
    res = ((1 + 1) * 9) + 6
    print(f"24点计算结果: {res}")
    """

    # 直接执行字符串代码
    eval(code_str)
    main()
