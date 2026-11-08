import time
import os
import json
from openai import OpenAI
from pydantic import BaseModel
import base64
from PIL import Image
import io

from .prompts import system_prompt
from tools.color_print import *

class CoT(BaseModel):
    chain_of_thought: str
    final_answer: str

def cot_forward(query, img_path, model):
    # 打开本地图片
    # with open(img_path, "rb") as image_file:
    #     image_data = image_file.read()
    # # 将图片数据转换为 base64 编码
    # image_base64 = base64.b64encode(image_data).decode('utf-8')
    # # 构建 data URL
    # image_url = f"data:image/jpeg;base64,{image_base64}"
    image_url = img_path
    
    prompt = """
    **Example**: 
    
    **Question**: 
    What is the result of 2 + 3 × 4?
    
    **Answer**:
    {{
        "chain_of_thought": "Step 1: Follow the order of mathematical operations (PEMDAS/BODMAS), which prioritizes multiplication over addition. First calculate the multiplication part: 3 × 4 = 12. Step 2: Add the result to 2, so 2 + 12 = 14. No other calculation order is valid here, as ignoring the operation priority would lead to wrong results like (2+3)×4=20.",
        "final_answer": "14"
    }}
    
    **Question**: 
    Evaluate the difficulty level (L1/L2/L3/L4) of the QA pair: Question - "What is 1+1?", Answer - "2".
    
    **Answer**:
    {{
        "chain_of_thought": "Step 1: Analyze the knowledge breadth: the question only involves basic arithmetic knowledge with no requirement for professional domain expertise. Step 2: Judge the reasoning depth: no logical deduction is needed, and the answer can be given directly. Step 3: Assess semantic complexity: the question is clearly stated without ambiguity, and the answer is concise and unique. Based on the above three points, it meets the criteria for L1 (easy) difficulty level.",
        "final_answer": "L1"
    }}
    
    **Question**: 
    {query}
    
    **Answer**:
    
    """.format(query=query)
    
    messages=[
        {"role": "system", "content": system_prompt},
        
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

    for attempt in range(3):
        try:
            response = model.get_response(messages=messages, json_format=None)
            response = response["response"][0]
            print(response)
            # input("点击继续")
            if type(response) == str:
                response = json.loads(response)
            
            cot = response["chain_of_thought"]
            answer = response["final_answer"]
            # printblue(f"{cot}")
            # printred(f"{answer}")
            # input("点击继续")
            # response_content = f"Chain of thought: {cot}\nFinal answer: {answer}"
            res = {
                "response": answer,
                "cot": cot,
            }
            
            return res
        
        except Exception as e:
            if attempt == 2:
                print(f"Failed to generate CoT and final answer after 3 attempts. Error: {str(e)}")
                answer = f"Failed to generate CoT and final answer after 3 attempts. Error: {str(e)}"
                res = {
                    "response": answer
                }
                return res
        
        # try:
        #     response = model.get_response(messages=messages, json_format=None)
        #     response = response["response"]
        #     res = {
        #         "response": response,
        #     }
        #     return res
        # except Exception as e:
        #     if attempt == 2:
        #         print(f"Failed to generate CoT and final answer after 3 attempts. Error: {str(e)}")
        #         answer = f"Failed to generate CoT and final answer after 3 attempts. Error: {str(e)}"
        #         res = {
        #             "response": answer
        #         }
        #         return res
