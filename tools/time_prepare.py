

import datetime
import time
import json



def main():
    json_path1 = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=ours3.charxiv.all/data=charxiv/llm=Qwen2-VL-7B-Instruct/prompt_method=our3/gen-reasoning_val.json"
    # json_path2 = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=timecut/data=mathvision/llm=Qwen2-VL-7B-Instruct/prompt_method=mctsv8/generation.json"
    json_path2 = ""
    if json_path1 == "":
        print("json_path1 为空")
        return
    
    with open(json_path1, "r",encoding="utf-8") as f:
        data1 = json.load(f)
    
    if json_path2 != "":
        with open(json_path2, "r",encoding="utf-8") as f:
            data2 = json.load(f)
    
    tol_time1 = tol_time2 = 0
    
    num1 = num2 = 0
    for key,value in data1.items():
        if value["response"] != "Failed to generate response!":
            num1 += 1
            tol_time1 += value["time"]

    if json_path2 != "":
        for key,value in data2.items():
            if value["response"] != "Failed to generate response!":
                num2 += 1
                tol_time2 += value["time"]

    print(num1,num2)
    
    if num1 > 0:
        print(f"avg time1: {tol_time1 / num1}")
    if num2 > 0:
        print(f"avg time2: {tol_time2 / num2}")
    
    time_more = abs(tol_time2 - tol_time1)
    
    print(f"time_more: {time_more}")
    
    hour = minute = second = 0
    
    hour = time_more // 3600
    minute = (time_more % 3600) // 60
    second = time_more % 60
    
    print(f"小时: {hour}")
    print(f"分钟: {minute}")
    print(f"秒钟: {second}")
    


if __name__ == "__main__":
    main()
    



















