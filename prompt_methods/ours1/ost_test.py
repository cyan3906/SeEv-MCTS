

import json
import re

from typing import Dict, Tuple


def desc(question): 
    
    plan_prompt = "Please describe in detail the image information related to the question."
    
    return plan_prompt


def concat_ost_steps(solution_trace: Dict[int, Dict[str, str]]) -> Tuple[str, int]:
    """Return: concatenated one-step thought steps, next one-step thought step id"""
    
    """
    
    solution_trace = {
        0: {"user_question": "计算123 + 456", "ost_step": {}},
        1: {"direct_answer": "579", "ost_step": {}}
    }
    
    """
    
    last_tuple = list(solution_trace.items())[-1] # 获取最后一个元组
    last_tuple_id, last_tuple_recording = last_tuple[0], last_tuple[1] # 参照上面给的例子
    assert "ost_step" in last_tuple_recording.keys()
    
    if len(last_tuple_recording["ost_step"]) > 0: # 有原始就遍历
        solution_trace_str = ""
        # step_id 为序号 step1 step2
        for step_id, step_text in last_tuple_recording["ost_step"].items():
            solution_trace_str += f"Step {step_id}: " + step_text + "\n"
            
        # 最后一个step_id 加1 作为下一个step_id
        return solution_trace_str, step_id + 1
    
    else:
        # no one-step thought step yet
        return "", 1
    
    
def




def main():
    
    pass





if __name__ == "__main__":
    main()