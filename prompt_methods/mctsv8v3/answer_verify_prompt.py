double_existing_reasoning_verify_prompt = """
    Provide you with two answer solutions. Please output the final answer.output "The answer is " at the end.
        
    **Output Example:**
    
    **Question**
    What is 2+3*6?
    
    **Solution 1:** 
    Sub-question 1: what is 3*6?
    Answer 1: 18.
    
    Sub-question 2: what is 2+18?
    Answer 2: 20.
    
    Sub-question 3: Now we can answer the question:What is 2+3*6?
    Answer 3: 20.
    
    **Solution 2:**
    Sub-question 1: what is 3*6?
    Answer 1: 18.
    
    Sub-question 2: what is 2+18?
    Answer 2: 21.
    
    Sub-question 3: Now we can answer the question:What is 2+3*6?
    Answer 3: 21.
    
    **Output**
    The answer is 21.
    
    **Question**
    {question}
    
    **Solution 1:**
    {answer1}
    
    **Solution 2:**
    {answer2}
    
    **Output**

"""

double_answer_verify_prompt = """
    Provide you with two answers to this question.Please output the answer that is more accurate.
    **Output Example:**

    **Question:**
    What is 2+3?\n
    Answer 1: 5
    Answer 2: 6

    **Output**
    The more accurate answer is 5.

    **Question:**
    What is sum of 2 and 3?\n
    Answer 1: 5
    Answer 2: 6

    **Output**
    The more accurate answer is 5.

    **Question:**
    {question}\n
    Answer 1: {answer1}
    Answer 2: {answer2}

    **Output**

"""

direct_answer_verify_prompt = """
    Provide you one question and answer. Please judge whether the answer is correct.
    **Output Example:**

    **Question**
    What is 2+3?\n
    Answer: 6

    **Output**
    No.The answer is not correct.

    **Question**
    What is sum of 2 and 3?\n
    Answer: 5

    **Output**
    Yes.The answer is correct.

    **Question**
    {question}\n
    Answer: {answer}

    **Output**

"""

rewrite_answer_prompt = """
    Provide you one question and False Answer. Please output the correct answer.
    **Output Example:**

    **Question**
    What is 2+3?\n
    False Answer: 6

    **Output**
    The correct answer is 5.

    **Question**
    What is sum of 2 and 3?\n
    False Answer: 6

    **Output**
    The correct answer is 5.

    **Question**
    {question}\n
    False Answer: {false_answer}

    **Output**

"""

existing_reasoning_verify_prompt = """
    Please generate the answer directly based on the existing reasoning.Please output "The final answer is " at the end.
    
    **Output Example:**
    
    **Question**
    What is 2+3+2+3?
    
    **Existing Reasoning**
    Sub-question 1: what is 2 + 3?
    Answer 1: The answer is 5.
    
    Sub-question 2: what is 5 + 2?
    Answer 2: The answer is 7.
    
    **Output**
    Sub-question 3: What is 2+3+2?
    Answer 3: The answer is 7.
    
    Sub-question 4: Now we can answer the question:What is 2+3+2+3?
    Answer 4: The answer is 10.
    
    The final answer is 10.
    
    **Question**
    What is 2+3*2?
    
    **Existing Reasoning**
    Sub-question 1: what is 3 * 2?
    Answer 1: The answer is 6.
    
    Sub-question 2: what is 6 + 2?
    Answer 2: The answer is 8.
    
    **Output**
    Sub-question 3: Now we can answer the question:What is 2+3*2?
    Answer 3: The answer is 8.
    
    The final answer is 8.
    
    **Question**
    {question}
    
    **Existing Reasoning**
    {answer}
    
    **Output**

"""


is_answer_consistent_prompt = """
    Act as an **Answer Consistency Evaluator** and strictly follow these requirements:
    1.  Task Goal: Judge whether the **core content of [Answer A] and [Answer B] is completely consistent**, ignoring any format differences.
    2.  Judgment Rules (Priority from High to Low):
        1.  Ignore format differences: Text/JSON/list format, word order, synonyms, punctuation, line breaks, uppercase/lowercase, redundant modifiers are all irrelevant.
        2.  Definition of core consistency: Full match of three elements—key objects (e.g., "different subplot number"), key conclusions (e.g., "split direction is different"), and key data (e.g., "vertical direction").
        3.  If any of the three key elements do not match, mark as "Inconsistent".
    3.  Output Requirements: **Must output strictly in JSON format, with no redundant text**. The JSON structure is fixed as follows:
    {{
        "is_consistent": true/false,
        "explanation": "Explanation for the judgment"
    }}

    [Answer A] {answer1}
    [Answer B] {answer2}
            
"""





