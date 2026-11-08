double_existing_reasoning_verify_prompt = """
    Provide you with two answers. Please output the final answer.output "The answer is " at the end.
        
    **Output Example:**
    
    **Question**
    What is 2+3*6?
    
    **Answer 1:** 
    Let us think step by step.
    Step 1: 2+3=5.
    Step 2: 5*6=30.
    Step 3: The answer is 30.
    
    **Answer 2:**
    Let us think step by step.
    Step 1: 3*6=18.
    Step 2: 2+18=20.
    Step 3: The answer is 20.
    
    **Output**
    The answer is 20.
    
    **Question**
    What is 2+3+4?
    
    **Answer 1:** 
    Let us think step by step.
    Step 1: 2+3=5.
    Step 2: 5+4=10.
    Step 3: The answer is 10.
    
    **Answer 2:**
    Let us think step by step.
    Step 1: 2+3=5.
    Step 2: 5+4=9.
    Step 3: The answer is 9.
    
    **Output**
    The answer is 9.
    
    **Question**
    {question}
    
    **Answer 1:**
    {answer1}
    
    **Answer 2:**
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
    Please generate the answer directly based on the existing reasoning.Please output "The answer is " at the end.
    
    **Output Example:**
    
    **Question**
    What is 2+3+2?
    
    **Existing Reasoning**
    Let us think step by step.
    Step 1: 2 + 3 = 5.
    Step 2: 5 + 2 = 7.
    
    **Output**
    The answer is 7.
    
    **Question**
    What is 2+3*2?
    
    **Existing Reasoning**
    Let us think step by step.
    Step 1: 3 * 2 = 6.
    Step 2: 6 + 2 = 8.

    **Output**
    The answer is 8.
    
    **Question**
    {question}
    
    **Existing Reasoning**
    {answer}
    
    **Output**

"""




