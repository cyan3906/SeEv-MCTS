
# double_answer_verify_prompt = """
# Given one question and two solutions.Please judge whether Solution 1 is superior to Solution 2 in terms of answer correctness, and output yes or no.

# **Output Example:**

# **Question:**
# There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?

# **Solution 1**
# Let's think step by step.
# Step 1: Identify the initial number of trees in the grove. The problem states the initial number is 15.
# Step 2: Identify the final number of trees after planting. The problem states the final number is 21.
# Step 3: Calculate the number of trees planted by subtracting the initial number from the final number.
# Step 4: Perform the subtraction: 21 - 15 = 6.
# Step 5: The answer is 6.

# **Solution 2**
# Let's think step by step.
# Step 1: Identify the initial number of trees. The problem states there are 15 trees.
# Step 2: Identify the final number of trees. The problem states there will be 21 trees.
# Step 3: Use multiplication to calculate the number of trees planted by setting the unknown number as x, so the equation is 15 × x = 21.
# Step 4: Solve the equation: x = 21 ÷ 15 = 1.4.
# Step 5: The answer is 6.

# """

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

direct_answer_verify_prefix = """
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


rewrite_answer_prefix = """
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

