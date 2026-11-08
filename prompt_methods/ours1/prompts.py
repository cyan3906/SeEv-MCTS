"""
tree search prompt
"""

init_system_prompt = """You are a helpful assistant."""

plan_prompt = """Given a question, please decompose it into sub-questions. For each sub-question, please answer it in a complete sentence, ending with "The answer is". When the original question is answerable, please start the sub-question with "Now we can answer the question:".

**Output Example:**

**Question:** Four years ago, Kody was only half as old as Mohamed. If Mohamed is currently twice as 30 years old, how old is Kody?

Sub-question 1: How old is Mohamed?
Answer 1: He is currently 30 * 2 = 60 years old. The answer is 60.

Sub-question 2: How old was Mohamed four years ago?
Answer 2: Four years ago, he must have been 60 - 4 = 56 years old. The answer is 56.

Sub-question 3: How old is Kody four years ago?
Answer 3: Four years ago, Kody was half as old as Mohamed. So Kody was 56 / 2 = 28 years old. The answer is 28.

Sub-question 4: How old is Kody now?
Answer 4: Kody is 28 + 4 = 32 years old. The answer is 32.

Sub-question 5: Now we can answer the question: How old is Kody?
Answer 5: Kody is currently 32 years old. The final answer is 32.


"""

question_prefix = "**Question:** {question}"
subquestion_prefix = "Sub-question {step}: {subquestion}"
overall_question_prefix = "Sub-question {step}: Now we can answer the question: {overall_question}"
answer_prefix = "Answer {step}: {answer}"

reward_prompt = """
Given a question and some sub-questions, determine whether the last sub-question is useful to answer the question. Output 'Yes' or 'No', and a reason.

**Output Example:**

**Question:** Four years ago, Kody was only half as old as Mohamed. If Mohamed is currently twice as 30 years old, how old is Kody?
Sub-question 1: How old is Mohamed?
Sub-question 2: How old was Mohamed four years ago?
New Sub-question 3: How old was Kody four years ago?
Is the new question useful? Yes. We need the answer to calculate how old is Kody now.

**Question:** Traci and Harris are baking cakes together. Traci has brought flour from her own house and Harris has 400g of flour in his house. Each cake needs 100g of flour and Traci and Harris have created 9 cakes each. How much flour, in grams, did Traci bring from her own house?
New Sub-question 1: How many cakes did Traci bring from her own house?
Is the new question useful? No. The new question is not related to the original question.

**Question:** A quantity surveyor is figuring the construction costs for a couple that wishes to build a house. The costs are as follows: land costs $50 per square meter, bricks cost $100 per 1000 bricks and roof tiles cost $10 per roof tile. If the house they wish to build requires 2000 square meters, 10000 bricks, and 500 roof tiles, how much construction costs are required for this project?
Sub-question 1: How much does the land cost?
Sub-question 2: How much do the bricks cost?
New Sub-question 3: How much do the roof tiles cost?
Is the new question useful? Yes. We need the answer to calculate the total construction costs.

**Question:** Wallace's water heater is twice the size of Catherine's water heater. If the capacity of Wallace's water heater is 40 gallons and it's 3/4 full, calculate the total number of gallons of water they both have if Catherine's water heater is also full with water to 3/4 of its capacity.
Sub-question 1: How much water is in Wallace's water heater?
New Sub-question 2: How much water do they have in total?
Is the new question useful? No. It is too hard to answer the new question based on the current information.


"""
new_subquestion_prefix = "New Sub-question {step}: {subquestion}"
reward_prefix = "Is the new question useful?"

reward_prompt_for_answer = """Given a question and some sub-questions and answers, determine whether the last answer of the last sub-question is correct. Only output 'Yes' or 'No'.

"""

reward_prefix_for_answer = "Is the answer correct?"



# ost_step 的奖励提示

ost_reward_prompt = """
Given a question and some steps , determine whether the last step is useful to answer the question. Output 'Yes' or 'No', and a reason.

**Output Example:**

**Question:** Four years ago, Kody was only half as old as Mohamed. If Mohamed is currently twice as 30 years old, how old is Kody?
Step 1: Determine Mohamed's current age.Mohamed is currently twice as 30 years old, so Mohamed is \(2 \times 30 = 60\) years old.
Step 2: Calculate Mohamed's age four years ago.Four years ago, Mohamed was \(60 - 4 = 56\) years old.
New Step 3: Determine Kody's age four years ago.Four years ago, Kody was only half as old as Mohamed, so Kody was \(56 / 2 = 28\) years old.
Is the new Step useful? Yes. We need the step to calculate Kody's age four years ago..

**Question:** Traci and Harris are baking cakes together. Traci has brought flour from her own house and Harris has 400g of flour in his house. Each cake needs 100g of flour and Traci and Harris have created 9 cakes each. How much flour, in grams, did Traci bring from her own house?
Step 1: Calculate the total amount of flour needed for all the cakes.Each cake requires 100g of flour, and they have made 9 cakes each, so the total amount of flour needed is 9 cakes * 100g/cake = 900g.
New Step 2: Calculate the total amount of flour available from both Traci and Harris.Harris has 400g of flour, and they have made 9 cakes each, so the total amount of flour available is 9 cakes * 100g/cake = 900g.
Is the new Step useful? No. The calculation of the total amount of flour is incorrect. The correct calculation should be: 2 × 9 cakes × 100g/cake = 1800g.

**Question:** A quantity surveyor is figuring the construction costs for a couple that wishes to build a house. The costs are as follows: land costs $50 per square meter, bricks cost $100 per 1000 bricks and roof tiles cost $10 per roof tile. If the house they wish to build requires 2000 square meters, 10000 bricks, and 500 roof tiles, how much construction costs are required for this project?
Step 1: Calculate the cost of the land.
- The cost of land is $50 per square meter.
- The house requires 2000 square meters.
- So, the cost of the land is 2000 * $50 = $100,000.
Step 2: Calculate the cost of the bricks.
- The cost of bricks is $100 per 1000 bricks.
- The house requires 10000 bricks.
- So, the cost of the bricks is 10000 / 1000 * $100 = $1000.
New Step 3: Calculate the cost of the roof tiles.
- The cost of roof tiles is $10 per roof tile.
- The house requires 500 roof tiles.
- So, the cost of the roof tiles is 500 * $10 = $5000.
Is the new Step useful? Yes. We need the step to calculate the total construction costs.

**Question:** Wallace's water heater is twice the size of Catherine's water heater. If the capacity of Wallace's water heater is 40 gallons and it's 3/4 full, calculate the total number of gallons of water they both have if Catherine's water heater is also full with water to 3/4 of its capacity.
Step 1: Identify Wallace's water heater capacity. The problem states it is 40 gallons.
New Step 2: Find Catherine's water heater capacity. Since Wallace's is twice the size of Catherine's, we calculate Catherine's capacity by multiplying Wallace's capacity by 2: 40 gallons × 2 = 80 gallons
Is the new Step useful? No. This contains a logical error: the direction of calculation for the "multiple relationship" is confused, and division should be used instead of multiplication.

"""

new_ost_prefix = "New Step {idx}: {step}"

reward_prefix = "Is the new Step useful?"







