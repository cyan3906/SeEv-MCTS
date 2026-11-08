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

new_ost_prefix = "New Step {idx}: {step}"

ost_reward_prefix = "Is the new Step useful?"

ost_reward_prompt2 = """
Given a question and some steps , determine whether the last step is logical and correct. Output 'Yes' or 'No', and a reason.

**Output Example:**

**Question:** What is the value of 2×(3+4) - 5?
Step 1: calculate the expression inside the parentheses: 3+4=7.
Step 2: perform the multiplication: 2×7=14.
New Step 3: subtract 5: 14-5=9.
Is the new Step logical and correct? Yes. Follows PEMDAS/BODMAS; 14-5=9 is mathematically correct.

**Question:** If all dogs are mammals and some mammals are carnivores, can we conclude that some dogs are carnivores?
Step 1: Premise 1: All dogs are mammals (D ⊂ M).
Step 2: Some mammals are carnivores (M ∩ C ≠ ∅).
New Step 3: Some dogs are carnivores (D ∩ C ≠ ∅).
Is the new Step logical and correct? No. Premises don’t confirm "some mammals" (carnivores) include dogs; conclusion isn’t guaranteed.

**Question:** Is it possible for a human to breathe underwater without any equipment?
Step 1: Humans rely on lungs to breathe, which extract oxygen from air.
Step 2: Water contains dissolved oxygen, but human lungs cannot efficiently absorb oxygen from water.
New Step 3: Therefore, humans cannot breathe underwater without equipment.
Is the new Step logical and correct? Yes. Logically consistent with biology: Steps 1-2 establish lung limitations; conclusion follows without contradiction.

**Question:** Xiao Ming bought 20 cakes. Starting from the first day, he eats twice as many cakes as the previous day. How many cakes does he have left after three days?
Step 1: Assume Xiao Ming eats 1 cake on the first day.
New Step 2: The answer is 1 cake.
Is the new Step logical and correct? No. The is no series of steps.
"""

# 写一版相关性的奖励提示

ost_reward_prompt2= """
Given a question and some steps , determine whether the last step is related to the question. Output 'Yes' or 'No', and a reason.

**Output Example:**

**Question:** What is the value of 3×(5+2) - 4?
Step 1: Calculate the parentheses: 5+2=7.
Step 2: Multiply: 3×7=21.
New Step 3: 21-4=17.
Is the new Step related to the question? Yes. The last step directly computes the final result of the question.

**Question:** Find the area of a rectangle with length 6cm and width 4cm?
Step 1: Recall the area formula: Area = length × width.
New Step 2: Calculate the area: 6cm × 4cm = 24cm².
Is the new Step related to the question? No. The last step computes perimeter instead of the required area, irrelevant to the question.

**Question:** A factory produces 1200 widgets per day. On Monday, it operates at full capacity; on Tuesday, it reduces production by 20% due to maintenance; on Wednesday, it increases Tuesday’s production by 15% to meet demand. If each widget is sold for $3.5, what is the total revenue from the three days of production?
Step 1: Calculate Monday’s production: 1200 widgets (full capacity).
Step 2: Calculate Tuesday’s production: 1200 × (1 - 20%) = 1200 × 0.8 = 960 widgets.
Step 3: Calculate Wednesday’s production: 960 × (1 + 15%) = 960 × 1.15 = 1104 widgets.
Step 4: Total production over three days: 1200 + 960 + 1104 = 3264 widgets.
New Step 5: Total revenue = 3264 × 3.5 = $11,424.
Is the new Step related to the question? Yes. The step is fully relevant to the problem’s objective.

**Question:** A car travels from City A to City B at an average speed of 60 km/h, taking 4 hours to reach. On the return trip from City B to City A, the car’s average speed increases by 20% due to less traffic. If the fuel consumption of the car is 8 liters per 100 km, how many liters of fuel are consumed for the round trip?
Step 1: Calculate the distance between City A and City B: Speed × Time = 60 × 4 = 240 km.
Step 2: Calculate the return trip speed: 60 × (1 + 20%) = 72 km/h.
Step 3: Calculate the return trip time: Distance ÷ Speed = 240 ÷ 72 ≈ 3.333 hours.
Step 4: Total round trip distance: 240 × 2 = 480 km.
New Step 5: Calculate the average speed for the round trip: Total distance ÷ Total time = 480 ÷ (4 + 3.333) ≈ 65.45 km/h.
Is the new Step related to the question? No. The step is unrelated to the question’s objective.
"""
# 写第二版相关性的奖励提示
ost_reward_prompt = """
Given a question and some steps , determine whether the last step is related to the question. Output 'Yes' or 'No', and a reason.

**Output Example:**

**Question:** What is the value of 3×(5+2) - 4?
Step 1: Calculate the parentheses: 5+2=7.
Step 2: Multiply: 3×7=21.
New Step 3: 21-4=17.
Is the new Step related to the question? Yes. The last step directly computes the final result of the question.

**Question:** Find the area of a rectangle with length 6cm and width 4cm?
Step 1: Recall the area formula: Area = length × width.
New Step 2: Calculate the area: 6cm × 4cm = 24cm².
Is the new Step related to the question? No. The last step computes perimeter instead of the required area, irrelevant to the question.

**Question:** A bakery sells croissants for $1.2 each. If Tom buys 15 croissants and pays with a $20 bill, how much change will he get?
Step 1: Calculate total cost: 15 × 1.2 = $18.
New Step 2: Change = 20 - 18 = $2.
Is the new Step related to the question? Yes. The step is fully relevant to the problem’s objective.

**Question:** A supermarket has 28 bottles of mineral water. If 9 bottles are sold each day, can they sell out all bottles in 3 days?
Step 1: Calculate total bottles sold in 3 days by multiplying daily sales by days, then compare with total stock.
New Step 2: Calculate the number of bottles left after 2 days: 28 - 9 - 9 = 10 bottles.
Is the new Step related to the question? No. New Step 2 computes remaining bottles after 2 days instead of total sales in 3 days (as required by Step 1), failing to address whether all bottles are sold out in 3 days.
"""

