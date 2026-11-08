
plan_prompt = """Given a question, please decompose it into sub-questions. For each sub-question, please answer it in a complete sentence, ending with "The answer is". When the original question is answerable, please start the sub-question with "Now we can answer the question:" For the generation of sub-questions, no answers or any explanations are required.

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