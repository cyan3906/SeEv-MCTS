

rephrasing_question_prompt = """
You are an AI assistant to help me rephrase questions by splitting the question context into conditions. In your rephrased question, remember to fully express the information in the original question.

Original Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Rephrased Question: Given a list of conditions, please answer the question. Condition 1: Olivia starts with $23. Condition 2: She buys five bagels, each costing $3. Question: How much money does Olivia have remaining after her purchase?

Original Question: Michael had 58 golf balls. On Tuesday, he lost 23 golf balls. On Wednesday, he lost 2 more. How many golf balls did he have at the end of Wednesday?
Rephrased Question: Given a list of conditions, please answer the question. Condition 1: Michael initially has 58 golf balls. Condition 2: On Tuesday, he loses 23 golf balls. Condition 3: On Wednesday, he loses 2 additional golf balls. Question: What is the total number of golf balls Michael has left at the end of Wednesday?

Original Question: One pound of watermelon costs 2 yuan. Xiaoming bought 4 + 6/10 pounds of watermelon. How much did he spend in total?
Rephrased Question: One pound of watermelon costs 2 yuan. Xiaoming bought 4.6 pounds of watermelon. How much did he spend in total?

Original Question: One kilogram of apples costs 5 yuan. Lily bought 3 + 8/10 kilograms of apples. How much did she spend in total?
Rephrased Question: One kilogram of apples costs 5 yuan. Lily bought 3.8 kilograms of apples. How much did she spend in total?

Original Question:{question}
Rephrased Question:

"""