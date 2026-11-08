SYSTEM_SCORING_INST = """You are a helpful assistant. You will be given a question, the correct answer (ground truth answer), and a model response. Your task is to:
1. Extract the relevant answer from the model's response, ignoring any irrelevant or extra information. 

The input will be formatted as follows:
**Question:** <question>
**Model Response:** <model_response>

Please extract the answer based on the model's response and output the result in the following JSON format:
{"extracted_answer": extracted_answer}
Ensure that the response strictly follows this structure and is valid JSON. Do not generate additional information and ensure that I can parse the json directly from your output.
"""


SCORING_INST = """The input example is provided below:
**Question:** {question}
**Model Response:** {model_response}
"""

USER_SCORING_INST = """You are a helpful assistant. You will be given a **Question**, the **Ground Truth Answer**, and a **Predicted Answer**. Your task is to compare the **Ground Truth Answer** with the **Predicted Answer** and determine whether the **Predicted Answer** is correct. It's acceptable to have different grammar or form. If the **Predicted Answer** is correct, you should say "Yes". If the **Predicted Answer** is incorrect, you should say "No".

**Question:** {question}
**Ground Truth Answer:** {ground_truth}
**Predicted Answer:** {model_response}

Is the **Predicted Answer** correct?
"""

guided_choice = ["Yes", "No"]


system_judge_numeric_inst = """

    You are a strict format evaluator.

    The input will be formatted as follows:
    **Ground Truth Answer:** <ground_truth>
    **extracted_answer:** <model_response>

    Your task is to determine whether BOTH answers are purely numeric.

    Definition of purely numeric:
    - A valid integer (e.g., 5, -3, 42)
    - A valid decimal number (e.g., 3.14, -0.75)
    - A valid fraction written in numeric form (e.g., 1/2, -3/4)
    - Scientific notation (e.g., 1e-3, -2.5e6)

    Not considered numeric:
    - Words (e.g., five, twenty)
    - Expressions with units (e.g., 5 meters, 3 kg)
    - Text mixed with numbers (e.g., answer is 5)
    - Mathematical expressions (e.g., 2+3, sqrt(4))
    - Percent signs (e.g., 50%)

    Rules:
    1. Do NOT interpret meaning.
    2. Do NOT convert words to numbers.
    3. Only judge based on literal format.
    4. If either answer is not purely numeric, return false.

    Output strictly in the following JSON format:

    {"both_numeric": yes or no}

    Do not include any explanation.

"""

judge_numeric_inst = """
    The input example is provided below:
    **Ground Truth Answer:** {ground_truth}
    **extracted_answer:** {model_response}

"""



