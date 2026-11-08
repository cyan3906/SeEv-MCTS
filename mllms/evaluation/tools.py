SYSTEM_SCORING_INST = """I am providing you a response from a model to a math problem, termed 'Model Response'. You should extract the answer from the response as 'Extracted Answer'. Directly output the extracted answer with no explanation.
    
    1.
    Model response: 'Rounded to two decimal places, the perimeter of the sector is approximately:\n\n(-2, 1)'
    Extracted Answer: (-2, 1)
    
    2.
    Model response: 'at those points.\n\nTherefore, the correct option that represents the meaning of the intersection points of the graphs is:\n\nD. They give the solutions to the equation $f(t)=g(t)$.",'
    Extracted Answer: D
    
    3.
    Model response: ' at 1 (there's a closed circle at y = 1), the range in interval notation is \\((-4, 1]\\).\n\nFinal values:\nDomain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)'
    Extracted Answer: Domain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)
    
    4.
    Model response: 'As it stands, I cannot provide the correct option letter because there isn't enough information to solve for 'y'.'
    Extracted Answer: null
    
    5.
    Model response: 'Given that AB = 17.6 meters, we can now substitute into the equation:\n\nd = 17.6 / cos(38\u00b0)\n\nTherefore, to one decimal place, the distance d between Ned and Bart is approximately 22.3 meters.'
    Extracted answer: 22.3 meters
    
    6.
    Model response:  have all the coefficients for the quadratic function:\n\\( f(x) = ax^2 + bx + c \\)\n\\( f(x) = -1x^2 - 2x + 1 \\)\n\nTherefore, the equation for the graphed function \\( f \\) is:\n\\( f(x) = -x^2 - 2x + 1 \\)"'
    Extracted answer: f(x) = -x^2 - 2x + 1
    
    7.

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


# USER_SCORING_INST = """You are a helpful assistant. You will be given a **Ground Truth Answer**, and a **Predicted Answer**. Your task is to compare the **Ground Truth Answer** with the **Predicted Answer** and determine whether the **Predicted Answer** is correct. It's acceptable to have different grammar or form. If the **Predicted Answer** is correct, you should say "Yes". If the **Predicted Answer** is incorrect, you should say "No".

# **Ground Truth Answer:** {ground_truth}
# **Predicted Answer:** {model_response}

# Is the **Predicted Answer** correct?
# """

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

    If both answers are purely numeric, return true.Else return false.
    
    Do not include any explanation.

"""

judge_numeric_inst = """
    The input example is provided below:
    **Ground Truth Answer:** {ground_truth}
    **extracted_answer:** {model_response}

"""


Mistral_SYSTEM_SCORING_INST = """You are a helpful assistant. You will be given a model response. Your task is to:
1. Extract the relevant answer from the model's response, ignoring any irrelevant or extra information. 

The input will be formatted as follows:
**Question:** <question>
**Model Response:** <model_response>

Please extract the answer based on the model's response and output the result in the following JSON format:
{"extracted_answer": extracted_answer}
Ensure that the response strictly follows this structure and is valid JSON. Do not generate additional information and ensure that I can parse the json directly from your output.
"""


Mistral_SCORING_INST = """
The input example is provided below:
**Model Response:** {model_response}

"""
