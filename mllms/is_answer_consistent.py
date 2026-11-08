
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
