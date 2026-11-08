DIVERSITY_ROLES = (
    "Use a different representation or decomposition of the problem.",
    "Develop an independent solution route that does not reuse the listed steps.",
    "Reason in reverse or through verification and counter-checking.",
)


DIVERSITY_PROMPT = """Generate exactly one new sub-question and its answer.

Main problem:
{question}

Reasoning history:
{history}

Already generated candidates:
{normal_candidates}

Required perspective:
{role}

Do not repeat or paraphrase an existing candidate. Output exactly:
Sub-question {depth}: <new sub-question>
Answer {depth}: <answer>
"""
