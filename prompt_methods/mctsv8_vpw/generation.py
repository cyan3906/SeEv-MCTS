from prompt_methods.mctsv8.utils_call import (
    build_reward_answer_messages,
    build_reward_subquestion_messages,
    extract_sub_questions_and_answers,
    default_call,
    generate_message,
    reward_call_batch,
)
from prompt_methods.mctsv8.prompts import (
    answer_prefix,
    overall_question_prefix,
    plan_prompt,
    question_prefix,
    subquestion_prefix,
)

from .prompts import DIVERSITY_PROMPT, DIVERSITY_ROLES


def _parse_candidate_response(response, depth):
    sub_question_marker = f"Sub-question {depth}:"
    answer_marker = f"Answer {depth}:"
    sub_question_start = response.rfind(sub_question_marker)
    if sub_question_start < 0:
        return "", ""
    answer_start = response.find(answer_marker, sub_question_start)
    if answer_start < 0:
        return "", ""
    sub_question = response[
        sub_question_start + len(sub_question_marker):answer_start
    ].strip()
    sub_answer = response[answer_start + len(answer_marker):].strip()
    if not sub_question or not sub_answer:
        return "", ""
    return sub_question, sub_answer


def generate_normal_candidates(
    cfg,
    image_url,
    model,
    question,
    sub_questions,
    sub_answers,
    depth,
    width,
):
    user_prompt = plan_prompt + question_prefix.format(question=question)
    assistant_prompt = ""
    for index, sub_question in enumerate(sub_questions, start=1):
        assistant_prompt += (
            f"{subquestion_prefix.format(step=index, subquestion=sub_question)}\n"
        )
        assistant_prompt += (
            f"{answer_prefix.format(step=index, answer=sub_answers[index - 1])}\n\n"
        )

    last_sub_question = depth == cfg.prompt_method.decompose_question_num
    if last_sub_question:
        assistant_prompt += overall_question_prefix.format(
            step=depth, overall_question=""
        )
    else:
        assistant_prompt += subquestion_prefix.format(
            step=depth, subquestion=""
        ).strip()

    stop_token = subquestion_prefix.format(
        step=depth + 1, subquestion=""
    ).strip()
    messages = generate_message(
        cfg,
        user_prompt,
        image_url,
        assistant_prompt=assistant_prompt,
        step=depth,
    )
    responses = default_call(
        cfg,
        messages,
        model,
        "",
        "decompose question",
        n=width,
        continue_final_message=True,
        stop_token=stop_token,
    )
    candidates = []
    for response in responses:
        sub_question, sub_answer = _parse_candidate_response(
            assistant_prompt + response, depth
        )
        if sub_question and sub_answer:
            candidates.append((sub_question, sub_answer))
    return candidates


def _format_history(sub_questions, sub_answers):
    if not sub_questions:
        return "(none)"
    return "\n".join(
        f"Sub-question {index}: {sub_question}\nAnswer {index}: {sub_answers[index - 1]}"
        for index, sub_question in enumerate(sub_questions, start=1)
    )


def _format_candidates(normal_candidates):
    if not normal_candidates:
        return "(none)"
    return "\n".join(
        f"Candidate {index}: {sub_question}\nCandidate answer {index}: {sub_answer}"
        for index, (sub_question, sub_answer) in enumerate(normal_candidates, start=1)
    )


def build_diversity_messages(
    cfg,
    image_url,
    question,
    sub_questions,
    sub_answers,
    normal_candidates,
    depth,
):
    history = _format_history(sub_questions, sub_answers)
    candidates = _format_candidates(normal_candidates)
    return [
        generate_message(
            cfg,
            DIVERSITY_PROMPT.format(
                question=question,
                history=history,
                normal_candidates=candidates,
                role=role,
                depth=depth,
            ),
            image_url,
            step=depth,
        )
        for role in DIVERSITY_ROLES
    ]


def generate_diverse_candidates(
    cfg,
    image_url,
    model,
    question,
    sub_questions,
    sub_answers,
    normal_candidates,
    depth,
):
    messages_batch = build_diversity_messages(
        cfg,
        image_url,
        question,
        sub_questions,
        sub_answers,
        normal_candidates,
        depth,
    )
    response_groups = model.get_response_batch(
        messages_batch,
        n=1,
        continue_final_message=False,
    )
    candidates = []
    for responses in response_groups:
        if not responses:
            continue
        sub_question, sub_answer = _parse_candidate_response(responses[0], depth)
        if sub_question and sub_answer:
            candidates.append((sub_question, sub_answer))
    return candidates


def score_candidates_batch(
    cfg,
    image_url,
    model,
    question,
    sub_questions,
    sub_answers,
    candidates,
):
    if not candidates:
        return []

    question_messages = [
        build_reward_subquestion_messages(
            cfg, image_url, question, sub_questions, candidate_question
        )
        for candidate_question, _ in candidates
    ]
    answer_messages = [
        build_reward_answer_messages(
            cfg,
            image_url,
            question,
            sub_questions,
            sub_answers,
            candidate_question,
            candidate_answer,
        )
        for candidate_question, candidate_answer in candidates
    ]
    rewards = reward_call_batch(
        cfg,
        question_messages + answer_messages,
        model,
        node_depth=len(sub_questions),
    )
    count = len(candidates)
    question_rewards = rewards[:count]
    answer_rewards = rewards[count:]
    return [
        {
            "sub_question_reward": question_reward,
            "sub_answer_reward": answer_reward,
            "base_sub_answer_reward": answer_reward,
            "p3pro_applied": False,
        }
        for question_reward, answer_reward in zip(
            question_rewards, answer_rewards
        )
    ]
