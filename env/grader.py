from __future__ import annotations

import math
import re
from typing import Callable, Dict, Iterable, Tuple

from env.models import Action, ActionType, EpisodeState, Reward, Sentiment
from env.tasks import SupportTask, get_task

ZERO = 0.0
ONE = 1.0
MIN_SCORE = 0.05
MAX_SCORE = 0.95
FALLBACK_SCORE = 0.1
GRADER_FALLBACK = 0.2


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _keyword_fraction(text: str, keywords: Iterable[str]) -> float:
    normalized = _normalize(text)
    keyword_list = list(keywords)
    if not keyword_list:
        return ZERO
    matches = sum(1 for keyword in keyword_list if keyword in normalized)
    return min(ONE, matches / len(keyword_list))


def normalize_score(score: float) -> float:
    try:
        score = float(score)
    except Exception:
        return FALLBACK_SCORE

    if score != score:
        return FALLBACK_SCORE

    if not math.isfinite(score):
        return FALLBACK_SCORE

    if score <= ZERO:
        return MIN_SCORE

    if score >= ONE:
        return MAX_SCORE

    return round(max(MIN_SCORE, min(score, MAX_SCORE)), 4)


def sentiment_to_score(sentiment: Sentiment) -> float:
    values = {
        Sentiment.ANGRY: ZERO,
        Sentiment.NEGATIVE: 0.33,
        Sentiment.NEUTRAL: 0.66,
        Sentiment.POSITIVE: ONE,
    }
    return values[sentiment]


def sentiment_improvement(previous: Sentiment, current: Sentiment) -> float:
    return max(ZERO, min(ONE, sentiment_to_score(current) - sentiment_to_score(previous)))


def efficiency_score(step_number: int, max_steps: int, solved: bool) -> float:
    if max_steps <= 0:
        return ZERO
    remaining = max(ZERO, (max_steps - step_number + 1) / max_steps)
    return round(min(ONE, remaining + (0.2 if solved else ZERO)), 4)


def policy_compliance(task: SupportTask, action: Action, state: EpisodeState) -> float:
    message = _normalize(action.message or "")
    score = ONE

    if "apology_required" in state.ticket_metadata.policy_flags:
        if state.step_count == 0 and action.action_type == ActionType.REPLY and not any(
            term in message for term in ["sorry", "apolog", "understand"]
        ):
            score -= 0.5

    if "no_promise_without_lookup" in state.ticket_metadata.policy_flags and any(
        phrase in message for phrase in ["refund issued", "it will arrive today", "guarantee delivery"]
    ):
        score -= 0.5

    if "billing_escalation_after_verification" in state.ticket_metadata.policy_flags:
        if action.action_type == ActionType.ESCALATE and not all(
            key in state.collected_info for key in task.required_info_keys
        ):
            score -= 0.7

    if "privacy_verification_required" in state.ticket_metadata.policy_flags:
        if action.action_type == ActionType.ESCALATE and not all(
            key in state.collected_info for key in task.required_info_keys
        ):
            score -= 0.2

    if action.action_type in {ActionType.REPLY, ActionType.REQUEST_INFO} and not action.message:
        score -= 0.5

    return max(ZERO, min(ONE, score))


def wrong_action_penalty(task: SupportTask, action: Action, state: EpisodeState) -> float:
    penalty = ZERO
    message = _normalize(action.message or "")

    if action.action_type == ActionType.ESCALATE and not task.escalation_required:
        penalty += 0.45

    if action.action_type == ActionType.ESCALATE and task.escalation_required and not all(
        key in state.collected_info for key in task.required_info_keys
    ):
        penalty += 0.45

    if action.action_type == ActionType.REQUEST_INFO and not task.required_info_keys:
        penalty += 0.2

    if task.difficulty == "easy" and action.action_type != ActionType.REPLY:
        penalty += 0.15

    if task.difficulty == "medium" and action.action_type == ActionType.REPLY and any(
        phrase in message for phrase in ["read our policy", "not my fault", "wait longer"]
    ):
        penalty += 0.35

    if task.difficulty == "hard" and action.action_type == ActionType.REPLY and "refund" in message:
        penalty += 0.2

    if action.action_type in {ActionType.REPLY, ActionType.REQUEST_INFO} and not action.message:
        penalty += 0.3

    if action.action_type == ActionType.REPLY and action.message:
        relevance = _keyword_fraction(message, task.success_keywords + task.resolution_keywords)
        if relevance < 0.2:
            penalty += 0.25

    if action.action_type == ActionType.REQUEST_INFO and action.message:
        request_relevance = _keyword_fraction(
            message,
            task.required_info_keys + task.resolution_keywords + ["order", "number", "email", "date", "charge"],
        )
        if request_relevance < 0.2:
            penalty += 0.2

    return max(ZERO, min(ONE, penalty))


def repeated_action_penalty(action: Action, state: EpisodeState) -> float:
    if not state.action_history:
        return ZERO
    streak = 0
    for previous_action in reversed(state.action_history):
        if previous_action == action.action_type:
            streak += 1
        else:
            break

    if streak == 0:
        return ZERO

    base = 0.15 if action.action_type == ActionType.REQUEST_INFO else 0.1
    return min(0.35, base * streak)


def excessive_step_penalty(step_number: int, max_steps: int) -> float:
    if step_number <= 1:
        return ZERO
    threshold = max(2, max_steps - 1)
    if step_number <= threshold:
        return ZERO
    return min(0.3, 0.08 * (step_number - threshold))


def step_decay(step_number: int) -> float:
    return min(ONE, 0.05 * step_number)


def _easy_correctness(task: SupportTask, action: Action) -> float:
    if action.action_type != ActionType.REPLY or not action.message:
        return ZERO
    message = _normalize(action.message)
    score = _keyword_fraction(message, task.success_keywords)
    if "forgot password" in message or "reset password" in message:
        score = max(score, 0.7)
    if "email" in message and "link" in message:
        score = max(score, 0.9)
    if "sign-in page" in message and "email" in message and "link" in message:
        score = 0.95
    return min(MAX_SCORE, score)


def _medium_correctness(task: SupportTask, action: Action) -> float:
    message = _normalize(action.message or "")
    if action.action_type == ActionType.REPLY:
        empathy = _keyword_fraction(message, task.empathy_keywords)
        solution = _keyword_fraction(message, ["check", "shipment", "tracking", "order"])
        if "order number" in message:
            solution = max(solution, 0.75)
        return min(ONE, empathy * 0.6 + solution * 0.4)

    if action.action_type == ActionType.REQUEST_INFO:
        if "order" in message and "number" in message:
            return 0.95
        if "order" in message:
            return 0.6
    if action.action_type == ActionType.ESCALATE and "manager" in message:
        return 0.3
    return ZERO


def _hard_correctness(task: SupportTask, action: Action, state: EpisodeState) -> float:
    message = _normalize(action.message or "")
    if action.action_type == ActionType.REPLY:
        empathy = _keyword_fraction(message, task.empathy_keywords)
        issue_ack = _keyword_fraction(message, ["duplicate", "charge", "billing"])
        return min(1.0, empathy * 0.5 + issue_ack * 0.5)

    if action.action_type == ActionType.REQUEST_INFO:
        targets = {
            "account_email": ["email", "account"],
            "charge_date": ["date", "charge"],
        }
        remaining = [key for key in task.required_info_keys if key not in state.collected_info]
        if not remaining:
            return 0.2
        covered = 0
        for key in remaining:
            if any(token in message for token in targets[key]):
                covered += 1
        return min(0.95, covered / len(remaining))

    if action.action_type == ActionType.ESCALATE:
        readiness = 0.95 if all(key in state.collected_info for key in task.required_info_keys) else 0.2
        context = 0.0
        if any(term in message for term in ["billing", "review", "duplicate", "refund"]):
            context = 0.2
        return min(0.95, readiness + context)

    return ZERO


def _grade_easy_task(task: SupportTask, action: Action, state: EpisodeState) -> float:
    del state
    action_correct = _easy_correctness(task, action)
    response_relevant = _keyword_fraction(action.message or "", task.resolution_keywords)
    efficient_steps = ONE if action.action_type == ActionType.REPLY else 0.2

    score = ZERO
    score += 0.5 * action_correct
    score += 0.35 * response_relevant
    score += 0.15 * efficient_steps
    return normalize_score(score)


def _grade_medium_task(task: SupportTask, action: Action, state: EpisodeState) -> float:
    del state
    action_correct = _medium_correctness(task, action)
    response_relevant = _keyword_fraction(action.message or "", task.resolution_keywords + task.success_keywords)
    sentiment_ready = _keyword_fraction(action.message or "", task.empathy_keywords)
    efficient_steps = ONE if action.action_type in {ActionType.REPLY, ActionType.REQUEST_INFO} else 0.3

    score = ZERO
    score += 0.4 * action_correct
    score += 0.25 * response_relevant
    score += 0.25 * sentiment_ready
    score += 0.1 * efficient_steps
    return normalize_score(score)


def _grade_hard_task(task: SupportTask, action: Action, state: EpisodeState) -> float:
    action_correct = _hard_correctness(task, action, state)
    response_relevant = _keyword_fraction(action.message or "", task.resolution_keywords + task.success_keywords)
    info_progress = len(state.collected_info) / max(1, len(task.required_info_keys))
    if action.action_type == ActionType.REQUEST_INFO:
        info_progress = min(ONE, info_progress + 0.5)
    efficient_steps = ONE if action.action_type in {ActionType.REQUEST_INFO, ActionType.ESCALATE} else 0.4

    score = ZERO
    score += 0.35 * action_correct
    score += 0.25 * response_relevant
    score += 0.25 * min(ONE, info_progress)
    score += 0.15 * efficient_steps
    return normalize_score(score)


def grade_easy(action: Action, state: EpisodeState) -> float:
    return _grade_easy_task(get_task("easy"), action, state)


def grade_medium(action: Action, state: EpisodeState) -> float:
    return _grade_medium_task(get_task("medium"), action, state)


def grade_hard(action: Action, state: EpisodeState) -> float:
    return _grade_hard_task(get_task("hard"), action, state)


TASK_GRADERS: Dict[str, Callable[[Action, EpisodeState], float]] = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}


def task_correctness(task: SupportTask, action: Action, state: EpisodeState) -> float:
    difficulty_graders: Dict[str, Callable[[SupportTask, Action, EpisodeState], float]] = {
        "easy": _grade_easy_task,
        "medium": _grade_medium_task,
        "hard": _grade_hard_task,
    }
    grader = difficulty_graders.get(task.difficulty, _grade_hard_task)
    return grader(task, action, state)


def grade(task_id: str, action, state) -> float:
    grader = TASK_GRADERS.get(task_id)

    if grader is None:
        return normalize_score(GRADER_FALLBACK)

    try:
        score = grader(action, state)
    except Exception:
        score = GRADER_FALLBACK

    return normalize_score(score)


def build_reward(
    task: SupportTask,
    action: Action,
    previous_state: EpisodeState,
    new_sentiment: Sentiment,
    solved: bool,
    delayed_bonus: float,
) -> Reward:
    step_number = previous_state.step_count + 1
    correctness = task_correctness(task, action, previous_state)
    sentiment_gain = sentiment_improvement(previous_state.sentiment, new_sentiment)
    efficiency = efficiency_score(step_number, previous_state.max_steps, solved)
    compliance = policy_compliance(task, action, previous_state)

    wrong_penalty = wrong_action_penalty(task, action, previous_state)
    repeat_penalty = repeated_action_penalty(action, previous_state)
    step_pen = excessive_step_penalty(step_number, previous_state.max_steps)
    decay = step_decay(step_number)

    weighted = (
        0.4 * correctness
        + 0.3 * sentiment_gain
        + 0.2 * efficiency
        + 0.1 * compliance
        + delayed_bonus
    )
    raw_score = weighted - wrong_penalty - repeat_penalty - step_pen - decay
    score = normalize_score(raw_score)

    return Reward(
        score=score,
        correctness=round(correctness, 4),
        sentiment_improvement=round(sentiment_gain, 4),
        efficiency=round(efficiency, 4),
        policy_compliance=round(compliance, 4),
        wrong_action_penalty=round(wrong_penalty, 4),
        repeated_action_penalty=round(repeat_penalty, 4),
        excessive_step_penalty=round(step_pen, 4),
        step_decay=round(decay, 4),
        reasoning=(
            "Deterministic weighted reward with task-specific correctness, dynamic sentiment change, "
            "efficiency, compliance, penalties, and delayed bonus."
        ),
    )


def terminal_success(task: SupportTask, state: EpisodeState) -> Tuple[bool, str]:
    if task.difficulty == "easy":
        if state.last_reward and state.last_reward.correctness >= 0.9:
            return True, "FAQ resolved with a complete reset path."
        return False, "Need a direct password reset answer."

    if task.difficulty == "medium":
        if "order_number" in state.collected_info and state.sentiment in {Sentiment.NEUTRAL, Sentiment.POSITIVE}:
            return True, "Customer de-escalated and order lookup information collected."
        return False, "Need empathy, better tone, and the order number."

    if task.escalation_required:
        required_ready = all(key in state.collected_info for key in task.required_info_keys)
        if state.resolution_status == "escalated" and required_ready:
            return True, "Required billing details collected before escalation."
        return False, "Collect required billing details before escalating."

    return False, "Task still in progress."


def extract_requested_info(customer_message: str) -> Dict[str, str]:
    normalized = customer_message.strip()
    lowered = normalized.lower()
    extracted: Dict[str, str] = {}

    if "order number is" in lowered:
        match = re.search(r"order number is\s+([A-Za-z0-9-]+)", normalized, re.IGNORECASE)
        if match:
            extracted["order_number"] = match.group(1)
    if "account email is" in lowered:
        match = re.search(r"account email is\s+([^\s.]+@[^\s.]+\.[^\s.]+)", normalized, re.IGNORECASE)
        if match:
            extracted["account_email"] = match.group(1)
    if "duplicate charge happened on" in lowered:
        match = re.search(r"duplicate charge happened on\s+([A-Za-z]+\s+\d{1,2})", normalized, re.IGNORECASE)
        if match:
            extracted["charge_date"] = match.group(1)

    return extracted
