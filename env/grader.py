def normalize_score(score: float) -> float:
    try:
        score = float(score)
    except Exception:
        return 0.01

    if score != score:
        return 0.01

    if score <= 0.0:
        return 0.01
    if score >= 1.0:
        return 0.99
    return score


def grade_easy(response, expected):
    """Keyword-based for password reset instructions."""
    keywords = ['password', 'reset', 'forgot', 'email', 'link', 'sign-in']
    message = (response.message or '').lower()
    matches = sum(1 for kw in keywords if kw in message)
    score = min(0.9, 0.3 + 0.1 * matches) if matches > 0 else 0.2
    return normalize_score(score)


def grade_medium(response, expected):
    """Empathy + request info for angry customer."""
    empathy_words = ['sorry', 'understand', 'frustrated', 'apologize', 'help']
    message = (response.message or '').lower()
    empathy_score = sum(1 for word in empathy_words if word in message) / len(empathy_words)
    action_type = getattr(response, 'action_type', '')
    action_name = getattr(action_type, 'value', str(action_type))
    action_bonus = 0.4 if action_name == 'request_info' else 0.0
    score = 0.3 + 0.4 * empathy_score + action_bonus
    return normalize_score(score)


def grade_hard(response, expected):
    """Multi-step: collect info then escalate."""
    # Support both dict-like task state and EpisodeState objects.
    if isinstance(expected, dict):
        collected_info = expected.get('collected_info', {})
        revealed_requirements = expected.get('revealed_requirements', [])
    else:
        collected_info = getattr(expected, 'collected_info', {})
        revealed_requirements = getattr(expected, 'revealed_requirements', [])

    collected = len(collected_info or {})
    revealed = len(revealed_requirements or [])
    progress = min(0.8, 0.2 + 0.1 * collected + 0.1 * revealed)
    action_type = getattr(response, 'action_type', '')
    action_name = getattr(action_type, 'value', str(action_type))
    if action_name == 'escalate' and progress > 0.4:
        progress += 0.2
    score = progress
    return normalize_score(score)


TASK_GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}

# IMPORTANT: force validator detection
_ = TASK_GRADERS
