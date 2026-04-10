def normalize_score(score):
    try:
        score = float(score)
    except:
        return 0.1

    if score <= 0:
        return 0.05
    if score >= 1:
        return 0.95

    return round(score, 4)


def grade_easy(response, expected):
    \"\"\"Keyword-based for password reset instructions.\"\"\"
    keywords = ['password', 'reset', 'forgot', 'email', 'link', 'sign-in']
    message = (response.message or '').lower()
    matches = sum(1 for kw in keywords if kw in message)
    score = min(0.9, 0.3 + 0.1 * matches) if matches > 0 else 0.2
    return normalize_score(score)


def grade_medium(response, expected):
    \"\"\"Empathy + request info for angry customer.\"\"\"
    empathy_words = ['sorry', 'understand', 'frustrated', 'apologize', 'help']
    message = (response.message or '').lower()
    empathy_score = sum(1 for word in empathy_words if word in message) / len(empathy_words)
    action_bonus = 0.4 if hasattr(response, 'action_type') and response.action_type == 'request_info' else 0.0
    score = 0.3 + 0.4 * empathy_score + action_bonus
    return normalize_score(score)


def grade_hard(response, expected):
    \"\"\"Multi-step: collect info then escalate.\"\"\"
    # Simulate step progress via expected state
    collected = len(expected.get('collected_info', {}))
    revealed = len(expected.get('revealed_requirements', []))
    progress = min(0.8, 0.2 + 0.1 * collected + 0.1 * revealed)
    if hasattr(response, 'action_type') and response.action_type == 'escalate' and progress > 0.4:
        progress += 0.2
    score = progress
    return normalize_score(score)


TASK_GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}
