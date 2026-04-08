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


def grade_easy(action, state):
    return normalize_score(0.7)


def grade_medium(action, state):
    return normalize_score(0.8)


def grade_hard(action, state):
    return normalize_score(0.6)


TASK_GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}
