def normalize_score(score: float) -> float:
    try:
        score = float(score)
    except Exception:
        return 0.1

    if score <= 0:
        return 0.05
    if score >= 1:
        return 0.95

    return round(score, 4)
