from env.grader import grade_easy, grade_hard, grade_medium


TASKS = {
    "easy": {
        "id": "easy",
        "task_id": "easy",
        "description": "Easy FAQ resolution",
        "grader": grade_easy,
        "grader_fn": grade_easy,
        "grader_path": "env.grader:grade_easy"
    },
    "medium": {
        "id": "medium",
        "task_id": "medium",
        "description": "Angry customer handling",
        "grader": grade_medium,
        "grader_fn": grade_medium,
        "grader_path": "env.grader:grade_medium"
    },
    "hard": {
        "id": "hard",
        "task_id": "hard",
        "description": "Multi-step escalation",
        "grader": grade_hard,
        "grader_fn": grade_hard,
        "grader_path": "env.grader:grade_hard"
    }
}


def list_tasks():
    return ["easy", "medium", "hard"]
