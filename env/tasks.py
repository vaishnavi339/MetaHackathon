TASKS = {
    "easy": {
        "id": "easy",
        "task_id": "easy",
        "description": "Easy FAQ resolution",
        "grader": "env.grader:grade_easy",
        "grader_fn": "env.grader:grade_easy"
    },
    "medium": {
        "id": "medium",
        "task_id": "medium",
        "description": "Angry customer handling",
        "grader": "env.grader:grade_medium",
        "grader_fn": "env.grader:grade_medium"
    },
    "hard": {
        "id": "hard",
        "task_id": "hard",
        "description": "Multi-step escalation",
        "grader": "env.grader:grade_hard",
        "grader_fn": "env.grader:grade_hard"
    }
}


def list_tasks():
    return ["easy", "medium", "hard"]
