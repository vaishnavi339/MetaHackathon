TASKS = {
    "easy": {
        "id": "easy",
        "task_id": "easy",
        "description": "Easy FAQ resolution",
        "grader": "easy"
    },
    "medium": {
        "id": "medium",
        "task_id": "medium",
        "description": "Angry customer handling",
        "grader": "medium"
    },
    "hard": {
        "id": "hard",
        "task_id": "hard",
        "description": "Multi-step escalation",
        "grader": "hard"
    }
}


def list_tasks():
    return ["easy", "medium", "hard"]
