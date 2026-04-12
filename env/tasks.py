from env.grader import grade_easy, grade_hard, grade_medium


TASKS = {
    "easy": {
        "id": "easy",
        "task_id": "easy",
        "name": "FAQ Resolution",
        "difficulty": "easy",
        "description": "Resolve a password reset question with accurate self-serve guidance.",
        "max_steps": 5,
        "grader": grade_easy,
        "grader_fn": grade_easy,
        "grader_path": "env.grader:grade_easy",
    },
    "medium": {
        "id": "medium",
        "task_id": "medium",
        "name": "Angry Customer Handling",
        "difficulty": "medium",
        "description": "De-escalate an upset customer and ask for the missing order details.",
        "max_steps": 5,
        "grader": grade_medium,
        "grader_fn": grade_medium,
        "grader_path": "env.grader:grade_medium",
    },
    "hard": {
        "id": "hard",
        "task_id": "hard",
        "name": "Multi-Step Billing Escalation",
        "difficulty": "hard",
        "description": "Collect the required billing context and escalate at the right time.",
        "max_steps": 5,
        "grader": grade_hard,
        "grader_fn": grade_hard,
        "grader_path": "env.grader:grade_hard",
    },
}


def list_task_ids():
    return list(TASKS.keys())


def list_tasks():
    return [TASKS[key] for key in list_task_ids()]
