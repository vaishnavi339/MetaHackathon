class BaseTask:
    def __init__(self, task_id):
        self.task_id = task_id

    def grader(self, action, state):
        from env.grader import grade

        return grade(self.task_id, action, state)


TASKS = {
    "easy": BaseTask("easy"),
    "medium": BaseTask("medium"),
    "hard": BaseTask("hard"),
}


def get_task(task_id):
    return TASKS[task_id]


def list_tasks():
    return ["easy", "medium", "hard"]
