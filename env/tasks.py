class EasyTask:
    task_id = "easy"

    def grader(self, action, state):
        return 0.7


class MediumTask:
    task_id = "medium"

    def grader(self, action, state):
        return 0.8


class HardTask:
    task_id = "hard"

    def grader(self, action, state):
        return 0.6


TASKS = {
    "easy": EasyTask(),
    "medium": MediumTask(),
    "hard": HardTask(),
}


def list_tasks():
    return ["easy", "medium", "hard"]
