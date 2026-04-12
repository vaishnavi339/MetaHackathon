from typing import Any

from fastapi import Body, FastAPI

from env.environment import CustomerSupportEnvironment
from env.models import Action
from env.tasks import list_tasks


app = FastAPI()

env = CustomerSupportEnvironment(task_id="easy")


@app.post("/reset")
def reset(task_id: str = "easy", payload: dict[str, Any] | None = Body(default=None)):
    global env
    if payload and isinstance(payload.get("task_id"), str):
        task_id = payload["task_id"]
    if task_id not in ["easy", "medium", "hard"]:
        task_id = "easy"
    env = CustomerSupportEnvironment(task_id=task_id)
    obs = env.reset(task_id=task_id)
    return obs.model_dump()


@app.post("/step")
def step(payload: dict[str, Any] | None = Body(default=None)):
    global env
    request_payload = payload if isinstance(payload, dict) else {}
    if isinstance(request_payload, dict) and "action" in request_payload:
        request_payload = request_payload["action"]
    try:
        action_obj = Action.model_validate(request_payload or {})
    except Exception:
        action_obj = Action(action_type="reply", message="Fallback action.")
    obs, reward, done, info = env.step(action_obj)
    return {
        "observation": obs.model_dump(),
        "reward": reward.score,
        "done": done,
        "info": info,
    }


@app.post("/state")
def state(payload: dict[str, Any] | None = Body(default=None)):
    global env
    if payload and isinstance(payload.get("task_id"), str) and payload["task_id"] != env.task_id:
        env = CustomerSupportEnvironment(task_id=payload["task_id"])
        env.reset(task_id=payload["task_id"])
    return env.state().model_dump()


@app.get("/tasks")
def tasks():
    task_defs = []
    for task in list_tasks():
        task_defs.append(
            {
                "id": task["id"],
                "task_id": task["task_id"],
                "name": task["name"],
                "difficulty": task["difficulty"],
                "description": task["description"],
                "max_steps": task["max_steps"],
                "grader": task["grader_path"],
                "grader_fn": task["grader_path"],
            }
        )

    return {
        "tasks": task_defs,
        "count": len(task_defs),
    }


@app.post("/grader")
def grader(payload: dict[str, Any] | None = Body(default=None)):
    global env
    if payload and isinstance(payload.get("task_id"), str) and payload["task_id"] != env.task_id:
        env = CustomerSupportEnvironment(task_id=payload["task_id"])
        env.reset(task_id=payload["task_id"])

    score = env.get_episode_score()
    return {
        "task_id": env.task_id,
        "score": score,
        "valid": 0.0 < score < 1.0,
    }


@app.get("/")
def root():
    return {
        "status": "ok",
        "environment": "Customer Support Environment",
        "tasks_endpoint": "/tasks",
        "grader_endpoint": "/grader",
    }


def main():
    return app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)
