from typing import Any

from fastapi import Body, FastAPI

from env.environment import CustomerSupportEnvironment
from env.models import Action


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
def step(action: dict):
    global env
    action_obj = Action.model_validate(action)
    obs, reward, done, info = env.step(action_obj)
    return {
        "observation": obs.model_dump(),
        "reward": reward.score,
        "done": done,
        "info": info,
    }


def main():
    return app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)
