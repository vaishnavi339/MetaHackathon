from fastapi import FastAPI
from env.environment import CustomerSupportEnvironment
from env.models import Action

app = FastAPI()

env = CustomerSupportEnvironment(task_id="easy")

@app.post("/reset")
def reset():
    global env
    env = CustomerSupportEnvironment(task_id="easy")
    obs = env.reset()
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
        "info": info
    }