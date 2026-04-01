from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from env.environment import CustomerSupportEnvironment
from env.models import Action


app = FastAPI(title="Customer Support OpenEnv Space")

_env = CustomerSupportEnvironment(task_id="easy")


class ResetRequest(BaseModel):
    task_id: str = Field(default="easy")


class StepRequest(BaseModel):
    action: Dict[str, Any]


@app.get("/")
def healthcheck() -> Dict[str, str]:
    return {
        "status": "ok",
        "message": "Customer Support Environment is running.",
    }


@app.post("/reset")
def reset(request: ResetRequest | None = None) -> Dict[str, Any]:
    task_id = "easy" if request is None else request.task_id
    try:
        observation = _env.reset(task_id=task_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return observation.model_dump(mode="json")


@app.post("/step")
def step(request: StepRequest) -> Dict[str, Any]:
    try:
        action = Action.model_validate(request.action)
        observation, reward, done, info = _env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "observation": observation.model_dump(mode="json"),
        "reward": reward.model_dump(mode="json"),
        "done": done,
        "info": info,
    }
