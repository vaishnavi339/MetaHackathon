from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from env.environment import CustomerSupportEnvironment
from env.models import Action


SYSTEM_PROMPT = """You are operating a customer support environment.
Choose exactly one action from: reply, request_info, escalate.
Return strict JSON only with:
{"action_type":"reply|request_info|escalate","message":"..."}"""


def build_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["HF_TOKEN"],
        base_url=os.environ["API_BASE_URL"],
    )


def choose_action(client: OpenAI, model_name: str, observation: Dict[str, Any]) -> Action:
    response = client.chat.completions.create(
        model=model_name,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(observation, default=str)},
        ],
    )
    payload = json.loads(response.choices[0].message.content or "{}")
    return Action.model_validate(payload)


def _format_reward(value: float) -> str:
    return f"{value:.2f}"


def _safe_action_name(action: Optional[Action]) -> str:
    if action is None:
        return "null"
    return action.action_type.value


def run_episode(task_id: str) -> None:
    rewards: List[str] = []
    step_number = 0
    success = False
    model_name = os.getenv("MODEL_NAME", "unknown-model")
    print(f"[START] task={task_id} env={CustomerSupportEnvironment.env_name} model={model_name}")

    env: Optional[CustomerSupportEnvironment] = None
    try:
        env = CustomerSupportEnvironment(task_id=task_id)
        client = build_client()
        observation = env.reset()

        done = False
        while not done:
            step_number += 1
            action: Optional[Action] = None
            error: Optional[str] = None
            reward_value = 0.0

            try:
                action = choose_action(client, model_name, observation.model_dump(mode="json"))
                observation, reward, done, _ = env.step(action)
                reward_value = reward.score
                success = done and env.state().resolution_status in {"resolved", "escalated"}
                rewards.append(_format_reward(reward_value))
            except Exception as exc:
                done = True
                error = str(exc).replace("\n", " ").strip() or "unknown_error"
                rewards.append(_format_reward(0.0))

            error_text = "null" if error is None else error
            done_text = "true" if done else "false"
            print(
                f"[STEP] step={step_number} action={_safe_action_name(action)} "
                f"reward={_format_reward(reward_value)} done={done_text} error={error_text}"
            )
    except Exception as exc:
        step_number += 1
        rewards.append(_format_reward(0.0))
        error_text = str(exc).replace("\n", " ").strip() or "unknown_error"
        print(
            f"[STEP] step={step_number} action=null reward=0.00 done=true error={error_text}"
        )

    success_text = "true" if success else "false"
    print(f"[END] success={success_text} steps={step_number} rewards={','.join(rewards)}")


if __name__ == "__main__":
    run_episode(task_id=os.getenv("TASK_ID", "hard_multi_step_escalation"))
