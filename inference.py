from __future__ import annotations

import json
import os
from typing import Any, Optional

print("[START] inference.py")

try:
    from openai import OpenAI
except Exception as e:
    print("[ERROR] OpenAI import failed:", e)
    print("[END] success=false steps=0 rewards=")
    raise SystemExit(0)

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

SYSTEM_PROMPT = """You are operating a customer support environment.
Choose exactly one action from: reply, request_info, escalate.
Return strict JSON only with:
{"action_type":"reply|request_info|escalate","message":"..."}"""


def _format_reward(value: float) -> str:
    return f"{value:.2f}"


def _safe_action_name(action: Any) -> str:
    if action is None:
        return "null"
    try:
        return action.action_type.value
    except Exception:
        return "null"


def _build_client() -> Optional[OpenAI]:
    if not HF_TOKEN:
        print("[ERROR] Missing HF_TOKEN")
        print("[END] success=false steps=0 rewards=")
        raise SystemExit(0)

    try:
        return OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN,
        )
    except Exception as e:
        print("[ERROR] Failed to init OpenAI client:", e)
        print("[END] success=false steps=0 rewards=")
        raise SystemExit(0)


def _choose_action(client: OpenAI, model_name: str, observation: dict[str, Any], action_cls: Any) -> Any:
    try:
        response = client.chat.completions.create(
            model=model_name,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(observation, default=str)},
            ],
        )
    except Exception as e:
        print("[ERROR] API call failed:", e)
        raise SystemExit(0)

    try:
        payload = json.loads(response.choices[0].message.content or "{}")
        return action_cls.model_validate(payload)
    except Exception as e:
        print("[ERROR] Failed to parse model response:", e)
        raise SystemExit(0)


def run_episode(task_id: str) -> None:
    rewards: list[str] = []
    step_number = 0
    success = False
    model_name = MODEL_NAME or "unknown-model"

    print(f"[STEP] stage=init task={task_id} model={model_name}")

    try:
        from env.environment import CustomerSupportEnvironment
        from env.models import Action
    except Exception as e:
        print("[ERROR] Environment import failed:", e)
        print("[END] success=false steps=0 rewards=")
        raise SystemExit(0)

    try:
        client = _build_client()
        env = CustomerSupportEnvironment(task_id=task_id)
        observation = env.reset()
    except SystemExit:
        raise
    except Exception as e:
        print("[ERROR] Failed to initialize environment:", e)
        print("[END] success=false steps=0 rewards=")
        raise SystemExit(0)

    done = False
    while not done:
        step_number += 1
        action = None
        reward_value = 0.1
        error_text = "null"

        try:
            action = _choose_action(client, model_name, observation.model_dump(mode="json"), Action)
            observation, reward, done, _ = env.step(action)
            reward_value = float(reward.score)
            rewards.append(_format_reward(reward_value))
            try:
                success = done and env.state().resolution_status in {"resolved", "escalated"}
            except Exception:
                success = False
        except SystemExit:
            rewards.append(_format_reward(0.1))
            print(
                f"[STEP] step={step_number} action={_safe_action_name(action)} "
                f"reward=0.10 done=true error=api_exit"
            )
            print(f"[END] success=false steps={step_number} rewards={','.join(rewards)}")
            raise
        except Exception as e:
            done = True
            error_text = str(e).replace("\n", " ").strip() or "unknown_error"
            rewards.append(_format_reward(0.1))

        done_text = "true" if done else "false"
        print(
            f"[STEP] step={step_number} action={_safe_action_name(action)} "
            f"reward={_format_reward(reward_value)} done={done_text} error={error_text}"
        )

    success_text = "true" if success else "false"
    print(f"[END] success={success_text} steps={step_number} rewards={','.join(rewards)}")


if __name__ == "__main__":
    try:
        run_episode(task_id=os.getenv("TASK_ID", "hard"))
    except SystemExit:
        raise
    except Exception as e:
        print("[ERROR] Unhandled runtime failure:", e)
        print("[END] success=false steps=0 rewards=")
        raise SystemExit(0)
