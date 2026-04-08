from __future__ import annotations

import json
import os

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

try:
    from openai import OpenAI
except Exception as e:
    print("[START] inference")
    print("[END] success=false error=OpenAI_import_failed")
    raise SystemExit(0)


def _end_with_error(error: str) -> None:
    print("[END] success=false error=" + error)
    raise SystemExit(0)


def _safe_score(value: object) -> str:
    try:
        score = float(value)
    except Exception:
        score = 0.1
    if score <= 0.0:
        score = 0.05
    if score >= 1.0:
        score = 0.95
    return f"{score:.2f}"


def _safe_action_name(action: object) -> str:
    try:
        action_type = getattr(action, "action_type", "reply")
        return getattr(action_type, "value", str(action_type))
    except Exception:
        return "reply"


def _choose_action(client: OpenAI, prompt: str, action_cls) -> object:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
    except Exception:
        return action_cls.model_validate(
            {
                "action_type": "reply",
                "message": "I understand the issue and I will help with the next step.",
            }
        )

    try:
        content = response.choices[0].message.content
    except Exception:
        content = "fallback"

    try:
        payload = json.loads(content) if content and content != "fallback" else {}
    except Exception:
        payload = {}

    if not isinstance(payload, dict):
        payload = {}

    payload.setdefault("action_type", "reply")
    payload.setdefault("message", "I understand the issue and I will help with the next step.")

    try:
        return action_cls.model_validate(payload)
    except Exception:
        return action_cls.model_validate(
            {
                "action_type": "reply",
                "message": "I understand the issue and I will help with the next step.",
            }
        )


def main() -> None:
    print("[START] inference")

    if not HF_TOKEN:
        _end_with_error("missing_token")

    try:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN,
        )
    except Exception as e:
        _end_with_error("client_init_failed")

    try:
        from env.environment import CustomerSupportEnvironment
        from env.models import Action
    except Exception:
        _end_with_error("environment_import_failed")

    task_id = os.getenv("TASK_ID", "easy")

    try:
        env = CustomerSupportEnvironment(task_id=task_id)
        observation = env.reset()
    except Exception:
        _end_with_error("environment_init_failed")

    final_score = "0.10"
    success = False
    max_steps = 5

    try:
        state = env.state()
        max_steps = max(1, int(getattr(state, "max_steps", 5)))
    except Exception:
        max_steps = 5

    try:
        for _ in range(max_steps):
            prompt = (
                "Choose exactly one action from: reply, request_info, escalate. "
                "Return strict JSON with action_type and message. "
                f"Task: {task_id}. "
                f"Observation: {json.dumps(observation.model_dump(mode='json'), default=str)}"
            )
            action = _choose_action(client, prompt, Action)
            print(f"[STEP] task={task_id} action={_safe_action_name(action)}")

            try:
                observation, reward, done, info = env.step(action)
                final_score = _safe_score(getattr(reward, "score", 0.1))
            except Exception:
                final_score = _safe_score(0.1)
                done = True
                info = {}

            if done:
                try:
                    status = info.get("resolution_status", "")
                    success = status in {"resolved", "escalated"}
                except Exception:
                    success = False
                break
    except Exception:
        _end_with_error("runtime_failure")

    print(f"[END] success={'true' if success else 'false'} score={final_score}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        print("[START] inference")
        print("[END] success=false error=unexpected_failure")
        raise SystemExit(0)
