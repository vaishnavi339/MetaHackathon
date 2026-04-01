---
title: Customer Support Env
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---






# AI Customer Support Simulation Environment

This repository provides a production-ready OpenEnv environment for training and evaluating agents on realistic customer support workflows. The environment is deterministic, fully typed with Pydantic, and designed for hackathon-style evaluation where stable scoring, real-world utility, and clean interfaces matter.

## Why This Is Useful

Customer support is a strong real-world benchmark for RL and agent systems because good performance depends on more than factual QA. An agent must understand intent, preserve context, manage user sentiment, follow policy, and escalate only when appropriate. That combination makes this environment a practical benchmark for training, evaluating, and comparing support-oriented LLM agents.

This environment simulates real customer support decision-making and can be used to benchmark LLM agents in production-like workflows.

## What This Environment Implements

- `reset()` returns the initial observation
- `step(action)` returns `(observation, reward, done, info)`
- `state()` returns the full current environment state
- Typed Pydantic models for `Observation`, `Action`, and `Reward`
- Exactly three tasks: easy, medium, and hard
- Deterministic graders with partial credit and reproducible reward signals
- Strict inference logging format required for external evaluation harnesses

## Directory Layout

```text
env/
  models.py
  environment.py
  tasks.py
  grader.py
openenv.yaml
inference.py
Dockerfile
README.md
```

## Environment Design

The domain is customer support automation. The agent must read a live ticket state, respond appropriately, maintain context across the conversation, and choose between direct support, information gathering, and escalation.

### Observation Fields

Each observation includes:

- `customer_message`
- `sentiment`
- `urgency`
- `conversation_history`
- `ticket_metadata`
- `allowed_actions`
- `step_count`
- `max_steps`
- `resolution_status`

### Action Space

The action model supports exactly these action types:

- `reply`
- `request_info`
- `escalate`

### Reward Model

The environment uses a weighted deterministic reward:

```text
reward =
    0.4 * correctness +
    0.3 * sentiment_improvement +
    0.2 * efficiency +
    0.1 * policy_compliance
    + delayed_bonus
    - wrong_action_penalty
    - repeated_action_penalty
    - excessive_step_penalty
    - step_decay
```

The reward is clipped to `0.0..1.0`.

This design keeps reward informative across the full trajectory instead of making learning depend only on sparse terminal outcomes.

### Reward Components

- `correctness`: task-specific quality of the chosen action and message
- `sentiment_improvement`: deterministic change in customer emotional state
- `efficiency`: reward for solving earlier and using steps well
- `policy_compliance`: adherence to support workflow policies
- `wrong_action_penalty`: penalizes bad choices like premature escalation
- `repeated_action_penalty`: penalizes repeating the same action without progress
- `excessive_step_penalty`: penalizes late-stage inefficiency
- `step_decay`: global step cost using `0.05 * step_number`

## Innovative Features

To improve realism and evaluation value, the environment includes:

- Dynamic sentiment changes
- Delayed reward effects for setting up later success
- Customer personality types that affect state transitions
- Hidden constraints that only become relevant if the agent gathers the right context
- A realistic escalation policy that rewards correct timing and penalizes premature handoff

## Tasks

There are exactly three tasks.

### 1. Easy: FAQ Resolution

- Scenario: password reset
- Goal: provide the correct self-serve reset instructions
- Grader style: deterministic keyword-based grading
- Typical successful path: answer directly with the reset flow and email link in one turn

### 2. Medium: Angry Customer Handling

- Scenario: delayed order with an upset customer
- Goal: show empathy, stabilize sentiment, and request the order number
- Grader style: deterministic tone-aware and solution-aware grading
- Typical successful path: empathetic reply followed by `request_info` for the order number

### 3. Hard: Multi-Step Escalation

- Scenario: duplicate billing charge
- Goal: collect hidden required details, preserve policy compliance, then escalate correctly
- Grader style: deterministic multi-step escalation logic
- Typical successful path: acknowledge issue, gather account email, gather charge date, escalate to billing review

## Baseline Results

Using a hand-authored strong policy and the deterministic transitions in this repo, the environment produces stable successful trajectories such as:

- Easy: resolved in 1 step with reward around `0.75`
- Medium: resolved in 2 steps with rewards around `0.53,0.74`
- Hard: escalated correctly in 4 steps with rewards around `0.65,0.51,0.37,0.57`

These baseline numbers are intended as reproducible sanity checks for local regression testing, not leaderboard claims.

## Local Usage

### Python Example

```python
from env.environment import CustomerSupportEnvironment
from env.models import Action

env = CustomerSupportEnvironment(task_id="easy_faq_resolution")
observation = env.reset()

observation, reward, done, info = env.step(
    Action(
        action_type="reply",
        message="Use the forgot password option on the sign-in page, then follow the reset link sent to your email."
    )
)
```

### Inference Variables

Set:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`
- optional `TASK_ID`

Then run:

```bash
python inference.py
```

### Required Inference Log Format

The script prints only these lines:

```text
[START] task=<task> env=<env> model=<model>
[STEP] step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> rewards=<r1,r2,...>
```

## Docker

Build:

```bash
docker build -t ai-customer-support-env .
```

Run:

```bash
docker run --rm \
  -e API_BASE_URL="$API_BASE_URL" \
  -e MODEL_NAME="$MODEL_NAME" \
  -e HF_TOKEN="$HF_TOKEN" \
  -e TASK_ID="hard_multi_step_escalation" \
  ai-customer-support-env
```

## Hugging Face Spaces Deployment

This repository is ready for a Docker-based Hugging Face Space:

1. Create a new Docker Space.
2. Upload this repository.
3. Add `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` as Space secrets.
4. Launch the Space.

## Validation Notes

- All graders are deterministic and reproducible.
- There are exactly three tasks.
- Reward scores remain in `0.0..1.0`.
- State transitions are rule-based with no randomness.
- The official action space is `reply`, `request_info`, and `escalate`.
- Legacy `ask_for_info` input is normalized to `request_info` for compatibility.
