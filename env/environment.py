from __future__ import annotations

from typing import Dict, Optional, Tuple

from pydantic import ValidationError

from env.grader import normalize_score
from env.models import (
    Action,
    ActionType,
    ConversationTurn,
    CustomerPersonality,
    EpisodeState,
    Observation,
    Reward,
    Sentiment,
    TicketMetadata,
    Urgency,
)
from env.tasks import TASKS, list_tasks


class CustomerSupportEnvironment:
    env_name = "AI Customer Support Simulation Environment"

    def __init__(self, task_id: str = "easy") -> None:
        self.task_id = task_id
        self._state: Optional[EpisodeState] = None

    def available_tasks(self) -> list[str]:
        return list_tasks()

    def available_task_definitions(self) -> list[object]:
        return [TASKS[key] for key in list_tasks()]

    def reset(self, task_id: Optional[str] = None) -> Observation:
        if task_id is not None:
            self.task_id = task_id

        if self.task_id not in TASKS:
            self.task_id = "easy"

        self._state = EpisodeState(
            task_id=self.task_id,
            task_title=f"{self.task_id.title()} Task",
            step_count=0,
            max_steps=5,
            done=False,
            sentiment=Sentiment.NEUTRAL,
            urgency=Urgency.MEDIUM,
            resolution_status="open",
            ticket_metadata=TicketMetadata(
                ticket_id=f"TASK-{self.task_id.upper()}",
                category="general",
                product="OpenEnv Hackathon",
                channel="chat",
                customer_tier="standard",
                personality=CustomerPersonality.PATIENT,
                policy_flags=[],
            ),
            collected_info={},
            revealed_requirements=[],
            conversation_history=[
                ConversationTurn(
                    speaker="customer",
                    message=f"This is the {self.task_id} task.",
                )
            ],
            action_history=[],
            last_reward=None,
            delayed_bonus_bank=0.0,
            metadata={},
        )
        return self._build_observation()

    def state(self) -> EpisodeState:
        if self._state is None:
            raise RuntimeError("Environment must be reset before state() is called.")
        return self._state.model_copy(deep=True)

    def step(self, action: Action | dict) -> Tuple[Observation, Reward, bool, Dict]:
        if self._state is None:
            self.reset(self.task_id)

        try:
            validated_action = action if isinstance(action, Action) else Action.model_validate(action)
        except ValidationError:
            validated_action = Action(
                action_type=ActionType.REPLY,
                message="Fallback action.",
            )

        self._state.step_count += 1
        self._state.action_history.append(validated_action.action_type)
        self._state.conversation_history.append(
            ConversationTurn(
                speaker="agent",
                message=validated_action.message or "Fallback action.",
            )
        )

        task = TASKS.get(self.task_id)

        try:
            score = task.grader(validated_action, self._state)
        except Exception:
            score = 0.2

        reward = Reward(
            score=normalize_score(score),
            correctness=normalize_score(score),
            sentiment_improvement=0.1,
            efficiency=0.5,
            policy_compliance=0.5,
            wrong_action_penalty=0.05,
            repeated_action_penalty=0.05,
            excessive_step_penalty=0.05,
            step_decay=0.05,
            reasoning="Deterministic global task grading.",
        )

        self._state.last_reward = reward
        done = self._state.step_count >= self._state.max_steps
        self._state.done = done
        if done:
            self._state.resolution_status = "resolved"

        observation = self._build_observation()
        info = {
            "task_id": self.task_id,
            "resolution_status": self._state.resolution_status,
        }
        return observation, reward, done, info

    def _build_observation(self) -> Observation:
        if self._state is None:
            raise RuntimeError("Environment must be reset before observations are built.")

        customer_message = self._state.conversation_history[-1].message
        return Observation(
            task_id=self._state.task_id,
            task_title=self._state.task_title,
            customer_message=customer_message,
            sentiment=self._state.sentiment,
            urgency=self._state.urgency,
            conversation_history=self._state.conversation_history,
            ticket_metadata=self._state.ticket_metadata,
            step_count=self._state.step_count,
            max_steps=self._state.max_steps,
            resolution_status=self._state.resolution_status,
        )
