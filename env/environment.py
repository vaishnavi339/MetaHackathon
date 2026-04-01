from __future__ import annotations

from copy import deepcopy
from typing import Dict, Optional, Tuple

from pydantic import ValidationError

from env.grader import build_reward, extract_requested_info, terminal_success
from env.models import (
    Action,
    ActionType,
    ConversationTurn,
    CustomerPersonality,
    EpisodeState,
    Observation,
    Reward,
    Sentiment,
)
from env.tasks import SupportTask, get_task, list_tasks


class CustomerSupportEnvironment:
    """Production-ready OpenEnv environment for customer support automation."""

    env_name = "AI Customer Support Simulation Environment"

    def __init__(self, task_id: str = "easy_faq_resolution") -> None:
        self.task_id = task_id
        self.task = get_task(task_id)
        self._state: Optional[EpisodeState] = None

    def available_tasks(self) -> list[SupportTask]:
        return list_tasks()

    def reset(self, task_id: Optional[str] = None) -> Observation:
        if task_id is not None:
            self.task_id = task_id
            self.task = get_task(task_id)

        opening_turn = ConversationTurn(speaker="customer", message=self.task.customer_message)
        self._state = EpisodeState(
            task_id=self.task.task_id,
            task_title=self.task.title,
            step_count=0,
            max_steps=self.task.max_steps,
            done=False,
            sentiment=self.task.initial_sentiment,
            urgency=self.task.urgency,
            resolution_status="open",
            ticket_metadata=self.task.ticket_metadata.model_copy(deep=True),
            collected_info={},
            revealed_requirements=[],
            conversation_history=[opening_turn],
            action_history=[],
            last_reward=None,
            delayed_bonus_bank=0.0,
            metadata=deepcopy(self.task.metadata),
        )
        return self._build_observation(self.task.customer_message)

    def state(self) -> EpisodeState:
        if self._state is None:
            raise RuntimeError("Environment must be reset before state() is called.")
        return self._state.model_copy(deep=True)

    def step(self, action: Action | dict) -> Tuple[Observation, Reward, bool, Dict]:
        if self._state is None:
            raise RuntimeError("Environment must be reset before step() is called.")
        if self._state.done:
            raise RuntimeError("Episode already completed. Call reset() to begin a new one.")
        try:
            validated_action = action if isinstance(action, Action) else Action.model_validate(action)
        except ValidationError as exc:
            return self._handle_invalid_action(exc)

        previous_state = self._state.model_copy(deep=True)
        self._state.step_count += 1
        agent_message = validated_action.message or self._default_message(validated_action.action_type)
        self._state.action_history.append(validated_action.action_type)
        self._state.conversation_history.append(
            ConversationTurn(speaker="agent", message=agent_message)
        )

        if validated_action.action_type == ActionType.ESCALATE:
            self._state.resolution_status = "escalated"

        customer_follow_up = self._generate_follow_up(validated_action, previous_state)
        if customer_follow_up:
            self._state.conversation_history.append(
                ConversationTurn(speaker="customer", message=customer_follow_up)
            )
            self._state.collected_info.update(extract_requested_info(customer_follow_up))

        self._update_hidden_requirements(validated_action)
        new_sentiment = self._simulate_sentiment(validated_action, previous_state)
        self._state.sentiment = new_sentiment

        provisional_success, terminal_reason = terminal_success(self.task, self._state)
        delayed_bonus = self._compute_delayed_bonus(validated_action, previous_state, provisional_success)
        reward = build_reward(
            task=self.task,
            action=validated_action,
            previous_state=previous_state,
            new_sentiment=new_sentiment,
            solved=provisional_success,
            delayed_bonus=delayed_bonus,
        )
        self._state.last_reward = reward

        solved, terminal_reason = terminal_success(self.task, self._state)
        out_of_steps = self._state.step_count >= self._state.max_steps
        self._state.done = solved or out_of_steps
        if solved and self._state.resolution_status == "open":
            self._state.resolution_status = "resolved"
        if out_of_steps and not solved and self._state.resolution_status == "open":
            self._state.resolution_status = "timeout"

        latest_customer_message = customer_follow_up or self._latest_customer_message()
        observation = self._build_observation(latest_customer_message)
        info = {
            "task_id": self.task.task_id,
            "difficulty": self.task.difficulty,
            "terminal_reason": terminal_reason,
            "resolution_status": self._state.resolution_status,
            "required_info_collected": sorted(self._state.collected_info.keys()),
            "revealed_requirements": list(self._state.revealed_requirements),
        }
        return observation, reward, self._state.done, info

    def _handle_invalid_action(self, exc: ValidationError) -> Tuple[Observation, Reward, bool, Dict]:
        if self._state is None:
            raise RuntimeError("Environment must be reset before invalid actions can be handled.")

        self._state.step_count += 1
        step_decay = min(1.0, 0.05 * self._state.step_count)
        reward = Reward(
            score=0.0,
            correctness=0.0,
            sentiment_improvement=0.0,
            efficiency=max(0.0, (self._state.max_steps - self._state.step_count + 1) / max(1, self._state.max_steps)),
            policy_compliance=0.0,
            wrong_action_penalty=1.0,
            repeated_action_penalty=0.0,
            excessive_step_penalty=0.0,
            step_decay=round(step_decay, 4),
            reasoning="Invalid action payload received; step penalized deterministically.",
        )
        self._state.last_reward = reward
        self._state.done = self._state.step_count >= self._state.max_steps
        if self._state.done:
            self._state.resolution_status = "timeout"

        observation = self._build_observation(self._latest_customer_message())
        info = {
            "task_id": self.task.task_id,
            "difficulty": self.task.difficulty,
            "terminal_reason": "Invalid action payload.",
            "resolution_status": self._state.resolution_status,
            "required_info_collected": sorted(self._state.collected_info.keys()),
            "revealed_requirements": list(self._state.revealed_requirements),
            "error": str(exc).replace("\n", " "),
        }
        return observation, reward, self._state.done, info

    def _build_observation(self, customer_message: str) -> Observation:
        if self._state is None:
            raise RuntimeError("Environment must be reset before observations are built.")
        return Observation(
            task_id=self.task.task_id,
            task_title=self.task.title,
            customer_message=customer_message,
            sentiment=self._state.sentiment,
            urgency=self._state.urgency,
            conversation_history=self._state.conversation_history,
            ticket_metadata=self._state.ticket_metadata,
            step_count=self._state.step_count,
            max_steps=self._state.max_steps,
            resolution_status=self._state.resolution_status,
        )

    def _simulate_sentiment(self, action: Action, previous_state: EpisodeState) -> Sentiment:
        message = (action.message or "").lower()
        current = previous_state.sentiment
        personality = previous_state.ticket_metadata.personality

        if self.task.difficulty == "easy":
            if action.action_type == ActionType.REPLY and "reset" in message and "email" in message:
                return Sentiment.POSITIVE
            if action.action_type == ActionType.ESCALATE:
                return Sentiment.NEGATIVE
            return current

        if self.task.difficulty == "medium":
            if action.action_type == ActionType.REPLY and any(
                token in message for token in ["sorry", "understand", "frustrating", "apolog"]
            ):
                return Sentiment.NEGATIVE
            if action.action_type == ActionType.REQUEST_INFO and "order" in message and "number" in message:
                return Sentiment.NEUTRAL
            if action.action_type == ActionType.ESCALATE:
                return Sentiment.NEGATIVE
            return Sentiment.ANGRY if personality == CustomerPersonality.FRUSTRATED else current

        if action.action_type == ActionType.REPLY and any(
            token in message for token in ["sorry", "billing", "duplicate", "review"]
        ):
            return Sentiment.NEGATIVE
        if action.action_type == ActionType.REQUEST_INFO:
            return Sentiment.NEUTRAL
        if action.action_type == ActionType.ESCALATE:
            ready = all(
                key in self._state.collected_info for key in self.task.required_info_keys
            )
            return Sentiment.NEUTRAL if ready else Sentiment.NEGATIVE
        return current

    def _generate_follow_up(self, action: Action, previous_state: EpisodeState) -> Optional[str]:
        if action.action_type not in {ActionType.REPLY, ActionType.REQUEST_INFO}:
            return None
        next_step = previous_state.step_count + 1
        return self.task.follow_up_customer_messages.get(next_step)

    def _update_hidden_requirements(self, action: Action) -> None:
        if self._state is None:
            return
        message = (action.message or "").lower()
        if (
            action.action_type == ActionType.ESCALATE
            and all(key in self._state.collected_info for key in self.task.required_info_keys)
            and "collect_required_info_before_escalation" not in self._state.revealed_requirements
        ):
            self._state.revealed_requirements.append("collect_required_info_before_escalation")
        if action.action_type == ActionType.ESCALATE and any(
            token in message for token in ["billing", "review", "refund"]
        ):
            if "mention_billing_review" not in self._state.revealed_requirements:
                self._state.revealed_requirements.append("mention_billing_review")

    def _compute_delayed_bonus(
        self,
        action: Action,
        previous_state: EpisodeState,
        solved: bool,
    ) -> float:
        if self._state is None:
            return 0.0

        bonus = 0.0
        if action.action_type == ActionType.REQUEST_INFO and any(
            key not in previous_state.collected_info for key in self.task.required_info_keys
        ):
            bonus += 0.05

        if solved and self.task.difficulty == "hard":
            bonus += 0.1
            if set(self._state.revealed_requirements) >= set(self.task.hidden_requirements):
                bonus += 0.05

        self._state.delayed_bonus_bank = min(1.0, previous_state.delayed_bonus_bank + bonus)
        return min(0.15, self._state.delayed_bonus_bank if solved else bonus)

    def _latest_customer_message(self) -> str:
        if self._state is None:
            return self.task.customer_message
        for turn in reversed(self._state.conversation_history):
            if turn.speaker == "customer":
                return turn.message
        return self.task.customer_message

    @staticmethod
    def _default_message(action_type: ActionType) -> str:
        defaults = {
            ActionType.REPLY: "I understand the issue and I am here to help.",
            ActionType.REQUEST_INFO: "Please share the information needed to continue.",
            ActionType.ESCALATE: "I am escalating this to the correct specialist team.",
        }
        return defaults[action_type]
