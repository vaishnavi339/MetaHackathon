from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    ANGRY = "angry"


class Urgency(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CustomerPersonality(str, Enum):
    PATIENT = "patient"
    FRUSTRATED = "frustrated"
    DEMANDING = "demanding"


class ActionType(str, Enum):
    REPLY = "reply"
    ESCALATE = "escalate"
    REQUEST_INFO = "request_info"


class ConversationTurn(BaseModel):
    speaker: Literal["customer", "agent", "system"]
    message: str


class TicketMetadata(BaseModel):
    ticket_id: str
    category: str
    product: str
    channel: str
    customer_tier: str
    locale: str = "en-US"
    personality: CustomerPersonality
    policy_flags: List[str] = Field(default_factory=list)


class Observation(BaseModel):
    task_id: str
    task_title: str
    customer_message: str
    sentiment: Sentiment
    urgency: Urgency
    conversation_history: List[ConversationTurn] = Field(default_factory=list)
    ticket_metadata: TicketMetadata
    allowed_actions: List[ActionType] = Field(
        default_factory=lambda: [
            ActionType.REPLY,
            ActionType.ESCALATE,
            ActionType.REQUEST_INFO,
        ]
    )
    step_count: int = 0
    max_steps: int = 5
    resolution_status: str = "open"


class Action(BaseModel):
    action_type: ActionType
    message: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_action_names(cls, value: object) -> object:
        if isinstance(value, dict) and value.get("action_type") == "ask_for_info":
            updated = dict(value)
            updated["action_type"] = "request_info"
            return updated
        return value


class Reward(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    correctness: float = Field(ge=0.0, le=1.0)
    sentiment_improvement: float = Field(ge=0.0, le=1.0)
    efficiency: float = Field(ge=0.0, le=1.0)
    policy_compliance: float = Field(ge=0.0, le=1.0)
    wrong_action_penalty: float = Field(ge=0.0, le=1.0)
    repeated_action_penalty: float = Field(ge=0.0, le=1.0)
    excessive_step_penalty: float = Field(ge=0.0, le=1.0)
    step_decay: float = Field(ge=0.0, le=1.0)
    reasoning: str


class EpisodeState(BaseModel):
    task_id: str
    task_title: str
    step_count: int
    max_steps: int
    done: bool
    sentiment: Sentiment
    urgency: Urgency
    resolution_status: str
    ticket_metadata: TicketMetadata
    collected_info: Dict[str, str] = Field(default_factory=dict)
    revealed_requirements: List[str] = Field(default_factory=list)
    conversation_history: List[ConversationTurn] = Field(default_factory=list)
    action_history: List[ActionType] = Field(default_factory=list)
    last_reward: Optional[Reward] = None
    delayed_bonus_bank: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: Dict[str, str] = Field(default_factory=dict)
