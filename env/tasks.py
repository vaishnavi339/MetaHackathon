from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field

from env.models import CustomerPersonality, Sentiment, TicketMetadata, Urgency


class SupportTask(BaseModel):
    task_id: str
    title: str
    difficulty: str
    description: str
    customer_message: str
    initial_sentiment: Sentiment
    urgency: Urgency
    max_steps: int
    ticket_metadata: TicketMetadata
    success_keywords: List[str] = Field(default_factory=list)
    empathy_keywords: List[str] = Field(default_factory=list)
    resolution_keywords: List[str] = Field(default_factory=list)
    required_info_keys: List[str] = Field(default_factory=list)
    hidden_requirements: List[str] = Field(default_factory=list)
    escalation_required: bool = False
    ideal_action_sequence: List[str] = Field(default_factory=list)
    follow_up_customer_messages: Dict[int, str] = Field(default_factory=dict)
    metadata: Dict[str, str] = Field(default_factory=dict)
    resolution_hint: str = ""


FAQ_TASK = SupportTask(
    task_id="easy_faq_resolution",
    title="Easy FAQ Resolution",
    difficulty="easy",
    description="Resolve a straightforward password reset ticket without escalation.",
    customer_message="Hi, I forgot my password. How can I reset it and get back into my account?",
    initial_sentiment=Sentiment.NEUTRAL,
    urgency=Urgency.LOW,
    max_steps=3,
    ticket_metadata=TicketMetadata(
        ticket_id="T-FAQ-1001",
        category="account_access",
        product="Meta Support Portal",
        channel="chat",
        customer_tier="standard",
        personality=CustomerPersonality.PATIENT,
        policy_flags=["self_service_allowed"],
    ),
    success_keywords=["reset", "password", "forgot password", "email", "link"],
    resolution_keywords=["sign-in page", "email", "reset link"],
    ideal_action_sequence=["reply"],
    resolution_hint="Direct the customer to the forgot-password flow and mention the email reset link.",
)


ANGRY_CUSTOMER_TASK = SupportTask(
    task_id="medium_angry_customer",
    title="Medium Angry Customer Handling",
    difficulty="medium",
    description="Handle a delayed delivery complaint with empathy and targeted troubleshooting.",
    customer_message=(
        "I am furious. My order was supposed to arrive three days ago and nobody has fixed this. "
        "What is going on?"
    ),
    initial_sentiment=Sentiment.ANGRY,
    urgency=Urgency.HIGH,
    max_steps=4,
    ticket_metadata=TicketMetadata(
        ticket_id="T-ORD-2048",
        category="order_delay",
        product="Meta Smart Glasses",
        channel="chat",
        customer_tier="priority",
        personality=CustomerPersonality.FRUSTRATED,
        policy_flags=["apology_required", "no_promise_without_lookup"],
    ),
    success_keywords=["sorry", "understand", "delay", "check", "order"],
    empathy_keywords=["sorry", "understand", "frustrating", "apologize"],
    resolution_keywords=["order number", "check the shipment", "tracking"],
    required_info_keys=["order_number"],
    ideal_action_sequence=["reply", "request_info", "reply"],
    follow_up_customer_messages={
        2: "My order number is 48291. I just need someone to check the shipment.",
    },
    resolution_hint="Acknowledge frustration, apologize, request the order number, and describe next steps.",
)


HARD_ESCALATION_TASK = SupportTask(
    task_id="hard_multi_step_escalation",
    title="Hard Multi-Step Escalation",
    difficulty="hard",
    description=(
        "Resolve a duplicate billing charge through a multi-step conversation with hidden requirements "
        "before escalating to billing operations."
    ),
    customer_message=(
        "I've been charged twice for the same subscription this month. I need this fixed immediately."
    ),
    initial_sentiment=Sentiment.NEGATIVE,
    urgency=Urgency.CRITICAL,
    max_steps=5,
    ticket_metadata=TicketMetadata(
        ticket_id="T-BILL-8871",
        category="billing_dispute",
        product="Meta Verified Subscription",
        channel="chat",
        customer_tier="business",
        personality=CustomerPersonality.DEMANDING,
        policy_flags=["billing_escalation_after_verification", "privacy_verification_required"],
    ),
    success_keywords=["billing", "duplicate", "charge", "refund", "escalate"],
    empathy_keywords=["sorry", "understand", "billing", "duplicate charge"],
    resolution_keywords=["billing team", "duplicate charge", "review", "refund"],
    required_info_keys=["account_email", "charge_date"],
    hidden_requirements=["mention_billing_review", "collect_required_info_before_escalation"],
    escalation_required=True,
    ideal_action_sequence=["reply", "request_info", "request_info", "escalate"],
    follow_up_customer_messages={
        2: "The account email is alex@example.com.",
        3: "The duplicate charge happened on March 28.",
    },
    metadata={"department": "billing_operations"},
    resolution_hint=(
        "Acknowledge the duplicate charge, gather account email and charge date, then escalate to billing."
    ),
)


TASK_REGISTRY: Dict[str, SupportTask] = {
    FAQ_TASK.task_id: FAQ_TASK,
    ANGRY_CUSTOMER_TASK.task_id: ANGRY_CUSTOMER_TASK,
    HARD_ESCALATION_TASK.task_id: HARD_ESCALATION_TASK,
}

TASK_ALIASES: Dict[str, str] = {
    "easy": FAQ_TASK.task_id,
    "medium": ANGRY_CUSTOMER_TASK.task_id,
    "hard": HARD_ESCALATION_TASK.task_id,
}


def get_task(task_id: str) -> SupportTask:
    task_id = TASK_ALIASES.get(task_id, task_id)
    if task_id not in TASK_REGISTRY:
        raise KeyError(f"Unknown task_id: {task_id}")
    return TASK_REGISTRY[task_id]


def list_tasks() -> List[SupportTask]:
    return [TASK_REGISTRY[key] for key in sorted(TASK_REGISTRY)]
