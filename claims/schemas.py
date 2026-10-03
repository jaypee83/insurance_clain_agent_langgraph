"""Structured outputs of the LLM calls inside each agent.

Pattern used by every agent: deterministic rules decide what is *checkable*
(dates, amounts, document presence); the LLM handles what needs *judgement*
(reading the narrative, mapping it to perils/exclusions, spotting
inconsistencies, writing the summary). The LLM never overrides a hard rule.
"""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class DocReviewLLM(BaseModel):
    inconsistencies: List[str] = Field(default_factory=list, description="Inconsistencies between the claim narrative and the documents (e.g., dates, names, event type). Empty if none.")
    notes: str = Field(description="One or two sentences on document quality/consistency")


class CoverageLLM(BaseModel):
    peril_covered: bool = Field(description="True if the incident described is one of the covered perils")
    exclusion_triggered: Optional[str] = Field(default=None, description="Name of the policy exclusion that applies, exactly as listed, or null")
    confidence: float = Field(ge=0, le=1, description="Confidence in this coverage opinion, 0-1")
    notes: str = Field(description="Short reasoning citing the policy wording")


class FraudNarrativeLLM(BaseModel):
    narrative_risk: float = Field(ge=0, le=1, description="0 = narrative fully plausible, 1 = strongly suggests fraud")
    red_flags: List[str] = Field(default_factory=list, description="Specific red flags found in the narrative or documents")
    notes: str = Field(description="One or two sentences of reasoning")


class ClaimSummaryLLM(BaseModel):
    headline: str = Field(description="One-line headline, e.g. 'Health claim ₹85,400 - recommended for auto-approval'")
    summary: str = Field(description="4-6 sentence adjuster-ready summary of the claim and the checks")
    key_facts: List[str] = Field(description="3-6 bullet facts (amounts, dates, policy status)")
    recommended_action: Literal["auto_approve", "reject", "human_review"]
    reviewer_focus: List[str] = Field(default_factory=list, description="What a human reviewer should check (empty if not escalated)")


class HumanDecision(BaseModel):
    decision: Literal["approve", "reject", "request_info"]
    reviewer: str = "claims.officer"
    notes: str = ""
    approved_amount: Optional[float] = None
