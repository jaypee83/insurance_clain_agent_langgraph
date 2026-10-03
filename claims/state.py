"""Graph state.

Each parallel branch writes to its OWN key (doc_check / eligibility / fraud),
so the three concurrent updates never collide. The only shared key written by
several nodes is `audit_log`, which has an `operator.add` reducer so parallel
appends are merged instead of raising InvalidUpdateError.
"""
import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict


class ClaimState(TypedDict, total=False):
    claim: Dict[str, Any]                 # input
    policy: Optional[Dict[str, Any]]      # looked up by intake
    doc_check: Dict[str, Any]             # Document Verification Agent
    eligibility: Dict[str, Any]           # Eligibility Check Agent
    fraud: Dict[str, Any]                 # Fraud Detection Agent
    decision: Dict[str, Any]              # Decision router (route + reasons)
    summary: Dict[str, Any]               # Claim Summary Agent
    human_review: Dict[str, Any]          # Human Approval Agent
    final: Dict[str, Any]                 # final outcome
    audit_log: Annotated[List[str], operator.add]
