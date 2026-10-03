"""The agents (graph nodes).

Every node is a pure function: state in -> partial state update out.
Rules produce facts; the LLM adds judgement via prompt | llm | parser.
"""
from __future__ import annotations

import json
from datetime import date
from typing import Any, Dict, List

from langchain_core.output_parsers import PydanticOutputParser
from langgraph.types import interrupt

from . import config as C
from .prompts import COVERAGE_PROMPT, DOC_REVIEW_PROMPT, FRAUD_PROMPT, SUMMARY_PROMPT
from .schemas import ClaimSummaryLLM, CoverageLLM, DocReviewLLM, FraudNarrativeLLM, HumanDecision
from .state import ClaimState

POLICIES = C.load_policies()


def _llm_structured(prompt, schema, variables: Dict[str, Any]):
    """prompt | llm | PydanticOutputParser; returns None if the LLM fails so the
    workflow degrades to rules-only instead of crashing (and the decision node
    escalates because confidence is unknown)."""
    parser = PydanticOutputParser(pydantic_object=schema)
    chain = prompt.partial(format_instructions=parser.get_format_instructions()) | C.get_llm() | parser
    try:
        return chain.invoke(variables)
    except Exception as e:  # noqa: BLE001 - we log and degrade gracefully
        print(f"[warn] LLM call for {schema.__name__} failed: {e}")
        return None


def _d(s: str) -> date:
    return date.fromisoformat(s)


def _claim_brief(claim: dict) -> str:
    c = {k: v for k, v in claim.items() if k != "documents"}
    c["documents_submitted"] = [d["doc_type"] for d in claim.get("documents", [])]
    return json.dumps(c, ensure_ascii=False)


def _inr(x: float) -> str:
    """Indian digit grouping: 1850000 -> ₹18,50,000"""
    n = str(int(round(x)))
    head, tail = n[:-3], n[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return "₹" + ",".join(parts + [tail]) if parts else "₹" + tail


# ------------------------------------------------------------------ intake
def intake(state: ClaimState) -> dict:
    claim = state["claim"]
    missing = [f for f in ("claim_id", "policy_id", "claim_type", "incident_date", "report_date", "claimed_amount") if not claim.get(f)]
    policy = POLICIES.get(claim.get("policy_id"))
    msg = f"intake: claim {claim.get('claim_id')} received; policy {'found' if policy else 'NOT FOUND'}"
    if missing:
        msg += f"; missing fields {missing}"
    return {"policy": policy, "audit_log": [msg]}


def route_after_intake(state: ClaimState):
    """Conditional edge #1: fan out to the 3 checks in parallel, or short-circuit."""
    if state.get("policy") is None:
        return "decide"
    return ["document_verification", "eligibility_check", "fraud_detection"]


# ------------------------------------------------------ Agent 1: documents
def document_verification(state: ClaimState) -> dict:
    claim, policy = state["claim"], state["policy"]
    docs = claim.get("documents", [])
    required = C.REQUIRED_DOCS.get(claim["claim_type"], [])
    provided = [d["doc_type"] for d in docs]
    missing = [r for r in required if r not in provided]
    issues: List[str] = []
    holder = policy["holder_name"].lower()
    for d in docs:
        f = d.get("fields", {})
        if f.get("name") and f["name"].lower() != holder:
            issues.append(f"{d['doc_type']}: name '{f['name']}' does not match policy holder '{policy['holder_name']}'")
        if d["doc_type"] in C.AMOUNT_DOCS and f.get("amount"):
            diff = abs(f["amount"] - claim["claimed_amount"]) / max(1, claim["claimed_amount"])
            if diff > C.AMOUNT_TOLERANCE:
                issues.append(f"{d['doc_type']}: document amount {_inr(f['amount'])} differs from claimed {_inr(claim['claimed_amount'])} by {diff:.0%}")
        if f.get("date") and _d(f["date"]) < _d(claim["incident_date"]) and d["doc_type"] not in ("id_proof", "ownership_proof"):
            issues.append(f"{d['doc_type']}: dated {f['date']}, before the incident date")
        if f.get("expiry_date") and _d(f["expiry_date"]) < _d(claim["incident_date"]):
            issues.append(f"{d['doc_type']}: expired on {f['expiry_date']} (before incident)")

    llm = _llm_structured(DOC_REVIEW_PROMPT, DocReviewLLM, {
        "claim": _claim_brief(claim), "documents": json.dumps(docs, ensure_ascii=False)})
    if llm:
        issues += [f"LLM: {i}" for i in llm.inconsistencies]
    result = {"required": required, "provided": provided, "missing": missing, "inconsistencies": issues,
              "complete": not missing, "llm_notes": llm.notes if llm else "LLM unavailable"}
    return {"doc_check": result,
            "audit_log": [f"document_verification: {len(provided)}/{len(required)} required docs, missing={missing}, issues={len(issues)}"]}


# ----------------------------------------------------- Agent 2: eligibility
def eligibility_check(state: ClaimState) -> dict:
    claim, p = state["claim"], state["policy"]
    inc = _d(claim["incident_date"])
    hard: List[str] = []
    warn: List[str] = []
    if p["status"] != "active":
        hard.append(f"Policy status is '{p['status']}'")
    if not (_d(p["start_date"]) <= inc <= _d(p["end_date"])):
        hard.append(f"Incident date {claim['incident_date']} is outside the policy period {p['start_date']} to {p['end_date']} (policy expired or not yet started)")
    if not p.get("premium_paid", False):
        hard.append("Premium not paid")
    if claim["claim_type"] != p["policy_type"]:
        hard.append(f"Claim type '{claim['claim_type']}' does not match policy type '{p['policy_type']}'")

    cov = _llm_structured(COVERAGE_PROMPT, CoverageLLM, {
        "policy_type": p["policy_type"], "perils": ", ".join(p["covered_perils"]),
        "exclusions": ", ".join(p["exclusions"]), "incident_type": claim["incident_type"],
        "description": claim["description"]})
    confidence = cov.confidence if cov else 0.0
    if cov:
        if not cov.peril_covered:
            hard.append(f"Peril '{claim['incident_type']}' is not covered")
        if cov.exclusion_triggered:
            hard.append(f"Exclusion applies: {cov.exclusion_triggered}")

    payable = max(0.0, min(claim["claimed_amount"], p["sum_insured"]) - p["deductible"])
    if claim["claimed_amount"] > p["sum_insured"]:
        warn.append(f"Claimed {_inr(claim['claimed_amount'])} exceeds sum insured {_inr(p['sum_insured'])}; payable capped")
    if cov and cov.confidence < C.MIN_LLM_CONFIDENCE:
        warn.append(f"Low confidence in coverage interpretation ({cov.confidence:.2f})")

    result = {"eligible": not hard, "hard_failures": hard, "warnings": warn, "policy_status": p["status"],
              "sum_insured": p["sum_insured"], "deductible": p["deductible"], "payable_amount": payable,
              "coverage_confidence": confidence, "llm_notes": cov.notes if cov else "LLM unavailable"}
    return {"eligibility": result,
            "audit_log": [f"eligibility_check: eligible={not hard}, failures={len(hard)}, payable={_inr(payable)}"]}


# ---------------------------------------------------------- Agent 3: fraud
FRAUD_WEIGHTS = {"early_claim": 0.30, "near_limit": 0.25, "frequent_claims": 0.20, "late_reporting": 0.15,
                 "amount_mismatch": 0.30, "round_amount": 0.10, "unusually_large": 0.15}


def fraud_detection(state: ClaimState) -> dict:
    claim, p = state["claim"], state["policy"]
    amt = claim["claimed_amount"]
    inc, rep = _d(claim["incident_date"]), _d(claim["report_date"])
    ind: List[Dict[str, Any]] = []

    def add(name, detail):
        ind.append({"indicator": name, "weight": FRAUD_WEIGHTS[name], "detail": detail})

    days_since_start = (inc - _d(p["start_date"])).days
    if 0 <= days_since_start < C.EARLY_CLAIM_DAYS:
        add("early_claim", f"Incident {days_since_start} days after policy inception")
    if amt >= 0.9 * p["sum_insured"]:
        add("near_limit", f"Claim is {amt / p['sum_insured']:.0%} of sum insured")
    if p.get("prior_claims_12m", 0) >= 2:
        add("frequent_claims", f"{p['prior_claims_12m']} prior claims in last 12 months")
    if (rep - inc).days > C.LATE_REPORTING_DAYS:
        add("late_reporting", f"Reported {(rep - inc).days} days after incident")
    for d in claim.get("documents", []):
        a = d.get("fields", {}).get("amount")
        if d["doc_type"] in C.AMOUNT_DOCS and a and abs(a - amt) / amt > C.AMOUNT_TOLERANCE:
            add("amount_mismatch", f"{d['doc_type']} shows {_inr(a)} vs claimed {_inr(amt)}")
            break
    if amt >= 50_000 and amt % 10_000 == 0:
        add("round_amount", f"Suspiciously round amount {_inr(amt)}")
    typical = C.TYPICAL_CLAIM.get(claim["claim_type"], amt)
    if amt > 3 * typical:
        add("unusually_large", f"{amt / typical:.1f}x the typical {claim['claim_type']} claim")

    rule_score = min(1.0, sum(i["weight"] for i in ind))
    nar = _llm_structured(FRAUD_PROMPT, FraudNarrativeLLM, {
        "claim": _claim_brief(claim), "rule_flags": "\n".join(f"- {i['detail']}" for i in ind) or "none"})
    narrative = nar.narrative_risk if nar else 0.0
    score = round(min(1.0, rule_score + 0.3 * narrative), 2)   # LLM can add at most 0.3
    level = "high" if score >= 0.7 else "medium" if score >= C.FRAUD_ESCALATE_SCORE else "low"
    result = {"fraud_score": score, "rule_score": round(rule_score, 2), "narrative_risk": narrative,
              "risk_level": level, "indicators": ind, "llm_red_flags": nar.red_flags if nar else [],
              "llm_notes": nar.notes if nar else "LLM unavailable"}
    return {"fraud": result, "audit_log": [f"fraud_detection: score={score} ({level}), {len(ind)} rule indicators"]}


# ------------------------------------------------------- decision (router)
def decide(state: ClaimState) -> dict:
    """Aggregates the three parallel results. Order of checks = policy priority."""
    claim = state["claim"]
    reasons: List[str] = []
    if state.get("policy") is None:
        route, reasons = "reject", [f"Policy {claim.get('policy_id')} not found"]
    else:
        dc, el, fr = state["doc_check"], state["eligibility"], state["fraud"]
        if dc["missing"]:
            route = "reject"
            reasons.append("Missing required documents: " + ", ".join(dc["missing"]))
        if el["hard_failures"]:
            route = "reject"
            reasons += el["hard_failures"]
        if not reasons:
            if fr["fraud_score"] >= C.FRAUD_ESCALATE_SCORE:
                reasons.append(f"Fraud score {fr['fraud_score']} >= {C.FRAUD_ESCALATE_SCORE} ({', '.join(i['indicator'] for i in fr['indicators'])})")
            if claim["claimed_amount"] >= C.HIGH_VALUE_THRESHOLD:
                reasons.append(f"High-value claim {_inr(claim['claimed_amount'])} >= {_inr(C.HIGH_VALUE_THRESHOLD)} threshold")
            if dc["inconsistencies"]:
                reasons += ["Document issue: " + i for i in dc["inconsistencies"]]
            if el["warnings"]:
                reasons += el["warnings"]
            if el["coverage_confidence"] < C.MIN_LLM_CONFIDENCE:
                reasons.append("Coverage opinion uncertain")
            route = "human_review" if reasons else "auto_approve"
            if route == "auto_approve":
                reasons = ["All documents present and consistent", "Policy active and incident covered",
                           f"Low fraud risk ({fr['fraud_score']})", "Amount below high-value threshold"]
        elif state["fraud"]["fraud_score"] >= C.FRAUD_ESCALATE_SCORE:
            reasons.append(f"Note: fraud score also elevated ({state['fraud']['fraud_score']})")
    return {"decision": {"route": route, "reasons": reasons},
            "audit_log": [f"decide: route={route}"]}


# --------------------------------------------------- Agent 4: claim summary
def claim_summary(state: ClaimState) -> dict:
    d = state["decision"]
    s = _llm_structured(SUMMARY_PROMPT, ClaimSummaryLLM, {
        "route": d["route"], "claim": _claim_brief(state["claim"]),
        "doc_check": json.dumps(state.get("doc_check", {}), ensure_ascii=False),
        "eligibility": json.dumps(state.get("eligibility", {}), ensure_ascii=False),
        "fraud": json.dumps(state.get("fraud", {}), ensure_ascii=False),
        "reasons": "\n".join(f"- {r}" for r in d["reasons"])})
    if s is None:
        s = ClaimSummaryLLM(headline=f"Claim {state['claim']['claim_id']}: {d['route']}", summary="; ".join(d["reasons"]),
                            key_facts=[], recommended_action=d["route"])
    s.recommended_action = d["route"]  # the summary may not override the decision
    return {"summary": s.model_dump(), "audit_log": ["claim_summary: generated"]}


def route_after_summary(state: ClaimState) -> str:
    """Conditional edge #2: only escalated claims go to a human."""
    return "human_approval" if state["decision"]["route"] == "human_review" else "finalize"


# -------------------------------------------------- Agent 5: human approval
def human_approval(state: ClaimState) -> dict:
    """Human-in-the-loop. `interrupt()` persists the state via the checkpointer
    and pauses the run; it resumes when the app calls
    graph.invoke(Command(resume={...}), config) with the reviewer's decision."""
    payload = {
        "claim_id": state["claim"]["claim_id"],
        "headline": state["summary"]["headline"],
        "summary": state["summary"]["summary"],
        "reasons": state["decision"]["reasons"],
        "fraud_score": state["fraud"]["fraud_score"] if state.get("fraud") else None,
        "payable_amount": state["eligibility"]["payable_amount"] if state.get("eligibility") else None,
        "question": "Approve or reject this claim?",
    }
    answer = interrupt(payload)
    decision = HumanDecision(**answer) if isinstance(answer, dict) else HumanDecision(decision=str(answer))
    return {"human_review": decision.model_dump(),
            "audit_log": [f"human_approval: {decision.reviewer} -> {decision.decision} ({decision.notes})"]}


# ------------------------------------------------------------------ finalize
def finalize(state: ClaimState) -> dict:
    route = state["decision"]["route"]
    hr = state.get("human_review")
    payable = state.get("eligibility", {}).get("payable_amount", 0)
    if hr:
        status = {"approve": "APPROVED (human)", "reject": "REJECTED (human)", "request_info": "PENDING - more information requested"}[hr["decision"]]
        amount = (hr.get("approved_amount") or payable) if hr["decision"] == "approve" else 0
    else:
        status = {"auto_approve": "APPROVED (auto)", "reject": "REJECTED (auto)"}[route]
        amount = payable if route == "auto_approve" else 0
    final = {"status": status, "route": route, "amount_payable": amount, "reasons": state["decision"]["reasons"]}
    return {"final": final, "audit_log": [f"finalize: {status}, payable={_inr(amount)}"]}
