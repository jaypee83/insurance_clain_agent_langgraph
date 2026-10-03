"""Offline, deterministic stand-in for GPT-4o-mini (testing/teaching only).

It is a real LangChain chat model: the agents call it through the same
`prompt | llm | PydanticOutputParser` chains, so the graph runs unchanged.
"""
import json
import re
from typing import Any, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

NARRATIVE_FLAGS = {
    r"paid in cash|cash payment": "Payment reportedly made in cash (hard to verify)",
    r"receipts? (were |was )?lost|lost (the )?(original|receipt)": "Original receipts lost; only duplicates supplied",
    r"friend'?s|relative'?s|family member'?s": "Treatment/repair by a related party (conflict of interest)",
    r"no witness": "No independent witnesses",
    r"urgent(ly)? need|settle (it )?quickly|immediate payment": "Pressure for quick settlement",
}


def _block(text: str, label: str) -> str:
    m = re.search(rf"{label}:\s*(.*?)(?:\n[A-Z][A-Z _-]+:|\Z)", text, re.S)
    return m.group(1).strip() if m else ""


class OfflineClaimsLLM(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "offline-claims-heuristic"

    def _generate(self, messages: List[BaseMessage], stop=None, run_manager=None, **kw: Any) -> ChatResult:
        text = "\n".join(str(m.content) for m in messages)
        task = re.search(r"TASK: (\w+)", text).group(1)
        out = getattr(self, "_" + task.lower())(text)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=json.dumps(out, ensure_ascii=False)))])

    def _doc_review(self, text: str) -> dict:
        return {"inconsistencies": [], "notes": "Document fields are consistent with the claim narrative (offline heuristic review)."}

    def _coverage(self, text: str) -> dict:
        perils = [p.strip().lower() for p in _block(text, "COVERED PERILS").split(",")]
        excl = [e.strip() for e in _block(text, "EXCLUSIONS").split(",") if e.strip()]
        incident = _block(text, "INCIDENT TYPE").lower()
        desc = _block(text, "DESCRIPTION").lower()
        hit = next((e for e in excl if re.search(rf"\b{re.escape(e.lower())}\b", desc)), None)  # word boundary: "war" must not match "warehouse"
        covered = incident in perils or any(p in desc for p in perils)
        return {"peril_covered": covered, "exclusion_triggered": hit,
                "confidence": 0.9 if covered else 0.7,
                "notes": (f"Incident '{incident}' is {'a covered peril' if covered else 'not among covered perils'}."
                          + (f" Narrative matches exclusion '{hit}'." if hit else " No exclusion wording matched."))}

    def _fraud_narrative(self, text: str) -> dict:
        desc = _block(text, "CLAIM").lower()
        flags = [msg for rx, msg in NARRATIVE_FLAGS.items() if re.search(rx, desc)]
        risk = min(1.0, 0.35 * len(flags))
        return {"narrative_risk": round(risk, 2), "red_flags": flags,
                "notes": "Narrative red flags detected." if flags else "Narrative is specific and plausible."}

    def _summary(self, text: str) -> dict:
        route = _block(text, "ROUTE").split()[0]
        claim = json.loads(re.search(r"CLAIM:\s*(\{.*\})\s*\n\s*DOCUMENT CHECK", text, re.S).group(1))
        reasons = [r.strip("- ").strip() for r in _block(text, "DECISION REASONS").splitlines() if r.strip()]
        label = {"auto_approve": "recommended for auto-approval", "reject": "recommended for rejection",
                 "human_review": "escalated for human review"}[route]
        from .nodes import _inr
        amt = _inr(claim["claimed_amount"])
        return {
            "headline": f"{claim['claim_type'].title()} claim {claim['claim_id']} for {amt} - {label}",
            "summary": (f"{claim['claimant_name']} filed a {claim['claim_type']} claim ({claim['incident_type']}) for {amt} "
                        f"on {claim['report_date']} for an incident on {claim['incident_date']}. "
                        f"Narrative: {claim['description']} Outcome of automated checks: {label}. "
                        f"Main reasons: {'; '.join(reasons[:3])}."),
            "key_facts": [f"Policy {claim['policy_id']}", f"Claimed amount {amt}",
                          f"Incident {claim['incident_date']}, reported {claim['report_date']}",
                          f"{len(claim.get('documents', []))} documents submitted"],
            "recommended_action": route,
            "reviewer_focus": reasons if route == "human_review" else [],
        }
