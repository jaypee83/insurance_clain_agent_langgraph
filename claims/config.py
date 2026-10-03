"""Business thresholds + LLM factory (OpenAI if a key exists, offline otherwise)."""
import json
import os
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"

# ---- business rules (tune per insurer) -----------------------------------
HIGH_VALUE_THRESHOLD = 1_000_000      # ₹10 lakh: always needs a human
FRAUD_ESCALATE_SCORE = 0.50           # fraud score at/above this goes to a human
MIN_LLM_CONFIDENCE = 0.60             # below this the coverage opinion is "uncertain"
AMOUNT_TOLERANCE = 0.10               # bill vs claimed amount tolerance
LATE_REPORTING_DAYS = 30
EARLY_CLAIM_DAYS = 30                 # claim soon after inception is a classic red flag
TYPICAL_CLAIM = {"health": 120_000, "motor": 90_000, "property": 500_000}  # for "unusually large" check

REQUIRED_DOCS = {
    "health":   ["claim_form", "id_proof", "hospital_bill", "discharge_summary"],
    "motor":    ["claim_form", "driving_license", "vehicle_rc", "repair_estimate", "police_report"],
    "property": ["claim_form", "id_proof", "ownership_proof", "damage_assessment", "incident_report"],
}
AMOUNT_DOCS = {"hospital_bill", "repair_estimate", "damage_assessment"}


def load_policies() -> dict:
    return {p["policy_id"]: p for p in json.loads((DATA / "policies.json").read_text())}


def load_scenarios() -> list:
    return json.loads((DATA / "scenarios.json").read_text())


def use_openai() -> bool:
    return bool(os.getenv("OPENAI_API_KEY")) and os.getenv("CLAIMS_OFFLINE", "0") != "1"


def get_llm():
    if use_openai():
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0, timeout=60, max_retries=2)
    from .offline import OfflineClaimsLLM
    return OfflineClaimsLLM()


def backend_name() -> str:
    return f"OpenAI ({os.getenv('OPENAI_MODEL', 'gpt-4o-mini')})" if use_openai() else "Offline demo model (rule-based)"
