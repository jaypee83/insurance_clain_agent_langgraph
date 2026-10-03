import copy
import os
import sys
from pathlib import Path

import pytest

os.environ["CLAIMS_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langgraph.types import Command  # noqa: E402

from claims.config import load_scenarios  # noqa: E402
from claims.graph import build_graph  # noqa: E402

SCEN = {s["claim"]["claim_id"]: s for s in load_scenarios()}


def run(claim, tid):
    g = build_graph()
    cfg = {"configurable": {"thread_id": tid}}
    g.invoke({"claim": claim}, cfg)
    return g, cfg


@pytest.mark.parametrize("cid", list(SCEN))
def test_scenario_routes(cid):
    g, cfg = run(SCEN[cid]["claim"], cid)
    assert g.get_state(cfg).values["decision"]["route"] == SCEN[cid]["expected"]


def test_checks_run_in_parallel_superstep():
    g = build_graph()
    steps = {}
    for ev in g.stream({"claim": SCEN["CLM-001"]["claim"]}, {"configurable": {"thread_id": "p"}}, stream_mode="debug"):
        if ev["type"] == "task":
            steps[ev["payload"]["name"]] = ev["step"]
    assert steps["document_verification"] == steps["eligibility_check"] == steps["fraud_detection"]
    assert steps["decide"] == steps["fraud_detection"] + 1


def test_human_in_the_loop_pause_and_resume():
    g, cfg = run(SCEN["CLM-005"]["claim"], "hitl")
    st = g.get_state(cfg)
    assert st.next == ("human_approval",)
    assert "High-value" in st.tasks[0].interrupts[0].value["reasons"][0]
    g.invoke(Command(resume={"decision": "approve", "reviewer": "t", "approved_amount": 1_000_000}), cfg)
    final = g.get_state(cfg).values["final"]
    assert final["status"] == "APPROVED (human)" and final["amount_payable"] == 1_000_000


def test_auto_approve_pays_after_deductible():
    g, cfg = run(SCEN["CLM-001"]["claim"], "ded")
    assert g.get_state(cfg).values["final"]["amount_payable"] == 85400 - 5000


def test_unknown_policy_short_circuits():
    c = copy.deepcopy(SCEN["CLM-001"]["claim"])
    c["policy_id"] = "NOPE-9"
    g, cfg = run(c, "nopol")
    v = g.get_state(cfg).values
    assert v["decision"]["route"] == "reject" and "doc_check" not in v


def test_amount_mismatch_escalates():
    c = copy.deepcopy(SCEN["CLM-001"]["claim"])
    c["claimed_amount"] = 120000       # bill still says 85,400
    g, cfg = run(c, "mismatch")
    v = g.get_state(cfg).values
    assert v["decision"]["route"] == "human_review"
    assert any(i["indicator"] == "amount_mismatch" for i in v["fraud"]["indicators"])


def test_word_boundary_exclusion():
    # 'war' exclusion must not fire on 'warehouse' (scenario 5)
    g, cfg = run(SCEN["CLM-005"]["claim"], "wb")
    assert g.get_state(cfg).values["eligibility"]["hard_failures"] == []
