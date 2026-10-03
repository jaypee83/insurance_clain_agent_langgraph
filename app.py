"""Insurance Claim Processing Agent - Streamlit UI (LangGraph + human-in-the-loop).

    streamlit run app.py
"""
import copy
import json
import os
import uuid
import warnings

import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Claim Processing Agent", page_icon="🛡️", layout="wide")

@st.cache_resource
def owner_key() -> str:
    """The app owner's key, read ONCE from Streamlit secrets (or the environment).
    It is never shown in the page, so visitors cannot see it."""
    try:
        k = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:  # no secrets file (e.g. running locally)
        k = ""
    return (k or os.getenv("OPENAI_API_KEY", "")).strip()


with st.sidebar:
    st.header("⚙️ Settings")
    OWNER = owner_key()
    if OWNER:
        st.success("Using the app owner's OpenAI key")
    user_key = st.text_input("Your own OpenAI API key (optional)", type="password", value="",
                             help="Leave empty to use the owner's key if one is set, otherwise the offline demo model.")
    offline = st.checkbox("Use offline demo model (no API calls)", value=not (OWNER or user_key))
    key = "" if offline else (user_key.strip() or OWNER)
    if key:
        os.environ["OPENAI_API_KEY"] = key
    else:
        os.environ.pop("OPENAI_API_KEY", None)

from langgraph.types import Command  # noqa: E402

from claims import config as C  # noqa: E402
from claims.graph import build_graph, graph_dot  # noqa: E402
from claims.nodes import POLICIES, _inr  # noqa: E402

with st.sidebar:
    st.caption(f"Backend: **{C.backend_name()}**")
    st.markdown(f"**Business rules**  \nHigh-value threshold: {_inr(C.HIGH_VALUE_THRESHOLD)}  \n"
                f"Fraud escalation score: {C.FRAUD_ESCALATE_SCORE}  \nMin. LLM confidence: {C.MIN_LLM_CONFIDENCE}")

# One compiled graph + one checkpointer per browser session, so paused claims survive reruns.
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
    st.session_state.claims = {}          # thread_id -> {"claim":..., "status":...}
graph = st.session_state.graph

NODE_LABEL = {"intake": "📥 Intake", "document_verification": "📄 Document Verification Agent",
              "eligibility_check": "✅ Eligibility Check Agent", "fraud_detection": "🕵️ Fraud Detection Agent",
              "decide": "🔀 Decision router", "claim_summary": "📝 Claim Summary Agent",
              "human_approval": "🧑‍⚖️ Human Approval Agent", "finalize": "🏁 Finalize"}

st.title("🛡️ Insurance Claim Processing Agent")
st.caption("LangGraph workflow: parallel verification, eligibility and fraud checks, then conditional routing "
           "(auto-approve, reject, or human approval).")

tab_new, tab_queue, tab_batch, tab_graph = st.tabs(["➕ Process a claim", "🧑‍⚖️ Human review queue",
                                                    "🧪 Run all test scenarios", "🗺️ Workflow graph"])


def run_claim(claim: dict):
    tid = f"{claim['claim_id']}-{uuid.uuid4().hex[:6]}"
    cfg = {"configurable": {"thread_id": tid}}
    log = st.container(border=True)
    log.markdown("**Live agent trace**")
    for upd in graph.stream({"claim": claim}, cfg, stream_mode="updates"):
        for node, val in upd.items():
            if node == "__interrupt__":
                log.warning("⏸ Workflow paused, waiting for human approval")
            else:
                msg = (val or {}).get("audit_log", [""])[-1] if isinstance(val, dict) else ""
                log.write(f"{NODE_LABEL.get(node, node)}: {msg.split(': ', 1)[-1]}")
    paused = bool(graph.get_state(cfg).next)
    st.session_state.claims[tid] = {"claim": claim, "status": "awaiting review" if paused else "done"}
    return tid


def show_result(tid: str):
    cfg = {"configurable": {"thread_id": tid}}
    v = graph.get_state(cfg).values
    d, s = v["decision"], v["summary"]
    route_badge = {"auto_approve": ":green[AUTO-APPROVE]", "reject": ":red[REJECT]", "human_review": ":orange[HUMAN REVIEW]"}
    st.subheader(s["headline"])
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**Route**  \n{route_badge[d['route']]}")
    c2.metric("Fraud score", v.get("fraud", {}).get("fraud_score", "-"))
    c3.metric("Payable (after deductible)", _inr(v.get("eligibility", {}).get("payable_amount", 0)))
    st.write(s["summary"])
    st.markdown("**Decision reasons**\n" + "\n".join(f"- {r}" for r in d["reasons"]))
    a, b, c = st.columns(3)
    if v.get("doc_check"):
        with a.expander("📄 Document verification", expanded=True):
            dc = v["doc_check"]
            st.write("Required:", ", ".join(dc["required"]))
            st.write("Missing:", ", ".join(dc["missing"]) or "none")
            st.write("Issues:", dc["inconsistencies"] or "none")
            st.caption(dc["llm_notes"])
    if v.get("eligibility"):
        with b.expander("✅ Eligibility", expanded=True):
            e = v["eligibility"]
            st.write("Eligible:", e["eligible"])
            st.write("Failures:", e["hard_failures"] or "none")
            st.write("Warnings:", e["warnings"] or "none")
            st.caption(e["llm_notes"])
    if v.get("fraud"):
        with c.expander("🕵️ Fraud detection", expanded=True):
            f = v["fraud"]
            st.progress(min(1.0, f["fraud_score"]), text=f"score {f['fraud_score']} ({f['risk_level']})")
            if f["indicators"]:
                st.dataframe(pd.DataFrame(f["indicators"]), hide_index=True)
            st.write("LLM red flags:", f["llm_red_flags"] or "none")
    if v.get("final"):
        fin = v["final"]
        (st.success if "APPROVED" in fin["status"] else st.error if "REJECTED" in fin["status"] else st.info)(
            f"**Final status: {fin['status']}**, amount payable {_inr(fin['amount_payable'])}")
        if v.get("human_review"):
            st.caption(f"Reviewer {v['human_review']['reviewer']}: {v['human_review']['notes']}")
    with st.expander("Audit log"):
        st.code("\n".join(v["audit_log"]))


def review_panel(tid: str):
    cfg = {"configurable": {"thread_id": tid}}
    state = graph.get_state(cfg)
    if not state.next:
        return
    payload = state.tasks[0].interrupts[0].value
    with st.container(border=True):
        st.markdown(f"### 🧑‍⚖️ Human approval required: {payload['claim_id']}")
        st.write(payload["summary"])
        st.markdown("**Why it was escalated:**\n" + "\n".join(f"- {r}" for r in payload["reasons"]))
        with st.form(f"review-{tid}"):
            decision = st.radio("Decision", ["approve", "reject", "request_info"], horizontal=True)
            amount = st.number_input("Approved amount (₹)", value=float(payload.get("payable_amount") or 0), step=1000.0)
            reviewer = st.text_input("Reviewer", "claims.officer")
            notes = st.text_area("Notes / justification")
            if st.form_submit_button("Submit decision and resume workflow", type="primary"):
                graph.invoke(Command(resume={"decision": decision, "reviewer": reviewer, "notes": notes,
                                             "approved_amount": amount if decision == "approve" else None}), cfg)
                st.session_state.claims[tid]["status"] = "done"
                st.rerun()


# ------------------------------------------------------------ process tab
with tab_new:
    scenarios = C.load_scenarios()
    choice = st.selectbox("Start from a test scenario", [s["title"] for s in scenarios])
    sc = copy.deepcopy(next(s for s in scenarios if s["title"] == choice))
    claim = sc["claim"]
    st.caption(f"Expected outcome: **{sc['expected']}**")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Claim details**")
        claim["claim_id"] = st.text_input("Claim ID", claim["claim_id"])
        claim["policy_id"] = st.selectbox("Policy", list(POLICIES), index=list(POLICIES).index(claim["policy_id"]))
        pol = POLICIES[claim["policy_id"]]
        st.caption(f"{pol['holder_name']} · {pol['policy_type']} · {pol['status']} · {pol['start_date']} to {pol['end_date']} · "
                   f"SI {_inr(pol['sum_insured'])}")
        claim["claimant_name"] = st.text_input("Claimant name", claim["claimant_name"])
        claim["claim_type"] = st.selectbox("Claim type", list(C.REQUIRED_DOCS), index=list(C.REQUIRED_DOCS).index(claim["claim_type"]))
        claim["incident_type"] = st.text_input("Incident type (peril)", claim["incident_type"])
        d1, d2 = st.columns(2)
        claim["incident_date"] = str(d1.date_input("Incident date", pd.to_datetime(claim["incident_date"]).date()))
        claim["report_date"] = str(d2.date_input("Report date", pd.to_datetime(claim["report_date"]).date()))
        claim["claimed_amount"] = st.number_input("Claimed amount (₹)", value=int(claim["claimed_amount"]), step=1000)
        claim["description"] = st.text_area("Incident description", claim["description"])
    with c2:
        st.markdown("**Supporting documents**")
        required = C.REQUIRED_DOCS[claim["claim_type"]]
        existing = {d["doc_type"]: d for d in claim["documents"]}
        docs = []
        for dt in required:
            has = st.checkbox(dt.replace("_", " ").title(), value=dt in existing, key=f"doc-{choice}-{dt}")
            if has:
                d = existing.get(dt, {"doc_type": dt, "filename": f"{dt}.pdf", "fields": {"name": claim["claimant_name"]}})
                if dt in C.AMOUNT_DOCS:
                    d["fields"]["amount"] = st.number_input(f"Amount on {dt} (₹)", value=int(d["fields"].get("amount", claim["claimed_amount"])),
                                                            step=1000, key=f"amt-{choice}-{dt}")
                docs.append(d)
        uploads = st.file_uploader("Attach files (optional; stored as extra evidence)", accept_multiple_files=True)
        for f in uploads or []:
            docs.append({"doc_type": "other", "filename": f.name, "fields": {}})
        claim["documents"] = docs
        with st.expander("Claim JSON sent to the agent"):
            st.json(claim)

    if st.button("🚀 Process claim", type="primary", use_container_width=True):
        st.session_state.current = run_claim(claim)

    if st.session_state.get("current"):
        tid = st.session_state.current
        show_result(tid)
        review_panel(tid)

# ------------------------------------------------------------ queue tab
with tab_queue:
    pending = [t for t, c in st.session_state.claims.items() if c["status"] == "awaiting review"]
    if not pending:
        st.info("No claims awaiting human review.")
    for tid in pending:
        review_panel(tid)
    done = [t for t, c in st.session_state.claims.items() if c["status"] == "done"]
    if done:
        st.markdown("#### Processed claims")
        st.dataframe(pd.DataFrame([{"thread": t, "claim": st.session_state.claims[t]["claim"]["claim_id"],
                                    "final": graph.get_state({"configurable": {"thread_id": t}}).values.get("final", {}).get("status")}
                                   for t in done]), hide_index=True, use_container_width=True)

# ------------------------------------------------------------ batch tab
with tab_batch:
    st.write("Runs all scenarios. Escalated claims stop at the human node; resolve them in the review queue.")
    if st.button("Run all scenarios"):
        rows = []
        for sc in C.load_scenarios():
            tid = f"batch-{sc['claim']['claim_id']}-{uuid.uuid4().hex[:4]}"
            cfg = {"configurable": {"thread_id": tid}}
            graph.invoke({"claim": sc["claim"]}, cfg)
            st_ = graph.get_state(cfg)
            v = st_.values
            paused = bool(st_.next)
            st.session_state.claims[tid] = {"claim": sc["claim"], "status": "awaiting review" if paused else "done"}
            rows.append({"Scenario": sc["title"], "Amount": _inr(sc["claim"]["claimed_amount"]), "Expected": sc["expected"],
                         "Route": v["decision"]["route"], "Pass": "✅" if v["decision"]["route"] == sc["expected"] else "❌",
                         "Fraud score": v.get("fraud", {}).get("fraud_score", "-"),
                         "Status": "⏸ awaiting human" if paused else v["final"]["status"],
                         "Top reason": v["decision"]["reasons"][0]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

# ------------------------------------------------------------ graph tab
with tab_graph:
    st.graphviz_chart(graph_dot())
    st.markdown("- **Dashed edges** are conditional. `intake` fans out to three checks that run in the **same super-step** "
                "(in parallel); `decide` waits for all three (fan-in).\n"
                "- `human_approval` calls `interrupt()`. The checkpointer saves state, and the reviewer's "
                "decision resumes it with `Command(resume=...)`.")
    with st.expander("Mermaid (generated by LangGraph)"):
        st.code(graph.get_graph().draw_mermaid(), language="text")
