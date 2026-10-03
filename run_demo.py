"""Runs every scenario in data/scenarios.json through the LangGraph agent.
Escalated claims are resumed with a scripted reviewer decision to show the
full human-in-the-loop round trip. Writes outputs/q2_results.md/json.

    python run_demo.py
"""
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from langgraph.types import Command

from claims.config import backend_name, load_scenarios
from claims.graph import build_graph
from claims.nodes import _inr

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

# Scripted reviewer decisions for escalated claims (what a claims officer might do)
REVIEWER = {
    "CLM-003": {"decision": "reject", "reviewer": "SIU.analyst", "notes": "Duplicate bills from a related-party clinic, cash payment, claim 9 days after inception. Refer to SIU."},
    "CLM-005": {"decision": "approve", "reviewer": "senior.adjuster", "notes": "Surveyor and fire-brigade reports corroborate loss.", "approved_amount": 1825000},
}


def main():
    graph = build_graph()
    rows, details, dump = [], [], []
    for i, sc in enumerate(load_scenarios()):
        claim = sc["claim"]
        cfg = {"configurable": {"thread_id": f"demo-{claim['claim_id']}"}}
        steps = []
        for upd in graph.stream({"claim": claim}, cfg, stream_mode="updates"):
            steps += list(upd.keys())
        state = graph.get_state(cfg)
        paused = bool(state.next)
        interrupt_payload = None
        if paused:
            interrupt_payload = state.tasks[0].interrupts[0].value
            for upd in graph.stream(Command(resume=REVIEWER[claim["claim_id"]]), cfg, stream_mode="updates"):
                steps += list(upd.keys())
        v = graph.get_state(cfg).values
        route = v["decision"]["route"]
        ok = "✅" if route == sc["expected"] else "❌"
        rows.append(f"| {sc['title']} | {_inr(claim['claimed_amount'])} | {sc['expected']} | {route} {ok} | "
                    f"{v.get('fraud', {}).get('fraud_score', '-')} | {v['final']['status']} | {_inr(v['final']['amount_payable'])} |")
        d = [f"### {sc['title']} ({claim['claim_id']})",
             f"- Node execution order: `{' → '.join(steps)}`",
             f"- Decision route: **{route}** (expected {sc['expected']})",
             "- Reasons: " + "; ".join(v["decision"]["reasons"])]
        if v.get("doc_check"):
            d.append(f"- Documents: missing = {v['doc_check']['missing'] or 'none'}; issues = {v['doc_check']['inconsistencies'] or 'none'}")
        if v.get("eligibility"):
            e = v["eligibility"]
            d.append(f"- Eligibility: eligible = {e['eligible']}; failures = {e['hard_failures'] or 'none'}; payable = {_inr(e['payable_amount'])}")
        if v.get("fraud"):
            f = v["fraud"]
            d.append(f"- Fraud: score {f['fraud_score']} ({f['risk_level']}); rule indicators = "
                     f"{[x['indicator'] for x in f['indicators']] or 'none'}; LLM red flags = {f['llm_red_flags'] or 'none'}")
        d.append(f"- Summary: *{v['summary']['headline']}*. {v['summary']['summary']}")
        if interrupt_payload:
            d.append(f"- ⏸ Paused for human review; reviewer decision: `{json.dumps(REVIEWER[claim['claim_id']])}`")
        d.append(f"- **Final: {v['final']['status']}**, payable {_inr(v['final']['amount_payable'])}")
        d.append("- Audit log:\n" + "\n".join(f"    - {a}" for a in v["audit_log"]))
        details.append("\n".join(d) + "\n")
        dump.append({"scenario": sc["title"], "expected": sc["expected"], "steps": steps,
                     "interrupt_payload": interrupt_payload, "state": v})

    md = [f"# Q2 claim scenarios\n\nBackend: {backend_name()}\n",
          "| Scenario | Amount | Expected | Route | Fraud score | Final status | Payable |",
          "|---|---|---|---|---|---|---|", *rows, "", "## Details\n", *details]
    (OUT / "q2_results.md").write_text("\n".join(md))
    (OUT / "q2_results.json").write_text(json.dumps(dump, indent=2, ensure_ascii=False, default=str))
    print("\n".join(md))


if __name__ == "__main__":
    main()
