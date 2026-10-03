"""LangGraph workflow.

                       ┌──────────────► document_verification ─┐
  START ─► intake ──(cond)──────────► eligibility_check ───────┼─► decide ─► claim_summary ─(cond)─► human_approval ─► finalize ─► END
              │        └──────────────► fraud_detection ───────┘                              └──────────────────────► finalize
              └─(policy not found)─────────────────────────────► decide

* Parallelism: the conditional edge after `intake` returns a LIST of three
  node names, so LangGraph runs them in the same super-step (concurrently).
* Join: `add_edge([...three...], "decide")` makes `decide` wait for ALL three.
* Human-in-the-loop: `human_approval` calls `interrupt()`; a checkpointer is
  required so the paused run can be resumed later by thread_id.
"""
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from .nodes import (claim_summary, decide, document_verification, eligibility_check, finalize, fraud_detection,
                    human_approval, intake, route_after_intake, route_after_summary)
from .state import ClaimState

CHECKS = ["document_verification", "eligibility_check", "fraud_detection"]


def build_graph(checkpointer=None):
    g = StateGraph(ClaimState)
    g.add_node("intake", intake)
    g.add_node("document_verification", document_verification)
    g.add_node("eligibility_check", eligibility_check)
    g.add_node("fraud_detection", fraud_detection)
    g.add_node("decide", decide)
    g.add_node("claim_summary", claim_summary)
    g.add_node("human_approval", human_approval)
    g.add_node("finalize", finalize)

    g.add_edge(START, "intake")
    g.add_conditional_edges("intake", route_after_intake, CHECKS + ["decide"])   # parallel fan-out
    g.add_edge(CHECKS, "decide")                                                   # fan-in (waits for all 3)
    g.add_edge("decide", "claim_summary")
    g.add_conditional_edges("claim_summary", route_after_summary,
                            {"human_approval": "human_approval", "finalize": "finalize"})
    g.add_edge("human_approval", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


def graph_dot() -> str:
    """Graphviz DOT for the UI (rendered client-side by st.graphviz_chart)."""
    return """digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef3fb", fontname="Helvetica"];
  START [shape=circle, label="", width=.25, fillcolor=black]; END [shape=doublecircle, label="", width=.2, fillcolor=black];
  START -> intake;
  intake -> document_verification [label="parallel", style=dashed];
  intake -> eligibility_check [style=dashed]; intake -> fraud_detection [style=dashed];
  intake -> decide [label="policy not found", style=dashed, color=gray];
  document_verification -> decide; eligibility_check -> decide; fraud_detection -> decide;
  decide -> claim_summary;
  claim_summary -> human_approval [label="human_review", style=dashed, color="#c0392b"];
  claim_summary -> finalize [label="auto_approve / reject", style=dashed];
  human_approval [fillcolor="#fdebd0", label="human_approval\\n(interrupt)"];
  human_approval -> finalize; finalize -> END; }"""
