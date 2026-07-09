"""
agent/graph.py
--------------
Assembles and compiles the TechVest Recruitment Agent LangGraph StateGraph.

Graph topology
--------------

    START
      │
      ▼
    injection_guard ──── (injected) ──► parse_resume  [skipped; index advances]
      │                                      │
      │ (clean)                              ▼
      └──────────────────────────► parse_resume_node
                                            │
                                            ▼
                                    fairness_check_node
                                            │
                                            ▼
                                    score_candidate_node
                                            │
                                   (more candidates?)
                              ┌─────────── ▼ ───────────┐
                              │ YES: loop back            │ NO
                              │ → injection_guard         │
                              └───────────────────────────┘
                                            │
                                            ▼
                                    rank_candidates_node
                                            │
                                            ▼
                                    generate_shortlist_node
                                            │
                              (shortlist empty?)
                         ┌──────────── ▼ ──────────────┐
                         │ YES → END                    │ NO
                         └──────────────────────────────┘
                                            │
                                            ▼
                                    check_availability_node
                                            │
                                            ▼
                                    human_approval_node   ← interrupt_before here
                                            │
                              (human_approved?)
                         ┌──────────── ▼ ──────────────┐
                         │ NO → END (rejected)          │ YES
                         └──────────────────────────────┘
                                            │
                                            ▼
                                    schedule_interview_node
                                            │
                                            ▼
                                          END

Exported symbols
----------------
build_graph()           build and return a compiled CompiledStateGraph
recruitment_graph       module-level compiled graph (default config)
GRAPH_CONFIG            default config dict to pass to graph.invoke()
RECURSION_LIMIT         default recursion limit (25 steps)

Usage
-----
    from agent.graph import recruitment_graph, GRAPH_CONFIG

    state = initial_state(jd_text, candidates)
    result = recruitment_graph.invoke(state, config=GRAPH_CONFIG)
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from loguru import logger

from agent.nodes import (
    check_availability_node,
    fairness_check_node,
    generate_shortlist_node,
    human_approval_node,
    injection_guard_node,
    parse_resume_node,
    rank_candidates_node,
    schedule_interview_node,
    score_candidate_node,
)
from agent.state import AgentState

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RECURSION_LIMIT: int = 25

# Default config dict — pass this to every graph.invoke() / graph.stream() call
GRAPH_CONFIG: dict[str, Any] = {
    "recursion_limit": RECURSION_LIMIT,
    "configurable": {
        "thread_id": "default",  # override per session
    },
}


# ---------------------------------------------------------------------------
# Conditional edge routing functions
# ---------------------------------------------------------------------------

def _route_after_injection_guard(
    state: AgentState,
) -> Literal["parse_resume_node", "injection_guard_node", "rank_candidates_node"]:
    """After the injection guard, decide what to do next.

    - If current_index < len(candidates): parse the next candidate.
    - If current_index >= len(candidates) AND we have scores: rank them.
    - If no candidates at all: go straight to END via rank (handles gracefully).
    """
    idx: int = state.get("current_index", 0)
    candidates: list = state.get("candidates", [])

    if idx < len(candidates):
        return "parse_resume_node"
    return "rank_candidates_node"


def _route_after_score(
    state: AgentState,
) -> Literal["injection_guard_node", "rank_candidates_node"]:
    """After scoring, loop back to process the next candidate or proceed to ranking."""
    idx: int = state.get("current_index", 0)
    candidates: list = state.get("candidates", [])

    if idx < len(candidates):
        return "injection_guard_node"   # process next candidate
    return "rank_candidates_node"       # all done → rank


def _route_after_shortlist(
    state: AgentState,
) -> Literal["check_availability_node", "__end__"]:
    """After shortlisting, proceed to availability only if there are shortlisted candidates."""
    shortlist: list = state.get("shortlist", [])
    if shortlist:
        return "check_availability_node"
    logger.info("route_after_shortlist | empty shortlist → END")
    return END


def _route_after_approval(
    state: AgentState,
) -> Literal["schedule_interview_node", "__end__"]:
    """After human approval node, proceed to schedule or end."""
    approved: bool = state.get("human_approved", False)
    if approved:
        return "schedule_interview_node"
    logger.info("route_after_approval | not approved → END")
    return END


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(
    checkpointer: MemorySaver | None = None,
    *,
    recursion_limit: int = RECURSION_LIMIT,
) -> Any:
    """Build and compile the TechVest Recruitment Agent StateGraph.

    Parameters
    ----------
    checkpointer:
        A LangGraph checkpoint backend.  Defaults to a fresh ``MemorySaver``
        (in-memory; suitable for development and testing).
        In production, swap for a ``SqliteSaver`` or ``PostgresSaver``.
    recursion_limit:
        Maximum number of node executions before the graph raises
        ``GraphRecursionError``.  Passed via the run config at invoke time.
        Default: 25.

    Returns
    -------
    CompiledStateGraph
        Ready to call via ``.invoke()``, ``.stream()``, or ``.astream()``.

    Notes
    -----
    ``interrupt_before=["human_approval_node"]`` is set so the graph pauses
    *before* entering the approval node.  The Streamlit UI reads the state,
    shows the shortlist, and resumes execution by calling ``.invoke()`` again
    with ``human_approved=True`` merged into the state.
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    graph = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("injection_guard_node",       injection_guard_node)
    graph.add_node("parse_resume_node",          parse_resume_node)
    graph.add_node("fairness_check_node",        fairness_check_node)
    graph.add_node("score_candidate_node",       score_candidate_node)
    graph.add_node("rank_candidates_node",       rank_candidates_node)
    graph.add_node("generate_shortlist_node",    generate_shortlist_node)
    graph.add_node("check_availability_node",    check_availability_node)
    graph.add_node("human_approval_node",        human_approval_node)
    graph.add_node("schedule_interview_node",    schedule_interview_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    graph.add_edge(START, "injection_guard_node")

    # ── Candidate processing loop ─────────────────────────────────────────────
    # injection_guard → parse OR rank (when all candidates done)
    graph.add_conditional_edges(
        "injection_guard_node",
        _route_after_injection_guard,
        {
            "parse_resume_node":   "parse_resume_node",
            "injection_guard_node": "injection_guard_node",
            "rank_candidates_node": "rank_candidates_node",
        },
    )

    # parse → fairness → score
    graph.add_edge("parse_resume_node",    "fairness_check_node")
    graph.add_edge("fairness_check_node",  "score_candidate_node")

    # score → loop back to injection_guard OR proceed to ranking
    graph.add_conditional_edges(
        "score_candidate_node",
        _route_after_score,
        {
            "injection_guard_node": "injection_guard_node",
            "rank_candidates_node": "rank_candidates_node",
        },
    )

    # ── Post-loop pipeline ────────────────────────────────────────────────────
    graph.add_edge("rank_candidates_node",    "generate_shortlist_node")

    # shortlist → availability OR end (empty shortlist)
    graph.add_conditional_edges(
        "generate_shortlist_node",
        _route_after_shortlist,
        {
            "check_availability_node": "check_availability_node",
            END: END,
        },
    )

    graph.add_edge("check_availability_node", "human_approval_node")

    # human approval → schedule OR end (rejected / no approval)
    graph.add_conditional_edges(
        "human_approval_node",
        _route_after_approval,
        {
            "schedule_interview_node": "schedule_interview_node",
            END: END,
        },
    )

    graph.add_edge("schedule_interview_node", END)

    # ── Compile ───────────────────────────────────────────────────────────────
    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_approval_node"],  # pause for Streamlit UI
    )

    logger.info(
        f"build_graph | compiled TechVest recruitment graph "
        f"(recursion_limit={recursion_limit}, interrupt_before=human_approval_node)"
    )
    return compiled


# ---------------------------------------------------------------------------
# Module-level default graph instance
# ---------------------------------------------------------------------------

# Shared in-memory checkpointer — one instance so threads share memory
_memory = MemorySaver()

recruitment_graph = build_graph(checkpointer=_memory)


# ---------------------------------------------------------------------------
# Helper: make a run config with a unique thread_id
# ---------------------------------------------------------------------------

def make_config(
    thread_id: str | None = None,
    recursion_limit: int = RECURSION_LIMIT,
) -> dict[str, Any]:
    """Return a LangGraph run config dict.

    Parameters
    ----------
    thread_id:
        Unique identifier for this conversation/run.  Used by MemorySaver
        to isolate state between runs.  Generates a UUID4 if not provided.
    recursion_limit:
        Max node executions before ``GraphRecursionError``.

    Returns
    -------
    dict
        Config ready to pass as the ``config`` argument to
        ``recruitment_graph.invoke()`` or ``recruitment_graph.stream()``.

    Examples
    --------
    >>> cfg = make_config(thread_id="session-abc")
    >>> result = recruitment_graph.invoke(state, config=cfg)
    """
    return {
        "recursion_limit": recursion_limit,
        "configurable": {
            "thread_id": thread_id or str(uuid.uuid4()),
        },
    }
