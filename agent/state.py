"""
agent/state.py
--------------
Defines AgentState — the single shared TypedDict passed between every
LangGraph node in the TechVest Recruitment Agent pipeline.

Design notes
------------
- LangGraph merges state by key; each node returns only the keys it modifies.
- Lists use Annotated[list, operator.add] so appending is safe across nodes.
- Primitive fields (bool, int, str, None) are overwritten directly.
- All fields have defaults so the graph can be cold-started with minimal input.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage


# ---------------------------------------------------------------------------
# AgentState
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    """Shared mutable state for the TechVest Recruitment Agent.

    Passed to every node; each node returns a partial dict containing only
    the keys it has updated.  LangGraph merges the partial dict into the
    running state.

    Field groups
    ------------
    Input
        job_description_text  Raw JD text supplied by the user.
        rubric_dict           Serialised Rubric; defaults to Rubric.default().

    Candidate pipeline
        candidates            Raw resume texts + file paths to process.
        candidate_profiles    Parsed CandidateProfile dicts (one per resume).
        scores                ScoreCard dicts; one per parsed candidate.
        shortlist             Subset of scores that passed the decision threshold.
        current_index         Pointer into `candidates`; drives the parse/score loop.

    Interview scheduling
        interview_slots       Available slot dicts returned by check_availability.
        selected_slot         The one slot chosen for scheduling.
        scheduled_interviews  Confirmed InterviewProposal dicts.

    Control flow
        human_approved        Set True by the human approval node; gates scheduling.
        approval_candidate_id Candidate ID awaiting human review.
        iteration_count       Monotonically incremented; checked against recursion_limit.
        error_message         Non-empty when a node fails gracefully.

    Audit
        audit_log             Append-only list of human-readable log lines.
        messages              LangChain message history (for LLM nodes).
    """

    # ── Input ─────────────────────────────────────────────────────────────────
    job_description_text: str
    rubric_dict: dict[str, Any]

    # ── Candidate pipeline ────────────────────────────────────────────────────
    # Each entry: {"raw_text": str, "file_path": str}
    candidates: list[dict[str, Any]]

    # Appended to as each candidate is processed
    candidate_profiles: Annotated[list[dict[str, Any]], operator.add]
    scores: Annotated[list[dict[str, Any]], operator.add]

    # Set after ranking
    shortlist: list[dict[str, Any]]

    # Loop pointer — which candidate in `candidates` is being processed
    current_index: int

    # ── Interview scheduling ──────────────────────────────────────────────────
    interview_slots: list[dict[str, Any]]
    selected_slot: dict[str, Any] | None
    scheduled_interviews: Annotated[list[dict[str, Any]], operator.add]

    # ── Control flow ──────────────────────────────────────────────────────────
    human_approved: bool
    approval_candidate_id: str | None
    iteration_count: int
    error_message: str

    # ── Audit ─────────────────────────────────────────────────────────────────
    audit_log: Annotated[list[str], operator.add]
    messages: Annotated[list[BaseMessage], operator.add]


# ---------------------------------------------------------------------------
# Factory: empty initial state
# ---------------------------------------------------------------------------

def initial_state(
    job_description_text: str,
    candidates: list[dict[str, Any]],
    rubric_dict: dict[str, Any] | None = None,
) -> AgentState:
    """Return a fully-initialised AgentState ready for graph.invoke().

    Parameters
    ----------
    job_description_text:
        Raw job description text (e.g. loaded from a .txt file).
    candidates:
        List of dicts, each with keys ``raw_text`` (str) and
        ``file_path`` (str), one per resume to evaluate.
    rubric_dict:
        Optional serialised ``Rubric`` dict.  Defaults to
        ``Rubric.default().model_dump()`` if not provided.

    Returns
    -------
    AgentState
        Populated TypedDict with sensible defaults for all fields.
    """
    from models.rubric import Rubric  # local import avoids circular deps

    return AgentState(
        # Input
        job_description_text=job_description_text,
        rubric_dict=rubric_dict or Rubric.default().model_dump(),
        # Candidate pipeline
        candidates=candidates,
        candidate_profiles=[],
        scores=[],
        shortlist=[],
        current_index=0,
        # Interview scheduling
        interview_slots=[],
        selected_slot=None,
        scheduled_interviews=[],
        # Control flow
        human_approved=False,
        approval_candidate_id=None,
        iteration_count=0,
        error_message="",
        # Audit
        audit_log=[],
        messages=[],
    )
