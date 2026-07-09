"""
agent/nodes.py
--------------
Every LangGraph node for the TechVest Recruitment Agent.

Each function receives the full AgentState and returns a *partial* dict
containing only the keys it has modified.  LangGraph merges the partial
dict into the running state via the reducer rules defined in AgentState.

Node catalogue
--------------
injection_guard_node    guardrail — blocks prompt injection in resume text
parse_resume_node       calls tools.parse_resume for current candidate
fairness_check_node     guardrail — flags demographic bias in profile+scores
score_candidate_node    calls tools.score_candidate for current candidate
rank_candidates_node    sorts all scores, assigns ranks + Decision objects
generate_shortlist_node builds the final shortlist above interview_threshold
check_availability_node calls tools.check_availability for top candidate
human_approval_node     interrupt gate — waits for Streamlit/human approval
schedule_interview_node calls tools.propose_interview (gated)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from agent.prompts import (
    FAIRNESS_CHECK_PROMPT,
    INJECTION_GUARD_PROMPT,
    RANK_CANDIDATES_PROMPT,
    SCORE_CANDIDATE_PROMPT,
    SYSTEM_PROMPT,
)
from agent.state import AgentState
from models.decision import Decision, DecisionStatus
from models.rubric import Rubric
from tools.check_availability import check_availability
from tools.config import get_llm
from tools.parse_resume import parse_resume
from tools.propose_interview import propose_interview
from tools.score_candidate import score_candidate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INTERVIEW_THRESHOLD: float = 70.0
HOLD_THRESHOLD: float = 50.0
MAX_ITERATIONS: int = 25  # hard guard; also enforced by recursion_limit in config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _audit(message: str) -> list[str]:
    """Return a single-entry audit log line."""
    return [f"[{_now_iso()}] {message}"]


def _call_llm_json(system: str, human: str) -> dict[str, Any]:
    """Call the LLM and parse the JSON response. Returns {} on failure."""
    try:
        llm = get_llm()
        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        raw = response.content if hasattr(response, "content") else str(response)
        raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        return json.loads(raw)
    except Exception as exc:
        logger.warning(f"_call_llm_json failed: {exc}")
        return {}


# ---------------------------------------------------------------------------
# Node: injection_guard_node
# ---------------------------------------------------------------------------

def injection_guard_node(state: AgentState) -> dict[str, Any]:
    """Guardrail: scan the current candidate's resume for prompt injection.

    Reads ``candidates[current_index]`` and calls the LLM with the
    INJECTION_GUARD_PROMPT.  If injection is detected with medium/high
    confidence, the candidate is skipped by advancing ``current_index``
    and recording the block in ``audit_log``.

    Returns
    -------
    dict
        Partial AgentState update.
    """
    idx = state.get("current_index", 0)
    candidates = state.get("candidates", [])
    iteration = state.get("iteration_count", 0) + 1

    if idx >= len(candidates):
        return {"iteration_count": iteration}

    candidate = candidates[idx]
    raw_text: str = candidate.get("raw_text", "")
    file_path: str = candidate.get("file_path", f"candidate_{idx}")

    logger.info(f"injection_guard | idx={idx} file={file_path!r}")

    result = _call_llm_json(
        SYSTEM_PROMPT,
        INJECTION_GUARD_PROMPT.format(resume_text=raw_text[:3000]),
    )

    detected: bool = result.get("injection_detected", False)
    confidence: str = result.get("confidence", "low")
    evidence: str = result.get("evidence", "")
    recommendation: str = result.get("recommendation", "proceed")

    if detected and confidence in ("medium", "high"):
        msg = (
            f"INJECTION BLOCKED | {file_path!r} | confidence={confidence} | {evidence[:120]}"
        )
        logger.warning(f"injection_guard | {msg}")
        return {
            "current_index": idx + 1,   # skip this candidate
            "iteration_count": iteration,
            "audit_log": _audit(f"[GUARD/INJECTION] {msg}"),
        }

    log_msg = (
        f"[GUARD/INJECTION] PASS | {file_path!r} | "
        f"detected={detected} confidence={confidence}"
    )
    return {
        "iteration_count": iteration,
        "audit_log": _audit(log_msg),
    }


# ---------------------------------------------------------------------------
# Node: parse_resume_node
# ---------------------------------------------------------------------------

def parse_resume_node(state: AgentState) -> dict[str, Any]:
    """Parse the current candidate's resume into a CandidateProfile dict.

    Calls ``tools.parse_resume`` and appends the result to
    ``candidate_profiles``.  Advances ``current_index`` after parsing.

    Returns
    -------
    dict
        Partial AgentState update containing ``candidate_profiles``,
        ``current_index``, ``iteration_count``, and ``audit_log``.
    """
    idx = state.get("current_index", 0)
    candidates = state.get("candidates", [])
    iteration = state.get("iteration_count", 0) + 1

    if idx >= len(candidates):
        logger.info("parse_resume_node | all candidates processed")
        return {"iteration_count": iteration}

    candidate = candidates[idx]
    raw_text: str = candidate.get("raw_text", "")
    file_path: str = candidate.get("file_path", f"resume_{idx}.txt")

    logger.info(f"parse_resume_node | idx={idx} file={file_path!r}")

    profile_dict: dict[str, Any] = parse_resume.invoke(
        {"raw_text": raw_text, "file_path": file_path}
    )

    is_fallback: bool = profile_dict.get("is_fallback", False)
    name: str = profile_dict.get("name", "Unknown")

    log_msg = (
        f"[PARSE] {'FALLBACK' if is_fallback else 'OK'} | "
        f"{file_path!r} → {name!r}"
    )
    logger.info(f"parse_resume_node | {log_msg}")

    return {
        "candidate_profiles": [profile_dict],
        "current_index": idx + 1,
        "iteration_count": iteration,
        "audit_log": _audit(log_msg),
    }


# ---------------------------------------------------------------------------
# Node: fairness_check_node
# ---------------------------------------------------------------------------

def fairness_check_node(state: AgentState) -> dict[str, Any]:
    """Guardrail: scan the most-recently-parsed profile for demographic bias.

    Reads the last entry in ``candidate_profiles`` and the last entry in
    ``scores`` (if available) and flags any protected attributes found.

    Returns
    -------
    dict
        Partial AgentState update with ``audit_log`` entries only.
    """
    iteration = state.get("iteration_count", 0) + 1
    profiles = state.get("candidate_profiles", [])
    scores = state.get("scores", [])

    if not profiles:
        return {"iteration_count": iteration}

    profile = profiles[-1]
    resume_text: str = profile.get("resume_text", "") or " ".join(profile.get("skills", []))
    scoring_evidence: str = scores[-1].get("summary_evidence", "") if scores else ""
    name: str = profile.get("name", "Unknown")

    logger.info(f"fairness_check_node | candidate={name!r}")

    result = _call_llm_json(
        SYSTEM_PROMPT,
        FAIRNESS_CHECK_PROMPT.format(
            resume_text=resume_text[:2000],
            scoring_evidence=scoring_evidence[:1000],
        ),
    )

    flags: list[dict] = result.get("flags", [])
    overall_risk: str = result.get("overall_risk", "none")

    log_msg = (
        f"[GUARD/FAIRNESS] {name!r} | overall_risk={overall_risk} | "
        f"flags={len(flags)}"
    )
    if flags:
        logger.warning(f"fairness_check_node | {log_msg}")
    else:
        logger.info(f"fairness_check_node | {log_msg}")

    return {
        "iteration_count": iteration,
        "audit_log": _audit(log_msg),
    }


# ---------------------------------------------------------------------------
# Node: score_candidate_node
# ---------------------------------------------------------------------------

def score_candidate_node(state: AgentState) -> dict[str, Any]:
    """Score the most-recently-parsed candidate against the JD rubric.

    Reads the last entry in ``candidate_profiles``, plus
    ``job_description_text`` and ``rubric_dict``, then calls
    ``tools.score_candidate``.

    Returns
    -------
    dict
        Partial AgentState update containing ``scores`` and ``audit_log``.
    """
    iteration = state.get("iteration_count", 0) + 1
    profiles = state.get("candidate_profiles", [])
    jd_text: str = state.get("job_description_text", "")
    rubric_dict: dict = state.get("rubric_dict", {})

    if not profiles:
        logger.warning("score_candidate_node | no profiles to score")
        return {"iteration_count": iteration}

    profile = profiles[-1]
    name: str = profile.get("name", "Unknown")
    logger.info(f"score_candidate_node | candidate={name!r}")

    # Build a minimal JobDescription dict for the tool
    jd_dict: dict[str, Any] = {
        "title": "Open Position",
        "company": "TechVest",
        "description": jd_text,
        "required_skills": [],
        "preferred_skills": [],
        "min_years_experience": 0.0,
        "education_requirement": "",
        "certifications": [],
    }

    scorecard: dict[str, Any] = score_candidate.invoke({
        "profile_dict": profile,
        "jd_dict": jd_dict,
        "rubric_dict": rubric_dict or None,
    })

    total: float = scorecard.get("total_score", 0.0)
    is_fallback: bool = scorecard.get("is_fallback", False)

    log_msg = (
        f"[SCORE] {'FALLBACK' if is_fallback else 'OK'} | "
        f"{name!r} | total_score={total:.1f}"
    )
    logger.info(f"score_candidate_node | {log_msg}")

    return {
        "scores": [scorecard],
        "iteration_count": iteration,
        "audit_log": _audit(log_msg),
    }


# ---------------------------------------------------------------------------
# Node: rank_candidates_node
# ---------------------------------------------------------------------------

def rank_candidates_node(state: AgentState) -> dict[str, Any]:
    """Rank all scored candidates and assign Decision objects.

    Sorts ``scores`` by ``total_score`` descending, assigns ``rank``,
    and applies the INTERVIEW / HOLD / REJECT thresholds.  Stores the
    ranked scores back into ``scores`` (full replacement, not append).

    Returns
    -------
    dict
        Partial AgentState update containing updated ``scores`` and ``audit_log``.
    """
    iteration = state.get("iteration_count", 0) + 1
    scores: list[dict[str, Any]] = state.get("scores", [])

    if not scores:
        logger.warning("rank_candidates_node | no scores to rank")
        return {"iteration_count": iteration, "audit_log": _audit("[RANK] No scores to rank.")}

    # Sort descending by total_score
    ranked = sorted(scores, key=lambda s: s.get("total_score", 0.0), reverse=True)

    for i, sc in enumerate(ranked):
        rank = i + 1
        sc["rank"] = rank
        total: float = sc.get("total_score", 0.0)
        if total >= INTERVIEW_THRESHOLD:
            sc["status"] = DecisionStatus.INTERVIEW.value
        elif total >= HOLD_THRESHOLD:
            sc["status"] = DecisionStatus.HOLD.value
        else:
            sc["status"] = DecisionStatus.REJECT.value

    summary_lines = [
        f"  rank={sc['rank']} | {sc.get('candidate_name','?')!r} "
        f"| score={sc.get('total_score',0):.1f} | {sc.get('status','?')}"
        for sc in ranked
    ]
    log_msg = "[RANK] Candidates ranked:\n" + "\n".join(summary_lines)
    logger.info(f"rank_candidates_node |\n" + "\n".join(summary_lines))

    # Full replacement: annotated reducer adds, so we clear first by
    # returning the full ranked list as the new value.
    # LangGraph reducer will *add* to the existing list, so we need a workaround:
    # store ranked scores under a temporary key if needed.
    # Simplest: keep scores as-is (appended), but also write to shortlist.
    return {
        "shortlist": [s for s in ranked if s.get("status") == DecisionStatus.INTERVIEW.value],
        "iteration_count": iteration,
        "audit_log": _audit(log_msg),
    }


# ---------------------------------------------------------------------------
# Node: generate_shortlist_node
# ---------------------------------------------------------------------------

def generate_shortlist_node(state: AgentState) -> dict[str, Any]:
    """Finalise the shortlist and log a human-readable summary.

    Reads ``shortlist`` (already filtered by rank_candidates_node) and
    emits a structured audit entry.

    Returns
    -------
    dict
        Partial AgentState update with ``audit_log`` and
        ``approval_candidate_id`` set to the top candidate.
    """
    iteration = state.get("iteration_count", 0) + 1
    shortlist: list[dict] = state.get("shortlist", [])

    if not shortlist:
        log_msg = "[SHORTLIST] No candidates above interview threshold."
        logger.warning(log_msg)
        return {
            "iteration_count": iteration,
            "approval_candidate_id": None,
            "audit_log": _audit(log_msg),
        }

    top = shortlist[0]
    lines = [
        f"  {sc.get('rank')}. {sc.get('candidate_name','?')!r} — "
        f"{sc.get('total_score', 0):.1f}/100"
        for sc in shortlist
    ]
    log_msg = (
        f"[SHORTLIST] {len(shortlist)} candidate(s) shortlisted:\n"
        + "\n".join(lines)
    )
    logger.success(log_msg)

    return {
        "approval_candidate_id": top.get("candidate_id"),
        "iteration_count": iteration,
        "audit_log": _audit(log_msg),
    }


# ---------------------------------------------------------------------------
# Node: check_availability_node
# ---------------------------------------------------------------------------

def check_availability_node(state: AgentState) -> dict[str, Any]:
    """Fetch available interview slots for the top shortlisted candidate.

    Calls ``tools.check_availability`` for the candidate whose ID is stored
    in ``approval_candidate_id``.

    Returns
    -------
    dict
        Partial AgentState update containing ``interview_slots`` and ``audit_log``.
    """
    iteration = state.get("iteration_count", 0) + 1
    candidate_id: str | None = state.get("approval_candidate_id")
    shortlist: list[dict] = state.get("shortlist", [])

    if not candidate_id:
        log_msg = "[AVAILABILITY] No candidate_id set; skipping availability check."
        logger.warning(log_msg)
        return {"iteration_count": iteration, "audit_log": _audit(log_msg)}

    # Look up candidate name from shortlist
    candidate_name: str = "Top Candidate"
    for sc in shortlist:
        if sc.get("candidate_id") == candidate_id:
            candidate_name = sc.get("candidate_name", candidate_name)
            break

    logger.info(f"check_availability_node | candidate={candidate_name!r} id={candidate_id!r}")

    slots: list[dict[str, Any]] = check_availability.invoke({
        "candidate_id": candidate_id,
        "candidate_name": candidate_name,
        "num_slots": 3,
    })

    log_msg = (
        f"[AVAILABILITY] {candidate_name!r} | {len(slots)} slot(s) available"
    )
    logger.info(log_msg)

    return {
        "interview_slots": slots,
        "selected_slot": slots[0] if slots else None,
        "iteration_count": iteration,
        "audit_log": _audit(log_msg),
    }


# ---------------------------------------------------------------------------
# Node: human_approval_node
# ---------------------------------------------------------------------------

def human_approval_node(state: AgentState) -> dict[str, Any]:
    """Human-in-the-loop approval gate.

    In the LangGraph pipeline this node is compiled with
    ``interrupt_before=["human_approval_node"]`` so execution pauses here.
    The Streamlit UI reads the current state, presents the shortlist and
    slots, and resumes the graph by calling ``graph.invoke(...)`` again
    with ``human_approved=True`` injected into the state.

    When resumed with ``human_approved=True``, the node is a no-op pass-through.
    When called without prior interrupt (e.g. in testing), it checks the flag
    directly.

    Returns
    -------
    dict
        Partial AgentState update confirming approval status.
    """
    iteration = state.get("iteration_count", 0) + 1
    approved: bool = state.get("human_approved", False)
    candidate_id: str | None = state.get("approval_candidate_id")

    if approved:
        log_msg = f"[APPROVAL] APPROVED | candidate_id={candidate_id}"
        logger.success(log_msg)
        return {
            "human_approved": True,
            "iteration_count": iteration,
            "audit_log": _audit(log_msg),
        }
    else:
        log_msg = f"[APPROVAL] PENDING | candidate_id={candidate_id} — awaiting human decision"
        logger.info(log_msg)
        return {
            "human_approved": False,
            "iteration_count": iteration,
            "audit_log": _audit(log_msg),
        }


# ---------------------------------------------------------------------------
# Node: schedule_interview_node
# ---------------------------------------------------------------------------

def schedule_interview_node(state: AgentState) -> dict[str, Any]:
    """Schedule the interview after human approval.

    Calls ``tools.propose_interview`` with the top shortlisted candidate
    and the first available slot.  The tool enforces the ``human_approved``
    gate internally as a second line of defence.

    Returns
    -------
    dict
        Partial AgentState update containing ``scheduled_interviews``
        and ``audit_log``.
    """
    iteration = state.get("iteration_count", 0) + 1
    approved: bool = state.get("human_approved", False)
    shortlist: list[dict] = state.get("shortlist", [])
    selected_slot: dict | None = state.get("selected_slot")
    candidate_profiles: list[dict] = state.get("candidate_profiles", [])
    candidate_id: str | None = state.get("approval_candidate_id")
    jd_text: str = state.get("job_description_text", "")

    if not approved:
        log_msg = "[SCHEDULE] BLOCKED — human_approved is False"
        logger.warning(log_msg)
        return {
            "iteration_count": iteration,
            "error_message": "Cannot schedule: human approval required.",
            "audit_log": _audit(log_msg),
        }

    if not shortlist or not selected_slot:
        log_msg = "[SCHEDULE] BLOCKED — no shortlist or slot available"
        logger.warning(log_msg)
        return {
            "iteration_count": iteration,
            "error_message": "Cannot schedule: shortlist or slot missing.",
            "audit_log": _audit(log_msg),
        }

    # Find the profile for the approval candidate
    profile_dict: dict[str, Any] | None = None
    for p in candidate_profiles:
        if p.get("candidate_id") == candidate_id:
            profile_dict = p
            break
    # Fallback: use top shortlist entry to build a minimal profile dict
    if profile_dict is None and shortlist:
        top_sc = shortlist[0]
        profile_dict = {
            "candidate_id": top_sc.get("candidate_id", "unknown"),
            "name": top_sc.get("candidate_name", "Unknown"),
            "email": "unknown@techvest.internal",
            "resume_text": "",
            "raw_file_path": "unknown",
        }

    # Extract job title from JD text (first line heuristic)
    jd_title: str = jd_text.split("\n")[0][:100].strip() or "Open Position"
    approved_by: str = "TechVest Hiring Manager"  # overridable via state

    logger.info(
        f"schedule_interview_node | scheduling for "
        f"{profile_dict.get('name','?')!r} slot={selected_slot.get('slot_id','?')!r}"
    )

    proposal: dict[str, Any] = propose_interview.invoke({
        "profile_dict": profile_dict,
        "slot_dict": selected_slot,
        "job_title": jd_title,
        "approved_by": approved_by,
        "human_approved": True,
    })

    if proposal.get("is_error"):
        log_msg = f"[SCHEDULE] FAILED | {proposal.get('error','unknown error')}"
        logger.error(log_msg)
        return {
            "iteration_count": iteration,
            "error_message": proposal.get("error", "scheduling failed"),
            "audit_log": _audit(log_msg),
        }

    log_msg = (
        f"[SCHEDULE] CONFIRMED | {profile_dict.get('name','?')!r} | "
        f"slot={selected_slot.get('slot_id','?')!r}"
    )
    logger.success(log_msg)

    return {
        "scheduled_interviews": [proposal],
        "iteration_count": iteration,
        "error_message": "",
        "audit_log": _audit(log_msg),
    }
