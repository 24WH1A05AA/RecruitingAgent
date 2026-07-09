"""
tools/propose_interview.py
--------------------------
LangChain tool: create a confirmed InterviewProposal after human approval.

This tool is a GATED action — it must only be called after
``human_approved = True`` is set in the agent state.  Calling it without
approval raises a ``PermissionError``.

Public API
----------
propose_interview(profile_dict, slot_dict, job_title, approved_by) -> dict
    Creates and returns an InterviewProposal dict representing a confirmed,
    human-approved interview booking.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from langchain.tools import tool
from loguru import logger

from models.candidate import CandidateProfile
from models.interview import InterviewProposal, InterviewSlot

# ---------------------------------------------------------------------------
# Audit helpers
# ---------------------------------------------------------------------------

def _write_audit_log(proposal: InterviewProposal) -> None:
    """Append a confirmation line to the audit log file."""
    import os

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "interview_proposals.jsonl")

    record = {
        "event": "interview_proposed",
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "candidate_id": proposal.candidate_id,
        "candidate_name": proposal.candidate_name,
        "slot_id": proposal.slot_id,
        "start_time": proposal.start_time.isoformat(),
        "interviewer": proposal.interviewer,
        "job_title": proposal.job_title,
        "approved_by": proposal.approved_by,
    }

    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")

    logger.debug(f"propose_interview | audit log written → {log_path}")


# ---------------------------------------------------------------------------
# LangChain tool
# ---------------------------------------------------------------------------

@tool
def propose_interview(
    profile_dict: dict[str, Any],
    slot_dict: dict[str, Any],
    job_title: str,
    approved_by: str,
    human_approved: bool = False,
) -> dict[str, Any]:
    """Create a confirmed interview proposal after explicit human approval.

    This is a GATED action.  The caller must pass ``human_approved=True``
    (set by the human approval node in the LangGraph pipeline).  Attempting
    to schedule without approval raises a ``PermissionError`` and returns
    an error dict — it never creates a proposal.

    Parameters
    ----------
    profile_dict:
        Dict of a ``CandidateProfile`` (as returned by ``parse_resume``).
        Must contain ``candidate_id``, ``name``, and optionally ``email``.
    slot_dict:
        Dict of an ``InterviewSlot`` (as returned by ``check_availability``).
        Must contain ``slot_id``, ``start_time``, ``end_time``,
        ``interviewer``, and ``location``.
    job_title:
        The role being interviewed for (e.g. "Senior ML Engineer").
    approved_by:
        Full name of the human reviewer who approved the interview.
        Required for the audit trail.
    human_approved:
        Must be ``True`` for the proposal to be created.  Defaults to
        ``False`` so accidental calls without the gate set are safe.

    Returns
    -------
    dict
        JSON-serialisable dict of a validated ``InterviewProposal``.
        Includes ``is_error=True`` and an ``error`` key when:
        - ``human_approved`` is ``False``
        - model validation fails
        - any unexpected error occurs

    Examples
    --------
    >>> result = propose_interview.invoke({
    ...     "profile_dict": profile.model_dump(),
    ...     "slot_dict": slot.model_dump(mode="json"),
    ...     "job_title": "Senior ML Engineer",
    ...     "approved_by": "Meera Nair",
    ...     "human_approved": True,
    ... })
    >>> proposal = InterviewProposal(**result)
    >>> print(proposal.to_confirmation_text())
    """
    candidate_name: str = profile_dict.get("name", "Unknown")
    candidate_id: str = profile_dict.get("candidate_id", "unknown")
    logger.info(
        f"propose_interview | candidate={candidate_name!r} "
        f"approved_by={approved_by!r} human_approved={human_approved}"
    )

    # ── Approval gate ─────────────────────────────────────────────────────────
    if not human_approved:
        msg = (
            f"Interview proposal for {candidate_name!r} BLOCKED: "
            "human_approved is False.  A human reviewer must approve "
            "before an interview can be scheduled."
        )
        logger.warning(f"propose_interview | {msg}")
        return {
            "is_error": True,
            "error": msg,
            "candidate_id": candidate_id,
            "candidate_name": candidate_name,
        }

    # ── Input validation ──────────────────────────────────────────────────────
    if not approved_by or not approved_by.strip():
        return {
            "is_error": True,
            "error": "approved_by must be a non-empty name.",
            "candidate_id": candidate_id,
            "candidate_name": candidate_name,
        }

    try:
        profile = CandidateProfile(**profile_dict)
        slot = InterviewSlot(**slot_dict)

        proposal = InterviewProposal(
            candidate_id=profile.candidate_id,
            candidate_name=profile.name,
            candidate_email=profile.email,
            slot_id=slot.slot_id,
            start_time=slot.start_time,
            end_time=slot.end_time,
            interviewer=slot.interviewer,
            location=slot.location,
            job_title=job_title,
            approved_by=approved_by.strip(),
            approved_at=datetime.now(tz=timezone.utc),
        )

        _write_audit_log(proposal)

        logger.success(
            f"propose_interview | CONFIRMED | {profile.name!r} "
            f"slot={slot.slot_id!r} start={slot.start_time.isoformat()}"
        )
        logger.info("\n" + proposal.to_confirmation_text())

        result = proposal.model_dump(mode="json")
        result["is_error"] = False
        result["confirmation_text"] = proposal.to_confirmation_text()
        return result

    except Exception as exc:
        logger.error(f"propose_interview | failed for {candidate_name!r}: {exc}")
        return {
            "is_error": True,
            "error": str(exc),
            "candidate_id": candidate_id,
            "candidate_name": candidate_name,
        }
