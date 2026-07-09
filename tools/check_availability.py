"""
tools/check_availability.py
----------------------------
LangChain tool: return available interview slots for a shortlisted candidate.

This is a stub implementation that generates deterministic mock slots.
In production this would call a calendar API (Google Calendar, Outlook, etc.).

Public API
----------
check_availability(candidate_id, candidate_name, num_slots) -> list[dict]
    Returns a list of InterviewSlot dicts representing available time windows.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from langchain.tools import tool
from loguru import logger

from models.interview import InterviewSlot, SlotStatus

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_NUM_SLOTS: int = 3
_SLOT_DURATION_MINUTES: int = 60
_DEFAULT_INTERVIEWER: str = "TechVest Hiring Manager"
_DEFAULT_INTERVIEWER_EMAIL: str = "hiring@techvest.io"
_IST = ZoneInfo("Asia/Kolkata")

# Hardcoded mock schedule: (weekday_offset_days, hour_IST)
# Generates slots starting from "next Monday" relative to call time
_SLOT_SCHEDULE: list[tuple[int, int]] = [
    (0, 10),   # Monday  10:00 IST
    (1, 15),   # Tuesday 15:00 IST
    (2, 11),   # Wednesday 11:00 IST
    (3, 14),   # Thursday 14:00 IST
    (4, 10),   # Friday  10:00 IST
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _next_weekday(base: datetime, target_weekday: int) -> datetime:
    """Return the next occurrence of target_weekday (0=Mon … 6=Sun) after base."""
    days_ahead = (target_weekday - base.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7  # always push to the *next* week, not today
    return base + timedelta(days=days_ahead)


def _generate_mock_slots(
    candidate_id: str,
    num_slots: int,
) -> list[InterviewSlot]:
    """Generate ``num_slots`` deterministic mock interview slots."""
    now = datetime.now(tz=_IST)
    # Start from next Monday
    monday = _next_weekday(now, 0)
    slots: list[InterviewSlot] = []

    for i, (day_offset, hour) in enumerate(_SLOT_SCHEDULE):
        if len(slots) >= num_slots:
            break
        slot_start = monday.replace(
            hour=hour, minute=0, second=0, microsecond=0
        ) + timedelta(days=day_offset)
        slot_end = slot_start + timedelta(minutes=_SLOT_DURATION_MINUTES)

        # Unique but stable slot_id tied to candidate and position
        slot_id = f"slot-{candidate_id[:8]}-{i + 1:02d}"

        slots.append(
            InterviewSlot(
                slot_id=slot_id,
                start_time=slot_start.astimezone(timezone.utc),
                end_time=slot_end.astimezone(timezone.utc),
                interviewer=_DEFAULT_INTERVIEWER,
                interviewer_email=_DEFAULT_INTERVIEWER_EMAIL,
                location="Google Meet (link will be sent via email)",
                status=SlotStatus.AVAILABLE,
            )
        )

    return slots


# ---------------------------------------------------------------------------
# LangChain tool
# ---------------------------------------------------------------------------

@tool
def check_availability(
    candidate_id: str,
    candidate_name: str = "Candidate",
    num_slots: int = _DEFAULT_NUM_SLOTS,
) -> list[dict[str, Any]]:
    """Return available interview slots for a shortlisted candidate.

    This tool generates mock interview slots for the week following the
    current date.  Each slot is a 60-minute window with the TechVest
    Hiring Manager.

    In production, replace ``_generate_mock_slots`` with a call to the
    company's calendar API (Google Calendar, Microsoft Graph, Calendly, etc.).

    Parameters
    ----------
    candidate_id:
        The candidate's unique identifier (``CandidateProfile.candidate_id``).
        Used to generate stable slot IDs.
    candidate_name:
        Display name for logging.
    num_slots:
        Number of slots to return (default 3, max 5).

    Returns
    -------
    list[dict]
        List of JSON-serialisable dicts, each representing an ``InterviewSlot``.
        Returns an empty list if generation fails.

    Examples
    --------
    >>> slots = check_availability.invoke({
    ...     "candidate_id": "abc-123",
    ...     "candidate_name": "Priya Sharma",
    ... })
    >>> for s in slots:
    ...     slot = InterviewSlot(**s)
    ...     print(slot.display())
    """
    num_slots = max(1, min(num_slots, 5))  # clamp to 1–5
    logger.info(
        f"check_availability | candidate={candidate_name!r} "
        f"id={candidate_id!r} requesting {num_slots} slot(s)"
    )

    try:
        slots = _generate_mock_slots(candidate_id, num_slots)
        result = [s.model_dump(mode="json") for s in slots]

        logger.success(
            f"check_availability | returned {len(result)} slot(s) "
            f"for {candidate_name!r}"
        )
        for slot in slots:
            logger.debug(f"  slot: {slot.display()}")

        return result

    except Exception as exc:
        logger.error(
            f"check_availability | failed for {candidate_name!r}: {exc}"
        )
        return []
