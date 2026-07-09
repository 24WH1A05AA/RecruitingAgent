"""
models/interview.py
-------------------
Pydantic models for interview scheduling.

Models
------
InterviewSlot      - an available time slot returned by check_availability()
InterviewProposal  - a confirmed, human-approved interview booking
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


class SlotStatus(str, Enum):
    """Lifecycle state of an interview slot."""

    AVAILABLE = "available"
    PROPOSED = "proposed"     # sent to candidate, awaiting confirmation
    CONFIRMED = "confirmed"   # human-approved and locked in
    CANCELLED = "cancelled"


class InterviewSlot(BaseModel):
    """An available interview time slot returned by check_availability().

    Immutable after construction.
    """

    model_config = {"frozen": True, "str_strip_whitespace": True}

    slot_id: str = Field(..., min_length=1, description="Unique identifier for this slot.")
    start_time: datetime = Field(..., description="Slot start time (timezone-aware).")
    end_time: datetime = Field(..., description="Slot end time (timezone-aware).")
    interviewer: str = Field(..., min_length=1, max_length=200, description="Interviewer full name.")
    interviewer_email: str = Field(default="", description="Interviewer contact e-mail.")
    location: str = Field(
        default="Video Call",
        description="Interview location or video link (e.g. 'Zoom', 'Google Meet').",
    )
    status: SlotStatus = Field(default=SlotStatus.AVAILABLE, description="Current slot status.")

    @model_validator(mode="after")
    def _end_after_start(self) -> "InterviewSlot":
        if self.end_time <= self.start_time:
            raise ValueError(
                f"end_time ({self.end_time}) must be after start_time ({self.start_time})."
            )
        return self

    @field_validator("start_time", "end_time", mode="before")
    @classmethod
    def _ensure_timezone_aware(cls, v: datetime) -> datetime:
        """Attach UTC if datetime is naive."""
        if isinstance(v, datetime) and v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    def duration_minutes(self) -> float:
        """Return slot duration in minutes."""
        return (self.end_time - self.start_time).total_seconds() / 60.0

    def is_available(self) -> bool:
        return self.status == SlotStatus.AVAILABLE

    def display(self) -> str:
        """Human-readable slot label for Streamlit UI."""
        return (
            f"{self.start_time.strftime('%A %d %b %Y %H:%M')} – "
            f"{self.end_time.strftime('%H:%M %Z')} with {self.interviewer}"
        )

    def __str__(self) -> str:
        return f"InterviewSlot(id={self.slot_id!r}, start={self.start_time.isoformat()}, status={self.status.value})"


class InterviewProposal(BaseModel):
    """A confirmed, human-approved interview booking.

    Created after human approval; immutable once recorded.
    """

    model_config = {"frozen": True, "str_strip_whitespace": True}

    # Candidate
    candidate_id: str = Field(..., min_length=1, description="Matches CandidateProfile.candidate_id.")
    candidate_name: str = Field(..., min_length=1, max_length=200, description="Candidate display name.")
    candidate_email: str = Field(default="", description="Candidate contact e-mail for calendar invite.")

    # Slot details (denormalised for portability)
    slot_id: str = Field(..., min_length=1, description="Matches InterviewSlot.slot_id.")
    start_time: datetime = Field(..., description="Interview start time (timezone-aware).")
    end_time: datetime = Field(..., description="Interview end time (timezone-aware).")
    interviewer: str = Field(..., min_length=1, max_length=200, description="Interviewer full name.")
    location: str = Field(default="Video Call", description="Location or video link.")

    # Job context
    job_title: str = Field(..., min_length=1, max_length=200, description="Role being interviewed for.")

    # Approval
    approved_by: str = Field(..., min_length=1, max_length=200, description="Name of human approver.")
    approved_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        description="UTC timestamp of human approval.",
    )

    # Audit
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        description="UTC timestamp when this proposal was created.",
    )
    notes: str = Field(default="", description="Optional scheduling notes.")

    @field_validator("start_time", "end_time", "approved_at", "created_at", mode="before")
    @classmethod
    def _ensure_timezone_aware(cls, v: datetime) -> datetime:
        if isinstance(v, datetime) and v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    @model_validator(mode="after")
    def _end_after_start(self) -> "InterviewProposal":
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be after start_time.")
        return self

    def to_confirmation_text(self) -> str:
        """Human-readable confirmation message."""
        return (
            f"Interview Confirmed\n"
            f"  Candidate : {self.candidate_name} ({self.candidate_email or 'no email'})\n"
            f"  Role      : {self.job_title}\n"
            f"  Time      : {self.start_time.strftime('%A %d %b %Y %H:%M %Z')}\n"
            f"  Duration  : {int((self.end_time - self.start_time).total_seconds() // 60)} min\n"
            f"  Interviewer: {self.interviewer}\n"
            f"  Location  : {self.location}\n"
            f"  Approved by: {self.approved_by} at {self.approved_at.isoformat()}"
        )

    def __str__(self) -> str:
        return (
            f"InterviewProposal(candidate={self.candidate_name!r}, "
            f"slot={self.slot_id!r}, start={self.start_time.isoformat()})"
        )
