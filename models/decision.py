"""
models/decision.py
------------------
Pydantic model for the hiring recommendation from the agent's decision node.

Models
------
DecisionStatus  - enum of possible hiring outcomes
Decision        - structured hiring recommendation for one candidate
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, model_validator


class DecisionStatus(str, Enum):
    """Possible hiring outcomes for a candidate."""

    INTERVIEW = "interview"   # Score >= interview_threshold
    HOLD = "hold"             # Score >= hold_threshold but < interview_threshold
    REJECT = "reject"         # Score < hold_threshold

    def is_positive(self) -> bool:
        """True for outcomes that move the candidate forward."""
        return self == DecisionStatus.INTERVIEW


class Decision(BaseModel):
    """Structured hiring recommendation for a single candidate.

    Produced by the LangGraph decision node after scoring is complete.
    Immutable; captures threshold values at decision time for full auditability.
    """

    model_config = {"frozen": True, "str_strip_whitespace": True}

    # Identity
    candidate_id: str = Field(..., min_length=1, description="Matches CandidateProfile.candidate_id.")
    candidate_name: str = Field(..., min_length=1, max_length=200, description="Display name.")

    # Outcome
    status: DecisionStatus = Field(..., description="Hiring recommendation: interview | hold | reject.")
    total_score: Annotated[float, Field(ge=0.0, le=100.0)] = Field(
        ..., description="total_score from ScoreCard at time of decision."
    )
    rank: int | None = Field(default=None, ge=1, description="Rank in scored cohort (1 = highest).")

    # Thresholds captured at decision time for audit trail
    interview_threshold: Annotated[float, Field(ge=0.0, le=100.0)] = Field(
        default=70.0, description="Minimum score for INTERVIEW recommendation."
    )
    hold_threshold: Annotated[float, Field(ge=0.0, le=100.0)] = Field(
        default=50.0, description="Minimum score for HOLD recommendation."
    )

    # Reasoning
    reasoning: str = Field(
        ...,
        min_length=10,
        description="Explanation of why this decision was reached, referencing score and threshold.",
    )
    fairness_flags: list[str] = Field(
        default_factory=list,
        description="Bias or fairness concerns flagged during evaluation. Empty = no concerns.",
    )

    # Audit
    decided_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        description="UTC timestamp when the decision was recorded.",
    )

    # Validators
    @model_validator(mode="after")
    def _status_consistent_with_thresholds(self) -> "Decision":
        if self.interview_threshold < self.hold_threshold:
            raise ValueError(
                f"interview_threshold ({self.interview_threshold}) must be >= "
                f"hold_threshold ({self.hold_threshold})."
            )
        score = self.total_score
        if score >= self.interview_threshold:
            expected = DecisionStatus.INTERVIEW
        elif score >= self.hold_threshold:
            expected = DecisionStatus.HOLD
        else:
            expected = DecisionStatus.REJECT

        if self.status != expected:
            raise ValueError(
                f"status '{self.status}' is inconsistent with "
                f"total_score={score}, interview_threshold={self.interview_threshold}, "
                f"hold_threshold={self.hold_threshold}. Expected: '{expected}'."
            )
        return self

    # Helpers
    def to_audit_line(self) -> str:
        """Compact single-line audit log entry."""
        flags = f" [flags: {', '.join(self.fairness_flags)}]" if self.fairness_flags else ""
        return (
            f"[{self.decided_at.isoformat()}] DECISION | {self.candidate_name} "
            f"(id={self.candidate_id}) | score={self.total_score:.1f} | "
            f"rank={self.rank} | status={self.status.value.upper()}{flags}"
        )

    def __str__(self) -> str:
        return (
            f"Decision(candidate={self.candidate_name!r}, "
            f"score={self.total_score:.1f}, status={self.status.value})"
        )
