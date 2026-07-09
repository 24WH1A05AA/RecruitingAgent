"""
models/trajectory.py
--------------------
Pydantic models for the agent's reasoning trace and audit trail.

Models
------
StepKind        - enum of node/action types in the graph
TrajectoryStep  - a single timestamped reasoning step
Trajectory      - ordered collection of steps for a full agent run
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class StepKind(str, Enum):
    """Categorises each step in the agent's reasoning trace."""

    PARSE_RESUME = "parse_resume"
    SCORE_CANDIDATE = "score_candidate"
    RANK_CANDIDATES = "rank_candidates"
    DECISION = "decision"
    CHECK_AVAILABILITY = "check_availability"
    HUMAN_APPROVAL = "human_approval"
    SCHEDULE_INTERVIEW = "schedule_interview"
    GUARDRAIL = "guardrail"
    AGENT_THOUGHT = "agent_thought"
    ERROR = "error"
    INFO = "info"


class TrajectoryStep(BaseModel):
    """A single timestamped event in the agent's reasoning trace. Immutable."""

    model_config = {"frozen": True, "str_strip_whitespace": True}

    kind: StepKind = Field(..., description="Category of this step.")
    node: str = Field(..., min_length=1, max_length=100, description="LangGraph node name.")

    # Optional candidate context
    candidate_id: str | None = Field(default=None, description="Related candidate ID, if applicable.")
    candidate_name: str | None = Field(default=None, description="Candidate display name.")

    # Content
    message: str = Field(..., min_length=1, description="Human-readable description of this step.")
    details: dict[str, Any] = Field(
        default_factory=dict, description="Structured payload (JSON-serialisable)."
    )
    is_error: bool = Field(default=False, description="True if this step represents an error.")

    # Timing
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        description="UTC timestamp when this step was recorded.",
    )
    iteration: int = Field(default=1, ge=1, description="Agent iteration counter at recording time.")

    @model_validator(mode="after")
    def _sync_error_kind_and_flag(self) -> "TrajectoryStep":
        """Keep StepKind.ERROR and is_error=True in sync."""
        if self.kind == StepKind.ERROR and not self.is_error:
            object.__setattr__(self, "is_error", True)
        return self

    def to_log_line(self) -> str:
        """Compact log line for audit files."""
        candidate_part = f" | candidate={self.candidate_name!r}" if self.candidate_name else ""
        error_part = " [ERROR]" if self.is_error else ""
        return (
            f"[{self.timestamp.isoformat()}] "
            f"iter={self.iteration:03d} | {self.kind.value} | {self.node}"
            f"{candidate_part}{error_part} | {self.message}"
        )

    def __str__(self) -> str:
        return f"TrajectoryStep(kind={self.kind.value}, node={self.node!r}, iter={self.iteration})"


class Trajectory(BaseModel):
    """Ordered collection of TrajectorySteps for a complete agent run.

    Mutable (not frozen) so steps can be appended during the graph run.
    """

    model_config = {"str_strip_whitespace": True}

    run_id: str = Field(..., min_length=1, description="Unique identifier for this agent run.")
    steps: list[TrajectoryStep] = Field(default_factory=list, description="Chronological steps.")
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        description="UTC timestamp when the run began.",
    )
    finished_at: datetime | None = Field(default=None, description="UTC timestamp on completion.")

    # Mutation helpers
    def add_step(
        self,
        kind: StepKind,
        node: str,
        message: str,
        *,
        candidate_id: str | None = None,
        candidate_name: str | None = None,
        details: dict[str, Any] | None = None,
        is_error: bool = False,
        iteration: int = 1,
    ) -> TrajectoryStep:
        """Append a new step and return it."""
        step = TrajectoryStep(
            kind=kind,
            node=node,
            message=message,
            candidate_id=candidate_id,
            candidate_name=candidate_name,
            details=details or {},
            is_error=is_error,
            iteration=iteration,
        )
        self.steps.append(step)
        return step

    def mark_finished(self) -> None:
        """Record the completion timestamp."""
        self.finished_at = datetime.now(tz=timezone.utc)

    # Query helpers
    def steps_for_candidate(self, candidate_id: str) -> list[TrajectoryStep]:
        return [s for s in self.steps if s.candidate_id == candidate_id]

    def errors(self) -> list[TrajectoryStep]:
        return [s for s in self.steps if s.is_error]

    def steps_by_kind(self, kind: StepKind) -> list[TrajectoryStep]:
        return [s for s in self.steps if s.kind == kind]

    # Export
    def to_log_lines(self) -> list[str]:
        return [s.to_log_line() for s in self.steps]

    def to_audit_text(self) -> str:
        """Full trace as a multi-line string."""
        finished = self.finished_at.isoformat() if self.finished_at else "IN PROGRESS"
        header = (
            f"=== TechVest Agent Run: {self.run_id} ===\n"
            f"Started : {self.started_at.isoformat()}\n"
            f"Finished: {finished}\n"
            f"Steps   : {len(self.steps)}\n"
            + "=" * 50
        )
        return header + "\n" + "\n".join(self.to_log_lines())

    def __len__(self) -> int:
        return len(self.steps)

    def __str__(self) -> str:
        return (
            f"Trajectory(run_id={self.run_id!r}, steps={len(self.steps)}, "
            f"errors={len(self.errors())})"
        )
