"""
evaluation/dataset/task_schema.py
----------------------------------
Pydantic schema for a single evaluation task in the TechVest evaluation dataset.

This module is entirely declarative — it defines data shapes and validation
rules only.  It never imports or invokes any agent component.

Design principles
-----------------
- All models are immutable (frozen=True) so tasks cannot be mutated after load.
- Every field carries a description and sensible defaults where optional.
- Validators enforce structural consistency (e.g. score ranges, tool ordering).
- Enums mirror the vocabulary used in the agent (DecisionStatus, tool names, node
  names) so evaluation results can be compared directly against agent outputs.

Classes
-------
ExpectedDecision        Enum of valid hiring outcomes (mirrors agent DecisionStatus)
ToolName                Enum of valid tool names (mirrors tools/__init__.py ALL_TOOLS)
NodeName                Enum of valid LangGraph node names (mirrors agent/graph.py)
GuardrailOutcome        Enum describing injection / fairness guard outcomes
ScoreRange              Min/max score band a candidate is expected to fall within
ExpectedCriterionScore  Per-criterion score expectation (raw score band + evidence keyword)
ExpectedTrace           Ordered sequence of nodes expected to fire for this task
PassCriteria            Structured acceptance conditions for a single task
EvaluationTask          Top-level task record — the primary schema of the dataset
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums — mirrors the agent's own vocabulary (no agent import needed)
# ---------------------------------------------------------------------------

class ExpectedDecision(str, Enum):
    """Hiring decision the agent is expected to produce for this task.

    Values deliberately match ``models.decision.DecisionStatus`` so string
    comparison works without importing the agent model.
    """

    INTERVIEW = "interview"
    """Score >= 70: candidate should be shortlisted for interview."""

    HOLD = "hold"
    """50 <= score < 70: candidate is borderline — place on hold."""

    REJECT = "reject"
    """Score < 50: candidate does not meet the minimum threshold."""

    BLOCKED = "blocked"
    """Candidate resume was blocked by a guardrail before scoring."""


class ToolName(str, Enum):
    """LangChain tool names available in the agent.

    Mirrors the ``tools/__init__.py`` ``ALL_TOOLS`` list.
    """

    PARSE_RESUME = "parse_resume"
    SCORE_CANDIDATE = "score_candidate"
    CHECK_AVAILABILITY = "check_availability"
    PROPOSE_INTERVIEW = "propose_interview"


class NodeName(str, Enum):
    """LangGraph node names in the recruitment graph.

    Mirrors the node names registered in ``agent/graph.py``
    ``build_graph()``.
    """

    INJECTION_GUARD = "injection_guard_node"
    PARSE_RESUME = "parse_resume_node"
    FAIRNESS_CHECK = "fairness_check_node"
    SCORE_CANDIDATE = "score_candidate_node"
    RANK_CANDIDATES = "rank_candidates_node"
    GENERATE_SHORTLIST = "generate_shortlist_node"
    CHECK_AVAILABILITY = "check_availability_node"
    HUMAN_APPROVAL = "human_approval_node"
    SCHEDULE_INTERVIEW = "schedule_interview_node"


class GuardrailOutcome(str, Enum):
    """Expected outcome of a guardrail node for this task."""

    PASS = "pass"
    """Guardrail found nothing suspicious — candidate proceeds."""

    BLOCKED = "blocked"
    """Guardrail detected a violation (injection / fairness) — candidate skipped."""

    FLAGGED = "flagged"
    """Guardrail raised a warning but did not block the candidate."""


# ---------------------------------------------------------------------------
# Score range model
# ---------------------------------------------------------------------------

class ScoreRange(BaseModel):
    """Inclusive score band [min_score, max_score] the agent is expected to produce.

    Validates that min <= max and both values are in the 0–100 range.
    """

    model_config = {"frozen": True}

    min_score: Annotated[float, Field(ge=0.0, le=100.0)] = Field(
        ...,
        description="Minimum acceptable total score (inclusive, 0–100).",
        examples=[55.0, 70.0],
    )
    max_score: Annotated[float, Field(ge=0.0, le=100.0)] = Field(
        ...,
        description="Maximum acceptable total score (inclusive, 0–100).",
        examples=[75.0, 95.0],
    )

    @model_validator(mode="after")
    def _min_lte_max(self) -> "ScoreRange":
        """Ensure min_score does not exceed max_score."""
        if self.min_score > self.max_score:
            raise ValueError(
                f"min_score ({self.min_score}) must be <= max_score ({self.max_score})."
            )
        return self

    def contains(self, score: float) -> bool:
        """Return True if ``score`` falls within [min_score, max_score]."""
        return self.min_score <= score <= self.max_score

    def __str__(self) -> str:
        return f"[{self.min_score:.1f}, {self.max_score:.1f}]"


# ---------------------------------------------------------------------------
# Per-criterion score expectation
# ---------------------------------------------------------------------------

class ExpectedCriterionScore(BaseModel):
    """Expected score band and evidence keywords for a single rubric criterion.

    Criteria names must match the TechVest default rubric:
    python_skills (35%), machine_learning (25%), projects (20%),
    communication (10%), education (10%).
    """

    model_config = {"frozen": True, "str_strip_whitespace": True}

    #: Criterion identifier — must match Rubric.criteria[*].name exactly.
    criterion: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description=(
            "Rubric criterion name. Must be one of: python_skills, machine_learning, "
            "projects, communication, education."
        ),
        examples=["python_skills", "machine_learning"],
    )

    #: Acceptable score band for this individual criterion (raw, 0–100).
    score_range: ScoreRange = Field(
        ...,
        description="Acceptable raw score range for this criterion (0–100).",
    )

    #: Keywords that should appear somewhere in the criterion's evidence string.
    evidence_keywords: list[str] = Field(
        default_factory=list,
        description=(
            "One or more keywords expected to appear in the LLM's evidence "
            "for this criterion. Used for keyword-presence checking, not exact match."
        ),
        examples=[["PyTorch", "NLP pipeline"], ["scikit-learn"]],
    )

    @field_validator("criterion")
    @classmethod
    def _valid_criterion_name(cls, v: str) -> str:
        """Validate criterion name against the known TechVest rubric criteria."""
        allowed = {
            "python_skills",
            "machine_learning",
            "projects",
            "communication",
            "education",
        }
        if v.lower() not in allowed:
            raise ValueError(
                f"Unknown criterion '{v}'. Must be one of: {sorted(allowed)}."
            )
        return v.lower()

    def __str__(self) -> str:
        return f"ExpectedCriterionScore(criterion={self.criterion!r}, range={self.score_range})"


# ---------------------------------------------------------------------------
# Expected trajectory (node execution sequence)
# ---------------------------------------------------------------------------

class ExpectedTrace(BaseModel):
    """Describes the expected LangGraph node execution sequence for a task.

    The ``nodes`` list is ordered — it defines which nodes should fire, in
    what order, for the agent to be considered correct on this task.
    """

    model_config = {"frozen": True}

    #: Ordered list of node names expected to execute.
    nodes: list[NodeName] = Field(
        ...,
        min_length=1,
        description=(
            "Ordered sequence of LangGraph node names expected to fire. "
            "Must contain at least injection_guard_node."
        ),
    )

    #: Whether the graph is expected to reach the human approval interrupt.
    reaches_human_approval: bool = Field(
        default=False,
        description=(
            "True if this task's candidate should reach the human_approval_node "
            "interrupt point (i.e. at least one candidate is shortlisted)."
        ),
    )

    #: Whether schedule_interview_node is expected to execute.
    scheduling_occurs: bool = Field(
        default=False,
        description=(
            "True if schedule_interview_node is expected to run. "
            "Requires reaches_human_approval=True."
        ),
    )

    #: Expected outcome of the injection guardrail for this task's resume.
    injection_guard_outcome: GuardrailOutcome = Field(
        default=GuardrailOutcome.PASS,
        description="Expected result of the injection_guard_node for this task.",
    )

    #: Expected outcome of the fairness check guardrail.
    fairness_check_outcome: GuardrailOutcome = Field(
        default=GuardrailOutcome.PASS,
        description="Expected result of the fairness_check_node for this task.",
    )

    @field_validator("nodes", mode="before")
    @classmethod
    def _nodes_not_empty(cls, v: list) -> list:
        if not v:
            raise ValueError("nodes list must contain at least one entry.")
        return v

    @model_validator(mode="after")
    def _scheduling_requires_approval(self) -> "ExpectedTrace":
        """Scheduling can only occur after human approval."""
        if self.scheduling_occurs and not self.reaches_human_approval:
            raise ValueError(
                "scheduling_occurs=True requires reaches_human_approval=True."
            )
        return self

    @model_validator(mode="after")
    def _injection_guard_first(self) -> "ExpectedTrace":
        """injection_guard_node must always be the first node in the trace."""
        if self.nodes and self.nodes[0] != NodeName.INJECTION_GUARD:
            raise ValueError(
                f"First node must be injection_guard_node, got '{self.nodes[0]}'."
            )
        return self

    def __str__(self) -> str:
        return (
            f"ExpectedTrace(nodes={[n.value for n in self.nodes]}, "
            f"approval={self.reaches_human_approval})"
        )


# ---------------------------------------------------------------------------
# Pass criteria
# ---------------------------------------------------------------------------

class PassCriteria(BaseModel):
    """Conditions that must hold for an agent run to be considered passing
    on this evaluation task.

    All boolean flags default to True (strict mode).  Set to False to
    explicitly declare that a criterion is not evaluated for this task.
    """

    model_config = {"frozen": True}

    #: Agent must produce the expected_decision value.
    decision_matches: bool = Field(
        default=True,
        description="Agent's hiring decision must match expected_decision.",
    )

    #: Agent's total score must fall within expected_score_range.
    score_in_range: bool = Field(
        default=True,
        description="Agent's total_score must lie within expected_score_range.",
    )

    #: All nodes in expected_trace.nodes must appear in agent's audit_log.
    trace_includes_required_nodes: bool = Field(
        default=True,
        description=(
            "All nodes listed in expected_trace.nodes must appear in the "
            "agent's audit log / trajectory."
        ),
    )

    #: At least one required_evidence keyword must appear in the agent's output.
    evidence_keywords_present: bool = Field(
        default=True,
        description=(
            "At least one keyword from required_evidence must appear in the "
            "agent's summary_evidence or criterion evidence strings."
        ),
    )

    #: All expected tools must have been called (present in agent messages).
    all_expected_tools_called: bool = Field(
        default=True,
        description=(
            "Every tool listed in expected_tools must appear in the agent's "
            "tool-call log / messages for this candidate."
        ),
    )

    #: Injection guard outcome must match expected_trace.injection_guard_outcome.
    injection_guard_outcome_matches: bool = Field(
        default=True,
        description=(
            "The injection guardrail result must match the expected outcome "
            "defined in expected_trace.injection_guard_outcome."
        ),
    )

    #: Per-criterion score ranges must hold (only checked when criteria_scores is set).
    per_criterion_scores_in_range: bool = Field(
        default=False,
        description=(
            "If True, every criterion in expected_criterion_scores must have "
            "its raw_score within the defined range. Defaults to False because "
            "LLM scores naturally vary."
        ),
    )

    def active_criteria(self) -> list[str]:
        """Return names of all pass criteria that are enabled (True)."""
        return [
            field_name
            for field_name, value in self.model_dump().items()
            if value is True
        ]

    def __str__(self) -> str:
        active = self.active_criteria()
        return f"PassCriteria(active={active})"


# ---------------------------------------------------------------------------
# Primary evaluation task model
# ---------------------------------------------------------------------------

class EvaluationTask(BaseModel):
    """A single evaluation task in the TechVest recruitment agent dataset.

    Each task represents one candidate scenario with a fully specified
    resume, expected agent behaviour, and acceptance conditions.

    This model is **read-only** — it never triggers agent execution.
    It is used by evaluators to compare expected vs actual agent outputs.

    Field groups
    ------------
    Identity
        task_id, title, description, tags, version
    Candidate
        candidate_name, candidate_resume
    Expectations
        expected_decision, expected_score_range, expected_criterion_scores,
        expected_tools, expected_trace, required_evidence
    Acceptance
        pass_criteria
    Metadata
        notes, created_by
    """

    model_config = {"frozen": True, "str_strip_whitespace": True}

    # ── Identity ──────────────────────────────────────────────────────────────

    task_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        pattern=r"^TASK-\d{3}$",
        description="Unique task identifier in the format TASK-NNN (e.g. TASK-001).",
        examples=["TASK-001", "TASK-010"],
    )

    title: str = Field(
        ...,
        min_length=5,
        max_length=200,
        description="Short human-readable title summarising the evaluation scenario.",
        examples=["Senior ML Engineer — Strong INTERVIEW candidate"],
    )

    description: str = Field(
        ...,
        min_length=20,
        description=(
            "Detailed description of what this task is testing and why the "
            "expected outcome is correct."
        ),
    )

    tags: list[str] = Field(
        default_factory=list,
        description=(
            "Scenario classification tags for filtering. "
            "e.g. ['interview', 'existing_candidate', 'high_score']"
        ),
        examples=[["interview", "existing_candidate", "high_score"]],
    )

    version: str = Field(
        default="1.0",
        description="Semver version of this task record.",
    )

    # ── Candidate ──────────────────────────────────────────────────────────────

    candidate_name: str = Field(
        ...,
        min_length=2,
        max_length=200,
        description="Full name of the candidate as it appears in the resume header.",
        examples=["Priya Sharma", "John Smith"],
    )

    candidate_resume: str = Field(
        ...,
        min_length=50,
        description=(
            "Complete raw resume text as it would be loaded from a .txt file "
            "and passed to the agent's parse_resume tool."
        ),
    )

    # ── Expectations ──────────────────────────────────────────────────────────

    expected_decision: ExpectedDecision = Field(
        ...,
        description=(
            "Hiring decision the agent should produce: interview | hold | reject | blocked. "
            "interview requires total_score >= 70. hold requires 50 <= score < 70. "
            "reject requires score < 50. blocked means a guardrail stopped the candidate."
        ),
    )

    expected_score_range: ScoreRange = Field(
        ...,
        description=(
            "Inclusive [min, max] band for the agent's total_score. "
            "Set a tight range for deterministic mock LLM tests; "
            "use a wider range for live LLM tolerance."
        ),
    )

    expected_criterion_scores: list[ExpectedCriterionScore] = Field(
        default_factory=list,
        description=(
            "Per-criterion score expectations. Leave empty to skip per-criterion "
            "checking. When provided, must cover at least one rubric criterion."
        ),
    )

    expected_tools: list[ToolName] = Field(
        ...,
        description=(
            "Ordered list of tool names the agent must invoke for this candidate. "
            "For a standard non-blocked candidate: "
            "[parse_resume, score_candidate, check_availability, propose_interview]. "
            "For a blocked candidate: [] or [parse_resume] only."
        ),
    )

    expected_trace: ExpectedTrace = Field(
        ...,
        description=(
            "Expected LangGraph node execution sequence and guardrail outcomes."
        ),
    )

    required_evidence: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "Keywords or phrases that must appear in the agent's scoring evidence "
            "or summary. Used for keyword-presence evaluation. "
            "At least one must match for the evidence criterion to pass."
        ),
        examples=[["PyTorch", "NLP pipeline", "production"], ["Django", "web scraper"]],
    )

    # ── Acceptance ────────────────────────────────────────────────────────────

    pass_criteria: PassCriteria = Field(
        default_factory=PassCriteria,
        description=(
            "Structured acceptance conditions. Defaults to strict mode "
            "(all boolean flags True except per_criterion_scores_in_range)."
        ),
    )

    # ── Metadata ──────────────────────────────────────────────────────────────

    notes: str = Field(
        default="",
        description=(
            "Free-text notes for evaluators: edge cases, known LLM variance, "
            "design rationale, or instructions for manual review."
        ),
    )

    created_by: str = Field(
        default="TechVest Evaluation Team",
        description="Author or team that created this evaluation task.",
    )

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("tags", mode="before")
    @classmethod
    def _strip_tags(cls, v: list) -> list[str]:
        """Strip whitespace and de-duplicate tags."""
        seen: set[str] = set()
        result: list[str] = []
        for tag in v:
            t = str(tag).strip().lower()
            if t and t not in seen:
                seen.add(t)
                result.append(t)
        return result

    @field_validator("expected_tools", mode="before")
    @classmethod
    def _tools_not_empty(cls, v: list) -> list:
        """Tools list may be empty for blocked candidates (guard fires first)."""
        # Allow empty list — blocked candidates have no tools called
        return v

    @model_validator(mode="after")
    def _decision_score_range_consistent(self) -> "EvaluationTask":
        """Enforce consistency between expected_decision and expected_score_range."""
        decision = self.expected_decision
        sr = self.expected_score_range

        if decision == ExpectedDecision.INTERVIEW:
            # Interview threshold is 70.0 in the agent
            if sr.max_score < 70.0:
                raise ValueError(
                    f"expected_decision=INTERVIEW but max_score={sr.max_score} < 70.0. "
                    "An INTERVIEW decision requires total_score >= 70."
                )
        elif decision == ExpectedDecision.HOLD:
            # Hold band: 50 <= score < 70
            if sr.max_score >= 70.0:
                raise ValueError(
                    f"expected_decision=HOLD but max_score={sr.max_score} >= 70.0. "
                    "A HOLD decision requires total_score < 70."
                )
            if sr.min_score < 50.0:
                raise ValueError(
                    f"expected_decision=HOLD but min_score={sr.min_score} < 50.0. "
                    "A HOLD decision requires total_score >= 50."
                )
        elif decision == ExpectedDecision.REJECT:
            # Reject: score < 50
            if sr.max_score >= 50.0:
                raise ValueError(
                    f"expected_decision=REJECT but max_score={sr.max_score} >= 50.0. "
                    "A REJECT decision requires total_score < 50."
                )
        # BLOCKED tasks may have any score range (score may not be computed at all)

        return self

    @model_validator(mode="after")
    def _blocked_decision_consistency(self) -> "EvaluationTask":
        """Blocked tasks must have injection_guard_outcome=BLOCKED in expected_trace."""
        if self.expected_decision == ExpectedDecision.BLOCKED:
            if self.expected_trace.injection_guard_outcome != GuardrailOutcome.BLOCKED:
                raise ValueError(
                    "expected_decision=BLOCKED requires "
                    "expected_trace.injection_guard_outcome=GuardrailOutcome.BLOCKED."
                )
        return self

    @model_validator(mode="after")
    def _no_duplicate_criteria(self) -> "EvaluationTask":
        """Ensure expected_criterion_scores has no duplicate criterion names."""
        if self.expected_criterion_scores:
            names = [c.criterion for c in self.expected_criterion_scores]
            if len(names) != len(set(names)):
                dupes = {n for n in names if names.count(n) > 1}
                raise ValueError(
                    f"Duplicate criterion names in expected_criterion_scores: {dupes}"
                )
        return self

    # ── Helpers ───────────────────────────────────────────────────────────────

    def is_blocked(self) -> bool:
        """Return True if this task expects the candidate to be blocked."""
        return self.expected_decision == ExpectedDecision.BLOCKED

    def is_positive(self) -> bool:
        """Return True if this task expects an INTERVIEW decision."""
        return self.expected_decision == ExpectedDecision.INTERVIEW

    def expected_node_names(self) -> list[str]:
        """Return expected node names as plain strings."""
        return [n.value for n in self.expected_trace.nodes]

    def expected_tool_names(self) -> list[str]:
        """Return expected tool names as plain strings."""
        return [t.value for t in self.expected_tools]

    def score_range_midpoint(self) -> float:
        """Return the midpoint of the expected score range."""
        sr = self.expected_score_range
        return (sr.min_score + sr.max_score) / 2.0

    def __str__(self) -> str:
        return (
            f"EvaluationTask(id={self.task_id!r}, candidate={self.candidate_name!r}, "
            f"decision={self.expected_decision.value}, "
            f"range={self.expected_score_range})"
        )

    def __repr__(self) -> str:
        return (
            f"EvaluationTask(task_id={self.task_id!r}, "
            f"candidate_name={self.candidate_name!r}, "
            f"expected_decision={self.expected_decision!r})"
        )
