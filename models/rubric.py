"""
models/rubric.py
----------------
Pydantic model for the weighted scoring rubric used by score_candidate().

Models
------
RubricCriterion  - a single criterion with weight and scoring guidance
Rubric           - the full rubric applied to a job description
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class RubricCriterion(BaseModel):
    """A single evaluation criterion with its weight and scoring guidance."""

    model_config = {"frozen": True, "str_strip_whitespace": True}

    name: str = Field(..., min_length=1, max_length=100, description="Criterion identifier.")
    description: str = Field(..., min_length=5, description="What is being evaluated.")
    weight: float = Field(..., gt=0.0, le=1.0, description="Fractional weight (e.g. 0.35).")
    scoring_guide: str = Field(
        default="", description="LLM guidance on how to assign a 0–100 raw score."
    )
    max_score: float = Field(default=100.0, gt=0.0, le=100.0, description="Max achievable raw score.")

    def __str__(self) -> str:
        return f"RubricCriterion(name={self.name!r}, weight={self.weight})"


# Default criteria matching the TechVest spec (weights sum to 1.0)
_DEFAULT_CRITERIA: list[dict] = [
    {
        "name": "python_skills",
        "description": "Depth and breadth of Python programming skills.",
        "weight": 0.35,
        "scoring_guide": (
            "80–100: expert Python (frameworks, async, testing). "
            "50–79: solid working knowledge. 0–49: limited evidence."
        ),
    },
    {
        "name": "machine_learning",
        "description": "Experience with ML/AI concepts, libraries, and real projects.",
        "weight": 0.25,
        "scoring_guide": (
            "80–100: hands-on ML deployment (scikit-learn, PyTorch, MLOps). "
            "50–79: academic/project exposure. 0–49: minimal evidence."
        ),
    },
    {
        "name": "projects",
        "description": "Quality and relevance of projects listed in the resume.",
        "weight": 0.20,
        "scoring_guide": (
            "80–100: production-grade or open-source projects with impact. "
            "50–79: meaningful side-projects. 0–49: academic exercises only."
        ),
    },
    {
        "name": "communication",
        "description": "Clarity of written communication inferred from the resume.",
        "weight": 0.10,
        "scoring_guide": (
            "80–100: well-structured, quantified impact statements. "
            "50–79: clear but basic. 0–49: vague or hard-to-parse."
        ),
    },
    {
        "name": "education",
        "description": "Relevance and level of formal education.",
        "weight": 0.10,
        "scoring_guide": (
            "80–100: relevant postgraduate degree (CS, ML, Data Science). "
            "60–79: relevant undergraduate degree. "
            "40–59: unrelated degree or equivalent experience. 0–39: no formal degree."
        ),
    },
]


class Rubric(BaseModel):
    """Full weighted rubric applied when scoring candidates.

    Validates that criterion weights sum to exactly 1.0 (±0.01 tolerance).
    """

    model_config = {"frozen": True, "str_strip_whitespace": True}

    name: str = Field(..., min_length=1, max_length=200, description="Human-readable rubric name.")
    criteria: list[RubricCriterion] = Field(..., min_length=1, description="Evaluation criteria.")
    version: str = Field(default="1.0", description="Semver version for rubric traceability.")
    notes: str = Field(default="", description="Free-text notes (JD source, revision history).")

    # Validators
    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> "Rubric":
        total = sum(c.weight for c in self.criteria)
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Criterion weights must sum to 1.0, but sum is {total:.4f}."
            )
        return self

    @model_validator(mode="after")
    def _no_duplicate_names(self) -> "Rubric":
        names = [c.name.lower() for c in self.criteria]
        if len(names) != len(set(names)):
            dupes = {n for n in names if names.count(n) > 1}
            raise ValueError(f"Duplicate criterion name(s): {dupes}")
        return self

    # Helpers
    def criterion_names(self) -> list[str]:
        return [c.name for c in self.criteria]

    def weight_for(self, name: str) -> float | None:
        return next(
            (c.weight for c in self.criteria if c.name.lower() == name.lower()), None
        )

    def as_prompt_text(self) -> str:
        """Render rubric as a formatted block for LLM prompts."""
        lines = [f"Rubric: {self.name} (v{self.version})", ""]
        for c in self.criteria:
            lines.append(f"  [{c.weight * 100:.0f}%] {c.name}: {c.description}")
            if c.scoring_guide:
                lines.append(f"       Guide: {c.scoring_guide}")
        return "\n".join(lines)

    @classmethod
    def default(cls) -> "Rubric":
        """Return the standard TechVest rubric for a Senior ML Engineer role."""
        return cls(
            name="TechVest Senior ML Engineer",
            criteria=[RubricCriterion(**c) for c in _DEFAULT_CRITERIA],
            version="1.0",
            notes="Default rubric based on TechVest Spec v1.",
        )

    def __str__(self) -> str:
        return f"Rubric(name={self.name!r}, criteria={len(self.criteria)}, v{self.version})"
