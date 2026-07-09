"""
models/scorecard.py
-------------------
Pydantic models for per-candidate scoring output.

Models
------
CriterionScore  - score + evidence for a single rubric criterion
ScoreCard       - full evaluation result for one candidate
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, field_validator, model_validator


class CriterionScore(BaseModel):
    """Score and supporting evidence for a single rubric criterion."""

    model_config = {"frozen": True, "str_strip_whitespace": True}

    criterion: str = Field(..., min_length=1, max_length=100, description="Criterion name.")
    raw_score: Annotated[float, Field(ge=0.0, le=100.0)] = Field(
        ..., description="Unweighted score on 0–100 scale."
    )
    weight: Annotated[float, Field(gt=0.0, le=1.0)] = Field(
        ..., description="Criterion weight as decimal fraction (e.g. 0.35)."
    )
    weighted_score: Annotated[float, Field(ge=0.0, le=100.0)] = Field(
        ..., description="raw_score × weight, contributing to total score."
    )
    evidence: str = Field(..., min_length=5, description="Justification for this score.")

    @model_validator(mode="after")
    def _check_weighted_consistency(self) -> "CriterionScore":
        expected = round(self.raw_score * self.weight, 4)
        if abs(self.weighted_score - expected) > 0.5:
            raise ValueError(
                f"weighted_score {self.weighted_score} is inconsistent with "
                f"raw_score {self.raw_score} × weight {self.weight} = {expected}."
            )
        return self


class ScoreCard(BaseModel):
    """Complete evaluation result for a single candidate.

    Immutable; total_score is validated against the sum of weighted criterion scores.
    """

    model_config = {"frozen": True, "str_strip_whitespace": True}

    # Identity
    candidate_id: str = Field(..., min_length=1, description="Matches CandidateProfile.candidate_id.")
    candidate_name: str = Field(..., min_length=1, max_length=200, description="Display name.")

    # Scores
    criterion_scores: list[CriterionScore] = Field(
        ..., min_length=1, description="One entry per evaluated rubric criterion."
    )
    total_score: Annotated[float, Field(ge=0.0, le=100.0)] = Field(
        ..., description="Sum of all weighted_scores (0–100)."
    )

    # Ranking (set after comparing all candidates)
    rank: int | None = Field(default=None, ge=1, description="Position in ranked shortlist (1 = best).")

    # Narrative
    summary_evidence: str = Field(default="", description="Overall LLM-generated justification.")

    # Validators
    @field_validator("criterion_scores", mode="before")
    @classmethod
    def _require_non_empty(cls, v: list) -> list:
        if not v:
            raise ValueError("criterion_scores must contain at least one entry.")
        return v

    @model_validator(mode="after")
    def _check_total_consistency(self) -> "ScoreCard":
        computed = sum(cs.weighted_score for cs in self.criterion_scores)
        if abs(self.total_score - computed) > 1.0:
            raise ValueError(
                f"total_score {self.total_score} differs from sum of weighted "
                f"criterion scores {computed:.4f} by more than 1.0."
            )
        return self

    # Helpers
    def score_for(self, criterion: str) -> CriterionScore | None:
        """Return the CriterionScore for a given name, or None."""
        return next(
            (cs for cs in self.criterion_scores if cs.criterion.lower() == criterion.lower()),
            None,
        )

    def as_dict_for_display(self) -> dict:
        """Flat dict for Streamlit tables / JSON export."""
        row: dict = {
            "candidate_id": self.candidate_id,
            "candidate_name": self.candidate_name,
            "total_score": round(self.total_score, 2),
            "rank": self.rank,
        }
        for cs in self.criterion_scores:
            row[f"{cs.criterion}_raw"] = round(cs.raw_score, 2)
            row[f"{cs.criterion}_weighted"] = round(cs.weighted_score, 2)
        return row

    def __str__(self) -> str:
        return (
            f"ScoreCard(candidate={self.candidate_name!r}, "
            f"total={self.total_score:.1f}, rank={self.rank})"
        )
