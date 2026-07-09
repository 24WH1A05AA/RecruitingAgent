"""
models/candidate.py
-------------------
Pydantic model for a parsed candidate resume.

Models
------
CandidateProfile  - structured representation of a parsed resume
"""

from __future__ import annotations

import re
import uuid
from typing import Annotated

from pydantic import BaseModel, Field, field_validator, model_validator


class CandidateProfile(BaseModel):
    """Structured representation of a candidate parsed from a resume.

    Immutable after construction to prevent accidental mutation inside graph nodes.
    """

    model_config = {"frozen": True, "str_strip_whitespace": True}

    # Identity
    candidate_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier (UUID4).",
    )
    name: str = Field(..., min_length=1, max_length=200, description="Full legal name.")
    email: str = Field(..., description="Primary contact e-mail address.")
    phone: str | None = Field(default=None, description="Contact phone (E.164 or local).")

    # Resume content
    resume_text: str = Field(..., min_length=10, description="Raw text from resume file.")
    raw_file_path: str = Field(..., min_length=1, description="Path to original resume file.")

    # Parsed attributes
    skills: list[str] = Field(default_factory=list, description="Technical and soft skills.")
    years_of_experience: Annotated[float, Field(ge=0.0, le=60.0)] = Field(
        default=0.0, description="Total professional experience in years."
    )
    education: list[str] = Field(default_factory=list, description="Degrees and institutions.")
    certifications: list[str] = Field(default_factory=list, description="Professional certifications.")
    projects: list[str] = Field(default_factory=list, description="Notable projects from resume.")

    # Validators
    @field_validator("skills", "education", "certifications", "projects", mode="before")
    @classmethod
    def _strip_list_items(cls, v: list) -> list[str]:
        return [item.strip() for item in v if isinstance(item, str) and item.strip()]

    @field_validator("email", mode="before")
    @classmethod
    def _validate_email(cls, v: str) -> str:
        v = str(v).strip().lower()
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", v):
            raise ValueError(f"'{v}' is not a valid email address.")
        return v

    @field_validator("phone", mode="before")
    @classmethod
    def _validate_phone(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = str(v).strip()
        if not v:
            return None
        if not re.fullmatch(r"[+\d\s\-().]{7,20}", v):
            raise ValueError(f"Phone number '{v}' does not look valid.")
        return v

    @model_validator(mode="after")
    def _check_name_not_placeholder(self) -> "CandidateProfile":
        forbidden = {"unknown", "n/a", "none", "candidate", "applicant"}
        if self.name.lower().strip() in forbidden:
            raise ValueError(f"Candidate name '{self.name}' looks like a placeholder.")
        return self

    # Helpers
    def has_skill(self, skill: str) -> bool:
        """Case-insensitive skill lookup."""
        return any(s.lower() == skill.lower() for s in self.skills)

    def skill_overlap(self, required: list[str]) -> list[str]:
        """Intersection of candidate skills and a required list."""
        required_lower = {s.lower() for s in required}
        return [s for s in self.skills if s.lower() in required_lower]

    def __str__(self) -> str:
        return (
            f"CandidateProfile(id={self.candidate_id!r}, name={self.name!r}, "
            f"yoe={self.years_of_experience}, skills={len(self.skills)})"
        )
