"""
models/job_description.py
-------------------------
Pydantic model for the job description fed into the TechVest Recruiting Agent.

Models
------
JobDescription  - structured representation of a job opening
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class JobDescription(BaseModel):
    """Structured representation of a job opening.

    Validated on construction; immutable after creation so it can be safely
    shared across all graph nodes without risk of mutation.
    """

    model_config = {"frozen": True, "str_strip_whitespace": True}

    # Core identity
    title: str = Field(
        ...,
        min_length=2,
        max_length=200,
        description="Job title (e.g. 'Senior ML Engineer').",
    )
    company: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Hiring company name.",
    )

    # Requirements
    required_skills: list[str] = Field(
        default_factory=list,
        description="Must-have skills the candidate must demonstrate.",
    )
    preferred_skills: list[str] = Field(
        default_factory=list,
        description="Nice-to-have skills that strengthen the application.",
    )
    min_years_experience: float = Field(
        default=0.0,
        ge=0.0,
        le=50.0,
        description="Minimum years of relevant professional experience required.",
    )
    education_requirement: str = Field(
        default="",
        description="Minimum education level (e.g. \"Bachelor's in CS or equivalent\").",
    )
    certifications: list[str] = Field(
        default_factory=list,
        description="Desired or required certifications.",
    )

    # Full text
    description: str = Field(
        ...,
        min_length=20,
        description="Full job description text as posted (used verbatim in LLM prompts).",
    )

    # Validators
    @field_validator("required_skills", "preferred_skills", "certifications", mode="before")
    @classmethod
    def _strip_list_items(cls, v: list) -> list[str]:
        """Strip whitespace and drop empty strings from list fields."""
        return [item.strip() for item in v if isinstance(item, str) and item.strip()]

    # Helpers
    def all_skills(self) -> list[str]:
        """Combined deduplicated list of required + preferred skills."""
        seen: set[str] = set()
        result: list[str] = []
        for skill in self.required_skills + self.preferred_skills:
            if skill.lower() not in seen:
                seen.add(skill.lower())
                result.append(skill)
        return result

    def to_prompt_block(self) -> str:
        """Render the JD as a compact block for inclusion in LLM prompts."""
        lines = [
            f"Job Title: {self.title}",
            f"Company  : {self.company}",
            f"Min Experience: {self.min_years_experience} years",
        ]
        if self.education_requirement:
            lines.append(f"Education: {self.education_requirement}")
        if self.required_skills:
            lines.append(f"Required Skills : {', '.join(self.required_skills)}")
        if self.preferred_skills:
            lines.append(f"Preferred Skills: {', '.join(self.preferred_skills)}")
        if self.certifications:
            lines.append(f"Certifications  : {', '.join(self.certifications)}")
        lines += ["", "Full Description:", self.description]
        return "\n".join(lines)

    def __str__(self) -> str:
        return f"JobDescription(title={self.title!r}, company={self.company!r})"
