"""
agent/prompts/resume_parsing.py
-------------------------------
Prompt for parsing raw resume text into structured JSON.
"""

from __future__ import annotations

from agent.prompts.base import PromptTemplate

TEMPLATE = """You are a precise resume parser.

Extract the following fields from the resume text below and return a single
valid JSON object — no markdown, no commentary, just JSON.

Fields to extract:
  name              : Full name from the resume header
  email             : Email address
  phone             : Phone number (null if absent)
  skills            : All technical skills, languages, frameworks, tools
  years_of_experience : Total years of professional work experience (float)
  education         : List of "Degree, Institution, Year" strings
  certifications    : Professional certifications only (not course completions)
  projects          : Up to 5 notable projects with one-line descriptions

Rules:
- Never invent data not present in the resume.
- years_of_experience: sum work history durations; use 0.0 if unclear.
- If a field is absent, use null (strings) or [] (lists).

Return exactly this JSON schema:
{{
  "name": "<string>",
  "email": "<string>",
  "phone": "<string or null>",
  "skills": ["<string>", ...],
  "years_of_experience": <float>,
  "education": ["<string>", ...],
  "certifications": ["<string>", ...],
  "projects": ["<string>", ...]
}}

RESUME:
{resume_text}
"""

PARSE_RESUME_PROMPT = PromptTemplate(
    name="resume_parsing",
    template=TEMPLATE,
    version="1.0",
    description="Extract structured candidate info from resume text",
    variables=("resume_text",),
)

def render(resume_text: str) -> str:
    """Render the resume parsing prompt.

    Parameters
    ----------
    resume_text:
        The raw resume text.

    Returns
    -------
    str
        The rendered prompt template.
    """
    return PARSE_RESUME_PROMPT.render(resume_text=resume_text)
