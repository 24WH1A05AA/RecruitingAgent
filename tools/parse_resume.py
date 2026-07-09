"""
tools/parse_resume.py
---------------------
LangChain tool: parse raw resume text into a structured CandidateProfile.

Public API
----------
parse_resume(raw_text, file_path) -> CandidateProfile
    Calls an OpenRouter LLM to extract structured fields from free-form
    resume text and returns a validated CandidateProfile Pydantic model.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from models.candidate import CandidateProfile
from tools.config import PARSE_MODEL, get_llm

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a precise resume parser for a technical recruiting system.

Your job is to extract structured information from raw resume text and return it
as a single valid JSON object — nothing else.

IMPORTANT RULES:
- Return ONLY raw JSON. No markdown fences, no extra text, no explanation.
- If a field cannot be determined, use its default: null for optional strings,
  [] for lists, 0.0 for numbers.
- Do NOT invent information that is not present in the resume.
- skills: extract ALL technical skills, programming languages, frameworks,
  tools, and platforms mentioned.
- years_of_experience: calculate from work history dates; use 0.0 if unclear.
- education: list each degree as "Degree, Institution, Year" when available.
- certifications: only real professional certifications (not course completions).
- projects: list up to 5 notable projects with one-line descriptions.
- name / email / phone: take directly from the resume header; never guess.

Return this exact JSON schema (all keys required):
{
  "name": "<string>",
  "email": "<string>",
  "phone": "<string or null>",
  "skills": ["<string>", ...],
  "years_of_experience": <float>,
  "education": ["<string>", ...],
  "certifications": ["<string>", ...],
  "projects": ["<string>", ...]
}"""

_HUMAN_TEMPLATE = """Parse the following resume and return structured JSON.

--- RESUME START ---
{resume_text}
--- RESUME END ---"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_json_response(raw: str) -> str:
    """Strip markdown code fences and leading/trailing whitespace."""
    # Remove ```json ... ``` or ``` ... ``` wrappers
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw.strip())
    return raw.strip()


def _extract_json(raw: str) -> dict[str, Any]:
    """Extract the first JSON object from a potentially noisy LLM response."""
    cleaned = _clean_json_response(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: find the outermost {...} block
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Could not parse JSON from LLM response:\n{raw[:500]}")


def _build_profile(parsed: dict[str, Any], file_path: str) -> CandidateProfile:
    """Construct and validate a CandidateProfile from the parsed LLM dict."""
    return CandidateProfile(
        candidate_id=str(uuid.uuid4()),
        name=parsed.get("name") or "Unknown Candidate",
        email=parsed.get("email") or "unknown@example.com",
        phone=parsed.get("phone"),
        resume_text="[extracted — stored separately]",
        raw_file_path=file_path,
        skills=parsed.get("skills") or [],
        years_of_experience=float(parsed.get("years_of_experience") or 0.0),
        education=parsed.get("education") or [],
        certifications=parsed.get("certifications") or [],
        projects=parsed.get("projects") or [],
    )


# ---------------------------------------------------------------------------
# Core logic (retried, not decorated as a tool so it can be unit-tested)
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _call_llm_for_parse(resume_text: str) -> dict[str, Any]:
    """Call the OpenRouter LLM and return the parsed JSON dict.

    Retried up to 3 times with exponential backoff on transient errors.
    """
    llm = get_llm(model=PARSE_MODEL)
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=_HUMAN_TEMPLATE.format(resume_text=resume_text)),
    ]
    response = llm.invoke(messages)
    raw_content = response.content if hasattr(response, "content") else str(response)
    return _extract_json(str(raw_content))


# ---------------------------------------------------------------------------
# LangChain tool
# ---------------------------------------------------------------------------

@tool
def parse_resume(raw_text: str, file_path: str = "unknown") -> dict[str, Any]:
    """Parse raw resume text and return a structured CandidateProfile as a dict.

    This tool extracts the following fields from free-form resume text using
    an OpenRouter LLM (Llama 3.1 8B free tier):

    - name, email, phone
    - skills (all technical skills, languages, frameworks, tools)
    - years_of_experience (calculated from work history)
    - education (degrees, institutions, years)
    - certifications (professional certs only)
    - projects (up to 5 notable projects)

    Parameters
    ----------
    raw_text:
        Raw text content extracted from the candidate's resume PDF or DOCX.
        Must be at least 10 characters.
    file_path:
        Path to the original resume file.  Used to populate
        ``CandidateProfile.raw_file_path``.  Defaults to ``"unknown"``.

    Returns
    -------
    dict
        JSON-serialisable dict representation of a validated
        ``CandidateProfile``.  Keys match the model fields.
        Always returns a complete dict — on LLM / parse failure a minimal
        fallback profile is returned so the agent pipeline can continue.

    Raises
    ------
    Does not raise.  All exceptions are caught, logged, and a fallback
    profile dict is returned with ``is_fallback=True``.

    Examples
    --------
    >>> result = parse_resume.invoke({"raw_text": resume_text, "file_path": "priya.txt"})
    >>> profile = CandidateProfile(**result)
    """
    logger.info(f"parse_resume | file={file_path!r} | text_len={len(raw_text)}")

    if not raw_text or len(raw_text.strip()) < 10:
        logger.warning("parse_resume | resume text too short, returning fallback profile")
        return _fallback_profile(file_path, reason="Resume text is empty or too short.")

    try:
        parsed = _call_llm_for_parse(raw_text)
        profile = _build_profile(parsed, file_path)
        logger.success(
            f"parse_resume | parsed candidate={profile.name!r} "
            f"skills={len(profile.skills)} yoe={profile.years_of_experience}"
        )
        result = profile.model_dump()
        result["is_fallback"] = False
        return result

    except Exception as exc:
        logger.error(f"parse_resume | failed for file={file_path!r}: {exc}")
        return _fallback_profile(file_path, reason=str(exc))


def _fallback_profile(file_path: str, reason: str) -> dict[str, Any]:
    """Return a minimal profile dict when parsing fails, flagged with is_fallback=True."""
    profile = CandidateProfile(
        name="Parse Failed",
        email="parse-failed@techvest.internal",
        resume_text="[parse failed — no content extracted]",
        raw_file_path=file_path,
    )
    result = profile.model_dump()
    result["is_fallback"] = True
    result["parse_error"] = reason
    return result
