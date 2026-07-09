"""
agent/prompts/fairness_check.py
-------------------------------
Prompt for auditing evaluations for fairness/demographic bias.
"""

from __future__ import annotations

from agent.prompts.base import PromptTemplate

TEMPLATE = """You are a fairness auditor reviewing a candidate evaluation.

Check whether any of the following demographic or protected-attribute signals
appear in the resume text or were used in the scoring evidence.

Protected attributes to check for:
  gender, age, race, ethnicity, religion, nationality, disability,
  marital status, pregnancy, sexual orientation, political affiliation

For each attribute found, describe exactly where it appeared and whether it
could have influenced scoring.

Resume text:
{resume_text}

Scoring evidence:
{scoring_evidence}

Return a JSON object:
{{
  "flags": [
    {{
      "attribute": "<protected attribute>",
      "location": "resume | scoring_evidence",
      "excerpt": "<direct quote>",
      "risk": "low | medium | high"
    }}
  ],
  "overall_risk": "none | low | medium | high",
  "recommendation": "<action to take, or 'No action required'>"
}}

If no protected attributes are found, return {{"flags": [], "overall_risk": "none",
"recommendation": "No action required"}}.
"""

FAIRNESS_CHECK_PROMPT = PromptTemplate(
    name="fairness_check",
    template=TEMPLATE,
    version="1.0",
    description="Detect demographic bias signals in a resume and scoring evidence",
    variables=("resume_text", "scoring_evidence"),
)

def render(resume_text: str, scoring_evidence: str) -> str:
    """Render the fairness check prompt.

    Parameters
    ----------
    resume_text:
        The candidate's resume text.
    scoring_evidence:
        The candidate scoring evidence.

    Returns
    -------
    str
        The rendered prompt template.
    """
    return FAIRNESS_CHECK_PROMPT.render(
        resume_text=resume_text,
        scoring_evidence=scoring_evidence,
    )
