"""
agent/prompts/decision.py
-------------------------
Prompt for making the hiring decision (ranking and status classification) on candidates.
"""

from __future__ import annotations

from agent.prompts.base import PromptTemplate

TEMPLATE = """You are a hiring manager reviewing a shortlist of scored candidates.

Rank the following candidates from highest to lowest total score.
For each, confirm their recommended status (interview / hold / reject) and provide
a one-sentence hiring rationale.

Candidates (JSON list of ScoreCard summaries):
{scorecards_json}

Job Description Title: {jd_title}
Interview Threshold  : {interview_threshold} / 100
Hold Threshold       : {hold_threshold} / 100

Return a JSON list ordered by rank (rank 1 = highest score):
[
  {{
    "rank": <int>,
    "candidate_id": "<string>",
    "candidate_name": "<string>",
    "total_score": <float>,
    "status": "interview" | "hold" | "reject",
    "rationale": "<one sentence>"
  }},
  ...
]
"""

RANK_CANDIDATES_PROMPT = PromptTemplate(
    name="decision",
    template=TEMPLATE,
    version="1.0",
    description="Rank + evidence generation across all candidates",
    variables=("scorecards_json", "jd_title", "interview_threshold", "hold_threshold"),
)

def render(
    scorecards_json: str,
    jd_title: str,
    interview_threshold: float,
    hold_threshold: float,
) -> str:
    """Render the candidate decision/ranking prompt.

    Parameters
    ----------
    scorecards_json:
        JSON string of candidate scorecards.
    jd_title:
        Title of the job description.
    interview_threshold:
        Threshold score for scheduling an interview.
    hold_threshold:
        Threshold score for putting a candidate on hold.

    Returns
    -------
    str
        The rendered prompt template.
    """
    return RANK_CANDIDATES_PROMPT.render(
        scorecards_json=scorecards_json,
        jd_title=jd_title,
        interview_threshold=interview_threshold,
        hold_threshold=hold_threshold,
    )
