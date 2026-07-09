"""
agent/prompts/shortlist.py
--------------------------
Prompt for generating an executive summary of shortlisted candidates.
"""

from __future__ import annotations

from agent.prompts.base import PromptTemplate

TEMPLATE = """You are a recruiting coordinator preparing a shortlist report.

Write a concise executive summary (3-5 sentences) of the shortlisted candidates
for the hiring manager.  Highlight the top candidate, note any notable strengths
across the group, and flag any concerns.

Job Title  : {jd_title}
Company    : {jd_company}
Shortlisted candidates (ranked):
{shortlist_summary}

Return plain prose — no JSON, no markdown headers.
"""

SHORTLIST_PROMPT = PromptTemplate(
    name="shortlist",
    template=TEMPLATE,
    version="1.0",
    description="Shortlist summary report generation",
    variables=("jd_title", "jd_company", "shortlist_summary"),
)

def render(jd_title: str, jd_company: str, shortlist_summary: str) -> str:
    """Render the shortlist summary prompt.

    Parameters
    ----------
    jd_title:
        Title of the job description.
    jd_company:
        Company name.
    shortlist_summary:
        Summary of the shortlisted candidates.

    Returns
    -------
    str
        The rendered prompt template.
    """
    return SHORTLIST_PROMPT.render(
        jd_title=jd_title,
        jd_company=jd_company,
        shortlist_summary=shortlist_summary,
    )
