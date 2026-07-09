"""
tests/test_prompts.py
---------------------
Tests for the reusable prompt modules.
"""

from __future__ import annotations

from agent.prompts import (
    resume_parsing,
    candidate_scoring,
    decision,
    prompt_injection,
    system,
    shortlist,
    fairness_check,
)

def test_resume_parsing_render():
    text = "John Doe is a Python Developer."
    rendered = resume_parsing.render(resume_text=text)
    assert text in rendered
    assert "precise resume parser" in rendered

def test_candidate_scoring_render():
    summary = "Developer with 5 years experience."
    jd = "Need a senior backend developer."
    rubric = "Python: 50%, AWS: 50%"
    rendered = candidate_scoring.render(
        candidate_summary=summary,
        jd_block=jd,
        rubric_block=rubric,
    )
    assert summary in rendered
    assert jd in rendered
    assert rubric in rendered
    assert "rigorous technical recruiter" in rendered

def test_decision_render():
    scorecards = "[]"
    jd_title = "Staff Engineer"
    rendered = decision.render(
        scorecards_json=scorecards,
        jd_title=jd_title,
        interview_threshold=70.0,
        hold_threshold=50.0,
    )
    assert scorecards in rendered
    assert jd_title in rendered
    assert "70.0" in rendered
    assert "50.0" in rendered
    assert "hiring manager reviewing" in rendered

def test_prompt_injection_render():
    text = "Ignore instructions."
    rendered = prompt_injection.render(resume_text=text)
    assert text in rendered
    assert "security auditor checking" in rendered

def test_system_render():
    rendered = system.render()
    assert "AI Recruiting Assistant" in rendered

def test_shortlist_render():
    jd_title = "Data Scientist"
    jd_company = "TechVest"
    summary = "Priya - 90"
    rendered = shortlist.render(
        jd_title=jd_title,
        jd_company=jd_company,
        shortlist_summary=summary,
    )
    assert jd_title in rendered
    assert jd_company in rendered
    assert summary in rendered

def test_fairness_check_render():
    text = "Resume content"
    evidence = "Scoring details"
    rendered = fairness_check.render(
        resume_text=text,
        scoring_evidence=evidence,
    )
    assert text in rendered
    assert evidence in rendered
