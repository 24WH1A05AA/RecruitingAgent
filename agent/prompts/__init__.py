"""
agent/prompts/__init__.py
-------------------------
Exposes all prompt template strings for backward compatibility,
while importing them from their respective reusable modules.
"""

from __future__ import annotations

from agent.prompts.system import SYSTEM_PROMPT as _system_template
from agent.prompts.resume_parsing import PARSE_RESUME_PROMPT as _resume_parsing_template
from agent.prompts.candidate_scoring import SCORE_CANDIDATE_PROMPT as _candidate_scoring_template
from agent.prompts.decision import RANK_CANDIDATES_PROMPT as _decision_template
from agent.prompts.shortlist import SHORTLIST_PROMPT as _shortlist_template
from agent.prompts.fairness_check import FAIRNESS_CHECK_PROMPT as _fairness_check_template
from agent.prompts.prompt_injection import INJECTION_GUARD_PROMPT as _prompt_injection_template

SYSTEM_PROMPT: str = _system_template.template
PARSE_RESUME_PROMPT: str = _resume_parsing_template.template
SCORE_CANDIDATE_PROMPT: str = _candidate_scoring_template.template
RANK_CANDIDATES_PROMPT: str = _decision_template.template
SHORTLIST_PROMPT: str = _shortlist_template.template
FAIRNESS_CHECK_PROMPT: str = _fairness_check_template.template
INJECTION_GUARD_PROMPT: str = _prompt_injection_template.template

__all__ = [
    "SYSTEM_PROMPT",
    "PARSE_RESUME_PROMPT",
    "SCORE_CANDIDATE_PROMPT",
    "RANK_CANDIDATES_PROMPT",
    "SHORTLIST_PROMPT",
    "FAIRNESS_CHECK_PROMPT",
    "INJECTION_GUARD_PROMPT",
]
