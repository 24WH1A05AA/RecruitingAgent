"""
tools/__init__.py
-----------------
Public interface for the TechVest Recruiting Agent tool layer.

Exports the four LangChain tools that the LangGraph agent can invoke,
plus the shared config utilities.

Usage
-----
    from tools import parse_resume, score_candidate, check_availability, propose_interview
    from tools.config import get_llm, get_settings
"""

from tools.check_availability import check_availability
from tools.parse_resume import parse_resume
from tools.propose_interview import propose_interview
from tools.score_candidate import score_candidate

# Convenience list for registering all tools with a LangChain agent
ALL_TOOLS = [parse_resume, score_candidate, check_availability, propose_interview]

__all__ = [
    "parse_resume",
    "score_candidate",
    "check_availability",
    "propose_interview",
    "ALL_TOOLS",
]
