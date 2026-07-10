"""
evaluation/security/__init__.py
------------------------------
Initialization for the TechVest Recruitment Agent security and red-team evaluation module.

Exposes:
- SecurityScenario: Pydantic model for security attack scenarios.
- SecurityFinding: Result of security checks.
- SecurityEvaluator: Audit engine that validates run behaviors.
- SecurityEvaluationReport: Formats and outputs MD, JSON, and Console reports.
"""

from evaluation.security.redteam import SecurityScenario, SecurityFinding, SecurityEvaluator, SecurityEvaluationReport

__all__ = [
    "SecurityScenario",
    "SecurityFinding",
    "SecurityEvaluator",
    "SecurityEvaluationReport",
]
