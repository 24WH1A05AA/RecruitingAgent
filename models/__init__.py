"""
models/__init__.py
------------------
Public interface for the TechVest Recruiting Agent model layer.

Usage
-----
    from models import CandidateProfile, ScoreCard, Rubric, Decision, Trajectory, InterviewSlot
"""

from models.candidate import CandidateProfile
from models.decision import Decision, DecisionStatus
from models.interview import InterviewProposal, InterviewSlot, SlotStatus
from models.rubric import Rubric, RubricCriterion
from models.scorecard import CriterionScore, ScoreCard
from models.trajectory import StepKind, Trajectory, TrajectoryStep

__all__ = [
    # Candidate
    "CandidateProfile",
    # Scoring
    "CriterionScore",
    "ScoreCard",
    # Rubric
    "RubricCriterion",
    "Rubric",
    # Decision
    "DecisionStatus",
    "Decision",
    # Trajectory / audit
    "StepKind",
    "TrajectoryStep",
    "Trajectory",
    # Interview scheduling
    "SlotStatus",
    "InterviewSlot",
    "InterviewProposal",
]
