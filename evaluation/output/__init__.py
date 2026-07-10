"""
evaluation/output/__init__.py
-----------------------------
Initialization for the TechVest Recruitment Agent output evaluation module.

Exposes:
- OutputEvaluator: Evaluates LLM outputs (Faithfulness, Relevancy, Completion, Decision Quality).
- OutputEvaluationResult: Schema for output validation results.
- OutputEvaluationReport: Consolidator of output evaluations and fairness audits.
- FairnessEvaluator: Compares base runs against name-swapped runs.
- FairnessResult: Schema for fairness validation results.
- swap_candidate_name: Helper function for demographic name replacement.
"""

from evaluation.output.output_evaluator import OutputEvaluator, OutputEvaluationResult, OutputEvaluationReport
from evaluation.output.fairness import FairnessEvaluator, FairnessResult, swap_candidate_name

__all__ = [
    "OutputEvaluator",
    "OutputEvaluationResult",
    "OutputEvaluationReport",
    "FairnessEvaluator",
    "FairnessResult",
    "swap_candidate_name",
]
