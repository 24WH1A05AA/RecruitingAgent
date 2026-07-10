"""
evaluation/__init__.py
----------------------
Top-level package for the TechVest Recruitment Agent evaluation framework.

This package is completely independent of the agent runtime.  It only defines
evaluation cases (tasks, datasets, helpers) — it never imports or invokes
any agent, LangGraph node, LLM tool, or Streamlit component.

Sub-packages
------------
evaluation.dataset
    Core dataset layer:
    - task_schema  : Pydantic models that describe one evaluation task
    - sample_tasks : 10 curated evaluation tasks as a JSON file
    - dataset      : EvaluationDataset loader and query class
    - tasks        : Convenience helper functions

Typical usage
-------------
    from evaluation.dataset import EvaluationDataset, get_task_by_id

    ds = EvaluationDataset.load()          # loads bundled sample_tasks.json
    task = get_task_by_id("TASK-001")
    interview_cases = ds.by_decision("interview")
"""

from evaluation.dataset import (
    EvaluationDataset,
    EvaluationTask,
    get_task_by_id,
    get_tasks_by_decision,
    get_tasks_by_tool,
    get_tasks_by_tag,
    summarize_dataset,
    validate_all_tasks,
)

__all__ = [
    # Dataset class
    "EvaluationDataset",
    # Schema model (re-exported for convenience)
    "EvaluationTask",
    # Helper functions
    "get_task_by_id",
    "get_tasks_by_decision",
    "get_tasks_by_tool",
    "get_tasks_by_tag",
    "summarize_dataset",
    "validate_all_tasks",
]

__version__ = "1.0.0"
