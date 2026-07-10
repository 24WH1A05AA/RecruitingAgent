"""
evaluation/dataset/__init__.py
------------------------------
Initialization for the TechVest evaluation dataset sub-package.

Exposes:
- EvaluationTask: Pydantic model for a single task schema
- EvaluationDataset: Dataset loader and query class
- get_task_by_id, get_tasks_by_decision, get_tasks_by_tool, get_tasks_by_tag: Helper functions
- summarize_dataset: Summarizes the dataset metrics
- validate_all_tasks: Validates the entire task dataset against Pydantic schemas
"""

from evaluation.dataset.task_schema import EvaluationTask
from evaluation.dataset.dataset import EvaluationDataset
from evaluation.dataset.tasks import (
    load_tasks,
    get_task,
    list_tasks,
    search_tasks,
    get_task_by_id,
    get_tasks_by_decision,
    get_tasks_by_tool,
    get_tasks_by_tag,
    summarize_dataset,
    validate_all_tasks,
)

__all__ = [
    "EvaluationTask",
    "EvaluationDataset",
    "load_tasks",
    "get_task",
    "list_tasks",
    "search_tasks",
    "get_task_by_id",
    "get_tasks_by_decision",
    "get_tasks_by_tool",
    "get_tasks_by_tag",
    "summarize_dataset",
    "validate_all_tasks",
]
