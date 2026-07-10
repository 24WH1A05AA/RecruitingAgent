"""
evaluation/dataset/tasks.py
---------------------------
Convenience helper functions for interacting with the evaluation dataset.
"""

from __future__ import annotations

from typing import List, Optional
from loguru import logger

from evaluation.dataset.dataset import EvaluationDataset
from evaluation.dataset.task_schema import EvaluationTask


def load_tasks() -> List[EvaluationTask]:
    """Load the default evaluation dataset tasks.

    Returns
    -------
    List[EvaluationTask]
        List of all validated evaluation tasks.
    """
    ds = EvaluationDataset.load()
    return ds.tasks


def get_task(task_id: str) -> Optional[EvaluationTask]:
    """Retrieve a single evaluation task by its unique task ID.

    Parameters
    ----------
    task_id : str
        Unique identifier (e.g., 'TASK-001').

    Returns
    -------
    Optional[EvaluationTask]
        The task instance, or None if not found.
    """
    ds = EvaluationDataset.load()
    return ds.get_task_by_id(task_id)


def list_tasks() -> List[EvaluationTask]:
    """List all available evaluation tasks in the dataset.

    Returns
    -------
    List[EvaluationTask]
        All evaluation tasks.
    """
    return load_tasks()


def search_tasks(query: str) -> List[EvaluationTask]:
    """Search tasks by title, description, candidate name, or tags (case-insensitive).

    Parameters
    ----------
    query : str
        The query string to search for.

    Returns
    -------
    List[EvaluationTask]
        Filter list of tasks matching the query.
    """
    tasks = load_tasks()
    q = query.lower()
    return [
        t for t in tasks
        if q in t.title.lower()
        or q in t.description.lower()
        or q in t.candidate_name.lower()
        or any(q in tag for tag in t.tags)
    ]


def get_task_by_id(task_id: str) -> Optional[EvaluationTask]:
    """Retrieve a single task by its unique ID. Alias for get_task."""
    return get_task(task_id)


def get_tasks_by_decision(decision: str) -> List[EvaluationTask]:
    """Retrieve all tasks matching a specific expected decision outcome.

    Parameters
    ----------
    decision : str
        Expected decision (e.g. 'interview', 'hold', 'reject', 'blocked').

    Returns
    -------
    List[EvaluationTask]
        List of matching tasks.
    """
    ds = EvaluationDataset.load()
    return ds.by_decision(decision)


def get_tasks_by_tool(tool_name: str) -> List[EvaluationTask]:
    """Retrieve all tasks expecting a specific tool to be invoked.

    Parameters
    ----------
    tool_name : str
        Name of the expected tool (e.g. 'parse_resume', 'score_candidate').

    Returns
    -------
    List[EvaluationTask]
        List of matching tasks.
    """
    ds = EvaluationDataset.load()
    return ds.by_tool(tool_name)


def get_tasks_by_tag(tag: str) -> List[EvaluationTask]:
    """Retrieve all tasks containing a specific tag.

    Parameters
    ----------
    tag : str
        Tag identifier.

    Returns
    -------
    List[EvaluationTask]
        List of matching tasks.
    """
    ds = EvaluationDataset.load()
    return ds.by_tag(tag)


def summarize_dataset() -> dict:
    """Return a dictionary summarizing the dataset metrics and counts.

    Returns
    -------
    dict
        Summary dictionary with counts of decisions, total tasks, and tags.
    """
    tasks = load_tasks()
    summary = {
        "total_tasks": len(tasks),
        "by_decision": {},
        "by_tags": {},
        "total_blocked": 0,
        "total_interview": 0,
        "total_hold": 0,
        "total_reject": 0,
    }
    for t in tasks:
        dec = t.expected_decision.value
        summary["by_decision"][dec] = summary["by_decision"].get(dec, 0) + 1
        
        # update decision counts
        if dec == "blocked":
            summary["total_blocked"] += 1
        elif dec == "interview":
            summary["total_interview"] += 1
        elif dec == "hold":
            summary["total_hold"] += 1
        elif dec == "reject":
            summary["total_reject"] += 1
            
        for tag in t.tags:
            summary["by_tags"][tag] = summary["by_tags"].get(tag, 0) + 1
            
    return summary


def validate_all_tasks() -> bool:
    """Validate all tasks against their schema.

    Returns
    -------
    bool
        True if all tasks validate successfully.

    Raises
    ------
    ValidationError
        If any task dataset file fails Pydantic validation.
    """
    try:
        load_tasks()
        return True
    except Exception as e:
        logger.error(f"Dataset validation failed: {e}")
        raise e
