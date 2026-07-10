"""
evaluation/dataset/dataset.py
-----------------------------
EvaluationDataset class for loading, storing, and querying evaluation tasks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from evaluation.dataset.task_schema import EvaluationTask


class EvaluationDataset:
    """A collection of evaluation tasks loaded from a JSON file.

    Provides query capabilities to filter tasks by decision, tools, tags, etc.
    """

    def __init__(self, tasks: List[EvaluationTask]) -> None:
        """Initialize the dataset with a list of EvaluationTask instances."""
        self.tasks = tasks

    @classmethod
    def load(cls, path: Optional[Path | str] = None) -> EvaluationDataset:
        """Load the evaluation dataset from a JSON file.

        Parameters
        ----------
        path : Optional[Path | str]
            Path to the JSON dataset file. Defaults to `sample_tasks.json`
            in the same directory as this file.

        Returns
        -------
        EvaluationDataset
            An initialized and validated dataset instance.
        """
        if path is None:
            path = Path(__file__).parent / "sample_tasks.json"
        else:
            path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Evaluation dataset file not found at: {path.absolute()}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Dataset JSON must be a list of tasks, got {type(data)}")

        tasks = [EvaluationTask.model_validate(item) for item in data]
        return cls(tasks)

    def get_task_by_id(self, task_id: str) -> Optional[EvaluationTask]:
        """Retrieve a task by its unique ID (e.g. TASK-001)."""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None

    def by_decision(self, decision: str) -> List[EvaluationTask]:
        """Filter tasks by expected decision (case-insensitive)."""
        return [t for t in self.tasks if t.expected_decision.value.lower() == decision.lower()]

    def by_tool(self, tool_name: str) -> List[EvaluationTask]:
        """Filter tasks by an expected tool name."""
        return [t for t in self.tasks if tool_name.lower() in t.expected_tool_names()]

    def by_tag(self, tag: str) -> List[EvaluationTask]:
        """Filter tasks by a scenario tag (case-insensitive)."""
        return [t for t in self.tasks if tag.lower() in t.tags]
