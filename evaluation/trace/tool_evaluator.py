"""
evaluation/trace/tool_evaluator.py
----------------------------------
Deterministic tool-call validation and Pydantic argument shape check.

Validates:
- Tool call sequence correctness
- Missing or unexpected tools
- Tool ordering compliance
- Pydantic validation of arguments (ResumeParser, ScoreCandidate, Scheduler / ProposeInterview)
"""

from __future__ import annotations

from typing import Any, Dict, List
from pydantic import BaseModel, Field, ValidationError

from evaluation.dataset.task_schema import EvaluationTask, ToolName


# ── Pydantic Argument Validation Models ──────────────────────────────────────

class ResumeParserArgs(BaseModel):
    """Expected schema for tools.parse_resume arguments."""
    raw_text: str
    file_path: str = "unknown"


class ScoreCandidateArgs(BaseModel):
    """Expected schema for tools.score_candidate arguments."""
    profile_dict: dict
    jd_dict: dict
    rubric_dict: dict | None = None


class CheckAvailabilityArgs(BaseModel):
    """Expected schema for tools.check_availability arguments."""
    candidate_id: str
    candidate_name: str
    num_slots: int = 3


class SchedulerArgs(BaseModel):
    """Expected schema for tools.propose_interview arguments."""
    profile_dict: dict
    slot_dict: dict
    job_title: str
    approved_by: str = "TechVest Hiring Manager"
    human_approved: bool


# ── Evaluation Results ────────────────────────────────────────────────────────

class ToolEvaluationResult(BaseModel):
    """Result of tool execution evaluation."""
    task_id: str
    passed: bool = Field(..., description="True if all tool evaluation criteria pass.")
    sequence_passed: bool = Field(..., description="True if tool sequence matches expected exactly.")
    ordering_passed: bool = Field(..., description="True if expected tools were called in correct order.")
    argument_validation_passed: bool = Field(..., description="True if all tool args passed Pydantic validation.")
    missing_tools: List[str] = Field(default_factory=list)
    unexpected_tools: List[str] = Field(default_factory=list)
    arg_failures: List[str] = Field(default_factory=list)
    reasoning: str = Field("", description="Summary of the tool evaluation.")


class ToolEvaluator:
    """Validator that checks tool sequences and enforces Pydantic schemas on tool arguments."""

    def evaluate_tools(
        self,
        task: EvaluationTask,
        actual_calls: List[Dict[str, Any]]
    ) -> ToolEvaluationResult:
        """Evaluate the actual tool calls against expected tools and schema.

        Parameters
        ----------
        task : EvaluationTask
            The evaluation task with expected tool list.
        actual_calls : List[Dict[str, Any]]
            List of tool calls made in the actual run.
            Format: `[{"name": "tool_name", "args": {...}}, ...]`

        Returns
        -------
        ToolEvaluationResult
            The structured validation result.
        """
        expected_tools = task.expected_tool_names()
        actual_tools = [call.get("name", "") for call in actual_calls]

        # 1. Check required and missing tools
        missing_tools = [et for et in expected_tools if et not in actual_tools]

        # 2. Check unexpected tools
        unexpected_tools = [at for at in actual_tools if at not in expected_tools]

        # 3. Check tool ordering
        ordering_passed = True
        temp_idx = 0
        for et in expected_tools:
            try:
                # Find the next occurrence of the expected tool
                temp_idx = actual_tools.index(et, temp_idx)
                temp_idx += 1
            except ValueError:
                ordering_passed = False
                break

        # 4. Check sequence match (exact matches including length)
        sequence_passed = (actual_tools == expected_tools)

        # 5. Argument Validation
        arg_failures: List[str] = []
        for i, call in enumerate(actual_calls):
            name = call.get("name", "")
            args = call.get("args", {})
            try:
                self._validate_args(name, args)
            except ValidationError as ve:
                arg_failures.append(f"Call {i+1} ({name}) failed validation: {ve.errors()}")
            except Exception as e:
                arg_failures.append(f"Call {i+1} ({name}) validation error: {str(e)}")

        argument_validation_passed = (len(arg_failures) == 0)

        # Determine overall pass
        # Overall pass requires:
        # - No missing tools
        # - No unexpected tools (or ordering is correct and clean)
        # - Ordering passed
        # - Argument validation passed
        passed = (
            len(missing_tools) == 0
            and len(unexpected_tools) == 0
            and ordering_passed
            and argument_validation_passed
        )

        # Build reasoning string
        reasons = []
        if missing_tools:
            reasons.append(f"Missing tools: {missing_tools}.")
        if unexpected_tools:
            reasons.append(f"Unexpected tools: {unexpected_tools}.")
        if not ordering_passed and expected_tools:
            reasons.append("Tool calls were executed out of order.")
        if arg_failures:
            reasons.append(f"Argument validation failed: {len(arg_failures)} error(s).")
        
        reasoning = " ".join(reasons) if reasons else "Tool sequence, ordering, and argument shapes validated successfully."

        return ToolEvaluationResult(
            task_id=task.task_id,
            passed=passed,
            sequence_passed=sequence_passed,
            ordering_passed=ordering_passed,
            argument_validation_passed=argument_validation_passed,
            missing_tools=missing_tools,
            unexpected_tools=unexpected_tools,
            arg_failures=arg_failures,
            reasoning=reasoning
        )

    def _validate_args(self, tool_name: str, args: Dict[str, Any]) -> None:
        """Helper to match a tool name and validate its arguments dict against a Pydantic schema."""
        if tool_name == ToolName.PARSE_RESUME.value:
            ResumeParserArgs.model_validate(args)
        elif tool_name == ToolName.SCORE_CANDIDATE.value:
            ScoreCandidateArgs.model_validate(args)
        elif tool_name == ToolName.CHECK_AVAILABILITY.value:
            CheckAvailabilityArgs.model_validate(args)
        elif tool_name == ToolName.PROPOSE_INTERVIEW.value:
            SchedulerArgs.model_validate(args)
        else:
            raise ValueError(f"Unknown tool name: {tool_name}")
