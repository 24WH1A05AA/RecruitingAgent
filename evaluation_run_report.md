# Trace and Tool Evaluation Framework

This package implements an offline validation layer to inspect LangGraph trajectories (node sequences) and tool calls without executing or modifying the graph.

Package path: [evaluation/trace/](file:///C:/Users/marut/OneDrive/Documents/VYSHU/TechVestAgenticAIWorkshop/TechVestWorkshop/RecruitingAgent/evaluation/trace/)

---

## 1. Trace Invariants

Instead of exact trajectory matching, the trace evaluator checks structural invariants across actual run node lists:

1.  **Parse before Score**: Checks that `parse_resume_node` runs before `score_candidate_node` (if both are executed).
2.  **Borderline candidates must visit Verifier**: Validates that borderline profiles visit the verifier (`fairness_check_node` and/or `human_approval_node`).
3.  **No scheduling before Human Approval**: Validates that scheduling cannot occur prior to a human approval checkpoint interrupt.
4.  **Human Gate must precede Scheduler**: Confirms `human_approval_node` comes before `schedule_interview_node`.
5.  **No invalid routing**: Asserts that the run starts with `injection_guard_node`, terminates immediately if blocked, and does not run post-shortlist nodes for rejected/hold candidates.

---

## 2. Referenceless Trajectory Judges

-   **Step Efficiency**: Calculates trajectory efficiency based on expected vs. actual step lengths (penalizes loops or redundant nodes). If `deepeval` is installed, it can use the `StepEfficiencyMetric`.
-   **Task Completion**: Verifies if the graph reached the expected target node for a given candidate type. Fallbacks to `TaskCompletionMetric` when `deepeval` is present.

---

## 3. Deterministic Tool Evaluators

-   **Sequence & Ordering**: Validates that all expected tools were called, and that they ran in the correct relative sequence without omitting steps or injecting out-of-order tools.
-   **Argument Validation**: Uses strict Pydantic schemas to validate the arguments passed to each tool:
    -   `parse_resume` is validated against `ResumeParserArgs` (requiring `raw_text` and optionally `file_path`).
    -   `score_candidate` is validated against `ScoreCandidateArgs` (requiring `profile_dict` and `jd_dict`).
    -   `check_availability` is validated against `CheckAvailabilityArgs` (requiring `candidate_id` and `candidate_name`).
    -   `propose_interview` is validated against `SchedulerArgs` (requiring `profile_dict`, `slot_dict`, `job_title`, and `human_approved`).

---

## 4. Multi-Format Report Generation

The `EvaluationReport` gathers results over multiple test cases and renders them:
1.  **JSON**: Excellent for structured output parsing or automated CI/CD checks (`report.generate_json()`).
2.  **Markdown**: Renders detailed lists of invariant/tool failures, overall metrics, and recommendations (`report.generate_markdown()`).
3.  **Console**: Prints a human-readable CLI summary to stdout (`report.print_console()`).

---

## Usage Example

```python
from evaluation.dataset.tasks import get_task
from evaluation.trace import TraceEvaluator, ToolEvaluator, EvaluationReport

# 1. Load task spec
task = get_task("TASK-001")

# 2. Extract actual trace and calls from a run
actual_nodes = ["injection_guard_node", "parse_resume_node", "fairness_check_node", "score_candidate_node", "rank_candidates_node", "generate_shortlist_node", "check_availability_node", "human_approval_node", "schedule_interview_node"]
actual_calls = [
    {"name": "parse_resume", "args": {"raw_text": "...", "file_path": "priya.txt"}},
    {"name": "score_candidate", "args": {"profile_dict": {}, "jd_dict": {}}},
    {"name": "check_availability", "args": {"candidate_id": "c1", "candidate_name": "Priya"}},
    {"name": "propose_interview", "args": {"profile_dict": {}, "slot_dict": {}, "job_title": "ML Engineer", "human_approved": True}}
]

# 3. Evaluate trace and tools
trace_res = TraceEvaluator().evaluate_trace(task, actual_nodes, {"human_approved": True})
tool_res = ToolEvaluator().evaluate_tools(task, actual_calls)

# 4. Generate report
report = EvaluationReport()
report.add_result(
    task_id=task.task_id,
    expected_nodes=task.expected_node_names(),
    actual_nodes=actual_nodes,
    expected_tools=task.expected_tool_names(),
    actual_tools=[call["name"] for call in actual_calls],
    trace_res=trace_res,
    tool_res=tool_res
)
report.print_console()
```
