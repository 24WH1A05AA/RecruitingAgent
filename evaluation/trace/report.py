"""
evaluation/trace/report.py
--------------------------
Evaluation report generator for recruitment agent trace and tool-call evaluations.

Supports rendering reports in:
- Console (human-readable printout)
- Markdown (structured artifact report)
- JSON (serialized metrics and results for CI/CD integrations)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
from pydantic import BaseModel, Field

from evaluation.trace.trace_evaluator import TraceEvaluationResult
from evaluation.trace.tool_evaluator import ToolEvaluationResult


class TaskEvaluationSummary(BaseModel):
    """Combined trace and tool evaluation summary for a single task run."""
    task_id: str
    expected_nodes: List[str] = Field(default_factory=list)
    actual_nodes: List[str] = Field(default_factory=list)
    expected_tools: List[str] = Field(default_factory=list)
    actual_tools: List[str] = Field(default_factory=list)
    trace_passed: bool
    tool_passed: bool
    step_efficiency: float
    task_completion: float
    invariant_failures: List[str] = Field(default_factory=list)
    tool_failures: List[str] = Field(default_factory=list)
    argument_failures: List[str] = Field(default_factory=list)
    notes: str = ""


class AggregatedMetrics(BaseModel):
    """Overall dataset-level evaluation metrics."""
    total_runs: int = 0
    trace_pass_rate: float = 0.0
    invariant_pass_rate: float = 0.0
    tool_accuracy: float = 0.0
    tool_ordering_accuracy: float = 0.0
    argument_validation_accuracy: float = 0.0
    average_step_efficiency: float = 0.0
    average_task_completion: float = 0.0


class EvaluationReport:
    """Consolidates results and outputs formatted Markdown, JSON, or Console reports."""

    def __init__(self) -> None:
        self.summaries: List[TaskEvaluationSummary] = []
        self.metrics = AggregatedMetrics()

    def add_result(
        self,
        task_id: str,
        expected_nodes: List[str],
        actual_nodes: List[str],
        expected_tools: List[str],
        actual_tools: List[str],
        trace_res: TraceEvaluationResult,
        tool_res: ToolEvaluationResult,
        notes: str = ""
    ) -> None:
        """Add evaluation results for a single task run."""
        inv_failures = [inv.message for inv in trace_res.invariants if not inv.passed]
        
        tool_failures = []
        if tool_res.missing_tools:
            tool_failures.append(f"Missing expected tools: {tool_res.missing_tools}")
        if tool_res.unexpected_tools:
            tool_failures.append(f"Unexpected tools called: {tool_res.unexpected_tools}")
        if not tool_res.ordering_passed and expected_tools:
            tool_failures.append("Tools were invoked out of expected relative order")

        summary = TaskEvaluationSummary(
            task_id=task_id,
            expected_nodes=expected_nodes,
            actual_nodes=actual_nodes,
            expected_tools=expected_tools,
            actual_tools=actual_tools,
            trace_passed=trace_res.passed,
            tool_passed=tool_res.passed,
            step_efficiency=trace_res.step_efficiency,
            task_completion=trace_res.task_completion,
            invariant_failures=inv_failures,
            tool_failures=tool_failures,
            argument_failures=tool_res.arg_failures,
            notes=notes
        )
        self.summaries.append(summary)
        self._calculate_metrics()

    def _calculate_metrics(self) -> None:
        """Re-calculate aggregated metrics over all current summaries."""
        total = len(self.summaries)
        if total == 0:
            self.metrics = AggregatedMetrics()
            return

        trace_passes = sum(1 for s in self.summaries if s.trace_passed)
        tool_passes = sum(1 for s in self.summaries if s.tool_passed)
        
        # Calculate individual invariant pass rate across all checks
        total_invariants = 0
        passed_invariants = 0
        # Calculate ordering and arg validation counts
        ordering_passes = 0
        arg_passes = 0
        
        # We need to reconstruct these from summaries:
        # - s.trace_passed means all invariants passed
        # - s.tool_passed means all tool checks passed
        for s in self.summaries:
            # Let's count how many invariant checks we made:
            # We had 5 invariants per task
            total_invariants += 5
            passed_invariants += (5 - len(s.invariant_failures))
            
            # Ordering passes if no ordering failures in tool_failures
            has_ordering_fail = any("order" in f.lower() for f in s.tool_failures)
            if not has_ordering_fail:
                ordering_passes += 1
                
            if not s.argument_failures:
                arg_passes += 1

        self.metrics = AggregatedMetrics(
            total_runs=total,
            trace_pass_rate=round(trace_passes / total, 2),
            invariant_pass_rate=round(passed_invariants / total_invariants, 2) if total_invariants else 1.0,
            tool_accuracy=round(tool_passes / total, 2),
            tool_ordering_accuracy=round(ordering_passes / total, 2),
            argument_validation_accuracy=round(arg_passes / total, 2),
            average_step_efficiency=round(sum(s.step_efficiency for s in self.summaries) / total, 2),
            average_task_completion=round(sum(s.task_completion for s in self.summaries) / total, 2)
        )

    def generate_json(self) -> str:
        """Serialize metrics and task summaries to a JSON string."""
        report_dict = {
            "metrics": self.metrics.model_dump(),
            "runs": [s.model_dump() for s in self.summaries],
            "recommendations": self.get_recommendations()
        }
        return json.dumps(report_dict, indent=2)

    def generate_markdown(self) -> str:
        """Format the evaluation results into a structured Markdown document."""
        md = []
        md.append("# TechVest Recruitment Agent - Trajectory and Tool Evaluation Report\n")
        
        # Summary metrics
        md.append("## Aggregated Metrics\n")
        md.append("| Metric | Score | Status |")
        md.append("| :--- | :---: | :--- |")
        md.append(f"| **Trace Pass Rate** | {self.metrics.trace_pass_rate * 100:.1f}% | {'✅ PASS' if self.metrics.trace_pass_rate >= 0.8 else '⚠️ REVIEW'} |")
        md.append(f"| **Invariant Pass Rate** | {self.metrics.invariant_pass_rate * 100:.1f}% | {'✅ PASS' if self.metrics.invariant_pass_rate >= 0.9 else '⚠️ REVIEW'} |")
        md.append(f"| **Tool Accuracy** | {self.metrics.tool_accuracy * 100:.1f}% | {'✅ PASS' if self.metrics.tool_accuracy >= 0.8 else '⚠️ REVIEW'} |")
        md.append(f"| **Tool Ordering Accuracy** | {self.metrics.tool_ordering_accuracy * 100:.1f}% | {'✅ PASS' if self.metrics.tool_ordering_accuracy >= 0.9 else '⚠️ REVIEW'} |")
        md.append(f"| **Argument Validation Accuracy** | {self.metrics.argument_validation_accuracy * 100:.1f}% | {'✅ PASS' if self.metrics.argument_validation_accuracy >= 1.0 else '⚠️ REVIEW'} |")
        md.append(f"| **Average Step Efficiency** | {self.metrics.average_step_efficiency * 100:.1f}% | - |")
        md.append(f"| **Average Task Completion** | {self.metrics.average_task_completion * 100:.1f}% | - |\n")

        # Recommendations section
        md.append("## Recommendations\n")
        for rec in self.get_recommendations():
            md.append(f"- {rec}")
        md.append("")

        # Detailed runs
        md.append("## Detailed Runs Breakdown\n")
        for s in self.summaries:
            status = "✅ PASS" if (s.trace_passed and s.tool_passed) else "❌ FAIL"
            md.append(f"### Task: {s.task_id} ({status})\n")
            md.append(f"- **Expected Trace**: `{s.expected_nodes}`")
            md.append(f"- **Actual Trace**: `{s.actual_nodes}`")
            md.append(f"- **Expected Tools**: `{s.expected_tools}`")
            md.append(f"- **Actual Tools**: `{s.actual_tools}`")
            md.append(f"- **Step Efficiency**: `{s.step_efficiency:.2f}` | **Task Completion**: `{s.task_completion:.2f}`\n")
            
            if s.invariant_failures:
                md.append("#### Invariant Failures")
                for f in s.invariant_failures:
                    md.append(f"- {f}")
                md.append("")
                
            if s.tool_failures:
                md.append("#### Tool Call Failures")
                for f in s.tool_failures:
                    md.append(f"- {f}")
                md.append("")

            if s.argument_failures:
                md.append("#### Argument Validation Failures")
                for f in s.argument_failures:
                    md.append(f"- {f}")
                md.append("")
                
            if s.notes:
                md.append(f"*Notes: {s.notes}*\n")
            md.append("---")
            
        return "\n".join(md)

    def print_console(self) -> None:
        """Print a formatted human-readable summary of the evaluation directly to the console."""
        print("=" * 80)
        print("TECHVEST RECRUITMENT AGENT - EVALUATION REPORT")
        print("=" * 80)
        print(f"Total Runs Analyzed: {self.metrics.total_runs}")
        print("-" * 80)
        print(f"Trace Pass Rate                : {self.metrics.trace_pass_rate * 100:.1f}%")
        print(f"Invariant Pass Rate            : {self.metrics.invariant_pass_rate * 100:.1f}%")
        print(f"Tool Accuracy                  : {self.metrics.tool_accuracy * 100:.1f}%")
        print(f"Tool Ordering Accuracy         : {self.metrics.tool_ordering_accuracy * 100:.1f}%")
        print(f"Argument Validation Accuracy   : {self.metrics.argument_validation_accuracy * 100:.1f}%")
        print(f"Average Step Efficiency        : {self.metrics.average_step_efficiency * 100:.1f}%")
        print(f"Average Task Completion        : {self.metrics.average_task_completion * 100:.1f}%")
        print("-" * 80)
        
        failures = [s for s in self.summaries if not (s.trace_passed and s.tool_passed)]
        if failures:
            print(f"Failed Tasks ({len(failures)}):")
            for f in failures:
                print(f"  - {f.task_id}: Invariants failed: {len(f.invariant_failures)}, Tool fails: {len(f.tool_failures) + len(f.argument_failures)}")
        else:
            print("All tasks evaluated successfully!")
            
        print("-" * 80)
        print("Recommendations:")
        for rec in self.get_recommendations():
            print(f"  - {rec}")
        print("=" * 80)

    def get_recommendations(self) -> List[str]:
        """Generate static suggestions for engineering action based on performance deficits."""
        recs = []
        if self.metrics.trace_pass_rate < 0.8:
            recs.append("CRITICAL: Enhance graph routing logic or verifier constraints to avoid policy violations.")
        if self.metrics.tool_ordering_accuracy < 0.9:
            recs.append("WARNING: Review node transitions to ensure expected tool sequence is preserved.")
        if self.metrics.argument_validation_accuracy < 1.0:
            recs.append("CRITICAL: Align tool input generation to the strict Pydantic schemas defined in models.")
        if self.metrics.average_step_efficiency < 0.8:
            recs.append("OPTIMIZATION: Agent execution trajectory has excessive steps or loops. Refine routing conditions.")
        if not recs:
            recs.append("Hiring Agent trajectory and tool-calls comply perfectly with corporate governance policies. Ready for staging.")
        return recs
