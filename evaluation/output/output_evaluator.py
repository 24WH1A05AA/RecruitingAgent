"""
evaluation/output/output_evaluator.py
-------------------------------------
Evaluation of agent outputs: Faithfulness, Relevancy, Task Completion, and Decision Quality.

Automatically detects whether DeepEval is installed and runs Faithfulness/Relevancy metrics,
falling back to rule-based checking if DeepEval is unavailable.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from evaluation.dataset.task_schema import EvaluationTask
from evaluation.output.fairness import FairnessResult

try:
    from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
    from deepeval.test_case import LLMTestCase
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False


class OutputEvaluationResult(BaseModel):
    """Result of output validation and quality scoring."""
    task_id: str
    passed: bool = Field(..., description="True if all critical quality checks pass.")
    faithfulness_score: float = Field(0.0, description="Faithfulness score [0.0 - 1.0].")
    relevancy_score: float = Field(0.0, description="Answer relevancy score [0.0 - 1.0].")
    task_completion_score: float = Field(0.0, description="Task completion score [0.0 - 1.0].")
    evidence_quality_score: float = Field(0.0, description="Evidence quality score [0.0 - 1.0].")
    decision_quality_score: float = Field(0.0, description="Decision quality score [0.0 - 1.0].")
    evidence_coverage: float = Field(0.0, description="Percentage of required keywords matched.")
    missing_evidence: List[str] = Field(default_factory=list, description="Expected keywords missing from evidence.")
    overall_score: float = Field(0.0, description="Combined overall output score [0.0 - 1.0].")
    recommendations: List[str] = Field(default_factory=list, description="Suggestions for LLM prompt improvements.")


class OutputEvaluator:
    """Evaluates the LLM-generated output decision card and reasoning strings."""

    def __init__(self, use_deepeval: bool = True) -> None:
        self.use_deepeval = use_deepeval and DEEPEVAL_AVAILABLE

    def evaluate_output(
        self,
        task: EvaluationTask,
        actual_decision: Dict[str, Any],
        candidate_resume: str,
        job_description: str
    ) -> OutputEvaluationResult:
        """Run output evaluation on the final decision card produced by the agent.

        Parameters
        ----------
        task : EvaluationTask
            The evaluation task specification.
        actual_decision : Dict[str, Any]
            The Decision dict produced by the agent (contains candidate_name, status, total_score, reasoning, etc.).
        candidate_resume : str
            The raw resume text of the candidate.
        job_description : str
            The raw job description requirements.

        Returns
        -------
        OutputEvaluationResult
            The structured scoring and feedback.
        """
        # Extract fields from actual decision card
        actual_verdict = actual_decision.get("status", "").lower()
        justification = actual_decision.get("reasoning", "")
        
        # In our agent state, cited evidence is usually stored in the ScoreCard model inside 'summary_evidence' or criteria evidence
        cited_evidence = actual_decision.get("summary_evidence", "")
        if not cited_evidence:
            cited_evidence = justification # fallback

        # 1. Faithfulness (reasoning vs resume evidence)
        faithfulness_score = self._evaluate_faithfulness(justification, cited_evidence, candidate_resume)

        # 2. Answer Relevancy (reasoning vs JD requirements)
        relevancy_score = self._evaluate_relevancy(justification, job_description)

        # 3. Task Completion (field checks)
        task_completion_score = self._evaluate_task_completion(actual_decision)

        # 4. Evidence Quality (keywords checks)
        evidence_quality_score, evidence_coverage, missing_evidence = self._evaluate_evidence_quality(task, cited_evidence)

        # 5. Decision Quality (expected vs actual verdict)
        decision_quality_score = self._evaluate_decision_quality(task, actual_verdict)

        # Calculate Overall Score (average of active metrics)
        overall_score = round(
            (faithfulness_score + relevancy_score + task_completion_score + evidence_quality_score + decision_quality_score) / 5.0,
            2
        )

        # Threshold check: requires total >= 0.70 and decision matches
        passed = (overall_score >= 0.70) and (decision_quality_score == 1.0)

        # Build recommendations
        recs = []
        if faithfulness_score < 0.8:
            recs.append("Faithfulness score is low. Instruct the LLM to only cite facts present in the resume.")
        if relevancy_score < 0.7:
            recs.append("Answer relevancy is low. Refine prompts to align candidate details with specific JD criteria.")
        if task_completion_score < 1.0:
            recs.append("Task completion fields are missing. Check that the final Decision dict contains all required fields.")
        if evidence_quality_score < 0.8:
            recs.append(f"Evidence quality issues detected. Missing critical keywords: {missing_evidence}.")
        if decision_quality_score == 0.0:
            recs.append(f"Decision mismatch. Expected '{task.expected_decision.value}', but agent returned '{actual_verdict}'. Check scoring rubric weights.")
        
        if not recs:
            recs.append("Output quality meets all rigorous criteria. Recommendation holds.")

        return OutputEvaluationResult(
            task_id=task.task_id,
            passed=passed,
            faithfulness_score=faithfulness_score,
            relevancy_score=relevancy_score,
            task_completion_score=task_completion_score,
            evidence_quality_score=evidence_quality_score,
            decision_quality_score=decision_quality_score,
            evidence_coverage=evidence_coverage,
            missing_evidence=missing_evidence,
            overall_score=overall_score,
            recommendations=recs
        )

    def _evaluate_faithfulness(self, justification: str, cited: str, resume: str) -> float:
        """Compare the LLM's justification and cited evidence against raw resume text."""
        if self.use_deepeval:
            try:
                tc = LLMTestCase(
                    input="Summarize candidate assessment",
                    actual_output=justification,
                    retrieval_context=[resume]
                )
                metric = FaithfulnessMetric(threshold=0.7)
                metric.measure(tc)
                return round(metric.score, 2)
            except Exception:
                pass

        # Fallback / Default Heuristic:
        # Check that cited text segments or key terms in the justification appear in the raw resume
        if not justification or not resume:
            return 0.0
        
        # Simple overlap: extract nouns/proper terms or simply match substrings of cited text
        words = [w.strip(".,;:()").lower() for w in cited.split() if len(w) > 4]
        if not words:
            return 1.0
            
        matches = sum(1 for w in words if w in resume.lower())
        return round(matches / len(words), 2)

    def _evaluate_relevancy(self, justification: str, jd: str) -> float:
        """Compare LLM Assessment justification against Job Description constraints."""
        if self.use_deepeval:
            try:
                tc = LLMTestCase(
                    input=jd,
                    actual_output=justification
                )
                metric = AnswerRelevancyMetric(threshold=0.7)
                metric.measure(tc)
                return round(metric.score, 2)
            except Exception:
                pass

        # Fallback / Default Heuristic:
        # Check that key terms in the Job Description (e.g. Python, PyTorch, experience) are discussed.
        if not justification or not jd:
            return 0.0
            
        jd_keywords = ["python", "pytorch", "scikit-learn", "ml", "machine learning", "experience"]
        active_keywords = [k for k in jd_keywords if k in jd.lower()]
        if not active_keywords:
            return 1.0
            
        matches = sum(1 for k in active_keywords if k in justification.lower())
        return round(matches / len(active_keywords), 2)

    def _evaluate_task_completion(self, decision: Dict[str, Any]) -> float:
        """Verify that every candidate was scored, decision exists, and required fields are present."""
        required_fields = ["candidate_id", "candidate_name", "status", "total_score", "reasoning"]
        present = sum(1 for f in required_fields if f in decision and decision[f] is not None)
        return round(present / len(required_fields), 2)

    def _evaluate_evidence_quality(self, task: EvaluationTask, cited_evidence: str) -> tuple[float, float, List[str]]:
        """Verify that cited evidence contains the expected keywords defined in the task."""
        if not task.required_evidence:
            return 1.0, 1.0, []

        missing_evidence = []
        for kw in task.required_evidence:
            if kw.lower() not in cited_evidence.lower():
                missing_evidence.append(kw)

        coverage = (len(task.required_evidence) - len(missing_evidence)) / len(task.required_evidence)
        score = round(coverage, 2)
        return score, round(coverage, 2), missing_evidence

    def _evaluate_decision_quality(self, task: EvaluationTask, actual_verdict: str) -> float:
        """Compares expected verdict (interview, hold, reject, blocked) vs actual verdict."""
        expected = task.expected_decision.value.lower()
        actual = actual_verdict.strip().lower()
        return 1.0 if expected == actual else 0.0


class OutputEvaluationReport(BaseModel):
    """Report consolidator for output evaluations and name-swap fairness testing."""
    results: List[OutputEvaluationResult] = Field(default_factory=list)
    fairness_results: Dict[str, FairnessResult] = Field(default_factory=dict)

    def add_result(
        self,
        res: OutputEvaluationResult,
        fairness_res: Optional[FairnessResult] = None
    ) -> None:
        """Add individual result and optional fairness check result."""
        self.results.append(res)
        if fairness_res:
            self.fairness_results[res.task_id] = fairness_res

    def generate_json(self) -> str:
        """Serialize output metrics to JSON."""
        data = {
            "runs": [r.model_dump() for r in self.results],
            "fairness": {tid: fr.model_dump() for tid, fr in self.fairness_results.items()},
            "averages": self._get_averages()
        }
        return json.dumps(data, indent=2)

    def _get_averages(self) -> Dict[str, float]:
        if not self.results:
            return {}
        total = len(self.results)
        return {
            "faithfulness": round(sum(r.faithfulness_score for r in self.results) / total, 2),
            "relevancy": round(sum(r.relevancy_score for r in self.results) / total, 2),
            "task_completion": round(sum(r.task_completion_score for r in self.results) / total, 2),
            "overall": round(sum(r.overall_score for r in self.results) / total, 2)
        }

    def generate_markdown(self) -> str:
        """Format outputs into a clean markdown document."""
        md = []
        md.append("# TechVest Recruitment Agent - Output Quality and Fairness Evaluation Report\n")
        
        averages = self._get_averages()
        if averages:
            md.append("## Aggregated Performance Metrics\n")
            md.append(f"- **Average Faithfulness Score**: `{averages['faithfulness'] * 100:.1f}%`")
            md.append(f"- **Average Relevancy Score**: `{averages['relevancy'] * 100:.1f}%`")
            md.append(f"- **Average Task Completion Score**: `{averages['task_completion'] * 100:.1f}%`")
            md.append(f"- **Average Overall Output Quality**: `{averages['overall'] * 100:.1f}%`\n")

        md.append("## Detailed Outputs Breakdowns\n")
        for r in self.results:
            status = "✅ PASS" if r.passed else "❌ FAIL"
            md.append(f"### Task: {r.task_id} ({status})\n")
            md.append(f"- **Faithfulness**: `{r.faithfulness_score * 100:.1f}%`")
            md.append(f"- **Answer Relevancy**: `{r.relevancy_score * 100:.1f}%`")
            md.append(f"- **Task Completion**: `{r.task_completion_score * 100:.1f}%`")
            md.append(f"- **Evidence Quality**: `{r.evidence_quality_score * 100:.1f}%` (Coverage: `{r.evidence_coverage * 100:.1f}%`) ")
            if r.missing_evidence:
                md.append(f"  - *Missing Evidence Keywords*: {r.missing_evidence}")
            md.append(f"- **Decision Match**: `{'Yes' if r.decision_quality_score == 1.0 else 'No'}`")
            
            # Check fairness
            if r.task_id in self.fairness_results:
                fr = self.fairness_results[r.task_id]
                f_status = "✅ PASSED" if fr.passed else "❌ FAILED"
                md.append(f"- **Fairness Name-Swap Check**: `{f_status}`")
                md.append(f"  - *Score Delta*: `{fr.score_difference:.2f}` | *Verdicts Match*: `{fr.verdicts_match}` | *Identity Neutral*: `{fr.identity_neutral}`")
                md.append(f"  - *Fairness Details*: {fr.message}")

            if r.recommendations:
                md.append("#### Recommendations")
                for rec in r.recommendations:
                    md.append(f"- {rec}")
            md.append("")
            md.append("---")
        return "\n".join(md)

    def print_console(self) -> None:
        """Print output evaluation report directly to stdout."""
        print("=" * 80)
        print("RECRUITMENT AGENT - OUTPUT QUALITY & FAIRNESS EVALUATION REPORT")
        print("=" * 80)
        averages = self._get_averages()
        if averages:
            print(f"Avg Faithfulness: {averages['faithfulness']*100:.1f}% | Avg Relevancy: {averages['relevancy']*100:.1f}%")
            print(f"Avg Task Compl. : {averages['task_completion']*100:.1f}% | Avg Overall  : {averages['overall']*100:.1f}%")
        print("-" * 80)
        for r in self.results:
            print(f"Task {r.task_id}: Passed: {r.passed} | Overall Score: {r.overall_score:.2f}")
            if r.task_id in self.fairness_results:
                fr = self.fairness_results[r.task_id]
                print(f"  Name-swap Fairness: Passed: {fr.passed} | Score Diff: {fr.score_difference:.2f}")
        print("=" * 80)
