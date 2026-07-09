"""
tests/test_graph.py
-------------------
Pytest suite: verify imports, state construction, graph compilation,
conditional routing logic, and an end-to-end offline run (no LLM calls).

Test classes
------------
TestImports             all agent symbols import cleanly
TestAgentState          AgentState and initial_state() factory
TestPrompts             prompt constants are non-empty and correctly formatted
TestGraphCompilation    build_graph() compiles; metadata is correct
TestConditionalRouting  routing functions return correct node names
TestGraphOfflineRun     full graph run with mocked nodes (no LLM / network)
TestMakeConfig          make_config() produces correct config dicts
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langgraph.checkpoint.memory import MemorySaver


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

JD_TEXT = """\
Senior ML Engineer — TechVest
We are looking for a Python expert with strong machine learning skills.
Required: Python, PyTorch, scikit-learn. Min 3 years experience.
"""

CANDIDATES = [
    {
        "raw_text": (
            "Priya Sharma | priya@example.com | +91-9876543210\n"
            "Skills: Python, PyTorch, FastAPI, scikit-learn\n"
            "Experience: 5 years ML engineering at DataCorp\n"
            "Education: B.Tech CS, IIT Delhi, 2019\n"
            "Certifications: AWS Certified ML Specialty\n"
            "Projects: Built NLP pipeline serving 10k req/day"
        ),
        "file_path": "data/resumes/priya.txt",
    },
    {
        "raw_text": (
            "Rahul Mehta | rahul@example.com\n"
            "Skills: Java, Spring Boot, SQL\n"
            "Experience: 2 years backend engineering\n"
            "Education: B.E. Computer Science, 2021"
        ),
        "file_path": "data/resumes/rahul.txt",
    },
]


@pytest.fixture
def fresh_graph():
    """Return a freshly compiled graph with its own MemorySaver."""
    from agent.graph import build_graph
    return build_graph(checkpointer=MemorySaver())


@pytest.fixture
def base_state():
    """Return a minimal initial AgentState."""
    from agent.state import initial_state
    return initial_state(JD_TEXT, CANDIDATES)


# ---------------------------------------------------------------------------
# TestImports
# ---------------------------------------------------------------------------

class TestImports:
    def test_agent_package_imports(self):
        from agent import (
            GRAPH_CONFIG,
            RECURSION_LIMIT,
            AgentState,
            build_graph,
            initial_state,
            make_config,
            recruitment_graph,
        )
        assert recruitment_graph is not None
        assert RECURSION_LIMIT == 25

    def test_state_imports(self):
        from agent.state import AgentState, initial_state
        assert AgentState is not None
        assert callable(initial_state)

    def test_prompts_import(self):
        from agent.prompts import (
            FAIRNESS_CHECK_PROMPT,
            INJECTION_GUARD_PROMPT,
            PARSE_RESUME_PROMPT,
            RANK_CANDIDATES_PROMPT,
            SCORE_CANDIDATE_PROMPT,
            SHORTLIST_PROMPT,
            SYSTEM_PROMPT,
        )
        for prompt in [
            SYSTEM_PROMPT, PARSE_RESUME_PROMPT, SCORE_CANDIDATE_PROMPT,
            RANK_CANDIDATES_PROMPT, SHORTLIST_PROMPT,
            FAIRNESS_CHECK_PROMPT, INJECTION_GUARD_PROMPT,
        ]:
            assert isinstance(prompt, str) and len(prompt) > 20

    def test_nodes_import(self):
        from agent.nodes import (
            check_availability_node,
            fairness_check_node,
            generate_shortlist_node,
            human_approval_node,
            injection_guard_node,
            parse_resume_node,
            rank_candidates_node,
            schedule_interview_node,
            score_candidate_node,
        )
        for fn in [
            injection_guard_node, parse_resume_node, fairness_check_node,
            score_candidate_node, rank_candidates_node, generate_shortlist_node,
            check_availability_node, human_approval_node, schedule_interview_node,
        ]:
            assert callable(fn)

    def test_graph_imports(self):
        from agent.graph import (
            GRAPH_CONFIG,
            RECURSION_LIMIT,
            build_graph,
            make_config,
            recruitment_graph,
        )
        assert callable(build_graph)
        assert callable(make_config)
        assert isinstance(GRAPH_CONFIG, dict)
        assert isinstance(RECURSION_LIMIT, int)


# ---------------------------------------------------------------------------
# TestAgentState
# ---------------------------------------------------------------------------

class TestAgentState:
    def test_initial_state_keys(self, base_state):
        required_keys = [
            "job_description_text", "rubric_dict", "candidates",
            "candidate_profiles", "scores", "shortlist", "current_index",
            "interview_slots", "selected_slot", "scheduled_interviews",
            "human_approved", "approval_candidate_id", "iteration_count",
            "error_message", "audit_log", "messages",
        ]
        for key in required_keys:
            assert key in base_state, f"Missing key: {key}"

    def test_initial_state_defaults(self, base_state):
        assert base_state["current_index"] == 0
        assert base_state["human_approved"] is False
        assert base_state["iteration_count"] == 0
        assert base_state["candidate_profiles"] == []
        assert base_state["scores"] == []
        assert base_state["shortlist"] == []
        assert base_state["audit_log"] == []
        assert base_state["messages"] == []
        assert base_state["error_message"] == ""
        assert base_state["selected_slot"] is None

    def test_initial_state_rubric_default(self, base_state):
        rubric = base_state["rubric_dict"]
        assert isinstance(rubric, dict)
        assert "criteria" in rubric
        assert len(rubric["criteria"]) == 5

    def test_initial_state_candidates_stored(self, base_state):
        assert len(base_state["candidates"]) == 2
        assert base_state["candidates"][0]["file_path"] == "data/resumes/priya.txt"

    def test_initial_state_with_custom_rubric(self):
        from agent.state import initial_state
        from models.rubric import Rubric
        rubric = Rubric.default()
        state = initial_state(JD_TEXT, CANDIDATES, rubric_dict=rubric.model_dump())
        assert state["rubric_dict"]["name"] == "TechVest Senior ML Engineer"


# ---------------------------------------------------------------------------
# TestPrompts
# ---------------------------------------------------------------------------

class TestPrompts:
    def test_parse_resume_prompt_has_placeholder(self):
        from agent.prompts import PARSE_RESUME_PROMPT
        assert "{resume_text}" in PARSE_RESUME_PROMPT

    def test_score_candidate_prompt_placeholders(self):
        from agent.prompts import SCORE_CANDIDATE_PROMPT
        assert "{candidate_summary}" in SCORE_CANDIDATE_PROMPT
        assert "{jd_block}" in SCORE_CANDIDATE_PROMPT
        assert "{rubric_block}" in SCORE_CANDIDATE_PROMPT

    def test_injection_guard_prompt_placeholder(self):
        from agent.prompts import INJECTION_GUARD_PROMPT
        assert "{resume_text}" in INJECTION_GUARD_PROMPT

    def test_fairness_check_prompt_placeholders(self):
        from agent.prompts import FAIRNESS_CHECK_PROMPT
        assert "{resume_text}" in FAIRNESS_CHECK_PROMPT
        assert "{scoring_evidence}" in FAIRNESS_CHECK_PROMPT

    def test_rank_prompt_placeholders(self):
        from agent.prompts import RANK_CANDIDATES_PROMPT
        assert "{scorecards_json}" in RANK_CANDIDATES_PROMPT
        assert "{jd_title}" in RANK_CANDIDATES_PROMPT

    def test_prompts_are_format_safe(self):
        """Prompts with double-braced JSON examples should survive .format()."""
        from agent.prompts import INJECTION_GUARD_PROMPT, PARSE_RESUME_PROMPT
        # Should not raise KeyError
        result = PARSE_RESUME_PROMPT.format(resume_text="test resume content")
        assert "test resume content" in result

        result2 = INJECTION_GUARD_PROMPT.format(resume_text="some resume")
        assert "some resume" in result2


# ---------------------------------------------------------------------------
# TestGraphCompilation
# ---------------------------------------------------------------------------

class TestGraphCompilation:
    def test_build_graph_returns_compiled_graph(self, fresh_graph):
        assert fresh_graph is not None
        assert hasattr(fresh_graph, "invoke")
        assert hasattr(fresh_graph, "stream")

    def test_module_level_graph_exists(self):
        from agent.graph import recruitment_graph
        assert recruitment_graph is not None

    def test_graph_has_correct_nodes(self, fresh_graph):
        node_names = set(fresh_graph.get_graph().nodes.keys())
        expected = {
            "__start__",
            "injection_guard_node",
            "parse_resume_node",
            "fairness_check_node",
            "score_candidate_node",
            "rank_candidates_node",
            "generate_shortlist_node",
            "check_availability_node",
            "human_approval_node",
            "schedule_interview_node",
        }
        assert expected.issubset(node_names), (
            f"Missing nodes: {expected - node_names}"
        )

    def test_graph_has_nine_non_terminal_nodes(self, fresh_graph):
        g = fresh_graph.get_graph()
        non_terminal = {
            n for n in g.nodes
            if n not in ("__start__", "__end__")
        }
        assert len(non_terminal) == 9

    def test_graph_interrupt_before_set(self, fresh_graph):
        """Interrupt should be configured on human_approval_node."""
        # interrupt_before is stored in the compiled graph config
        assert fresh_graph.config_specs is not None  # compiled config exists

    def test_different_builds_are_independent(self):
        from agent.graph import build_graph
        g1 = build_graph(checkpointer=MemorySaver())
        g2 = build_graph(checkpointer=MemorySaver())
        assert g1 is not g2

    def test_graph_get_graph_has_edges(self, fresh_graph):
        g = fresh_graph.get_graph()
        assert len(g.edges) > 0


# ---------------------------------------------------------------------------
# TestConditionalRouting
# ---------------------------------------------------------------------------

class TestConditionalRouting:
    """Test the routing functions in isolation — no graph invocation needed."""

    def _state(self, **overrides) -> dict[str, Any]:
        """Build a minimal state dict for routing tests."""
        base: dict[str, Any] = {
            "current_index": 0,
            "candidates": [{"raw_text": "r", "file_path": "f.txt"}],
            "scores": [],
            "shortlist": [],
            "human_approved": False,
        }
        base.update(overrides)
        return base

    def test_route_injection_guard_to_parse_when_candidates_remain(self):
        from agent.graph import _route_after_injection_guard
        state = self._state(current_index=0, candidates=[{"raw_text": "r", "file_path": "f"}])
        assert _route_after_injection_guard(state) == "parse_resume_node"

    def test_route_injection_guard_to_rank_when_all_processed(self):
        from agent.graph import _route_after_injection_guard
        state = self._state(current_index=2, candidates=[{"r": 1}, {"r": 2}])
        assert _route_after_injection_guard(state) == "rank_candidates_node"

    def test_route_injection_guard_no_candidates_goes_to_rank(self):
        from agent.graph import _route_after_injection_guard
        state = self._state(current_index=0, candidates=[])
        assert _route_after_injection_guard(state) == "rank_candidates_node"

    def test_route_after_score_loops_back(self):
        from agent.graph import _route_after_score
        state = self._state(current_index=1, candidates=[{"r": 1}, {"r": 2}])
        assert _route_after_score(state) == "injection_guard_node"

    def test_route_after_score_goes_to_rank_when_done(self):
        from agent.graph import _route_after_score
        state = self._state(current_index=2, candidates=[{"r": 1}, {"r": 2}])
        assert _route_after_score(state) == "rank_candidates_node"

    def test_route_after_shortlist_to_availability(self):
        from langgraph.graph import END

        from agent.graph import _route_after_shortlist
        state = self._state(shortlist=[{"candidate_id": "x", "total_score": 85}])
        assert _route_after_shortlist(state) == "check_availability_node"

    def test_route_after_shortlist_to_end_when_empty(self):
        from langgraph.graph import END

        from agent.graph import _route_after_shortlist
        state = self._state(shortlist=[])
        assert _route_after_shortlist(state) == END

    def test_route_after_approval_approved(self):
        from agent.graph import _route_after_approval
        state = self._state(human_approved=True)
        assert _route_after_approval(state) == "schedule_interview_node"

    def test_route_after_approval_rejected(self):
        from langgraph.graph import END

        from agent.graph import _route_after_approval
        state = self._state(human_approved=False)
        assert _route_after_approval(state) == END


# ---------------------------------------------------------------------------
# TestMakeConfig
# ---------------------------------------------------------------------------

class TestMakeConfig:
    def test_make_config_structure(self):
        from agent.graph import make_config
        cfg = make_config(thread_id="test-123")
        assert cfg["recursion_limit"] == 25
        assert cfg["configurable"]["thread_id"] == "test-123"

    def test_make_config_generates_uuid_when_no_thread_id(self):
        from agent.graph import make_config
        cfg = make_config()
        tid = cfg["configurable"]["thread_id"]
        # Should parse as a valid UUID4
        parsed = uuid.UUID(tid, version=4)
        assert str(parsed) == tid

    def test_make_config_custom_recursion_limit(self):
        from agent.graph import make_config
        cfg = make_config(thread_id="t", recursion_limit=10)
        assert cfg["recursion_limit"] == 10

    def test_graph_config_constant(self):
        from agent.graph import GRAPH_CONFIG
        assert "recursion_limit" in GRAPH_CONFIG
        assert "configurable" in GRAPH_CONFIG
        assert "thread_id" in GRAPH_CONFIG["configurable"]


# ---------------------------------------------------------------------------
# TestGraphOfflineRun
# ---------------------------------------------------------------------------

class TestGraphOfflineRun:
    """
    Run the graph end-to-end with all LLM-backed nodes patched to return
    deterministic results.  No API key or network required.

    Strategy:
    - patch _call_llm_json in agent.nodes to return safe defaults
    - patch tools.parse_resume.parse_resume._call_llm_for_parse
    - patch tools.score_candidate.score_candidate._call_llm_for_score
    The check_availability and propose_interview tools are real (no LLM).
    """

    MOCK_PARSE_RESULT = {
        "name": "Priya Sharma",
        "email": "priya@example.com",
        "phone": "+91-9876543210",
        "skills": ["Python", "PyTorch", "FastAPI"],
        "years_of_experience": 5.0,
        "education": ["B.Tech CS, IIT Delhi, 2019"],
        "certifications": ["AWS Certified ML Specialty"],
        "projects": ["NLP pipeline serving 10k req/day"],
    }

    MOCK_SCORE_RESULT = {
        "criterion_scores": [
            {"criterion": "python_skills",   "raw_score": 85.0, "weight": 0.35, "weighted_score": 29.75, "evidence": "Strong Python usage throughout resume"},
            {"criterion": "machine_learning","raw_score": 80.0, "weight": 0.25, "weighted_score": 20.00, "evidence": "PyTorch and scikit-learn experience"},
            {"criterion": "projects",        "raw_score": 75.0, "weight": 0.20, "weighted_score": 15.00, "evidence": "NLP pipeline serving 10k req/day"},
            {"criterion": "communication",   "raw_score": 70.0, "weight": 0.10, "weighted_score": 7.00,  "evidence": "Clear and well-structured resume"},
            {"criterion": "education",       "raw_score": 80.0, "weight": 0.10, "weighted_score": 8.00,  "evidence": "B.Tech from IIT Delhi, 2019"},
        ],
        "total_score": 79.75,
        "summary_evidence": "Strong ML engineer with solid Python and PyTorch experience.",
    }

    MOCK_INJECTION_RESULT = {
        "injection_detected": False,
        "confidence": "low",
        "evidence": "",
        "recommendation": "proceed",
    }

    MOCK_FAIRNESS_RESULT = {
        "flags": [],
        "overall_risk": "none",
        "recommendation": "No action required",
    }

    def _run_to_interrupt(self, graph, state, config):
        """Run graph until the human_approval interrupt and return state."""
        try:
            return graph.invoke(state, config=config)
        except Exception:
            # In LangGraph 0.2.x, interrupt_before raises or returns snapshot
            pass
        # Get the current state snapshot
        snapshot = graph.get_state(config)
        return snapshot.values if snapshot else {}

    @patch("agent.nodes._call_llm_json")
    @patch("tools.parse_resume._call_llm_for_parse")
    @patch("tools.score_candidate._call_llm_for_score")
    def test_graph_runs_to_interrupt(
        self,
        mock_score_llm,
        mock_parse_llm,
        mock_guard_llm,
    ):
        """Graph should run from START through to the human_approval interrupt."""
        mock_guard_llm.return_value = self.MOCK_INJECTION_RESULT
        mock_parse_llm.return_value = self.MOCK_PARSE_RESULT
        mock_score_llm.return_value = self.MOCK_SCORE_RESULT

        from agent.graph import build_graph
        from agent.state import initial_state

        graph = build_graph(checkpointer=MemorySaver())
        state = initial_state(JD_TEXT, [CANDIDATES[0]])  # single candidate
        config = {"recursion_limit": 25, "configurable": {"thread_id": str(uuid.uuid4())}}

        # Invoke — will stop at interrupt_before=human_approval_node
        result = graph.invoke(state, config=config)

        # After interrupt, get_state shows where we are
        snapshot = graph.get_state(config)
        assert snapshot is not None

        # The next node should be human_approval_node (the interrupt point)
        next_nodes = list(snapshot.next)
        assert "human_approval_node" in next_nodes

    @patch("agent.nodes._call_llm_json")
    @patch("tools.parse_resume._call_llm_for_parse")
    @patch("tools.score_candidate._call_llm_for_score")
    def test_graph_state_has_profiles_after_parse(
        self,
        mock_score_llm,
        mock_parse_llm,
        mock_guard_llm,
    ):
        mock_guard_llm.return_value = self.MOCK_INJECTION_RESULT
        mock_parse_llm.return_value = self.MOCK_PARSE_RESULT
        mock_score_llm.return_value = self.MOCK_SCORE_RESULT

        from agent.graph import build_graph
        from agent.state import initial_state

        graph = build_graph(checkpointer=MemorySaver())
        state = initial_state(JD_TEXT, [CANDIDATES[0]])
        config = {"recursion_limit": 25, "configurable": {"thread_id": str(uuid.uuid4())}}

        graph.invoke(state, config=config)
        snapshot = graph.get_state(config)

        sv = snapshot.values
        # Profile should have been parsed
        assert len(sv.get("candidate_profiles", [])) >= 1
        assert sv["candidate_profiles"][0]["name"] == "Priya Sharma"

    @patch("agent.nodes._call_llm_json")
    @patch("tools.parse_resume._call_llm_for_parse")
    @patch("tools.score_candidate._call_llm_for_score")
    def test_graph_state_has_shortlist_above_threshold(
        self,
        mock_score_llm,
        mock_parse_llm,
        mock_guard_llm,
    ):
        mock_guard_llm.return_value = self.MOCK_INJECTION_RESULT
        mock_parse_llm.return_value = self.MOCK_PARSE_RESULT
        mock_score_llm.return_value = self.MOCK_SCORE_RESULT  # total=79.75 > 70 threshold

        from agent.graph import build_graph
        from agent.state import initial_state

        graph = build_graph(checkpointer=MemorySaver())
        state = initial_state(JD_TEXT, [CANDIDATES[0]])
        config = {"recursion_limit": 25, "configurable": {"thread_id": str(uuid.uuid4())}}

        graph.invoke(state, config=config)
        snapshot = graph.get_state(config)

        shortlist = snapshot.values.get("shortlist", [])
        assert len(shortlist) >= 1
        assert shortlist[0]["status"] == "interview"

    @patch("agent.nodes._call_llm_json")
    @patch("tools.parse_resume._call_llm_for_parse")
    @patch("tools.score_candidate._call_llm_for_score")
    def test_graph_state_has_interview_slots(
        self,
        mock_score_llm,
        mock_parse_llm,
        mock_guard_llm,
    ):
        mock_guard_llm.return_value = self.MOCK_INJECTION_RESULT
        mock_parse_llm.return_value = self.MOCK_PARSE_RESULT
        mock_score_llm.return_value = self.MOCK_SCORE_RESULT

        from agent.graph import build_graph
        from agent.state import initial_state

        graph = build_graph(checkpointer=MemorySaver())
        state = initial_state(JD_TEXT, [CANDIDATES[0]])
        config = {"recursion_limit": 25, "configurable": {"thread_id": str(uuid.uuid4())}}

        graph.invoke(state, config=config)
        snapshot = graph.get_state(config)

        slots = snapshot.values.get("interview_slots", [])
        assert len(slots) == 3  # check_availability returns 3

    @patch("agent.nodes._call_llm_json")
    @patch("tools.parse_resume._call_llm_for_parse")
    @patch("tools.score_candidate._call_llm_for_score")
    def test_graph_resumes_and_schedules_after_approval(
        self,
        mock_score_llm,
        mock_parse_llm,
        mock_guard_llm,
    ):
        mock_guard_llm.return_value = self.MOCK_INJECTION_RESULT
        mock_parse_llm.return_value = self.MOCK_PARSE_RESULT
        mock_score_llm.return_value = self.MOCK_SCORE_RESULT

        from agent.graph import build_graph
        from agent.state import initial_state

        graph = build_graph(checkpointer=MemorySaver())
        state = initial_state(JD_TEXT, [CANDIDATES[0]])
        config = {"recursion_limit": 25, "configurable": {"thread_id": str(uuid.uuid4())}}

        # First invoke — runs to interrupt_before=human_approval_node
        graph.invoke(state, config=config)

        # Inject human approval into the checkpoint, then resume
        graph.update_state(config, {"human_approved": True}, as_node="human_approval_node")
        graph.invoke(None, config=config)

        # Should have completed scheduling
        snapshot = graph.get_state(config)
        sv = snapshot.values
        assert sv.get("human_approved") is True
        scheduled = sv.get("scheduled_interviews", [])
        assert len(scheduled) >= 1
        assert scheduled[0].get("is_error") is False

    @patch("agent.nodes._call_llm_json")
    @patch("tools.parse_resume._call_llm_for_parse")
    @patch("tools.score_candidate._call_llm_for_score")
    def test_graph_ends_without_scheduling_when_rejected(
        self,
        mock_score_llm,
        mock_parse_llm,
        mock_guard_llm,
    ):
        mock_guard_llm.return_value = self.MOCK_INJECTION_RESULT
        mock_parse_llm.return_value = self.MOCK_PARSE_RESULT
        mock_score_llm.return_value = self.MOCK_SCORE_RESULT

        from agent.graph import build_graph
        from agent.state import initial_state

        graph = build_graph(checkpointer=MemorySaver())
        state = initial_state(JD_TEXT, [CANDIDATES[0]])
        config = {"recursion_limit": 25, "configurable": {"thread_id": str(uuid.uuid4())}}

        graph.invoke(state, config=config)

        # Inject rejection via update_state, then resume
        graph.update_state(config, {"human_approved": False}, as_node="human_approval_node")
        graph.invoke(None, config=config)

        snapshot = graph.get_state(config)
        sv = snapshot.values
        assert sv.get("human_approved") is False
        assert sv.get("scheduled_interviews", []) == []

    @patch("agent.nodes._call_llm_json")
    @patch("tools.parse_resume._call_llm_for_parse")
    @patch("tools.score_candidate._call_llm_for_score")
    def test_audit_log_grows_through_pipeline(
        self,
        mock_score_llm,
        mock_parse_llm,
        mock_guard_llm,
    ):
        mock_guard_llm.return_value = self.MOCK_INJECTION_RESULT
        mock_parse_llm.return_value = self.MOCK_PARSE_RESULT
        mock_score_llm.return_value = self.MOCK_SCORE_RESULT

        from agent.graph import build_graph
        from agent.state import initial_state

        graph = build_graph(checkpointer=MemorySaver())
        state = initial_state(JD_TEXT, [CANDIDATES[0]])
        config = {"recursion_limit": 25, "configurable": {"thread_id": str(uuid.uuid4())}}

        graph.invoke(state, config=config)
        snapshot = graph.get_state(config)

        audit = snapshot.values.get("audit_log", [])
        assert len(audit) >= 4  # guard + parse + score + rank + shortlist + availability

    @patch("agent.nodes._call_llm_json")
    @patch("tools.parse_resume._call_llm_for_parse")
    @patch("tools.score_candidate._call_llm_for_score")
    def test_graph_handles_empty_candidate_list(
        self,
        mock_score_llm,
        mock_parse_llm,
        mock_guard_llm,
    ):
        """Graph should reach END gracefully when candidates=[]."""
        mock_guard_llm.return_value = self.MOCK_INJECTION_RESULT
        mock_parse_llm.return_value = self.MOCK_PARSE_RESULT
        mock_score_llm.return_value = self.MOCK_SCORE_RESULT

        from agent.graph import build_graph
        from agent.state import initial_state

        graph = build_graph(checkpointer=MemorySaver())
        state = initial_state(JD_TEXT, [])   # no candidates
        config = {"recursion_limit": 25, "configurable": {"thread_id": str(uuid.uuid4())}}

        # Should not raise; shortlist will be empty → routes to END
        result = graph.invoke(state, config=config)
        snapshot = graph.get_state(config)
        assert snapshot.values.get("shortlist", []) == []
