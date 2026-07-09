"""
tests/test_tools_smoke.py
-------------------------
Smoke tests for the four TechVest LangChain tools.
No LLM calls are made — only deterministic / offline behaviour is tested.
"""

from __future__ import annotations

import uuid

import pytest

from models.candidate import CandidateProfile
from models.interview import InterviewProposal, InterviewSlot, SlotStatus
from models.rubric import Rubric
from tools import ALL_TOOLS, check_availability, parse_resume, propose_interview, score_candidate
from tools.config import PARSE_MODEL, SCORE_MODEL


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_profile() -> CandidateProfile:
    return CandidateProfile(
        candidate_id=str(uuid.uuid4()),
        name="Priya Sharma",
        email="priya@example.com",
        resume_text="Experienced ML engineer with 5 years of Python and PyTorch.",
        raw_file_path="data/resumes/priya.txt",
        skills=["Python", "PyTorch", "FastAPI", "scikit-learn"],
        years_of_experience=5.0,
        education=["B.Tech Computer Science, IIT Delhi, 2019"],
        certifications=["AWS Certified ML Specialty"],
        projects=["Built an NLP pipeline serving 10k req/day"],
    )


@pytest.fixture
def availability_slots() -> list[dict]:
    return check_availability.invoke(
        {"candidate_id": "test-001", "candidate_name": "Priya Sharma", "num_slots": 3}
    )


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

class TestToolRegistration:
    def test_all_four_tools_present(self):
        names = [t.name for t in ALL_TOOLS]
        assert names == ["parse_resume", "score_candidate", "check_availability", "propose_interview"]

    def test_tools_have_descriptions(self):
        for t in ALL_TOOLS:
            assert t.description and len(t.description) > 10

    def test_config_model_constants(self):
        assert "llama" in PARSE_MODEL.lower()
        assert "llama" in SCORE_MODEL.lower()


# ---------------------------------------------------------------------------
# check_availability
# ---------------------------------------------------------------------------

class TestCheckAvailability:
    def test_returns_requested_number_of_slots(self, availability_slots):
        assert len(availability_slots) == 3

    def test_all_slots_are_available(self, availability_slots):
        for s in availability_slots:
            slot = InterviewSlot(**s)
            assert slot.is_available()
            assert slot.status == SlotStatus.AVAILABLE

    def test_slot_duration_is_60_minutes(self, availability_slots):
        for s in availability_slots:
            slot = InterviewSlot(**s)
            assert slot.duration_minutes() == 60.0

    def test_end_time_after_start_time(self, availability_slots):
        for s in availability_slots:
            slot = InterviewSlot(**s)
            assert slot.end_time > slot.start_time

    def test_slots_are_timezone_aware(self, availability_slots):
        for s in availability_slots:
            slot = InterviewSlot(**s)
            assert slot.start_time.tzinfo is not None
            assert slot.end_time.tzinfo is not None

    def test_clamps_num_slots_to_max_5(self):
        slots = check_availability.invoke(
            {"candidate_id": "test-002", "candidate_name": "Rahul", "num_slots": 99}
        )
        assert len(slots) <= 5

    def test_returns_empty_list_on_invalid_candidate(self):
        # Should not raise — returns what it can
        slots = check_availability.invoke(
            {"candidate_id": "", "candidate_name": "", "num_slots": 1}
        )
        assert isinstance(slots, list)

    def test_slot_display_returns_string(self, availability_slots):
        slot = InterviewSlot(**availability_slots[0])
        display = slot.display()
        assert isinstance(display, str)
        assert slot.interviewer in display


# ---------------------------------------------------------------------------
# propose_interview
# ---------------------------------------------------------------------------

class TestProposeInterview:
    def test_blocked_when_not_approved(self, sample_profile, availability_slots):
        first_slot = InterviewSlot(**availability_slots[0])
        result = propose_interview.invoke({
            "profile_dict": sample_profile.model_dump(),
            "slot_dict": first_slot.model_dump(mode="json"),
            "job_title": "Senior ML Engineer",
            "approved_by": "Meera Nair",
            "human_approved": False,
        })
        assert result["is_error"] is True
        assert "human_approved" in result["error"].lower() or "blocked" in result["error"].lower()

    def test_confirmed_with_approval(self, sample_profile, availability_slots):
        first_slot = InterviewSlot(**availability_slots[0])
        result = propose_interview.invoke({
            "profile_dict": sample_profile.model_dump(),
            "slot_dict": first_slot.model_dump(mode="json"),
            "job_title": "Senior ML Engineer",
            "approved_by": "Meera Nair",
            "human_approved": True,
        })
        assert result["is_error"] is False

    def test_proposal_fields_correct(self, sample_profile, availability_slots):
        first_slot = InterviewSlot(**availability_slots[0])
        result = propose_interview.invoke({
            "profile_dict": sample_profile.model_dump(),
            "slot_dict": first_slot.model_dump(mode="json"),
            "job_title": "Senior ML Engineer",
            "approved_by": "Meera Nair",
            "human_approved": True,
        })
        proposal = InterviewProposal(
            **{k: v for k, v in result.items() if k not in ("is_error", "confirmation_text")}
        )
        assert proposal.candidate_name == "Priya Sharma"
        assert proposal.approved_by == "Meera Nair"
        assert proposal.job_title == "Senior ML Engineer"

    def test_confirmation_text_present(self, sample_profile, availability_slots):
        first_slot = InterviewSlot(**availability_slots[0])
        result = propose_interview.invoke({
            "profile_dict": sample_profile.model_dump(),
            "slot_dict": first_slot.model_dump(mode="json"),
            "job_title": "Senior ML Engineer",
            "approved_by": "Meera Nair",
            "human_approved": True,
        })
        assert "confirmation_text" in result
        assert "Priya Sharma" in result["confirmation_text"]
        assert "Meera Nair" in result["confirmation_text"]

    def test_blocked_with_empty_approver(self, sample_profile, availability_slots):
        first_slot = InterviewSlot(**availability_slots[0])
        result = propose_interview.invoke({
            "profile_dict": sample_profile.model_dump(),
            "slot_dict": first_slot.model_dump(mode="json"),
            "job_title": "Senior ML Engineer",
            "approved_by": "",
            "human_approved": True,
        })
        assert result["is_error"] is True

    def test_audit_log_written(self, sample_profile, availability_slots, tmp_path, monkeypatch):
        """Audit log file should be created after a confirmed proposal."""
        import os
        monkeypatch.chdir(tmp_path)
        first_slot = InterviewSlot(**availability_slots[0])
        result = propose_interview.invoke({
            "profile_dict": sample_profile.model_dump(),
            "slot_dict": first_slot.model_dump(mode="json"),
            "job_title": "Senior ML Engineer",
            "approved_by": "Meera Nair",
            "human_approved": True,
        })
        assert result["is_error"] is False
        log_file = tmp_path / "logs" / "interview_proposals.jsonl"
        assert log_file.exists()
        content = log_file.read_text()
        assert "Priya Sharma" in content


# ---------------------------------------------------------------------------
# Rubric
# ---------------------------------------------------------------------------

class TestRubric:
    def test_default_rubric_weights_sum_to_one(self):
        r = Rubric.default()
        total = sum(c.weight for c in r.criteria)
        assert abs(total - 1.0) < 0.01

    def test_default_rubric_has_five_criteria(self):
        r = Rubric.default()
        assert len(r.criteria) == 5

    def test_rubric_as_prompt_text(self):
        r = Rubric.default()
        text = r.as_prompt_text()
        assert "python_skills" in text
        assert "35%" in text
