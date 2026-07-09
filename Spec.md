# TechVest Recruitment Agent

## Overview

The TechVest Recruitment Agent is an autonomous AI hiring assistant built using LangGraph.

The agent evaluates multiple candidates against a Job Description (JD), automatically chooses which tools to call, scores applicants using a weighted rubric, generates a ranked shortlist, and proposes interview slots while ensuring human approval before any scheduling action.

The system demonstrates autonomous decision making, tool usage, persistent state management, guardrails, and reasoning traceability.

---

# Objectives

- Parse resumes into structured profiles
- Score candidates against the Job Description
- Rank candidates
- Recommend Interview / Hold / Reject
- Log every reasoning step
- Protect against prompt injection
- Ensure fairness
- Require human approval before interview scheduling

---

# Tech Stack

- Python
- LangGraph
- LangChain
- OpenAI GPT-4o Mini
- Pydantic
- Streamlit
- SQLite (optional)

---

# Folder Structure

RecruitmentAgent/

├── app.py

├── agent/

│ ├── graph.py

│ ├── state.py

│ ├── nodes.py

│ └── prompts.py

├── tools/

│ ├── parse_resume.py

│ ├── score_candidate.py

│ ├── availability.py

│ └── schedule.py

├── models/

│ ├── candidate.py

│ ├── rubric.py

│ └── scorecard.py

├── data/

│ ├── job_description.txt

│ ├── priya.txt

│ ├── rahul.txt

│ └── meera.txt

├── logs/

├── requirements.txt

└── README.md

---

# Features

## Resume Parsing

Input:
Resume Text

Output:
CandidateProfile

Fields

- Name
- Skills
- Education
- Experience
- Projects
- Certifications

---

## Candidate Scoring

Weighted Rubric

Python Skills (35%)

Machine Learning (25%)

Projects (20%)

Communication (10%)

Education (10%)

Output

- Criterion Score
- Evidence
- Final Score

---

## Ranking

Highest score first.

Recommendation

- Interview
- Hold
- Reject

---

## Availability Tool

Returns mock interview slots.

Example

Monday 10 AM

Tuesday 3 PM

Wednesday 11 AM

---

## Schedule Tool

Accepts

Candidate

Slot

Returns confirmation.

Requires Human Approval.

---

# LangGraph Nodes

START

↓

Parse Resume

↓

Score Candidate

↓

Decision Node

↓

Availability Tool

↓

Human Approval

↓

Schedule Interview

↓

END

---

# Agent State

Job Description

Rubric

Candidate List

Candidate Profiles

Scores

Shortlist

Trajectory Logs

Approval Status

---

# Guardrails

Human Approval

Prompt Injection Detection

Fairness Check

Step Limit

Audit Logs

---

# Streamlit UI

Dashboard

Candidate Cards

Score Breakdown

Evidence

Reasoning Trace

Approve Interview Button

Guardrail Status

---

# Future Improvements

Calendar Integration

Email Notifications

ATS Integration

Multi-Agent Architecture

MCP Support
