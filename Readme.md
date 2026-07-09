# TechVest Recruitment Agent

## Overview

TechVest Recruitment Agent is an autonomous AI hiring assistant built with LangGraph.

It evaluates candidates against a Job Description, automatically decides which tools to use, scores applicants using a weighted rubric, ranks them, and proposes interviews with mandatory human approval.

---

## Features

✔ Autonomous Agent

✔ Resume Parsing

✔ Candidate Scoring

✔ Ranking

✔ Evidence-based Justification

✔ Human Approval

✔ Prompt Injection Protection

✔ Fairness Check

✔ Audit Logs

✔ Streamlit Dashboard

---

## Workflow

Job Description

↓

Load Candidates

↓

Parse Resume

↓

Score Candidate

↓

Rank Candidates

↓

Generate Shortlist

↓

Check Availability

↓

Human Approval

↓

Schedule Interview

---

## Tech Stack

Python

LangGraph

LangChain

OpenAI GPT-4o Mini

Pydantic

Streamlit

---

## Installation

```bash
git clone <repo>

cd RecruitmentAgent
```

Create Virtual Environment

```bash
python -m venv .venv
```

Activate

Windows

```bash
.venv\Scripts\activate
```

Linux

```bash
source .venv/bin/activate
```

Install

```bash
pip install -r requirements.txt
```

Run

```bash
streamlit run app.py
```

---

## Project Structure

```text
agent/
tools/
models/
data/
logs/
```

---

## Tools

### parse_resume()

Extracts structured candidate profile.

### score_candidate()

Scores against rubric.

### check_availability()

Returns interview slots.

### propose_interview()

Schedules interview after approval.

---

## Guardrails

Human Approval

Prompt Injection Detection

Fairness Validation

Iteration Limit

Audit Logging

---

## Output

Ranked Shortlist

Evidence

Scores

Reasoning Trace

Interview Recommendation

---

## Future Scope

Real Calendar Integration

ATS Integration

Email Automation

CrewAI Version

MCP Tools
