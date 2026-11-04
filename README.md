# Meridian — AI Academic Advisor

> AI-powered academic advisor for Penn State students — multi-agent LLM orchestration,
> transcript-aware RAG, and deterministic graduation planning, built with React, Flask, and LangChain.

🌐 **Live Demo:** https://3.145.213.72/

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Node](https://img.shields.io/badge/Node.js-18+-green?logo=node.js&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![LangChain](https://img.shields.io/badge/LangChain-0.2-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Overview

Meridian supports:

- transcript upload and session-scoped retrieval
- semester-by-semester graduation planning
- next-semester course recommendations
- Penn State policy and deadline lookup
- course, major, and prerequisite question answering
- source-backed web research for current or unexpected factual questions
- faculty-facing major roadmap graph generation

---

## Why This Project Is Interesting

This codebase is not a basic chatbot wrapper. It demonstrates:

- multi-agent orchestration with explicit planning, execution, and validation stages
- hybrid AI systems design combining LLM reasoning with deterministic academic logic
- retrieval-augmented generation over transcripts, courses, majors, and policy data
- tool-aware execution with controlled web research and source validation
- operational observability through per-request debug traces and regression tests

---

## System Architecture

```text
Browser
  -> React Client
      -> Flask AI API
          -> Orchestrator Agent
          -> Executor Agent
          -> Finalizer Agent
          -> Shared Research Pipeline
          -> Local RAG Stores
          -> Deterministic Graduation Planner
      -> Node / Express Auth API
          -> MySQL
```

### Main Services

#### 1. React Frontend

Location: `IAAS/client`

Responsibilities:

- renders the student advising chat experience
- manages per-tab session state
- uploads transcript PDFs
- sends conversation history with each request
- surfaces faculty roadmap generation views

#### 2. Node Authentication Service

Location: `IAAS/client/server`

Responsibilities:

- register and login endpoints
- MySQL user storage
- lightweight auth support for the UI

#### 3. Flask AI Backend

Location: `IAAS/LLMAcadamic-Advisor`

Responsibilities:

- transcript ingestion and retrieval setup
- multi-agent reasoning pipeline
- deterministic graduation planning
- policy and deadline lookup
- web-backed research for fresh facts
- debug trace generation

---

## AI Multi-Agentic Workflow

The active AI workflow is a three-stage agent pipeline with a shared research system behind it.

### Agent 1: Orchestrator

Primary responsibility:

- understand the user message in context
- correct spelling and grammar when useful
- resolve follow-up questions against prior conversation
- decide what type of answer is needed
- choose whether tools are required
- produce a structured execution plan

Inputs:

- latest user message
- conversation history
- current server-local date/time
- active task context from prior turns

Output:

- `OrchestrationPlan`

What it decides:

- domain: transcript, course, major, policy, or general academic chat
- deliverable: direct answer, audit, single-semester plan, multi-semester plan, full graduation plan, or lookup result
- plan scope and semester horizon
- allowed tools
- executor path
- success criteria and failure constraints

### Agent 2: Executor

Primary responsibility:

- execute the orchestrator plan exactly
- call tools only when needed
- combine deterministic outputs and LLM reasoning into a grounded draft

Core executor paths:

- `conversation_profile_recall`
- `conversation_plan_slice`
- `policy_lookup`
- `research_lookup`
- transcript-, course-, major-, and planning-related handlers

Tools available to the executor:

- conversation history and active task context
- transcript retriever
- course retriever
- majors retriever
- policy retriever
- deterministic graduation planner
- prerequisite graph utilities
- shared web research pipeline

### Agent 3: Finalizer

Primary responsibility:

- rewrite the executor draft into a polished user-facing response
- keep the answer friendly, professional, and appropriately detailed
- validate that the draft matches the original plan
- check citations, freshness, answer scope, and date requirements
- trigger one controlled retry if the draft fails validation

Validation examples:

- a single-semester request must not return a full graduation plan
- a current deadline question should not return without a concrete or clearly exhausted source-backed answer
- a no-tools conversation-only request must not silently call tools

### Shared Research Pipeline

The web research layer is shared infrastructure rather than a top-level agent, but it is a critical part of the system.

Responsibilities:

- resolve research context from the question and conversation
- infer campus, term, year, freshness needs, and exact-date requirements
- choose a search strategy
- retrieve evidence from official sources first
- rank, fetch, and extract candidate facts
- return a source-backed research result to the executor

Provider ladder:

1. official Penn State seed URLs
2. official domain crawl
3. DuckDuckGo discovery
4. Tavily fallback when configured
5. browser fallback for difficult pages
6. deterministic claim extraction
7. LLM synthesis over verified evidence

### Inner Agent Communication

The internal handoff between stages is explicit.

1. `Orchestrator -> Executor`
   Sends a structured `OrchestrationPlan` with the normalized question, requested scope, allowed tools, executor path, and success criteria.

2. `Executor -> Finalizer`
   Sends an `ExecutorResult` containing the draft response, tools used, execution metadata, and any search resolution artifacts.

3. `Finalizer -> Executor` (optional)
   Sends a single retry reason if validation fails, for example scope mismatch, missing concrete date, or missing source.

4. `System -> Debug Trace`
   Records orchestrator, executor, finalizer, tool usage, research rounds, quality checks, and repair attempts.

---

## Retrieval-Augmented Generation (RAG)

Meridian uses RAG to ground responses in academic data instead of relying on free-form model memory alone.

### What is retrieved

The system retrieves from four main knowledge sources:

- uploaded transcript PDFs for transcript-aware advising
- `processed_psu_courses.csv` for course information and prerequisites
- `all_abington_majors_combined.csv` for major requirements
- `PennState_Combined_Advising_and_Registrar_VERBATIM.pdf` for policy and registrar guidance

### How the RAG pipeline works

1. A transcript is uploaded through `/llm/setup-environment`.
2. The backend hashes the file and creates a session-scoped transcript collection.
3. The transcript is loaded, chunked, lightly redacted, embedded, and stored in Chroma.
4. Static academic sources such as courses, majors, and policy documents are also indexed in local vector stores.
5. At response time, the executor retrieves the most relevant chunks for the current question.
6. The retrieved context is passed into the response-building stage so the answer is grounded in local evidence.

### Why RAG matters here

RAG improves:

- accuracy for transcript-specific advising
- consistency for course and major questions
- grounded policy responses from local Penn State material
- explainability, because answers are tied to retrievable academic context

### RAG and the rest of the system

RAG is one layer of the architecture, not the whole system.

- for static academic knowledge, the system prefers local retrieval first
- for planning, retrieval is combined with deterministic graduation logic
- for current or time-sensitive facts, the system escalates from local RAG to the shared web research pipeline

This design keeps the system both grounded and adaptable: local retrieval handles stable university knowledge, while web research handles freshness-sensitive questions that embeddings alone cannot answer reliably.

---

## Engineering Highlights

- **Hybrid reasoning model**: LLMs handle planning and synthesis; deterministic code handles transcript parsing, prerequisite ordering, and schedule generation.
- **Tool-aware orchestration**: tool usage is explicit, constrained, and validated after execution.
- **Search for both expected and unexpected questions**: the system can route known policy questions and open-ended factual questions through the same research pipeline.
- **Session continuity**: follow-up questions like `current semester` or `what about Abington` can continue the active reasoning thread.
- **Performance-conscious web research**: the search pipeline uses staged query expansion, bounded concurrency, short-lived caches, and early exits when an authoritative answer is found.
- **Observability**: each response can return a detailed internal trace for debugging and evaluation.

---

## Key Technical Components

### Frontend

- React 18
- React Router
- MUI (Material UI)
- Axios

### Backend

- Flask + Flask-CORS
- Express (Node.js auth service)
- MySQL

### AI / Retrieval

- Google Gemini (`gemini-3.5-flash-lite`) via OpenAI-compatible API
- LangChain 0.2
- ChromaDB
- PDF ingestion with `pypdf`
- Local CSV- and PDF-backed retrieval corpora

### Research / Verification

- `duckduckgo-search`
- `tavily-python` (optional, when configured)
- `requests`
- Playwright fallback for JavaScript-heavy pages (optional)

---

## Repository Layout

```text
.
├── README.md
├── requirements.txt                        # Core Python deps (Flask, LangChain, Chroma…)
├── IAAS/
│   ├── client/                             # React frontend + Node auth server
│   │   ├── src/
│   │   ├── server/                         # Express auth API
│   │   └── package.json
│   ├── LLMAcadamic-Advisor/                # Flask AI backend
│   │   ├── api/
│   │   │   └── api.py
│   │   ├── LLM/
│   │   │   ├── main_copy.py
│   │   │   ├── orchestration_pipeline.py
│   │   │   ├── web_search_pipeline.py
│   │   │   ├── graduationFlowchart.py
│   │   │   ├── graphExperiment.py
│   │   │   ├── majorsGraph.py
│   │   │   ├── processed_psu_courses.csv
│   │   │   └── PennState_Combined_Advising_and_Registrar_VERBATIM.pdf
│   │   ├── requirements.txt                # Extra deps (duckduckgo, tavily)
│   │   └── .env.example
│   ├── test_orchestration_pipeline.py
│   └── test_web_search_pipeline.py
```

---

## API Endpoints

### Flask AI API (port 5002)

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/llm/setup-environment` | Upload a transcript and initialize retrieval |
| `POST` | `/llm/generate-response` | Generate the main advising response |
| `POST` | `/llm/clear-conversation` | Clear in-memory conversation and active research context |
| `POST` | `/llm/generate-major-graph` | Generate a major roadmap graph |

### Node Auth API (port 5001)

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/register` | Register a new user |
| `POST` | `/api/login` | Authenticate a user and return a JWT |

---

## Local Development

### Prerequisites

- Python 3.10+
- Node.js 18+ and npm
- MySQL (only required if you want auth endpoints — set `MYSQL_DISABLE=true` to skip)
- Google Gemini API key ([get one here](https://aistudio.google.com/app/apikey))

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd AI_ACADEMIC_ADVISOR
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt -r IAAS/LLMAcadamic-Advisor/requirements.txt
```

Optional — browser fallback support for JavaScript-heavy pages:

```bash
playwright install chromium
```

### 3. Install frontend and auth dependencies

```bash
cd IAAS/client && npm install
cd server && npm install && cd ../..
```

### 4. Configure environment variables

Copy the example env file and fill in your keys:

```bash
cp IAAS/LLMAcadamic-Advisor/.env.example IAAS/LLMAcadamic-Advisor/.env
```

Minimum required variables in `.env`:

```bash
# LLM provider
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-gemini-api-key-here
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
GEMINI_MODEL=gemini-3.5-flash-lite
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
```

Optional variables:

```bash
MERIDIAN_DEBUG_TRACE=true          # enables per-request internal trace output
MYSQL_DISABLE=true              # run without MySQL (skips auth layer)
LANGCHAIN_API_KEY=...           # LangSmith tracing (optional)
LANGCHAIN_PROJECT=...           # LangSmith project name (optional)
TAVILY_API_KEY=...              # Tavily web search fallback (optional)
```

> **Warning:** Never commit your `.env` file. It is listed in `.gitignore`.

### 5. Start the application

```bash
cd IAAS/client
npm start
```

This concurrently starts all three services:

| Service | URL |
|---------|-----|
| React frontend | http://localhost:3000 |
| Node auth service | http://localhost:5001 |
| Flask AI backend | http://127.0.0.1:5002 |

---

## Testing

### AI pipeline unit tests

```bash
source .venv/bin/activate
python3 -m unittest IAAS/test_orchestration_pipeline.py IAAS/test_web_search_pipeline.py
```

### Manual conversation smoke test

> The Flask service must already be running before this test.

```bash
python IAAS/test_ai_conversation.py
```

---

## Current Status

This is a fully functional engineering prototype and AI systems portfolio project.

**Known limitations:**

| Area | Status |
|------|--------|
| Auth layer | Lightweight JWT — not hardened for production |
| State management | Some state is process-local; would need Redis or a DB for multi-instance deploys |
| Dependency manifests | Two `requirements.txt` files (root + sub-package) — intentional separation |
| Secret handling | Requires manual `.env` setup; no secrets manager integration yet |

---

## Summary

Meridian is a practical example of modern AI application engineering:

- a real frontend and backend stack
- explicit multi-agent orchestration
- deterministic planning for high-stakes academic logic
- source-aware research for fresh information
- traceable and testable behavior instead of opaque prompt-only answers

It demonstrates both software engineering discipline and applied AI systems thinking.

---

## Core Files To Read First

If you want to understand the system quickly, start here:

1. `IAAS/client/src/components/UnifiedChat.jsx` — main chat UI
2. `IAAS/client/src/services/apiService.js` — frontend API layer
3. `IAAS/LLMAcadamic-Advisor/api/api.py` — Flask route handlers
4. `IAAS/LLMAcadamic-Advisor/LLM/orchestration_pipeline.py` — full agent pipeline
5. `IAAS/LLMAcadamic-Advisor/LLM/main_copy.py` — executor logic
6. `IAAS/LLMAcadamic-Advisor/LLM/web_search_pipeline.py` — web research system
7. `IAAS/LLMAcadamic-Advisor/LLM/graduationFlowchart.py` — deterministic graduation planner

---

## License

MIT License — © 2024 Neerad Sood

---

## Author

**Neerad Sood**
