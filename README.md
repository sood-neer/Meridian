# Meridian — AI Academic Advisor (Leo)
### National Institute of Technology Hamirpur (NIT Hamirpur)

> Multi-agent LLM academic advisor and graduation planning system for **National Institute of Technology Hamirpur (NIT Hamirpur)** students — featuring curriculum-grounded RAG, NEP-2020 8-semester course planning, prerequisite graph validation, official web research, and open demo access.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Node](https://img.shields.io/badge/Node.js-18+-green?logo=node.js&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![LangChain](https://img.shields.io/badge/LangChain-0.2-orange)
![Tests](https://img.shields.io/badge/Tests-124%20Passing-success)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Overview

**Meridian** features **Leo**, an AI Academic Advisor tuned for undergraduate engineering students at **NIT Hamirpur**:

- **Curriculum-Grounded Planning:** Grounded in 695 official undergraduate courses indexed directly from the official NIT Hamirpur Bachelor Course Structure & Syllabus ([nith.ac.in/bachelor-course-structure-syllabus](https://nith.ac.in/bachelor-course-structure-syllabus)).
- **NEP-2020 Degree Structures:** Supports B.Tech programs across all major engineering disciplines (Computer Science & Engineering, Electronics & Communication Engineering, Electrical Engineering, Mechanical Engineering, Civil Engineering, Chemical Engineering, Materials Science & Engineering) across 8 semesters and 160 total credits (~20 credits/semester).
- **Indian Academic Context:** Built-in awareness of the 10-point CGPA grading scale ($O=10$, $A^+=9$, $A=8$, $B^+=7$, $B=6$, $C=5$, $D=4$, $F=0$), continuous internal evaluations (quizzes, mid-terms, labs), backlog clearance rules, and Training & Placement (T&P) Cell requirements.
- **Deterministic Prerequisite Graph:** Validates course sequences and hard prerequisites (e.g. `CS-111` $\rightarrow$ `CS-121` $\rightarrow$ `CS-211`) to prevent scheduling invalid courses.
- **Official Web Research Pipeline:** Prioritizes `site:nith.ac.in` for real-time institute deadlines, academic calendars, ordinances, and administrative leadership lookups.
- **Zero-Barrier Demo Access:** Runs out-of-the-box with 1-click quick-login (`MYSQL_DISABLE=true`), eliminating database setup barriers for local exploration and evaluation.

---

## System Architecture

```text
Student / Faculty Browser
  │
  ├── React Client (Port 3000) ─────────────── Unified Chat, Demo Quick-Login, Roadmap Graph
  │     │
  │     ├── Node Auth API (Port 5001) ──────── Optional MySQL user storage (Bypassed in Demo mode)
  │     │
  │     └── Flask AI Backend (Port 5002) ───── Multi-Agent Advisory Engine
  │           ├── Orchestrator Agent (Intent classification, plan scoping, persona)
  │           ├── Executor Agent (Path routing, RAG retrievers, tool execution)
  │           │     ├── Deterministic Graduation Flowchart (NEP-2020 160-cr scheduler)
  │           │     ├── Prerequisite DAG Engine (Hard prereqs, coreqs, level flooring)
  │           │     ├── Local Chroma Vectorstores (NITH Courses, Majors, Regulations)
  │           │     └── Shared Web Research Pipeline (Official nith.ac.in crawlers)
  │           └── Finalizer Agent (Grounded verification, prereq validation, tone polish)
```

---

## AI Multi-Agentic Workflow

### 1. Orchestrator Agent
- Resolves student intent (course info, graduation plan, policy/deadline, transcript audit, general chat).
- Enforces the **Leo** advising persona at **NIT Hamirpur**.
- Produces a strongly-typed `OrchestrationPlan` defining required tools, target major, semester horizon, and validation constraints.

### 2. Executor Agent
- Executes the orchestration plan across local retrieval and analytical tools:
  - `majors_retriever`: Catalog specifications from `nith_majors.csv`.
  - `course_retriever`: Syllabi and descriptions from `nith_courses.csv`.
  - `prerequisite_graph`: Direct graph inspection of course dependencies.
  - `policy_retriever`: Institute academic ordinances and regulations.
  - `graduationFlowchart`: 8-semester course scheduling with level-flooring and credit balancing.
  - `web_search`: Authoritative crawling of `nith.ac.in`.

### 3. Finalizer & Prerequisite Verifier
- Enforces strict prerequisite satisfaction before responses reach the student.
- Replaces vague placeholders with categorized course buckets.
- Formats structured semester breakdown cards (`Semester N: ... Total Credits: N`).
- Includes contextual guidance on grading, continuous evaluation, and next academic steps.

---

## Academic Data & Catalog Specifications

### NIT Hamirpur B.Tech Curricula
All degree specifications reflect NEP-2020 guidelines:
- **Total Credits:** 160 credits across 8 semesters.
- **Normal Semester Load:** ~18–22 credits per semester.
- **Course Coding Scheme:** Hyphenated departmental codes:
  - `CS`: Computer Science and Engineering (`CS-111`, `CS-121`, `CS-211`, `CS-212`, `CS-499`...)
  - `EC`: Electronics and Communication Engineering (`EC-101`, `EC-211`...)
  - `EE`: Electrical Engineering (`EE-101`, `EE-211`...)
  - `ME`: Mechanical Engineering (`ME-101`, `ME-102`...)
  - `CE`: Civil Engineering (`CE-101`...)
  - `MA`: Mathematics (`MA-111`, `MA-121`...)
  - `PH`: Physics (`PH-101`, `PH-102`...)
  - `CY`: Chemistry (`CY-101`, `CY-102`...)
  - `HS`: Humanities & Social Sciences (`HS-101`, `HS-102`...)
  - `MB`: Management Studies (`MB-101`...)
  - `EN`: Energy & Environmental Engineering (`EN-101`...)

---

## Repository Structure

```text
.
├── README.md                                  # Complete project documentation
├── requirements.txt                           # Core Python dependencies
├── .gitignore                                 # Production gitignore rules
├── IAAS/
│   ├── client/                                # React 18 frontend + Node auth server
│   │   ├── src/
│   │   │   ├── components/UnifiedChat.jsx     # Leo Academic Advising interface
│   │   │   ├── pages/LoginPage.jsx            # 1-click Demo Mode Quick Login
│   │   │   ├── pages/FacultyDashboard.jsx     # Major roadmap graph viewer
│   │   │   └── services/apiService.js         # API integration layer
│   │   ├── server/server.js                   # Node auth backend with demo fallback
│   │   └── setup.sh                           # Unified startup script
│   ├── LLMAcadamic-Advisor/                   # Flask AI Engine
│   │   ├── api/api.py                         # REST API route handlers
│   │   ├── LLM/
│   │   │   ├── orchestration_pipeline.py      # Orchestrator & multi-agent pipeline
│   │   │   ├── main_copy.py                   # Executor & tool orchestration
│   │   │   ├── graduationFlowchart.py         # Deterministic NITH course scheduler
│   │   │   ├── web_search_pipeline.py         # Official nith.ac.in research pipeline
│   │   │   ├── build_nith_datasets.py         # NITH dataset extraction tool
│   │   │   ├── nith_courses.csv               # 74 verified NITH courses & syllabi
│   │   │   └── nith_majors.csv                # 5 NITH B.Tech degree requirements
│   │   └── .env.example                       # Example environment configuration
│   ├── test_orchestration_pipeline.py         # Multi-agent regression tests
│   ├── test_web_search_pipeline.py            # Web retrieval tests
│   ├── test_course_prerequisites.py           # Prerequisite validation tests
│   └── test_api_session_store.py              # Session management tests
```

---

## API Endpoints

### Flask AI Backend (Port `5002`)

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/llm/generate-response` | Primary academic advising chat endpoint |
| `POST` | `/llm/setup-environment` | Ingests student transcript and sets up session RAG |
| `POST` | `/llm/generate-major-graph` | Generates prerequisite graph for a specified major |
| `POST` | `/llm/clear-conversation` | Resets active conversation context and memory |

### Node Auth Service (Port `5001`)

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/login` | Returns JWT (instant mock credentials in demo mode) |
| `POST` | `/api/register` | User registration (bypassed when `MYSQL_DISABLE=true`) |

---

## Getting Started

### 1. Prerequisites
- Python 3.10+
- Node.js 18+ and npm
- Google Gemini API key ([Google AI Studio](https://aistudio.google.com/app/apikey))

### 2. Clone and Setup Environment

```bash
git clone <repo-url>
cd AI_ACADEMIC_ADVISOR

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt -r IAAS/LLMAcadamic-Advisor/requirements.txt

# Install frontend and auth dependencies
cd IAAS/client && npm install
cd server && npm install && cd ../..
```

### 3. Environment Configuration

Copy the example environment file:

```bash
cp IAAS/LLMAcadamic-Advisor/.env.example IAAS/LLMAcadamic-Advisor/.env
```

Configure `IAAS/LLMAcadamic-Advisor/.env`:

```ini
# Google Gemini API Configuration
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-gemini-api-key-here
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
GEMINI_MODEL=gemini-3.5-flash-lite
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-2

# Disable MySQL requirement for local demo mode
MYSQL_DISABLE=true
```

> **Note:** Real `.env` files are excluded by `.gitignore` and must never be committed.

### 4. Running the Application

Launch all services concurrently using the root client runner:

```bash
cd IAAS/client
MYSQL_DISABLE=true npm start
```

This starts:
- **React Frontend:** [http://localhost:3000](http://localhost:3000) (Demo mode active — zero login required)
- **Node Auth API:** [http://localhost:5001](http://localhost:5001)
- **Flask AI Engine:** [http://127.0.0.1:5002](http://127.0.0.1:5002)

---

## Verification & Testing

### Python Test Suite
Run the 124 unit and regression tests:

```bash
./.venv/bin/python3 -m unittest \
  IAAS/test_orchestration_pipeline.py \
  IAAS/test_web_search_pipeline.py \
  IAAS/test_course_prerequisites.py \
  IAAS/test_api_session_store.py
```
*Result: 124 passed, 0 failed.*

### React Frontend Test Suite
Run component and UI rendering tests:

```bash
cd IAAS/client
npm test -- --watchAll=false
```
*Result: 7 passed, 7 total.*

### Production Build
Verify frontend production compilation:

```bash
cd IAAS/client
npm run build
```
*Result: Compiled successfully.*

---

## License

MIT License — © 2024–2026 Neerad Sood
