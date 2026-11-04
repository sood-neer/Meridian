import os
import sys

from dotenv import load_dotenv

# Load .env from IAAS/LLMAcadamic-Advisor or repo root
_llm_dir = os.path.dirname(os.path.abspath(__file__))
for _env_path in [
    os.path.join(_llm_dir, "..", ".env"),  # IAAS/LLMAcadamic-Advisor/.env
    os.path.join(_llm_dir, "..", "..", "..", ".env"),  # repo root
]:
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
        break
else:
    load_dotenv()
import re
import json
import shutil
import hashlib
import pickle
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any, Callable
from urllib.parse import urlparse, urljoin
import time as _dbg_time

_DBG_LOG_PATH = os.getenv("MERIDIAN_DEBUG_LOG_PATH", "").strip()
_DBG_SESSION_ID = os.getenv("MERIDIAN_DEBUG_SESSION_ID", "local")
def _dbg(hyp: str, location: str, message: str, data: Dict[str, Any]) -> None:
    if not _DBG_LOG_PATH:
        return
    try:
        with open(_DBG_LOG_PATH, "a") as _f:
            _f.write(json.dumps({
                "sessionId": _DBG_SESSION_ID,
                "runId": "r1",
                "hypothesisId": hyp,
                "location": location,
                "message": message,
                "data": data,
                "timestamp": int(_dbg_time.time() * 1000),
            }) + "\n")
    except Exception:
        pass
from html import unescape

# Add current directory to Python path BEFORE importing local modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document

import pandas as pd

# Make sure local modules can be imported


from graphExperiment import (
    build_course_catalog_index,
    course_graph,
    format_course_prerequisite_summary,
    get_course_prerequisites,
    get_major_course_codes,
)
from orchestration_pipeline import (
    ADVISOR_PERSONA,
    AGENT_DISPLAY_NAME,
    ExecutorResult,
    OrchestrationPlan,
    PRIVATE_REASONING_GUIDANCE,
    _canonical_path_for_domain,
    _canonical_tools_for_path,
    canonicalize_semester_blocks,
    _parse_prereq_graph_text,
    _requires_grounded_answer,
    _looks_like_plan_request,
    run_three_stage_pipeline,
    sanitize_public_response,
)
from web_search_pipeline import (
    ResolvedResearchAnswer,
    clear_active_search_context,
    run_search_pipeline,
)


# =========================
# Globals
# =========================

pdf_retriever = None
course_retriever = None  # For individual course queries (processed_psu_courses.csv)
majors_retriever = None  # For major requirements queries (all_abington_majors_combined(5).csv)
policy_retriever = None  # For advising/registrar policy PDF queries
policy_vectorstore = None
llm1 = None
llm2 = None
llm_classifier = None  # Shared reasoning/orchestration model
has_transcript = False
stored_pdf_docs = None  # Store original PDF docs for deterministic extraction
latest_student_data = None  # Cache most recent extracted student data
execution_trace_store = {}  # Per-session lightweight execution traces

def is_gemini_active() -> bool:
    provider = os.getenv("LLM_PROVIDER", "").lower()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    return bool(gemini_key) and (provider == "gemini" or not os.getenv("OPENAI_API_KEY") or os.getenv("USE_GEMINI", "").lower() == "true")

def get_default_model_name() -> str:
    if is_gemini_active():
        return os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    return os.getenv("ADVISOR_MODEL_NAME", "gpt-5.4-mini")

PRIMARY_MODEL_NAME = get_default_model_name()
CLASSIFIER_MODEL_NAME = os.getenv("ADVISOR_CLASSIFIER_MODEL_NAME", PRIMARY_MODEL_NAME)

def create_chat_model(model_name: str = None, temperature: float = 0):
    """Instantiate ChatOpenAI targeting either OpenAI or Gemini OpenAI-compatible endpoint."""
    if is_gemini_active():
        gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        gemini_model = model_name or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        base_url = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
        return ChatOpenAI(
            base_url=base_url,
            api_key=gemini_key,
            model_name=gemini_model,
            temperature=temperature,
            max_retries=2,
            request_timeout=35
        )
    return ChatOpenAI(model_name=model_name or PRIMARY_MODEL_NAME, temperature=temperature)

def create_embeddings_model():
    """Instantiate embeddings model targeting either OpenAI or Gemini OpenAI-compatible endpoint."""
    if is_gemini_active():
        gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        base_url = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
        target_model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
        return OpenAIEmbeddings(
            base_url=base_url,
            api_key=gemini_key,
            model=target_model,
            dimensions=1536,
            check_embedding_ctx_length=False
        )
    return OpenAIEmbeddings()

# Absolute paths for Chroma persistence
LLM_DIR = os.path.abspath(os.path.dirname(__file__))
CHROMA_DATA_DIR = os.path.join(LLM_DIR, "chroma_data")
COLLECTION_METADATA_FILE = os.path.join(CHROMA_DATA_DIR, "collection_metadata.json")
POLICY_PDF_PATH = os.path.join(LLM_DIR, "PennState_Combined_Advising_and_Registrar_VERBATIM.pdf")
COURSE_RETRIEVER_FORMAT_VERSION = "course-prereq-canonical-v2"

# =========================
# Helper Functions for Document Processing
# =========================

def get_document_hash(file_path: str) -> str:
    """Generate SHA256 hash of PDF file."""
    with open(file_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def get_transcript_collection_name(user_id: str, doc_hash: str) -> str:
    """Generate collection name: transcript_userid_hash8"""
    return f"transcript_{user_id}_{doc_hash[:8]}"

def chunk_with_metadata(pdf_docs: List[Document]) -> List[Document]:
    """Chunk PDF with page and section metadata."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=130
    )
    chunks = text_splitter.split_documents(pdf_docs)
    
    # Add metadata
    for i, chunk in enumerate(chunks):
        page = chunk.metadata.get("page", 0)
        content = chunk.page_content
        
        # Detect section
        section = "general"
        if "Not Satisfied" in content or "Unmet" in content:
            section = "unmet_requirements"
        elif "In Progress" in content:
            section = "in_progress"
        elif "Course History" in content or re.search(r'\b[A-Z]{2,}\s+\d{3}\b', content):
            section = "course_history"
        elif "Major" in content and "Option" in content:
            section = "major_info"
        
        chunk.metadata.update({
            "page": page,
            "section": section,
            "chunk_id": f"page_{page}_chunk_{i}"
        })
    
    return chunks


def is_policy_question(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    policy_terms = [
        "advisor", "adviser", "advising", "advising center",
        "drop", "add", "late drop", "withdraw", "audit",
        "leave of absence", "declare major", "declare minor",
        "double major", "concurrent major", "change my major",
        "what-if report", "lionpath", "academic requirements",
        "entry to major", "early progress report", "internship",
        "aleks", "world campus", "registration", "bursar",
        "tuition penalty", "credits", "f grade", "gpa"
    ]
    return any(term in t for term in policy_terms)


def is_academic_or_policy_question(text: str) -> bool:
    """
    Check if the question is about academics or Penn State policy.
    Used to: (1) allow web search fallback, (2) reject non-academic questions with scope message.
    """
    if not text or len(text.strip()) < 5:
        return False
    t = text.lower()
    academic_policy_terms = [
        # Policy / institutional
        "advisor", "adviser", "advising", "drop", "add", "withdraw", "audit",
        "leave of absence", "declare major", "declare minor", "registration",
        "bursar", "tuition", "gpa", "grade", "credit", "lionpath",
        "penn state", "psu", "university policy", "academic policy",
        "academic requirement", "graduation", "degree", "enrollment",
        "course", "class", "major", "minor", "prerequisite",
        "semester", "transcript", "degree audit", "what-if",
        "internship", "co-op", "study abroad", "transfer credit",
    ]
    return any(term in t for term in academic_policy_terms)


def _response_indicates_no_answer_found(response: str) -> bool:
    """True only when the LLM response is *entirely* a "couldn't answer" message.

    This gatekeeper exists to detect total give-ups (e.g., a 30-character
    "I don't have any information on that") so we can fall through to the
    web-search backstop. It must NOT fire on real, structured plans that
    happen to acknowledge one missing data point — that mistake threw away
    a perfectly good Art + Econ double-major plan in production because
    the synthesizer (correctly!) said "I don't have X for the option
    track" inside an otherwise complete answer.

    Logic:
      * Short / empty responses → no answer.
      * Response contains a "Semester N" block, two or more PSU-style
        course codes, or is reasonably long (≥ 800 chars) → real answer
        regardless of the indicator phrases.
      * Otherwise apply the phrase scan.
    """
    if not response or len(response.strip()) < 20:
        return True
    if re.search(r"\bSemester\s+\d", response, re.IGNORECASE):
        return False
    course_codes = re.findall(r"\b[A-Z]{2,}\s+\d{2,3}[A-Z]?\b", response)
    if len(set(c.upper() for c in course_codes)) >= 2:
        return False
    if len(response.strip()) > 800:
        return False
    lower = response.lower()
    indicators = [
        "couldn't find", "could not find", "isn't present", "not in the policy",
        "not found in", "unable to find", "i don't have", "i do not have",
        "couldn't find that", "no relevant", "please rephrase", "not available",
        "couldn't verify", "could not verify",
    ]
    return any(ind in lower for ind in indicators)


def _is_time_sensitive_identity_question(text: str) -> bool:
    """Detect questions that require up-to-date role holder verification."""
    if not text:
        return False
    t = text.lower()
    role_terms = [
        "chancellor", "program chair", "chair", "dean", "director",
        "president", "provost", "head of", "department head",
    ]
    asking_identity = any(
        p in t for p in [
            "who is", "who's", "current", "currently", "now", "at present",
            "who serves as", "who holds",
        ]
    )
    return asking_identity and any(r in t for r in role_terms)


def _is_time_sensitive_factual_question(text: str) -> bool:
    """Generic detector for questions requiring fresh, source-backed facts."""
    if not text:
        return False
    t = text.lower().strip()
    starters = re.search(r"^(who|what|when)\b", t) is not None
    freshness_terms = any(k in t for k in ["current", "currently", "latest", "now", "this semester", "this year", "upcoming"])
    institution_terms = any(k in t for k in [
        "penn state", "psu", "abington", "chancellor", "chair", "dean", "director",
        "deadline", "calendar", "registration", "policy"
    ])
    return (starters and institution_terms) or (freshness_terms and institution_terms)


def _build_execution_profile(
    contextual_question: str,
    query_type: str,
    reasoning_plan: Dict[str, object]
) -> Dict[str, object]:
    """Universal execution profile used by all query paths."""
    tools = reasoning_plan.get("tools", {}) if reasoning_plan else {}
    web_enabled = bool(tools.get("web_search")) or is_academic_or_policy_question(contextual_question)
    mode = "none"
    require_web_precheck = False
    must_include_source = False
    prefer_current_or_future = False

    if _question_needs_specific_date(contextual_question):
        mode = "date"
        require_web_precheck = True
        prefer_current_or_future = True
    elif _is_time_sensitive_factual_question(contextual_question) and query_type in {"chat", "policy", "major", "course"}:
        mode = "fresh_fact"
        require_web_precheck = True
        must_include_source = True

    return {
        "verification_mode": mode,
        "require_web_precheck": require_web_precheck and web_enabled,
        "must_include_source": must_include_source,
        "prefer_current_or_future": prefer_current_or_future,
        "web_enabled": web_enabled,
        "identity_like": _is_time_sensitive_identity_question(contextual_question),
        "source_domain_hints": _extract_domain_hints(contextual_question),
    }


def _recent_user_messages(conversation_history: Optional[List[dict]], limit: int = 10) -> List[str]:
    if not conversation_history:
        return []
    msgs = [str(m.get("content", "")).strip() for m in conversation_history if m.get("role") == "user"]
    return [m for m in msgs[-limit:] if m]


def _is_vague_followup(text: str) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    explicit_followup_terms = [
        "current semester", "this semester", "for this semester", "for the current semester",
        "that one", "for that", "for this", "what about this", "after that", "same one",
    ]
    if any(term in t for term in explicit_followup_terms):
        return True

    # Term/session-only answers are typical follow-ups to deadline questions.
    if len(t.split()) <= 5 and (
        re.search(r"\b(spring|summer|fall|winter)\b", t)
        or re.search(r"\b20\d{2}\b", t)
        or "regular session" in t
        or "7-week" in t
        or "7 week" in t
        or "session" in t
    ):
        return True

    # Treat full WH questions as new topics unless explicit follow-up markers exist.
    if re.search(r"^(who|what|when|where|why|how)\b", t) and len(t.split()) >= 4:
        return False

    return False


def _contextualize_policy_question(user_question: str, conversation_history: Optional[List[dict]] = None) -> str:
    """Attach prior policy/deadline context for vague follow-up messages."""
    q = (user_question or "").strip()
    if not q:
        return q
    if not conversation_history or not _is_vague_followup(q):
        return q

    prior_user = _recent_user_messages(conversation_history, limit=8)
    for msg in reversed(prior_user):
        m = msg.lower()
        if any(k in m for k in ["deadline", "late drop", "withdraw", "drop deadline", "drop period"]):
            return f"{msg}. Clarification: {q}"
    return q


def _get_current_datetime_context() -> str:
    now = datetime.now()
    if now.month <= 5:
        current_term = "Spring"
    elif now.month <= 8:
        current_term = "Summer"
    else:
        current_term = "Fall"
    return (
        f"Current local date/time: {now.strftime('%A, %B %d, %Y at %I:%M %p')}. "
        f"Treat 'current semester' as {current_term} {now.year} unless the user specifies another term."
    )


def _infer_term_year_from_context(
    user_question: str,
    conversation_history: Optional[List[dict]] = None
) -> Tuple[Optional[str], Optional[int]]:
    """Infer target term/year, preferring current or upcoming semester for vague requests."""
    q = _contextualize_policy_question(user_question, conversation_history).lower()
    now = datetime.now()
    current_year = now.year

    if now.month <= 5:
        current_term = "spring"
    elif now.month <= 8:
        current_term = "summer"
    else:
        current_term = "fall"

    year_match = re.search(r"\b(20\d{2})\b", q)
    target_year = int(year_match.group(1)) if year_match else None

    term = None
    for term_token in ["spring", "summer", "fall", "winter"]:
        if term_token in q:
            term = term_token
            break

    if "next semester" in q or "upcoming semester" in q:
        if current_term == "spring":
            term, target_year = "summer", current_year
        elif current_term == "summer":
            term, target_year = "fall", current_year
        else:
            term, target_year = "spring", current_year + 1
    elif "current semester" in q or "this semester" in q:
        term = term or current_term
        target_year = target_year or current_year

    if target_year is None:
        target_year = current_year

    return term, target_year


def _question_needs_specific_date(text: str) -> bool:
    """Detect date/deadline style questions that usually need a concrete date."""
    if not text:
        return False
    t = text.lower()
    date_terms = [
        "deadline", "due date", "what date", "exact date", "by when",
        "when is", "last day", "cutoff", "closing date", "ends on",
    ]
    academic_event_terms = [
        "late drop", "drop", "withdraw", "registration", "add period",
        "semester start", "semester end", "tuition penalty",
    ]
    return any(dt in t for dt in date_terms) and any(ev in t for ev in academic_event_terms)


def _has_concrete_date_in_text(text: str) -> bool:
    """Heuristic check for concrete calendar dates in a response."""
    if not text:
        return False
    t = text.lower()
    month_pattern = r"\b(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)\b"
    numeric_date_pattern = r"\b\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?\b"
    long_date_pattern = r"\b\d{1,2}(st|nd|rd|th)?\b"
    has_month = re.search(month_pattern, t) is not None
    has_numeric = re.search(numeric_date_pattern, t) is not None
    # A month + day signal like "March 22" (allow optional ordinal suffix)
    has_month_and_day = has_month and re.search(long_date_pattern, t) is not None
    return has_numeric or has_month_and_day


def _extract_dates_from_text(text: str) -> List[str]:
    """Extract concrete date strings from free text."""
    if not text:
        return []
    month_date = re.findall(
        r"\b(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December)\s+\d{1,2}(?:,\s*\d{2,4})?\b",
        text,
        flags=re.IGNORECASE,
    )
    numeric_date = re.findall(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b", text)
    dates: List[str] = []
    for d in month_date + numeric_date:
        cleaned = " ".join(d.split()).strip()
        if cleaned not in dates:
            dates.append(cleaned)
    return dates


def _parse_date_candidate(date_text: str, fallback_year: Optional[int] = None) -> Optional[datetime]:
    """Parse supported date formats into datetime for ranking/validation."""
    if not date_text:
        return None
    s = date_text.strip()
    s = re.sub(r'(\d)(st|nd|rd|th)\b', r'\1', s, flags=re.IGNORECASE)
    s = re.sub(r'\s+', ' ', s)

    formats = [
        "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y",
        "%m/%d/%Y", "%m-%d-%Y", "%m/%d/%y", "%m-%d-%y",
        "%B %d", "%b %d", "%m/%d", "%m-%d",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(s, fmt)
            if "%Y" not in fmt and "%y" not in fmt:
                dt = dt.replace(year=fallback_year or datetime.now().year)
            return dt
        except Exception:
            continue
    return None


def _extract_json_object_from_text(raw_text: str) -> Optional[Dict[str, Any]]:
    if not raw_text:
        return None
    raw = str(raw_text).strip()
    if "```" in raw:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            raw = raw[start:end]
    try:
        parsed = json.loads(raw)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _build_deadline_search_queries(
    user_question: str,
    conversation_history: Optional[List[dict]] = None
) -> List[str]:
    """Build targeted PSU policy/deadline web queries."""
    context_q = _contextualize_policy_question(user_question, conversation_history)
    term, year = _infer_term_year_from_context(user_question, conversation_history)
    year_txt = str(year) if year else ""
    term_txt = (term or "current").strip()
    campus_txt = "Penn State Abington" if "abington" in context_q.lower() else "Penn State"

    query_variants = [
        f"{campus_txt} {term_txt} {year_txt} academic calendar late drop",
        f"{campus_txt} {term_txt} {year_txt} academic calendar drop add",
        f"{campus_txt} {term_txt} {year_txt} early drop deadline",
        f"{campus_txt} {term_txt} {year_txt} late drop deadline",
        f"site:psu.edu {campus_txt} {term_txt} {year_txt} academic calendar",
        f"site:pennstate.edu {campus_txt} {term_txt} {year_txt} late drop deadline",
        f"site:psu.edu {campus_txt} {term_txt} {year_txt} drop add deadline",
    ]

    deduped: List[str] = []
    seen = set()
    for query in query_variants:
        cleaned = " ".join(query.split()).strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            deduped.append(cleaned)
    return deduped[:8]


def _lookup_exact_deadline_dates(
    user_question: str,
    conversation_history: Optional[List[dict]] = None,
    trace: Optional[Dict[str, Any]] = None,
    user_id: str = "default",
) -> Optional[str]:
    """Look up exact deadline dates through the shared research pipeline."""
    contextual_question = _contextualize_policy_question(user_question, conversation_history)
    result = _run_shared_search_pipeline(
        user_id=user_id,
        question_text=contextual_question,
        conversation_history=conversation_history,
        trace=trace,
        preferred_strategy="deadline_exact_date",
        preferred_domains=["abington.psu.edu", "psu.edu", "pennstate.edu"],
        required_domain_mode="strict",
    )
    answer_text = _research_answer_text(result)
    if not answer_text:
        return None
    if _response_indicates_no_answer_found(answer_text) and not result.source_urls:
        return None
    return answer_text


def _build_identity_search_queries(user_question: str) -> List[str]:
    q = (user_question or "").lower()
    keyword_q = _keywordize_query(user_question)
    scope_abington = "abington" in q
    base = []
    if scope_abington:
        base.extend([
            f"site:abington.psu.edu {keyword_q}",
            f"Penn State Abington {keyword_q}",
        ])
    base.extend([
        f"site:psu.edu {keyword_q}",
        f"Penn State {keyword_q}",
    ])
    if "abington" in q:
        base.insert(0, f"Penn State Abington {keyword_q}")
    if "chancellor" in q:
        if scope_abington:
            base.insert(0, "site:abington.psu.edu chancellor")
        else:
            base.insert(0, "site:psu.edu leadership")
            base.insert(1, "Penn State leadership")
    if "program chair" in q or "chair" in q:
        base.insert(0, "site:abington.psu.edu computer science program chair")
        base.insert(1, "site:abington.psu.edu computer science faculty")
    if "provost" in q:
        base.insert(0, "site:psu.edu provost")
        base.insert(1, "Penn State provost")
    # Deduplicate while preserving order
    seen = set()
    out = []
    for b in base:
        if b not in seen:
            seen.add(b)
            out.append(b)
    return out[:6]


def _extract_domain_hints(question: str) -> List[str]:
    q = (question or "").lower()
    hints: List[str] = []
    if any(k in q for k in ["penn state", "psu"]):
        hints.extend(["psu.edu", "pennstate.edu"])
    # Default to PSU domains for in-scope academic advising questions in this app.
    if not hints and is_academic_or_policy_question(question):
        hints.extend(["psu.edu", "pennstate.edu"])
    return hints


def _domain_from_url(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""


def _is_domain_allowed(url: str, domain_hints: List[str]) -> bool:
    if not domain_hints:
        return True
    d = _domain_from_url(url)
    return any(d.endswith(h) for h in domain_hints)


def _keywordize_query(query: str) -> str:
    q = (query or "").lower()
    q = re.sub(r"[^a-z0-9\s]", " ", q)
    stop = {
        "who", "what", "when", "where", "why", "how", "is", "are", "the", "a", "an",
        "at", "of", "for", "to", "current", "currently", "please", "me", "tell",
        "then", "uh", "um", "about",
    }
    toks = [t for t in q.split() if t and t not in stop]
    return " ".join(toks[:10]).strip() or q.strip()


def _normalize_question_for_search(question: str) -> str:
    q = (question or "").strip().lower()
    # Fix common typo/extra-token patterns that hurt retrieval quality.
    q = re.sub(r"\bwho\s+is\s+then\b", "who is the", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def _expand_academic_query_terms(question: str) -> str:
    q = _normalize_question_for_search(question)
    q = re.sub(r"\bcs\b", "computer science", q, flags=re.IGNORECASE)
    q = re.sub(r"\bclasses\b", "courses", q, flags=re.IGNORECASE)
    if is_academic_or_policy_question(q) and "penn state" not in q and "psu" not in q:
        q = f"penn state {q}"
    return " ".join(q.split()).strip()


def _relax_search_query(query: str) -> str:
    """Relax strict operators/phrasing to improve hit rate."""
    q = (query or "").strip()
    q = re.sub(r"\bsite:[^\s]+\s*", "", q, flags=re.IGNORECASE)
    q = q.replace('"', '').replace("  ", " ").strip()
    return q


def _build_general_search_queries(user_question: str) -> List[str]:
    q = (user_question or "").strip()
    if not q:
        return []
    lower_q = q.lower()
    keyword_q = _keywordize_query(q)
    variants = [
        q,
        keyword_q,
        f"{q} official source",
        f"{q} latest update",
    ]
    if "penn state" in lower_q or "psu" in lower_q or "abington" in lower_q:
        variants.extend([
            f"Penn State {q}",
            f"site:psu.edu {q}",
            f"site:pennstate.edu {q}",
        ])

    out: List[str] = []
    seen = set()
    for v in variants:
        vv = " ".join(v.split()).strip()
        if vv and vv not in seen:
            seen.add(vv)
            out.append(vv)
    return out[:8]


def _run_multi_query_web_search(
    query_variants: List[str],
    trace: Optional[Dict[str, Any]] = None,
    max_per_query: int = 5,
    domain_hints: Optional[List[str]] = None,
    require_domain_match: bool = False
) -> List[dict]:
    """Universal retrieval: strict queries first, then relaxed fallback."""
    from duckduckgo_search import DDGS

    if domain_hints is None:
        domain_hints = []
    gathered: List[dict] = []
    seen = set()

    def _collect(query: str) -> int:
        hits = 0
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_per_query))
        _trace_event(trace, "execute", "Executed web query", {"query": query, "results": len(results)})
        for r in results:
            href = str(r.get("href", "") or "")
            key = href or f"{r.get('title','')}-{r.get('body','')[:120]}"
            if key in seen:
                continue
            seen.add(key)
            gathered.append(r)
            hits += 1
        return hits

    # Pass 1: strict/original variants
    for q in query_variants[:8]:
        try:
            _collect(q)
        except Exception as e:
            _trace_event(trace, "execute", "Web query failed", {"query": q, "error": str(e)})

    # Pass 2: relaxed variants if we got nothing
    if len(gathered) == 0:
        relaxed = []
        for q in query_variants[:8]:
            rq = _relax_search_query(q)
            if rq and rq not in relaxed:
                relaxed.append(rq)
        for q in relaxed[:8]:
            try:
                _collect(q)
            except Exception as e:
                _trace_event(trace, "execute", "Relaxed web query failed", {"query": q, "error": str(e)})

    # Domain filtering (authoritative-source preference/requirement)
    if domain_hints:
        filtered = [r for r in gathered if _is_domain_allowed(str(r.get("href", "") or ""), domain_hints)]
        _trace_event(
            trace,
            "execute",
            "Applied domain filtering",
            {
                "domain_hints": domain_hints,
                "before": len(gathered),
                "after": len(filtered),
                "required": require_domain_match
            }
        )
        if filtered:
            return filtered
        if require_domain_match:
            return []

    return gathered


def _strip_html_to_text(html: str) -> str:
    if not html:
        return ""
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_candidate_links(base_url: str, html: str) -> List[str]:
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html or "", flags=re.IGNORECASE)
    base_domain = _domain_from_url(base_url)
    candidates: List[str] = []
    for h in hrefs:
        if not h or h.startswith("#") or h.startswith("mailto:") or h.startswith("tel:"):
            continue
        full = urljoin(base_url, h)
        if not full.startswith("http"):
            continue
        if _domain_from_url(full) != base_domain:
            continue
        # Skip static assets and noisy non-content URLs.
        low = full.lower()
        if any(low.endswith(ext) for ext in [".css", ".js", ".ico", ".png", ".jpg", ".jpeg", ".svg", ".woff", ".woff2", ".ttf", ".pdf"]):
            continue
        if any(tok in low for tok in ["/sites/", "/themes/", "/files/css/", "/files/js/"]):
            continue
        candidates.append(full)
    # Prefer leadership/administration-relevant paths
    priority_tokens = ["leadership", "administration", "about", "chancellor", "provost", "faculty", "staff", "directory"]
    candidates = sorted(
        list(dict.fromkeys(candidates)),
        key=lambda u: sum(1 for t in priority_tokens if t in u.lower()),
        reverse=True,
    )
    return candidates[:20]


def _extract_role_keywords(question: str) -> List[str]:
    q = (question or "").lower()
    kws: List[str] = []
    for k in ["chancellor", "provost", "president", "program chair", "chair", "dean", "director"]:
        if k in q:
            kws.append(k)
    return kws or ["leadership"]


def _is_plausible_person_name(name: str) -> bool:
    if not name:
        return False
    parts = [p for p in re.split(r"\s+", name.strip()) if p]
    if len(parts) < 2:
        return False
    bad_tokens = {
        "development", "office", "staff", "administration", "admissions",
        "directory", "results", "campus", "leadership", "university",
        "department", "college", "program", "penn", "state", "abington"
    }
    first = parts[0].lower().strip(".,")
    second = parts[1].lower().strip(".,")
    if first in bad_tokens or second in bad_tokens:
        return False
    # Must look like person-case words (allow apostrophes/hyphens)
    token_ok = all(re.match(r"^[A-Z][a-zA-Z'\-]+\.?$", p) for p in parts[:2])
    return bool(token_ok)


def _extract_role_holder_candidates(text: str, role_keywords: List[str]) -> List[str]:
    """Extract candidate role-holder names from role-aware text patterns."""
    if not text:
        return []
    candidates: List[str] = []
    role_alt = "|".join(re.escape(k) for k in role_keywords if k)

    patterns = [
        # Chancellor Dr. Jane Doe
        rf"\b(?:{role_alt})\b[^A-Za-z]{{0,20}}(Dr\.\s+[A-Z][a-zA-Z'\-]+\s+[A-Z][a-zA-Z'\-]+)",
        # Chancellor Jane Doe
        rf"\b(?:{role_alt})\b[^A-Za-z]{{0,20}}([A-Z][a-zA-Z'\-]+\s+[A-Z][a-zA-Z'\-]+)",
        # Jane Doe, Chancellor
        rf"\b([A-Z][a-zA-Z'\-]+\s+[A-Z][a-zA-Z'\-]+)\b[^A-Za-z]{{0,20}}\b(?:{role_alt})\b",
    ]

    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            cand = m.group(1).strip()
            # normalize "dr." casing
            cand = re.sub(r"^dr\.\s+", "Dr. ", cand, flags=re.IGNORECASE)
            cl = cand.lower()
            if any(rk in cl for rk in ["president", "chancellor", "provost", "dean", "director", "chair"]):
                continue
            if _is_plausible_person_name(cand) and cand not in candidates:
                candidates.append(cand)
    return candidates


def _fetch_official_role_holder(
    question: str,
    domain_hints: List[str],
    trace: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Deterministic fallback: scrape known official leadership pages and extract likely role holder.
    """
    q = _normalize_question_for_search(question)
    role_keywords = _extract_role_keywords(q)
    scope_abington = "abington" in q
    candidate_urls = []
    if scope_abington:
        candidate_urls.extend([
            "https://www.abington.psu.edu/",
            "https://www.abington.psu.edu/about",
            "https://www.abington.psu.edu/about/administration",
            "https://www.abington.psu.edu/faculty-and-staff",
        ])
    else:
        candidate_urls.extend([
            "https://www.psu.edu/",
            "https://www.psu.edu/about/leadership/",
            "https://www.psu.edu/about/administration/",
        ])
    if "provost" in q:
        candidate_urls.insert(0, "https://www.psu.edu/about/administration/")
    if "chancellor" in q:
        if scope_abington:
            candidate_urls.insert(0, "https://www.abington.psu.edu/about/leadership")
        else:
            candidate_urls.insert(0, "https://www.psu.edu/about/leadership/")

    for url in candidate_urls:
        if domain_hints and not _is_domain_allowed(url, domain_hints):
            continue
        try:
            _trace_tool(trace, "requests_get", "official page fallback", metadata={"url": url})
            resp = requests.get(url, timeout=7, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code != 200:
                _trace_event(trace, "execute", "Official page fetch non-200", {"url": url, "status": resp.status_code})
                continue
            pages_to_scan = [(url, resp.text)]
            for link in _extract_candidate_links(url, resp.text)[:8]:
                try:
                    _trace_tool(trace, "requests_get", "official link follow", metadata={"url": link})
                    lr = requests.get(link, timeout=7, headers={"User-Agent": "Mozilla/5.0"})
                    if lr.status_code == 200:
                        pages_to_scan.append((link, lr.text))
                except Exception:
                    continue

            for page_url, html in pages_to_scan:
                text = _strip_html_to_text(html)
                low = text.lower()
                if scope_abington and "abington" not in low:
                    continue
                if not any(k in low for k in role_keywords):
                    continue
                # Role-aware extraction first (higher precision), then fallback.
                candidates = _extract_role_holder_candidates(text, role_keywords)
                if candidates:
                    return f"The current office holder appears to be {candidates[0]}.\nSource: {page_url}"

                # Fallback: search around role windows for plausible person names.
                for k in role_keywords:
                    for m in re.finditer(re.escape(k), low):
                        start = max(0, m.start() - 180)
                        end = min(len(text), m.end() + 220)
                        window = text[start:end]
                        win_candidates = _extract_role_holder_candidates(window, [k])
                        if win_candidates:
                            return f"The current office holder appears to be {win_candidates[0]}.\nSource: {page_url}"
        except Exception as e:
            _trace_event(trace, "execute", "Official page fetch failed", {"url": url, "error": str(e)})
            continue
    return None


def _extract_person_name_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    # Prefer "Dr. First Last" style names
    m = re.search(r"\b(Dr\.\s+[A-Z][a-zA-Z'-]+\s+[A-Z][a-zA-Z'-]+)\b", text)
    if m:
        return m.group(1).strip()
    # Fallback: First Last (two capitalized words) with optional middle initial
    m2 = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z]\.)?\s+[A-Z][a-z]+)\b", text)
    if m2:
        candidate = m2.group(1).strip()
        bad_starts = {
            "as", "the", "this", "that", "for", "from", "with", "in", "on",
            "staff", "office", "penn", "state", "chancellor", "provost",
            "dean", "director", "president"
        }
        first = candidate.split()[0].lower() if candidate.split() else ""
        if candidate.lower() not in {"penn state", "state university"} and first not in bad_starts:
            return candidate
    return None


def _rank_identity_result(result: dict, role_keywords: List[str]) -> int:
    title = str(result.get("title", "") or "")
    body = str(result.get("body", "") or "")
    href = str(result.get("href", "") or "").lower()
    text = f"{title}\n{body}".lower()
    score = 0
    if "abington.psu.edu" in href:
        score += 6
    elif "psu.edu" in href or "pennstate.edu" in href:
        score += 4
    if any(k in text for k in role_keywords):
        score += 4
    if "directory" in href or "faculty" in href or "leadership" in href or "administration" in href:
        score += 2
    if "linkedin" in href or "facebook" in href:
        score -= 3
    return score


def _rank_deadline_date_candidates(
    results: List[dict],
    target_year: Optional[int] = None,
    prefer_current_or_future: bool = True,
    strict_target_year: bool = False,
    strict_current_or_future: bool = False
) -> List[Tuple[datetime, int]]:
    """Extract and score date candidates from search results."""
    scored: List[Tuple[datetime, int]] = []
    now = datetime.now()

    for r in results:
        title = str(r.get("title", "") or "")
        body = str(r.get("body", "") or "")
        href = str(r.get("href", "") or "")
        snippet = f"{title}\n{body}"
        snippet_lower = snippet.lower()
        has_deadline_terms = any(k in snippet_lower for k in ["late drop", "drop deadline", "withdraw", "deadline", "drop"])
        is_calendar_like = any(k in href.lower() for k in ["calendar", "academic", "registrar", "deadline", "lionpath"])

        for d in _extract_dates_from_text(snippet):
            dt = _parse_date_candidate(d, fallback_year=target_year or now.year)
            if not dt:
                continue

            score = 0
            if "psu.edu" in href or "pennstate.edu" in href:
                score += 3
            if has_deadline_terms:
                score += 2
            if "late drop" in snippet_lower:
                score += 3
            if is_calendar_like:
                score += 2
            if target_year is not None:
                if strict_target_year and dt.year != target_year:
                    continue
                score += 3 if dt.year == target_year else -3
            if prefer_current_or_future:
                if strict_current_or_future and dt.date() < now.date():
                    continue
                score += 2 if dt.date() >= now.date() else -4

            # Penalize clearly stale years unless user explicitly asked for them
            if target_year is None and dt.year < now.year:
                score -= 3

            scored.append((dt, score))

    scored.sort(key=lambda x: (x[1], -abs((x[0].date() - datetime.now().date()).days)), reverse=True)
    return scored


def _web_search_and_synthesize(
    user_question: str,
    date_only: bool = False,
    conversation_history: Optional[List[dict]] = None,
    prefer_current_or_future: bool = False,
    trace: Optional[Dict[str, Any]] = None,
    force_fresh_identity: bool = False,
    user_id: str = "default",
) -> Optional[str]:
    """Run the shared research pipeline and return the synthesized answer text."""
    try:
        strategy = "generic_authoritative_lookup"
        required_domain_mode = None
        preferred_domains = None
        if date_only:
            strategy = "deadline_exact_date"
            preferred_domains = ["abington.psu.edu", "psu.edu", "pennstate.edu"]
            required_domain_mode = "strict"
        elif force_fresh_identity:
            strategy = "identity_current_role"
        result = _run_shared_search_pipeline(
            user_id=user_id,
            question_text=user_question,
            conversation_history=conversation_history,
            trace=trace,
            preferred_strategy=strategy,
            preferred_domains=preferred_domains,
            required_domain_mode=required_domain_mode,
        )
        answer_text = _research_answer_text(result)
        if not answer_text:
            return "DATE_NOT_FOUND" if date_only else None
        return answer_text
    except Exception as e:
        print(f"⚠️ Web search failed: {e}")
        _trace_tool(trace, "web_search_pipeline", "shared research pipeline", status="failed", metadata={"error": str(e)})
        return "DATE_NOT_FOUND" if date_only else None


def _observe_and_repair_response(
    user_question: str,
    response: str,
    reasoning_plan: Dict[str, object],
    query_type: str,
    conversation_history: Optional[List[dict]] = None,
    trace: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Universal observe/repair stage for all query types:
      - Validate whether response satisfies plan intent.
      - Attempt one repair pass using available tools.
    """
    candidate = (response or "").strip()
    if not candidate:
        candidate = "I couldn't produce a clear answer. Please rephrase your question."

    issues: List[str] = []
    contextual_question = str(reasoning_plan.get("contextual_question") or user_question)
    tools = reasoning_plan.get("tools", {}) or {}
    wants_web = bool(tools.get("web_search"))
    execution_profile = reasoning_plan.get("execution_profile", {}) or {}
    verification_mode = str(execution_profile.get("verification_mode", "none"))
    requires_source = bool(execution_profile.get("must_include_source", False))
    identity_like = bool(execution_profile.get("identity_like", False))
    source_domain_hints = execution_profile.get("source_domain_hints", []) or []

    if _response_indicates_no_answer_found(candidate) and wants_web and is_academic_or_policy_question(contextual_question):
        issues.append("no_answer")

    if verification_mode == "date":
        dates = _extract_dates_from_text(candidate)
        if not dates:
            issues.append("missing_date")
        else:
            term, target_year = _infer_term_year_from_context(contextual_question, conversation_history)
            parsed = _parse_date_candidate(dates[0], fallback_year=target_year or datetime.now().year)
            cql = contextual_question.lower()
            current_or_future_requested = any(t in cql for t in ["current semester", "this semester", "next semester", "upcoming"])
            if parsed and current_or_future_requested and parsed.date() < (datetime.now().date() - timedelta(days=120)):
                issues.append("stale_date")

    if reasoning_plan.get("intent") == "recommendation":
        horizon = int(reasoning_plan.get("semester_horizon", 1) or 1)
        if horizon >= 2 and "semester +2" not in candidate.lower():
            issues.append("missing_horizon")

    if requires_source:
        web_used = False
        if trace and trace.get("tools_used"):
            web_used = any(str(t.get("tool", "")).lower() == "duckduckgo_search" for t in trace.get("tools_used", []))
        if not web_used:
            issues.append("unverified_fresh_fact")
        if "source:" not in candidate.lower() and "unavailable" not in candidate.lower():
            issues.append("missing_source")
        elif "unavailable" not in candidate.lower() and not _response_has_allowed_source(candidate, source_domain_hints):
            issues.append("untrusted_source")
        m_name = re.search(r"appears to be\s+([^\n\.]+)", candidate, flags=re.IGNORECASE)
        if m_name:
            extracted_name = m_name.group(1).strip()
            if not _is_plausible_person_name(extracted_name):
                issues.append("low_confidence_name")

    _trace_quality_check(
        trace,
        "response_satisfies_plan",
        len(issues) == 0,
        {
            "issues": issues,
            "intent": reasoning_plan.get("intent"),
            "query_type": query_type,
        }
    )

    if not issues:
        return candidate

    if trace and trace.get("enabled"):
        trace["repairs"].append({
            "ts": datetime.now().isoformat(),
            "issues": issues,
            "attempt": "repair_pass_1",
        })

    # Repair path 1: web search backfill for missing/no-answer policy/course/major/chat contexts.
    if wants_web and is_academic_or_policy_question(contextual_question):
        if verification_mode == "date" and any(i in issues for i in ["missing_date", "stale_date", "no_answer"]):
            web = _web_search_and_synthesize(
                contextual_question,
                date_only=True,
                conversation_history=conversation_history,
                prefer_current_or_future=True,
                trace=trace
            )
            if web and web != "DATE_NOT_FOUND":
                _trace_event(trace, "observe", "Repair succeeded with date web search", {"issues": issues})
                return web
        if "no_answer" in issues:
            web = _web_search_and_synthesize(contextual_question, conversation_history=conversation_history, trace=trace)
            if web:
                _trace_event(trace, "observe", "Repair succeeded with generic web search", {"issues": issues})
                return web
        if any(i in issues for i in ["unverified_fresh_fact", "missing_source", "untrusted_source", "low_confidence_name"]):
            web = _web_search_and_synthesize(
                contextual_question,
                conversation_history=conversation_history,
                trace=trace,
                force_fresh_identity=identity_like
            )
            if web:
                _trace_event(trace, "observe", "Repair succeeded with fresh-fact verification web search", {"issues": issues})
                return web

    # Repair path 2: recommendation horizon mismatch fallback.
    if "missing_horizon" in issues and reasoning_plan.get("intent") == "recommendation":
        _trace_event(trace, "observe", "Applied recommendation horizon fallback", {"issues": issues})
        return candidate + "\n\nI can also provide Semester +2 explicitly if you want me to regenerate with a 2-semester horizon."

    return candidate


def _term_or_year_was_provided(text: str, conversation_history: Optional[List[dict]] = None) -> bool:
    term, year = _infer_term_year_from_context(text, conversation_history)
    q = (text or "").lower()
    has_explicit_term = any(t in q for t in ["spring", "summer", "fall", "winter", "current semester", "this semester", "next semester", "upcoming semester"])
    has_explicit_year = re.search(r"\b20\d{2}\b", q) is not None
    return has_explicit_term or has_explicit_year or (term is not None and year is not None)


def _format_deadline_answer(date_text: str, question_text: str, conversation_history: Optional[List[dict]] = None) -> str:
    """Return a short, contextual deadline answer with term/year."""
    term, year = _infer_term_year_from_context(question_text, conversation_history)
    term_label = f"{(term or 'requested term').title()} {year}".strip()
    parsed = _parse_date_candidate(date_text, fallback_year=year or datetime.now().year)
    normalized = parsed.strftime("%B %d, %Y") if parsed else date_text
    return f"Late-drop deadline ({term_label}): {normalized}."


def _extract_source_urls_from_response(response: str) -> List[str]:
    if not response:
        return []
    return re.findall(r"https?://[^\s)]+", response)


def _response_has_allowed_source(response: str, domain_hints: List[str]) -> bool:
    if not domain_hints:
        return True
    urls = _extract_source_urls_from_response(response)
    if not urls:
        return False
    return any(_is_domain_allowed(u, domain_hints) for u in urls)


def _resolved_search_to_metadata(result: Optional[ResolvedResearchAnswer]) -> Dict[str, Any]:
    if result is None:
        return {}
    return {
        "search_resolution": result.to_dict(),
    }


def _run_shared_search_pipeline(
    user_id: str,
    question_text: str,
    conversation_history: Optional[List[dict]],
    trace: Optional[Dict[str, Any]],
    preferred_strategy: Optional[str] = None,
    preferred_domains: Optional[List[str]] = None,
    required_domain_mode: Optional[str] = None,
) -> ResolvedResearchAnswer:
    return run_search_pipeline(
        user_id=user_id,
        user_question=question_text,
        corrected_question=question_text,
        conversation_history=conversation_history,
        llm=llm2,
        trace=trace,
        preferred_strategy=preferred_strategy,
        preferred_domains=preferred_domains,
        required_domain_mode=required_domain_mode,
    )


def _research_answer_text(result: Optional[ResolvedResearchAnswer]) -> Optional[str]:
    if result is None:
        return None
    answer = str(result.answer_text or "").strip()
    return answer or None

def track_collection_creation(collection_name: str, user_id: str):
    """Track when collection was created for TTL cleanup."""
    metadata = {}
    if os.path.exists(COLLECTION_METADATA_FILE):
        try:
            with open(COLLECTION_METADATA_FILE, 'r') as f:
                metadata = json.load(f)
        except:
            metadata = {}
    
    metadata[collection_name] = {
        "user_id": user_id,
        "created_at": datetime.now().isoformat()
    }
    
    with open(COLLECTION_METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)

def cleanup_old_collections(ttl_days=7):
    """Remove transcript collections older than TTL."""
    if not os.path.exists(COLLECTION_METADATA_FILE):
        return
    
    try:
        with open(COLLECTION_METADATA_FILE, 'r') as f:
            metadata = json.load(f)
    except:
        return
    
    cutoff = datetime.now() - timedelta(days=ttl_days)
    to_remove = []
    
    for collection_name, info in metadata.items():
        try:
            created_at = datetime.fromisoformat(info['created_at'])
            if created_at < cutoff:
                to_remove.append(collection_name)
                # Delete collection directory
                collection_path = os.path.join(CHROMA_DATA_DIR, "transcripts", collection_name)
                if os.path.exists(collection_path):
                    shutil.rmtree(collection_path)
        except:
            continue
    
    # Update metadata
    for name in to_remove:
        del metadata[name]
    
    with open(COLLECTION_METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    if to_remove:
        print(f"🧹 Cleaned up {len(to_remove)} old collections")


def _is_debug_enabled(flag: Any) -> bool:
    if isinstance(flag, bool):
        return flag
    if isinstance(flag, str):
        return flag.strip().lower() in {"1", "true", "yes", "on"}
    return bool(flag)


def _new_execution_trace(user_id: str, user_question: str, enabled: bool) -> Dict[str, Any]:
    return {
        "enabled": enabled,
        "user_id": user_id,
        "user_question": user_question,
        "started_at": datetime.now().isoformat(),
        "plan": {},
        "orchestrator": {},
        "executor": {},
        "finalizer": {},
        "search_pipeline": {
            "context": {},
            "plan": {},
            "rounds": [],
            "sources_considered": [],
            "sources_used": [],
            "exhaustion_reason": None,
            "runs": [],
        },
        "events": [],
        "tools_used": [],
        "quality_checks": [],
        "repairs": [],
        "result_preview": "",
        "finished_at": None,
    }


def _trace_event(trace: Optional[Dict[str, Any]], stage: str, message: str, data: Optional[Dict[str, Any]] = None):
    if not trace or not trace.get("enabled"):
        return
    trace["events"].append({
        "ts": datetime.now().isoformat(),
        "stage": stage,
        "message": message,
        "data": data or {},
    })


def _trace_tool(
    trace: Optional[Dict[str, Any]],
    tool_name: str,
    purpose: str,
    status: str = "used",
    metadata: Optional[Dict[str, Any]] = None,
):
    if not trace or not trace.get("enabled"):
        return
    trace["tools_used"].append({
        "ts": datetime.now().isoformat(),
        "tool": tool_name,
        "purpose": purpose,
        "status": status,
        "metadata": metadata or {},
    })


def _trace_quality_check(
    trace: Optional[Dict[str, Any]],
    check_name: str,
    passed: bool,
    details: Optional[Dict[str, Any]] = None,
):
    if not trace or not trace.get("enabled"):
        return
    trace["quality_checks"].append({
        "ts": datetime.now().isoformat(),
        "check": check_name,
        "passed": passed,
        "details": details or {},
    })


def get_execution_trace_for_user(user_id: str) -> Optional[Dict[str, Any]]:
    return execution_trace_store.get(user_id)


def clear_search_context_for_user(user_id: str):
    clear_active_search_context(user_id)


def reset_uploaded_transcript_state() -> None:
    """
    Drop any uploaded-transcript state held at module scope so subsequent queries
    do not reuse a previous upload's retriever or extracted student data.

    This is what "clear chat" needs in addition to wiping conversation history —
    otherwise the agent still sees the last student's transcript even after the
    UI has marked the document as removed.
    """
    global pdf_retriever, stored_pdf_docs, latest_student_data, has_transcript
    pdf_retriever = None
    stored_pdf_docs = None
    latest_student_data = None
    has_transcript = False


class CachedEmbeddings:
    """Wrapper around OpenAIEmbeddings that caches results."""
    
    def __init__(self, embedding_fn, cache_dir=None):
        self.embedding_fn = embedding_fn
        if cache_dir is None:
            cache_dir = os.path.join(CHROMA_DATA_DIR, "embedding_cache")
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_path(self, text: str) -> str:
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{text_hash}.pkl")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents with caching."""
        embeddings = []
        texts_to_embed = []
        indices_to_embed = []
        
        for i, text in enumerate(texts):
            cache_path = self._get_cache_path(text)
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'rb') as f:
                        embeddings.append((i, pickle.load(f)))
                except:
                    texts_to_embed.append(text)
                    indices_to_embed.append(i)
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)
        
        # Embed uncached texts
        if texts_to_embed:
            new_embeddings = self.embedding_fn.embed_documents(texts_to_embed)
            for idx, text, emb in zip(indices_to_embed, texts_to_embed, new_embeddings):
                cache_path = self._get_cache_path(text)
                try:
                    with open(cache_path, 'wb') as f:
                        pickle.dump(emb, f)
                except:
                    pass
                embeddings.append((idx, emb))
        
        # Return in original order
        return [emb for _, emb in sorted(embeddings)]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query with caching."""
        cache_path = self._get_cache_path(text)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        
        embedding = self.embedding_fn.embed_query(text)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
        except:
            pass
        return embedding

# =========================
# Environment / Setup
# =========================

def setup_environment(whatIfUpload=None, user_id='default'):
    """
    Initialize environment, LLMs, vectorstores for PDF + courses.
    If whatIfUpload is None or has no 'transcript', run in no-transcript mode.
    """
    global pdf_retriever, course_retriever, majors_retriever, llm1, llm2, llm_classifier, has_transcript, stored_pdf_docs, latest_student_data

    has_transcript = False  # reset flag each call
    
    # Ensure Chroma data directory exists
    os.makedirs(CHROMA_DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(CHROMA_DATA_DIR, "courses"), exist_ok=True)
    os.makedirs(os.path.join(CHROMA_DATA_DIR, "transcripts"), exist_ok=True)

    #  Environment Variables (set via .env or system env)
    if is_gemini_active() or not os.getenv('LANGCHAIN_API_KEY'):
        os.environ['LANGCHAIN_TRACING_V2'] = 'false'
    else:
        os.environ.setdefault('LANGCHAIN_TRACING_V2', 'true')
        os.environ.setdefault('LANGCHAIN_ENDPOINT', 'https://api.smith.langchain.com')
        _lc_project = os.getenv('LANGCHAIN_PROJECT', 'iaas-advisor')
        os.environ['LANGCHAIN_PROJECT'] = _lc_project

    if not os.getenv('OPENAI_API_KEY') and not is_gemini_active():
        raise RuntimeError(
            "Neither OPENAI_API_KEY nor GEMINI_API_KEY is set. Add it to .env or set the environment variable. "
            "See .env.example for the format."
        )
    if not is_gemini_active() and not os.getenv('LANGCHAIN_API_KEY'):
        print("⚠️ LANGCHAIN_API_KEY is not set (optional, for LangSmith tracing).")

    # ---- Optional transcript / What-If report ----
    if whatIfUpload is not None and 'transcript' in whatIfUpload.files:
        transcript_file = whatIfUpload.files['transcript']
        filename = secure_filename(transcript_file.filename)
        temp_pdf_path = f"/tmp/{filename}"
        transcript_file.save(temp_pdf_path)

        # Generate document hash
        doc_hash = get_document_hash(temp_pdf_path)
        collection_name = get_transcript_collection_name(user_id, doc_hash)
        collection_path = os.path.join(CHROMA_DATA_DIR, "transcripts", collection_name)
        
        # Check if collection exists (cached)
        if os.path.exists(collection_path):
            try:
                cached_embeddings = CachedEmbeddings(create_embeddings_model())
                vectorstore_pdf = Chroma(
                    collection_name=collection_name,
                    persist_directory=os.path.join(CHROMA_DATA_DIR, "transcripts"),
                    embedding_function=cached_embeddings
                )
                pdf_retriever = vectorstore_pdf.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 10, "fetch_k": 40, "lambda_mult": 0.5}
                )
                # For cached transcripts, reload PDF for deterministic extraction
                loader = PyPDFLoader(temp_pdf_path)
                stored_pdf_docs = redact_student_info(loader.load())
                try:
                    latest_student_data = extract_student_data_deterministic(stored_pdf_docs)
                except Exception as e:
                    print(f"⚠️ Failed to cache student data from cached transcript: {e}")
                has_transcript = True
                print(f"✅ Using cached transcript: {collection_name}")
            except Exception as e:
                print(f"⚠️ Error loading cached transcript: {e}, recreating...")
                # Fall through to create new
                vectorstore_pdf = None
        else:
            vectorstore_pdf = None
        
        # Create new collection if not cached
        if vectorstore_pdf is None:
            loader = PyPDFLoader(temp_pdf_path)
            pdf_docs = redact_student_info(loader.load())
            stored_pdf_docs = pdf_docs  # Store for deterministic extraction
            try:
                latest_student_data = extract_student_data_deterministic(pdf_docs)
            except Exception as e:
                print(f"⚠️ Failed to cache student data on upload: {e}")
            pdf_splits = chunk_with_metadata(pdf_docs)

            # Use cached embeddings
            cached_embeddings = CachedEmbeddings(create_embeddings_model())
            vectorstore_pdf = Chroma.from_documents(
                pdf_splits,
                collection_name=collection_name,
                persist_directory=os.path.join(CHROMA_DATA_DIR, "transcripts"),
                embedding=cached_embeddings
            )

            # Track collection creation for TTL cleanup
            track_collection_creation(collection_name, user_id)

            pdf_retriever = vectorstore_pdf.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 10, "fetch_k": 40, "lambda_mult": 0.5}
            )
            print(f"✅ Transcript loaded and vectorstore built: {collection_name}")
        has_transcript = True
    else:
        pdf_retriever = None
        stored_pdf_docs = None
        latest_student_data = None
        has_transcript = False
        print("ℹ️ No transcript provided. Running in no-transcript mode.")

    # ---- Load course-level data (processed_psu_courses.csv) for individual course queries ----
    courses_csv_path = os.path.join(os.path.dirname(__file__), "processed_psu_courses.csv")
    if not os.path.exists(courses_csv_path):
        alt_path = os.path.join(os.path.dirname(__file__), "..", "LLM", "processed_psu_courses.csv")
        if os.path.exists(alt_path):
            courses_csv_path = alt_path
        else:
            print("⚠️ processed_psu_courses.csv not found")
            courses_csv_path = None
    
    if courses_csv_path and os.path.exists(courses_csv_path):
        df_courses = pd.read_csv(courses_csv_path)
        catalog_codes = [str(c) for c in df_courses["course_number"].dropna().tolist()]
        catalog_index = build_course_catalog_index(catalog_codes)
        course_texts = []
        course_metadata = []

        for _, row in df_courses.iterrows():
            course_text = f"Course: {row.get('course_number', '')}. "
            course_text += f"Title: {row.get('course_name', '')}. "
            course_text += f"Description: {row.get('description', '')}. "
            prereq_summary = format_course_prerequisite_summary(row, catalog_codes=catalog_index)
            if prereq_summary:
                course_text += f"{prereq_summary} "
            course_texts.append(course_text)
            
            course_code = str(row.get('course_number', '')).strip()
            course_metadata.append({
                "course_code": course_code,
                "department": course_code.split()[0] if ' ' in course_code else ''
            })
        
        chroma_courses_dir = os.path.join(CHROMA_DATA_DIR, "courses")
        course_fingerprint_path = os.path.join(chroma_courses_dir, ".source_fingerprint")
        course_source_fingerprint = hashlib.sha256(
            f"{COURSE_RETRIEVER_FORMAT_VERSION}:{get_document_hash(courses_csv_path)}".encode("utf-8")
        ).hexdigest()
        if os.path.exists(chroma_courses_dir):
            existing_fingerprint = ""
            if os.path.exists(course_fingerprint_path):
                with open(course_fingerprint_path, "r", encoding="utf-8") as _fp:
                    existing_fingerprint = _fp.read().strip()
            if existing_fingerprint != course_source_fingerprint:
                shutil.rmtree(chroma_courses_dir)

        def _write_course_fingerprint():
            os.makedirs(chroma_courses_dir, exist_ok=True)
            with open(course_fingerprint_path, "w", encoding="utf-8") as _fp:
                _fp.write(course_source_fingerprint)

        try:
            if os.path.exists(chroma_courses_dir):
                try:
                    cached_embeddings = CachedEmbeddings(create_embeddings_model())
                    vectorstore_courses = Chroma(
                        persist_directory=chroma_courses_dir,
                        embedding_function=cached_embeddings
                    )
                    _ = vectorstore_courses.as_retriever(search_kwargs={"k": 1})
                    print("✅ Using existing course-level vectorstore.")
                except Exception as e:
                    print(f"⚠️ Course vectorstore corrupted, recreating... Error: {e}")
                    if os.path.exists(chroma_courses_dir):
                        shutil.rmtree(chroma_courses_dir)
                    cached_embeddings = CachedEmbeddings(create_embeddings_model())
                    vectorstore_courses = Chroma.from_texts(
                        texts=course_texts,
                        embedding=cached_embeddings,
                        metadatas=course_metadata,
                        persist_directory=chroma_courses_dir
                    )
                    vectorstore_courses.persist()
                    _write_course_fingerprint()
                    print("✅ Created new course-level vectorstore.")
            else:
                cached_embeddings = CachedEmbeddings(create_embeddings_model())
                vectorstore_courses = Chroma.from_texts(
                    texts=course_texts,
                    embedding=cached_embeddings,
                    metadatas=course_metadata,
                    persist_directory=chroma_courses_dir
                )
                vectorstore_courses.persist()
                _write_course_fingerprint()
                print("✅ Created new course-level vectorstore.")
        except Exception as e:
            print(f"⚠️ Error with course vectorstore: {e}")
            if os.path.exists(chroma_courses_dir):
                shutil.rmtree(chroma_courses_dir)
            cached_embeddings = CachedEmbeddings(create_embeddings_model())
            vectorstore_courses = Chroma.from_texts(
                texts=course_texts,
                embedding=cached_embeddings,
                metadatas=course_metadata,
                persist_directory=chroma_courses_dir
            )
            vectorstore_courses.persist()
            _write_course_fingerprint()

        course_retriever = vectorstore_courses.as_retriever(search_kwargs={"k": 5})
        print("✅ Course-level vectorstore initialized.")
    else:
        course_retriever = None
        print("⚠️ Course-level retriever not available (processed_psu_courses.csv missing)")

    # ---- Load majors CSV & build Chroma vectorstore for major requirements queries ----
    majors_csv_path = os.path.join(os.path.dirname(__file__), "all_abington_majors_combined(5).csv")
    if not os.path.exists(majors_csv_path):
        alt_path = os.path.join(os.path.dirname(__file__), "..", "LLM", "all_abington_majors_combined(5).csv")
        if os.path.exists(alt_path):
            majors_csv_path = alt_path
        else:
            print("⚠️ all_abington_majors_combined(5).csv not found")
            majors_csv_path = None
    
    if majors_csv_path and os.path.exists(majors_csv_path):
        df_majors = pd.read_csv(majors_csv_path)
        majors_texts = []
        majors_metadata = []
        
        for _, row in df_majors.iterrows():
            text_data = ""
            metadata = {}
            for col in df_majors.columns:
                value = str(row[col]) if not pd.isna(row[col]) else ""
                text_data += f"{col}: {value}. "
                metadata[col] = str(value) if value else ""
            majors_texts.append(text_data.strip())
            majors_metadata.append(metadata)
        
        chroma_majors_dir = os.path.join(CHROMA_DATA_DIR, "majors")
        try:
            if os.path.exists(chroma_majors_dir):
                try:
                    cached_embeddings = CachedEmbeddings(create_embeddings_model())
                    vectorstore_majors = Chroma(
                        persist_directory=chroma_majors_dir,
                        embedding_function=cached_embeddings
                    )
                    _ = vectorstore_majors.as_retriever(search_kwargs={"k": 1})
                    print("✅ Using existing majors vectorstore.")
                except Exception as e:
                    print(f"⚠️ Majors vectorstore corrupted, recreating... Error: {e}")
                    if os.path.exists(chroma_majors_dir):
                        shutil.rmtree(chroma_majors_dir)
                    cached_embeddings = CachedEmbeddings(create_embeddings_model())
                    vectorstore_majors = Chroma.from_texts(
                        texts=majors_texts,
                        embedding=cached_embeddings,
                        metadatas=majors_metadata,
                        persist_directory=chroma_majors_dir
                    )
                    vectorstore_majors.persist()
                    print("✅ Created new majors vectorstore.")
            else:
                cached_embeddings = CachedEmbeddings(create_embeddings_model())
                vectorstore_majors = Chroma.from_texts(
                    texts=majors_texts,
                    embedding=cached_embeddings,
                    metadatas=majors_metadata,
                    persist_directory=chroma_majors_dir
                )
                vectorstore_majors.persist()
                print("✅ Created new majors vectorstore.")
        except Exception as e:
            print(f"⚠️ Error with majors vectorstore: {e}")
            if os.path.exists(chroma_majors_dir):
                shutil.rmtree(chroma_majors_dir)
            cached_embeddings = CachedEmbeddings(create_embeddings_model())
            vectorstore_majors = Chroma.from_texts(
                texts=majors_texts,
                embedding=cached_embeddings,
                metadatas=majors_metadata,
                persist_directory=chroma_majors_dir
            )
            vectorstore_majors.persist()
        
        majors_retriever = vectorstore_majors.as_retriever(search_kwargs={"k": 3})
        print("✅ Majors vectorstore initialized.")
    else:
        majors_retriever = None
        print("⚠️ Majors retriever not available (all_abington_majors_combined(5).csv missing)")

    # ---- Load advising/registrar policy PDF into vectorstore ----
    global policy_retriever, policy_vectorstore
    policy_pdf_path = os.path.join(LLM_DIR, "PennState_Combined_Advising_and_Registrar_VERBATIM.pdf")
    if os.path.exists(policy_pdf_path):
        chroma_policy_dir = os.path.join(CHROMA_DATA_DIR, "policies")
        policy_collection = "advising_registrar_policies_v1"
        try:
            if os.path.exists(chroma_policy_dir):
                try:
                    cached_embeddings = CachedEmbeddings(create_embeddings_model())
                    policy_vectorstore = Chroma(
                        collection_name=policy_collection,
                        persist_directory=chroma_policy_dir,
                        embedding_function=cached_embeddings
                    )
                    _ = policy_vectorstore.as_retriever(search_kwargs={"k": 1})
                    print("✅ Using existing policy vectorstore.")
                except Exception as e:
                    print(f"⚠️ Policy vectorstore corrupted, recreating... Error: {e}")
                    if os.path.exists(chroma_policy_dir):
                        shutil.rmtree(chroma_policy_dir)
                    cached_embeddings = CachedEmbeddings(create_embeddings_model())
                    loader = PyPDFLoader(policy_pdf_path)
                    policy_docs = loader.load()
                    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
                    policy_chunks = splitter.split_documents(policy_docs)
                    for chunk in policy_chunks:
                        chunk.metadata["source"] = "advising_registrar_policy"
                    policy_vectorstore = Chroma.from_documents(
                        policy_chunks,
                        collection_name=policy_collection,
                        persist_directory=chroma_policy_dir,
                        embedding=cached_embeddings
                    )
                    policy_vectorstore.persist()
                    print("✅ Created new policy vectorstore.")
            else:
                cached_embeddings = CachedEmbeddings(create_embeddings_model())
                loader = PyPDFLoader(policy_pdf_path)
                policy_docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
                policy_chunks = splitter.split_documents(policy_docs)
                for chunk in policy_chunks:
                    chunk.metadata["source"] = "advising_registrar_policy"
                policy_vectorstore = Chroma.from_documents(
                    policy_chunks,
                    collection_name=policy_collection,
                    persist_directory=chroma_policy_dir,
                    embedding=cached_embeddings
                )
                policy_vectorstore.persist()
                print("✅ Created new policy vectorstore.")
        except Exception as e:
            print(f"⚠️ Error with policy vectorstore: {e}")
            policy_vectorstore = None
        policy_retriever = policy_vectorstore.as_retriever(search_kwargs={"k": 4}) if policy_vectorstore else None
        if policy_retriever:
            print("✅ Policy retriever initialized.")
    else:
        policy_retriever = None
        print("⚠️ Policy PDF not found for RAG.")

    # ---- LLM Instances ----
    global llm1, llm2, llm_classifier
    llm1 = create_chat_model(PRIMARY_MODEL_NAME, temperature=0)
    llm2 = create_chat_model(PRIMARY_MODEL_NAME, temperature=0)
    llm_classifier = create_chat_model(CLASSIFIER_MODEL_NAME, temperature=0)
    print("✅ LLMs initialized.")


# =========================
# Helper Functions
# =========================

def redact_student_info(docs):
    """
    Roughly redact student name and ID from the PDF text.
    """
    redacted_docs = []

    for doc in docs:
        content = doc.page_content

        # Mask patterns like "Name: John Doe 123456789"
        content = re.sub(
            r"\b[A-Z][a-z]+\s[A-Z][a-z]+(?=:\s?\d{6,})",
            "[REDACTED NAME]",
            content
        )

        # Mask 9-digit ID
        content = re.sub(r"\b\d{9}\b", "[REDACTED ID]", content)

        # Optionally mask first occurrence of a "First Last" pattern
        content = re.sub(
            r"\b[A-Z][a-z]+\s[A-Z][a-z]+\b",
            "[REDACTED NAME]",
            content,
            count=1
        )

        doc.page_content = content
        redacted_docs.append(doc)

    return redacted_docs


def extract_completed_and_ip_courses(text):
    pattern = re.findall(r"\b[A-Z]{2,}\s+[0-9]{3}\b", text)
    courses = {re.sub(r"\s+", " ", course.strip()) for course in pattern}
    return courses


def extract_courses_from_term_lines(full_text: str) -> Tuple[List[str], List[str]]:
    """
    Extract completed and in-progress courses from term-based lines like:
    "FA 2024 CMPSC 131 ... 3.00 A" or "... 3.00 IP"
    """
    term_pattern = re.compile(r"\b(FA|SP|SU)\s+\d{4}\s+([A-Z]{2,})\s+(\d{1,3}[A-Z]?)\b")
    grade_pattern = re.compile(r"\b(A\+|A-|A|B\+|B-|B|C\+|C-|C|D\+|D-|D|F|IP|W|WF|LD)\b")

    completed = []
    in_progress = []
    completed_seen = set()
    in_progress_seen = set()

    for match in term_pattern.finditer(full_text):
        course_code = f"{match.group(2)} {match.group(3)}"
        context = full_text[match.end():match.end() + 80]
        grade_match = grade_pattern.search(context)
        grade = grade_match.group(1) if grade_match else ""

        if grade == "IP":
            if course_code not in in_progress_seen:
                in_progress.append(course_code)
                in_progress_seen.add(course_code)
            continue
        if grade in {"W", "WF", "LD", "F"}:
            continue
        if course_code not in completed_seen:
            completed.append(course_code)
            completed_seen.add(course_code)

    return completed, in_progress


def extract_total_credits_from_text(full_text: str) -> Optional[Tuple[float, float, float]]:
    """
    Extract total required/used/needed credits from summary lines.
    Prefer the line with the largest required credits (e.g., 120.00).
    """
    pattern = re.compile(
        r"Units:\s*([\d.]+)\s*required,\s*([\d.]+)\s*used,\s*([\d.]+)\s*needed",
        re.IGNORECASE
    )
    matches = []
    for match in pattern.finditer(full_text):
        try:
            required = float(match.group(1))
            used = float(match.group(2))
            needed = float(match.group(3))
            matches.append((required, used, needed))
        except Exception:
            continue

    if not matches:
        return None

    required, used, needed = max(matches, key=lambda t: (t[0], t[2]))
    return required, used, needed


def extract_student_data_deterministic(pdf_docs: List[Document]) -> dict:
    """
    Extract student data from full transcript text using deterministic parsing.
    More accurate and faster than RAG for structured data.
    """
    full_text = "\n".join(doc.page_content for doc in pdf_docs)
    
    # Extract major - try multiple patterns
    major = None
    
    # Pattern 1: "Major/Option: Computer Science, B.S. (Abington) Req Term: ..."
    # Stop at "Req Term" or end of line
    major_match = re.search(r'Major/Option[:\s]+([^\n]+?)(?:\s+Req\s+Term|$)', full_text, re.IGNORECASE)
    if major_match:
        major = major_match.group(1).strip()
        # Clean up - remove "Req Term" and anything after it
        major = re.sub(r'\s+Req\s+Term.*$', '', major, flags=re.IGNORECASE).strip()
        # Remove trailing punctuation/whitespace
        major = major.rstrip('.,;:')
    
    # Pattern 2: Look for major name with degree type pattern: "Name, B.S. (Location)"
    if not major or len(major) < 5 or 'Req Term' in major:
        # Look for pattern like "Computer Science, B.S. (Abington)"
        degree_pattern = r'([A-Z][a-zA-Z\s&]+(?:,\s*[A-Z][a-zA-Z\s&]+)*)\s*,\s*B\.?[AS]\.?\s*\([^)]+\)'
        match = re.search(degree_pattern, full_text)
        if match:
            major = match.group(1).strip()
            # Try to get the full string including degree
            full_match = re.search(rf'{re.escape(major)}[^\n]*B\.?[AS]\.?[^\n]*\([^)]+\)', full_text, re.IGNORECASE)
            if full_match:
                major = full_match.group(0).strip()
    
    # Pattern 3: If still not found, look for common major keywords
    if not major or len(major) < 5:
        # Look for major names near "Major/Option" label
        major_section = re.search(r'Major/Option[:\s]+([^\n]{10,100})', full_text, re.IGNORECASE)
        if major_section:
            section_text = major_section.group(1)
            # Try to extract just the major name part (before "Req Term" or other labels)
            clean_major = re.sub(r'\s+Req\s+Term.*$', '', section_text, flags=re.IGNORECASE).strip()
            if clean_major and len(clean_major) > 5:
                major = clean_major.split(',')[0].strip()  # Take first part before comma if exists
    
    # Extract completed and in-progress courses (term-based lines)
    completed_courses, in_progress = extract_courses_from_term_lines(full_text)

    # Extract total credits from summary lines
    totals = extract_total_credits_from_text(full_text)
    if totals:
        credits_required, credits_completed, credits_needed = totals
    else:
        credits_match = re.search(
            r'(\d+\.\d+)\s+units?\s+required.*?(\d+\.\d+)\s+used',
            full_text,
            re.IGNORECASE
        )
        credits_completed = float(credits_match.group(2)) if credits_match else 0.0
        credits_required = float(credits_match.group(1)) if credits_match else 120.0
        credits_needed = max(0.0, credits_required - credits_completed)
    
    # Extract unmet requirements (deterministic parsing)
    unmet_requirements = extract_not_satisfied_sections_deterministic(full_text)
    
    # Normalize major name if extracted
    if major:
        major = normalize_major_name(major)
    
    return {
        "major": major,
        "completed_courses": completed_courses,
        "in_progress_courses": in_progress,
        "credits_completed": credits_completed,
        "credits_required": credits_required,
        "credits_needed": credits_needed,
        "unmet_requirements": unmet_requirements
    }


def _clean_major_phrase(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    stopwords = {
        "major", "degree", "program", "student", "studying", "study", "in", "of",
        "ba", "bs", "b", "a", "s", "abington", "penn", "state", "university",
        "freshman", "sophomore", "junior", "senior", "first", "second", "third", "fourth", "year"
    }
    tokens = [t for t in text.split() if t not in stopwords and len(t) > 2]
    return " ".join(tokens)


def normalize_major_name(extracted_major: str) -> str:
    """
    Normalize extracted major name to match CSV format using fuzzy matching.
    """
    if not extracted_major or len(extracted_major.strip()) < 3:
        return None
    
    user_major = extracted_major.strip()
    user_clean = _clean_major_phrase(user_major)
    user_tokens = {w.lower() for w in re.split(r"[^A-Za-z]+", user_clean) if len(w) > 2}
    if not user_tokens and not user_clean:
        return None

    # Load majors CSV to get available majors
    majors_csv_path = os.path.join(os.path.dirname(__file__), "all_abington_majors_combined(5).csv")
    if not os.path.exists(majors_csv_path):
        alt_path = os.path.join(os.path.dirname(__file__), "..", "LLM", "all_abington_majors_combined(5).csv")
        if os.path.exists(alt_path):
            majors_csv_path = alt_path
        else:
            return extracted_major  # Return as-is if CSV not found
    
    try:
        df_majors = pd.read_csv(majors_csv_path)
        available_majors = df_majors['Major Name'].tolist()
        cleaned_majors = {major: _clean_major_phrase(major) for major in available_majors}
        
        # First, try exact match (case-insensitive)
        for major in available_majors:
            if user_major.lower() == major.lower().strip():
                return major

        # Cleaned exact match (removes degree/abington/etc.)
        for major, cleaned in cleaned_majors.items():
            if user_clean and user_clean == cleaned:
                return major

        # Token subset match: require at least 2 overlapping tokens
        best_token_match = None
        best_token_count = 0
        for major in available_majors:
            major_tokens = {w.lower() for w in re.split(r"[^A-Za-z]+", cleaned_majors.get(major, "")) if len(w) > 2}
            overlap = user_tokens.intersection(major_tokens) if user_tokens else set()
            if len(overlap) >= 2 and len(overlap) > best_token_count:
                best_token_match = major
                best_token_count = len(overlap)
        if best_token_match:
            return best_token_match

        # Single-token match for common majors (e.g., "business")
        if len(user_tokens) == 1:
            token = next(iter(user_tokens))
            best_single = None
            for major, cleaned in cleaned_majors.items():
                if token in cleaned:
                    # Prefer the shortest cleaned major containing the token
                    if not best_single or len(cleaned) < len(cleaned_majors.get(best_single, "")):
                        best_single = major
            if best_single:
                return best_single

        # Substring match (either direction)
        for major in available_majors:
            cleaned = cleaned_majors.get(major, "")
            if user_clean and (user_clean in cleaned or cleaned in user_clean):
                return major

        # Try fuzzy matching with higher threshold to avoid wrong majors
        try:
            from fuzzywuzzy import process
            best_match = process.extractOne(user_clean or user_major, available_majors)
            if best_match and best_match[1] >= 70:  # slightly more forgiving
                return best_match[0]
        except ImportError:
            try:
                import difflib
                candidates = difflib.get_close_matches(user_clean or user_major, available_majors, n=1, cutoff=0.6)
                if candidates:
                    return candidates[0]
            except Exception:
                pass

        # If nothing matches, return None to force the caller to ask for clarification
        return None
    except Exception as e:
        print(f"⚠️ Error normalizing major name: {e}")
        return None


def extract_not_satisfied_sections_deterministic(full_text: str) -> List[dict]:
    """Extract unmet requirements using deterministic parsing (faster than RAG)."""
    unmet = []
    
    # Find all "Not Satisfied" sections
    not_satisfied_pattern = r'Not Satisfied.*?Units:\s*([\d.]+)\s*required.*?([\d.]+)\s*used.*?([\d.]+)\s*needed'
    for match in re.finditer(not_satisfied_pattern, full_text, re.DOTALL | re.IGNORECASE):
        try:
            unmet.append({
                "category": "Requirement",
                "required": float(match.group(1)),
                "used": float(match.group(2)),
                "needed": float(match.group(3))
            })
        except:
            continue
    
    return unmet


def extract_not_satisfied_sections(docs):
    """
    Parse What-If report text to find 'Not Satisfied' / unmet requirement blocks.
    Expects docs to be a list of Documents (e.g., from pdf_retriever.invoke()).
    """
    all_text = "\n".join(doc.page_content for doc in docs)
    print(f"🔍 Analyzing text sample: {all_text[:500]}...")

    # Pattern 1: Standard "Not Satisfied" sections
    pattern1 = re.compile(
        r"(.*?)Not Satisfied.*?Units:\s*([\d.]+)\s*required,\s*([\d.]+)\s*used(?:,\s*([\d.]+)\s*needed)?",
        re.DOTALL | re.IGNORECASE
    )
    results1 = pattern1.findall(all_text)
    print(f"🔍 Pattern 1 matches: {len(results1)}")

    # Pattern 2: Course tables with "needed" credits
    pattern2 = re.compile(
        r"(Units:\s*[\d.]+\s*required,\s*[\d.]+\s*used,\s*([\d.]+)\s*needed)",
        re.IGNORECASE
    )
    results2 = pattern2.findall(all_text)
    print(f"🔍 Pattern 2 matches: {len(results2)}")

    # Pattern 3: More flexible pattern for course tables
    pattern3 = re.compile(
        r"(Units:\s*([\d.]+)\s*required,\s*([\d.]+)\s*used,\s*([\d.]+)\s*needed)",
        re.IGNORECASE
    )
    results3 = pattern3.findall(all_text)
    print(f"🔍 Pattern 3 matches: {len(results3)}")

    extracted = []

    # Process Pattern 1: Not Satisfied sections
    for match in results1:
        category, required, used, needed = match
        extracted.append({
            "category": category.strip(),
            "required": float(required),
            "used": float(used),
            "needed": float(needed) if needed else round(float(required) - float(used), 2),
            "type": "not_satisfied"
        })

    # Process Pattern 2 & 3 only if near "Not Satisfied" context
    for m in pattern2.finditer(all_text):
        full_match = m.group(1)
        needed = m.group(2)
        start_idx = m.start()
        context_start = max(0, start_idx - 500)
        context_text = all_text[context_start:start_idx]
        if re.search(r"Not\s+Satisfied|Complete\s+the\s+following|Unmet|Needed", context_text, re.IGNORECASE):
            if float(needed) > 0:
                extracted.append({
                    "category": "Course Requirements Table",
                    "required": 0,
                    "used": 0,
                    "needed": float(needed),
                    "type": "course_table",
                    "context": full_match.strip()
                })

    for m in pattern3.finditer(all_text):
        full_match = m.group(1)
        required = m.group(2)
        used = m.group(3)
        needed = m.group(4)
        start_idx = m.start()
        context_start = max(0, start_idx - 500)
        context_text = all_text[context_start:start_idx]
        if re.search(r"Not\s+Satisfied|Complete\s+the\s+following|Unmet|Needed", context_text, re.IGNORECASE):
            if float(needed) > 0:
                extracted.append({
                    "category": "Course Requirements Table (Detailed)",
                    "required": float(required),
                    "used": float(used),
                    "needed": float(needed),
                    "type": "course_table_detailed",
                    "context": full_match.strip()
                })

    return extracted


# Valid intent labels from LLM classifier (maps to route types)
INTENT_TO_ROUTE = {
    "transcript_review": "transcript",
    "graduation_plan": "transcript",
    "policy": "policy",
    "course_info": "course",
    "major_requirements": "major",
    "general_chat": "chat",
}


def _classify_intent_llm(
    user_question: str,
    has_transcript: bool,
    conversation_history: Optional[List[dict]] = None,
) -> Optional[str]:
    """
    Use LLM to classify user intent when keyword routing is ambiguous.
    Returns route type: 'course', 'major', 'transcript', 'policy', or 'chat'
    """
    global llm_classifier
    if llm_classifier is None:
        try:
            from langchain_openai import ChatOpenAI
            llm_classifier = ChatOpenAI(model_name=CLASSIFIER_MODEL_NAME, temperature=0)
        except Exception:
            return None

    recent_context = ""
    if conversation_history and len(conversation_history) > 0:
        recent = conversation_history[-2:]
        recent_context = "\nRecent messages: " + " | ".join(
            f"{m.get('role', '')}: {str(m.get('content', ''))[:80]}..." for m in recent
        )

    system_prompt = (
        "You are an intent classifier for a Penn State Abington advising assistant. "
        "Classify the student's message into exactly one routing label.\n\n"
        + PRIVATE_REASONING_GUIDANCE
        + "\n\nOutput contract:\n"
        "- Return ONLY one lowercase label from this list: transcript_review, graduation_plan, policy, "
        "course_info, major_requirements, general_chat.\n"
        "- Do not return JSON, markdown, punctuation, confidence, or explanation.\n\n"
        "Label definitions with examples:\n"
        "- transcript_review: The student wants their academic record reviewed or audited. "
        "Examples: 'audit my requirements', 'review my transcript', 'where do I stand?'.\n"
        "- graduation_plan: The student wants a semester-by-semester graduation plan or course schedule. "
        "Examples: 'make me a graduation plan', 'what should I take next semester?'.\n"
        "- policy: The student asks about institutional rules, deadlines, or processes. "
        "Examples: 'late drop deadline', 'how do I audit a course?', 'withdraw policy'.\n"
        "- course_info: The student asks about a specific course's prerequisites, title, description, or role. "
        "Examples: 'what are CMPSC 488 prereqs?', 'what is ENGL 15?'.\n"
        "- major_requirements: The student asks what a major requires in general, not their personal record. "
        "Examples: 'what courses are required for Computer Science?', 'Business major prescribed courses'.\n"
        "- general_chat: Anything else or unclear intent. Example: 'thanks', 'can you explain that?'."
    )
    prompt = f"""Student has transcript uploaded: {has_transcript}
{recent_context}

Question: "{user_question}"

Return one label:"""

    try:
        result = llm_classifier.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ]).content.strip().lower()
        # Extract first valid intent if LLM added extra text
        for intent in INTENT_TO_ROUTE:
            if intent in result:
                return INTENT_TO_ROUTE[intent]
        return None
    except Exception as e:
        print(f"⚠️ Intent classification failed: {e}")
        return None


def _route_query_deterministic_fallback(
    user_question: str,
    has_transcript: bool = False,
    conversation_history: Optional[List[dict]] = None,
) -> str:
    """Last-resort route when the LLM classifier is unavailable or invalid."""
    question_lower = user_question.lower()

    # Course-code detection is only a fallback; the LLM route owns normal flow.
    if re.search(r'\b[A-Z]{2,}\s+\d{3}[A-Z]?\b', user_question, re.IGNORECASE):
        return 'course'

    transcript_terms = [
        'graduation plan', 'generate plan', 'create plan', 'semester plan',
        'my transcript', 'my courses', 'what do i need', 'generate plan',
        'my academic', 'my requirements', 'my progress', 'my record',
        'next semester', 'upcoming semester', 'what should i take next',
        'recommend courses', 'recommend me courses',
        'audit my document', 'audit my transcript', 'audit my academic requirements',
    ]
    if any(term in question_lower for term in transcript_terms):
        return 'transcript'

    year_terms = ['freshman', 'sophomore', 'junior', 'senior', 'first-year', 'second-year', 'third-year', 'fourth-year']
    major_terms = ['computer science', 'biology', 'business', 'psychology', 'engineering', 'math', 'chemistry', 'physics', 'major', 'studying']
    if any(y in question_lower for y in year_terms) and any(m in question_lower for m in major_terms):
        return 'transcript'

    if any(term in question_lower for term in ['major requirement', 'major courses', 'prescribed', 'option', 'what courses do i need for']):
        return 'major'

    return 'chat'


def route_query(
    user_question: str,
    has_transcript: bool = False,
    conversation_history: Optional[List[dict]] = None,
) -> str:
    """
    LLM-first routing with deterministic fallback only when classification fails.
    Returns: 'course', 'major', 'transcript', 'policy', or 'chat'
    """
    classified_route = _classify_intent_llm(user_question, has_transcript, conversation_history)
    if classified_route:
        return classified_route
    return _route_query_deterministic_fallback(user_question, has_transcript, conversation_history)


def _extract_requested_semesters(user_question: str, conversation_history: Optional[List[dict]] = None) -> int:
    """Infer how many upcoming semesters the user is asking for."""
    q = (user_question or "").lower()
    # Numeric form: "next 2 semesters"
    m = re.search(r"\b(?:next\s+)?(\d+)\s+semesters?\b", q)
    if m:
        try:
            return max(1, min(8, int(m.group(1))))
        except Exception:
            pass

    # Word form: "next two semesters"
    words_to_num = {
        "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8,
    }
    for w, n in words_to_num.items():
        if re.search(rf"\b(?:next\s+)?{w}\s+semesters?\b", q):
            return n

    # Continuation phrasing ("for the semester after that too")
    continuation_terms = [
        "after that", "semester after that", "following semester",
        "and after that", "too", "also",
    ]
    if any(t in q for t in continuation_terms):
        # If prior context was recommendation-like, interpret as 2 upcoming semesters.
        if conversation_history:
            recent_context = " ".join(
                str(m.get("content", "")).lower()
                for m in conversation_history[-8:]
            )
            if any(k in recent_context for k in ["recommend", "next semester", "courses for next", "semester +1"]):
                return 2

    # Default
    return 1


def _build_reasoning_plan_hardcoded(
    user_question: str,
    conversation_history: Optional[List[dict]],
    has_transcript: bool,
) -> Dict[str, object]:
    """Fallback: hardcoded reasoning plan when LLM fails or returns invalid output."""
    q = (user_question or "").lower()
    recommendation_terms = [
        "next semester", "upcoming semester", "what should i take next",
        "recommend courses", "recommend me courses", "what should i take",
    ]
    followup_reco_terms = [
        "after that", "semester after that", "following semester",
        "for the semester after", "and after that", "too",
    ]
    audit_terms = ["audit", "review", "check", "analyze", "analyse", "evaluate", "where do i stand"]
    full_plan_terms = ["full graduation plan", "graduation plan", "semester-by-semester", "full plan"]

    recent_context_text = ""
    if conversation_history:
        recent_context_text = " ".join(str(m.get("content", "")).lower() for m in conversation_history[-8:])
    is_followup_recommendation = any(t in q for t in followup_reco_terms) and any(
        k in recent_context_text for k in ["recommend", "next semester", "courses for next", "semester +1"]
    )

    tools = {"local_retrievers": True, "web_search": False}

    if any(t in q for t in recommendation_terms) or is_followup_recommendation:
        return {
            "intent": "recommendation",
            "needed_data": ["transcript_plan_state"],
            "delivery": "semester_recommendations",
            "semester_horizon": _extract_requested_semesters(user_question, conversation_history),
            "tools": tools,
            "web_search_mode": "never",
        }

    if any(t in q for t in audit_terms) and ("transcript" in q or "document" in q or "requirements" in q or has_transcript):
        return {
            "intent": "audit",
            "needed_data": ["transcript_summary", "unmet_requirements"],
            "delivery": "detailed_but_concise_audit",
            "semester_horizon": 0,
            "tools": tools,
            "web_search_mode": "never",
        }

    if any(t in q for t in full_plan_terms):
        return {
            "intent": "full_plan",
            "needed_data": ["transcript_or_major_year"],
            "delivery": "full_semester_plan",
            "semester_horizon": 8,
            "tools": tools,
            "web_search_mode": "never",
        }

    contextual_q = _contextualize_policy_question(user_question, conversation_history)
    routed = route_query(contextual_q, has_transcript=has_transcript, conversation_history=conversation_history)
    if is_academic_or_policy_question(contextual_q):
        tools["web_search"] = True
    web_search_mode = "if_missing"
    if _is_time_sensitive_factual_question(contextual_q):
        web_search_mode = "verify_current_fact"
    recent_policy_deadline = any(k in recent_context_text for k in ["late drop", "deadline", "withdraw", "drop period"])
    if _question_needs_specific_date(contextual_q) or (_is_vague_followup(user_question) and recent_policy_deadline):
        routed = "policy"
        web_search_mode = "verify_date"
    return {
        "intent": routed,
        "needed_data": [],
        "delivery": "default",
        "semester_horizon": 0,
        "tools": tools,
        "web_search_mode": web_search_mode,
        "contextual_question": contextual_q,
    }


def _build_reasoning_plan(
    user_question: str,
    conversation_history: Optional[List[dict]],
    has_transcript: bool,
) -> Dict[str, object]:
    """
    LLM-based reasoning layer:
      1) What does user want?
      2) What data is needed?
      3) How should we deliver?
    Falls back to hardcoded logic if LLM fails or returns invalid JSON.
    """
    global llm_classifier
    if llm_classifier is None:
        try:
            from langchain_openai import ChatOpenAI
            llm_classifier = ChatOpenAI(model_name=CLASSIFIER_MODEL_NAME, temperature=0)
        except Exception:
            return _build_reasoning_plan_hardcoded(user_question, conversation_history, has_transcript)

    recent_context = ""
    if conversation_history and len(conversation_history) > 0:
        recent = conversation_history[-6:]
        recent_context = "\nRecent conversation:\n" + "\n".join(
            f"{m.get('role', 'user')}: {str(m.get('content', ''))[:150]}" for m in recent
        )

    system_prompt = (
        "You are the REASONING ROUTER for a Penn State Abington advising chatbot. "
        "Your job is to classify the student's intent and decide what data the response pipeline needs.\n\n"
        + PRIVATE_REASONING_GUIDANCE
        + "\n\nOutput contract:\n"
        "- Return ONLY valid JSON. No markdown, no code fence, no comments, no prose before or after.\n"
        "- Use exactly these keys: intent, needed_data, delivery, semester_horizon, tools, web_search_mode, "
        "contextual_question.\n\n"
        "Private decision checklist (do not reveal):\n"
        "1. Read the current question and recent conversation together.\n"
        "2. Resolve vague follow-ups by carrying forward the latest relevant topic.\n"
        "3. Pick the narrowest intent that can answer the student safely.\n"
        "4. Decide whether transcript state, retrievers, or web search are required.\n"
        "5. Set a delivery shape and semester horizon consistent with the user's wording.\n\n"
        "intent values and examples:\n"
        "- recommendation: Student wants course recommendations for next/upcoming semester(s). "
        "Example: 'what should I take next semester?'.\n"
        "- audit: Student wants review/analysis of their transcript or requirements. "
        "Example: 'audit my transcript'.\n"
        "- full_plan: Student wants a full semester-by-semester graduation plan. "
        "Example: 'give me a full graduation plan'.\n"
        "- policy: Institutional rules, dates, procedures, or deadlines. "
        "Example: 'what is the late drop deadline?'.\n"
        "- course: Specific course info such as prerequisites, description, title, or credits. "
        "Example: 'what are CMPSC 488 prerequisites?'.\n"
        "- major: General catalog major requirements, not personal transcript review. "
        "Example: 'what does the Computer Science major require?'.\n"
        "- transcript: Other transcript/planning requests. Example: 'how many credits do I have left?'.\n"
        "- chat: General or unclear questions. Example: 'thanks' or 'can you explain?'.\n\n"
        "needed_data examples:\n"
        "- recommendation -> [\"transcript_plan_state\"].\n"
        "- audit -> [\"transcript_summary\", \"unmet_requirements\"].\n"
        "- full_plan -> [\"transcript_or_major_year\"].\n"
        "- policy/course/major/chat -> [] unless the question explicitly needs transcript context.\n\n"
        "delivery values and examples:\n"
        "- semester_recommendations for recommendation.\n"
        "- detailed_but_concise_audit for audit.\n"
        "- full_semester_plan for full_plan.\n"
        "- default otherwise.\n\n"
        "semester_horizon examples:\n"
        "- 'next semester' -> 1.\n"
        "- 'next two semesters' -> 2.\n"
        "- 'full graduation plan' -> 8.\n"
        "- policy/course/major/chat -> 0.\n\n"
        "tools object:\n"
        "- local_retrievers: usually true for academic questions. Example: course prerequisites need local_retrievers=true.\n"
        "- web_search: true when policy/date/current information may need verification. Example: deadline questions need web_search=true.\n\n"
        "web_search_mode values and examples:\n"
        "- never: recommendation, audit, full_plan.\n"
        "- verify_date: deadline/date questions such as late drop, withdrawal, registration windows.\n"
        "- verify_current_fact: time-sensitive factual questions.\n"
        "- if_missing: general policy/academic questions where local data may be incomplete.\n\n"
        "contextual_question examples:\n"
        "- If the student asks 'what about the deadline?' after discussing late drop, write "
        "'What is the late drop deadline?'.\n"
        "- If the question is already clear, copy it with typo fixes only."
    )

    user_prompt = f"""Student has transcript uploaded: {has_transcript}
{recent_context}

Current question: "{user_question or ''}"

JSON:"""

    try:
        response = llm_classifier.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]).content.strip()

        # Extract JSON from response (handle markdown code blocks)
        raw = response
        if "```" in raw:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                raw = raw[start:end]
        else:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                raw = raw[start:end]

        plan = json.loads(raw)

        # Validate and normalize
        valid_intents = {"recommendation", "audit", "full_plan", "policy", "course", "major", "transcript", "chat"}
        intent = str(plan.get("intent", "chat")).lower().strip()
        if intent not in valid_intents:
            intent = "chat"

        needed_data = plan.get("needed_data")
        if not isinstance(needed_data, list):
            needed_data = []

        valid_deliveries = {"semester_recommendations", "detailed_but_concise_audit", "full_semester_plan", "default"}
        delivery = str(plan.get("delivery", "default")).lower().strip()
        if delivery not in valid_deliveries:
            delivery = "default"

        try:
            horizon = int(plan.get("semester_horizon", 0))
            horizon = max(0, min(8, horizon))
        except (TypeError, ValueError):
            horizon = 0

        tools = plan.get("tools") or {}
        if not isinstance(tools, dict):
            tools = {"local_retrievers": True, "web_search": False}
        tools.setdefault("local_retrievers", True)
        tools.setdefault("web_search", bool(tools.get("web_search", False)))

        valid_modes = {"never", "if_missing", "verify_current_fact", "verify_date"}
        web_search_mode = str(plan.get("web_search_mode", "if_missing")).lower().strip()
        if web_search_mode not in valid_modes:
            web_search_mode = "if_missing"

        contextual_q = plan.get("contextual_question")
        if contextual_q is None or not isinstance(contextual_q, str):
            contextual_q = _contextualize_policy_question(user_question, conversation_history) or user_question

        # For recommendation, ensure semester_horizon is sensible (LLM may miss "next 2 semesters")
        if intent == "recommendation" and horizon == 0:
            horizon = _extract_requested_semesters(user_question, conversation_history)

        return {
            "intent": intent,
            "needed_data": needed_data,
            "delivery": delivery,
            "semester_horizon": horizon,
            "tools": tools,
            "web_search_mode": web_search_mode,
            "contextual_question": str(contextual_q).strip() or user_question,
        }
    except Exception as e:
        print(f"⚠️ LLM reasoning plan failed ({e}), using hardcoded fallback")
        return _build_reasoning_plan_hardcoded(user_question, conversation_history, has_transcript)


def _course_level_num(code: str) -> int:
    m = re.search(r'\b(\d{3})\b', code or '')
    try:
        return int(m.group(1)) if m else 999
    except Exception:
        return 999

def _normalize_year_token(year: str) -> Optional[str]:
    """Normalize year tokens to freshman/sophomore/junior/senior."""
    if not year:
        return None
    year_lower = year.strip().lower()
    year_mapping = {
        'first-year': 'freshman', '1st year': 'freshman', 'first year': 'freshman',
        'second-year': 'sophomore', '2nd year': 'sophomore', 'second year': 'sophomore',
        'third-year': 'junior', '3rd year': 'junior', 'third year': 'junior',
        'fourth-year': 'senior', '4th year': 'senior', 'fourth year': 'senior'
    }
    if year_lower in year_mapping:
        return year_mapping[year_lower]
    if year_lower in ['freshman', 'sophomore', 'junior', 'senior']:
        return year_lower
    return None

def _fuzzy_year_from_text(text: str) -> Optional[str]:
    """Handle common typos like 'freshamn' by fuzzy matching."""
    import difflib
    tokens = re.split(r'[^A-Za-z]+', text or "")
    choices = ['freshman', 'sophomore', 'junior', 'senior', 'firstyear', 'secondyear', 'thirdyear', 'fourthyear']
    for token in tokens:
        if len(token) < 5:
            continue
        token_lower = token.lower()
        match = difflib.get_close_matches(token_lower, choices, n=1, cutoff=0.78)
        if match:
            normalized = match[0]
            normalized = normalized.replace('firstyear', 'freshman').replace('secondyear', 'sophomore') \
                                   .replace('thirdyear', 'junior').replace('fourthyear', 'senior')
            return _normalize_year_token(normalized)
    return None

def _format_course_for_display(course: str) -> str:
    try:
        from graduationFlowchart import format_planned_course_label
        return format_planned_course_label(course)
    except Exception:
        if course.startswith("GENED__"):
            parts = course.split("__")
            key = parts[1] if len(parts) > 1 else "GENED"
            return f"Gen Ed ({key})"
        if course.startswith("OPT__"):
            return "Option Elective"
        if course.startswith("FREE__") or course.startswith("ELECTIVE__FREE__"):
            return "Free Elective"
        return course

def _format_plan_structure(plan_structure: dict, start_semester_number: int = 1) -> str:
    semesters = plan_structure.get("semesters", []) if plan_structure else []
    prereq_map = plan_structure.get("prerequisites", {}) if plan_structure else {}
    course_titles = plan_structure.get("course_titles", {}) if plan_structure else {}
    if not semesters:
        return "I couldn't generate a semester plan. Please try again or upload your transcript."
    lines = []
    for idx, sem in enumerate(sorted(semesters, key=lambda s: s.get("semester", 0))):
        display_semester = start_semester_number + idx
        lines.append(f"Semester {display_semester}:")
        courses = sem.get("courses", [])
        for course in courses:
            prereqs = prereq_map.get(course, []) or []
            prereq_text = ", ".join(_format_course_for_display(p) for p in prereqs) if prereqs else "None"
            display_course = _format_course_for_display(course)
            course_title = str(course_titles.get(course) or course_titles.get(str(course).upper()) or "").strip()
            if course_title and display_course == course:
                display_course = f"{display_course}: {course_title}"
            lines.append(f"{display_course} (prerequisites: {prereq_text})")
        lines.append(f"Total Credits: {sem.get('credits', len(courses) * 3)}")
        lines.append("---")
    return "\n".join(lines).rstrip("-\n")


def _slice_plan_structure(plan_structure: dict, semester_horizon: int) -> dict:
    if not plan_structure or semester_horizon <= 0:
        return plan_structure
    semesters = list(plan_structure.get("semesters", []) or [])
    sliced_semesters = sorted(semesters, key=lambda s: s.get("semester", 0))[:semester_horizon]
    return {
        "semesters": sliced_semesters,
        "total_credits": sum(int(s.get("credits", 0)) for s in sliced_semesters),
        "prerequisites": dict(plan_structure.get("prerequisites", {}) or {}),
    }


def _extract_semester_blocks_from_text(plan_text: str) -> List[str]:
    if not plan_text:
        return []
    plan_text = canonicalize_semester_blocks(plan_text)
    matches = re.finditer(
        r"((?:[-*]\s*)?(?:\*\*)?Semester\s+\+?\d+:(?:\*\*)?\s*\n.*?(?:[-*]\s*)?(?:\*\*)?Total Credits:(?:\*\*)?\s*\d+)",
        plan_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    blocks: List[str] = []
    for match in matches:
        block = match.group(1).strip()
        if block:
            blocks.append(block)
    return blocks


def _find_recent_assistant_plan_text(conversation_history: Optional[List[dict]]) -> Optional[str]:
    if not conversation_history:
        return None
    for msg in reversed(conversation_history):
        if msg.get("role") != "assistant":
            continue
        content = str(msg.get("content", "") or "")
        if _extract_semester_blocks_from_text(content):
            return content
    return None


def _slice_semester_plan_text(plan_text: str, semester_horizon: int) -> str:
    if semester_horizon <= 0:
        return plan_text
    blocks = _extract_semester_blocks_from_text(plan_text)
    if not blocks:
        return plan_text
    return "\n---\n".join(blocks[:semester_horizon]).strip()


# ---------------------------------------------------------------------------
# Target-semester parsing.
#
# A question like "first semester courses for computer science" or
# "3rd semester of biology, B.S." pins a specific semester of the standard
# graduation plan. We map this onto:
#   - semester_number in 1..8 (absolute index from freshman fall)
#   - year_level for generate_plan_from_major_and_year (so the deterministic
#     planner starts its output at the right semester number)
#   - horizon (1 for a single semester, 2 for a full academic year)
#
# This is used by the major_lookup executor path to route semester-specific
# major questions to the deterministic planner instead of LLM synthesis.
# ---------------------------------------------------------------------------

_ORDINAL_TO_SEMESTER = {
    "1st": 1, "first": 1,
    "2nd": 2, "second": 2,
    "3rd": 3, "third": 3,
    "4th": 4, "fourth": 4,
    "5th": 5, "fifth": 5,
    "6th": 6, "sixth": 6,
    "7th": 7, "seventh": 7,
    "8th": 8, "eighth": 8,
}

_YEAR_SEMESTER_START = {"freshman": 1, "sophomore": 3, "junior": 5, "senior": 7}


def _year_level_for_semester(n: int) -> str:
    if n <= 2:
        return "freshman"
    if n <= 4:
        return "sophomore"
    if n <= 6:
        return "junior"
    return "senior"


def _extract_target_semester_request(question: str) -> Optional[Dict[str, Any]]:
    """Parse a natural-language question for a specific target semester.

    Returns ``{"semester_number": N, "year_level": str, "horizon": int}`` when
    the question pins a concrete semester; otherwise ``None``.

    Supported phrasings (case-insensitive):
      - ``first semester``, ``1st semester``, ``semester 1``  -> (1, freshman, 1)
      - ``3rd semester``, ``semester 3``                       -> (3, sophomore, 1)
      - ``freshman fall``, ``sophomore spring``                -> (start/+1, 1)
      - ``freshman year``, ``junior year``                     -> (start, year, 2)
    """
    if not question:
        return None
    lowered = question.lower()

    for word, n in _ORDINAL_TO_SEMESTER.items():
        if re.search(rf"\b{word}\s+semester\b", lowered):
            return {
                "semester_number": n,
                "year_level": _year_level_for_semester(n),
                "horizon": 1,
            }

    m = re.search(r"\bsemester\s+(\d)\b", lowered)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 8:
            return {
                "semester_number": n,
                "year_level": _year_level_for_semester(n),
                "horizon": 1,
            }

    for year, start in _YEAR_SEMESTER_START.items():
        if re.search(rf"\b{year}\s+fall\b", lowered):
            return {"semester_number": start, "year_level": year, "horizon": 1}
        if re.search(rf"\b{year}\s+spring\b", lowered):
            return {"semester_number": start + 1, "year_level": year, "horizon": 1}

    for year, start in _YEAR_SEMESTER_START.items():
        if re.search(rf"\b{year}\s+year\b", lowered):
            return {"semester_number": start, "year_level": year, "horizon": 2}

    return None


def _extract_semester_block_by_number(plan_text: str, semester_number: int) -> Optional[str]:
    """Return the block labeled ``Semester N:`` from a formatted plan.

    ``_format_plan_structure`` labels blocks with absolute semester numbers
    (e.g. a plan generated for a sophomore starts at ``Semester 3:``), so we
    can pick out the target semester directly rather than relying on position.
    """
    for block in _extract_semester_blocks_from_text(plan_text):
        m = re.search(r"semester\s+\+?(\d+)", block, flags=re.IGNORECASE)
        if m and int(m.group(1)) == semester_number:
            return block
    return None


def _build_available_tool_inventory(has_transcript_for_plan: bool) -> List[str]:
    tools = ["conversation_history", "deterministic_planner", "web_search"]
    if has_transcript_for_plan:
        tools.extend(["transcript_state", "transcript_retriever"])
    if pdf_retriever is not None and "transcript_retriever" not in tools:
        tools.append("transcript_retriever")
    if course_retriever is not None:
        tools.append("course_retriever")
    if majors_retriever is not None:
        tools.append("majors_retriever")
    if policy_retriever is not None:
        tools.append("policy_retriever")
    if course_graph is not None:
        tools.append("prerequisite_graph")
    return list(dict.fromkeys(tools))


def get_student_data_from_transcript() -> Optional[dict]:
    """Return deterministic student data if transcript is loaded."""
    global stored_pdf_docs, latest_student_data
    if stored_pdf_docs:
        try:
            latest_student_data = extract_student_data_deterministic(stored_pdf_docs)
            return latest_student_data
        except Exception as e:
            print(f"⚠️ Failed to extract student data deterministically: {e}")
    return latest_student_data

def process_psu_courses(user_question):
    """Process course-level queries using course_retriever."""
    global course_retriever
    if course_retriever is None:
        return "Course information not available."
    relevant_courses = course_retriever.invoke(user_question)
    if not relevant_courses:
        return "No relevant courses found."
    context_summary = "\n\n".join(doc.page_content for doc in relevant_courses)
    return context_summary


def extract_major_from_summary(summary_text):
    match = re.search(r"Major/Option\s*:\s*(.+)", summary_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        return None


def extract_completed_courses_from_record(student_record):
    """
    Extract all completed and in-progress course codes from the LLM's
    structured student record summary.
    """
    patterns = [
        r'\b([A-Z]{2,}\s+\d{3}[A-Z]?)\b',  # e.g. "CMPSC 472" or "MATH 140A"
        r'\b([A-Z]{2,}\s+\d{3})\b',
        r'([A-Z]{2,}\s+\d{3}[A-Z]?)',
    ]

    courses = []
    for pattern in patterns:
        matches = re.findall(pattern, student_record)
        courses.extend(matches)

    seen = set()
    unique_courses = []
    for course in courses:
        normalized = course.replace(' ', ' ').strip()
        if normalized not in seen:
            seen.add(normalized)
            unique_courses.append(normalized)

    print(f" Extracted {len(unique_courses)} courses: {unique_courses[:10]}...")
    return unique_courses


def validate_prerequisites_in_plan(course_list, major_graph, completed_courses=None):
    """
    Check that for each course in course_list, all prerequisite courses
    are either completed or appear earlier in the list.
    """
    if completed_courses is None:
        completed_courses = []

    invalid_courses = []
    all_available_courses = set(completed_courses + course_list)

    for course in course_list:
        prereqs = [
            u for u, v, d in major_graph.in_edges(course, data=True)
            if d.get("type") == "prereq"
        ]

        if prereqs:
            missing_prereqs = [pr for pr in prereqs if pr not in all_available_courses]

            if missing_prereqs:
                invalid_courses.append({
                    'course': course,
                    'missing_prereqs': missing_prereqs
                })

    return invalid_courses


def enhance_graduation_plan_with_prerequisites(graduation_plan, major_graph):
    """
    Scan the generated plan and, for each course, append a clean
    "prerequisites: ..." clause based on the course graph.
    """
    enhanced_plan = graduation_plan

    course_pattern = r'\b([A-Z]{2,}\s+\d{3})\b'
    courses_in_plan = re.findall(course_pattern, graduation_plan)

    for course in courses_in_plan:
        prereqs = [
            u for u, v, d in major_graph.in_edges(course, data=True)
            if d.get("type") == "prereq"
        ]

        if prereqs:
            course_section_pattern = rf'\*\*{re.escape(course)}[^*]*\*\*'
            course_match = re.search(course_section_pattern, enhanced_plan)

            if course_match:
                course_text = course_match.group()

                # Remove any existing prereq text
                course_text = re.sub(
                    r',\s*prerequisites:\s*None',
                    '',
                    course_text
                )
                course_text = re.sub(
                    r',\s*prerequisites:\s*[^,]*(?:,\s*prerequisites:\s*[^,]*)*',
                    '',
                    course_text
                )

                prereq_text = f", prerequisites: {', '.join(prereqs)}"
                enhanced_plan = enhanced_plan.replace(
                    course_match.group(),
                    course_text + prereq_text
                )

    return enhanced_plan


# =========================
# MAIN: Transcript-based plan
# =========================

def aggregate_and_generate_response(user_question=None):
    """
    Main function to generate graduation plan using deterministic extraction and planning.
    Requires that a transcript was uploaded and stored_pdf_docs is available.
    """
    global stored_pdf_docs, has_transcript
    
    if not has_transcript or stored_pdf_docs is None:
        return "No transcript found. Please upload a What-If Report, or tell me your major and year level (e.g., 'I'm a Computer Science freshman') to generate a graduation plan."

    print("\n🔹 Using Deterministic Extraction and Planning...")

    # Step 1: Deterministic extraction (no RAG)
    student_data = extract_student_data_deterministic(stored_pdf_docs)
    print(f"✅ Extracted major: {student_data.get('major')}")
    completed_courses = student_data.get("completed_courses", [])
    in_progress_courses = student_data.get("in_progress_courses", [])
    completed_and_ip = []
    for course in completed_courses + in_progress_courses:
        if course and course not in completed_and_ip:
            completed_and_ip.append(course)

    print(f"✅ Extracted {len(completed_courses)} completed courses")
    if in_progress_courses:
        print(f"✅ Extracted {len(in_progress_courses)} in-progress courses")
    print(f"✅ Credits completed: {student_data.get('credits_completed')}")
    print(f"✅ Credits needed: {student_data.get('credits_needed')}")

    # Step 2: Deterministic planning
    from graduationFlowchart import generate_semester_plan
    
    courses_csv = os.path.join(os.path.dirname(__file__), "processed_psu_courses.csv")
    majors_csv = os.path.join(os.path.dirname(__file__), "all_abington_majors_combined(5).csv")

    # Ensure CSV paths exist
    if not os.path.exists(courses_csv):
        alt_courses = os.path.join(os.path.dirname(__file__), "..", "LLM", "processed_psu_courses.csv")
        if os.path.exists(alt_courses):
            courses_csv = alt_courses
    
    if not os.path.exists(majors_csv):
        alt_majors = os.path.join(os.path.dirname(__file__), "..", "LLM", "all_abington_majors_combined(5).csv")
        if os.path.exists(alt_majors):
            majors_csv = alt_majors
    
    # Normalize major name before planning
    extracted_major = student_data.get("major")
    if extracted_major:
        normalized_major = normalize_major_name(extracted_major)
        if normalized_major != extracted_major:
            print(f"🔄 Normalized major: '{extracted_major}' → '{normalized_major}'")
        student_data["major"] = normalized_major
    else:
        # Try to extract from PDF text using LLM as fallback
        print("⚠️ Could not extract major deterministically, trying LLM extraction...")
        try:
            transcript_context = "\n".join(doc.page_content[:500] for doc in stored_pdf_docs[:3])
            major_extraction = llm1.invoke([
                SystemMessage(content=(
                    "You are a precise transcript information extractor. Extract ONLY the student's declared "
                    "major from the transcript excerpt.\n\n"
                    + PRIVATE_REASONING_GUIDANCE
                    + "\n\nOutput contract:\n"
                    "- Return only the full major name, no label and no explanation.\n"
                    "- Prefer the exact university wording when visible. Example output: "
                    "Computer Science, B.S. (Abington)\n"
                    "- If no major is visible, return UNKNOWN."
                )),
                HumanMessage(content=f"Transcript excerpt:\n{transcript_context[:1000]}\n\nMajor:")
            ]).content.strip()
            if major_extraction and major_extraction.strip().upper() != "UNKNOWN":
                student_data["major"] = normalize_major_name(major_extraction)
        except:
            pass
    
    if not student_data.get("major"):
        return "I couldn't determine your major from the transcript. Please specify your major (e.g., 'Computer Science, B.S. (Abington)') or contact support."
    
    try:
        plan_structure = generate_semester_plan(
            major_name=student_data.get("major"),
            completed_courses=completed_and_ip,
            unmet_requirements=student_data.get("unmet_requirements", []),
            courses_csv_path=courses_csv,
            majors_csv_path=majors_csv,
            credits_needed=student_data.get("credits_needed")
        )
        print(f"✅ Generated plan with {len(plan_structure['semesters'])} semesters")
    except ValueError as e:
        error_msg = str(e)
        if "not found in majors CSV" in error_msg:
            # Try fuzzy matching to suggest correct major
            print(f"⚠️ Major '{student_data.get('major')}' not found, trying fuzzy match...")
            try:
                from fuzzywuzzy import process
                df_majors = pd.read_csv(majors_csv)
                available_majors = df_majors['Major Name'].tolist()
                best_match = process.extractOne(student_data.get("major"), available_majors)
                if best_match and best_match[1] >= 50:
                    suggested_major = best_match[0]
                    return f"I couldn't find the exact major '{student_data.get('major')}' in the system. Did you mean '{suggested_major}'? Please verify your major name or contact support."
            except:
                pass
            return f"I couldn't find the major '{student_data.get('major')}' in the system. Please verify your major name is correct, or contact support."
        else:
            print(f"⚠️ Error in deterministic planning: {e}")
            import traceback
            traceback.print_exc()
            return f"I encountered an error generating your plan: {str(e)}. Please try again or contact support."
    except Exception as e:
        print(f"⚠️ Error in deterministic planning: {e}")
        import traceback
        traceback.print_exc()
        return f"I encountered an error generating your plan: {str(e)}. Please try again or contact support."

    # Step 3: deterministic formatting for consistency across all plans
    # (ensures GenEd category labels are shown, not raw placeholders).
    print("\n🔹 Formatting plan deterministically...")
    return _format_plan_structure(plan_structure, start_semester_number=1)
    course_list = re.findall(r'\b([A-Z]{2,}\s+\d{3})\b', enhanced_plan)
    validation_warnings = validate_prerequisites_in_plan(
        course_list,
        major_specific_graph,
        completed_courses
    )

    if validation_warnings:
        print("⚠️ Validation warnings (not shown to user):")
        for warning in validation_warnings:
            print(f"   - {warning['course']}: needs {', '.join(warning['missing_prereqs'])}")

    return enhanced_plan


def simple_extract_major_and_year(user_text):
    """Simple pattern-based extraction as fallback."""
    import re
    major = None
    year = None
    
    # Common major patterns
    major_patterns = [
        r'(?:studying|major in|major is|studies)\s+([^,\.\n]+)',
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:major|degree)',
        r'(computer science|biology|business|engineering|psychology|mathematics|chemistry|physics|english|history|economics|accounting|finance|marketing|management)',
    ]
    
    # Year patterns
    year_patterns = [
        r'\b(freshman|sophomore|junior|senior|first-year|second-year|third-year|fourth-year|1st year|2nd year|3rd year|4th year)\b',
    ]
    
    text_lower = user_text.lower()
    
    # Extract year
    for pattern in year_patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            year_str = match.group(1).lower()
            year_mapping = {
                'first-year': 'freshman', '1st year': 'freshman', 'first year': 'freshman',
                'second-year': 'sophomore', '2nd year': 'sophomore', 'second year': 'sophomore',
                'third-year': 'junior', '3rd year': 'junior', 'third year': 'junior',
                'fourth-year': 'senior', '4th year': 'senior', 'fourth year': 'senior'
            }
            year = year_mapping.get(year_str, year_str)
            if year in ['freshman', 'sophomore', 'junior', 'senior']:
                break
    
    # Extract major - try to find in CSV
    try:
        csv_path = os.path.join(os.path.dirname(__file__), "all_abington_majors_combined(5).csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(os.path.dirname(__file__), "..", "LLM", "all_abington_majors_combined(5).csv")
        majors_df = pd.read_csv(csv_path)
        
        # Common major keywords
        major_keywords = {
            'computer science': ['computer science', 'cs', 'comp sci'],
            'biology': ['biology', 'bio'],
            'business': ['business', 'business administration'],
            'psychology': ['psychology', 'psych'],
            'mathematics': ['mathematics', 'math'],
        }
        
        for keyword, variations in major_keywords.items():
            if any(var in text_lower for var in variations):
                # Find matching major in CSV
                matching = majors_df[majors_df['Major Name'].str.contains(keyword, case=False, na=False)]
                if not matching.empty:
                    major = matching.iloc[0]['Major Name']
                    break
        
        # If not found, try direct search
        if not major:
            for major_name in majors_df['Major Name'].tolist():
                major_lower = major_name.lower()
                # Check if any significant word from major name is in user text
                major_words = [w for w in major_lower.split() if len(w) > 3]
                if any(word in text_lower for word in major_words):
                    major = major_name
                    break
    except Exception as e:
        print(f"⚠️ Error in simple extraction: {e}")
    
    return major, year

def extract_major_and_year_from_text(user_text, conversation_history):
    """Extract major and year level from user text using LLM."""
    try:
        # Get list of available majors from CSV
        csv_path = os.path.join(os.path.dirname(__file__), "all_abington_majors_combined(5).csv")
        if not os.path.exists(csv_path):
            # Try alternative path
            csv_path = os.path.join(os.path.dirname(__file__), "..", "LLM", "all_abington_majors_combined(5).csv")
        majors_df = pd.read_csv(csv_path)
        available_majors = majors_df['Major Name'].tolist()[:50]  # Limit to first 50 for prompt size
        
        # Build context from conversation history
        history_context = ""
        if conversation_history and len(conversation_history) > 0:
            recent_history = conversation_history[-4:]  # Last 4 messages
            history_context = "\n\nPrevious conversation:\n"
            for msg in recent_history:
                role = "Student" if msg.get('role') == 'user' else "AI"
                history_context += f"{role}: {msg.get('content', '')}\n"
        
        # Use LLM to extract major and year
        # Show some common majors first
        common_majors = [m for m in available_majors if any(x in m.lower() for x in ['computer', 'biology', 'business', 'psychology', 'mathematics'])][:10]
        other_majors = [m for m in available_majors if m not in common_majors][:10]
        
        extraction_prompt = f"""Task: Extract the student's major and year level from the current message and recent conversation.

Available major examples:
- Common majors: {', '.join(common_majors)}
- Other available majors: {', '.join(other_majors)}

Allowed year values:
- freshman
- sophomore
- junior
- senior
- UNKNOWN

Current student message:
"{user_text}"
{history_context}

Extraction rules:
1. Reason privately across the current message and conversation history.
2. Extract BOTH fields when both are present, even if grammar is messy.
3. Map first-year/1st year -> freshman, second-year/2nd year -> sophomore, third-year/3rd year -> junior, fourth-year/4th year -> senior.
4. Use the closest listed major name when the student uses an abbreviation such as CS, bio, psych, or business.
5. If a field is not supported by the message or history, return UNKNOWN for that field.

Output contract:
Return exactly two lines and no extra text:
MAJOR: [exact major name from list, closest listed major, or UNKNOWN]
YEAR: [freshman OR sophomore OR junior OR senior OR UNKNOWN]

Examples:
Student: "I'm a computer science freshman"
MAJOR: Computer Science, B.S. (Abington)
YEAR: freshman

Student: "biology sophomore"
MAJOR: Biology, B.S. (Abington)
YEAR: sophomore

Student: "I'm a first year psych major"
MAJOR: Psychological and Social Sciences, B.A. (Abington)
YEAR: freshman

Student: "give me a plan"
MAJOR: UNKNOWN
YEAR: UNKNOWN"""
        
        extraction_result = llm1.invoke([
            SystemMessage(content=(
                "You are a precise information extractor for academic advising. "
                "Extract only the requested fields from student text and recent conversation.\n\n"
                + PRIVATE_REASONING_GUIDANCE
                + "\n\nDo not infer unsupported facts. Return exactly the requested two-line format."
            )),
            HumanMessage(content=extraction_prompt)
        ]).content
        
        # Parse the result - be more flexible with parsing
        major = None
        year = None
        
        # Try multiple parsing strategies
        extraction_result_lower = extraction_result.lower()
        
        # Strategy 1: Look for explicit MAJOR: and YEAR: labels
        for line in extraction_result.split('\n'):
            line_upper = line.upper()
            if 'MAJOR:' in line_upper:
                major = line.split(':', 1)[1].strip()
                if major == 'UNKNOWN':
                    major = None
            elif 'YEAR:' in line_upper:
                year = line.split(':', 1)[1].strip().lower()
                if year == 'UNKNOWN':
                    year = None
                # Normalize year terms
                year_mapping = {
                    'first-year': 'freshman', '1st year': 'freshman', 'first year': 'freshman',
                    'second-year': 'sophomore', '2nd year': 'sophomore', 'second year': 'sophomore',
                    'third-year': 'junior', '3rd year': 'junior', 'third year': 'junior',
                    'fourth-year': 'senior', '4th year': 'senior', 'fourth year': 'senior'
                }
                year = year_mapping.get(year, year)
        
        # Strategy 2: If year not found, try direct pattern matching in the LLM response
        if not year:
            year_patterns = [
                r'\b(freshman|sophomore|junior|senior)\b',
                r'\b(first-year|second-year|third-year|fourth-year)\b',
                r'\b(1st year|2nd year|3rd year|4th year)\b'
            ]
            for pattern in year_patterns:
                match = re.search(pattern, extraction_result_lower, re.IGNORECASE)
                if match:
                    year_str = match.group(1).lower()
                    year_mapping = {
                        'first-year': 'freshman', '1st year': 'freshman', 'first year': 'freshman',
                        'second-year': 'sophomore', '2nd year': 'sophomore', 'second year': 'sophomore',
                        'third-year': 'junior', '3rd year': 'junior', 'third year': 'junior',
                        'fourth-year': 'senior', '4th year': 'senior', 'fourth year': 'senior'
                    }
                    year = year_mapping.get(year_str, year_str)
                    if year in ['freshman', 'sophomore', 'junior', 'senior']:
                        break
        
        # Strategy 3: If still no year, check the original user text and conversation history
        if not year:
            # Check current user message
            year_patterns = [
                r'\b(freshman|sophomore|junior|senior)\b',
                r'\b(first-year|second-year|third-year|fourth-year)\b',
                r'\b(1st year|2nd year|3rd year|4th year)\b'
            ]
            for pattern in year_patterns:
                match = re.search(pattern, user_text.lower(), re.IGNORECASE)
                if match:
                    year_str = match.group(1).lower()
                    year_mapping = {
                        'first-year': 'freshman', '1st year': 'freshman', 'first year': 'freshman',
                        'second-year': 'sophomore', '2nd year': 'sophomore', 'second year': 'sophomore',
                        'third-year': 'junior', '3rd year': 'junior', 'third year': 'junior',
                        'fourth-year': 'senior', '4th year': 'senior', 'fourth year': 'senior'
                    }
                    year = year_mapping.get(year_str, year_str)
                    if year in ['freshman', 'sophomore', 'junior', 'senior']:
                        break
            
            # Check conversation history if still not found
            if not year and conversation_history:
                for msg in reversed(conversation_history[-4:]):
                    if msg.get('role') == 'user':
                        msg_text = msg.get('content', '').lower()
                        for pattern in year_patterns:
                            match = re.search(pattern, msg_text, re.IGNORECASE)
                            if match:
                                year_str = match.group(1).lower()
                                year_mapping = {
                                    'first-year': 'freshman', '1st year': 'freshman', 'first year': 'freshman',
                                    'second-year': 'sophomore', '2nd year': 'sophomore', 'second year': 'sophomore',
                                    'third-year': 'junior', '3rd year': 'junior', 'third year': 'junior',
                                    'fourth-year': 'senior', '4th year': 'senior', 'fourth year': 'senior'
                                }
                                year = year_mapping.get(year_str, year_str)
                                if year in ['freshman', 'sophomore', 'junior', 'senior']:
                                    break
                        if year:
                            break

        # Strategy 4: Fuzzy year matching for typos (e.g., "freshamn")
        if not year:
            year = _fuzzy_year_from_text(user_text)
        
        # Try to match major name with available majors using fuzzy matching if needed
        if major and major != 'UNKNOWN':
            # Check if major is in the list or find close match
            df = majors_df
            matching_major = df[df['Major Name'].str.contains(major, case=False, na=False)]
            if not matching_major.empty:
                major = matching_major.iloc[0]['Major Name']
            else:
                # Try fuzzy matching on first few words
                try:
                    from fuzzywuzzy import process
                    major_names = majors_df['Major Name'].tolist()
                    best_match = process.extractOne(major, major_names)
                    if best_match and best_match[1] >= 70:  # 70% similarity threshold
                        major = best_match[0]
                    else:
                        major = None
                except ImportError:
                    # Fallback: try partial string matching
                    for major_name in majors_df['Major Name'].tolist():
                        if major.lower() in major_name.lower() or major_name.lower() in major.lower():
                            major = major_name
                            break
                    else:
                        major = None
        
        # Final fallback: simple deterministic extraction from user text
        if not major or not year:
            simple_major, simple_year = simple_extract_major_and_year(user_text)
            major = major or simple_major
            year = year or simple_year

        # Fuzzy year matching for misspellings in user text
        if not year:
            year = _fuzzy_year_from_text(user_text)

        # Normalize major to a CSV-backed canonical name
        if major:
            normalized_major = normalize_major_name(major)
            if normalized_major:
                major = normalized_major
        
        print(f"🔍 Extracted - Major: {major}, Year: {year}")
        print(f"🔍 LLM raw response: {extraction_result[:200]}...")  # Debug: show first 200 chars of LLM response
        return major, year
        
    except Exception as e:
        print(f"⚠️ Error extracting major/year: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def _extract_major_year_from_request_context(
    user_question: str,
    conversation_history: Optional[List[dict]],
) -> Tuple[Optional[str], Optional[str]]:
    major_from_text, year_from_text = None, None
    try:
        major_from_text, year_from_text = extract_major_and_year_from_text(user_question, conversation_history)
        if not major_from_text or not year_from_text:
            simple_major, simple_year = simple_extract_major_and_year(user_question)
            major_from_text = major_from_text or simple_major
            year_from_text = year_from_text or simple_year
        if not year_from_text:
            year_from_text = _fuzzy_year_from_text(user_question)
    except Exception as e:
        print(f"⚠️ Error extracting major/year from request context: {e}")
    return major_from_text, year_from_text


def _extract_major_year_from_conversation_history(
    conversation_history: Optional[List[dict]],
) -> Tuple[Optional[str], Optional[str]]:
    if not conversation_history:
        return None, None

    major = None
    year = None

    for preferred_role in ("user", "assistant"):
        for message in reversed(conversation_history):
            if message.get("role") != preferred_role:
                continue
            content = str(message.get("content", "") or "").strip()
            if not content:
                continue

            simple_major, simple_year = simple_extract_major_and_year(content)
            if simple_major and not major:
                major = simple_major
            if simple_year and not year:
                year = _normalize_year_token(simple_year) or simple_year
            if not year:
                fuzzy_year = _fuzzy_year_from_text(content)
                if fuzzy_year:
                    year = fuzzy_year
            if major and year:
                return major, year

    user_text = "\n".join(
        str(message.get("content", "") or "")
        for message in conversation_history
        if message.get("role") == "user"
    ).strip()
    if user_text:
        simple_major, simple_year = simple_extract_major_and_year(user_text)
        major = major or simple_major
        year = year or (_normalize_year_token(simple_year) if simple_year else None) or _fuzzy_year_from_text(user_text)

    return major, year


def _build_conversation_profile_recall_draft(
    conversation_history: Optional[List[dict]],
    student_data: Optional[dict],
) -> str:
    major, year = _extract_major_year_from_conversation_history(conversation_history)
    if student_data:
        major = major or student_data.get("major")

    if major and year:
        return f"From earlier in our conversation, you said you're a {year} majoring in {major}."
    if year:
        return f"From earlier in our conversation, you said you're a {year}. I don't have your major with confidence."
    if major:
        return f"From earlier in our conversation, you said your major is {major}. I don't have your year level with confidence."
    return "I couldn't determine your year level and major from the earlier conversation with enough confidence."


def _build_reasoning_support_context(
    plan: OrchestrationPlan,
    conversation_history: Optional[List[dict]],
    student_data: Optional[dict],
    draft_response: str,
    metadata: Optional[Dict[str, Any]],
) -> str:
    sections: List[str] = []
    metadata = metadata or {}

    sections.append(
        "Reason from the user's request and conversation history first. "
        "Treat any helper-generated draft as supporting evidence, not the only valid path."
    )

    major_from_history, year_from_history = _extract_major_year_from_conversation_history(conversation_history)
    recovered_facts = []
    if year_from_history:
        recovered_facts.append(f"year={year_from_history}")
    if major_from_history:
        recovered_facts.append(f"major={major_from_history}")
    if recovered_facts:
        sections.append("Recovered conversation facts: " + ", ".join(recovered_facts))

    if not plan.use_tools:
        recent_plan = _find_recent_assistant_plan_text(conversation_history)
        if recent_plan:
            sections.append("Relevant prior plan from the conversation:\n" + recent_plan[:2500])

    if student_data:
        transcript_facts = []
        if student_data.get("major"):
            transcript_facts.append(f"transcript_major={student_data.get('major')}")
        if student_data.get("credits_completed") is not None:
            transcript_facts.append(f"credits_completed={student_data.get('credits_completed')}")
        if student_data.get("credits_needed") is not None:
            transcript_facts.append(f"credits_needed={student_data.get('credits_needed')}")
        if transcript_facts:
            sections.append("Transcript facts: " + ", ".join(transcript_facts))

    if plan.executor_path == "conversation_plan_slice":
        prior_plan = _find_recent_assistant_plan_text(conversation_history)
        if prior_plan:
            sections.append("Most recent semester-by-semester plan from the conversation:\n" + prior_plan[:3500])
        if draft_response:
            sections.append("Deterministic helper suggestion for the requested scope:\n" + draft_response[:1500])
    elif draft_response:
        sections.append("Supporting draft or retrieved answer:\n" + draft_response[:1500])

    source = metadata.get("source")
    if source:
        sections.append(f"Draft source: {source}")

    search_resolution = metadata.get("search_resolution")
    if isinstance(search_resolution, dict):
        source_urls = list(search_resolution.get("source_urls") or [])
        if source_urls:
            sections.append("Verified web sources: " + ", ".join(source_urls[:4]))
        if search_resolution.get("exhaustion_reason"):
            sections.append(f"Search exhaustion reason: {search_resolution.get('exhaustion_reason')}")

    return "\n\n".join(section for section in sections if section).strip()


def _build_transcript_plan_draft(
    plan: OrchestrationPlan,
    user_question: str,
    conversation_history: Optional[List[dict]],
    has_transcript_for_plan: bool,
) -> Tuple[str, List[str], Dict[str, Any]]:
    tools_used = ["deterministic_planner"]

    if has_transcript_for_plan:
        full_plan_text = aggregate_and_generate_response(user_question)
        if plan.plan_scope in {"single_semester", "multi_semester"}:
            sliced = _slice_semester_plan_text(full_plan_text, plan.semester_horizon)
            if sliced:
                return sliced, ["transcript_state"] + tools_used, {"source": "transcript_plan"}
        return full_plan_text, ["transcript_state"] + tools_used, {"source": "transcript_plan"}

    major_from_text, year_from_text = _extract_major_year_from_request_context(
        plan.corrected_question or user_question,
        conversation_history,
    )
    if not major_from_text or not year_from_text:
        return (
            (
                "Happy to put together a semester plan for you! To make it accurate I just need two quick "
                "details:\n\n"
                "- Your **major** (e.g., Computer Science, Business, Biology)\n"
                "- Your current **year level** (freshman, sophomore, junior, or senior)\n\n"
                "Once you share those (for example, 'Computer Science sophomore' or 'Business junior'), I "
                "can build a 15-credit plan that matches your stage in the program. If you also upload your "
                "academic transcript, I can personalize it around what you've already completed.\n\n"
                "What's your major and year level?"
            ),
            [],
            {"source": "missing_major_year"},
        )

    full_plan_text = generate_plan_from_major_and_year(major_from_text, year_from_text)
    if plan.plan_scope in {"single_semester", "multi_semester"}:
        sliced = _slice_semester_plan_text(full_plan_text, plan.semester_horizon)
        if sliced:
            return sliced, tools_used, {"source": "major_year_plan", "major": major_from_text, "year": year_from_text}
    return full_plan_text, tools_used, {"source": "major_year_plan", "major": major_from_text, "year": year_from_text}


def _build_transcript_audit_draft(student_data: Optional[dict], user_question: str) -> str:
    if not student_data:
        return "I couldn't read a transcript. Please upload one so I can answer transcript-specific questions."

    credits_needed = student_data.get("credits_needed", "unknown")
    credits_completed = student_data.get("credits_completed", "unknown")
    unmet = student_data.get("unmet_requirements", [])
    if unmet and all(
        isinstance(x, dict) and x.get("category", "").strip().lower() in {"", "requirement"}
        for x in unmet
    ) and stored_pdf_docs:
        try:
            parsed_unmet = extract_not_satisfied_sections(stored_pdf_docs)
            if parsed_unmet:
                unmet = parsed_unmet
        except Exception as e:
            print(f"⚠️ Could not enrich unmet requirement details: {e}")

    def _format_unmet_item(item) -> str:
        if isinstance(item, dict):
            category = str(item.get("category", "Requirement")).strip() or "Requirement"
            needed = item.get("needed", 0)
            used = item.get("used", None)
            required = item.get("required", None)
            if required is not None and used is not None:
                return f"- {category}: {needed} credits needed ({used}/{required} completed)"
            return f"- {category}: {needed} credits needed"
        if isinstance(item, (list, tuple)):
            return f"- Requirement: {', '.join(str(x) for x in item)}"
        return f"- Requirement: {str(item)}"

    unmet_summary = "\n".join(_format_unmet_item(item) for item in unmet[:8]) if unmet else "No unmet requirements found."
    return (
        f"Academic Audit Summary\n"
        f"- Major: {student_data.get('major', 'Unknown')}\n"
        f"- Credits completed: {credits_completed}\n"
        f"- Credits remaining: {credits_needed}\n"
        f"- In-progress courses: {', '.join(student_data.get('in_progress_courses', [])[:8]) if student_data.get('in_progress_courses') else 'None'}\n"
        f"- Recently completed courses: {', '.join(student_data.get('completed_courses', [])[-8:]) if student_data.get('completed_courses') else 'Not detected'}\n"
        f"- Key unmet requirements:\n{unmet_summary}\n\n"
        f"Next step: ask 'recommend me courses for next 2 semesters' for concrete scheduling, or ask for a full graduation plan."
    )


def _build_course_lookup_draft(user_question: str, trace: Optional[Dict[str, Any]], user_id: str = "default") -> Tuple[str, List[str], Dict[str, Any]]:
    return _build_course_lookup_draft_with_tools(user_question, trace, None, user_id=user_id)


def _build_course_lookup_draft_with_tools(
    user_question: str,
    trace: Optional[Dict[str, Any]],
    allowed_tools: Optional[set],
    user_id: str = "default",
) -> Tuple[str, List[str], Dict[str, Any]]:
    expanded_question = _expand_academic_query_terms(user_question)
    course_match = re.search(r'\b([A-Z]{2,})\s+(\d{3}[A-Z]?)\b', user_question, re.IGNORECASE)
    if course_match and (allowed_tools is None or "prerequisite_graph" in allowed_tools):
        course_code = f"{course_match.group(1).upper()} {course_match.group(2)}"
        try:
            if course_graph:
                prereq_text = get_course_prerequisites(course_code, course_graph)
                return prereq_text, ["prerequisite_graph"], {"source": "prerequisite_graph"}
        except Exception as e:
            print(f"⚠️ Could not fetch prerequisites from graph: {e}")

    if allowed_tools is None or "course_retriever" in allowed_tools:
        course_info = process_psu_courses(expanded_question)
        if course_info and "No relevant courses found" not in course_info and "not available" not in course_info.lower():
            return course_info, ["course_retriever"], {"source": "course_retriever"}

    if allowed_tools is None or "web_search" in allowed_tools:
        search_result = _run_shared_search_pipeline(
            user_id=user_id,
            question_text=expanded_question,
            conversation_history=None,
            trace=trace,
            preferred_strategy="generic_authoritative_lookup",
            preferred_domains=["psu.edu", "pennstate.edu"],
            required_domain_mode="preferred",
        )
        web_response = _research_answer_text(search_result)
        if web_response:
            return web_response, ["web_search"], {
                "source": "shared_search_pipeline",
                **_resolved_search_to_metadata(search_result),
            }
    return (
        (
            "I want to make sure I give you the right course details, but I couldn't locate that course in the "
            "catalog. Could you double-check the course code (Penn State codes look like 'CMPSC 131' or "
            "'MATH 140')? If you share the subject area or what you're hoping the course covers, I can also "
            "suggest a few alternatives."
        ),
        [],
        {"source": "course_lookup_not_found"},
    )


def _build_policy_lookup_draft(
    user_question: str,
    conversation_history: Optional[List[dict]],
    trace: Optional[Dict[str, Any]],
    allowed_tools: Optional[set] = None,
    retry_reason: Optional[str] = None,
    user_id: str = "default",
) -> Tuple[str, List[str], Dict[str, Any]]:
    response = None
    tools_used: List[str] = []
    metadata: Dict[str, Any] = {"source": "policy_lookup"}
    contextual_question = user_question
    execution_profile = _build_execution_profile(
        contextual_question,
        "policy",
        {"tools": {"web_search": True}},
    )
    requires_exact_date = execution_profile.get("verification_mode") == "date"

    if requires_exact_date and (allowed_tools is None or "web_search" in allowed_tools):
        search_result = _run_shared_search_pipeline(
            user_id=user_id,
            question_text=contextual_question,
            conversation_history=conversation_history,
            trace=trace,
            preferred_strategy="deadline_exact_date",
            preferred_domains=["abington.psu.edu", "psu.edu", "pennstate.edu"],
            required_domain_mode="strict",
        )
        exact_date_response = _research_answer_text(search_result)
        if exact_date_response:
            metadata.update(_resolved_search_to_metadata(search_result))
            return exact_date_response, ["web_search"], metadata

    if (allowed_tools is None or "policy_retriever" in allowed_tools) and policy_retriever:
        try:
            policy_docs = policy_retriever.invoke(contextual_question)
            if policy_docs:
                tools_used.append("policy_retriever")
                policy_context = "\n\n".join(doc.page_content[:900] for doc in policy_docs[:4])
                response = llm2.invoke([
                    SystemMessage(content=(
                        ADVISOR_PERSONA
                        + "\n\nFor this turn you are the POLICY ANSWERER. Answer a Penn State policy, "
                        "deadline, registration, or advising-process question using ONLY the provided policy text.\n\n"
                        + PRIVATE_REASONING_GUIDANCE
                        + "\n\nPrivate answer checklist (do not reveal):\n"
                        "1. Identify the exact policy question and whether it asks for a date, process, eligibility, or office.\n"
                        "2. Locate the supporting sentence(s) in the provided policy context.\n"
                        "3. Answer only what the context supports.\n"
                        "4. If the context does not answer it, say that plainly and suggest the correct office to contact.\n\n"
                        "Output guidance with examples:\n"
                        "- Direct supported answer: 'The available policy text says students should contact the Registrar for X.'\n"
                        "- Missing answer: 'I do not see that deadline in the policy text provided here, so confirm it with the Registrar.'\n"
                        "- Exact-date questions: include the concrete date only if it appears in the context. Example: "
                        "'The late-drop deadline shown in the provided text is April 5, 2026.'\n\n"
                        "Output contract:\n"
                        "- Return student-facing prose only.\n"
                        "- Put the direct answer first, then any grounded next steps.\n"
                        "- Do not include JSON, markdown tables, internal notes, or private reasoning.\n\n"
                        "Hard rules:\n"
                        "- Do not guess dates, deadlines, URLs, office names, or policy details.\n"
                        "- Do not cite internal retrieval mechanics.\n"
                        "- Keep the answer concise, then add grounded next steps when helpful."
                    )),
                    HumanMessage(content=f"{_get_current_datetime_context()}\nPolicy context:\n{policy_context}\n\nQuestion: {contextual_question}"),
                ]).content

                if requires_exact_date:
                    if not _has_concrete_date_in_text(response):
                        if allowed_tools is None or "web_search" in allowed_tools:
                            exact_date_response = _lookup_exact_deadline_dates(
                                contextual_question,
                                conversation_history=conversation_history,
                                trace=trace,
                                user_id=user_id,
                            )
                            if exact_date_response:
                                return exact_date_response, tools_used + ["web_search"], metadata
                            web_response = _web_search_and_synthesize(
                                contextual_question,
                                date_only=True,
                                conversation_history=conversation_history,
                                prefer_current_or_future=True,
                                trace=trace,
                                user_id=user_id,
                            )
                            if web_response and web_response != "DATE_NOT_FOUND":
                                return _format_deadline_answer(web_response, contextual_question, conversation_history), tools_used + ["web_search"], metadata
                    else:
                        local_dates = _extract_dates_from_text(response)
                        if local_dates:
                            return _format_deadline_answer(local_dates[0], contextual_question, conversation_history), tools_used, metadata

                if not _response_indicates_no_answer_found(response):
                    return response, tools_used, metadata
        except Exception as e:
            print(f"⚠️ Could not retrieve policy information: {e}")

    if allowed_tools is None or "web_search" in allowed_tools:
        preferred_strategy = "deadline_exact_date" if requires_exact_date else "generic_authoritative_lookup"
        search_result = _run_shared_search_pipeline(
            user_id=user_id,
            question_text=contextual_question,
            conversation_history=conversation_history,
            trace=trace,
            preferred_strategy=preferred_strategy,
            preferred_domains=["abington.psu.edu", "psu.edu", "pennstate.edu"],
            required_domain_mode="strict" if requires_exact_date else "preferred",
        )
        web_response = _research_answer_text(search_result)
        if web_response:
            metadata.update(_resolved_search_to_metadata(search_result))
            return web_response, tools_used + ["web_search"], metadata
    return "I couldn't find that policy information. Please rephrase your question or check the Penn State website directly.", tools_used, metadata


def _build_semester_specific_major_draft(
    user_question: str,
    target: Dict[str, Any],
    conversation_history: Optional[List[dict]],
    has_transcript_for_plan: bool,
    trace: Optional[Dict[str, Any]],
    allowed_tools: Optional[set],
    user_id: str = "default",
) -> Tuple[str, List[str], Dict[str, Any]]:
    """Generate the concrete courses for a specific target semester of a major.

    Behavior contract (matches the product requirement):
      - No transcript uploaded: return the courses that appear in the standard
        graduation plan for that semester of the major/option. 15 credits,
        deterministic (driven by ``generate_plan_from_major_and_year`` which
        uses the course graph + CSV requirements, not the LLM).
      - Transcript uploaded: return a personalized recommendation for the
        student's next semester based on what they still need to satisfy
        (delegated to ``aggregate_and_generate_response``).
    """
    target_semester = int(target.get("semester_number") or 0)
    year_level = target.get("year_level") or "freshman"
    horizon = max(1, int(target.get("horizon") or 1))

    if has_transcript_for_plan:
        try:
            full_plan_text = aggregate_and_generate_response(user_question)
            tools_used = ["transcript_state", "deterministic_planner"]
            sliced = _slice_semester_plan_text(full_plan_text, horizon) or full_plan_text
            return sliced, tools_used, {
                "source": "personalized_semester_plan",
                "target_semester": target_semester,
                "horizon": horizon,
            }
        except Exception as e:
            print(f"⚠️ Personalized semester-specific plan failed: {e}")

    major_from_text, _year_from_text = _extract_major_year_from_request_context(
        user_question, conversation_history
    )
    if not major_from_text:
        return "", [], {"source": "semester_specific_missing_major"}

    try:
        full_plan_text = generate_plan_from_major_and_year(major_from_text, year_level)
    except Exception as e:
        print(f"⚠️ Deterministic semester-specific plan failed: {e}")
        return "", [], {"source": "semester_specific_planner_failed", "error": str(e)}

    if not full_plan_text or full_plan_text.startswith("I couldn't"):
        return full_plan_text or "", [], {
            "source": "semester_specific_no_plan",
            "major": major_from_text,
            "year_level": year_level,
        }

    tools_used = ["deterministic_planner"]
    if horizon == 1:
        block = _extract_semester_block_by_number(full_plan_text, target_semester)
        if block:
            return block, tools_used, {
                "source": "major_year_semester_plan",
                "target_semester": target_semester,
                "major": major_from_text,
                "year_level": year_level,
                "horizon": 1,
            }

    sliced = _slice_semester_plan_text(full_plan_text, horizon) or full_plan_text
    return sliced, tools_used, {
        "source": "major_year_semester_plan",
        "target_semester": target_semester,
        "major": major_from_text,
        "year_level": year_level,
        "horizon": horizon,
    }


# =============================================================================
# Executor tool registry — single source of truth for "data retrieval as a tool"
#
# The orchestrator now produces a ``retrieval_plan`` (an ordered list of
# {tool, query, reason} steps). The executor stops deciding which tools to
# call on its own; it dispatches every step through this registry. Each
# callable takes a query string plus a context dict and returns formatted
# retrieval text (empty string if nothing was found). Errors are swallowed
# inside each wrapper so a failing tool never crashes the dispatcher.
# =============================================================================


def _call_majors_retriever(query: str, ctx: Dict[str, Any]) -> str:
    if majors_retriever is None:
        # #region debug log (H-R1: retriever None-state)
        _dbg("H-R1", "main_copy.py:_call_majors_retriever", "retriever is None", {
            "query_first_120": (query or "")[:120],
        })
        # #endregion
        return ""
    try:
        docs = majors_retriever.invoke(query)
    except Exception as e:
        # #region debug log (H-R2: retriever exception)
        _dbg("H-R2", "main_copy.py:_call_majors_retriever", "exception", {
            "query_first_120": (query or "")[:120],
            "error_type": type(e).__name__,
            "error_str": str(e)[:300],
        })
        # #endregion
        print(f"⚠️ majors_retriever failed: {e}")
        return ""
    # #region debug log (H-R3: docs returned)
    _dbg("H-R3", "main_copy.py:_call_majors_retriever", "docs returned", {
        "query_first_120": (query or "")[:120],
        "docs_count": len(docs or []),
        "docs_sizes": [len(getattr(d, "page_content", "") or "") for d in (docs or [])[:6]],
        "doc0_head_200": ((docs[0].page_content or "") if docs else "")[:200],
    })
    # #endregion
    if not docs:
        return ""
    return "\n\n".join(doc.page_content[:1200] for doc in docs[:4])


def _call_course_retriever(query: str, ctx: Dict[str, Any]) -> str:
    if course_retriever is None:
        # #region debug log (H-R1: retriever None-state)
        _dbg("H-R1", "main_copy.py:_call_course_retriever", "retriever is None", {
            "query_first_120": (query or "")[:120],
        })
        # #endregion
        return ""
    try:
        expanded = _expand_academic_query_terms(query)
        docs = course_retriever.invoke(expanded)
    except Exception as e:
        # #region debug log (H-R2: retriever exception)
        _dbg("H-R2", "main_copy.py:_call_course_retriever", "exception", {
            "query_first_120": (query or "")[:120],
            "expanded_first_120": (expanded if 'expanded' in locals() else "")[:120],
            "error_type": type(e).__name__,
            "error_str": str(e)[:300],
        })
        # #endregion
        print(f"⚠️ course_retriever failed: {e}")
        return ""
    # #region debug log (H-R3: docs returned)
    _dbg("H-R3", "main_copy.py:_call_course_retriever", "docs returned", {
        "query_first_120": (query or "")[:120],
        "expanded_first_120": (expanded or "")[:120],
        "docs_count": len(docs or []),
        "docs_sizes": [len(getattr(d, "page_content", "") or "") for d in (docs or [])[:8]],
        "doc0_head_200": ((docs[0].page_content or "") if docs else "")[:200],
    })
    # #endregion
    if not docs:
        return ""
    return "\n\n".join(doc.page_content[:700] for doc in docs[:8])


def _call_prerequisite_graph(query: str, ctx: Dict[str, Any]) -> str:
    """Look up enrollment + concurrent prereqs for every course code in the query.

    The planner's query is expected to be a comma-separated or whitespace list
    of candidate course codes (e.g. "CMPSC 131,CMPSC 132,MATH 140"). We extract
    every PSU-style code and call the underlying graph for each. The output
    format mirrors ``get_course_prerequisites`` so the synthesizer (and the
    deterministic prereq validator) can parse it predictably.
    """
    if course_graph is None:
        return ""
    # 1-3 digit numbers with optional single-letter suffix to cover the full
    # PSU catalog shape (SOC 1, ART 11, MATH 21, ENGL 15, CMPSC 131, ART 122Y).
    codes = re.findall(r"\b([A-Z]{2,})\s+(\d{1,3}[A-Z]?)\b", query, flags=re.IGNORECASE)
    if not codes:
        return ""
    seen: List[str] = []
    parts: List[str] = []
    for prefix, number in codes:
        code = f"{prefix.upper()} {number}"
        if code in seen:
            continue
        seen.append(code)
        try:
            text = get_course_prerequisites(code, course_graph)
        except Exception as e:
            print(f"⚠️ prerequisite_graph failed for {code}: {e}")
            continue
        if text:
            parts.append(text.strip())
    return "\n\n".join(parts)


def _call_policy_retriever(query: str, ctx: Dict[str, Any]) -> str:
    if policy_retriever is None:
        return ""
    try:
        docs = policy_retriever.invoke(query)
    except Exception as e:
        print(f"⚠️ policy_retriever failed: {e}")
        return ""
    if not docs:
        return ""
    return "\n\n".join(doc.page_content[:900] for doc in docs[:4])


def _call_web_search(query: str, ctx: Dict[str, Any]) -> str:
    user_id = str(ctx.get("user_id") or "default")
    trace = ctx.get("trace")
    try:
        result = _run_shared_search_pipeline(
            user_id=user_id,
            question_text=query,
            conversation_history=None,
            trace=trace,
            preferred_strategy="generic_authoritative_lookup",
            preferred_domains=["abington.psu.edu", "psu.edu", "pennstate.edu"],
            required_domain_mode="preferred",
        )
    except Exception as e:
        print(f"⚠️ web_search failed: {e}")
        return ""
    return _research_answer_text(result) or ""


# Registry keyed by the same tool names the orchestrator uses in
# ``allowed_tools`` and ``retrieval_plan``. Adding a new tool is a one-line
# change here plus an entry in ``_TOOLS_BY_PATH``.
EXECUTOR_TOOLS: Dict[str, Callable[[str, Dict[str, Any]], str]] = {
    "majors_retriever":    _call_majors_retriever,
    "course_retriever":    _call_course_retriever,
    "prerequisite_graph":  _call_prerequisite_graph,
    "policy_retriever":    _call_policy_retriever,
    "web_search":          _call_web_search,
}


# PSU course numbers are 2-3 digits with an optional single-letter suffix
# (e.g. ART 11, MATH 21, ENGL 15, CMPSC 131, ENGL 202D, ART 122Y). Older
# regexes in this file require exactly 3 digits, which silently drops every
# 2-digit code and is part of the reason the user's Art + Accounting plan
# could not retrieve prereqs for ART 11. We also accept 1-digit PSU codes
# (SOC 1, CAS 1, ART 1) — confirmed by runtime logs from the Psych double
# major, where "SOC 1" escaped both the rewrite and the backfill scan.
_COURSE_CODE_FINDER = re.compile(r"\b([A-Z]{2,})\s+(\d{1,3}[A-Z]?)\b")
_PREREQ_HEADER_FINDER = re.compile(
    r"Prerequisites for\s+([A-Z]{2,})\s+(\d{1,3}[A-Z]?)\s*:", re.IGNORECASE
)
_PREREQ_NOT_FOUND_FINDER = re.compile(
    r"No course named '\s*([A-Z]{2,})\s+(\d{1,3}[A-Z]?)\s*' found\.", re.IGNORECASE
)


def _normalize_course_code(prefix: str, number: str) -> str:
    return f"{prefix.upper().strip()} {number.upper().strip()}"


def _collect_codes_from_text(text: str) -> List[str]:
    """Return every PSU-style course code found in ``text`` (deduped, ordered)."""
    if not text:
        return []
    seen: Dict[str, None] = {}  # ordered set
    for prefix, number in _COURSE_CODE_FINDER.findall(text):
        seen[_normalize_course_code(prefix, number)] = None
    return list(seen.keys())


def _collect_codes_already_attempted(prereq_text: str) -> set:
    """Course codes the prereq_graph already TRIED to look up.

    A code counts as "attempted" if either:
      - the bundle contains a ``Prerequisites for X:`` header for it (success), or
      - the bundle contains a ``No course named 'X' found.`` line for it (tried-and-empty).

    Both shapes mean the executor already asked the graph about that code, so
    a follow-up retrieval would just repeat the same call. The backfill pass
    only targets codes that have been NEITHER attempted nor returned.
    """
    attempted: set = set()
    if not prereq_text:
        return attempted
    for prefix, number in _PREREQ_HEADER_FINDER.findall(prereq_text):
        attempted.add(_normalize_course_code(prefix, number))
    for prefix, number in _PREREQ_NOT_FOUND_FINDER.findall(prereq_text):
        attempted.add(_normalize_course_code(prefix, number))
    return attempted


def _backfill_missing_prereqs(
    bundle: Dict[str, Dict[str, str]],
    plan: OrchestrationPlan,
    ctx: Dict[str, Any],
    tools_used: List[str],
) -> None:
    """Self-consistency pass: backfill prereq retrieval for every course in
    the bundle that doesn't yet have a prereq entry.

    Why this exists: a planner can be lazy and only enumerate a handful of
    course codes in its ``prerequisite_graph`` step. The bundle that comes
    back will then have ``majors_retriever`` results listing dozens of
    courses while ``prerequisite_graph`` only covered six. Without this
    pass the synthesizer is forced to write ``(prerequisites: not retrieved)``
    on every uncovered course — which is exactly the bug the user reported
    on the Art + Accounting double-major plan.

    The pass is intentionally generic: it does NOT know anything about Art
    or Accounting or which majors the planner picked. It just enforces the
    bundle invariant "every course code we mention should also have a
    prereq lookup attempt", which is the universal contract the synthesizer
    relies on. Mutates ``bundle`` and ``tools_used`` in place.
    """
    if "prerequisite_graph" not in EXECUTOR_TOOLS:
        return
    if plan.allowed_tools and "prerequisite_graph" not in plan.allowed_tools:
        return
    callable_ = EXECUTOR_TOOLS.get("prerequisite_graph")
    if callable_ is None:
        return

    # Collect every code the bundle mentions across ALL tools.
    mentioned: Dict[str, None] = {}
    for payload in bundle.values():
        for code in _collect_codes_from_text(payload.get("results", "") or ""):
            mentioned[code] = None
    if not mentioned:
        return

    existing_prereq_text = ""
    if "prerequisite_graph" in bundle:
        existing_prereq_text = bundle["prerequisite_graph"].get("results", "") or ""
    attempted = _collect_codes_already_attempted(existing_prereq_text)

    # Anything we've never asked the graph about needs a follow-up call.
    missing = [code for code in mentioned if code not in attempted]
    # #region debug log (H-B3: backfill scope)
    _dbg("H-B3", "main_copy.py:_backfill_missing_prereqs", "scope", {
        "mentioned_count": len(mentioned),
        "mentioned_sample": list(mentioned.keys())[:30],
        "attempted_count": len(attempted),
        "attempted_sample": list(attempted)[:30],
        "missing_count": len(missing),
        "missing_sample": missing[:30],
    })
    # #endregion
    if not missing:
        return

    # Chunk the lookup so a single huge query string never blows past
    # tokenizer limits. 25 codes per call is well under any reasonable cap.
    new_chunks: List[str] = []
    for i in range(0, len(missing), 25):
        chunk = missing[i:i + 25]
        query = ", ".join(chunk)
        try:
            chunk_text = callable_(query, ctx) or ""
        except Exception as e:  # pragma: no cover - tool wrapper already swallows
            print(f"⚠️ prereq backfill failed for chunk {i}: {e}")
            continue
        if chunk_text:
            new_chunks.append(chunk_text)

    if not new_chunks:
        return
    backfill_text = "\n\n".join(new_chunks)

    if "prerequisite_graph" in bundle:
        bundle["prerequisite_graph"]["results"] = (
            (existing_prereq_text + "\n\n---\nAuto-backfill for codes the "
             "planner did not pre-list:\n\n" + backfill_text).strip()
        )
        # Note in the reason that we extended the planner's query so the
        # executor's metadata accurately reflects what was retrieved.
        prior_reason = bundle["prerequisite_graph"].get("reason", "") or ""
        bundle["prerequisite_graph"]["reason"] = (
            (prior_reason + " (auto-extended to cover every code mentioned in the bundle)").strip()
        )
    else:
        bundle["prerequisite_graph"] = {
            "query": "auto-backfill: " + ", ".join(missing[:25]) + (
                f" (and {len(missing) - 25} more)" if len(missing) > 25 else ""
            ),
            "reason": (
                "Bundle self-consistency: every course mentioned in the bundle must have "
                "a prereq lookup attempt before synthesis."
            ),
            "results": backfill_text,
        }
        if "prerequisite_graph" not in tools_used:
            tools_used.append("prerequisite_graph")


# Match the WHOLE inline prereq parenthetical so we can replace it as a unit.
# Accepts "(prerequisites: ...)", "(prereq: ...)", "(prereqs: ...)", etc.
_INLINE_PREREQ_REPLACE_RE = re.compile(
    r"\(\s*prereq(?:uisite)?s?\s*:\s*[^)]*\)",
    re.IGNORECASE,
)

# Match a single "Prerequisites for X:" block in the graph output, capturing
# the raw body (everything until the next header or end-of-text). We preserve
# the body VERBATIM rather than re-parsing it into codes — the parser silently
# drops 2-digit codes (e.g., "Math 22" in MATH 140's prereq), which is
# exactly what made the LLM say "(prerequisites: None)" for MATH 140.
_PREREQ_BLOCK_HEADER_RE = re.compile(
    r"Prerequisites for\s+([A-Z]{2,})\s+(\d{1,3}[A-Z]?)\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_ENFORCED_PREREQ_LINE_RE = re.compile(
    r"-\s*Enforced prerequisite at enrollment\s*:\s*(.*?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _extract_raw_enforced_prereq_map(prereq_graph_text: str) -> Dict[str, str]:
    """Return ``{COURSE_CODE: raw enforced-prereq text}`` from the graph.

    Unlike :func:`_parse_prereq_graph_text` (which extracts codes only and
    silently drops 2-digit prereqs), this preserves the verbatim text from
    the "Enforced prerequisite at enrollment:" line. That's what we hand to
    the rewrite — accurate verbose text beats concise wrong text every
    time.
    """
    result: Dict[str, str] = {}
    if not prereq_graph_text:
        return result
    headers = list(_PREREQ_BLOCK_HEADER_RE.finditer(prereq_graph_text))
    for idx, hm in enumerate(headers):
        code = _normalize_course_code(hm.group(1), hm.group(2))
        body_start = hm.end()
        body_end = headers[idx + 1].start() if idx + 1 < len(headers) else len(prereq_graph_text)
        body = prereq_graph_text[body_start:body_end]
        em = _ENFORCED_PREREQ_LINE_RE.search(body)
        if em:
            payload = em.group(1).strip()
        else:
            payload = "None"
        result[code] = payload or "None"
    return result


def _rewrite_inline_prereqs_from_graph(
    response_text: str,
    prereq_graph_text: str,
) -> str:
    """Overwrite every '(prerequisites: ...)' annotation with verified text
    from the prereq graph.

    Why this exists: the synthesizer LLM is the wrong tool for transcribing
    prereq lists. Given a wall of graph data, it routinely either drops a
    real prereq (writes 'None' for MATH 140) or upgrades a 'concurrent'
    option to a fake enforced prereq (writes 'ARTH 250' for PHOTO 201). A
    short deterministic rewrite eliminates both failure modes — we let the
    LLM keep writing the annotation in its preferred shape, then replace
    the parenthetical contents with the truth from the graph before
    shipping.

    For each line that already carries an inline '(prerequisites: ...)':
      * Find the FIRST course code on the line that's known to the graph
        (this is the course being recommended; codes after 'examples:' on
        categorized placeholder lines are sample picks, not recommendations).
      * Replace the parenthetical with the verbatim "Enforced prerequisite
        at enrollment" text from the graph, or "(prerequisites: None)" when
        the graph says None.
      * Lines whose course is not in the graph are left untouched.
    """
    if not response_text or not prereq_graph_text:
        # #region debug log (H-B1: rewrite no-op early)
        _dbg("H-B1", "main_copy.py:_rewrite_inline_prereqs_from_graph", "early-exit empty input", {
            "response_len": len(response_text or ""),
            "graph_text_len": len(prereq_graph_text or ""),
        })
        # #endregion
        return response_text
    raw_map = _extract_raw_enforced_prereq_map(prereq_graph_text)
    # #region debug log (H-B1+H-B2: extracted graph map)
    _dbg("H-B1+H-B2", "main_copy.py:_rewrite_inline_prereqs_from_graph", "extracted raw_map", {
        "map_size": len(raw_map),
        "map_keys_sample": list(raw_map.keys())[:40],
        "graph_text_len": len(prereq_graph_text or ""),
    })
    # #endregion
    if not raw_map:
        return response_text

    skipped: List[Dict[str, Any]] = []

    def fix_line(line: str) -> str:
        if not _INLINE_PREREQ_REPLACE_RE.search(line):
            return line
        codes = _COURSE_CODE_FINDER.findall(line)
        payload: Optional[str] = None
        normalized_codes: List[str] = []
        for prefix, number in codes:
            up = _normalize_course_code(prefix, number)
            normalized_codes.append(up)
            if up in raw_map:
                payload = raw_map[up]
                break
        if payload is None:
            skipped.append({
                "line_first_200": line.strip()[:200],
                "codes_on_line": normalized_codes,
            })
            return line
        clean = payload.strip().rstrip(".")
        if clean.lower() in {"", "none"}:
            new_annotation = "(prerequisites: None)"
        else:
            new_annotation = f"(prerequisites: {clean})"
        return _INLINE_PREREQ_REPLACE_RE.sub(new_annotation, line, count=1)

    out = "\n".join(fix_line(line) for line in response_text.splitlines())
    # #region debug log (H-B2: lines the rewrite couldn't fix)
    _dbg("H-B1+H-B2", "main_copy.py:_rewrite_inline_prereqs_from_graph", "lines skipped (no map hit)", {
        "skipped_count": len(skipped),
        "skipped_sample": skipped[:15],
    })
    # #endregion
    return out


def _execute_retrieval_plan(
    plan: OrchestrationPlan,
    trace: Optional[Dict[str, Any]] = None,
    user_id: str = "default",
) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    """Dispatch every step in ``plan.retrieval_plan`` through ``EXECUTOR_TOOLS``.

    Returns a labeled bundle keyed by tool name and the list of tools that
    actually returned content (this becomes ``tools_used`` for downstream
    validation). If the planner lists the same tool twice, the second call's
    output is concatenated with a separator so both queries' results survive.

    After the planner-defined steps complete, a structural self-consistency
    pass (``_backfill_missing_prereqs``) makes one final ``prerequisite_graph``
    call covering every course code the bundle mentions but the planner's
    prereq query missed. This is what stops the synthesizer from writing
    ``(prerequisites: not retrieved)`` on courses it actually could have
    looked up.
    """
    bundle: Dict[str, Dict[str, str]] = {}
    tools_used: List[str] = []
    if not plan.retrieval_plan:
        return bundle, tools_used

    ctx: Dict[str, Any] = {"user_id": user_id, "trace": trace, "plan": plan}
    # #region debug log (H-A1: per-step retrieval contract)
    _dbg("H-A1+H-B1", "main_copy.py:_execute_retrieval_plan", "plan steps about to execute", {
        "plan_majors": list(getattr(plan, "majors", None) or []),
        "step_count": len(plan.retrieval_plan or []),
        "steps": [
            {"tool": str(s.get("tool", "")), "query": str(s.get("query", ""))[:120]}
            for s in (plan.retrieval_plan or [])
        ],
    })
    # #endregion
    for step in plan.retrieval_plan:
        tool = str(step.get("tool", "")).strip()
        query = str(step.get("query", "")).strip()
        reason = str(step.get("reason", "")).strip()
        if not tool or not query:
            continue
        if plan.allowed_tools and tool not in plan.allowed_tools:
            continue
        callable_ = EXECUTOR_TOOLS.get(tool)
        if callable_ is None:
            continue
        results = callable_(query, ctx) or ""
        # #region debug log (H-B1: per-tool result size)
        _dbg("H-A1+H-B1", "main_copy.py:_execute_retrieval_plan", "tool returned", {
            "tool": tool,
            "query_first_120": query[:120],
            "result_len": len(results or ""),
            "result_head_300": (results or "")[:300],
        })
        # #endregion
        if not results:
            continue
        if tool in bundle:
            bundle[tool]["query"] = bundle[tool]["query"] + " | " + query
            bundle[tool]["reason"] = bundle[tool]["reason"] or reason
            bundle[tool]["results"] = bundle[tool]["results"] + "\n\n---\n\n" + results
        else:
            bundle[tool] = {"query": query, "reason": reason, "results": results}
            tools_used.append(tool)

    # Bundle self-consistency: the planner's prereq step is allowed to be
    # narrow ("ART 11, ART 110, ART 111, ART 122, ACCTG 211, ACCTG 212") even
    # though the actual bundle ends up mentioning 30+ codes. We backfill so
    # every recommendable course has a prereq attempt before synthesis.
    if _is_course_recommendation_deliverable(plan):
        _backfill_missing_prereqs(bundle, plan, ctx, tools_used)
    return bundle, tools_used


def _format_retrieval_bundle_for_prompt(bundle: Dict[str, Dict[str, str]]) -> str:
    """Render the bundle as labeled sections the synthesis LLM can read.

    Each section is clearly headed by tool name so the model — and the tests —
    can verify which tool produced which evidence."""
    if not bundle:
        return "(no retrieval results available)"
    lines: List[str] = []
    for tool, payload in bundle.items():
        lines.append(f"### {tool}")
        if payload.get("reason"):
            lines.append(f"_reason_: {payload['reason']}")
        lines.append(f"_query_: {payload.get('query', '')}")
        lines.append("")
        lines.append((payload.get("results") or "").strip())
        lines.append("")
    return "\n".join(lines).strip()


def _multi_major_rule_block(plan: Optional["OrchestrationPlan"]) -> str:
    """Inject the MULTI-MAJOR RULE into the synthesis prompt when applicable.

    Returns an empty string for single-major (or unknown-major) plans so the
    rest of the prompt is unchanged. For double/multi-major plans it lists
    every requested major and tells the synthesizer it MUST cover required
    courses from each, label which course belongs to which major, and call
    out shared / double-counting courses explicitly.
    """
    majors = list(getattr(plan, "majors", None) or []) if plan is not None else []
    if len(majors) <= 1:
        return ""
    listed = ", ".join(majors)
    return (
        "MULTI-MAJOR RULE (enforced; the response will be re-tried if violated):\n"
        f"The student is pursuing {len(majors)} majors: {listed}. The response MUST "
        "include required courses from EACH of those majors. For every course you "
        "schedule, label which major it serves (e.g. '[Art]', '[Accounting]', or "
        "'[Shared/double-counts]' for a course that satisfies a requirement for "
        "more than one major). Identify shared/double-counting courses explicitly so "
        "the student understands when a single course knocks out two requirements. "
        "Pull the requirements for each major from its dedicated section in the "
        "verified data below. Do NOT ship a single-major plan in response to a "
        "multi-major request.\n\n"
    )


def _build_major_lookup_draft(
    user_question: str,
    trace: Optional[Dict[str, Any]],
    allowed_tools: Optional[set] = None,
    user_id: str = "default",
    plan: Optional[OrchestrationPlan] = None,
    retry_reason: Optional[str] = None,
) -> Tuple[str, List[str], Dict[str, Any]]:
    """Major-lookup executor: planner-driven retrieval + bundle-grounded synthesis.

    Control flow:
      1. Dispatch every step in ``plan.retrieval_plan`` through ``EXECUTOR_TOOLS``.
         If the planner left it empty (or no plan was passed), fall back to the
         canonical major-lookup retrieval set so we never silently skip a tool.
      2. If retrieval returned anything, synthesize a single grounded answer
         using ADVISOR_PERSONA + the labeled bundle + the planner's
         ``synthesis_brief``. The synthesizer is told which tool produced
         which evidence and is required to schedule courses only when the
         ``prerequisite_graph`` section confirms prereqs are satisfied.
      3. If retrieval came back completely empty, escalate to web_search.
      4. If even web_search has nothing, return a warm soft-ask asking the
         student to confirm the major name.
    """
    bundle: Dict[str, Dict[str, str]] = {}
    tools_used: List[str] = []

    if plan is not None and plan.retrieval_plan:
        bundle, tools_used = _execute_retrieval_plan(plan, trace=trace, user_id=user_id)

    if not bundle:
        # Safety net: planner produced no retrieval_plan AND no override step
        # populated it (e.g. degraded retrievers). Run the canonical major-lookup
        # fan-out directly through the registry so we never reach synthesis with
        # an empty bundle.
        ctx = {"user_id": user_id, "trace": trace, "plan": plan}
        for tool in ("majors_retriever", "course_retriever", "prerequisite_graph"):
            if allowed_tools is not None and tool not in allowed_tools:
                continue
            callable_ = EXECUTOR_TOOLS.get(tool)
            if callable_ is None:
                continue
            results = callable_(user_question, ctx) or ""
            if results:
                bundle[tool] = {
                    "query": user_question,
                    "reason": "Canonical retrieval fallback (planner did not provide a step).",
                    "results": results,
                }
                tools_used.append(tool)

    if bundle:
        synthesis_brief = (plan.synthesis_brief if plan is not None else "") or (
            "Build a grounded answer to the student's question using ONLY the verified data shown below. "
            "If they asked for a single semester, return exactly one Semester N: ... Total Credits: N block."
        )
        retry_directive = ""
        if retry_reason == "prerequisite_violation":
            retry_directive = (
                "\n\nIMPORTANT RETRY: The previous attempt scheduled a course whose enrollment "
                "prerequisite was not satisfied in an earlier semester. Re-read the prerequisite_graph "
                "section carefully. For every course you place in Semester N, every code listed under "
                "'Enforced prerequisite at enrollment' MUST appear in some Semester < N of your response, "
                "OR be listed under 'Enforced concurrent at enrollment' (meaning it can be in the same "
                "semester). If you cannot satisfy a prereq, swap the course out for one whose prereqs are "
                "satisfied. Do not repeat the previous violation."
            )
        elif retry_reason == "under_specified_recommendation":
            retry_directive = (
                "\n\nIMPORTANT RETRY: The previous attempt left a slot in a semester block as a bare "
                "placeholder like 'Elective', 'Gen Ed', or 'Option Course'. The student cannot register "
                "for that — every line in a Semester N: ... block MUST be one of:\n"
                "  (a) A specific course code with title pulled from the bundle "
                "(e.g. 'BIOL 110: Biology Concepts and Biodiversity').\n"
                "  (b) A categorized placeholder of the form "
                "'<Tag/Category>: <Bucket name> elective — examples: <CODE 1>, <CODE 2>'. "
                "Use the actual taxonomy that appears in the majors_retriever section "
                "(GH, GA, GS, GN, GQ, GHW, GWS, US, IL, humanities, social science, "
                "natural science, arts, quantification, etc.) and pull example codes "
                "from the course_retriever section. If you know the bucket category at all "
                "(humanities, writing/speaking, social science, etc.) you MUST use this form. "
                "Form (c) is a last resort.\n"
                "  (c) An honest deferral RESERVED for slots whose bucket category is genuinely "
                "unknown: 'Open slot — we don't have a category for this slot yet; "
                "please confirm with your advisor.'\n"
                "Do not repeat the previous bare-placeholder lines."
            )
        elif retry_reason == "missing_inline_prereqs":
            retry_directive = (
                "\n\nIMPORTANT RETRY: The previous attempt recommended one or more courses "
                "in a 'Semester N:' block without showing their prerequisites on the same "
                "line. Every line inside a semester block that contains a course code MUST "
                "end with '(prerequisites: <list>)' or '(prerequisites: None)'. Use the "
                "prereq data shown in the verified data below. If a course's prereqs are "
                "genuinely not in the verified data, DROP that course from the semester "
                "block and replace it with one whose prereqs are known; do NOT write "
                "'(prerequisites: not retrieved)' or any similar placeholder — students "
                "find that confusing. Prose references to courses outside semester blocks "
                "do not need prereq annotations."
            )
        elif retry_reason == "missing_major_coverage":
            requested_majors = ", ".join(getattr(plan, "majors", None) or []) or "(unspecified)"
            retry_directive = (
                "\n\nIMPORTANT RETRY: The previous attempt did not cover every major the "
                f"student asked about. The requested majors are: {requested_majors}. The "
                "response MUST mention each of these majors by name and include required "
                "courses from EACH of them. Label every course with the major it serves "
                "(e.g. '[Art]', '[Accounting]', '[Shared/double-counts]'). Pull each "
                "major's requirements from its dedicated section in the verified data "
                "below. Do not repeat the previous single-major answer."
            )
        system_content = (
            ADVISOR_PERSONA
            + "\n\nFor this turn you are the major-lookup synthesizer. You receive verified "
            "Penn State data below, organized into labeled sections. Ground every Penn State-specific "
            "claim in that data. Use exact course codes and titles as they appear there.\n\n"
            + PRIVATE_REASONING_GUIDANCE
            + "\n\nPrivate synthesis checklist (do not reveal):\n"
            "1. Identify the requested major(s), semester horizon, student year, and whether transcript data is present.\n"
            "2. List candidate courses only from the verified data.\n"
            "3. Check prerequisite_graph before placing each course in a semester.\n"
            "4. Fill gen-ed/elective buckets with categorized labels and examples when exact courses are not verified.\n"
            "5. Confirm the response scope: one semester, multiple semesters, or direct major requirements.\n\n"
            "Required semester block shape:\n"
            "Semester N:\n"
            "COURSE 123: Course Title (prerequisites: None)\n"
            "COURSE 234: Course Title (prerequisites: COURSE 123)\n"
            "Gen Ed (GH - Humanities) — examples: ART 001, HIST 148 (prerequisites: None)\n"
            "Total Credits: 15\n\n"
            "Good examples:\n"
            "- 'PSU 1: First-Year Seminar Abington (prerequisites: None)' when PSU 1 appears in verified course data.\n"
            "- 'CMPSC 488: Senior Project (prerequisites: CMPSC 487W)' only when prerequisite_graph supports that edge.\n"
            "- 'Gen Ed (GQ - Quantification) — examples: STAT 200, MATH 110' when the exact bucket is known but the final choice needs advisor confirmation.\n\n"
            "Bad examples:\n"
            "- 'Elective' with no category or examples.\n"
            "- 'CMPSC 132 (prerequisites: None)' when prerequisite_graph says CMPSC 131 is enforced.\n"
            "- A quoted one-paragraph plan such as 'Semester 1: ... Total Credits: 15'.\n\n"
            "USER-FACING LANGUAGE RULE: never use internal terms in your answer. The student does not "
            "know what the following words mean and must NOT see them: 'retrieved bundle', 'retrieval', "
            "'retrieved', 'bundle', 'majors_retriever', 'course_retriever', 'prerequisite_graph', "
            "'planner', 'executor', 'synthesizer'. If a piece of information isn't in the data below, "
            "just say so in plain student-facing language (e.g., 'this isn't in our course data' or "
            "'you'll want to confirm this with an advisor').\n\n"
            "PREREQUISITE RULES (enforced; the response will be re-tried if violated):\n"
            "- Every course you place in Semester N must have its 'Enforced prerequisite at enrollment' "
            "satisfied by a course scheduled in some Semester < N of THIS SAME response, OR appear in "
            "'Enforced concurrent at enrollment' (which permits the prereq and the course in the same "
            "semester).\n"
            "- For a first-semester / freshman recommendation, only schedule courses whose 'Enforced "
            "prerequisite at enrollment' is None (placement tests and high-school background do not "
            "count as college-level prereqs).\n"
            "- If the verified data has no prereq entry for a course you want to recommend, do NOT "
            "recommend it. Say so explicitly.\n\n"
            "INLINE PREREQUISITE ANNOTATION RULE (enforced; the response will be re-tried if violated):\n"
            "Every line that contains a course code inside a 'Semester N: ... Total Credits: N' block "
            "MUST end with a parenthetical of the form '(prerequisites: <list>)'. Always include the "
            "annotation — the system rewrites the contents from verified data before shipping, so just "
            "make sure each course line has the parenthetical present. Example shape:\n"
            "  'CMPSC 132: Programming and Computation II (prerequisites: CMPSC 131)'\n"
            "  'MATH 140: Calculus I (prerequisites: None)'\n"
            "This is required for every course recommendation; the student needs to see what each "
            "course requires before they can register.\n\n"
            "RECOMMENDATION SPECIFICITY RULES (enforced; the response will be re-tried if violated):\n"
            "Every line inside a 'Semester N: ... Total Credits: N' block MUST be one of:\n"
            "  (a) A specific course code with title, followed by the inline prereq annotation "
            "(see rule above).\n"
            "  (b) A clearly categorized placeholder using the actual gen-ed taxonomy. The line MUST "
            "contain at least one of:\n"
            "      - A gen-ed/program tag: GH, GA, GS, GN, GQ, GHW, GWS, US, IL, GFL.\n"
            "      - A category word: humanities, social science, natural science, arts, quantification, "
            "health and wellness, writing, speaking, cultural diversity, US cultures, international cultures, "
            "foreign language, capstone, honors, seminar.\n"
            "      Whenever possible, append concrete example course codes using the form "
            "'— examples: CODE 1, CODE 2, CODE 3'.\n"
            "      If you can identify the bucket category AT ALL — even just 'humanities' or "
            "'writing/speaking' — you MUST use this form (b). Do NOT fall back to form (c).\n"
                "  (c) An honest deferral RESERVED for slots whose bucket category is genuinely "
                "unknown: 'Open slot — we don't have a category for this slot yet; please confirm "
                "with your advisor.' Do not use this form when you already know the bucket (e.g. "
                "'humanities slot', 'writing/speaking slot') — that is form (b) territory and "
                "apologizing is the wrong move.\n"
            "NEVER write a bare 'Elective', 'Gen Ed', 'General Education', 'Option Course', "
            "'Major Elective', or 'Free Elective' line. Those are not actionable for the student and the "
            "validator will reject the response.\n\n"
            + _multi_major_rule_block(plan)
            + "Other rules:\n"
            "- Prefer a realistic 15-17 credit load unless the question asks otherwise.\n"
            "- If the user asked for a single semester, return exactly one Semester N: ... Total Credits: N block.\n"
            "- If the user asked for multiple semesters, structure your answer semester by semester.\n"
            "- If the verified data does not contain enough information to answer, say so explicitly "
            "and list what is missing in plain language (e.g. 'prereq info for X isn't in our records'). "
            "Do not hallucinate course codes or titles."
            + retry_directive
        )
        human_content = (
            f"Question: {user_question}\n\n"
            f"Brief:\n{synthesis_brief}\n\n"
            f"Verified Penn State data:\n{_format_retrieval_bundle_for_prompt(bundle)}"
        )
        try:
            response = llm2.invoke([
                SystemMessage(content=system_content),
                HumanMessage(content=human_content),
            ]).content
            if response and not _response_indicates_no_answer_found(response):
                # Overwrite every '(prerequisites: ...)' annotation with the
                # verified text from the prereq graph BEFORE the response is
                # returned. The synthesizer LLM consistently misreads this
                # data (writes 'None' for courses that have prereqs, or
                # upgrades concurrent options to fake enforced prereqs);
                # rewriting deterministically from the graph eliminates the
                # entire class of false-info bugs in one pass.
                prereq_text_for_rewrite = ""
                if "prerequisite_graph" in bundle:
                    prereq_text_for_rewrite = bundle["prerequisite_graph"].get("results", "") or ""
                if prereq_text_for_rewrite:
                    response = _rewrite_inline_prereqs_from_graph(response, prereq_text_for_rewrite)
                # #region debug log (H-B1+H-B2: post-rewrite "not retrieved" survivors + multi-major surface)
                _not_retrieved_lines = [
                    ln.strip()[:200]
                    for ln in (response or "").splitlines()
                    if "prerequisites: not retrieved" in ln.lower()
                ]
                _resp_lower_dbg = (response or "").lower()
                _dbg("H-A1+H-A3+H-B1+H-B2", "main_copy.py:_build_major_lookup_draft",
                     "post-rewrite executor draft snapshot", {
                    "response_len": len(response or ""),
                    "not_retrieved_count": len(_not_retrieved_lines),
                    "not_retrieved_sample": _not_retrieved_lines[:10],
                    "mentions_history": "history" in _resp_lower_dbg,
                    "mentions_computer_science": "computer science" in _resp_lower_dbg,
                    "mentions_cs": " cs " in (" " + _resp_lower_dbg + " "),
                    "plan_majors": list(getattr(plan, "majors", None) or []) if plan is not None else [],
                    "deliverable": getattr(plan, "deliverable", None) if plan is not None else None,
                    "executor_path": getattr(plan, "executor_path", None) if plan is not None else None,
                    "tools_used": list(tools_used or []),
                })
                # #endregion
                metadata: Dict[str, Any] = {
                    "source": "major_lookup_rag",
                    "retrieval_bundle": {
                        tool: {
                            "query": payload.get("query", ""),
                            "reason": payload.get("reason", ""),
                            "results_preview": (payload.get("results") or "")[:600],
                        }
                        for tool, payload in bundle.items()
                    },
                    "used_majors_retriever": "majors_retriever" in tools_used,
                    "used_course_retriever": "course_retriever" in tools_used,
                    "used_prerequisite_graph": "prerequisite_graph" in tools_used,
                    "retry_reason": retry_reason,
                }
                # The prereq validator (orchestration_pipeline.validate_final_response)
                # reads this full text to verify every recommended course's enrollment
                # prereqs are satisfied in earlier semesters of the response.
                if "prerequisite_graph" in bundle:
                    metadata["prereq_graph_text"] = bundle["prerequisite_graph"].get("results", "")
                return response, tools_used, metadata
        except Exception as e:
            print(f"⚠️ major_lookup LLM synthesis failed: {e}")

    if allowed_tools is None or "web_search" in allowed_tools:
        search_result = _run_shared_search_pipeline(
            user_id=user_id,
            question_text=user_question,
            conversation_history=None,
            trace=trace,
            preferred_strategy="generic_authoritative_lookup",
            preferred_domains=["abington.psu.edu", "psu.edu", "pennstate.edu"],
            required_domain_mode="preferred",
        )
        web_response = _research_answer_text(search_result)
        if web_response and _web_fallback_is_structurally_acceptable(web_response, plan):
            return web_response, tools_used + ["web_search"], {
                "source": "shared_search_pipeline",
                **_resolved_search_to_metadata(search_result),
            }
        # Web search returned text, but it is structurally unacceptable for
        # this deliverable (e.g. course-recommendation deliverable with no
        # Semester N: ... Total Credits: N block and no course codes).
        # Honest unknown beats confident garbage: fall through to the
        # soft-ask below rather than ship the registrar-page prose.
    return (
        _major_lookup_soft_ask(plan, bundle),
        tools_used,
        {"source": "major_lookup_not_found"},
    )


def _is_course_recommendation_deliverable(plan: Optional["OrchestrationPlan"]) -> bool:
    """True when the plan asked for an actual semester-by-semester course plan.

    Used to gate the web-search fallback acceptance check: vague registrar
    prose is fine for a "what is the Art major about" question but is NEVER
    an acceptable answer to "give me a 4-semester plan".
    """
    if plan is None:
        return False
    scope = (getattr(plan, "plan_scope", "") or "").lower()
    deliverable = (getattr(plan, "deliverable", "") or "").lower()
    if scope in {"single_semester", "multi_semester", "full_graduation"}:
        return True
    if deliverable in {
        "single_semester_plan",
        "multi_semester_plan",
        "full_graduation_plan",
    }:
        return True
    return False


def _web_fallback_is_structurally_acceptable(
    response_text: str,
    plan: Optional["OrchestrationPlan"],
) -> bool:
    """Structural acceptance check for the web_search fallback.

    For course-recommendation deliverables the response MUST contain at
    least one ``Semester N: ... Total Credits: N`` block AND at least one
    course code. If both are missing, the web fallback is shipping vague
    registrar prose instead of an actual plan and we reject it. (Trace 2.)

    For any other deliverable (lookup, direct_answer, audit) we accept the
    web response as-is since it is just being used as a fact-finding aid.
    """
    if not response_text:
        return False
    if not _is_course_recommendation_deliverable(plan):
        return True
    has_semester_block = bool(_extract_semester_blocks_from_text(response_text))
    from orchestration_pipeline import COURSE_CODE_RE  # local import avoids cycles
    has_course_code = bool(COURSE_CODE_RE.search(response_text))
    return has_semester_block and has_course_code


def _major_lookup_soft_ask(
    plan: Optional["OrchestrationPlan"],
    bundle: Optional[Dict[str, Dict[str, str]]],
) -> str:
    """Honest soft-ask returned when major_lookup cannot ship a real plan.

    Lists what was retrieved (if anything) so the student can see what we
    have and what we're asking them to confirm. Always ends with a question
    so the persona contract holds.
    """
    retrieved_tools = sorted(bundle.keys()) if bundle else []
    bundle_summary = ""
    if retrieved_tools:
        bundle_summary = (
            f" I did pull data from {', '.join(retrieved_tools)}, but it wasn't "
            "enough to lay out a credible semester-by-semester plan."
        )
    return (
        "I want to give you a solid answer here, but I couldn't pin down everything "
        "I'd need to map out an accurate course plan from what we have so far."
        f"{bundle_summary} "
        "Could you confirm the full program name(s) as they appear in the Penn State "
        "catalog (for example, 'Computer Science, B.S. (Abington)' or 'Accounting, B.S. "
        "(Abington)')? If this is a double major, please list both. Once I have that I "
        "can pull the requirements and map out the right courses for you."
    )


def _build_general_academic_chat_draft(
    user_question: str,
    conversation_history: Optional[List[dict]],
    student_data: Optional[dict],
    trace: Optional[Dict[str, Any]],
    allowed_tools: Optional[set],
    user_id: str = "default",
) -> Tuple[str, List[str], Dict[str, Any]]:
    if not is_academic_or_policy_question(user_question):
        return (
            "Sorry, this question is beyond the scope of this conversation. I can only help with academic advising, course planning, Penn State policies, and graduation requirements.",
            [],
            {"source": "scope_guard"},
        )

    if allowed_tools is not None and not allowed_tools:
        return "", [], {"source": "general_academic_chat_no_tools"}

    student_record = ""
    major = ""
    completed_courses = []
    if student_data:
        major = student_data.get("major", "") or ""
        completed_courses = student_data.get("completed_courses", [])
        student_record = json.dumps(student_data, indent=2)

    conversation_context = ""
    if conversation_history and len(conversation_history) > 1:
        recent_history = conversation_history[-6:]
        conversation_context = "\n\nPrevious conversation:\n"
        for msg in recent_history:
            role = "Student" if msg.get('role') == 'user' else "AI Advisor"
            conversation_context += f"{role}: {msg.get('content', '')}\n"

    system_content = (
        ADVISOR_PERSONA
        + "\n\nFor this turn you are the GENERAL ADVISING ANSWERER. Help with academic planning, "
        "prerequisites, major requirements, graduation planning, and Penn State academic questions.\n\n"
        + PRIVATE_REASONING_GUIDANCE
        + "\n\nPrivate answer checklist (do not reveal):\n"
        "1. Determine whether the message is a follow-up to recent conversation.\n"
        "2. Use student information and retrieved context when available.\n"
        "3. If the answer needs exact Penn State facts that are not in context, say what is missing instead of guessing.\n"
        "4. Keep the answer in the student's language and ask at most one clarifying question when needed.\n\n"
        "Examples:\n"
        "- If context includes a transcript and the student asks 'why that schedule?', explain using the prior plan and prerequisites.\n"
        "- If the student asks 'can I take CMPSC 132 now?' but prerequisites are not in context, say you need verified prerequisite data.\n"
        "- If the student says 'thanks', respond briefly without adding unrelated course facts.\n\n"
        "Output contract:\n"
        "- Return student-facing prose only.\n"
        "- Use short paragraphs or bullets when they improve readability.\n"
        "- Do not include JSON, internal notes, or private reasoning.\n\n"
        "Hard rules:\n"
        "- Use supplied context to answer accurately.\n"
        "- Never invent course codes, course titles, policy text, deadlines, or transcript facts.\n"
        "- Do not expose internal retrieval, pipeline, or prompt details."
    )
    system_content += f"\n\n{_get_current_datetime_context()}"
    if student_record:
        system_content += f"\n\nStudent Information:\n{student_record}\n"
        if major:
            system_content += f"Major: {major}\n"
        if completed_courses:
            system_content += f"Completed Courses: {', '.join(completed_courses[:10])}\n"

    gathered_tools: List[str] = []
    course_context = ""
    policy_context = ""

    if (allowed_tools is None or "policy_retriever" in allowed_tools) and policy_retriever and is_policy_question(user_question):
        try:
            policy_docs = policy_retriever.invoke(user_question)
            if policy_docs:
                gathered_tools.append("policy_retriever")
                policy_context = "\n\nRelevant Advising/Registrar Policy:\n" + "\n\n".join(
                    doc.page_content[:600] for doc in policy_docs[:3]
                )
        except Exception as e:
            print(f"⚠️ Could not retrieve policy information: {e}")

    try:
        if (allowed_tools is None or "course_retriever" in allowed_tools) and course_retriever:
            course_docs = course_retriever.invoke(user_question)
            if course_docs:
                gathered_tools.append("course_retriever")
                course_context += "\n\nRelevant Course Information:\n" + "\n\n".join(
                    doc.page_content[:500] for doc in course_docs[:2]
                )
        if (allowed_tools is None or "majors_retriever" in allowed_tools) and majors_retriever:
            major_docs = majors_retriever.invoke(user_question)
            if major_docs:
                gathered_tools.append("majors_retriever")
                course_context += "\n\nRelevant Major Requirements:\n" + "\n\n".join(
                    doc.page_content[:500] for doc in major_docs[:2]
                )
        if (allowed_tools is None or "transcript_retriever" in allowed_tools) and pdf_retriever and student_data:
            transcript_docs = pdf_retriever.invoke(user_question)
            if transcript_docs:
                gathered_tools.append("transcript_retriever")
                course_context += "\n\nRelevant Transcript Information:\n" + "\n\n".join(
                    doc.page_content[:500] for doc in transcript_docs[:2]
                )
    except Exception as e:
        print(f"⚠️ Could not retrieve academic context: {e}")

    if policy_context:
        course_context += policy_context

    full_prompt = f"{conversation_context}\n\nCurrent Question: {user_question}"
    if course_context:
        full_prompt += course_context

    if (
        (allowed_tools is None or "web_search" in allowed_tools)
        and (not course_context or len(course_context.strip()) < 300)
        and is_academic_or_policy_question(user_question)
    ):
        search_result = _run_shared_search_pipeline(
            user_id=user_id,
            question_text=user_question,
            conversation_history=conversation_history,
            trace=trace,
            preferred_strategy="generic_authoritative_lookup",
        )
        web_context = _research_answer_text(search_result)
        if web_context:
            gathered_tools.append("web_search")
            full_prompt += f"\n\nWeb research notes:\n{web_context}"

    response = llm2.invoke([
        SystemMessage(content=system_content),
        HumanMessage(content=full_prompt),
    ]).content

    if (allowed_tools is None or "web_search" in allowed_tools) and _response_indicates_no_answer_found(response):
        search_result = _run_shared_search_pipeline(
            user_id=user_id,
            question_text=user_question,
            conversation_history=conversation_history,
            trace=trace,
            preferred_strategy="generic_authoritative_lookup",
        )
        web_response = _research_answer_text(search_result)
        if web_response:
            gathered_tools.append("web_search")
            return web_response, list(dict.fromkeys(gathered_tools)), {
                "source": "shared_search_pipeline",
                **_resolved_search_to_metadata(search_result),
            }

    return response, list(dict.fromkeys(gathered_tools)), {"source": "general_academic_chat"}


def _build_research_lookup_draft(
    user_id: str,
    user_question: str,
    conversation_history: Optional[List[dict]],
    trace: Optional[Dict[str, Any]],
    allowed_tools: Optional[set],
) -> Tuple[str, List[str], Dict[str, Any]]:
    if allowed_tools is not None and "web_search" not in allowed_tools:
        return (
            "I don't have web research enabled for this response.",
            [],
            {"source": "research_lookup_disabled"},
        )
    search_result = _run_shared_search_pipeline(
        user_id=user_id,
        question_text=user_question,
        conversation_history=conversation_history,
        trace=trace,
        preferred_strategy="generic_authoritative_lookup",
    )
    answer_text = _research_answer_text(search_result)
    if answer_text:
        return answer_text, ["web_search"], {
            "source": "shared_search_pipeline",
            **_resolved_search_to_metadata(search_result),
        }
    return (
        "I couldn't verify the answer with enough confidence from the available sources.",
        ["web_search"],
        {
            "source": "shared_search_pipeline",
            **_resolved_search_to_metadata(search_result),
        },
    )


def _build_executor_draft(
    user_id: str,
    plan: OrchestrationPlan,
    user_question: str,
    conversation_history: Optional[List[dict]],
    has_transcript_for_plan: bool,
    student_data: Optional[dict],
    trace: Optional[Dict[str, Any]],
    retry_reason: Optional[str] = None,
) -> Tuple[str, List[str], Dict[str, Any]]:
    # Structural retry: if the finalizer flagged an ungrounded specificity
    # (course codes etc. with zero tools used), upgrade this plan to a
    # grounded path for this attempt before dispatching. This protects every
    # unexpected question, not just the ones we have keywords for.
    if retry_reason == "ungrounded_specificity":
        available = _build_available_tool_inventory(has_transcript_for_plan)
        canonical_path = _canonical_path_for_domain(plan.domain, plan.deliverable)
        canonical_tools = _canonical_tools_for_path(canonical_path, available)
        if not canonical_tools:
            # Domain doesn't map to a grounded path; fall back to web research.
            canonical_path = "research_lookup"
            canonical_tools = _canonical_tools_for_path(canonical_path, available)
        plan.use_tools = True
        plan.executor_path = canonical_path
        plan.allowed_tools = canonical_tools

    allowed_tools = set(plan.allowed_tools or [])

    if plan.executor_path == "conversation_plan_slice":
        prior_plan = _find_recent_assistant_plan_text(conversation_history)
        if prior_plan:
            sliced = _slice_semester_plan_text(prior_plan, max(1, plan.semester_horizon))
            if sliced:
                return sliced, [], {"source": "conversation_plan_slice"}
        return (
            "I couldn't find a recent semester-by-semester plan in the conversation to narrow down. Please ask for the plan again and then I can trim it to the next semester.",
            [],
            {"source": "conversation_plan_slice_missing_context"},
        )

    if plan.executor_path == "deterministic_plan":
        return _build_transcript_plan_draft(plan, user_question, conversation_history, has_transcript_for_plan)

    if plan.executor_path == "transcript_audit":
        return _build_transcript_audit_draft(student_data, user_question), ["transcript_state"] if student_data else [], {"source": "transcript_audit"}

    if plan.executor_path == "conversation_profile_recall":
        return (
            _build_conversation_profile_recall_draft(conversation_history, student_data),
            [],
            {"source": "conversation_profile_recall"},
        )

    if plan.executor_path == "course_lookup":
        response, tools_used, metadata = _build_course_lookup_draft_with_tools(
            plan.corrected_question or user_question,
            trace,
            allowed_tools,
            user_id=user_id,
        )
        return response, tools_used, metadata

    if plan.executor_path == "policy_lookup":
        response, tools_used, metadata = _build_policy_lookup_draft(
            plan.corrected_question or user_question,
            conversation_history,
            trace,
            allowed_tools,
            retry_reason=retry_reason,
            user_id=user_id,
        )
        return response, tools_used, metadata

    if plan.executor_path == "major_lookup":
        is_plan_request = (
            plan.plan_scope in {"single_semester", "multi_semester", "full_graduation"}
            or plan.deliverable in {"single_semester_plan", "multi_semester_plan", "full_graduation_plan"}
            or _looks_like_plan_request(plan.corrected_question or user_question)
        )
        is_multi_major_request = len(getattr(plan, "majors", None) or []) > 1
        if is_plan_request and not has_transcript_for_plan and not is_multi_major_request:
            major_from_text, year_from_text = _extract_major_year_from_request_context(
                plan.corrected_question or user_question,
                conversation_history,
            )
            if major_from_text:
                year_for_plan = year_from_text or "freshman"
                full_plan_text = generate_plan_from_major_and_year(major_from_text, year_for_plan)
                if plan.plan_scope in {"single_semester", "multi_semester"}:
                    sliced = _slice_semester_plan_text(full_plan_text, plan.semester_horizon)
                    if sliced:
                        return sliced, ["deterministic_planner"], {
                            "source": "standard_major_plan",
                            "major": major_from_text,
                            "year": year_for_plan,
                        }
                return full_plan_text, ["deterministic_planner"], {
                    "source": "standard_major_plan",
                    "major": major_from_text,
                    "year": year_for_plan,
                }

        # If the question pins a specific semester of the major
        # (e.g. "first semester CS", "3rd semester biology"), route to the
        # deterministic planner: with-transcript -> personalized next-semester
        # plan, without -> the courses that appear in the standard graduation
        # plan for that semester of the major/option.
        target_semester_request = _extract_target_semester_request(
            plan.corrected_question or user_question
        )
        if target_semester_request is not None and not is_multi_major_request:
            response, tools_used, metadata = _build_semester_specific_major_draft(
                user_question=plan.corrected_question or user_question,
                target=target_semester_request,
                conversation_history=conversation_history,
                has_transcript_for_plan=has_transcript_for_plan,
                trace=trace,
                allowed_tools=allowed_tools,
                user_id=user_id,
            )
            if response:
                return response, tools_used, metadata
            # Fall through to generic major_lookup if the semester-specific
            # path could not produce a concrete plan (e.g. missing major).

        response, tools_used, metadata = _build_major_lookup_draft(
            plan.corrected_question or user_question,
            trace,
            allowed_tools,
            user_id=user_id,
            plan=plan,
            retry_reason=retry_reason,
        )
        return response, tools_used, metadata

    if plan.executor_path == "research_lookup":
        return _build_research_lookup_draft(
            user_id=user_id,
            user_question=plan.corrected_question or user_question,
            conversation_history=conversation_history,
            trace=trace,
            allowed_tools=allowed_tools,
        )

    if not plan.use_tools:
        # Safety net: if the orchestrator somehow left us in a "no tools" state
        # for a domain that needs grounding, escalate to web_search rather than
        # returning an empty draft (which invites the finalizer to hallucinate).
        if _requires_grounded_answer(plan.domain, plan.deliverable):
            available = _build_available_tool_inventory(has_transcript_for_plan)
            canonical_tools = _canonical_tools_for_path(
                _canonical_path_for_domain(plan.domain, plan.deliverable), available
            ) or _canonical_tools_for_path("research_lookup", available)
            plan.use_tools = True
            plan.allowed_tools = canonical_tools
            allowed_tools = set(canonical_tools)
        else:
            return "", [], {"source": "general_academic_chat_no_tools"}

    response, tools_used, metadata = _build_general_academic_chat_draft(
        plan.corrected_question or user_question,
        conversation_history,
        student_data,
        trace,
        allowed_tools,
        user_id=user_id,
    )
    return response, tools_used, metadata


def _run_executor_stage(
    plan: OrchestrationPlan,
    draft_response: str,
    conversation_history: Optional[List[dict]],
    supporting_context: str,
    retry_reason: Optional[str] = None,
) -> str:
    recent_messages = []
    for message in (conversation_history or [])[-4:]:
        role = str(message.get("role", "user"))
        content = str(message.get("content", "") or "")
        recent_messages.append(f"{role}: {content[:250]}")

    system_prompt = (
        ADVISOR_PERSONA
        + "\n\nFor this turn you are the EXECUTOR STAGE. Produce the core answer using the execution plan, "
        "conversation history, supporting context, and draft answer.\n\n"
        + PRIVATE_REASONING_GUIDANCE
        + "\n\nPrivate execution checklist (do not reveal):\n"
        "1. Confirm the execution plan's domain, deliverable, scope, and must_not_do list.\n"
        "2. Compare the draft against supporting context and remove unsupported facts.\n"
        "3. Preserve the requested scope exactly.\n"
        "4. If semester blocks already exist in the draft, preserve their block count and structure.\n"
        "5. If no tools are required, answer only from conversation history and available context.\n\n"
        "Examples:\n"
        "- Single-semester scope: return exactly one Semester N block, not a full 8-semester plan.\n"
        "- Policy direct answer: answer the policy question and do not add unrelated course recommendations.\n"
        "- Profile recall: if the student earlier said 'junior Computer Science', answer that directly without retrieval.\n\n"
        "Output contract:\n"
        "- Return the draft answer text only, improved for accuracy and scope.\n"
        "- Do not return JSON, internal notes, or private reasoning.\n"
        "- Preserve semester block formatting when present.\n\n"
        "Hard rules:\n"
        "- Treat helper outputs and drafts as evidence, but do not repeat unsupported or contradictory claims.\n"
        "- If the draft already contains semester blocks, preserve those blocks VERBATIM and do not add extra semesters.\n"
        "- Do not reveal private reasoning or internal pipeline details."
    )
    user_prompt = (
        f"{_get_current_datetime_context()}\n"
        f"Original question: {plan.original_question}\n"
        f"Corrected question: {plan.corrected_question}\n"
        f"Resolved need: {plan.resolved_user_need}\n"
        f"Domain: {plan.domain}\n"
        f"Deliverable: {plan.deliverable}\n"
        f"Plan scope: {plan.plan_scope}\n"
        f"Semester horizon: {plan.semester_horizon}\n"
        f"Must not do: {plan.must_not_do}\n"
        f"Retry reason: {retry_reason or 'none'}\n"
        f"Recent conversation:\n" + ("\n".join(recent_messages) if recent_messages else "[none]") + "\n\n"
        f"Supporting context:\n{supporting_context or '[none]'}\n\n"
        f"Draft answer:\n{draft_response}"
    )
    raw_output = llm2.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]).content
    candidate = str(raw_output or "").strip()
    if not candidate:
        return draft_response or "I need a bit more detail to answer that. Please ask a more specific academic advising question."

    draft_blocks = _extract_semester_blocks_from_text(draft_response)
    candidate_blocks = _extract_semester_blocks_from_text(candidate)
    if draft_blocks:
        expected_blocks = 1 if plan.plan_scope == "single_semester" else plan.semester_horizon if plan.plan_scope == "multi_semester" else len(draft_blocks)
        if plan.plan_scope in {"single_semester", "multi_semester"} and len(candidate_blocks) != expected_blocks:
            return draft_response
    return candidate


def _execute_orchestration_plan(
    user_id: str,
    plan: OrchestrationPlan,
    user_question: str,
    conversation_history: Optional[List[dict]],
    has_transcript_for_plan: bool,
    student_data: Optional[dict],
    trace: Optional[Dict[str, Any]],
    retry_reason: Optional[str] = None,
) -> ExecutorResult:
    draft_response, tools_used, metadata = _build_executor_draft(
        user_id=user_id,
        plan=plan,
        user_question=user_question,
        conversation_history=conversation_history,
        has_transcript_for_plan=has_transcript_for_plan,
        student_data=student_data,
        trace=trace,
        retry_reason=retry_reason,
    )
    supporting_context = _build_reasoning_support_context(
        plan=plan,
        conversation_history=conversation_history,
        student_data=student_data,
        draft_response=draft_response,
        metadata=metadata,
    )
    final_response = _run_executor_stage(
        plan=plan,
        draft_response=draft_response,
        conversation_history=conversation_history,
        supporting_context=supporting_context,
        retry_reason=retry_reason,
    )
    return ExecutorResult(
        response=final_response,
        tools_used=list(dict.fromkeys(tools_used)),
        metadata={
            "draft_preview": draft_response[:500],
            "supporting_context_preview": supporting_context[:500],
            "retry_reason": retry_reason,
            **metadata,
        },
    )


_FOUL_LANGUAGE_TOPIC_RE = re.compile(
    r"\b(?:profan(?:ity|e)|swear(?:ing)?|curse\s+words?|cuss(?:ing)?|foul\s+language|"
    r"vulgar\s+language|obscene\s+wording|bad\s+words?|offensive\s+language)\b",
    re.IGNORECASE,
)
_FOUL_LANGUAGE_REQUEST_RE = re.compile(
    r"\b(?:say|write|use|give|list|generate|repeat|quote|translate|make|create|provide|"
    r"approve|endorse|condone|justify|normalize|example|examples)\b",
    re.IGNORECASE,
)


def _is_foul_language_request(user_question: str) -> bool:
    """Detect requests to produce or endorse foul language before calling the model."""
    question = user_question or ""
    if not _FOUL_LANGUAGE_TOPIC_RE.search(question):
        return False
    return bool(_FOUL_LANGUAGE_REQUEST_RE.search(question))


def _foul_language_refusal() -> str:
    return (
        f"I can't help produce, repeat, approve, or endorse foul language. "
        f"{AGENT_DISPLAY_NAME} will keep the wording professional and can help rephrase the idea respectfully."
    )


def generate_conversational_response(
    user_question,
    conversation_history,
    user_id='default',
    user_has_transcript=None,
    debug_mode: bool = False
):
    """Generate a response through the orchestrator -> executor -> finalizer pipeline."""
    print("\n🔹 Generating Conversational Response...")

    global llm1, llm2, llm_classifier, has_transcript, stored_pdf_docs, latest_student_data
    global execution_trace_store

    trace_enabled = _is_debug_enabled(debug_mode)
    trace = _new_execution_trace(user_id=user_id, user_question=user_question, enabled=trace_enabled)
    _trace_event(
        trace,
        "start",
        "Request received",
        {"has_history": bool(conversation_history), "history_len": len(conversation_history or [])},
    )

    if _is_foul_language_request(user_question):
        refusal = sanitize_public_response(_foul_language_refusal())
        trace["result_preview"] = refusal[:400]
        trace["finished_at"] = datetime.now().isoformat()
        execution_trace_store[user_id] = trace
        return refusal

    if user_has_transcript is not None:
        has_transcript_for_plan = user_has_transcript and (stored_pdf_docs is not None)
    else:
        has_transcript_for_plan = stored_pdf_docs is not None and has_transcript

    if llm1 is None or llm2 is None or llm_classifier is None:
        print("⚠️ LLMs not initialized. Initializing now...")
        try:
            setup_environment(None)
        except Exception as e:
            print(f"⚠️ Error initializing LLMs: {e}")
            if not os.getenv('OPENAI_API_KEY'):
                raise RuntimeError("GEMINI_API_KEY not set. Add it to .env in IAAS/LLMAcadamic-Advisor/")
            from langchain_openai import ChatOpenAI
            llm1 = ChatOpenAI(model_name=PRIMARY_MODEL_NAME, temperature=0)
            llm2 = ChatOpenAI(model_name=PRIMARY_MODEL_NAME, temperature=0)
            llm_classifier = ChatOpenAI(model_name=CLASSIFIER_MODEL_NAME, temperature=0)

    student_data = get_student_data_from_transcript()
    if not student_data and has_transcript_for_plan and stored_pdf_docs:
        try:
            student_data = extract_student_data_deterministic(stored_pdf_docs)
            latest_student_data = student_data
        except Exception as e:
            print(f"⚠️ Failed to extract student data deterministically: {e}")
            student_data = latest_student_data

    available_tools = _build_available_tool_inventory(has_transcript_for_plan)
    validation_helpers = {
        "needs_specific_date": _question_needs_specific_date,
        "has_concrete_date": _has_concrete_date_in_text,
        "is_time_sensitive_fact": _is_time_sensitive_factual_question,
        "extract_source_urls": _extract_source_urls_from_response,
        "indicates_no_answer": _response_indicates_no_answer_found,
        # Lets the finalizer re-verify any '(prerequisites: ...)' annotations
        # it ADDS (e.g. in 'Things to consider next' bullets) against the
        # verified prereq graph from the executor, just like the executor
        # already does on its own draft.
        "rewrite_inline_prereqs": _rewrite_inline_prereqs_from_graph,
    }

    try:
        final_response, plan = run_three_stage_pipeline(
            user_question=user_question,
            conversation_history=conversation_history,
            has_transcript=has_transcript_for_plan,
            available_tools=available_tools,
            orchestrator_llm=llm_classifier,
            finalizer_llm=llm1,
            execute_plan=lambda orchestration_plan, retry_reason=None: _execute_orchestration_plan(
                user_id=user_id,
                plan=orchestration_plan,
                user_question=user_question,
                conversation_history=conversation_history,
                has_transcript_for_plan=has_transcript_for_plan,
                student_data=student_data,
                trace=trace,
                retry_reason=retry_reason,
            ),
            trace=trace,
            trace_event=_trace_event,
            trace_quality_check=_trace_quality_check,
            validation_helpers=validation_helpers,
        )
        trace["plan"] = plan.to_dict()
        final_response = sanitize_public_response(final_response)
        trace["result_preview"] = final_response[:400]
        trace["finished_at"] = datetime.now().isoformat()
        execution_trace_store[user_id] = trace
        return final_response
    except Exception as e:
        import traceback
        print(f"Error in conversational response pipeline: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        fallback = "I apologize, but I encountered an error processing your question. Please try rephrasing it or ask a more specific academic advising question."
        fallback = sanitize_public_response(fallback)
        trace["result_preview"] = fallback[:400]
        trace["finished_at"] = datetime.now().isoformat()
        execution_trace_store[user_id] = trace
        return fallback


# =========================
# NEW: Major + Year (no transcript)
# =========================

def generate_plan_from_major_and_year(major, year_level, target_grad_in_years=4):
    """
    Generate a graduation plan when the student does NOT upload a transcript.
    Uses major, year_level, course graph, and CSV-based major requirements.
    """
    global llm1, llm2, course_graph
    
    # Ensure LLMs are initialized
    if llm1 is None or llm2 is None:
        print("⚠️ LLMs not initialized. Initializing now...")
        try:
            setup_environment(None)  # Initialize without transcript
        except Exception as e:
            print(f"⚠️ Error in setup_environment: {e}")
            if not os.getenv('OPENAI_API_KEY'):
                raise RuntimeError("GEMINI_API_KEY not set. Add it to .env in IAAS/LLMAcadamic-Advisor/")
            from langchain_openai import ChatOpenAI
            llm1 = ChatOpenAI(model_name=PRIMARY_MODEL_NAME, temperature=0)
            llm2 = ChatOpenAI(model_name=PRIMARY_MODEL_NAME, temperature=0)
    
    # Ensure course_graph is available
    if 'course_graph' not in globals() or course_graph is None:
        print("⚠️ Course graph not available. Importing...")
        try:
            from graphExperiment import course_graph as cg
            course_graph = cg
        except Exception as e:
            print(f"⚠️ Could not import course_graph: {e}")
            course_graph = None
    
    # Normalize year input (handles typos and variants)
    year_level = _normalize_year_token(year_level) or year_level
    if year_level not in ['freshman', 'sophomore', 'junior', 'senior']:
        year_level = _fuzzy_year_from_text(year_level or "") or "freshman"

    # Calculate semesters completed and remaining
    typical_total_credits = 120
    year_lower = year_level.lower()

    if year_lower == "freshman":
        assumed_completed = 0
        semesters_completed = 0
    elif year_lower == "sophomore":
        assumed_completed = 30
        semesters_completed = 2
    elif year_lower == "junior":
        assumed_completed = 60
        semesters_completed = 4
    elif year_lower == "senior":
        assumed_completed = 90
        semesters_completed = 6
    else:
        assumed_completed = 0
        semesters_completed = 0

    total_credits_needed = max(0, typical_total_credits - assumed_completed)
    semesters_remaining = 8 - semesters_completed
    credits_per_semester = total_credits_needed / semesters_remaining if semesters_remaining > 0 else 15
    # Keep plans spread across expected timeline (freshman->8, sophomore->6, etc.)
    # Prefer ~15-credit semesters when possible, allow up to 18 only when needed.
    planning_min_credits = 15
    planning_max_credits = 15 if total_credits_needed <= semesters_remaining * 15 else 18
    
    print(f"📊 Student status: {year_level}, {assumed_completed} credits completed, {total_credits_needed} credits needed")
    print(f"📊 Semesters remaining: {semesters_remaining} ({credits_per_semester:.1f} credits per semester)")

    # Normalize major and load CSVs
    major_norm = normalize_major_name(major)
    if not major_norm:
        return (
            f"I want to build out the right plan for you, but I couldn't match '{major}' to a Penn State Abington program. "
            "Could you share the exact program name as it appears in the catalog "
            "(for example, 'Business, B.A. (Abington)' or 'Computer Science, B.S. (Abington)')? "
            "Once I have the official name I can pull the requirements and lay out a 15-credit semester."
        )
    courses_csv = os.path.join(os.path.dirname(__file__), "processed_psu_courses.csv")
    majors_csv = os.path.join(os.path.dirname(__file__), "all_abington_majors_combined(5).csv")
    if not os.path.exists(courses_csv):
        alt_courses = os.path.join(os.path.dirname(__file__), "..", "LLM", "processed_psu_courses.csv")
        if os.path.exists(alt_courses):
            courses_csv = alt_courses
    if not os.path.exists(majors_csv):
        alt_majors = os.path.join(os.path.dirname(__file__), "..", "LLM", "all_abington_majors_combined(5).csv")
        if os.path.exists(alt_majors):
            majors_csv = alt_majors

    # Deterministic schedule using graduationFlowchart
    from graduationFlowchart import generate_semester_plan
    try:
        # First build a full plan to infer completed courses for upperclassmen
        full_plan = generate_semester_plan(
            major_name=major_norm,
            completed_courses=[],
            unmet_requirements=[],
            courses_csv_path=courses_csv,
            majors_csv_path=majors_csv,
            credits_needed=typical_total_credits,
            max_semesters=8,
            max_credits_per_sem=planning_max_credits,
            min_credits_per_sem=planning_min_credits
        )
        completed_courses = []
        if semesters_completed > 0:
            for sem in full_plan.get("semesters", []):
                if sem.get("semester", 0) <= semesters_completed:
                    completed_courses.extend(sem.get("courses", []))

        plan_structure = generate_semester_plan(
            major_name=major_norm,
            completed_courses=completed_courses,
            unmet_requirements=[],
            courses_csv_path=courses_csv,
            majors_csv_path=majors_csv,
            credits_needed=total_credits_needed,
            max_semesters=8,
            max_credits_per_sem=planning_max_credits,
            min_credits_per_sem=planning_min_credits
        )
        print(f"✅ Generated deterministic plan with {len(plan_structure.get('semesters', []))} semesters")
    except Exception as e:
        print(f"⚠️ Error generating deterministic plan (no transcript): {e}")
        # Try fuzzy match on major name as a fallback
        try:
            dfm = pd.read_csv(majors_csv)
            try:
                from fuzzywuzzy import process
                best_match = process.extractOne(major_norm, dfm['Major Name'].tolist())
                if best_match and best_match[1] >= 60:
                    print(f"🔄 Retrying with fuzzy matched major: {best_match[0]}")
                    plan_structure = generate_semester_plan(
                        major_name=best_match[0],
                        completed_courses=[],
                        unmet_requirements=[],
                        courses_csv_path=courses_csv,
                        majors_csv_path=majors_csv,
                        credits_needed=total_credits_needed,
                        max_semesters=8,
                        max_credits_per_sem=planning_max_credits,
                        min_credits_per_sem=planning_min_credits
                    )
                else:
                    return "I couldn't generate a plan without your transcript right now. Please upload your transcript or try again."
            except Exception:
                # Fallback to stdlib fuzzy matching if fuzzywuzzy is unavailable
                import difflib
                candidates = difflib.get_close_matches(
                    major_norm,
                    dfm['Major Name'].tolist(),
                    n=1,
                    cutoff=0.6
                )
                if candidates:
                    print(f"🔄 Retrying with difflib matched major: {candidates[0]}")
                    plan_structure = generate_semester_plan(
                        major_name=candidates[0],
                        completed_courses=[],
                        unmet_requirements=[],
                        courses_csv_path=courses_csv,
                        majors_csv_path=majors_csv,
                        credits_needed=total_credits_needed,
                        max_semesters=8,
                        max_credits_per_sem=planning_max_credits,
                        min_credits_per_sem=planning_min_credits
                    )
                else:
                    return "I couldn't generate a plan without your transcript right now. Please upload your transcript or try again."
        except Exception as e2:
            print(f"⚠️ Fuzzy fallback failed: {e2}")
            return "I couldn't generate a plan without your transcript right now. Please upload your transcript or try again."

    # Deterministic formatting (avoid LLM hallucinations)
    start_semester_number = semesters_completed + 1 if semesters_completed > 0 else 1
    return _format_plan_structure(plan_structure, start_semester_number=start_semester_number)


# =========================
# Flask App / Endpoint
# =========================

app = Flask(__name__)

@app.route("/generate_plan", methods=["POST"])
def generate_plan_route():
    """
    Unified endpoint:
    - If 'transcript' file is present → transcript-based plan.
    - Else → expects JSON or form with 'major' and 'year' to generate plan.
    """
    # If there's a transcript file, use transcript-based mode
    if 'transcript' in request.files:
        setup_environment(request)
        plan = aggregate_and_generate_response(
            user_question="Generate my graduation plan from my transcript."
        )
        return jsonify({"plan": plan})

    # Otherwise, expect major + year
    data = request.get_json(silent=True) or request.form
    major = data.get("major")
    year_level = data.get("year")

    if not major or not year_level:
        return jsonify({
            "error": "Please provide 'major' and 'year' (e.g., 'Computer Science, BS (CMPAB_BS)' and 'junior')."
        }), 400

    setup_environment(None)
    plan = generate_plan_from_major_and_year(
        major=major,
        year_level=year_level
    )
    return jsonify({"plan": plan})


if __name__ == "__main__":
    # For local testing
    # Make sure OPENAI_API_KEY and LANGCHAIN_API_KEY are set in your environment first.
    app.run(host="0.0.0.0", port=5000, debug=True)
