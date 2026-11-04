from __future__ import annotations

import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from html import unescape
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib import request as urllib_request
from urllib.parse import urljoin, urlparse

try:
    import requests
except Exception:  # pragma: no cover - stdlib fallback for reduced test environments
    requests = None

try:
    from langchain_core.messages import HumanMessage, SystemMessage
except Exception:  # pragma: no cover - fallback for reduced test environments
    @dataclass
    class SystemMessage:
        content: str

    @dataclass
    class HumanMessage:
        content: str


CURRENT_DATETIME_FORMAT = "Current local date/time: %A, %B %d, %Y at %I:%M %p."

OFFICIAL_SOURCE_SEEDS: Dict[str, List[str]] = {
    "nit_hamirpur_policy": [
        "https://nith.ac.in/",
        "https://nith.ac.in/bachelor-course-structure-syllabus",
        "https://nith.ac.in/academic-calendar",
        "https://nith.ac.in/ordinance-regulations",
        "https://nith.ac.in/computer-science-engineering",
    ],
    "nit_hamirpur_identity": [
        "https://nith.ac.in/",
        "https://nith.ac.in/administration",
        "https://nith.ac.in/deans",
        "https://nith.ac.in/directors-desk",
    ],
    "penn_state_policy": [
        "https://www.registrar.psu.edu/academic-calendars/",
        "https://www.registrar.psu.edu/registration/dropadd.cfm",
        "https://www.registrar.psu.edu/registration/",
        "https://www.psu.edu/",
        "https://www.abington.psu.edu/",
        "https://www.abington.psu.edu/advising",
        "https://www.abington.psu.edu/academics/academic-advising",
    ],
    "penn_state_identity": [
        "https://www.psu.edu/about/leadership/",
        "https://www.psu.edu/about/administration/",
        "https://www.abington.psu.edu/about",
        "https://www.abington.psu.edu/about/leadership",
        "https://www.abington.psu.edu/about/administration",
    ],
}

SOURCE_POLICY_MATRIX: Dict[str, Dict[str, Any]] = {
    "nith_policy": {
        "required_domain_mode": "strict",
        "preferred_domains": ["nith.ac.in"],
        "min_citations": 1,
    },
    "psu_policy": {
        "required_domain_mode": "strict",
        "preferred_domains": ["abington.psu.edu", "psu.edu", "pennstate.edu"],
        "min_citations": 1,
    },
    "organization_role": {
        "required_domain_mode": "preferred",
        "preferred_domains": [],
        "min_citations": 1,
    },
    "generic_fact": {
        "required_domain_mode": "preferred",
        "preferred_domains": [],
        "min_citations": 1,
    },
    "generic_fallback": {
        "required_domain_mode": "none",
        "preferred_domains": [],
        "min_citations": 2,
    },
}

PERSON_NAME_RE = re.compile(
    r"\b(?:Dr\.\s+)?[A-Z][a-zA-Z'\-]+(?:\s+[A-Z]\.)?\s+[A-Z][a-zA-Z'\-]+\b"
)
DATE_RE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\s+\d{1,2}(?:,\s*\d{4})?\b",
    flags=re.IGNORECASE,
)
YEAR_RE = re.compile(r"\b(20\d{2})\b")
TERM_RE = re.compile(r"\b(spring|summer|fall|winter)\b", flags=re.IGNORECASE)

CURRENT_TASK_CONTEXT_STORE: Dict[str, Dict[str, Any]] = {}

SEARCH_RESULTS_TTL_SECONDS = int(os.getenv("SEARCH_RESULTS_TTL_SECONDS", "300"))
FETCH_TEXT_TTL_SECONDS = int(os.getenv("FETCH_TEXT_TTL_SECONDS", "900"))
MAX_CONCURRENT_FETCHES = max(1, int(os.getenv("SEARCH_MAX_CONCURRENT_FETCHES", "6")))
MAX_CONCURRENT_SEARCHES = max(1, int(os.getenv("SEARCH_MAX_CONCURRENT_SEARCHES", "4")))
INITIAL_QUERY_BATCH_SIZE = max(1, int(os.getenv("SEARCH_INITIAL_QUERY_BATCH_SIZE", "2")))
EXPANDED_QUERY_BATCH_SIZE = max(INITIAL_QUERY_BATCH_SIZE, int(os.getenv("SEARCH_EXPANDED_QUERY_BATCH_SIZE", "5")))
INITIAL_SEARCH_FETCH_LIMIT = max(1, int(os.getenv("SEARCH_INITIAL_FETCH_LIMIT", "6")))
EXPANDED_SEARCH_FETCH_LIMIT = max(INITIAL_SEARCH_FETCH_LIMIT, int(os.getenv("SEARCH_EXPANDED_FETCH_LIMIT", "10")))
MAX_BROWSER_FALLBACKS = max(1, int(os.getenv("SEARCH_MAX_BROWSER_FALLBACKS", "2")))
MIN_BROWSER_FALLBACK_TEXT_LENGTH = max(80, int(os.getenv("SEARCH_BROWSER_TEXT_MIN", "160")))

SEARCH_RESULTS_CACHE: Dict[Tuple[str, str, int], Tuple[float, List[dict]]] = {}
FETCH_TEXT_CACHE: Dict[Tuple[str, int], Tuple[float, Tuple[Optional[str], Optional[str]]]] = {}
BROWSER_FETCH_CACHE: Dict[Tuple[str, int], Tuple[float, Tuple[Optional[str], Optional[str]]]] = {}
CACHE_LOCK = threading.Lock()


@dataclass
class SearchContext:
    original_question: str
    normalized_question: str
    corrected_question: str
    conversation_summary: str
    current_datetime: str
    campus: Optional[str] = None
    term: Optional[str] = None
    year: Optional[int] = None
    domain_hints: List[str] = field(default_factory=list)
    requires_freshness: bool = False
    requires_exact_date: bool = False
    requires_authoritative_sources: bool = False
    active_task_kind: str = "generic_authoritative_lookup"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SearchPlan:
    strategy: str
    query_variants: List[str]
    preferred_domains: List[str]
    required_domain_mode: str
    providers: List[str]
    fetch_mode: str
    extraction_mode: str
    max_rounds: int
    allow_browser_fallback: bool
    success_criteria: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SearchEvidence:
    url: str
    title: str = ""
    snippet: str = ""
    fetched_text: str = ""
    domain: str = ""
    publish_date: Optional[str] = None
    authority_score: float = 0.0
    relevance_score: float = 0.0
    freshness_score: float = 0.0
    extracted_claims: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResolvedResearchAnswer:
    answer_text: str
    citations: List[str] = field(default_factory=list)
    confidence: float = 0.0
    source_urls: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    attempted_strategies: List[str] = field(default_factory=list)
    exhausted: bool = False
    strategy: str = ""
    completed_ladder: bool = False
    exhaustion_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def clear_active_search_context(user_id: str):
    CURRENT_TASK_CONTEXT_STORE.pop(user_id, None)


def reset_search_pipeline_caches():
    with CACHE_LOCK:
        SEARCH_RESULTS_CACHE.clear()
        FETCH_TEXT_CACHE.clear()
        BROWSER_FETCH_CACHE.clear()


def _cache_get(cache: Dict[Any, Tuple[float, Any]], key: Any, ttl_seconds: int) -> Optional[Any]:
    with CACHE_LOCK:
        cached = cache.get(key)
        if not cached:
            return None
        stored_at, value = cached
        if (time.time() - stored_at) > ttl_seconds:
            cache.pop(key, None)
            return None
        return value


def _cache_set(cache: Dict[Any, Tuple[float, Any]], key: Any, value: Any):
    with CACHE_LOCK:
        cache[key] = (time.time(), value)


def _parallel_map(items: Sequence[Any], worker: Any, max_workers: int) -> List[Any]:
    if not items:
        return []
    bounded_workers = max(1, min(max_workers, len(items)))
    if bounded_workers == 1:
        results: List[Any] = []
        for item in items:
            try:
                result = worker(item)
            except Exception:
                result = None
            if result is not None:
                results.append(result)
        return results

    indexed_results: Dict[int, Any] = {}
    with ThreadPoolExecutor(max_workers=bounded_workers) as executor:
        future_to_index = {
            executor.submit(worker, item): index
            for index, item in enumerate(items)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
            except Exception:
                result = None
            if result is not None:
                indexed_results[index] = result
    return [indexed_results[index] for index in sorted(indexed_results)]


def current_datetime_context(now: Optional[datetime] = None) -> str:
    current = now or datetime.now()
    return current.strftime(CURRENT_DATETIME_FORMAT)


def normalize_question_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def summarize_conversation(conversation_history: Optional[List[dict]], limit: int = 6) -> str:
    if not conversation_history:
        return ""
    lines: List[str] = []
    for message in conversation_history[-limit:]:
        role = str(message.get("role", "user"))
        content = normalize_question_text(str(message.get("content", "") or ""))
        if content:
            lines.append(f"{role}: {content[:300]}")
    return "\n".join(lines)


def _domain_from_url(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""


def _question_tokens(question: str) -> List[str]:
    stop_words = {
        "a", "an", "and", "are", "at", "be", "by", "for", "from", "how",
        "i", "in", "is", "it", "latest", "me", "my", "of", "on", "or",
        "please", "tell", "the", "this", "to", "today", "what", "when",
        "where", "which", "who", "why", "you",
    }
    lowered = re.sub(r"[^a-z0-9\s]", " ", (question or "").lower())
    return [token for token in lowered.split() if token and token not in stop_words]


def _is_followup_like(question: str) -> bool:
    lowered = normalize_question_text(question).lower()
    if not lowered:
        return False
    if len(lowered.split()) <= 4:
        return True
    return any(
        token in lowered
        for token in [
            "current semester",
            "this semester",
            "what about",
            "and for",
            "how about",
            "find exact",
            "exact dates",
            "spring 20",
            "fall 20",
            "summer 20",
            "winter 20",
        ]
    )


def _detect_campus(text: str) -> Optional[str]:
    lowered = (text or "").lower()
    if "abington" in lowered:
        return "Penn State Abington"
    if "penn state" in lowered or "psu" in lowered:
        return "Penn State"
    if "nit hamirpur" in lowered or "nith" in lowered or "hamirpur" in lowered:
        return "NIT Hamirpur"
    return None


def _infer_term_and_year(question: str, active_context: Optional[Dict[str, Any]] = None, now: Optional[datetime] = None) -> Tuple[Optional[str], Optional[int]]:
    current = now or datetime.now()
    lowered = (question or "").lower()
    year_match = YEAR_RE.search(lowered)
    year = int(year_match.group(1)) if year_match else None
    term_match = TERM_RE.search(lowered)
    term = term_match.group(1).lower() if term_match else None

    if "current semester" in lowered or "this semester" in lowered or "current term" in lowered:
        if current.month <= 5:
            term = "spring"
        elif current.month <= 8:
            term = "summer"
        else:
            term = "fall"
        year = current.year

    if not term and not year and any(token in lowered for token in ["deadline", "late drop", "early drop", "drop-add", "drop add", "today", "current"]):
        if current.month <= 5:
            term = "spring"
        elif current.month <= 8:
            term = "summer"
        else:
            term = "fall"
        year = current.year

    if not term and active_context:
        term = active_context.get("term")
    if not year and active_context:
        year = active_context.get("year")
    return term, year


def _extract_domain_hints(question: str, campus: Optional[str] = None) -> List[str]:
    hints: List[str] = []
    lowered = (question or "").lower()
    campus_lower = (campus or "").lower()
    if any(token in lowered for token in ["nit hamirpur", "nith", "hamirpur"]) or "hamirpur" in campus_lower:
        hints.append("nith.ac.in")
    if "abington" in lowered or "abington" in campus_lower:
        hints.append("abington.psu.edu")
    if any(token in lowered for token in ["penn state", "psu"]) or "penn state" in campus_lower:
        hints.extend(["psu.edu", "pennstate.edu"])
    return list(dict.fromkeys(hints))


def _requires_exact_date(question: str) -> bool:
    lowered = (question or "").lower()
    deadline_terms = ["deadline", "late drop", "early drop", "drop-add", "drop add", "calendar date", "exact date"]
    return any(term in lowered for term in deadline_terms) or ("when is" in lowered and "date" in lowered)


def _requires_freshness(question: str) -> bool:
    lowered = (question or "").lower()
    return any(
        token in lowered
        for token in [
            "current", "currently", "latest", "today", "this semester", "this year",
            "upcoming", "now", "deadline", "ceo", "president", "chancellor",
        ]
    )


def _requires_authoritative_sources(question: str) -> bool:
    lowered = (question or "").lower()
    return any(
        token in lowered
        for token in [
            "deadline", "policy", "official", "ceo", "president", "chancellor",
            "registrar", "academic calendar", "what is the current", "who is the current",
            "when was", "how many", "how much", "rate",
        ]
    )


def _classify_task_kind(question: str, active_context: Optional[Dict[str, Any]]) -> str:
    lowered = (question or "").lower()
    if any(term in lowered for term in ["late drop", "early drop", "drop-add", "drop add", "deadline"]):
        return "deadline_exact_date"
    if any(term in lowered for term in ["chancellor", "ceo", "president", "provost", "director", "chair", "dean"]):
        return "identity_current_role"
    if any(term in lowered for term in ["office page", "website", "where is", "registrar page", "calendar page"]):
        return "official_domain_lookup"
    if any(term in lowered for term in ["when ", "who ", "what ", "where ", "how many", "how much", "rate", "released"]):
        return "generic_authoritative_lookup"
    if active_context and _is_followup_like(question):
        return str(active_context.get("active_task_kind") or "generic_authoritative_lookup")
    return "broad_research_fallback"


def resolve_search_context(
    user_id: str,
    original_question: str,
    corrected_question: Optional[str],
    conversation_history: Optional[List[dict]],
    now: Optional[datetime] = None,
) -> SearchContext:
    current = now or datetime.now()
    normalized_question = normalize_question_text(original_question)
    corrected = normalize_question_text(corrected_question or normalized_question)
    active_context = CURRENT_TASK_CONTEXT_STORE.get(user_id) or {}
    conversation_summary = summarize_conversation(conversation_history)
    campus = _detect_campus(corrected) or _detect_campus(conversation_summary) or active_context.get("campus")
    term, year = _infer_term_and_year(corrected, active_context, current)
    task_kind = _classify_task_kind(corrected, active_context)
    context = SearchContext(
        original_question=original_question,
        normalized_question=normalized_question,
        corrected_question=corrected,
        conversation_summary=conversation_summary,
        current_datetime=current_datetime_context(current),
        campus=campus,
        term=term,
        year=year,
        domain_hints=_extract_domain_hints(corrected, campus),
        requires_freshness=_requires_freshness(corrected),
        requires_exact_date=_requires_exact_date(corrected),
        requires_authoritative_sources=_requires_authoritative_sources(corrected),
        active_task_kind=task_kind,
    )
    return context


def _keywordized_query(question: str) -> str:
    tokens = _question_tokens(question)
    return " ".join(tokens[:10]).strip() or normalize_question_text(question)


def _preferred_policy_key(context: SearchContext) -> str:
    if any(domain.endswith("nith.ac.in") for domain in context.domain_hints):
        return "nith_policy"
    if any(domain.endswith(("psu.edu", "pennstate.edu")) for domain in context.domain_hints):
        return "psu_policy"
    if context.active_task_kind == "identity_current_role":
        return "organization_role"
    if context.active_task_kind == "broad_research_fallback":
        return "generic_fallback"
    return "generic_fact"


def _build_deadline_queries(context: SearchContext) -> List[str]:
    campus = context.campus or "NIT Hamirpur"
    term = (context.term or "current").title()
    year = str(context.year or datetime.now().year)
    question = context.corrected_question
    queries = [
        f"site:nith.ac.in {campus} academic calendar",
        f"site:nith.ac.in bachelor course structure syllabus",
        f"site:nith.ac.in {campus} registration deadline",
        f"{campus} {term} {year} academic calendar",
        f"{campus} {term} {year} late drop deadline",
        f"site:psu.edu {campus} {term} {year} academic calendar late drop",
        question,
    ]
    return list(dict.fromkeys(" ".join(q.split()).strip() for q in queries if q.strip()))


def _build_identity_queries(context: SearchContext) -> List[str]:
    campus = context.campus or ""
    question = context.corrected_question
    queries = [
        question,
        f"{campus} {question}".strip(),
        f"{question} official",
    ]
    if context.domain_hints:
        primary_domain = context.domain_hints[0]
        queries.append(f"site:{primary_domain} {question}")
    return list(dict.fromkeys(" ".join(q.split()).strip() for q in queries if q.strip()))


def _build_generic_queries(context: SearchContext) -> List[str]:
    question = context.corrected_question
    keywordized = _keywordized_query(question)
    queries = [question, keywordized, f"{keywordized} official source", f"{keywordized} latest"]
    for domain in context.domain_hints[:2]:
        queries.append(f"site:{domain} {question}")
    return list(dict.fromkeys(" ".join(q.split()).strip() for q in queries if q.strip()))


def plan_search(
    context: SearchContext,
    preferred_strategy: Optional[str] = None,
    preferred_domains: Optional[Sequence[str]] = None,
    required_domain_mode: Optional[str] = None,
) -> SearchPlan:
    strategy = preferred_strategy or context.active_task_kind
    policy_key = _preferred_policy_key(context)
    defaults = SOURCE_POLICY_MATRIX[policy_key]
    preferred = list(dict.fromkeys(list(preferred_domains or []) + list(defaults["preferred_domains"]) + list(context.domain_hints)))
    domain_mode = required_domain_mode or defaults["required_domain_mode"]

    if strategy == "deadline_exact_date":
        queries = _build_deadline_queries(context)
        extraction_mode = "deadline_rows"
    elif strategy == "identity_current_role":
        queries = _build_identity_queries(context)
        extraction_mode = "named_role"
    elif strategy == "official_domain_lookup":
        queries = _build_generic_queries(context)
        extraction_mode = "official_page"
    elif strategy == "structured_fact_lookup":
        queries = _build_generic_queries(context)
        extraction_mode = "structured_fact"
    elif strategy == "generic_authoritative_lookup":
        queries = _build_generic_queries(context)
        extraction_mode = "generic_claim"
    else:
        strategy = "broad_research_fallback"
        queries = _build_generic_queries(context)
        extraction_mode = "generic_claim"

    success_criteria = ["Return an evidence-backed answer with at least one citation."]
    if context.requires_exact_date:
        success_criteria.append("Return an absolute Month DD, YYYY date when the date can be verified.")
    if domain_mode == "strict":
        success_criteria.append("Use the required official domain for the final answer.")

    return SearchPlan(
        strategy=strategy,
        query_variants=queries,
        preferred_domains=preferred,
        required_domain_mode=domain_mode,
        providers=["official_seed_fetch", "official_domain_crawl", "duckduckgo", "tavily", "browser_fallback"],
        fetch_mode="fetch_and_extract",
        extraction_mode=extraction_mode,
        max_rounds=5,
        allow_browser_fallback=True,
        success_criteria=success_criteria,
    )


def _trace_search_run(trace: Optional[Dict[str, Any]], context: SearchContext, plan: SearchPlan) -> Optional[Dict[str, Any]]:
    if not trace:
        return None
    pipeline = trace.setdefault(
        "search_pipeline",
        {
            "context": {},
            "plan": {},
            "rounds": [],
            "sources_considered": [],
            "sources_used": [],
            "exhaustion_reason": None,
            "runs": [],
        },
    )
    pipeline.setdefault("context", {})
    pipeline.setdefault("plan", {})
    pipeline.setdefault("rounds", [])
    pipeline.setdefault("sources_considered", [])
    pipeline.setdefault("sources_used", [])
    pipeline.setdefault("exhaustion_reason", None)
    pipeline.setdefault("runs", [])
    run = {
        "context": context.to_dict(),
        "plan": plan.to_dict(),
        "rounds": [],
        "sources_considered": [],
        "sources_used": [],
        "exhaustion_reason": None,
    }
    pipeline["runs"].append(run)
    pipeline["context"] = run["context"]
    pipeline["plan"] = run["plan"]
    pipeline["rounds"] = run["rounds"]
    pipeline["sources_considered"] = run["sources_considered"]
    pipeline["sources_used"] = run["sources_used"]
    pipeline["exhaustion_reason"] = run["exhaustion_reason"]
    return run


def _trace_search_round(
    trace: Optional[Dict[str, Any]],
    run: Optional[Dict[str, Any]],
    provider: str,
    details: Dict[str, Any],
):
    if not trace or run is None:
        return
    round_info = {"provider": provider, "details": details}
    run["rounds"].append(round_info)
    trace["search_pipeline"]["rounds"] = run["rounds"]


def _trace_search_sources(
    trace: Optional[Dict[str, Any]],
    run: Optional[Dict[str, Any]],
    evidence: Iterable[SearchEvidence],
    used: bool = False,
):
    if not trace or run is None:
        return
    dest = run["sources_used"] if used else run["sources_considered"]
    for item in evidence:
        serialized = {
            "url": item.url,
            "title": item.title,
            "domain": item.domain,
            "authority_score": item.authority_score,
            "relevance_score": item.relevance_score,
            "freshness_score": item.freshness_score,
        }
        if serialized not in dest:
            dest.append(serialized)
    trace["search_pipeline"]["sources_considered"] = run["sources_considered"]
    trace["search_pipeline"]["sources_used"] = run["sources_used"]


def _set_trace_exhaustion(trace: Optional[Dict[str, Any]], run: Optional[Dict[str, Any]], reason: Optional[str]):
    if not trace or run is None:
        return
    run["exhaustion_reason"] = reason
    trace["search_pipeline"]["exhaustion_reason"] = reason


def _fetch_url_text(url: str, timeout: int = 10) -> Tuple[Optional[str], Optional[str]]:
    if requests is None:
        try:
            req = urllib_request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib_request.urlopen(req, timeout=timeout) as response:
                status = getattr(response, "status", 200)
                if status != 200:
                    return None, f"status_{status}"
                charset = response.headers.get_content_charset() or "utf-8"
                return response.read().decode(charset, errors="ignore"), None
        except Exception as exc:
            return None, str(exc)
    try:
        response = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code != 200:
            return None, f"status_{response.status_code}"
        return response.text, None
    except Exception as exc:
        return None, str(exc)


def _fetch_url_text_cached(url: str, timeout: int = 10) -> Tuple[Optional[str], Optional[str]]:
    cache_key = (url, timeout)
    cached = _cache_get(FETCH_TEXT_CACHE, cache_key, FETCH_TEXT_TTL_SECONDS)
    if cached is not None:
        return cached
    result = _fetch_url_text(url, timeout=timeout)
    _cache_set(FETCH_TEXT_CACHE, cache_key, result)
    return result


def _browser_fetch_url_text(url: str, timeout_ms: int = 15000) -> Tuple[Optional[str], Optional[str]]:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        return None, f"playwright_unavailable: {exc}"
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=timeout_ms)
            content = page.content()
            browser.close()
            return content, None
    except Exception as exc:
        return None, str(exc)


def _browser_fetch_url_text_cached(url: str, timeout_ms: int = 15000) -> Tuple[Optional[str], Optional[str]]:
    cache_key = (url, timeout_ms)
    cached = _cache_get(BROWSER_FETCH_CACHE, cache_key, FETCH_TEXT_TTL_SECONDS)
    if cached is not None:
        return cached
    result = _browser_fetch_url_text(url, timeout_ms=timeout_ms)
    _cache_set(BROWSER_FETCH_CACHE, cache_key, result)
    return result


def _search_with_duckduckgo(query: str, max_results: int = 5) -> List[dict]:
    try:
        from duckduckgo_search import DDGS
    except Exception:
        return []
    try:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    except Exception:
        return []


def _search_with_tavily(query: str, max_results: int = 5) -> List[dict]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return []
    try:
        from tavily import TavilyClient
    except Exception:
        return []
    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, max_results=max_results)
    except Exception:
        return []
    results = []
    for item in response.get("results", []) or []:
        results.append(
            {
                "title": item.get("title", ""),
                "body": item.get("content", ""),
                "href": item.get("url", ""),
            }
        )
    return results


def _run_search_provider_cached(provider_name: str, query: str, max_results: int = 5) -> List[dict]:
    normalized_query = normalize_question_text(query).lower()
    cache_key = (provider_name, normalized_query, max_results)
    cached = _cache_get(SEARCH_RESULTS_CACHE, cache_key, SEARCH_RESULTS_TTL_SECONDS)
    if cached is not None:
        return list(cached)
    results = _run_search_provider(provider_name, query, max_results=max_results)
    _cache_set(SEARCH_RESULTS_CACHE, cache_key, list(results))
    return results


def _strip_html_to_text(html: str) -> str:
    if not html:
        return ""
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_links(base_url: str, html: str) -> List[str]:
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html or "", flags=re.IGNORECASE)
    base_domain = _domain_from_url(base_url)
    candidates: List[str] = []
    for href in hrefs:
        if not href or href.startswith("#") or href.startswith(("mailto:", "tel:")):
            continue
        full = urljoin(base_url, href)
        if not full.startswith("http"):
            continue
        if _domain_from_url(full) != base_domain:
            continue
        lowered = full.lower()
        if any(lowered.endswith(ext) for ext in [".css", ".js", ".ico", ".png", ".jpg", ".jpeg", ".svg", ".pdf", ".woff", ".woff2"]):
            continue
        candidates.append(full)
    priority_tokens = ["calendar", "registrar", "drop", "late", "deadline", "leadership", "administration", "about", "directory", "office"]
    candidates = sorted(
        list(dict.fromkeys(candidates)),
        key=lambda url: sum(1 for token in priority_tokens if token in url.lower()),
        reverse=True,
    )
    return candidates[:20]


def _domain_matches_required(url: str, preferred_domains: Sequence[str], required_mode: str) -> bool:
    if required_mode == "none" or not preferred_domains:
        return True
    domain = _domain_from_url(url)
    if required_mode == "strict":
        return any(domain.endswith(preferred) for preferred in preferred_domains)
    return True


def _score_authority(url: str, preferred_domains: Sequence[str], required_mode: str) -> float:
    domain = _domain_from_url(url)
    score = 0.0
    if preferred_domains and any(domain.endswith(preferred) for preferred in preferred_domains):
        score += 8.0 if required_mode == "strict" else 5.0
    if domain.endswith(".gov") or domain.endswith(".edu"):
        score += 4.0
    if any(token in domain for token in ["wikipedia.org", "reddit.com", "facebook.com", "linkedin.com"]):
        score -= 4.0
    return score


def _score_relevance(question: str, title: str, text: str) -> float:
    question_tokens = set(_question_tokens(question))
    if not question_tokens:
        return 0.0
    haystack_tokens = set(_question_tokens(f"{title} {text}"))
    overlap = len(question_tokens & haystack_tokens)
    return float(overlap)


def _score_freshness(context: SearchContext, title: str, text: str) -> float:
    score = 0.0
    if context.year and str(context.year) in f"{title} {text}":
        score += 2.0
    if context.requires_freshness:
        current_year = str(datetime.now().year)
        if current_year in f"{title} {text}":
            score += 1.0
    return score


def _preliminary_result_score(result: dict, context: SearchContext, plan: SearchPlan) -> float:
    url = str(result.get("href", "") or "").strip()
    title = str(result.get("title", "") or "")
    snippet = str(result.get("body", "") or "")
    score = _score_authority(url, plan.preferred_domains, plan.required_domain_mode)
    score += _score_relevance(context.corrected_question, title, snippet)
    score += _score_freshness(context, title, snippet)
    if any(token in url.lower() for token in ["calendar", "registrar", "deadline", "drop", "leadership", "about"]):
        score += 1.5
    return score


def _requested_deadline_labels(question: str) -> List[str]:
    lowered = (question or "").lower()
    labels: List[str] = []
    if "late drop" in lowered:
        labels.append("late_drop")
    if any(token in lowered for token in ["early drop", "drop-add", "drop add", "regular drop-add"]):
        labels.append("early_drop")
    return labels or ["late_drop", "early_drop"]


def _source_urls_from_clusters(clusters: Sequence[Dict[str, Any]], limit: int = 5) -> List[str]:
    urls: List[str] = []
    for cluster in clusters[:3]:
        urls.extend(source["url"] for source in cluster.get("sources", []))
    return list(dict.fromkeys(url for url in urls if url))[:limit]


def _has_required_sources(source_urls: Sequence[str], plan: SearchPlan, context: SearchContext) -> bool:
    policy_key = _preferred_policy_key(context)
    min_citations = int(SOURCE_POLICY_MATRIX.get(policy_key, {}).get("min_citations", 1))
    unique_urls = list(dict.fromkeys(url for url in source_urls if url))
    if plan.required_domain_mode == "strict":
        unique_urls = [
            url for url in unique_urls
            if _domain_matches_required(url, plan.preferred_domains, "strict")
        ]
    return len(unique_urls) >= min_citations


def _should_try_browser_for_evidence(evidence: SearchEvidence, context: SearchContext, plan: SearchPlan) -> bool:
    if not plan.allow_browser_fallback:
        return False
    if evidence.fetched_text and len(evidence.fetched_text) >= MIN_BROWSER_FALLBACK_TEXT_LENGTH and evidence.extracted_claims:
        return False
    if plan.required_domain_mode == "strict":
        return _domain_matches_required(evidence.url, plan.preferred_domains, "strict")
    return evidence.authority_score >= 4.0 or any(
        domain and evidence.domain.endswith(domain)
        for domain in plan.preferred_domains
    )


def _candidate_sentences(text: str) -> List[str]:
    chunks = re.split(r"(?<=[\.\?!])\s+", text or "")
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _extract_deadline_claims(text: str, context: SearchContext) -> List[Dict[str, Any]]:
    claims: List[Dict[str, Any]] = []
    lowered = text.lower()
    labels = {
        "late_drop": ["late drop", "late-drop"],
        "early_drop": ["early drop", "drop/add", "drop add", "regular drop-add"],
    }
    target_year = context.year or datetime.now().year
    for label, tokens in labels.items():
        for token in tokens:
            for match in re.finditer(re.escape(token), lowered):
                start = max(0, match.start() - 220)
                end = min(len(text), match.end() + 260)
                window = text[start:end]
                for date_candidate in DATE_RE.findall(window):
                    parsed = _parse_date_candidate(date_candidate, fallback_year=target_year)
                    if not parsed:
                        continue
                    claims.append(
                        {
                            "kind": "date",
                            "label": label,
                            "value": parsed.strftime("%B %d, %Y"),
                            "normalized": f"{label}:{parsed.date().isoformat()}",
                            "supporting_text": window.strip(),
                        }
                    )
    return claims


def _extract_role_claims(text: str, context: SearchContext) -> List[Dict[str, Any]]:
    claims: List[Dict[str, Any]] = []
    lowered = text.lower()
    role_tokens = ["ceo", "president", "chancellor", "provost", "dean", "director", "chair"]
    active_role_tokens = [token for token in role_tokens if token in context.corrected_question.lower()] or role_tokens
    for token in active_role_tokens:
        for match in re.finditer(re.escape(token), lowered):
            start = max(0, match.start() - 160)
            end = min(len(text), match.end() + 200)
            window = text[start:end]
            person_match = PERSON_NAME_RE.search(window)
            if person_match:
                value = person_match.group(0).strip()
                claims.append(
                    {
                        "kind": "person",
                        "label": token,
                        "value": value,
                        "normalized": f"{token}:{value.lower()}",
                        "supporting_text": window.strip(),
                    }
                )
    return claims


def _extract_fact_claims(text: str, context: SearchContext) -> List[Dict[str, Any]]:
    claims: List[Dict[str, Any]] = []
    question_lower = context.corrected_question.lower()
    target_year = context.year or datetime.now().year
    for sentence in _candidate_sentences(text):
        if _score_relevance(context.corrected_question, "", sentence) <= 0:
            continue
        if "when" in question_lower:
            for date_candidate in DATE_RE.findall(sentence):
                parsed = _parse_date_candidate(date_candidate, fallback_year=target_year)
                if parsed:
                    claims.append(
                        {
                            "kind": "date",
                            "label": "answer",
                            "value": parsed.strftime("%B %d, %Y"),
                            "normalized": parsed.date().isoformat(),
                            "supporting_text": sentence,
                        }
                    )
        elif "who" in question_lower:
            person_match = PERSON_NAME_RE.search(sentence)
            if person_match:
                value = person_match.group(0).strip()
                claims.append(
                    {
                        "kind": "person",
                        "label": "answer",
                        "value": value,
                        "normalized": value.lower(),
                        "supporting_text": sentence,
                    }
                )
        else:
            claims.append(
                {
                    "kind": "sentence",
                    "label": "answer",
                    "value": sentence.strip(),
                    "normalized": re.sub(r"\s+", " ", sentence.lower()).strip(),
                    "supporting_text": sentence.strip(),
                }
            )
    return claims[:8]


def _extract_claims_for_evidence(evidence: SearchEvidence, context: SearchContext, plan: SearchPlan) -> List[Dict[str, Any]]:
    text = evidence.fetched_text or evidence.snippet or evidence.title
    if not text:
        return []
    if plan.extraction_mode == "deadline_rows":
        return _extract_deadline_claims(text, context)
    if plan.extraction_mode == "named_role":
        return _extract_role_claims(text, context)
    return _extract_fact_claims(text, context)


def _parse_date_candidate(date_text: str, fallback_year: Optional[int] = None) -> Optional[datetime]:
    cleaned = normalize_question_text(date_text).rstrip(".,")
    formats = ["%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y", "%B %d", "%b %d"]
    for fmt in formats:
        try:
            parsed = datetime.strptime(cleaned, fmt)
            if "%Y" not in fmt and fallback_year:
                parsed = parsed.replace(year=fallback_year)
            return parsed
        except Exception:
            continue
    return None


def _build_evidence(url: str, title: str, snippet: str, fetched_text: str, context: SearchContext, plan: SearchPlan) -> SearchEvidence:
    evidence = SearchEvidence(
        url=url,
        title=title,
        snippet=snippet,
        fetched_text=fetched_text,
        domain=_domain_from_url(url),
    )
    evidence.authority_score = _score_authority(url, plan.preferred_domains, plan.required_domain_mode)
    evidence.relevance_score = _score_relevance(context.corrected_question, title, fetched_text or snippet)
    evidence.freshness_score = _score_freshness(context, title, fetched_text or snippet)
    evidence.extracted_claims = _extract_claims_for_evidence(evidence, context, plan)
    return evidence


def _fetch_and_build_evidence(
    url: str,
    title: str,
    snippet: str,
    context: SearchContext,
    plan: SearchPlan,
    allow_browser_fallback: bool,
) -> Optional[SearchEvidence]:
    html_text, error = _fetch_url_text_cached(url)
    fetched_text = _strip_html_to_text(html_text or "")
    if (not fetched_text or len(fetched_text) < MIN_BROWSER_FALLBACK_TEXT_LENGTH) and allow_browser_fallback:
        browser_html, browser_error = _browser_fetch_url_text_cached(url)
        if browser_html:
            fetched_text = _strip_html_to_text(browser_html)
        elif error is None:
            error = browser_error
    if error and not snippet and not fetched_text:
        return None
    return _build_evidence(url, title, snippet, fetched_text, context, plan)


def _run_official_seed_fetch(context: SearchContext, plan: SearchPlan) -> List[SearchEvidence]:
    seeds: List[str] = []
    if any(domain.endswith("nith.ac.in") for domain in plan.preferred_domains):
        if plan.strategy == "identity_current_role":
            seeds.extend(OFFICIAL_SOURCE_SEEDS["nit_hamirpur_identity"])
        else:
            seeds.extend(OFFICIAL_SOURCE_SEEDS["nit_hamirpur_policy"])
    elif any(domain.endswith(("psu.edu", "pennstate.edu")) for domain in plan.preferred_domains):
        if plan.strategy == "identity_current_role":
            seeds.extend(OFFICIAL_SOURCE_SEEDS["penn_state_identity"])
        else:
            seeds.extend(OFFICIAL_SOURCE_SEEDS["penn_state_policy"])
    filtered_urls = [
        url
        for url in list(dict.fromkeys(seeds))[:8]
        if _domain_matches_required(url, plan.preferred_domains, plan.required_domain_mode)
    ]
    return _parallel_map(
        filtered_urls,
        lambda url: _fetch_and_build_evidence(url, "", "", context, plan, allow_browser_fallback=False),
        MAX_CONCURRENT_FETCHES,
    )


def _run_official_domain_crawl(context: SearchContext, plan: SearchPlan, seed_evidence: Sequence[SearchEvidence]) -> List[SearchEvidence]:
    def _seed_candidates(seed: SearchEvidence) -> List[str]:
        text_html, error = _fetch_url_text_cached(seed.url)
        if error or not text_html:
            return []
        return [
            candidate
            for candidate in _extract_links(seed.url, text_html)[:8]
            if _domain_matches_required(candidate, plan.preferred_domains, plan.required_domain_mode)
        ]

    candidate_lists = _parallel_map(seed_evidence[:4], _seed_candidates, MAX_CONCURRENT_FETCHES)
    candidate_urls: List[str] = []
    for candidates in candidate_lists:
        candidate_urls.extend(candidates or [])

    evidence = _parallel_map(
        list(dict.fromkeys(candidate_urls)),
        lambda candidate: _fetch_and_build_evidence(candidate, "", "", context, plan, allow_browser_fallback=False),
        MAX_CONCURRENT_FETCHES,
    )
    unique: Dict[str, SearchEvidence] = {}
    for item in evidence:
        unique[item.url] = item
    return list(unique.values())


def _run_search_provider(provider_name: str, query: str, max_results: int = 5) -> List[dict]:
    if provider_name == "duckduckgo":
        return _search_with_duckduckgo(query, max_results=max_results)
    if provider_name == "tavily":
        return _search_with_tavily(query, max_results=max_results)
    return []


def _build_search_evidence_from_results(
    results: Sequence[dict],
    context: SearchContext,
    plan: SearchPlan,
    fetch_limit: int,
) -> List[SearchEvidence]:
    unique_results: Dict[str, dict] = {}
    for result in results:
        url = str(result.get("href", "") or "").strip()
        if not url:
            continue
        if plan.required_domain_mode == "strict" and not _domain_matches_required(url, plan.preferred_domains, plan.required_domain_mode):
            continue
        if url not in unique_results:
            unique_results[url] = result

    ranked_results = sorted(
        unique_results.values(),
        key=lambda item: _preliminary_result_score(item, context, plan),
        reverse=True,
    )[:fetch_limit]

    evidence = _parallel_map(
        ranked_results,
        lambda result: _fetch_and_build_evidence(
            str(result.get("href", "") or "").strip(),
            str(result.get("title", "") or ""),
            str(result.get("body", "") or ""),
            context,
            plan,
            allow_browser_fallback=False,
        ),
        MAX_CONCURRENT_FETCHES,
    )

    if plan.allow_browser_fallback:
        browser_candidates = [
            item for item in evidence
            if _should_try_browser_for_evidence(item, context, plan)
        ][:MAX_BROWSER_FALLBACKS]
        if browser_candidates:
            refreshed = _parallel_map(
                browser_candidates,
                lambda item: _fetch_and_build_evidence(
                    item.url,
                    item.title,
                    item.snippet,
                    context,
                    plan,
                    allow_browser_fallback=True,
                ),
                min(MAX_BROWSER_FALLBACKS, MAX_CONCURRENT_FETCHES),
            )
            refreshed_by_url = {item.url: item for item in refreshed}
            evidence = [refreshed_by_url.get(item.url, item) for item in evidence]
    unique: Dict[str, SearchEvidence] = {}
    for item in evidence:
        unique[item.url] = item
    return list(unique.values())


def _cluster_claims(evidence_list: Sequence[SearchEvidence]) -> List[Dict[str, Any]]:
    clusters: Dict[str, Dict[str, Any]] = {}
    for evidence in evidence_list:
        evidence_score = evidence.authority_score + evidence.relevance_score + evidence.freshness_score
        for claim in evidence.extracted_claims:
            key = str(claim.get("normalized") or claim.get("value") or "").strip().lower()
            if not key:
                continue
            cluster = clusters.setdefault(
                key,
                {
                    "claim": claim,
                    "score": 0.0,
                    "sources": [],
                },
            )
            cluster["score"] += evidence_score
            cluster["sources"].append({"url": evidence.url, "title": evidence.title})
    ranked = sorted(clusters.values(), key=lambda item: item["score"], reverse=True)
    return ranked


def _format_citations(source_urls: Sequence[str]) -> str:
    unique = list(dict.fromkeys(url for url in source_urls if url))
    if not unique:
        return ""
    return "\n".join(f"Source: {url}" for url in unique[:3])


def _synthesize_with_llm(
    llm: Any,
    context: SearchContext,
    plan: SearchPlan,
    top_clusters: Sequence[Dict[str, Any]],
) -> Optional[str]:
    if llm is None or not top_clusters:
        return None
    claims_text = []
    for index, cluster in enumerate(top_clusters[:4], start=1):
        claim = cluster["claim"]
        sources = ", ".join(source["url"] for source in cluster.get("sources", [])[:3])
        claims_text.append(
            f"[{index}] kind={claim.get('kind')} value={claim.get('value')} support={claim.get('supporting_text')} sources={sources}"
        )
    try:
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are Leo, a factual answer synthesizer for academic advising research. Answer the user's "
                        "question using ONLY the extracted claims and cited sources provided by the search pipeline.\n\n"
                        "Language safety: never write, quote, list, transform, repeat, praise, or endorse profanity, "
                        "foul language, vulgar insults, slurs, or obscene wording. If source text contains foul "
                        "language, summarize it neutrally without repeating the wording.\n\n"
                        "Reason privately before answering: compare the claims, prefer the most directly supported "
                        "and most current source, detect conflicts, and decide whether the evidence is sufficient. "
                        "Do not reveal private chain-of-thought.\n\n"
                        "Output contract:\n"
                        "- Be concise and directly answer the question first.\n"
                        "- Cite sources explicitly using the provided source URLs.\n"
                        "- If claims conflict, state the conflict and identify which source supports each claim.\n"
                        "- If the evidence is insufficient, say so instead of guessing.\n\n"
                        "Examples:\n"
                        "- Supported date: 'The late-drop deadline is April 5, 2026. Source: https://...'\n"
                        "- Conflicting evidence: 'One source lists April 5, while another lists April 7; confirm with the Registrar.'\n"
                        "- Insufficient evidence: 'I do not see that deadline in the extracted sources, so I cannot verify it here.'"
                    )
                ),
                HumanMessage(
                    content=(
                        f"{context.current_datetime}\nQuestion: {context.corrected_question}\n"
                        f"Strategy: {plan.strategy}\n\nClaims:\n" + "\n".join(claims_text)
                    )
                ),
            ]
        )
    except Exception:
        return None
    content = str(getattr(response, "content", response) or "").strip()
    return content or None


def _build_deadline_answer(context: SearchContext, clusters: Sequence[Dict[str, Any]]) -> Optional[str]:
    requested_labels = set(_requested_deadline_labels(context.corrected_question))
    late_drop = None
    early_drop = None
    sources: List[str] = []
    for cluster in clusters:
        claim = cluster["claim"]
        label = str(claim.get("label") or "")
        if label == "late_drop" and "late_drop" in requested_labels and late_drop is None:
            late_drop = str(claim.get("value") or "")
            sources.extend(source["url"] for source in cluster.get("sources", []))
        if label == "early_drop" and "early_drop" in requested_labels and early_drop is None:
            early_drop = str(claim.get("value") or "")
            sources.extend(source["url"] for source in cluster.get("sources", []))
    if not late_drop and not early_drop:
        return None
    lines = []
    target_label = " ".join(part for part in [context.campus or "Requested campus", (context.term or "").title(), str(context.year or "")] if part).strip()
    prefix = f"{target_label} drop deadlines:".strip()
    lines.append(prefix)
    if early_drop:
        lines.append(f"- Early drop deadline: {early_drop}")
    if late_drop:
        lines.append(f"- Late drop deadline: {late_drop}")
    citation_block = _format_citations(sources)
    if citation_block:
        lines.append(citation_block)
    return "\n".join(line for line in lines if line)


def _build_role_answer(clusters: Sequence[Dict[str, Any]]) -> Optional[str]:
    if not clusters:
        return None
    claim = clusters[0]["claim"]
    value = str(claim.get("value") or "").strip()
    if not value:
        return None
    sources = [source["url"] for source in clusters[0].get("sources", [])]
    answer = f"The current answer appears to be {value}."
    citation_block = _format_citations(sources)
    return answer + (f"\n{citation_block}" if citation_block else "")


def _build_generic_answer(
    context: SearchContext,
    clusters: Sequence[Dict[str, Any]],
    llm: Any,
) -> Optional[str]:
    synthesized = _synthesize_with_llm(llm, context, SearchPlan(
        strategy=context.active_task_kind,
        query_variants=[],
        preferred_domains=[],
        required_domain_mode="preferred",
        providers=[],
        fetch_mode="fetch_and_extract",
        extraction_mode="generic_claim",
        max_rounds=0,
        allow_browser_fallback=False,
        success_criteria=[],
    ), clusters)
    if synthesized:
        return synthesized
    if not clusters:
        return None
    claim = clusters[0]["claim"]
    value = str(claim.get("value") or "").strip()
    if not value:
        return None
    sources = [source["url"] for source in clusters[0].get("sources", [])]
    citation_block = _format_citations(sources)
    return value + (f"\n{citation_block}" if citation_block else "")


def _rank_evidence(evidence_list: Sequence[SearchEvidence]) -> List[SearchEvidence]:
    return sorted(
        evidence_list,
        key=lambda item: item.authority_score + item.relevance_score + item.freshness_score + (2.0 if item.extracted_claims else 0.0),
        reverse=True,
    )


def _resolve_answer_from_evidence(
    context: SearchContext,
    plan: SearchPlan,
    evidence_list: Sequence[SearchEvidence],
    llm: Any,
) -> Tuple[Optional[str], List[str], List[Dict[str, Any]], List[SearchEvidence]]:
    ranked_evidence = _rank_evidence(evidence_list)
    clusters = _cluster_claims(ranked_evidence)
    answer_text: Optional[str] = None
    if plan.strategy == "deadline_exact_date":
        answer_text = _build_deadline_answer(context, clusters)
    elif plan.strategy == "identity_current_role":
        answer_text = _build_role_answer(clusters)
    else:
        answer_text = _build_generic_answer(context, clusters, llm)
    source_urls = _source_urls_from_clusters(clusters, limit=5)
    return answer_text, source_urls, clusters, ranked_evidence


def _can_early_exit(
    context: SearchContext,
    plan: SearchPlan,
    answer_text: Optional[str],
    source_urls: Sequence[str],
    clusters: Sequence[Dict[str, Any]],
) -> bool:
    if not answer_text or not clusters:
        return False
    if context.requires_exact_date and not DATE_RE.search(answer_text):
        return False
    if not _has_required_sources(source_urls, plan, context):
        return False
    top_score = float(clusters[0].get("score", 0.0)) if clusters else 0.0
    if plan.required_domain_mode == "strict":
        return top_score >= 6.0
    return top_score >= 3.0


def _record_active_search_context(user_id: str, context: SearchContext, plan: SearchPlan, source_urls: Sequence[str]):
    CURRENT_TASK_CONTEXT_STORE[user_id] = {
        "campus": context.campus,
        "term": context.term,
        "year": context.year,
        "active_task_kind": plan.strategy,
        "domain_hints": list(plan.preferred_domains),
        "last_source_urls": list(dict.fromkeys(url for url in source_urls if url))[:5],
        "updated_at": datetime.now().isoformat(),
    }


def _build_success_answer(
    user_id: str,
    context: SearchContext,
    plan: SearchPlan,
    answer_text: str,
    source_urls: Sequence[str],
    attempted_strategies: Sequence[str],
) -> ResolvedResearchAnswer:
    citations = list(dict.fromkeys(url for url in source_urls if url))[:3]
    all_sources = list(dict.fromkeys(url for url in source_urls if url))[:5]
    _record_active_search_context(user_id, context, plan, all_sources)
    return ResolvedResearchAnswer(
        answer_text=answer_text,
        citations=citations,
        confidence=min(0.95, 0.35 + 0.1 * max(1, len(citations))),
        source_urls=all_sources,
        gaps=[],
        attempted_strategies=list(dict.fromkeys(str(strategy) for strategy in attempted_strategies if strategy)),
        exhausted=False,
        strategy=plan.strategy,
        completed_ladder=True,
        exhaustion_reason=None,
    )


def _build_exhausted_answer(context: SearchContext, considered: Sequence[SearchEvidence], reason: str) -> ResolvedResearchAnswer:
    best_sources = [item.url for item in sorted(
        considered,
        key=lambda evidence: evidence.authority_score + evidence.relevance_score + evidence.freshness_score,
        reverse=True,
    )[:3]]
    checked = "\n".join(f"Source checked: {url}" for url in best_sources if url)
    answer_lines = [
        "I couldn't verify the answer with enough confidence after exhausting the available research steps.",
    ]
    if checked:
        answer_lines.append(checked)
    answer_lines.append(f"Reason: {reason}.")
    return ResolvedResearchAnswer(
        answer_text="\n".join(answer_lines),
        citations=list(best_sources),
        confidence=0.0,
        source_urls=list(best_sources),
        gaps=[reason],
        attempted_strategies=[context.active_task_kind],
        exhausted=True,
        strategy=context.active_task_kind,
        completed_ladder=True,
        exhaustion_reason=reason,
    )


def _query_batches(queries: Sequence[str]) -> List[List[str]]:
    normalized_queries = [query for query in queries if query]
    if not normalized_queries:
        return []
    first_batch = normalized_queries[:INITIAL_QUERY_BATCH_SIZE]
    remaining = normalized_queries[INITIAL_QUERY_BATCH_SIZE:EXPANDED_QUERY_BATCH_SIZE]
    batches = [first_batch]
    if remaining:
        batches.append(remaining)
    return [batch for batch in batches if batch]


def _run_provider_batch(provider: str, queries: Sequence[str], max_results: int = 5) -> Tuple[List[dict], int]:
    if not queries:
        return [], 0

    cached_query_count = 0
    for query in queries:
        cache_key = (provider, normalize_question_text(query).lower(), max_results)
        if _cache_get(SEARCH_RESULTS_CACHE, cache_key, SEARCH_RESULTS_TTL_SECONDS) is not None:
            cached_query_count += 1

    results = _parallel_map(
        list(queries),
        lambda query: _run_search_provider_cached(provider, query, max_results=max_results),
        MAX_CONCURRENT_SEARCHES,
    )
    flattened: List[dict] = []
    for query_results in results:
        flattened.extend(query_results or [])
    return flattened, cached_query_count


def run_search_pipeline(
    user_id: str,
    user_question: str,
    corrected_question: Optional[str],
    conversation_history: Optional[List[dict]],
    llm: Any = None,
    trace: Optional[Dict[str, Any]] = None,
    preferred_strategy: Optional[str] = None,
    preferred_domains: Optional[Sequence[str]] = None,
    required_domain_mode: Optional[str] = None,
) -> ResolvedResearchAnswer:
    context = resolve_search_context(
        user_id=user_id,
        original_question=user_question,
        corrected_question=corrected_question,
        conversation_history=conversation_history,
    )
    plan = plan_search(
        context=context,
        preferred_strategy=preferred_strategy,
        preferred_domains=preferred_domains,
        required_domain_mode=required_domain_mode,
    )
    run = _trace_search_run(trace, context, plan)

    considered: List[SearchEvidence] = []
    used: List[SearchEvidence] = []
    attempted_strategies = [plan.strategy]

    seed_evidence = _run_official_seed_fetch(context, plan)
    if seed_evidence:
        considered.extend(seed_evidence)
    _trace_search_round(trace, run, "official_seed_fetch", {"count": len(seed_evidence)})
    _trace_search_sources(trace, run, seed_evidence, used=False)

    answer_text, source_urls, clusters, ranked_evidence = _resolve_answer_from_evidence(context, plan, considered, llm)
    if _can_early_exit(context, plan, answer_text, source_urls, clusters):
        used = [item for item in ranked_evidence if item.url in source_urls[:5]]
        _trace_search_sources(trace, run, used, used=True)
        _trace_search_round(trace, run, "fast_path_exit", {"stage": "official_seed_fetch", "source_count": len(source_urls)})
        _set_trace_exhaustion(trace, run, None)
        return _build_success_answer(user_id, context, plan, str(answer_text), source_urls, attempted_strategies)

    crawl_evidence = _run_official_domain_crawl(context, plan, seed_evidence)
    if crawl_evidence:
        considered.extend(crawl_evidence)
    _trace_search_round(trace, run, "official_domain_crawl", {"count": len(crawl_evidence)})
    _trace_search_sources(trace, run, crawl_evidence, used=False)

    seen_urls = {item.url for item in considered}
    answer_text, source_urls, clusters, ranked_evidence = _resolve_answer_from_evidence(context, plan, considered, llm)
    if _can_early_exit(context, plan, answer_text, source_urls, clusters):
        used = [item for item in ranked_evidence if item.url in source_urls[:5]]
        _trace_search_sources(trace, run, used, used=True)
        _trace_search_round(trace, run, "fast_path_exit", {"stage": "official_domain_crawl", "source_count": len(source_urls)})
        _set_trace_exhaustion(trace, run, None)
        return _build_success_answer(user_id, context, plan, str(answer_text), source_urls, attempted_strategies)

    query_batches = _query_batches(plan.query_variants[:EXPANDED_QUERY_BATCH_SIZE])
    for batch_index, query_batch in enumerate(query_batches):
        search_results: List[dict] = []
        fetch_limit = INITIAL_SEARCH_FETCH_LIMIT if batch_index == 0 else EXPANDED_SEARCH_FETCH_LIMIT
        for provider in ["duckduckgo", "tavily"]:
            provider_results, cached_queries = _run_provider_batch(provider, query_batch, max_results=5)
            if provider_results:
                search_results.extend(provider_results)
            _trace_search_round(
                trace,
                run,
                provider,
                {
                    "batch": batch_index + 1,
                    "queries": len(query_batch),
                    "results": len(provider_results),
                    "cached_queries": cached_queries,
                },
            )

        search_evidence = _build_search_evidence_from_results(search_results, context, plan, fetch_limit=fetch_limit)
        for item in search_evidence:
            if item.url not in seen_urls:
                considered.append(item)
                seen_urls.add(item.url)
        _trace_search_sources(trace, run, search_evidence, used=False)

        answer_text, source_urls, clusters, ranked_evidence = _resolve_answer_from_evidence(context, plan, considered, llm)
        if _can_early_exit(context, plan, answer_text, source_urls, clusters):
            used = [item for item in ranked_evidence if item.url in source_urls[:5]]
            _trace_search_sources(trace, run, used, used=True)
            _trace_search_round(
                trace,
                run,
                "fast_path_exit",
                {"stage": f"search_batch_{batch_index + 1}", "source_count": len(source_urls)},
            )
            _set_trace_exhaustion(trace, run, None)
            return _build_success_answer(user_id, context, plan, str(answer_text), source_urls, attempted_strategies)

    answer_text, source_urls, clusters, ranked_evidence = _resolve_answer_from_evidence(context, plan, considered, llm)

    if answer_text:
        used.extend(item for item in ranked_evidence if item.url in source_urls[:5])
        _trace_search_sources(trace, run, used, used=True)
        _set_trace_exhaustion(trace, run, None)
        return _build_success_answer(user_id, context, plan, str(answer_text), source_urls, attempted_strategies)

    reason = "no authoritative claim cluster could be extracted"
    _record_active_search_context(user_id, context, plan, [item.url for item in ranked_evidence[:5]])
    _set_trace_exhaustion(trace, run, reason)
    return _build_exhausted_answer(context, ranked_evidence, reason)
