import sys
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch


LLM_DIR = Path(__file__).resolve().parent / "LLMAcadamic-Advisor" / "LLM"
if str(LLM_DIR) not in sys.path:
    sys.path.insert(0, str(LLM_DIR))

import web_search_pipeline as wsp  # noqa: E402


class WebSearchPipelineTests(unittest.TestCase):
    def setUp(self):
        wsp.CURRENT_TASK_CONTEXT_STORE.clear()
        wsp.reset_search_pipeline_caches()

    def test_resolve_search_context_uses_current_semester_and_active_thread(self):
        wsp.CURRENT_TASK_CONTEXT_STORE["policy-user"] = {
            "campus": "Penn State Abington",
            "term": "spring",
            "year": 2026,
            "active_task_kind": "deadline_exact_date",
        }

        context = wsp.resolve_search_context(
            user_id="policy-user",
            original_question="current semester",
            corrected_question="current semester",
            conversation_history=[
                {"role": "user", "content": "when is the late drop deadline for abington"},
            ],
            now=datetime(2026, 3, 21, 13, 0, 0),
        )

        self.assertEqual(context.term, "spring")
        self.assertEqual(context.year, 2026)
        self.assertEqual(context.campus, "Penn State Abington")
        self.assertEqual(context.active_task_kind, "deadline_exact_date")

    def test_run_search_pipeline_uses_secondary_provider_when_primary_is_junk(self):
        def fake_fetch(url, timeout=10):
            if "registrar.psu.edu" in url:
                return (
                    """
                    <html><body>
                    Spring 2026 academic calendar.
                    Late drop deadline: March 27, 2026.
                    Early drop deadline: January 18, 2026.
                    </body></html>
                    """,
                    None,
                )
            return ("<html><body>junk</body></html>", None)

        with patch.object(wsp, "_run_official_seed_fetch", return_value=[]), \
             patch.object(wsp, "_run_official_domain_crawl", return_value=[]), \
             patch.object(wsp, "_search_with_duckduckgo", return_value=[{"title": "Dictionary", "body": "late means delayed", "href": "https://example.com/late"}]), \
             patch.object(wsp, "_search_with_tavily", return_value=[{"title": "Registrar Calendar", "body": "Spring 2026 academic calendar late drop deadline", "href": "https://registrar.psu.edu/academic-calendars/2025-26.cfm"}]), \
             patch.object(wsp, "_fetch_url_text", side_effect=fake_fetch):
            result = wsp.run_search_pipeline(
                user_id="policy-user",
                user_question="when is the late drop deadline for abington",
                corrected_question="When is the late drop deadline for Penn State Abington Spring 2026?",
                conversation_history=[
                    {"role": "user", "content": "current semester"},
                    {"role": "assistant", "content": "The current semester is Spring 2026."},
                ],
                llm=None,
                preferred_strategy="deadline_exact_date",
                preferred_domains=["abington.psu.edu", "psu.edu", "pennstate.edu"],
                required_domain_mode="strict",
            )

        self.assertIn("March 27, 2026", result.answer_text)
        self.assertIn("Source:", result.answer_text)
        self.assertIn("registrar.psu.edu", "\n".join(result.source_urls))
        self.assertTrue(result.completed_ladder)

    def test_run_search_pipeline_answers_unexpected_generic_fact(self):
        def fake_fetch(url, timeout=10):
            return (
                """
                <html><body>
                Python 3.12 was released on October 2, 2023.
                </body></html>
                """,
                None,
            )

        with patch.object(wsp, "_run_official_seed_fetch", return_value=[]), \
             patch.object(wsp, "_run_official_domain_crawl", return_value=[]), \
             patch.object(wsp, "_search_with_duckduckgo", return_value=[{"title": "Python Release", "body": "Python 3.12 was released on October 2, 2023.", "href": "https://www.python.org/downloads/release/python-3120/"}]), \
             patch.object(wsp, "_search_with_tavily", return_value=[]), \
             patch.object(wsp, "_fetch_url_text", side_effect=fake_fetch):
            result = wsp.run_search_pipeline(
                user_id="generic-user",
                user_question="when was Python 3.12 released",
                corrected_question="When was Python 3.12 released?",
                conversation_history=[],
                llm=None,
            )

        self.assertIn("October 02, 2023", result.answer_text)
        self.assertIn("python.org", "\n".join(result.source_urls))
        self.assertFalse(result.exhausted)

    def test_run_search_pipeline_returns_exhaustion_with_checked_sources(self):
        def fake_fetch(url, timeout=10):
            return ("<html><body>No exact answer here.</body></html>", None)

        with patch.object(wsp, "_run_official_seed_fetch", return_value=[]), \
             patch.object(wsp, "_run_official_domain_crawl", return_value=[]), \
             patch.object(wsp, "_search_with_duckduckgo", return_value=[{"title": "AMD leadership", "body": "Leadership overview.", "href": "https://www.amd.com/en/company"}]), \
             patch.object(wsp, "_search_with_tavily", return_value=[]), \
             patch.object(wsp, "_fetch_url_text", side_effect=fake_fetch):
            result = wsp.run_search_pipeline(
                user_id="exhaust-user",
                user_question="who is the current ceo of amd",
                corrected_question="Who is the current CEO of AMD?",
                conversation_history=[],
                llm=None,
            )

        self.assertTrue(result.exhausted)
        self.assertIn("Source checked:", result.answer_text)
        self.assertTrue(result.source_urls)

    def test_run_search_pipeline_backfills_missing_trace_runs_key(self):
        trace = {
            "enabled": False,
            "search_pipeline": {
                "context": {},
                "plan": {},
                "rounds": [],
                "sources_considered": [],
                "sources_used": [],
                "exhaustion_reason": None,
            },
        }

        with patch.object(wsp, "_run_official_seed_fetch", return_value=[]), \
             patch.object(wsp, "_run_official_domain_crawl", return_value=[]), \
             patch.object(wsp, "_search_with_duckduckgo", return_value=[]), \
             patch.object(wsp, "_search_with_tavily", return_value=[]):
            result = wsp.run_search_pipeline(
                user_id="trace-user",
                user_question="when is the late drop deadline for this current semester",
                corrected_question="When is the late drop deadline for this current semester?",
                conversation_history=[],
                llm=None,
                trace=trace,
                preferred_strategy="deadline_exact_date",
                preferred_domains=["abington.psu.edu", "psu.edu", "pennstate.edu"],
                required_domain_mode="strict",
            )

        self.assertIn("runs", trace["search_pipeline"])
        self.assertTrue(trace["search_pipeline"]["runs"])
        self.assertTrue(result.exhausted)

    def test_run_search_pipeline_uses_fast_exit_before_web_providers(self):
        seed_evidence = [
            wsp.SearchEvidence(
                url="https://registrar.psu.edu/academic-calendars/2025-26.cfm",
                title="Spring 2026 calendar",
                domain="registrar.psu.edu",
                authority_score=12.0,
                relevance_score=4.0,
                freshness_score=2.0,
                extracted_claims=[
                    {
                        "kind": "date",
                        "label": "late_drop",
                        "value": "March 27, 2026",
                        "normalized": "late_drop:2026-03-27",
                        "supporting_text": "Late drop deadline: March 27, 2026.",
                    }
                ],
            )
        ]

        with patch.object(wsp, "_run_official_seed_fetch", return_value=seed_evidence), \
             patch.object(wsp, "_run_official_domain_crawl", return_value=[]), \
             patch.object(wsp, "_search_with_duckduckgo") as duck, \
             patch.object(wsp, "_search_with_tavily") as tavily:
            result = wsp.run_search_pipeline(
                user_id="fast-exit-user",
                user_question="when is the late drop deadline for abington",
                corrected_question="When is the late drop deadline for Penn State Abington Spring 2026?",
                conversation_history=[],
                llm=None,
                preferred_strategy="deadline_exact_date",
                preferred_domains=["abington.psu.edu", "psu.edu", "pennstate.edu"],
                required_domain_mode="strict",
            )

        self.assertIn("March 27, 2026", result.answer_text)
        self.assertFalse(duck.called)
        self.assertFalse(tavily.called)

    def test_run_search_pipeline_reuses_cached_search_and_fetch_results(self):
        fetch_calls = {"count": 0}

        def fake_fetch(url, timeout=10):
            fetch_calls["count"] += 1
            return (
                """
                <html><body>
                Python 3.12 was released on October 2, 2023.
                </body></html>
                """,
                None,
            )

        with patch.object(wsp, "_run_official_seed_fetch", return_value=[]), \
             patch.object(wsp, "_run_official_domain_crawl", return_value=[]), \
             patch.object(wsp, "_run_search_provider", return_value=[{"title": "Python Release", "body": "Python 3.12 was released on October 2, 2023.", "href": "https://www.python.org/downloads/release/python-3120/"}]) as run_provider, \
             patch.object(wsp, "_fetch_url_text", side_effect=fake_fetch):
            first = wsp.run_search_pipeline(
                user_id="cache-user",
                user_question="when was Python 3.12 released",
                corrected_question="When was Python 3.12 released?",
                conversation_history=[],
                llm=None,
            )
            second = wsp.run_search_pipeline(
                user_id="cache-user",
                user_question="when was Python 3.12 released",
                corrected_question="When was Python 3.12 released?",
                conversation_history=[],
                llm=None,
            )

        self.assertIn("October 02, 2023", first.answer_text)
        self.assertIn("October 02, 2023", second.answer_text)
        self.assertEqual(fetch_calls["count"], 1)
        self.assertEqual(run_provider.call_count, 4)

    def test_run_search_pipeline_limits_browser_fallback_candidates(self):
        def fake_fetch(url, timeout=10):
            return ("<html><body>short</body></html>", None)

        browser_calls = {"count": 0}

        def fake_browser(url, timeout_ms=15000):
            browser_calls["count"] += 1
            suffix = browser_calls["count"]
            return (
                f"""
                <html><body>
                Spring 2026 late drop deadline: March 2{suffix}, 2026.
                </body></html>
                """,
                None,
            )

        with patch.object(wsp, "_run_official_seed_fetch", return_value=[]), \
             patch.object(wsp, "_run_official_domain_crawl", return_value=[]), \
             patch.object(
                 wsp,
                 "_search_with_duckduckgo",
                 return_value=[
                     {"title": "Calendar 1", "body": "deadline", "href": "https://registrar.psu.edu/calendar-1"},
                     {"title": "Calendar 2", "body": "deadline", "href": "https://registrar.psu.edu/calendar-2"},
                     {"title": "Calendar 3", "body": "deadline", "href": "https://registrar.psu.edu/calendar-3"},
                 ],
             ), \
             patch.object(wsp, "_search_with_tavily", return_value=[]), \
             patch.object(wsp, "_fetch_url_text", side_effect=fake_fetch), \
             patch.object(wsp, "_browser_fetch_url_text", side_effect=fake_browser):
            result = wsp.run_search_pipeline(
                user_id="browser-user",
                user_question="when is the late drop deadline for abington",
                corrected_question="When is the late drop deadline for Penn State Abington Spring 2026?",
                conversation_history=[],
                llm=None,
                preferred_strategy="deadline_exact_date",
                preferred_domains=["abington.psu.edu", "psu.edu", "pennstate.edu"],
                required_domain_mode="strict",
            )

        self.assertIn("Late drop deadline", result.answer_text)
        self.assertLessEqual(browser_calls["count"], wsp.MAX_BROWSER_FALLBACKS)


if __name__ == "__main__":
    unittest.main()
