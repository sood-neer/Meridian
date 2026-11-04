import sys
import unittest
from pathlib import Path
from typing import List
from unittest.mock import patch


LLM_DIR = Path(__file__).resolve().parent / "LLMAcadamic-Advisor" / "LLM"
if str(LLM_DIR) not in sys.path:
    sys.path.insert(0, str(LLM_DIR))

from orchestration_pipeline import ExecutorResult, build_orchestration_plan, run_three_stage_pipeline  # noqa: E402

MAIN_COPY_IMPORT_ERROR = None
try:
    import main_copy  # noqa: E402
except Exception as exc:  # pragma: no cover - environment-dependent
    main_copy = None
    MAIN_COPY_IMPORT_ERROR = exc


class FakeResponse:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    def __init__(self, responses=None):
        self.responses = list(responses or [""])
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.responses:
            content = self.responses.pop(0)
        else:
            content = ""
        return FakeResponse(content)


FULL_PLAN_TEXT = """Semester 3:
BA 243 (prerequisites: None)
CAS 352 (prerequisites: None)
Total Credits: 15
---
Semester 4:
MGMT 301 (prerequisites: None)
STAT 200 (prerequisites: None)
Total Credits: 15
---
Semester 5:
FIN 301 (prerequisites: None)
SCM 301 (prerequisites: None)
Total Credits: 15"""

MARKDOWN_PLAN_TEXT = """Here's a suggested graduation plan for a junior Computer Science major.

**Semester 5:**
- CMPSC 313
- MATH 425
- **Total Credits:** 6

---

**Semester 6:**
- CMPSC 330
- CMPSC 360
- **Total Credits:** 15"""


class OrchestrationPipelineTests(unittest.TestCase):
    def test_build_orchestration_plan_narrows_prior_plan_without_tools(self):
        history = [
            {"role": "user", "content": "give me a graduation plan for a sophmore in business managment"},
            {"role": "assistant", "content": FULL_PLAN_TEXT},
        ]
        plan, metadata = build_orchestration_plan(
            user_question="how about just a plan for next semester only",
            conversation_history=history,
            has_transcript=False,
            available_tools=["conversation_history", "deterministic_planner", "transcript_state"],
            llm=FakeLLM([""]),
        )

        self.assertEqual(plan.plan_scope, "single_semester")
        self.assertEqual(plan.deliverable, "single_semester_plan")
        self.assertFalse(plan.use_tools)
        self.assertEqual(plan.executor_path, "conversation_plan_slice")
        self.assertIn("sophomore", build_orchestration_plan(
            user_question="give me a graduation plan for a sophmore in business managment",
            conversation_history=[],
            has_transcript=False,
            available_tools=["deterministic_planner"],
            llm=FakeLLM([""]),
        )[0].corrected_question.lower())
        self.assertTrue(metadata["recent_plan_available"])

    def test_build_orchestration_plan_corrects_typos(self):
        plan, _ = build_orchestration_plan(
            user_question="give me a graduation plan for a sophmore in business managment",
            conversation_history=[],
            has_transcript=False,
            available_tools=["deterministic_planner"],
            llm=FakeLLM([""]),
        )

        self.assertIn("sophomore", plan.corrected_question.lower())
        self.assertIn("management", plan.corrected_question.lower())
        self.assertEqual(plan.deliverable, "full_graduation_plan")

    def test_build_orchestration_plan_overrides_bad_slice_for_profile_recall(self):
        history = [
            {"role": "user", "content": "give me a graduation plan for a junior computer science major"},
            {"role": "assistant", "content": MARKDOWN_PLAN_TEXT},
            {"role": "user", "content": "just for next semester"},
            {"role": "assistant", "content": "As a junior Computer Science major, here is a suggested course plan for your next semester."},
        ]
        bad_orchestrator_json = """{
          "normalized_question": "What year and major am I again?",
          "corrected_question": "What year and major am I again?",
          "resolved_user_need": "User needs to confirm their academic year and major.",
          "domain": "general_academic_chat",
          "deliverable": "direct_answer",
          "plan_scope": "single_semester",
          "semester_horizon": "next semester",
          "use_tools": false,
          "allowed_tools": [],
          "executor_path": "conversation_plan_slice",
          "must_not_do": [],
          "success_criteria": "User confirms their academic year and major."
        }"""

        plan, metadata = build_orchestration_plan(
            user_question="what year and major am i again?",
            conversation_history=history,
            has_transcript=False,
            available_tools=["conversation_history", "deterministic_planner"],
            llm=FakeLLM([bad_orchestrator_json]),
        )

        self.assertEqual(plan.plan_scope, "direct_answer")
        self.assertEqual(plan.executor_path, "conversation_profile_recall")
        self.assertEqual(plan.semester_horizon, 0)
        self.assertFalse(plan.use_tools)
        self.assertTrue(metadata["recent_plan_available"])

    def test_build_orchestration_plan_keeps_unexpected_followup_in_reasoning_mode(self):
        history = [
            {"role": "user", "content": "give me a graduation plan for a junior computer science major"},
            {"role": "assistant", "content": MARKDOWN_PLAN_TEXT},
        ]
        bad_orchestrator_json = """{
          "normalized_question": "Why did you recommend that schedule?",
          "corrected_question": "Why did you recommend that schedule?",
          "resolved_user_need": "Explain the plan.",
          "domain": "general_academic_chat",
          "deliverable": "direct_answer",
          "plan_scope": "single_semester",
          "semester_horizon": "next semester",
          "use_tools": false,
          "allowed_tools": [],
          "executor_path": "conversation_plan_slice",
          "must_not_do": [],
          "success_criteria": "Explain the recommendation."
        }"""

        plan, _ = build_orchestration_plan(
            user_question="why did you recommend that schedule?",
            conversation_history=history,
            has_transcript=False,
            available_tools=["conversation_history", "deterministic_planner"],
            llm=FakeLLM([bad_orchestrator_json]),
        )

        self.assertEqual(plan.plan_scope, "direct_answer")
        self.assertEqual(plan.executor_path, "general_academic_chat")
        self.assertFalse(plan.use_tools)

    def test_build_orchestration_plan_forces_policy_followup_to_direct_answer(self):
        history = [
            {"role": "user", "content": "when is the late drop and early drop deadlines for this current semester"},
            {"role": "assistant", "content": "Tell me your campus and semester and I can narrow it down."},
            {"role": "user", "content": "I am at penn state abington"},
            {"role": "assistant", "content": "I can help find the exact dates if you tell me the semester."},
        ]
        bad_orchestrator_json = """{
          "normalized_question": "spring 2026",
          "corrected_question": "What are the exact Penn State Abington academic calendar dates for Spring 2026?",
          "resolved_user_need": "The user is asking for the Spring 2026 semester dates.",
          "domain": "policy",
          "deliverable": "policy_lookup",
          "plan_scope": "single_semester",
          "semester_horizon": "spring_2026",
          "use_tools": true,
          "allowed_tools": ["policy_retriever", "web_search"],
          "executor_path": "policy_lookup",
          "must_not_do": [],
          "success_criteria": "Return the exact dates."
        }"""

        plan, _ = build_orchestration_plan(
            user_question="spring 2026",
            conversation_history=history,
            has_transcript=False,
            available_tools=["conversation_history", "policy_retriever", "web_search"],
            llm=FakeLLM([bad_orchestrator_json]),
        )

        self.assertEqual(plan.domain, "policy")
        self.assertEqual(plan.plan_scope, "direct_answer")
        self.assertEqual(plan.executor_path, "policy_lookup")
        self.assertEqual(plan.semester_horizon, 0)

    def test_build_orchestration_plan_routes_unexpected_fact_to_research_lookup(self):
        plan, _ = build_orchestration_plan(
            user_question="when was Python 3.12 released",
            conversation_history=[],
            has_transcript=False,
            available_tools=["conversation_history", "web_search"],
            llm=FakeLLM([""]),
        )

        self.assertEqual(plan.plan_scope, "direct_answer")
        self.assertTrue(plan.use_tools)
        self.assertEqual(plan.executor_path, "research_lookup")
        self.assertIn("web_search", plan.allowed_tools)

    def test_generate_conversational_response_slices_prior_plan_followup(self):
        if main_copy is None:
            self.skipTest(f"main_copy unavailable in this environment: {MAIN_COPY_IMPORT_ERROR}")
        user_id = "pipeline-followup-test"
        main_copy.execution_trace_store.clear()
        history = [
            {"role": "user", "content": "give me a graduation plan for a sophmore in business managment"},
            {"role": "assistant", "content": FULL_PLAN_TEXT},
        ]

        with patch.object(main_copy, "llm_classifier", FakeLLM([""])), \
             patch.object(main_copy, "llm1", FakeLLM([""])), \
             patch.object(main_copy, "llm2", FakeLLM([""])), \
             patch.object(main_copy, "has_transcript", False), \
             patch.object(main_copy, "stored_pdf_docs", None), \
             patch.object(main_copy, "latest_student_data", None):
            response = main_copy.generate_conversational_response(
                "how about just a plan for next semester only",
                history,
                user_id=user_id,
                user_has_transcript=False,
            )

        self.assertIn("Semester 3:", response)
        self.assertNotIn("Semester 4:", response)

        trace = main_copy.get_execution_trace_for_user(user_id)
        self.assertEqual(trace["orchestrator"]["plan_scope"], "single_semester")
        self.assertEqual(trace["executor"]["attempts"][0]["result"]["tools_used"], [])

    def test_generate_conversational_response_recalls_major_and_year_from_history(self):
        if main_copy is None:
            self.skipTest(f"main_copy unavailable in this environment: {MAIN_COPY_IMPORT_ERROR}")
        user_id = "pipeline-profile-recall-test"
        main_copy.execution_trace_store.clear()
        history = [
            {"role": "user", "content": "give me a graduation plan for a junior computer science major"},
            {"role": "assistant", "content": MARKDOWN_PLAN_TEXT},
            {"role": "user", "content": "what year and major am i again?"},
        ]
        bad_orchestrator_json = """{
          "normalized_question": "What year and major am I again?",
          "corrected_question": "What year and major am I again?",
          "resolved_user_need": "User needs to confirm their academic year and major.",
          "domain": "general_academic_chat",
          "deliverable": "direct_answer",
          "plan_scope": "single_semester",
          "semester_horizon": "next semester",
          "use_tools": false,
          "allowed_tools": [],
          "executor_path": "conversation_plan_slice",
          "must_not_do": [],
          "success_criteria": "User confirms their academic year and major."
        }"""

        with patch.object(main_copy, "llm_classifier", FakeLLM([bad_orchestrator_json])), \
             patch.object(main_copy, "llm1", FakeLLM([""])), \
             patch.object(main_copy, "llm2", FakeLLM([""])), \
             patch.object(main_copy, "has_transcript", False), \
             patch.object(main_copy, "stored_pdf_docs", None), \
             patch.object(main_copy, "latest_student_data", None):
            response = main_copy.generate_conversational_response(
                "what year and major am i again?",
                history[:-1],
                user_id=user_id,
                user_has_transcript=False,
                debug_mode=True,
            )

        self.assertIn("junior", response.lower())
        self.assertIn("computer science", response.lower())

        trace = main_copy.get_execution_trace_for_user(user_id)
        self.assertEqual(trace["orchestrator"]["executor_path"], "conversation_profile_recall")
        self.assertEqual(trace["orchestrator"]["plan_scope"], "direct_answer")

    def test_run_three_stage_pipeline_retries_on_scope_mismatch(self):
        attempts = []
        trace = {"enabled": True, "events": [], "quality_checks": []}

        def execute_plan(plan, retry_reason=None):
            attempts.append(retry_reason)
            if len(attempts) == 1:
                return ExecutorResult(response=FULL_PLAN_TEXT, tools_used=[], metadata={})
            return ExecutorResult(
                response="Semester 3:\nBA 243 (prerequisites: None)\nTotal Credits: 15",
                tools_used=[],
                metadata={},
            )

        final_response, plan = run_three_stage_pipeline(
            user_question="just next semester only",
            conversation_history=[{"role": "assistant", "content": FULL_PLAN_TEXT}],
            has_transcript=False,
            available_tools=["conversation_history"],
            orchestrator_llm=FakeLLM([""]),
            finalizer_llm=FakeLLM(["", ""]),
            execute_plan=execute_plan,
            trace=trace,
            validation_helpers={},
        )

        self.assertEqual(plan.plan_scope, "single_semester")
        self.assertEqual(attempts, [None, "scope_mismatch_single_semester"])
        self.assertIn("Semester 3:", final_response)
        self.assertNotIn("Semester 4:", final_response)


class StructuralRoutingRegressionTests(unittest.TestCase):
    """Regression tests for the domain-driven override contract.

    These tests encode the *design* — not specific keywords — so they cover the
    whole class of "unexpected question gets silently downgraded to
    general_academic_chat with no tools" bugs.
    """

    AVAILABLE = [
        "conversation_history",
        "deterministic_planner",
        "transcript_state",
        "course_retriever",
        "majors_retriever",
        "policy_retriever",
        "prerequisite_graph",
        "web_search",
    ]

    @unittest.skipIf(main_copy is None, f"main_copy unavailable: {MAIN_COPY_IMPORT_ERROR}")
    def test_legacy_route_query_honors_llm_before_keyword_fallback(self):
        """The fallback router must not let keyword/course-code rules override
        an LLM classification. This guards the same failure class as the
        double-major issue: structured model routing owns the route, and
        deterministic rules are only a fallback.
        """
        with patch.object(main_copy, "_classify_intent_llm", return_value="policy"):
            route = main_copy.route_query("What is the CMPSC 488 late drop policy?")

        self.assertEqual(route, "policy")

    @unittest.skipIf(main_copy is None, f"main_copy unavailable: {MAIN_COPY_IMPORT_ERROR}")
    def test_legacy_route_query_uses_keywords_only_when_llm_classification_fails(self):
        with patch.object(main_copy, "_classify_intent_llm", return_value=None):
            route = main_copy.route_query("What are the prerequisites for CMPSC 488?")

        self.assertEqual(route, "course")

    def _plan(self, question, llm_json="", history=None, has_transcript=False, available=None):
        return build_orchestration_plan(
            user_question=question,
            conversation_history=history or [],
            has_transcript=has_transcript,
            available_tools=available or self.AVAILABLE,
            llm=FakeLLM([llm_json]),
        )[0]

    def test_llm_domain_major_survives_bad_executor_path(self):
        """If the LLM correctly identifies domain=major but picks a wrong path
        (e.g. conversation_plan_slice), the repair phase must keep the domain
        and align the path + tools. Previously this was silently downgraded."""
        llm_json = """{
          "normalized_question": "give me first semester courses for computer science",
          "corrected_question": "Give me first semester courses for computer science.",
          "resolved_user_need": "First-semester CS course recommendations.",
          "domain": "major",
          "deliverable": "single_semester_plan",
          "plan_scope": "single_semester",
          "semester_horizon": 1,
          "use_tools": false,
          "allowed_tools": [],
          "executor_path": "conversation_plan_slice",
          "must_not_do": [],
          "success_criteria": []
        }"""
        plan = self._plan("give me first semester courses for computer science", llm_json)
        self.assertEqual(plan.domain, "major")
        self.assertEqual(plan.executor_path, "major_lookup")
        self.assertTrue(plan.use_tools)
        self.assertIn("majors_retriever", plan.allowed_tools)
        self.assertIn("course_retriever", plan.allowed_tools)

    def test_grounded_domain_forces_tools_when_llm_says_false(self):
        """The LLM is allowed to be wrong about use_tools. Tool guardrail forces
        True for any grounded domain (course/major/policy/transcript)."""
        for domain, expected_path, expected_tool in [
            ("major", "major_lookup", "majors_retriever"),
            ("course", "course_lookup", "course_retriever"),
            ("policy", "policy_lookup", "policy_retriever"),
        ]:
            with self.subTest(domain=domain):
                llm_json = (
                    '{"normalized_question":"q","corrected_question":"Q?",'
                    '"resolved_user_need":"x","domain":"' + domain + '",'
                    '"deliverable":"' + domain + '_lookup","plan_scope":"direct_answer",'
                    '"semester_horizon":0,"use_tools":false,"allowed_tools":[],'
                    '"executor_path":"general_academic_chat",'
                    '"must_not_do":[],"success_criteria":[]}'
                )
                plan = self._plan("some unexpected " + domain + " question", llm_json)
                self.assertTrue(plan.use_tools, f"{domain} must force use_tools=True")
                self.assertEqual(plan.executor_path, expected_path)
                self.assertIn(expected_tool, plan.allowed_tools)

    def test_keyword_signal_never_downgrades_grounded_domain(self):
        """If the LLM correctly sets domain=major but no keyword in
        _is_major_question matches the question, the domain must be preserved.
        This is the core anti-regression: no more keyword coverage needed."""
        llm_json = """{
          "normalized_question": "which electives pair well with a stats minor",
          "corrected_question": "Which electives pair well with a stats minor?",
          "resolved_user_need": "Elective suggestions for stats minor.",
          "domain": "major",
          "deliverable": "major_lookup",
          "plan_scope": "direct_answer",
          "semester_horizon": 0,
          "use_tools": true,
          "allowed_tools": ["majors_retriever"],
          "executor_path": "major_lookup",
          "must_not_do": [],
          "success_criteria": []
        }"""
        plan = self._plan("Which electives pair well with a stats minor?", llm_json)
        self.assertEqual(plan.domain, "major")
        self.assertEqual(plan.executor_path, "major_lookup")
        self.assertTrue(plan.use_tools)

    def test_degraded_retrievers_filtered_but_web_search_remains(self):
        """If a retriever is unavailable (degraded chromadb), the tool list
        must still include web_search so the executor can fall back gracefully."""
        available = [t for t in self.AVAILABLE if t != "majors_retriever"]
        llm_json = """{
          "normalized_question": "cs major requirements",
          "corrected_question": "CS major requirements?",
          "resolved_user_need": "CS major reqs.",
          "domain": "major",
          "deliverable": "major_lookup",
          "plan_scope": "direct_answer",
          "semester_horizon": 0,
          "use_tools": true,
          "allowed_tools": ["majors_retriever"],
          "executor_path": "major_lookup",
          "must_not_do": [],
          "success_criteria": []
        }"""
        plan = self._plan("CS major requirements?", llm_json, available=available)
        self.assertEqual(plan.executor_path, "major_lookup")
        self.assertNotIn("majors_retriever", plan.allowed_tools)
        self.assertIn("web_search", plan.allowed_tools)
        self.assertIn("course_retriever", plan.allowed_tools)

    def test_ungrounded_specificity_validation_flags_hallucinated_codes(self):
        """Validation must flag responses with PSU course codes when the
        executor reported zero tools used. This is the finalizer-layer guard
        that catches hallucinations no matter how they got there."""
        from orchestration_pipeline import (
            ExecutorResult,
            OrchestrationPlan,
            validate_final_response,
        )

        plan = OrchestrationPlan(
            original_question="what courses for freshman cs",
            normalized_question="what courses for freshman cs",
            corrected_question="What courses for freshman CS?",
            resolved_user_need="CS first-sem courses.",
            domain="major",
            deliverable="major_lookup",
            plan_scope="direct_answer",
            semester_horizon=0,
            use_tools=True,
            allowed_tools=["majors_retriever"],
            executor_path="major_lookup",
        )
        result = ExecutorResult(
            response="Take CMPSC 131 and MATH 140.",
            tools_used=[],  # no tools used → any course code is ungrounded
            metadata={},
        )
        validation = validate_final_response(
            plan=plan,
            response_text="Take CMPSC 131 and MATH 140.",
            executor_result=result,
        )
        self.assertIn("ungrounded_specificity", validation["issues"])

    def test_ungrounded_specificity_ignored_when_tools_were_used(self):
        """If retrievers were invoked, specific course codes in the response
        are considered grounded and must not be flagged."""
        from orchestration_pipeline import (
            ExecutorResult,
            OrchestrationPlan,
            validate_final_response,
        )

        plan = OrchestrationPlan(
            original_question="what courses for freshman cs",
            normalized_question="what courses for freshman cs",
            corrected_question="What courses for freshman CS?",
            resolved_user_need="x",
            domain="major",
            deliverable="major_lookup",
            plan_scope="direct_answer",
            semester_horizon=0,
            use_tools=True,
            allowed_tools=["majors_retriever"],
            executor_path="major_lookup",
        )
        result = ExecutorResult(
            response="Take CMPSC 131 and MATH 140.",
            tools_used=["majors_retriever"],
            metadata={},
        )
        validation = validate_final_response(
            plan=plan,
            response_text="Take CMPSC 131 and MATH 140.",
            executor_result=result,
        )
        self.assertNotIn("ungrounded_specificity", validation["issues"])

    def test_course_code_regex_accepts_short_psu_codes(self):
        from orchestration_pipeline import COURSE_CODE_RE

        text = "PSU 1, ART 11, SOC 1, ENGL 15, and CMPSC 131 are valid course codes."
        self.assertEqual(
            [m.group(0).upper() for m in COURSE_CODE_RE.finditer(text)],
            ["PSU 1", "ART 11", "SOC 1", "ENGL 15", "CMPSC 131"],
        )

    def test_compact_semester_line_is_canonicalized_before_validation(self):
        from orchestration_pipeline import OrchestrationPlan, validate_final_response

        plan = OrchestrationPlan(
            original_question="give me a CS plan",
            normalized_question="give me a cs plan",
            corrected_question="Give me a Computer Science plan.",
            resolved_user_need="Plan one semester.",
            domain="major",
            deliverable="single_semester_plan",
            plan_scope="single_semester",
            semester_horizon=1,
            use_tools=True,
            allowed_tools=["majors_retriever", "course_retriever"],
            executor_path="major_lookup",
        )
        response = (
            '"Semester 1: PSU 1: First-Year Seminar Abington (prerequisites: None) '
            'CMPSC 131: Programming and Computation I (prerequisites: None) '
            'MATH 140: Calculus I (prerequisites: None) Total Credits: 7"'
        )
        result = ExecutorResult(
            response=response,
            tools_used=["majors_retriever", "course_retriever"],
            metadata={},
        )

        report = validate_final_response(plan, response, result)
        self.assertEqual(report["semester_block_count"], 1)
        self.assertNotIn("missing_inline_prereqs", report["issues"])
        self.assertNotIn("scope_mismatch_single_semester", report["issues"])

    def test_ungrounded_specificity_retry_upgrades_plan(self):
        """End-to-end: when the first attempt produces hallucinated codes with
        zero tools, the retry must re-execute with use_tools=True and the
        canonical toolset for the domain."""
        attempts = []

        def execute_plan(plan, retry_reason=None):
            attempts.append({"retry_reason": retry_reason, "use_tools": plan.use_tools,
                             "path": plan.executor_path, "tools": list(plan.allowed_tools)})
            if retry_reason is None:
                # First attempt: pretend the draft hallucinated course codes with no tools.
                return ExecutorResult(
                    response="For a CS first semester, take CMPSC 131 and MATH 140.",
                    tools_used=[],
                    metadata={},
                )
            # Retry: grounded answer with a tool used.
            return ExecutorResult(
                response="Based on the catalog: CMPSC 131, MATH 140.",
                tools_used=["majors_retriever"],
                metadata={},
            )

        llm_json = """{
          "normalized_question": "q",
          "corrected_question": "Q?",
          "resolved_user_need": "x",
          "domain": "major",
          "deliverable": "major_lookup",
          "plan_scope": "direct_answer",
          "semester_horizon": 0,
          "use_tools": true,
          "allowed_tools": ["majors_retriever", "course_retriever", "web_search"],
          "executor_path": "major_lookup",
          "must_not_do": [],
          "success_criteria": []
        }"""
        trace = {"enabled": True, "events": [], "quality_checks": []}
        _, plan = run_three_stage_pipeline(
            user_question="what should I take freshman year for CS",
            conversation_history=[],
            has_transcript=False,
            available_tools=self.AVAILABLE,
            orchestrator_llm=FakeLLM([llm_json]),
            finalizer_llm=FakeLLM(["", ""]),
            execute_plan=execute_plan,
            trace=trace,
            validation_helpers={},
        )
        self.assertEqual(len(attempts), 2)
        self.assertIsNone(attempts[0]["retry_reason"])
        self.assertEqual(attempts[1]["retry_reason"], "ungrounded_specificity")

    def test_canonical_helpers_are_exhaustive_across_domains(self):
        """Every valid domain must have a canonical executor_path, and every
        grounded domain's canonical path must have a non-empty tool list."""
        from orchestration_pipeline import (
            VALID_DOMAINS,
            _canonical_path_for_domain,
            _canonical_tools_for_path,
            _requires_grounded_answer,
        )
        available = self.AVAILABLE
        for domain in VALID_DOMAINS:
            path = _canonical_path_for_domain(domain, "direct_answer")
            self.assertTrue(path, f"{domain} has no canonical path")
            if _requires_grounded_answer(domain, "direct_answer"):
                tools = _canonical_tools_for_path(path, available)
                self.assertTrue(tools, f"{domain}/{path} has empty canonical tool set")


@unittest.skipIf(main_copy is None, f"main_copy unavailable: {MAIN_COPY_IMPORT_ERROR}")
class SemesterSpecificMajorRoutingTests(unittest.TestCase):
    """The routing contract for 'specific semester of a specific major' questions.

    Product requirement:
      - No transcript uploaded -> return the courses from the standard
        graduation plan for that semester of the major/option.
      - Transcript uploaded    -> personalized next-semester recommendation
        based on what the student still needs.

    These tests exercise the parser, dispatch branch, and end-to-end behavior
    without hitting the LLM or the real vector stores.
    """

    def test_extract_target_semester_ordinals(self):
        cases = [
            ("first semester courses for computer science",     1, "freshman",  1),
            ("1st semester of the CS major",                    1, "freshman",  1),
            ("second semester classes for business",            2, "freshman",  1),
            ("third semester of biology, b.s.",                 3, "sophomore", 1),
            ("what should I take 5th semester for data sciences", 5, "junior",  1),
            ("semester 7 for finance",                          7, "senior",   1),
        ]
        for question, sem_n, year, horizon in cases:
            with self.subTest(question=question):
                out = main_copy._extract_target_semester_request(question)
                self.assertIsNotNone(out, f"expected a target for: {question}")
                self.assertEqual(out["semester_number"], sem_n)
                self.assertEqual(out["year_level"], year)
                self.assertEqual(out["horizon"], horizon)

    def test_extract_target_semester_year_and_term(self):
        out = main_copy._extract_target_semester_request("sophomore fall plan for CS")
        self.assertEqual(out, {"semester_number": 3, "year_level": "sophomore", "horizon": 1})
        out = main_copy._extract_target_semester_request("freshman spring for biology")
        self.assertEqual(out, {"semester_number": 2, "year_level": "freshman", "horizon": 1})
        out = main_copy._extract_target_semester_request("junior year classes for finance")
        self.assertEqual(out, {"semester_number": 5, "year_level": "junior", "horizon": 2})

    def test_extract_target_semester_none_for_generic_major_q(self):
        self.assertIsNone(main_copy._extract_target_semester_request("what are the CS major requirements"))
        self.assertIsNone(main_copy._extract_target_semester_request("give me a graduation plan for CS"))
        self.assertIsNone(main_copy._extract_target_semester_request(""))

    def test_extract_semester_block_by_number_picks_correct_block(self):
        block = main_copy._extract_semester_block_by_number(FULL_PLAN_TEXT, 4)
        self.assertIsNotNone(block)
        self.assertIn("Semester 4:", block)
        self.assertIn("Total Credits: 15", block)
        self.assertNotIn("Semester 3:", block)

    def test_no_transcript_returns_standard_plan_block(self):
        """Without a transcript, the helper should call the deterministic
        planner and return exactly the requested semester's block."""
        fake_plan = (
            "Semester 1:\n"
            "CMPSC 131 (prerequisites: None)\n"
            "MATH 140 (prerequisites: None)\n"
            "ENGL 15 (prerequisites: None)\n"
            "Total Credits: 15\n"
            "---\n"
            "Semester 2:\n"
            "CMPSC 132 (prerequisites: CMPSC 131)\n"
            "MATH 141 (prerequisites: MATH 140)\n"
            "Total Credits: 15"
        )
        with patch.object(main_copy, "generate_plan_from_major_and_year", return_value=fake_plan) as planner, \
             patch.object(
                 main_copy,
                 "_extract_major_year_from_request_context",
                 return_value=("Computer Science, B.S. (Abington)", None),
             ):
            response, tools_used, metadata = main_copy._build_semester_specific_major_draft(
                user_question="first semester courses for computer science",
                target={"semester_number": 1, "year_level": "freshman", "horizon": 1},
                conversation_history=[],
                has_transcript_for_plan=False,
                trace=None,
                allowed_tools=None,
            )
        planner.assert_called_once()
        self.assertIn("Semester 1:", response)
        self.assertIn("CMPSC 131", response)
        self.assertNotIn("Semester 2:", response)
        self.assertIn("Total Credits: 15", response)
        self.assertEqual(tools_used, ["deterministic_planner"])
        self.assertEqual(metadata["source"], "major_year_semester_plan")
        self.assertEqual(metadata["target_semester"], 1)

    def test_transcript_path_uses_personalized_aggregator(self):
        """With a transcript, the helper must delegate to the personalized
        aggregator (not the generic major+year planner)."""
        personalized = (
            "Semester 5:\n"
            "CMPSC 360 (prerequisites: None)\n"
            "STAT 318 (prerequisites: None)\n"
            "Total Credits: 15"
        )
        with patch.object(main_copy, "aggregate_and_generate_response", return_value=personalized) as agg, \
             patch.object(main_copy, "generate_plan_from_major_and_year") as planner:
            response, tools_used, metadata = main_copy._build_semester_specific_major_draft(
                user_question="what should I take next semester",
                target={"semester_number": 5, "year_level": "junior", "horizon": 1},
                conversation_history=[],
                has_transcript_for_plan=True,
                trace=None,
                allowed_tools=None,
            )
        agg.assert_called_once()
        planner.assert_not_called()
        self.assertIn("Semester 5:", response)
        self.assertIn("transcript_state", tools_used)
        self.assertIn("deterministic_planner", tools_used)
        self.assertEqual(metadata["source"], "personalized_semester_plan")

    def test_missing_major_returns_empty_for_graceful_fallback(self):
        """When we can't extract the major and the student has no transcript,
        return empty so the dispatch can fall through to generic major_lookup."""
        with patch.object(
            main_copy,
            "_extract_major_year_from_request_context",
            return_value=(None, None),
        ):
            response, tools_used, metadata = main_copy._build_semester_specific_major_draft(
                user_question="what should I take first semester",
                target={"semester_number": 1, "year_level": "freshman", "horizon": 1},
                conversation_history=[],
                has_transcript_for_plan=False,
                trace=None,
                allowed_tools=None,
            )
        self.assertEqual(response, "")
        self.assertEqual(tools_used, [])
        self.assertEqual(metadata["source"], "semester_specific_missing_major")

    def test_major_lookup_tools_include_deterministic_planner(self):
        """Without this, tools_used from the deterministic planner would be
        flagged as disallowed by validate_final_response."""
        from orchestration_pipeline import _TOOLS_BY_PATH
        self.assertIn("deterministic_planner", _TOOLS_BY_PATH["major_lookup"])
        self.assertIn("transcript_state", _TOOLS_BY_PATH["major_lookup"])


class PersonalityRegressionTests(unittest.TestCase):
    """Locks in the advisor persona contract.

    These tests exist so that a future edit can't quietly drop the persona,
    the 'Things to consider next' enrichment, the soft-ask behavior, or the
    semester-block preservation rule. They check both the structural shape
    (constants, prompt strings) and a representative end-to-end behavior
    (finalizer preserves semester blocks; soft-ask drafts are warm).
    """

    def test_advisor_persona_constant_defines_all_four_traits(self):
        """The persona is the single source of truth for the agent's voice.
        It must encode: maximal helpfulness, encouragement, soft-ask on
        missing info, and grounded behavior. Removing any of these is a
        product-level regression, not a stylistic edit."""
        from orchestration_pipeline import ADVISOR_PERSONA

        self.assertIsInstance(ADVISOR_PERSONA, str)
        self.assertGreater(len(ADVISOR_PERSONA), 200, "persona must be substantive")

        lowered = ADVISOR_PERSONA.lower()
        self.assertIn("nit hamirpur", lowered, "persona must name the school")
        self.assertIn("encourag", lowered, "persona must mention encouragement")
        self.assertIn("things to consider next", lowered, "persona must define enrichment section")
        self.assertIn("soft-ask", lowered, "persona must define soft-ask behavior")
        self.assertIn("grounded", lowered, "persona must require grounded answers")
        self.assertIn("semester n", lowered, "persona must require semester-block preservation")

    def test_finalizer_prompt_uses_persona_and_required_shape(self):
        """The finalizer is the last layer the user sees, so it must apply
        the persona and produce the required response shape (warm answer +
        'Things to consider next' + closing nudge/clarifying question).
        This is a source-level check because the prompt itself is the
        contract — we can't reliably observe it through the LLM's output."""
        import inspect
        import orchestration_pipeline

        source = inspect.getsource(orchestration_pipeline.run_finalizer_stage)
        self.assertIn("ADVISOR_PERSONA", source, "finalizer must inject the persona")
        self.assertIn("Things to consider next", source)
        self.assertIn("VERBATIM", source, "finalizer must enforce semester-block preservation")
        self.assertIn("clarifying question", source, "finalizer must allow soft-ask close")

    def test_finalizer_preserves_semester_blocks_verbatim(self):
        """If the executor draft contains a 'Semester N: ... Total Credits: N'
        block, the finalizer must not mangle it. Downstream UI parsing and
        plan-slicing rely on the exact shape. We simulate a finalizer LLM
        that echoes the draft and assert the block survives end to end."""
        from orchestration_pipeline import (
            ExecutorResult,
            OrchestrationPlan,
            run_finalizer_stage,
        )

        semester_block = (
            "Semester 1:\n"
            "CMPSC 121: Introduction to Programming Techniques (3 credits)\n"
            "MATH 140: Calculus with Analytic Geometry I (4 credits)\n"
            "ENGL 015: Rhetoric and Composition (3 credits)\n"
            "PSYCH 100: Introductory Psychology (3 credits)\n"
            "KINES 084: First-Year Seminar (2 credits)\n"
            "Total Credits: 15"
        )
        draft = f"Here is a recommended first-semester plan for Computer Science.\n\n{semester_block}"

        plan = OrchestrationPlan(
            original_question="first semester courses for computer science",
            normalized_question="first semester courses for computer science",
            corrected_question="first semester courses for computer science",
            resolved_user_need="recommend first-semester CS courses",
            domain="major",
            deliverable="major_lookup",
            plan_scope="direct_answer",
            semester_horizon=0,
            use_tools=True,
            allowed_tools=["majors_retriever"],
            executor_path="major_lookup",
            must_not_do=[],
            success_criteria=["Concrete 15-credit semester block returned."],
        )
        executor_result = ExecutorResult(
            response=draft,
            tools_used=["majors_retriever"],
            metadata={"source": "major_lookup_rag"},
            notes=[],
        )

        # FakeLLM that echoes the draft verbatim, wrapped in the JSON shape
        # the finalizer expects, plus warm prose + enrichment.
        echo_payload = (
            "{\n"
            '  "final_response": "Great question! Here is a recommended first-semester plan for Computer Science.\\n\\n'
            + semester_block.replace("\n", "\\n")
            + '\\n\\nThings to consider next:\\n- Confirm placement test results.\\n- Meet your advisor.",\n'
            '  "finalizer_notes": "ok"\n'
            "}"
        )
        finalizer_llm = FakeLLM([echo_payload])

        final_response, _meta = run_finalizer_stage(
            plan=plan,
            executor_result=executor_result,
            conversation_history=[],
            llm=finalizer_llm,
        )

        self.assertIn(semester_block, final_response, "Semester block must survive finalizer verbatim")
        self.assertIn("Things to consider next", final_response)

    @unittest.skipIf(main_copy is None, f"main_copy unavailable: {MAIN_COPY_IMPORT_ERROR}")
    def test_user_facing_executor_drafters_inject_persona(self):
        """The major/policy/general-chat drafters compose their own LLM prose
        and therefore must arrive with the right voice — relying on the
        finalizer alone leaves a window where the executor's own LLM produces
        cold/terse text that the finalizer then has to repair against."""
        import inspect

        for fn_name in (
            "_build_major_lookup_draft",
            "_build_policy_lookup_draft",
            "_build_general_academic_chat_draft",
            "_run_executor_stage",
        ):
            fn = getattr(main_copy, fn_name, None)
            self.assertIsNotNone(fn, f"main_copy is missing {fn_name}")
            source = inspect.getsource(fn)
            self.assertIn(
                "ADVISOR_PERSONA",
                source,
                f"{fn_name} must inject ADVISOR_PERSONA into its system prompt",
            )

    @unittest.skipIf(main_copy is None, f"main_copy unavailable: {MAIN_COPY_IMPORT_ERROR}")
    def test_missing_info_uses_warm_soft_ask(self):
        """When critical info is missing (here: major + year for a plan
        request), the response must be a warm soft-ask, not a one-liner.
        It should: (a) acknowledge the request, (b) ask a specific
        clarifying question, (c) hint at how the answer will improve once
        the info arrives. This protects the soft-ask behavior contract."""
        from orchestration_pipeline import OrchestrationPlan

        plan = OrchestrationPlan(
            original_question="give me a plan",
            normalized_question="give me a plan",
            corrected_question="give me a plan",
            resolved_user_need="generate a semester plan",
            domain="transcript",
            deliverable="full_graduation_plan",
            plan_scope="full_graduation",
            semester_horizon=8,
            use_tools=True,
            allowed_tools=["deterministic_planner", "transcript_state"],
            executor_path="deterministic_plan",
            must_not_do=[],
            success_criteria=["Return a graduation plan."],
        )

        with patch.object(main_copy, "_extract_major_year_from_request_context", return_value=(None, None)), \
             patch.object(main_copy, "aggregate_and_generate_response", return_value=""):
            response, tools_used, metadata = main_copy._build_transcript_plan_draft(
                plan=plan,
                user_question="give me a plan",
                conversation_history=[],
                has_transcript_for_plan=False,
            )

        self.assertEqual(metadata.get("source"), "missing_major_year")
        self.assertEqual(tools_used, [])
        # Must NOT be the old terse one-liner.
        self.assertNotEqual(
            response.strip(),
            "I need your major and year level to build that plan. "
            "For example: 'Business sophomore' or 'Computer Science junior'.",
        )
        # Must read like a warm soft-ask.
        lowered = response.lower()
        self.assertIn("major", lowered)
        self.assertIn("year", lowered)
        self.assertIn("?", response, "soft-ask must end with a clarifying question")
        self.assertGreater(len(response), 200, "soft-ask must be a real explanation, not a one-liner")
        self.assertTrue(
            any(token in lowered for token in ("happy", "glad", "love to", "want to")),
            "soft-ask must open in a warm/encouraging voice",
        )


class PromptEngineeringRegressionTests(unittest.TestCase):
    """Locks in the prompt-engineering contract for model-facing prompts."""

    def test_private_reasoning_guidance_is_safe_and_reusable(self):
        from orchestration_pipeline import PRIVATE_REASONING_GUIDANCE

        lowered = PRIVATE_REASONING_GUIDANCE.lower()
        self.assertIn("reason privately", lowered)
        self.assertIn("do not reveal private chain-of-thought", lowered)
        self.assertIn("evidence summary", lowered)

    def test_advisor_persona_names_leo_and_blocks_foul_language(self):
        from orchestration_pipeline import ADVISOR_PERSONA, AGENT_DISPLAY_NAME, LANGUAGE_SAFETY_POLICY

        self.assertEqual(AGENT_DISPLAY_NAME, "Leo")
        self.assertIn("Leo", ADVISOR_PERSONA)
        lowered_policy = LANGUAGE_SAFETY_POLICY.lower()
        self.assertIn("never write, quote, list, transform, repeat", lowered_policy)
        self.assertIn("foul language", lowered_policy)
        self.assertIn("endorse", lowered_policy)

    def test_public_response_sanitizer_removes_foul_language(self):
        from orchestration_pipeline import sanitize_public_response

        blocked_word = "s" + "h" + "i" + "t"
        clean = sanitize_public_response(f"That is {blocked_word}.")

        self.assertNotIn(blocked_word, clean.lower())
        self.assertIn("[removed]", clean)

    @unittest.skipIf(main_copy is None, f"main_copy unavailable: {MAIN_COPY_IMPORT_ERROR}")
    def test_foul_language_requests_are_refused_before_model_call(self):
        self.assertTrue(main_copy._is_foul_language_request("please list curse words"))
        refusal = main_copy._foul_language_refusal()
        self.assertIn("can't help", refusal)
        self.assertIn("Leo", refusal)
        self.assertIn("endorse foul language", refusal)

        response = main_copy.generate_conversational_response(
            "please list curse words",
            [],
            user_id="clean-language-test",
        )
        self.assertEqual(response, refusal)

    def test_model_system_prompts_include_output_contracts_and_examples(self):
        import inspect
        import orchestration_pipeline

        prompt_sources = [
            inspect.getsource(orchestration_pipeline.build_orchestration_plan),
            inspect.getsource(orchestration_pipeline.run_finalizer_stage),
        ]
        if main_copy is not None:
            prompt_sources.extend([
                inspect.getsource(main_copy._classify_intent_llm),
                inspect.getsource(main_copy._build_reasoning_plan),
                inspect.getsource(main_copy._build_policy_lookup_draft),
                inspect.getsource(main_copy._build_major_lookup_draft),
                inspect.getsource(main_copy._build_general_academic_chat_draft),
                inspect.getsource(main_copy._run_executor_stage),
            ])

        for source in prompt_sources:
            self.assertIn("PRIVATE_REASONING_GUIDANCE", source)
            self.assertIn("example", source.lower())
            self.assertTrue(
                "Output contract" in source
                or "Required response shape" in source
                or "Required semester block shape" in source,
                "prompt should define a concrete output contract",
            )


class RetrievalPlanContractTests(unittest.TestCase):
    """Locks in the planner-driven retrieval contract.

    These tests exist so a future edit can't reintroduce executor-side
    keyword conditionals for tool selection. The contract is:

      - The orchestrator emits ``retrieval_plan`` and ``synthesis_brief``.
      - For course-recommendation deliverables, the override layer
        guarantees a ``prerequisite_graph`` step is present.
      - The executor mechanically dispatches every step through
        ``EXECUTOR_TOOLS`` and never picks tools on its own.
      - The synthesis prompt arrives with a labeled bundle, one section
        per tool that returned content.
      - A deterministic prerequisite-satisfaction validator catches
        scheduling violations (the CMPSC 131/132 regression) and
        produces a ``prerequisite_violation`` retry signal.
    """

    AVAILABLE = [
        "conversation_history",
        "deterministic_planner",
        "transcript_state",
        "course_retriever",
        "majors_retriever",
        "policy_retriever",
        "prerequisite_graph",
        "web_search",
    ]

    def _plan(self, question, llm_json="", history=None, has_transcript=False, available=None):
        return build_orchestration_plan(
            user_question=question,
            conversation_history=history or [],
            has_transcript=has_transcript,
            available_tools=available or self.AVAILABLE,
            llm=FakeLLM([llm_json]),
        )[0]

    def test_override_injects_canonical_retrieval_plan_for_course_recommendation(self):
        """If the planner emits a major_lookup deliverable but no retrieval_plan,
        the override layer MUST fill in canonical steps that include
        ``prerequisite_graph`` — otherwise the executor synthesizes without
        prereq edges and re-ships the CMPSC 131/132 bug."""
        llm_json = """{
          "normalized_question": "first semester courses for computer science",
          "corrected_question": "First semester courses for computer science?",
          "resolved_user_need": "First-semester CS course recommendations.",
          "domain": "major",
          "deliverable": "single_semester_plan",
          "plan_scope": "single_semester",
          "semester_horizon": 1,
          "use_tools": true,
          "allowed_tools": ["majors_retriever", "course_retriever", "prerequisite_graph"],
          "executor_path": "major_lookup",
          "must_not_do": [],
          "success_criteria": []
        }"""
        plan = self._plan("first semester courses for computer science", llm_json)
        self.assertTrue(plan.retrieval_plan, "override must populate empty retrieval_plan")
        tools_in_plan = [step["tool"] for step in plan.retrieval_plan]
        self.assertIn("prerequisite_graph", tools_in_plan,
                      "course-recommendation deliverable MUST include prerequisite_graph")
        self.assertIn("majors_retriever", tools_in_plan)

    def test_planner_emitted_retrieval_plan_is_preserved_when_valid(self):
        """When the planner produces a complete retrieval_plan (with
        prerequisite_graph already present), the override layer must NOT
        clobber the planner's queries — order and queries are the planner's
        deliverable, the override is just a safety net."""
        llm_json = """{
          "normalized_question": "first semester courses for computer science",
          "corrected_question": "First semester courses for computer science?",
          "resolved_user_need": "First-semester CS course recommendations.",
          "domain": "major",
          "deliverable": "single_semester_plan",
          "plan_scope": "single_semester",
          "semester_horizon": 1,
          "use_tools": true,
          "allowed_tools": ["majors_retriever", "course_retriever", "prerequisite_graph"],
          "executor_path": "major_lookup",
          "must_not_do": [],
          "success_criteria": [],
          "retrieval_plan": [
            {"tool": "majors_retriever", "query": "Computer Science B.S. Abington requirements", "reason": "Get major requirements."},
            {"tool": "course_retriever", "query": "CMPSC 131 MATH 140 ENGL 15 candidate first-semester courses", "reason": "Get titles and credits."},
            {"tool": "prerequisite_graph", "query": "CMPSC 131,CMPSC 132,MATH 140", "reason": "Avoid scheduling violations."}
          ],
          "synthesis_brief": "Build a 15-credit Semester 1 schedule for CS using only courses with no enrollment prereqs."
        }"""
        plan = self._plan("first semester courses for computer science", llm_json)
        tools_in_plan = [step["tool"] for step in plan.retrieval_plan]
        self.assertEqual(
            tools_in_plan,
            ["majors_retriever", "course_retriever", "prerequisite_graph"],
            "planner's order must survive untouched",
        )
        majors_step = plan.retrieval_plan[0]
        self.assertEqual(majors_step["query"], "Computer Science B.S. Abington requirements")
        self.assertIn("Build a 15-credit", plan.synthesis_brief)

    @unittest.skipIf(main_copy is None, f"main_copy unavailable: {MAIN_COPY_IMPORT_ERROR}")
    def test_executor_dispatches_every_retrieval_plan_step(self):
        """``_execute_retrieval_plan`` is the executor's only path to data.
        It must call every tool the planner listed, with the planner's
        exact query, and only those tools. No keyword logic, no implicit
        fan-out."""
        from orchestration_pipeline import OrchestrationPlan

        plan = OrchestrationPlan(
            original_question="first semester CS",
            normalized_question="first semester CS",
            corrected_question="first semester CS",
            resolved_user_need="x",
            domain="major",
            deliverable="single_semester_plan",
            plan_scope="single_semester",
            semester_horizon=1,
            use_tools=True,
            allowed_tools=["majors_retriever", "course_retriever", "prerequisite_graph"],
            executor_path="major_lookup",
            retrieval_plan=[
                {"tool": "majors_retriever", "query": "MAJ_Q", "reason": "r1"},
                {"tool": "course_retriever", "query": "CRS_Q", "reason": "r2"},
                {"tool": "prerequisite_graph", "query": "PRQ_Q", "reason": "r3"},
            ],
            synthesis_brief="brief",
        )

        recorded = []

        def fake_tool(name, payload):
            def _call(query, ctx):
                recorded.append((name, query))
                return payload
            return _call

        fake_registry = {
            "majors_retriever":   fake_tool("majors_retriever", "MAJORS_RESULT"),
            "course_retriever":   fake_tool("course_retriever", "COURSES_RESULT"),
            "prerequisite_graph": fake_tool("prerequisite_graph", "PREREQ_RESULT"),
        }
        with patch.dict(main_copy.EXECUTOR_TOOLS, fake_registry, clear=True):
            bundle, tools_used = main_copy._execute_retrieval_plan(plan)

        self.assertEqual(
            recorded,
            [("majors_retriever", "MAJ_Q"),
             ("course_retriever", "CRS_Q"),
             ("prerequisite_graph", "PRQ_Q")],
            "every step must dispatch in order with the planner's query",
        )
        self.assertEqual(tools_used, ["majors_retriever", "course_retriever", "prerequisite_graph"])
        self.assertEqual(set(bundle.keys()), {"majors_retriever", "course_retriever", "prerequisite_graph"})
        self.assertEqual(bundle["prerequisite_graph"]["results"], "PREREQ_RESULT")

    @unittest.skipIf(main_copy is None, f"main_copy unavailable: {MAIN_COPY_IMPORT_ERROR}")
    def test_synthesis_prompt_contains_every_retrieved_section(self):
        """The synthesis LLM must see a labeled bundle — one ### TOOL header
        per tool that returned content — so it can attribute claims to
        specific tools (and so the prereq directive is enforceable)."""
        from orchestration_pipeline import OrchestrationPlan

        plan = OrchestrationPlan(
            original_question="first semester CS",
            normalized_question="first semester CS",
            corrected_question="first semester CS",
            resolved_user_need="x",
            domain="major",
            deliverable="single_semester_plan",
            plan_scope="single_semester",
            semester_horizon=1,
            use_tools=True,
            allowed_tools=["majors_retriever", "course_retriever", "prerequisite_graph"],
            executor_path="major_lookup",
            retrieval_plan=[
                {"tool": "majors_retriever", "query": "MAJ_Q", "reason": "r1"},
                {"tool": "course_retriever", "query": "CRS_Q", "reason": "r2"},
                {"tool": "prerequisite_graph", "query": "PRQ_Q", "reason": "r3"},
            ],
            synthesis_brief="Build a 15-credit Semester 1 schedule for CS.",
        )

        fake_registry = {
            "majors_retriever":   lambda q, ctx: "MAJORS_RESULT_TEXT",
            "course_retriever":   lambda q, ctx: "COURSES_RESULT_TEXT",
            "prerequisite_graph": lambda q, ctx: "PREREQ_RESULT_TEXT",
        }
        fake_llm = FakeLLM(["Semester 1:\nCMPSC 131\nTotal Credits: 3"])

        with patch.dict(main_copy.EXECUTOR_TOOLS, fake_registry, clear=True), \
             patch.object(main_copy, "llm2", fake_llm):
            response, tools_used, metadata = main_copy._build_major_lookup_draft(
                user_question="first semester CS",
                trace=None,
                allowed_tools={"majors_retriever", "course_retriever", "prerequisite_graph"},
                plan=plan,
            )

        self.assertEqual(len(fake_llm.calls), 1, "synthesizer must run exactly once")
        # The HumanMessage is the second message in the call.
        human_msg = fake_llm.calls[0][1]
        prompt_text = getattr(human_msg, "content", "")
        for header in ("### majors_retriever", "### course_retriever", "### prerequisite_graph"):
            self.assertIn(header, prompt_text, f"prompt must contain {header}")
        for payload in ("MAJORS_RESULT_TEXT", "COURSES_RESULT_TEXT", "PREREQ_RESULT_TEXT"):
            self.assertIn(payload, prompt_text)
        self.assertIn("Build a 15-credit Semester 1 schedule for CS.", prompt_text,
                      "synthesis_brief must reach the prompt")
        self.assertIn("prereq_graph_text", metadata,
                      "metadata must carry full prereq text for the validator")
        self.assertEqual(set(tools_used), {"majors_retriever", "course_retriever", "prerequisite_graph"})

    def test_prereq_validator_flags_cmpsc_131_and_132_in_same_semester(self):
        """Exact regression for the bug the user pasted: CMPSC 131 and
        CMPSC 132 placed in Semester 1 together, with the bundle stating
        CMPSC 132's enrollment prereq is CMPSC 131. The validator MUST
        catch this and ``validate_final_response`` MUST surface
        ``prerequisite_violation``."""
        from orchestration_pipeline import (
            OrchestrationPlan,
            _validate_prerequisite_satisfaction,
            validate_final_response,
        )

        response = (
            "Here is your first semester plan.\n\n"
            "Semester 1:\n"
            "CMPSC 131: Programming and Computation I\n"
            "CMPSC 132: Programming and Computation II\n"
            "MATH 140: Calculus I\n"
            "Total Credits: 9"
        )
        prereq_text = (
            "Prerequisites for CMPSC 131:\n"
            "- Enforced prerequisite at enrollment: None\n"
            "- Enforced concurrent at enrollment: None\n\n"
            "Prerequisites for CMPSC 132:\n"
            "- Enforced prerequisite at enrollment: CMPSC 131\n"
            "- Enforced concurrent at enrollment: None\n\n"
            "Prerequisites for MATH 140:\n"
            "- Enforced prerequisite at enrollment: None\n"
            "- Enforced concurrent at enrollment: None\n"
        )

        violation = _validate_prerequisite_satisfaction(response, prereq_text)
        self.assertIsNotNone(violation, "validator must catch CMPSC 132 placed before its prereq")
        self.assertIn("CMPSC 132", violation)
        self.assertIn("CMPSC 131", violation)

        plan = OrchestrationPlan(
            original_question="first semester courses for CS",
            normalized_question="first semester courses for CS",
            corrected_question="first semester courses for CS",
            resolved_user_need="x",
            domain="major",
            deliverable="single_semester_plan",
            plan_scope="single_semester",
            semester_horizon=1,
            use_tools=True,
            allowed_tools=["majors_retriever", "course_retriever", "prerequisite_graph"],
            executor_path="major_lookup",
        )
        executor_result = ExecutorResult(
            response=response,
            tools_used=["majors_retriever", "course_retriever", "prerequisite_graph"],
            metadata={"prereq_graph_text": prereq_text},
            notes=[],
        )
        report = validate_final_response(plan, response, executor_result)
        self.assertFalse(report["passed"])
        self.assertIn("prerequisite_violation", report["issues"])
        self.assertIn("prereq_violation_detail", report)

    def test_prereq_validator_passes_when_prereq_in_earlier_semester(self):
        """Same prereq bundle, valid layout: CMPSC 131 in Semester 1,
        CMPSC 132 in Semester 2. The validator must accept this and
        ``validate_final_response`` must not flag a prereq violation."""
        from orchestration_pipeline import (
            OrchestrationPlan,
            _validate_prerequisite_satisfaction,
            validate_final_response,
        )

        response = (
            "Two-semester plan:\n\n"
            "Semester 1:\n"
            "CMPSC 131: Programming and Computation I\n"
            "MATH 140: Calculus I\n"
            "Total Credits: 6\n\n"
            "Semester 2:\n"
            "CMPSC 132: Programming and Computation II\n"
            "MATH 141: Calculus II\n"
            "Total Credits: 6"
        )
        prereq_text = (
            "Prerequisites for CMPSC 131:\n"
            "- Enforced prerequisite at enrollment: None\n"
            "- Enforced concurrent at enrollment: None\n\n"
            "Prerequisites for CMPSC 132:\n"
            "- Enforced prerequisite at enrollment: CMPSC 131\n"
            "- Enforced concurrent at enrollment: None\n\n"
            "Prerequisites for MATH 140:\n"
            "- Enforced prerequisite at enrollment: None\n"
            "- Enforced concurrent at enrollment: None\n\n"
            "Prerequisites for MATH 141:\n"
            "- Enforced prerequisite at enrollment: MATH 140\n"
            "- Enforced concurrent at enrollment: None\n"
        )

        violation = _validate_prerequisite_satisfaction(response, prereq_text)
        self.assertIsNone(violation, "valid layout must produce no violation")

        plan = OrchestrationPlan(
            original_question="two-semester CS plan",
            normalized_question="two-semester CS plan",
            corrected_question="two-semester CS plan",
            resolved_user_need="x",
            domain="major",
            deliverable="multi_semester_plan",
            plan_scope="multi_semester",
            semester_horizon=2,
            use_tools=True,
            allowed_tools=["majors_retriever", "course_retriever", "prerequisite_graph"],
            executor_path="major_lookup",
        )
        executor_result = ExecutorResult(
            response=response,
            tools_used=["majors_retriever", "course_retriever", "prerequisite_graph"],
            metadata={"prereq_graph_text": prereq_text},
            notes=[],
        )
        report = validate_final_response(plan, response, executor_result)
        self.assertNotIn("prerequisite_violation", report["issues"])


class SpecificityContractTests(unittest.TestCase):
    """Locks in the universal recommendation-specificity contract.

    The product rule is structural: the student must be able to act on every
    line in a semester block. A line is acceptable when it contains a
    specific course code, OR a categorized placeholder (gen-ed tag like GH/GA
    or a category word like humanities/social science), OR an explicit
    advisor/catalog deferral. Anything else (bare 'Elective', 'Gen Ed',
    'Option Course', etc.) is unactionable and must be re-tried.

    These tests are deliberately major-agnostic and gen-ed-system-agnostic.
    Adding a new bucket to the catalog must not require changing the
    validator — only updating the taxonomy data.
    """

    def _plan(self, deliverable="single_semester_plan", scope="single_semester"):
        from orchestration_pipeline import OrchestrationPlan
        return OrchestrationPlan(
            original_question="first semester courses",
            normalized_question="first semester courses",
            corrected_question="first semester courses",
            resolved_user_need="x",
            domain="major",
            deliverable=deliverable,
            plan_scope=scope,
            semester_horizon=1,
            use_tools=True,
            allowed_tools=["majors_retriever", "course_retriever", "prerequisite_graph"],
            executor_path="major_lookup",
        )

    def test_bare_elective_line_fails_specificity(self):
        """The exact bug the user reported: 'Option Elective (3 credits)'
        with no category, no code, and no deferral. Must be flagged."""
        from orchestration_pipeline import _validate_recommendation_specificity
        response = (
            "Semester 1:\n"
            "BIOL 110: Biology Concepts (3 credits)\n"
            "MATH 140: Calculus I (4 credits)\n"
            "Option Elective (3 credits)\n"
            "Total Credits: 10"
        )
        violation = _validate_recommendation_specificity(response)
        self.assertIsNotNone(violation, "bare 'Option Elective' must be flagged")
        self.assertIn("Option Elective", violation)
        self.assertIn("Semester 1", violation)

    def test_bare_gen_ed_line_fails_specificity(self):
        """Bare 'Gen Ed (3 credits)' is the same class of bug as Elective.
        The validator must not require keyword routing — the trigger word
        is enough to demand a category descriptor."""
        from orchestration_pipeline import _validate_recommendation_specificity
        response = (
            "Semester 2:\n"
            "BIOL 220: Genetics (3 credits)\n"
            "Gen Ed (3 credits)\n"
            "Total Credits: 6"
        )
        violation = _validate_recommendation_specificity(response)
        self.assertIsNotNone(violation)
        self.assertIn("Gen Ed", violation)

    def test_categorized_placeholder_passes(self):
        """A placeholder with a recognized taxonomy descriptor is actionable
        because the student can pick from a clear bucket. Both gen-ed tags
        ('GH') and category words ('humanities') must satisfy the rule."""
        from orchestration_pipeline import _validate_recommendation_specificity
        responses = [
            (
                "Semester 1:\n"
                "BIOL 110: Biology (3 credits)\n"
                "GH Humanities elective (3 credits)\n"
                "Total Credits: 6"
            ),
            (
                "Semester 1:\n"
                "BIOL 110: Biology (3 credits)\n"
                "Social science elective — examples: PSYCH 100, SOC 001 (3 credits)\n"
                "Total Credits: 6"
            ),
            (
                "Semester 1:\n"
                "BIOL 110: Biology (3 credits)\n"
                "GWS writing/speaking course (3 credits)\n"
                "Total Credits: 6"
            ),
        ]
        for response in responses:
            with self.subTest(response=response[:80]):
                self.assertIsNone(
                    _validate_recommendation_specificity(response),
                    f"Categorized placeholder must pass:\n{response}",
                )

    def test_explicit_advisor_deferral_passes(self):
        """When the bundle has no eligible courses, the synthesizer is
        allowed to honestly say so. The student still knows what to do."""
        from orchestration_pipeline import _validate_recommendation_specificity
        response = (
            "Semester 1:\n"
            "BIOL 110: Biology (3 credits)\n"
            "Open elective slot — please confirm with your advisor or check the gen-ed catalog (3 credits)\n"
            "Total Credits: 6"
        )
        self.assertIsNone(_validate_recommendation_specificity(response))

    def test_specific_course_codes_alone_pass(self):
        """A response composed only of real course codes is the strongest
        form of specificity and must always pass."""
        from orchestration_pipeline import _validate_recommendation_specificity
        response = (
            "Semester 1:\n"
            "BIOL 110: Biology Concepts (3 credits)\n"
            "MATH 140: Calculus I (4 credits)\n"
            "ENGL 015: Rhetoric and Composition (3 credits)\n"
            "PSYCH 100: Introductory Psychology (3 credits)\n"
            "KINES 084: First-Year Seminar (2 credits)\n"
            "Total Credits: 15"
        )
        self.assertIsNone(_validate_recommendation_specificity(response))

    def test_validate_final_response_surfaces_under_specified_recommendation(self):
        """End-to-end: when a course-recommendation response contains a
        bare placeholder, validate_final_response must surface
        ``under_specified_recommendation`` in issues so the executor's
        retry path can fix it. This is the contract the synthesizer's
        retry directive depends on."""
        from orchestration_pipeline import validate_final_response

        response = (
            "Semester 1:\n"
            "BIOL 110: Biology Concepts (3 credits)\n"
            "MATH 140: Calculus I (4 credits)\n"
            "Elective (3 credits)\n"
            "Total Credits: 10"
        )
        executor_result = ExecutorResult(
            response=response,
            tools_used=["majors_retriever", "course_retriever"],
            metadata={},
            notes=[],
        )
        report = validate_final_response(self._plan(), response, executor_result)
        self.assertFalse(report["passed"])
        self.assertIn("under_specified_recommendation", report["issues"])
        self.assertIn("specificity_violation_detail", report)

    def test_validator_skips_non_recommendation_deliverables(self):
        """The specificity rule must not fire for non-course-recommendation
        deliverables (a policy_lookup answering 'what is the elective policy'
        legitimately uses the word 'elective' in prose)."""
        from orchestration_pipeline import OrchestrationPlan, validate_final_response

        plan = OrchestrationPlan(
            original_question="What is the policy for choosing electives?",
            normalized_question="What is the policy for choosing electives?",
            corrected_question="What is the policy for choosing electives?",
            resolved_user_need="x",
            domain="policy",
            deliverable="policy_lookup",
            plan_scope="direct_answer",
            semester_horizon=0,
            use_tools=True,
            allowed_tools=["policy_retriever"],
            executor_path="policy_lookup",
        )
        # Note: NO semester block — just prose. Even if there were an
        # 'elective' mention, the validator only walks semester blocks.
        response = (
            "You can choose any elective from your major's option list. "
            "Talk to your advisor for guidance."
        )
        executor_result = ExecutorResult(
            response=response,
            tools_used=["policy_retriever"],
            metadata={},
            notes=[],
        )
        report = validate_final_response(plan, response, executor_result)
        self.assertNotIn("under_specified_recommendation", report["issues"])


class InlinePrereqContractTests(unittest.TestCase):
    """Locks in the universal "every recommended course shows its prereqs
    inline" contract.

    The product rule is: when the system tells a student to take a course,
    the prereqs must be on the same line so the student knows what is
    required to enroll. The check is purely structural — does every line
    that contains a course code also contain a ``(prerequisites: ...)``
    parenthetical? — so it works for any major, any catalog year, and any
    new course added to the bundle without changing the validator.

    Lines that don't contain a course code (gen-ed placeholders, deferral
    lines, prose, headers, totals) are exempt because there is no specific
    course whose prereqs would even apply.
    """

    def _plan(self, deliverable="single_semester_plan", scope="single_semester"):
        from orchestration_pipeline import OrchestrationPlan
        return OrchestrationPlan(
            original_question="first semester courses for biology and psychology",
            normalized_question="first semester courses",
            corrected_question="first semester courses",
            resolved_user_need="x",
            domain="major",
            deliverable=deliverable,
            plan_scope=scope,
            semester_horizon=1,
            use_tools=True,
            allowed_tools=["majors_retriever", "course_retriever", "prerequisite_graph"],
            executor_path="major_lookup",
        )

    def test_course_line_without_inline_prereqs_fails(self):
        """The exact bug the user reported: 'BIOL 110: Biology Concepts and
        Biodiversity' on its own line, no '(prerequisites: ...)' annotation.
        Must be flagged so the synthesizer puts the context back."""
        from orchestration_pipeline import _validate_inline_prereqs
        response = (
            "Semester 1:\n"
            "BIOL 110: Biology: Basic Concepts and Biodiversity\n"
            "CHEM 110: Chemical Principles I\n"
            "Total Credits: 6"
        )
        violation = _validate_inline_prereqs(response)
        self.assertIsNotNone(violation, "Course line without inline prereqs must be flagged")
        self.assertIn("Semester 1", violation)
        self.assertIn("BIOL 110", violation)

    def test_course_line_with_prereqs_none_passes(self):
        """A course with no enrollment prereqs is annotated with
        '(prerequisites: None)'. That counts as a valid annotation — the
        rule is "did the synthesizer surface prereq context?", not
        "did the prereq list have items?"."""
        from orchestration_pipeline import _validate_inline_prereqs
        response = (
            "Semester 1:\n"
            "BIOL 110: Biology Concepts (prerequisites: None)\n"
            "MATH 140: Calculus I (prerequisites: None)\n"
            "Total Credits: 6"
        )
        self.assertIsNone(_validate_inline_prereqs(response))

    def test_course_line_with_prereq_codes_passes(self):
        """The shape produced by aggregate_and_generate_response in the
        deterministic_plan path. Must continue to pass after the new
        validator lands so we do not regress that path."""
        from orchestration_pipeline import _validate_inline_prereqs
        response = (
            "Semester 1:\n"
            "DS 200 (prerequisites: None)\n"
            "MATH 140 (prerequisites: None)\n"
            "Total Credits: 6\n"
            "---\n"
            "Semester 2:\n"
            "CMPSC 121 (prerequisites: MATH 110, MATH 140)\n"
            "MATH 141 (prerequisites: MATH 140)\n"
            "Total Credits: 6"
        )
        self.assertIsNone(_validate_inline_prereqs(response))

    def test_course_line_with_not_retrieved_passes(self):
        """When the bundle has no prereq entry for a course, the
        synthesizer is required to say so explicitly rather than omit
        the annotation. Honest-unknown still satisfies the rule."""
        from orchestration_pipeline import _validate_inline_prereqs
        response = (
            "Semester 1:\n"
            "BIOL 220W: Genetics (prerequisites: not retrieved)\n"
            "Total Credits: 3"
        )
        self.assertIsNone(_validate_inline_prereqs(response))

    def test_categorized_placeholder_lines_are_exempt(self):
        """Lines without a course code (gen-ed placeholders, deferral
        lines) carry no specific course whose prereqs would apply, so the
        inline-prereq rule does NOT apply to them."""
        from orchestration_pipeline import _validate_inline_prereqs
        response = (
            "Semester 1:\n"
            "BIOL 110: Biology Concepts (prerequisites: None)\n"
            "GH Humanities elective — examples: HIST 020, ENGL 015\n"
            "Open slot — please confirm with your advisor\n"
            "Total Credits: 9"
        )
        self.assertIsNone(_validate_inline_prereqs(response))

    def test_validate_final_response_surfaces_missing_inline_prereqs(self):
        """End-to-end: a course-recommendation response where the courses
        lack inline prereq annotations must surface
        ``missing_inline_prereqs`` so the executor's retry path can fix it.
        This is the contract the new retry directive depends on."""
        from orchestration_pipeline import validate_final_response

        response = (
            "Semester 1:\n"
            "BIOL 110: Biology: Basic Concepts and Biodiversity\n"
            "CHEM 110: Chemical Principles I\n"
            "MATH 140: Calculus with Analytic Geometry I\n"
            "PSYCH 100: Introductory Psychology\n"
            "Total Credits: 12"
        )
        executor_result = ExecutorResult(
            response=response,
            tools_used=["majors_retriever", "course_retriever", "prerequisite_graph"],
            metadata={},
            notes=[],
        )
        report = validate_final_response(self._plan(), response, executor_result)
        self.assertFalse(report["passed"])
        self.assertIn("missing_inline_prereqs", report["issues"])
        self.assertIn("inline_prereq_violation_detail", report)

    def test_validator_skips_non_recommendation_deliverables(self):
        """A policy_lookup answer that mentions a course code in prose
        ('CMPSC 132 has the same enrollment requirements as CMPSC 131')
        must not trip the inline-prereq rule — the rule only walks
        semester blocks and only fires for course-recommendation
        deliverables."""
        from orchestration_pipeline import OrchestrationPlan, validate_final_response

        plan = OrchestrationPlan(
            original_question="What is the prereq policy?",
            normalized_question="prereq policy",
            corrected_question="prereq policy",
            resolved_user_need="x",
            domain="policy",
            deliverable="policy_lookup",
            plan_scope="direct_answer",
            semester_horizon=0,
            use_tools=True,
            allowed_tools=["policy_retriever"],
            executor_path="policy_lookup",
        )
        response = (
            "Penn State enforces enrollment prerequisites strictly. "
            "For example, CMPSC 132 requires CMPSC 131 to be completed first."
        )
        executor_result = ExecutorResult(
            response=response,
            tools_used=["policy_retriever"],
            metadata={},
            notes=[],
        )
        report = validate_final_response(plan, response, executor_result)
        self.assertNotIn("missing_inline_prereqs", report["issues"])

    def test_categorized_placeholder_preferred_over_deferral_in_prompt(self):
        """The synthesis prompt must explicitly tell the LLM that when it
        knows the bucket category (humanities, writing/speaking, social
        science, etc.), it MUST use the categorized-placeholder form (b)
        and NOT fall back to the apologetic deferral form (c). This is
        what stops 'Open humanities slot — bundle does not list eligible
        courses' from ever shipping when 'Humanities' is enough info.

        We verify this at the prompt level (rather than runtime) so the
        contract is locked even if the test environment lacks LLM
        credentials."""
        if main_copy is None:
            self.skipTest(f"main_copy unavailable in this environment: {MAIN_COPY_IMPORT_ERROR}")
        import inspect

        source = inspect.getsource(main_copy._build_major_lookup_draft)
        # The system prompt must prefer (b) over (c) and reserve (c) for
        # cases where the bucket is *genuinely* unidentifiable.
        self.assertIn("you MUST use this form", source)
        self.assertIn("RESERVED for slots whose bucket category", source)
        # The retry directive for under_specified_recommendation must
        # repeat the same hierarchy so a retry does not re-introduce the
        # apologetic deferral.
        self.assertIn("Form (c) is a last resort", source)

    def test_inline_prereq_rule_documented_in_synthesis_prompt(self):
        """The synthesis prompt must contain the inline-prereq rule so
        the LLM emits the annotation on first attempt rather than only
        on retry. Locks the prompt-level contract."""
        if main_copy is None:
            self.skipTest(f"main_copy unavailable in this environment: {MAIN_COPY_IMPORT_ERROR}")
        import inspect

        source = inspect.getsource(main_copy._build_major_lookup_draft)
        self.assertIn("INLINE PREREQUISITE ANNOTATION RULE", source)
        self.assertIn("(prerequisites:", source)
        # Retry directive must exist for this failure mode.
        self.assertIn("missing_inline_prereqs", source)


class HorizonCoercionTests(unittest.TestCase):
    """Locks in tolerant ``semester_horizon`` parsing.

    The orchestrator LLM is allowed to express the planning horizon in any
    natural shape — integer, integer-with-noun, hyphenated, semantic phrase.
    The previous ``int(raw)`` parser silently dropped any non-numeric value
    to ``0``, which then paired with ``plan_scope=multi_semester`` to produce
    the contradictory ``Do not return more than 1 semesters`` directive that
    crashed Trace 2. These tests fix the contract so that never happens
    again, regardless of how creative the planner gets.
    """

    def test_integer_inputs_pass_through(self):
        from orchestration_pipeline import _coerce_semester_horizon
        self.assertEqual(_coerce_semester_horizon(8, "multi_semester", 0), 8)
        self.assertEqual(_coerce_semester_horizon(0, "direct_answer", 0), 0)
        self.assertEqual(_coerce_semester_horizon(99, "full_graduation", 0), 8)
        self.assertEqual(_coerce_semester_horizon(-1, "direct_answer", 0), 0)

    def test_strings_with_embedded_numbers(self):
        """The exact Trace 2 failure: '8 semesters' must coerce to 8, not 0."""
        from orchestration_pipeline import _coerce_semester_horizon
        cases = [
            ("8 semesters", 8),
            ("8-semester", 8),
            ("next 4", 4),
            ("the next 2 semesters", 2),
            ("1 semester", 1),
            ("3 sem", 3),
            ("8", 8),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(
                    _coerce_semester_horizon(raw, "multi_semester", 0),
                    expected,
                )

    def test_semantic_phrases_and_word_numbers(self):
        from orchestration_pipeline import _coerce_semester_horizon
        cases = [
            ("all_remaining_semesters", 8),
            ("until graduation", 8),
            ("through graduation", 8),
            ("full graduation", 8),
            ("entire plan", 8),
            ("next semester", 1),
            ("upcoming semester", 1),
            ("four semesters", 4),
            ("two", 2),
            ("eight", 8),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(
                    _coerce_semester_horizon(raw, "multi_semester", 0),
                    expected,
                )

    def test_empty_or_unparseable_falls_back_to_scope_default(self):
        """When the planner omits or garbles the horizon, infer it from the
        scope so we never produce a contradictory pair like
        (multi_semester, 0)."""
        from orchestration_pipeline import _coerce_semester_horizon
        cases = [
            (None, "full_graduation", 8),
            ("", "full_graduation", 8),
            ("???", "full_graduation", 8),
            (None, "multi_semester", 4),
            ("", "single_semester", 1),
            (None, "direct_answer", 0),
            ("nonsense", "audit", 0),
        ]
        for raw, scope, expected in cases:
            with self.subTest(raw=raw, scope=scope):
                self.assertEqual(
                    _coerce_semester_horizon(raw, scope, 99),
                    expected,
                )

    def test_reconcile_raises_zero_horizon_for_multi_scope(self):
        """The post-coercion invariant: scope and horizon must agree.
        ``multi_semester`` + ``0`` must be raised to the scope default,
        otherwise ``_build_must_not_do`` builds the bug-2 directive."""
        from orchestration_pipeline import _reconcile_scope_and_horizon
        self.assertEqual(_reconcile_scope_and_horizon("multi_semester", 0), 4)
        self.assertEqual(_reconcile_scope_and_horizon("full_graduation", 0), 8)
        self.assertEqual(_reconcile_scope_and_horizon("single_semester", 5), 1)
        self.assertEqual(_reconcile_scope_and_horizon("direct_answer", 5), 0)
        self.assertEqual(_reconcile_scope_and_horizon("audit", 3), 0)
        # Healthy values are passed through.
        self.assertEqual(_reconcile_scope_and_horizon("multi_semester", 4), 4)

    def test_end_to_end_trace2_horizon_string_no_longer_breaks_must_not_do(self):
        """End-to-end regression for Trace 2: when the planner emits
        ``semester_horizon='8 semesters'`` paired with
        ``plan_scope='multi_semester'`` and ``deliverable='multi_semester_plan'``,
        the resulting plan must NOT carry 'Do not return more than 1 semesters'.
        That directive is the symptom that caused the synthesizer to fail
        scope validation and silently fall through to web_search."""
        from orchestration_pipeline import _sanitize_plan_dict, OrchestrationPlan

        fallback = OrchestrationPlan(
            original_question="x",
            normalized_question="x",
            corrected_question="x",
            resolved_user_need="x",
            domain="major",
            deliverable="multi_semester_plan",
            plan_scope="multi_semester",
            semester_horizon=0,
            use_tools=True,
        )
        parsed = {
            "domain": "major",
            "deliverable": "multi_semester_plan",
            "plan_scope": "multi_semester",
            "semester_horizon": "8 semesters",
            "use_tools": True,
            "executor_path": "major_lookup",
        }
        plan = _sanitize_plan_dict(
            parsed, fallback, available_tools=["majors_retriever", "course_retriever"]
        )
        self.assertEqual(plan.semester_horizon, 8)
        self.assertEqual(plan.plan_scope, "multi_semester")


class CoherentLLMPlanProtectionTests(unittest.TestCase):
    """Locks in the rule that ``_apply_deterministic_overrides`` must NEVER
    downgrade a coherent grounded LLM plan via Phase 2 keyword rewriting.

    Trace 1 was the canonical failure: the orchestrator LLM produced a
    perfect ``major + major_lookup + 5-step retrieval_plan`` for an
    Art-and-Accounting double-major question, and Phase 2 rewrote it to
    ``transcript + deterministic_plan`` because the ``corrected_question``
    contained the words "course plan". The deterministic_plan path only
    knows how to handle a single major, which is how the user got an
    Art-only sample syllabus from a double-major request.
    """

    def _trace_1_orchestrator_json(self) -> str:
        """The relevant fields from Trace 1's actual orchestrator output."""
        return """{
          "normalized_question": "Sample syllabus or example course plan for an Art and Accounting double major",
          "corrected_question": "Sample syllabus or example course plan for an Art and Accounting double major",
          "resolved_user_need": "Provide a sample first-year plan covering both majors.",
          "domain": "major",
          "deliverable": "major_lookup",
          "plan_scope": "direct_answer",
          "semester_horizon": 0,
          "use_tools": true,
          "allowed_tools": ["majors_retriever", "course_retriever", "prerequisite_graph"],
          "executor_path": "major_lookup",
          "must_not_do": [],
          "success_criteria": "Includes specific courses for both Art and Accounting.",
          "retrieval_plan": [
            {"tool": "majors_retriever", "query": "Art major requirements", "purpose": "Pull Art catalog"},
            {"tool": "majors_retriever", "query": "Accounting major requirements", "purpose": "Pull Accounting catalog"},
            {"tool": "course_retriever", "query": "Art foundation courses", "purpose": "Identify intro Art courses"},
            {"tool": "course_retriever", "query": "Accounting prerequisite courses", "purpose": "Identify intro Accounting courses"},
            {"tool": "prerequisite_graph", "query": "ART/ACCTG prereq chains", "purpose": "Build prereq graph"}
          ],
          "synthesis_brief": "Combine retrievals into a year-1 sample syllabus covering both majors."
        }"""

    def test_trace1_coherent_major_plan_survives_overrides(self):
        plan, _ = build_orchestration_plan(
            user_question="give me a sample syllubus for an art and accounting double major",
            conversation_history=[],
            has_transcript=False,
            available_tools=[
                "majors_retriever",
                "course_retriever",
                "prerequisite_graph",
                "deterministic_planner",
                "transcript_state",
                "conversation_history",
            ],
            llm=FakeLLM([self._trace_1_orchestrator_json()]),
        )

        self.assertEqual(plan.domain, "major", "Coherent major plan must survive overrides")
        self.assertEqual(plan.executor_path, "major_lookup",
                         "Coherent major plan must keep its major_lookup path")
        self.assertEqual(plan.deliverable, "major_lookup")
        self.assertTrue(plan.use_tools, "Grounded plans never get use_tools=false")
        self.assertEqual(
            len(plan.retrieval_plan or []),
            5,
            "All 5 LLM retrieval steps must survive overrides",
        )

    def test_incoherent_plan_still_gets_keyword_repair(self):
        """The coherence gate is conservative: when the LLM ships an
        incoherent plan (empty domain or empty retrieval_plan), Phase 2
        keyword fallback is allowed to take over and route the question."""
        empty_orchestrator_json = """{
          "normalized_question": "give me a graduation plan for an art major",
          "corrected_question": "give me a graduation plan for an art major",
          "resolved_user_need": "Provide a graduation plan.",
          "domain": "",
          "deliverable": "direct_answer",
          "plan_scope": "direct_answer",
          "semester_horizon": 0,
          "use_tools": false,
          "allowed_tools": [],
          "executor_path": "general_academic_chat",
          "must_not_do": [],
          "success_criteria": "",
          "retrieval_plan": []
        }"""

        plan, _ = build_orchestration_plan(
            user_question="give me a graduation plan for an art major",
            conversation_history=[],
            has_transcript=False,
            available_tools=["deterministic_planner", "transcript_state", "conversation_history"],
            llm=FakeLLM([empty_orchestrator_json]),
        )

        # Phase 2 is allowed to fire here because the LLM plan was incoherent
        # (empty domain, empty retrieval_plan). The keyword fallback routes
        # to a transcript / planning path.
        self.assertIn(plan.domain, {"transcript", "major"})
        self.assertNotEqual(plan.executor_path, "general_academic_chat")

    def test_coherent_helper_distinguishes_coherence_classes(self):
        from orchestration_pipeline import (
            _llm_plan_is_coherent,
            OrchestrationPlan,
        )

        coherent = OrchestrationPlan(
            original_question="x",
            normalized_question="x",
            corrected_question="x",
            resolved_user_need="x",
            domain="major",
            deliverable="major_lookup",
            plan_scope="direct_answer",
            semester_horizon=0,
            use_tools=True,
            executor_path="major_lookup",
            retrieval_plan=[{"tool": "majors_retriever", "query": "Art"}],
        )
        self.assertTrue(_llm_plan_is_coherent(coherent))

        no_retrieval = OrchestrationPlan(
            original_question="x",
            normalized_question="x",
            corrected_question="x",
            resolved_user_need="x",
            domain="major",
            deliverable="major_lookup",
            plan_scope="direct_answer",
            semester_horizon=0,
            use_tools=True,
            executor_path="major_lookup",
            retrieval_plan=[],
        )
        self.assertFalse(_llm_plan_is_coherent(no_retrieval))

        wrong_domain = OrchestrationPlan(
            original_question="x",
            normalized_question="x",
            corrected_question="x",
            resolved_user_need="x",
            domain="general_academic_chat",
            deliverable="direct_answer",
            plan_scope="direct_answer",
            semester_horizon=0,
            use_tools=True,
            executor_path="general_academic_chat",
            retrieval_plan=[{"tool": "web_search", "query": "x"}],
        )
        self.assertFalse(_llm_plan_is_coherent(wrong_domain))

        no_tools = OrchestrationPlan(
            original_question="x",
            normalized_question="x",
            corrected_question="x",
            resolved_user_need="x",
            domain="major",
            deliverable="major_lookup",
            plan_scope="direct_answer",
            semester_horizon=0,
            use_tools=False,
            executor_path="major_lookup",
            retrieval_plan=[{"tool": "majors_retriever", "query": "Art"}],
        )
        self.assertFalse(_llm_plan_is_coherent(no_tools))


class WebFallbackAcceptanceTests(unittest.TestCase):
    """Locks in the rule that the web_search fallback in
    ``_build_major_lookup_draft`` must NOT ship vague registrar prose for a
    course-recommendation deliverable. (Trace 2.)

    Honest unknown beats confident garbage: when the web fallback returns
    text that lacks any ``Semester N: ... Total Credits: N`` block AND any
    course code for a single/multi/full plan, we suppress it and return a
    soft-ask listing what was retrieved instead.
    """

    def _make_plan(self, scope: str, deliverable: str) -> "OrchestrationPlan":
        from orchestration_pipeline import OrchestrationPlan
        return OrchestrationPlan(
            original_question="x",
            normalized_question="x",
            corrected_question="x",
            resolved_user_need="x",
            domain="major",
            deliverable=deliverable,
            plan_scope=scope,
            semester_horizon=4,
            use_tools=True,
            executor_path="major_lookup",
            retrieval_plan=[{"tool": "majors_retriever", "query": "Art"}],
        )

    def test_acceptance_check_rejects_vague_prose_for_multi_semester_plan(self):
        """Trace 2's exact failure: web fallback returned bullet-list prose
        with no Semester block and no course code, and the system shipped it.
        The acceptance check must reject that."""
        if main_copy is None:
            self.skipTest(f"main_copy not importable: {MAIN_COPY_IMPORT_ERROR}")
        vague = (
            "- Semester 1 - Intro Art course\n"
            "- Semester 2 - More art classes\n"
            "Visit the Penn State Abington registrar for details."
        )
        plan = self._make_plan("multi_semester", "multi_semester_plan")
        self.assertFalse(
            main_copy._web_fallback_is_structurally_acceptable(vague, plan),
            "Vague prose with no semester block and no course code must be rejected",
        )

    def test_acceptance_check_accepts_real_plan_with_block_and_codes(self):
        """A real semester block with course codes is accepted."""
        if main_copy is None:
            self.skipTest(f"main_copy not importable: {MAIN_COPY_IMPORT_ERROR}")
        good = (
            "Semester 1:\n"
            "- ART 010: Drawing I (prerequisites: None) — 3 credits\n"
            "- ENGL 015: Rhetoric and Composition (prerequisites: None) — 3 credits\n"
            "Total Credits: 6\n"
        )
        plan = self._make_plan("multi_semester", "multi_semester_plan")
        self.assertTrue(
            main_copy._web_fallback_is_structurally_acceptable(good, plan),
        )

    def test_acceptance_check_accepts_anything_for_lookup_deliverables(self):
        """Lookup-style deliverables (direct_answer / major_lookup) accept
        any non-empty web response — a registrar paragraph is a valid
        answer to 'tell me about the Art major'."""
        if main_copy is None:
            self.skipTest(f"main_copy not importable: {MAIN_COPY_IMPORT_ERROR}")
        prose = "The Art major at Penn State Abington offers concentrations in painting, sculpture, and digital media."
        plan = self._make_plan("direct_answer", "major_lookup")
        self.assertTrue(
            main_copy._web_fallback_is_structurally_acceptable(prose, plan),
        )
        self.assertTrue(
            main_copy._web_fallback_is_structurally_acceptable(prose, None),
            "No plan at all is treated as a non-recommendation deliverable",
        )

    def test_build_major_lookup_draft_returns_soft_ask_when_web_fallback_is_vague(self):
        """End-to-end: when retrieval returns nothing AND the web fallback
        returns vague prose, ``_build_major_lookup_draft`` must return the
        honest soft-ask, not the vague prose."""
        if main_copy is None:
            self.skipTest(f"main_copy not importable: {MAIN_COPY_IMPORT_ERROR}")
        from orchestration_pipeline import OrchestrationPlan

        plan = OrchestrationPlan(
            original_question="art and accounting double major plan",
            normalized_question="art and accounting double major plan",
            corrected_question="art and accounting double major plan",
            resolved_user_need="multi-semester plan covering both majors",
            domain="major",
            deliverable="multi_semester_plan",
            plan_scope="multi_semester",
            semester_horizon=4,
            use_tools=True,
            executor_path="major_lookup",
            retrieval_plan=[],  # empty so we hit the canonical fallback
        )

        # All retrievers return empty so the bundle stays empty and the
        # function falls through to the web_search branch.
        empty_registry = {
            "majors_retriever":   lambda q, ctx: "",
            "course_retriever":   lambda q, ctx: "",
            "prerequisite_graph": lambda q, ctx: "",
        }
        fake_search_result = {
            "answer": "- Semester 1 - Intro Art class\n- Semester 2 - More art\n",
            "citations": [],
        }

        with patch.dict(main_copy.EXECUTOR_TOOLS, empty_registry, clear=True), \
             patch.object(
                 main_copy,
                 "_run_shared_search_pipeline",
                 lambda **kw: fake_search_result,
             ), \
             patch.object(
                 main_copy,
                 "_research_answer_text",
                 lambda r: r.get("answer", ""),
             ), \
             patch.object(
                 main_copy,
                 "_resolved_search_to_metadata",
                 lambda r: {},
             ):
            response, tools_used, metadata = main_copy._build_major_lookup_draft(
                user_question="art and accounting double major plan",
                trace=None,
                allowed_tools={
                    "majors_retriever", "course_retriever",
                    "prerequisite_graph", "web_search",
                },
                plan=plan,
            )

        self.assertEqual(metadata.get("source"), "major_lookup_not_found")
        self.assertNotIn("Semester 1 - Intro Art class", response,
                         "Vague web fallback must not be shipped")
        self.assertTrue(
            response.endswith("?") or "?" in response,
            "Soft-ask must contain a clarifying question",
        )

    def test_build_major_lookup_draft_passes_through_real_plan_from_web_search(self):
        """Sanity check: when the web fallback DOES return a structurally
        acceptable plan (semester block + course code), it must be shipped."""
        if main_copy is None:
            self.skipTest(f"main_copy not importable: {MAIN_COPY_IMPORT_ERROR}")
        from orchestration_pipeline import OrchestrationPlan

        plan = OrchestrationPlan(
            original_question="x",
            normalized_question="x",
            corrected_question="x",
            resolved_user_need="x",
            domain="major",
            deliverable="multi_semester_plan",
            plan_scope="multi_semester",
            semester_horizon=4,
            use_tools=True,
            executor_path="major_lookup",
            retrieval_plan=[],
        )
        empty_registry = {
            "majors_retriever":   lambda q, ctx: "",
            "course_retriever":   lambda q, ctx: "",
            "prerequisite_graph": lambda q, ctx: "",
        }
        good_plan_text = (
            "Semester 1:\n"
            "- ART 010: Drawing I (prerequisites: None)\n"
            "Total Credits: 3"
        )
        with patch.dict(main_copy.EXECUTOR_TOOLS, empty_registry, clear=True), \
             patch.object(main_copy, "_run_shared_search_pipeline",
                          lambda **kw: {"answer": good_plan_text}), \
             patch.object(main_copy, "_research_answer_text",
                          lambda r: r.get("answer", "")), \
             patch.object(main_copy, "_resolved_search_to_metadata", lambda r: {}):
            response, tools_used, metadata = main_copy._build_major_lookup_draft(
                user_question="art plan",
                trace=None,
                allowed_tools={"majors_retriever", "course_retriever",
                               "prerequisite_graph", "web_search"},
                plan=plan,
            )

        self.assertEqual(metadata.get("source"), "shared_search_pipeline")
        self.assertIn("ART 010", response)


class MultiMajorContractTests(unittest.TestCase):
    """Locks in first-class multi-major awareness in the orchestrator + executor.

    Trace 1's Art-only answer to an Art-and-Accounting double-major question
    was the symptom that motivated this contract. The fix is structural:
    the orchestrator extracts ``majors``, the override layer fans canonical
    retrieval out per major, and the validator flags any final response that
    omits a requested major.
    """

    def test_sanitizer_parses_list_of_majors(self):
        from orchestration_pipeline import _sanitize_plan_dict, OrchestrationPlan
        fallback = OrchestrationPlan(
            original_question="x", normalized_question="x",
            corrected_question="x", resolved_user_need="x",
            domain="major", deliverable="major_lookup",
            plan_scope="direct_answer", semester_horizon=0,
            use_tools=True,
        )
        parsed = {
            "domain": "major",
            "deliverable": "major_lookup",
            "plan_scope": "direct_answer",
            "use_tools": True,
            "executor_path": "major_lookup",
            "majors": ["art", "ACCOUNTING ", "art", "  "],  # noisy + dup + empty
        }
        plan = _sanitize_plan_dict(parsed, fallback, available_tools=["majors_retriever"])
        self.assertEqual(plan.majors, ["Art", "Accounting"])

    def test_sanitizer_parses_string_with_conjunctions(self):
        """The planner is allowed to be lazy and emit a single string."""
        from orchestration_pipeline import _sanitize_majors_list
        self.assertEqual(
            _sanitize_majors_list("art and accounting", []),
            ["Art", "Accounting"],
        )
        self.assertEqual(
            _sanitize_majors_list("psych / bio", []),
            ["Psych", "Bio"],
        )
        self.assertEqual(
            _sanitize_majors_list("computer science, math & english", []),
            ["Computer Science", "Math", "English"],
        )
        self.assertEqual(_sanitize_majors_list("", ["Existing"]), ["Existing"])
        self.assertEqual(_sanitize_majors_list(None, []), [])

    def test_canonical_retrieval_plan_fans_out_per_major(self):
        """Without per-major fan-out, a single combined retrieval like
        'art and accounting double major' usually only hits the Art catalog
        and Accounting requirements never reach the synthesizer."""
        from orchestration_pipeline import _canonical_retrieval_plan
        steps = _canonical_retrieval_plan(
            executor_path="major_lookup",
            corrected_question="art and accounting double major plan",
            available_tools=["majors_retriever", "course_retriever", "prerequisite_graph"],
            majors=["Art", "Accounting"],
        )
        # 3 tools × 2 majors = 6 steps.
        self.assertEqual(len(steps), 6)
        majors_steps = [s for s in steps if s["tool"] == "majors_retriever"]
        self.assertEqual(len(majors_steps), 2,
                         "One majors_retriever query per major")
        queries_lower = " ".join(s.get("query", "").lower() for s in steps)
        self.assertIn("art", queries_lower)
        self.assertIn("accounting", queries_lower)

    def test_canonical_retrieval_plan_single_major_unchanged(self):
        """Single-major plans keep the original 3-step layout."""
        from orchestration_pipeline import _canonical_retrieval_plan
        steps = _canonical_retrieval_plan(
            executor_path="major_lookup",
            corrected_question="computer science first semester",
            available_tools=["majors_retriever", "course_retriever", "prerequisite_graph"],
            majors=["Computer Science"],
        )
        self.assertEqual(len(steps), 3)
        self.assertEqual([s["tool"] for s in steps],
                         ["majors_retriever", "course_retriever", "prerequisite_graph"])

    @unittest.skipIf(main_copy is None, f"main_copy unavailable: {MAIN_COPY_IMPORT_ERROR}")
    def test_executor_keeps_multi_major_plan_on_llm_retrieval_path(self):
        """The LLM orchestrator's multi-major route must not be overwritten by
        the single-major deterministic planner shortcut.

        This is the Biology + Computer Science failure mode: the route correctly
        knows there are two majors, but a later executor shortcut extracted one
        major and generated a single-major standard plan.
        """
        from orchestration_pipeline import OrchestrationPlan

        plan = OrchestrationPlan(
            original_question="biology and computer science double major plan",
            normalized_question="biology and computer science double major plan",
            corrected_question="biology and computer science double major plan",
            resolved_user_need="full plan covering both majors",
            domain="major",
            deliverable="full_graduation_plan",
            plan_scope="full_graduation",
            semester_horizon=8,
            use_tools=True,
            allowed_tools=["majors_retriever", "course_retriever", "prerequisite_graph"],
            executor_path="major_lookup",
            retrieval_plan=[{"tool": "majors_retriever", "query": "Biology", "reason": "r"}],
            majors=["Biology", "Computer Science"],
        )

        with patch.object(main_copy, "generate_plan_from_major_and_year") as deterministic_planner, \
             patch.object(
                 main_copy,
                 "_build_major_lookup_draft",
                 return_value=("multi-major grounded draft", ["majors_retriever"], {"source": "major_lookup_rag"}),
             ) as major_lookup:
            response, tools_used, metadata = main_copy._build_executor_draft(
                user_id="test",
                plan=plan,
                user_question=plan.corrected_question,
                conversation_history=[],
                has_transcript_for_plan=False,
                student_data=None,
                trace=None,
            )

        deterministic_planner.assert_not_called()
        major_lookup.assert_called_once()
        self.assertEqual(response, "multi-major grounded draft")
        self.assertEqual(tools_used, ["majors_retriever"])
        self.assertEqual(metadata["source"], "major_lookup_rag")

    def test_validator_flags_single_major_response_to_double_major_request(self):
        from orchestration_pipeline import _validate_multi_major_coverage
        art_only = (
            "Semester 1:\n"
            "- ART 010: Drawing I (prerequisites: None)\n"
            "Total Credits: 3"
        )
        detail = _validate_multi_major_coverage(art_only, ["Art", "Accounting"])
        self.assertIsNotNone(detail)
        self.assertIn("Accounting", detail)

    def test_validator_rejects_multi_major_plan_with_unlabeled_course_lines(self):
        from orchestration_pipeline import _validate_multi_major_coverage

        generic_double_major = (
            "This plan covers Biology and Computer Science.\n"
            "Semester 1:\n"
            "- BIOL 110: Biology: Basic Concepts and Biodiversity (prerequisites: None)\n"
            "- CMPSC 131: Programming and Computation I (prerequisites: None)\n"
            "Total Credits: 7"
        )

        detail = _validate_multi_major_coverage(
            generic_double_major,
            ["Biology", "Computer Science"],
        )
        self.assertIsNotNone(detail)
        self.assertIn("without a major label", detail)

    def test_validator_passes_when_both_majors_named(self):
        from orchestration_pipeline import _validate_multi_major_coverage
        covered = (
            "This plan covers your Art and Accounting double major.\n"
            "Semester 1:\n"
            "- ART 010: Drawing I (prerequisites: None) [Art]\n"
            "- ACCTG 211: Financial and Managerial Accounting (prerequisites: None) [Accounting]\n"
            "Total Credits: 6"
        )
        self.assertIsNone(_validate_multi_major_coverage(covered, ["Art", "Accounting"]))

    def test_validate_final_response_surfaces_missing_major_coverage(self):
        from orchestration_pipeline import (
            validate_final_response, OrchestrationPlan, ExecutorResult,
        )
        plan = OrchestrationPlan(
            original_question="art and accounting double major plan",
            normalized_question="art and accounting double major plan",
            corrected_question="art and accounting double major plan",
            resolved_user_need="multi-semester plan covering both majors",
            domain="major",
            deliverable="multi_semester_plan",
            plan_scope="multi_semester",
            semester_horizon=4,
            use_tools=True,
            executor_path="major_lookup",
            retrieval_plan=[{"tool": "majors_retriever", "query": "Art"}],
            majors=["Art", "Accounting"],
        )
        art_only_response = (
            "Semester 1:\n"
            "- ART 010: Drawing I (prerequisites: None)\n"
            "- ART 020: Painting I (prerequisites: None)\n"
            "Total Credits: 6\n"
        )
        result = validate_final_response(
            plan,
            art_only_response,
            ExecutorResult(
                response=art_only_response,
                tools_used=["majors_retriever"],
                metadata={},
            ),
        )
        self.assertIn("missing_major_coverage", result["issues"])
        self.assertIn("multi_major_coverage_detail", result)
        self.assertIn("Accounting", result["multi_major_coverage_detail"])


class PrereqBackfillTests(unittest.TestCase):
    """Locks in the bundle self-consistency invariant: every course code the
    bundle mentions must have a ``prerequisite_graph`` lookup attempt before
    synthesis runs.

    This is the structural fix for the "(prerequisites: not retrieved)
    everywhere" bug on the Art + Accounting double-major plan. The planner
    is allowed to be lazy and only list a handful of codes in its
    ``prerequisite_graph`` step — the executor is responsible for noticing
    that the bundle mentions more codes than the planner asked about, and
    issuing a follow-up call to cover the rest. No keywords involved; the
    rule is purely "the bundle must be self-consistent".
    """

    def _plan(self) -> "OrchestrationPlan":
        from orchestration_pipeline import OrchestrationPlan
        return OrchestrationPlan(
            original_question="art and accounting double major plan",
            normalized_question="x",
            corrected_question="x",
            resolved_user_need="x",
            domain="major",
            deliverable="full_graduation_plan",
            plan_scope="full_graduation",
            semester_horizon=8,
            use_tools=True,
            executor_path="major_lookup",
            allowed_tools=["majors_retriever", "course_retriever", "prerequisite_graph"],
            retrieval_plan=[
                {"tool": "majors_retriever", "query": "Art major requirements", "reason": "r1"},
                {"tool": "majors_retriever", "query": "Accounting major requirements", "reason": "r2"},
                # Planner's prereq step is intentionally narrow — only 3 codes
                # — even though the majors_retriever bundle below mentions
                # many more. This is the "lazy planner" case the backfill
                # exists to cover.
                {"tool": "prerequisite_graph", "query": "ART 110, ART 111, ACCTG 211", "reason": "r3"},
            ],
            majors=["Art", "Accounting"],
        )

    def test_collect_codes_from_text_dedupes_and_orders(self):
        if main_copy is None:
            self.skipTest(f"main_copy not importable: {MAIN_COPY_IMPORT_ERROR}")
        text = (
            "Prescribed Courses: ART 11, ART 110, ART 111, ART 122Y, ARTH 111. "
            "Required: ACCTG 211, ECON 102, ECON 102, MGMT 301."
        )
        codes = main_copy._collect_codes_from_text(text)
        # Order preserved, duplicates dropped.
        self.assertEqual(
            codes,
            ["ART 11", "ART 110", "ART 111", "ART 122Y",
             "ARTH 111", "ACCTG 211", "ECON 102", "MGMT 301"],
        )

    def test_collect_codes_already_attempted_reads_both_shapes(self):
        if main_copy is None:
            self.skipTest(f"main_copy not importable: {MAIN_COPY_IMPORT_ERROR}")
        text = (
            "Prerequisites for ART 110:\n- Enforced prerequisite at enrollment: None\n\n"
            "No course named 'ART 122' found.\n\n"
            "Prerequisites for ACCTG 211:\n- Enforced prerequisite at enrollment: MATH 21"
        )
        attempted = main_copy._collect_codes_already_attempted(text)
        self.assertEqual(attempted, {"ART 110", "ART 122", "ACCTG 211"})

    def test_backfill_issues_followup_for_codes_not_covered_by_planner(self):
        """End-to-end: the planner's prereq step only listed 3 codes but
        majors_retriever mentions 5 distinct codes. The backfill must issue
        one ``prerequisite_graph`` call covering the missing 2 (ECON 102 and
        MGMT 301) and merge the result into the bundle. Without this, the
        synthesizer writes ``(prerequisites: not retrieved)`` for ECON 102
        and MGMT 301 — which is the exact bug from the user's trace."""
        if main_copy is None:
            self.skipTest(f"main_copy not importable: {MAIN_COPY_IMPORT_ERROR}")

        prereq_calls: List[str] = []

        def fake_prereq(query, ctx):
            prereq_calls.append(query)
            # Return one entry per code in the query.
            codes = main_copy._collect_codes_from_text(query)
            parts = [
                f"Prerequisites for {code}:\n- Enforced prerequisite at enrollment: None\n"
                f"- Enforced concurrent at enrollment: None"
                for code in codes
            ]
            return "\n\n".join(parts)

        majors_text = (
            "Major Name: Art, B.A. Prescribed: ART 110, ART 111. "
            "Major Name: Accounting, B.S. Prescribed: ACCTG 211, ECON 102, MGMT 301."
        )
        fake_registry = {
            "majors_retriever":   lambda q, ctx: majors_text,
            "course_retriever":   lambda q, ctx: "",  # not exercised here
            "prerequisite_graph": fake_prereq,
        }

        plan = self._plan()
        with patch.dict(main_copy.EXECUTOR_TOOLS, fake_registry, clear=True):
            bundle, tools_used = main_copy._execute_retrieval_plan(plan)

        self.assertIn("prerequisite_graph", bundle)
        prereq_text = bundle["prerequisite_graph"]["results"]
        # Every code mentioned in majors_retriever now has an entry, including
        # the ones the planner did NOT pre-list.
        for code in ("ART 110", "ART 111", "ACCTG 211", "ECON 102", "MGMT 301"):
            self.assertIn(f"Prerequisites for {code}:", prereq_text,
                          f"backfill must cover {code}")

        # The original planner call ran first; the backfill ran second with
        # only the missing codes.
        self.assertEqual(len(prereq_calls), 2)
        self.assertEqual(
            main_copy._collect_codes_from_text(prereq_calls[0]),
            ["ART 110", "ART 111", "ACCTG 211"],
        )
        # Backfill should target only the codes the planner missed.
        self.assertEqual(
            set(main_copy._collect_codes_from_text(prereq_calls[1])),
            {"ECON 102", "MGMT 301"},
        )
        self.assertIn("prerequisite_graph", tools_used)

    def test_backfill_skips_already_attempted_no_course_codes(self):
        """If the planner queried a code and the graph returned 'No course
        named X found.', that code counts as ATTEMPTED — re-querying would
        just repeat the same negative answer. Only truly un-queried codes
        should be in the follow-up call."""
        if main_copy is None:
            self.skipTest(f"main_copy not importable: {MAIN_COPY_IMPORT_ERROR}")

        prereq_calls: List[str] = []

        def fake_prereq(query, ctx):
            prereq_calls.append(query)
            codes = main_copy._collect_codes_from_text(query)
            parts = []
            for code in codes:
                if code == "ART 11":
                    parts.append(f"No course named '{code}' found.")
                else:
                    parts.append(
                        f"Prerequisites for {code}:\n"
                        f"- Enforced prerequisite at enrollment: None\n"
                        f"- Enforced concurrent at enrollment: None"
                    )
            return "\n\n".join(parts)

        majors_text = "Prescribed: ART 11, ART 110, ECON 102."
        fake_registry = {
            "majors_retriever":   lambda q, ctx: majors_text,
            "course_retriever":   lambda q, ctx: "",
            "prerequisite_graph": fake_prereq,
        }

        from orchestration_pipeline import OrchestrationPlan
        plan = OrchestrationPlan(
            original_question="x", normalized_question="x",
            corrected_question="x", resolved_user_need="x",
            domain="major", deliverable="single_semester_plan",
            plan_scope="single_semester", semester_horizon=1,
            use_tools=True, executor_path="major_lookup",
            allowed_tools=["majors_retriever", "prerequisite_graph"],
            retrieval_plan=[
                {"tool": "majors_retriever", "query": "Art", "reason": "r1"},
                # Planner asked for ART 11 (which the graph doesn't have)
                # and ART 110 (which it does). Backfill should only chase
                # ECON 102 — NOT re-query ART 11.
                {"tool": "prerequisite_graph", "query": "ART 11, ART 110", "reason": "r2"},
            ],
        )
        with patch.dict(main_copy.EXECUTOR_TOOLS, fake_registry, clear=True):
            main_copy._execute_retrieval_plan(plan)

        self.assertEqual(len(prereq_calls), 2)
        backfill_codes = set(main_copy._collect_codes_from_text(prereq_calls[1]))
        self.assertEqual(
            backfill_codes,
            {"ECON 102"},
            "Backfill must skip codes the graph already returned an empty answer for",
        )

    def test_backfill_skipped_for_non_recommendation_deliverables(self):
        """The bundle self-consistency rule only matters for course
        recommendation deliverables. A simple ``major_lookup`` describing
        what a major IS does not need every mentioned code prereq-checked."""
        if main_copy is None:
            self.skipTest(f"main_copy not importable: {MAIN_COPY_IMPORT_ERROR}")

        prereq_calls: List[str] = []

        def fake_prereq(query, ctx):
            prereq_calls.append(query)
            return f"Prerequisites for ART 110:\n- Enforced prerequisite at enrollment: None"

        from orchestration_pipeline import OrchestrationPlan
        plan = OrchestrationPlan(
            original_question="what is the art major", normalized_question="x",
            corrected_question="x", resolved_user_need="x",
            domain="major", deliverable="major_lookup",
            plan_scope="direct_answer", semester_horizon=0,
            use_tools=True, executor_path="major_lookup",
            allowed_tools=["majors_retriever", "prerequisite_graph"],
            retrieval_plan=[
                {"tool": "majors_retriever", "query": "Art major",  "reason": "r1"},
                {"tool": "prerequisite_graph", "query": "ART 110", "reason": "r2"},
            ],
        )
        majors_text = "Prescribed: ART 11, ART 110, ART 122Y, ECON 102."
        with patch.dict(main_copy.EXECUTOR_TOOLS, {
            "majors_retriever":   lambda q, ctx: majors_text,
            "prerequisite_graph": fake_prereq,
        }, clear=True):
            main_copy._execute_retrieval_plan(plan)

        # Only the planner's single prereq step ran; backfill did not fire.
        self.assertEqual(len(prereq_calls), 1)
        self.assertNotIn("ECON 102", prereq_calls[0])

    def test_backfill_creates_prereq_section_when_planner_omitted_it(self):
        """If the planner didn't include a prereq step at all but the
        deliverable is a course recommendation, backfill must still run and
        seed the prereq section so the synthesizer has something to cite."""
        if main_copy is None:
            self.skipTest(f"main_copy not importable: {MAIN_COPY_IMPORT_ERROR}")

        prereq_calls: List[str] = []

        def fake_prereq(query, ctx):
            prereq_calls.append(query)
            codes = main_copy._collect_codes_from_text(query)
            return "\n\n".join(
                f"Prerequisites for {c}:\n- Enforced prerequisite at enrollment: None"
                for c in codes
            )

        from orchestration_pipeline import OrchestrationPlan
        plan = OrchestrationPlan(
            original_question="art plan", normalized_question="x",
            corrected_question="x", resolved_user_need="x",
            domain="major", deliverable="single_semester_plan",
            plan_scope="single_semester", semester_horizon=1,
            use_tools=True, executor_path="major_lookup",
            allowed_tools=["majors_retriever", "prerequisite_graph"],
            retrieval_plan=[
                {"tool": "majors_retriever", "query": "Art major", "reason": "r1"},
            ],
        )
        majors_text = "Prescribed: ART 110, ART 111."
        with patch.dict(main_copy.EXECUTOR_TOOLS, {
            "majors_retriever":   lambda q, ctx: majors_text,
            "prerequisite_graph": fake_prereq,
        }, clear=True):
            bundle, tools_used = main_copy._execute_retrieval_plan(plan)

        self.assertIn("prerequisite_graph", bundle,
                      "Backfill must seed prereq section when planner skipped it")
        self.assertIn("Prerequisites for ART 110:", bundle["prerequisite_graph"]["results"])
        self.assertIn("Prerequisites for ART 111:", bundle["prerequisite_graph"]["results"])
        self.assertIn("prerequisite_graph", tools_used)


class InlinePrereqRewriteTests(unittest.TestCase):
    """Locks in the deterministic '(prerequisites: ...)' rewrite.

    The synthesizer LLM is the wrong tool for transcribing prereq lists. It
    routinely either drops a real prereq (writes 'None' for MATH 140) or
    upgrades a 'concurrent' option to a fake enforced prereq (writes
    'ARTH 250' for PHOTO 201). The rewrite reads the verified prereq graph
    and overwrites the parenthetical contents on each course line so the
    student never sees a hallucinated annotation. These tests exist so
    nobody can quietly remove that safety net.
    """

    GRAPH_TEXT = (
        "Prerequisites for MATH 140:\n"
        "- Enforced prerequisite at enrollment: Math 22 and Math 26 or "
        "satisfactory performance on the mathematics placement examination.\n"
        "- Enforced concurrent at enrollment: None\n\n"
        "Prerequisites for PHOTO 201:\n"
        "- Enforced prerequisite at enrollment: None\n"
        "- Enforced concurrent at enrollment: ARTH 250\n\n"
        "Prerequisites for ART 110:\n"
        "- Enforced prerequisite at enrollment: None\n"
        "- Enforced concurrent at enrollment: None\n\n"
        "Prerequisites for ART 280:\n"
        "- Enforced prerequisite at enrollment: ART 110 or ART 111\n"
        "- Enforced concurrent at enrollment: None"
    )

    def test_overwrites_false_none_with_real_prereqs(self):
        """LLM wrote '(prerequisites: None)' for MATH 140 — graph says it
        actually has a real prereq. Rewrite must replace 'None' with the
        verified prereq text."""
        if main_copy is None:
            self.skipTest(f"main_copy not importable: {MAIN_COPY_IMPORT_ERROR}")
        response = "- MATH 140: Calculus I (prerequisites: None)"
        rewritten = main_copy._rewrite_inline_prereqs_from_graph(
            response, self.GRAPH_TEXT
        )
        self.assertNotIn("(prerequisites: None)", rewritten)
        self.assertIn("Math 22", rewritten)

    def test_overwrites_fake_prereq_when_only_concurrent(self):
        """LLM wrote '(prerequisites: ARTH 250)' for PHOTO 201 — graph
        only lists ARTH 250 as concurrent (not enforced). Rewrite must
        downgrade to 'None'."""
        if main_copy is None:
            self.skipTest(f"main_copy not importable: {MAIN_COPY_IMPORT_ERROR}")
        response = "- PHOTO 201: Studio Photography (prerequisites: ARTH 250)"
        rewritten = main_copy._rewrite_inline_prereqs_from_graph(
            response, self.GRAPH_TEXT
        )
        self.assertIn("(prerequisites: None)", rewritten)
        self.assertNotIn("ARTH 250", rewritten)

    def test_preserves_correct_annotations(self):
        """ART 280 already has the right prereq written verbatim. Rewrite
        must be idempotent (don't garble correct entries)."""
        if main_copy is None:
            self.skipTest(f"main_copy not importable: {MAIN_COPY_IMPORT_ERROR}")
        response = "- ART 280: Drawing II (prerequisites: ART 110 or ART 111)"
        rewritten = main_copy._rewrite_inline_prereqs_from_graph(
            response, self.GRAPH_TEXT
        )
        self.assertIn("(prerequisites: ART 110 or ART 111)", rewritten)

    def test_leaves_lines_without_inline_annotation_alone(self):
        """Headers, plain prose, and section dividers must not be touched."""
        if main_copy is None:
            self.skipTest(f"main_copy not importable: {MAIN_COPY_IMPORT_ERROR}")
        response = (
            "### Year 1\n"
            "**Semester 1:**\n"
            "Hi — here is your plan.\n"
            "**Total Credits: 15**"
        )
        rewritten = main_copy._rewrite_inline_prereqs_from_graph(
            response, self.GRAPH_TEXT
        )
        self.assertEqual(rewritten, response)

    def test_leaves_lines_for_unknown_courses_alone(self):
        """If the line's course is not in the graph at all, the LLM's
        annotation stays — we don't have ground truth to overwrite with."""
        if main_copy is None:
            self.skipTest(f"main_copy not importable: {MAIN_COPY_IMPORT_ERROR}")
        response = "- ZZZZ 999: Mystery Course (prerequisites: not retrieved)"
        rewritten = main_copy._rewrite_inline_prereqs_from_graph(
            response, self.GRAPH_TEXT
        )
        self.assertEqual(rewritten, response)

    def test_categorized_placeholder_with_examples_uses_first_real_code(self):
        """Lines like 'GH Humanities elective — examples: ART 110, ART 111'
        contain multiple codes. The first code that's in the graph is the
        recommendation target. ART 110 is in the graph (None prereqs), so
        the rewrite normalizes the annotation to '(prerequisites: None)'."""
        if main_copy is None:
            self.skipTest(f"main_copy not importable: {MAIN_COPY_IMPORT_ERROR}")
        response = (
            "- GH Humanities elective — examples: ART 110, ART 111 "
            "(prerequisites: None)"
        )
        rewritten = main_copy._rewrite_inline_prereqs_from_graph(
            response, self.GRAPH_TEXT
        )
        self.assertIn("(prerequisites: None)", rewritten)

    def test_empty_graph_text_passes_through(self):
        if main_copy is None:
            self.skipTest(f"main_copy not importable: {MAIN_COPY_IMPORT_ERROR}")
        response = "- MATH 140: Calculus I (prerequisites: None)"
        rewritten = main_copy._rewrite_inline_prereqs_from_graph(response, "")
        self.assertEqual(rewritten, response)


class FinalizerUniversalPrereqTests(unittest.TestCase):
    """Locks in the universal-inline-prereq rule for the finalizer.

    The rule: any course code mentioned ANYWHERE in the final response —
    including 'Things to consider next' bullets, prose, and follow-up
    questions — must end with '(prerequisites: ...)'. Two pieces have to
    work together for this to hold:

      1. The finalizer system prompt tells the LLM to add the annotation
         to every course mention.
      2. The deterministic rewrite pass runs again on the FINALIZER's
         output (not just the executor draft) so any annotation contents
         the LLM invented get overwritten with verified graph data.

    This test file locks in piece (2). Piece (1) is enforced by prompt
    text and is regression-tested via the existing
    InlinePrereqContractTests.
    """

    def test_finalizer_stage_rewrites_inline_prereqs_via_helper(self):
        from orchestration_pipeline import (  # local import to avoid module-load surprises
            ExecutorResult,
            OrchestrationPlan,
            run_finalizer_stage,
        )

        # Simulate a finalizer LLM that added a course mention with the
        # WRONG prereq annotation in a 'Things to consider next' bullet.
        bad_final = (
            '{"final_response":"Here is your plan.\\n\\nThings to consider '
            'next:\\n- CMPSC 132 (prerequisites: None) is a good follow-up '
            'next semester.","finalizer_notes":""}'
        )
        finalizer_llm = FakeLLM(responses=[bad_final])

        plan = OrchestrationPlan(
            original_question="cs plan",
            normalized_question="cs plan",
            corrected_question="cs plan",
            resolved_user_need="x",
            domain="major",
            deliverable="single_semester_plan",
            plan_scope="single_semester",
            semester_horizon=1,
            use_tools=True,
            executor_path="major_lookup",
            allowed_tools=["majors_retriever", "prerequisite_graph"],
        )
        # Verified graph data: CMPSC 132 actually requires CMPSC 131.
        prereq_graph_text = (
            "Prerequisites for CMPSC 132:\n"
            "- Enforced prerequisite at enrollment: CMPSC 131\n"
            "- Enforced concurrent at enrollment: None"
        )
        executor_result = ExecutorResult(
            response="Draft answer.",
            tools_used=["majors_retriever", "prerequisite_graph"],
            metadata={"prereq_graph_text": prereq_graph_text},
        )

        if main_copy is None:
            self.skipTest(f"main_copy not importable: {MAIN_COPY_IMPORT_ERROR}")
        helpers = {
            "rewrite_inline_prereqs": main_copy._rewrite_inline_prereqs_from_graph,
        }

        final_response, meta = run_finalizer_stage(
            plan=plan,
            executor_result=executor_result,
            conversation_history=None,
            llm=finalizer_llm,
            validation_helpers=helpers,
        )
        # The LLM said "(prerequisites: None)"; the rewrite pass must have
        # overwritten it with the verified value from the graph.
        self.assertIn("CMPSC 131", final_response)
        self.assertNotIn("CMPSC 132 (prerequisites: None)", final_response)

    def test_finalizer_stage_skips_rewrite_when_no_helper_provided(self):
        """Backward-compat: if validation_helpers omits the helper, the
        finalizer must still work (no crash)."""
        from orchestration_pipeline import (
            ExecutorResult,
            OrchestrationPlan,
            run_finalizer_stage,
        )

        finalizer_llm = FakeLLM(responses=['{"final_response":"hello","finalizer_notes":""}'])
        plan = OrchestrationPlan(
            original_question="q",
            normalized_question="q",
            corrected_question="q",
            resolved_user_need="x",
            domain="general_academic_chat",
            deliverable="direct_answer",
            plan_scope="direct_answer",
            semester_horizon=0,
            use_tools=False,
            executor_path="general_chat",
            allowed_tools=[],
        )
        executor_result = ExecutorResult(
            response="Draft.",
            tools_used=[],
            metadata={},
        )
        final_response, _ = run_finalizer_stage(
            plan=plan,
            executor_result=executor_result,
            conversation_history=None,
            llm=finalizer_llm,
            validation_helpers={},  # no helper
        )
        self.assertIn("hello", final_response)

    def test_finalizer_stage_skips_rewrite_when_no_prereq_graph_text(self):
        """If the executor never ran the prereq graph (e.g. policy lookup),
        the rewrite is a no-op and the finalizer's output is untouched."""
        from orchestration_pipeline import (
            ExecutorResult,
            OrchestrationPlan,
            run_finalizer_stage,
        )

        finalizer_llm = FakeLLM(
            responses=['{"final_response":"Drop deadline is Friday.","finalizer_notes":""}']
        )
        plan = OrchestrationPlan(
            original_question="when is drop deadline",
            normalized_question="when is drop deadline",
            corrected_question="when is drop deadline",
            resolved_user_need="x",
            domain="policy",
            deliverable="policy_lookup",
            plan_scope="direct_answer",
            semester_horizon=0,
            use_tools=True,
            executor_path="policy_lookup",
            allowed_tools=["policy_retriever"],
        )
        executor_result = ExecutorResult(
            response="Draft.",
            tools_used=["policy_retriever"],
            metadata={},  # no prereq_graph_text
        )
        if main_copy is None:
            self.skipTest(f"main_copy not importable: {MAIN_COPY_IMPORT_ERROR}")
        helpers = {
            "rewrite_inline_prereqs": main_copy._rewrite_inline_prereqs_from_graph,
        }
        final_response, _ = run_finalizer_stage(
            plan=plan,
            executor_result=executor_result,
            conversation_history=None,
            llm=finalizer_llm,
            validation_helpers=helpers,
        )
        self.assertIn("Drop deadline is Friday", final_response)


class NoAnswerFoundGatekeeperTests(unittest.TestCase):
    """Locks in the fix for the production bug where a real Art + Econ
    double-major plan was silently thrown away because the synthesizer
    correctly said 'I don't have X for the option track' inside an
    otherwise complete answer.

    The gatekeeper exists to detect TOTAL give-ups (short
    "I couldn't find anything" responses) so we can fall through to
    web-search backstop. It must NOT fire on real, structured plans that
    happen to acknowledge a missing data point — that's the whole reason
    we want the LLM to be honest about gaps in the first place.
    """

    def test_short_giveup_response_still_caught(self):
        if main_copy is None:
            self.skipTest(f"main_copy not importable: {MAIN_COPY_IMPORT_ERROR}")
        self.assertTrue(
            main_copy._response_indicates_no_answer_found(
                "I don't have any information on that."
            )
        )

    def test_empty_response_caught(self):
        if main_copy is None:
            self.skipTest(f"main_copy not importable: {MAIN_COPY_IMPORT_ERROR}")
        self.assertTrue(main_copy._response_indicates_no_answer_found(""))
        self.assertTrue(main_copy._response_indicates_no_answer_found("   "))

    def test_response_with_semester_block_kept_even_with_phrase(self):
        """The exact production failure: a real plan with a semester block
        that mentions 'I don't have' for one missing piece. Must NOT be
        treated as a give-up."""
        if main_copy is None:
            self.skipTest(f"main_copy not importable: {MAIN_COPY_IMPORT_ERROR}")
        response = (
            "Here is a draft 8-semester plan for a double major in Art "
            "and Economics.\n\n"
            "Semester 1:\n"
            "- ART 110: Drawing I\n"
            "- ECON 102: Microeconomics\n"
            "Total Credits: 15\n\n"
            "Note: I don't have the option-track data for the Art BFA, "
            "so confirm with your advisor."
        )
        self.assertFalse(main_copy._response_indicates_no_answer_found(response))

    def test_response_with_multiple_course_codes_kept(self):
        """Two or more PSU course codes signals a real, content-bearing
        response — keep it even if the phrase appears."""
        if main_copy is None:
            self.skipTest(f"main_copy not importable: {MAIN_COPY_IMPORT_ERROR}")
        response = (
            "Suggested first-year courses: ENGL 15, MATH 140, ECON 102, "
            "ART 110. I don't have the official advising notes for this "
            "specific combination."
        )
        self.assertFalse(main_copy._response_indicates_no_answer_found(response))

    def test_long_response_kept_even_with_phrase(self):
        """A long response (>800 chars) is clearly not a one-line giveup
        regardless of which phrase it contains."""
        if main_copy is None:
            self.skipTest(f"main_copy not importable: {MAIN_COPY_IMPORT_ERROR}")
        response = (
            "Here's a detailed walkthrough of the double-major path. "
            * 30
            + " I don't have the option-track names handy."
        )
        self.assertGreater(len(response), 800)
        self.assertFalse(main_copy._response_indicates_no_answer_found(response))

    def test_short_response_with_phrase_and_no_codes_caught(self):
        """A short response that's just a 'no answer' message — still
        treated as a giveup."""
        if main_copy is None:
            self.skipTest(f"main_copy not importable: {MAIN_COPY_IMPORT_ERROR}")
        response = (
            "Sorry — I couldn't find anything matching that question in "
            "our records."
        )
        self.assertTrue(main_copy._response_indicates_no_answer_found(response))


class GraphBuilderTwoDigitPrereqTests(unittest.TestCase):
    """Locks in the 2-digit course-code fix in graphExperiment.

    The graph builder's prereq-text parser used to require exactly 3 digits
    (`\\d{3}`), which silently dropped every 2-digit prereq at graph-build
    time. That's the deepest layer of the "prerequisites: None for courses
    that have prereqs" bug — by the time the LLM and the executor see the
    graph, the 2-digit prereqs are already gone. These tests prove the
    extractor now keeps both shapes.
    """

    def test_extractor_keeps_two_digit_codes(self):
        try:
            import graphExperiment
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"graphExperiment not importable: {exc}")
        # Real shape from the PSU CSV for MATH 140's prereq_data.
        sample = (
            "Enforced Prerequisite at Enrollment: MATH 22 AND MATH 26 OR "
            "MATH 26 AND satisfactory placement OR MATH 40 OR MATH 41"
        )
        prereqs, concurrents, _, _ = graphExperiment.extract_prereq_and_concurrent(sample)
        # All four 2-digit gateways must survive parsing.
        for code in ("MATH 22", "MATH 26", "MATH 40", "MATH 41"):
            self.assertIn(code, prereqs, f"2-digit prereq {code} dropped by extractor")

    def test_extractor_still_keeps_three_digit_codes(self):
        try:
            import graphExperiment
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"graphExperiment not importable: {exc}")
        sample = "Enforced Prerequisite at Enrollment: CMPSC 121 AND MATH 140"
        prereqs, _, _, _ = graphExperiment.extract_prereq_and_concurrent(sample)
        self.assertIn("CMPSC 121", prereqs)
        self.assertIn("MATH 140", prereqs)

    def test_parse_course_codes_from_requirements_keeps_two_digit_codes(self):
        try:
            import graphExperiment
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"graphExperiment not importable: {exc}")
        s = "ART 11, ART 110, ENGL 15, ENGL 202D, MATH 140"
        codes = graphExperiment.parse_course_codes_from_requirements(s)
        for code in ("ART 11", "ART 110", "ENGL 15", "ENGL 202D", "MATH 140"):
            self.assertIn(code, codes)


if __name__ == "__main__":
    unittest.main()
