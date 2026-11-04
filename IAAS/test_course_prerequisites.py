import sys
import unittest
from pathlib import Path


LLM_DIR = Path(__file__).resolve().parent / "LLMAcadamic-Advisor" / "LLM"
if str(LLM_DIR) not in sys.path:
    sys.path.insert(0, str(LLM_DIR))

import graphExperiment  # noqa: E402
import graduationFlowchart  # noqa: E402


class CoursePrerequisiteTests(unittest.TestCase):
    def test_suffixed_prerequisite_is_not_duplicated_as_concurrent(self):
        text = graphExperiment.get_course_prerequisites("CMPSC 488", graphExperiment.course_graph)

        self.assertIn("Enforced prerequisite at enrollment: CMPSC 487W", text)
        self.assertIn("Enforced concurrent at enrollment: None", text)
        self.assertNotIn("Enforced concurrent at enrollment (can be taken together): CMPSC 487", text)

    def test_base_only_extracted_code_resolves_to_unique_catalog_suffix(self):
        catalog_index = graphExperiment.build_course_catalog_index(["CMPSC 487W", "CMPSC 488", "MATH 311M"])

        enforced, concurrent = graphExperiment.parse_prerequisite_fields(
            "",
            "['CMPSC 487']",
            catalog_codes=catalog_index,
        )

        self.assertEqual(enforced, ["CMPSC 487W"])
        self.assertEqual(concurrent, [])

        enforced, concurrent = graphExperiment.parse_prerequisite_fields(
            "",
            "['MATH 311']",
            catalog_codes=catalog_index,
        )

        self.assertEqual(enforced, ["MATH 311M"])
        self.assertEqual(concurrent, [])

    def test_direct_prerequisite_data_preserves_suffix_and_nonbreaking_space(self):
        catalog_index = graphExperiment.build_course_catalog_index(["CMPSC 487W", "CMPSC 488"])

        enforced, concurrent = graphExperiment.parse_prerequisite_fields(
            "['CMPSC\\xa0487W']",
            "['CMPSC 487']",
            catalog_codes=catalog_index,
        )

        self.assertEqual(enforced, ["CMPSC 487W"])
        self.assertEqual(concurrent, [])

    def test_parse_course_codes_from_requirements_preserves_suffixes(self):
        codes = graphExperiment.parse_course_codes_from_requirements(
            "ART 11, ART 110, ENGL 15, ENGL 202D, MATH 311M, MATH 140"
        )

        for code in ("ART 11", "ART 110", "ENGL 15", "ENGL 202D", "MATH 311M", "MATH 140"):
            self.assertIn(code, codes)

    def test_standard_freshman_plan_includes_first_year_seminar(self):
        courses_csv = LLM_DIR / "processed_psu_courses.csv"
        majors_csv = LLM_DIR / "all_abington_majors_combined(5).csv"

        plan = graduationFlowchart.generate_semester_plan(
            major_name="Computer Science, B.S. (Abington)",
            completed_courses=[],
            unmet_requirements=[],
            courses_csv_path=str(courses_csv),
            majors_csv_path=str(majors_csv),
            credits_needed=120,
            max_semesters=8,
            max_credits_per_sem=18,
            min_credits_per_sem=15,
        )

        first_semester = next(s for s in plan["semesters"] if s["semester"] == 1)
        self.assertIn("PSU 1", first_semester["courses"])
        self.assertEqual(plan["prerequisites"].get("PSU 1"), [])
        self.assertEqual(plan["course_titles"].get("PSU 1"), "First-Year Seminar Abington")

    def test_wordy_gened_tokens_render_as_catalog_codes(self):
        self.assertEqual(
            graduationFlowchart.format_planned_course_label("GENED__HEALTH_AND_WELLNESS_GHW__1"),
            "Gen Ed (GHW - Health and Wellness)",
        )
        self.assertEqual(
            graduationFlowchart.format_planned_course_label("GENED__HUMANITIES_GH__1"),
            "Gen Ed (GH - Humanities)",
        )


if __name__ == "__main__":
    unittest.main()
