"""
graduationFlowchart.py (v4 – STRICT PREREQUISITES)

Fixes:
1) A course may ONLY be scheduled after all prereqs are completed in PRIOR semesters
2) No same-semester prereq violations (CMPSC 131 → 132 fixed)
3) No impossible semester-1 placements (MATH 401, MATH 410, DS 220 fixed)
"""

from __future__ import annotations

import ast
import io
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import pandas as pd
import networkx as nx


COURSE_TOKEN_RE = re.compile(r"\b([A-Z]{2,8}(?:/[A-Z]{2,8})*)[-\s]+(\d{1,3}[A-Z]?)\b")
GENED_CREDITS_RE = re.compile(r"([A-Za-z \-\(\)]+):\s*(\d+)\s*credits", re.IGNORECASE)

GENED_CATEGORY_LABELS = {
    "GA": "Arts",
    "GH": "Humanities",
    "GN": "Natural Sciences",
    "GQ": "Quantification",
    "GS": "Social and Behavioral Sciences",
    "GWS": "Writing and Speaking",
    "GHW": "Health and Wellness",
}


def format_planned_course_label(course: str) -> str:
    """Convert internal planner tokens to student-friendly labels."""
    if not isinstance(course, str):
        return str(course)

    if course.startswith("GENED__"):
        parts = course.split("__")
        key = parts[1].upper() if len(parts) > 1 else "GENED"
        if key in {"ELECTIVE", "GENED"}:
            return "Gen Ed Elective"

        normalized_key = key.replace("_", "/")
        if normalized_key in GENED_CATEGORY_LABELS:
            return f"Gen Ed ({normalized_key} - {GENED_CATEGORY_LABELS[normalized_key]})"

        if "/" in normalized_key:
            pieces = [p.strip().upper() for p in normalized_key.split("/") if p.strip()]
            if pieces and all(re.fullmatch(r"G[A-Z]{1,3}", p) for p in pieces):
                desc = "/".join(GENED_CATEGORY_LABELS.get(p, p) for p in pieces)
                return f"Gen Ed ({'/'.join(pieces)} - {desc})"
            gened_codes = [p for p in pieces if p in GENED_CATEGORY_LABELS]
            if gened_codes:
                code = gened_codes[-1]
                return f"Gen Ed ({code} - {GENED_CATEGORY_LABELS[code]})"

        if re.fullmatch(r"G[A-Z]{1,3}", normalized_key):
            return f"Gen Ed ({normalized_key})"

        return f"Gen Ed ({normalized_key.title()})"

    if course.startswith("FREE__") or course.startswith("ELECTIVE__FREE__"):
        return "Free Elective"

    if course.startswith("OPT__"):
        return "Option Elective"

    return course


def _course_level(c: str) -> int:
    """Return the 3-digit catalog level for a course code.

    Handles both plain codes (``CMPSC 132``) and codes with letter suffixes used
    by Penn State to tag writing/honors/etc. (``CMPSC 487W``, ``ENGL 202C``,
    ``DS 440W``). A strict ``\\b(\\d{1,3})\\b`` boundary would fail on the
    suffixed form because ``\\d\\b`` requires a non-word char after the digits,
    and a letter is still a word char — so ``487W`` never matched and the
    function used to return 9999 for every W/C/H/M-suffixed course.
    """
    m = re.search(r"(?<![A-Z0-9])(\d{1,3})(?=[A-Z]?\b|\s|$)", c)
    return int(m.group(1)) if m else 9999


def _min_sem_by_level(course: str) -> int:
    """
    Universal course-level floor: a course numbered in a given range cannot be scheduled
    earlier than a reasonable typical semester for that level. This catches courses whose
    prereqs are recorded as non-course text (e.g., IST 495 "Prior approval of proposed
    assignment by instructor") and would otherwise land in semester 1.

    - Placeholder buckets (GENED__, OPT__, FREE__, ELECTIVE__) are unconstrained.
    - level < 300 (100s/200s): sem 1+ (freshman/sophomore intro)
    - 300 <= level < 400: sem 3+ (sophomore/junior)
    - 400 <= level < 500: sem 5+ (junior/senior)
    - level >= 500: sem 7+ (senior capstone)
    """
    if not isinstance(course, str):
        return 1
    if course.startswith(("GENED__", "OPT__", "FREE__", "ELECTIVE__")):
        return 1
    lvl = _course_level(course)
    if lvl == 9999:
        return 1
    if lvl < 300:
        return 1
    if lvl < 400:
        return 3
    if lvl < 500:
        return 5
    return 7


def _normalize_course_code(code: str) -> str:
    """Normalize a course code: strip, uppercase, collapse whitespace.
    Preserves both NITH hyphenated style (CS-111) and PSU space style (CMPSC 131).
    Converts 'MA111' -> 'MA-111' for NITH-style short dept codes."""
    code = re.sub(r"\s+", " ", code.strip().upper())
    # Rehydrate NITH codes that lost their hyphen: 'MA111' -> 'MA-111'
    code = re.sub(r"^([A-Z]{2,4})(\d{3})$", r"\1-\2", code)
    return code


def _parse_credits(value, default: int = 0) -> int:
    """
    Parse credit values that may be ranges like "82–85" or "90-110".
    Uses the max numeric value found; falls back to default if none.
    """
    if value is None:
        return default
    text = str(value)
    nums = [int(n) for n in re.findall(r"\d+", text)]
    return max(nums) if nums else default


def _parse_course_credit_value(value, default: int = 3) -> int:
    text = str(value or "")
    credit_match = re.search(r"(\d+(?:\s*[-–]\s*\d+)?)\s+Credits?", text, re.IGNORECASE)
    if credit_match:
        return _parse_credits(credit_match.group(1), default=default)
    return default


def _clean_course_title(value: str) -> str:
    title = str(value or "").replace("\n", " ").replace("\t", " ")
    title = re.sub(r"\s+", " ", title).strip()
    title = re.sub(r"\s+\d+(?:\s*[-–]\s*\d+)?\s+Credits?.*$", "", title, flags=re.IGNORECASE)
    return title.strip()


def _build_course_metadata(courses_csv_path: str) -> Tuple[Dict[str, str], Dict[str, int]]:
    titles: Dict[str, str] = {}
    credits: Dict[str, int] = {}
    try:
        df = pd.read_csv(courses_csv_path)
    except Exception:
        return titles, credits
    for _, row in df.iterrows():
        code = _normalize_course_code(str(row.get("course_number", "") or ""))
        if not code:
            continue
        title = _clean_course_title(row.get("course_name", ""))
        if title:
            titles[code] = title
        credits[code] = _parse_course_credit_value(row.get("course_name", ""), default=3)
    return titles, credits


def _first_year_seminar_code(course_titles: Dict[str, str]) -> Optional[str]:
    for code in ("PSU 1", "PSU 101"):
        if code in course_titles:
            return code
    return None


def load_prereq_map(courses_csv_path: str) -> Dict[str, List[str]]:
    df = pd.read_csv(courses_csv_path)
    prereq_map: Dict[str, List[str]] = {}
    
    # Build a mapping of base course codes (without suffixes) to actual course codes
    # e.g., "MATH 311" -> ["MATH 311M", "MATH 311W"]
    base_to_actual: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        course = str(row.get("course_number", "")).strip()
        # Extract base course code (e.g., "MATH 311" from "MATH 311M")
        base_match = re.match(r'([A-Z]{2,})\s+(\d{1,3})', course)
        if base_match:
            base_code = f"{base_match.group(1)} {base_match.group(2)}"
            if base_code not in base_to_actual:
                base_to_actual[base_code] = []
            base_to_actual[base_code].append(course)

    for _, row in df.iterrows():
        course = str(row.get("course_number", "")).strip()

        # Prefer `prerequisite_data` (direct raw prereq expression) over
        # `prerequisite_course_numbers` (which is transitively-flattened and pollutes
        # direct prereqs with prereqs-of-prereqs, causing same-semester violations
        # like scheduling CMPSC 463 in the same semester as its real prereq CMPSC 462).
        prereqs: List[str] = []
        raw_direct = row.get("prerequisite_data", None)
        if isinstance(raw_direct, str) and raw_direct.strip():
            try:
                raw_list = ast.literal_eval(raw_direct)
                if isinstance(raw_list, (list, tuple)):
                    raw_text = " ".join(str(x) for x in raw_list)
                else:
                    raw_text = str(raw_list)
            except Exception:
                raw_text = raw_direct
            for m in COURSE_TOKEN_RE.finditer(raw_text):
                prereqs.append(_normalize_course_token(m.group(1), m.group(2)))

        # Fallback: use the flattened column if the direct column gave us nothing.
        if not prereqs:
            raw = row.get("prerequisite_course_numbers", None)
            if isinstance(raw, str) and raw.strip():
                try:
                    parsed = ast.literal_eval(raw)
                    if isinstance(parsed, (list, tuple)):
                        prereqs = [str(x).strip() for x in parsed]
                except Exception:
                    pass
        
        # Expand prerequisites: if a prereq is a base code (e.g., "MATH 311"),
        # check if any actual courses match (e.g., "MATH 311M", "MATH 311W")
        expanded_prereqs = []
        for p in prereqs:
            expanded_prereqs.append(p)  # Keep original
            # Also check if this is a base code that maps to actual courses
            if p in base_to_actual:
                # If the base code exists, we consider the prerequisite satisfied
                # if any of the actual courses are in the plan
                # For now, just add the base code as a valid prerequisite
                pass  # We'll handle this in the scheduling logic
        
        prereq_map[course] = prereqs
        
        # Also add base code entries for matching (e.g., "MATH 311" for "MATH 311M")
        base_match = re.match(r'([A-Z]{2,})\s+(\d{1,3})', course)
        if base_match:
            base_code = f"{base_match.group(1)} {base_match.group(2)}"
            # If base code doesn't exist, create it with the same prerequisites
            # This allows "MATH 311" prerequisite to match "MATH 311M" or "MATH 311W"
            if base_code not in prereq_map:
                prereq_map[base_code] = prereqs.copy()
            elif base_code != course:
                # If base code already exists and is different from course,
                # merge prerequisites (union of all variants' prerequisites)
                existing_prereqs = set(prereq_map[base_code])
                existing_prereqs.update(prereqs)
                prereq_map[base_code] = list(existing_prereqs)

    return prereq_map


def add_all_prerequisites_to_plan(
    planned_courses: List[str],
    prereq_map: Dict[str, List[str]],
    max_depth: int = 10
) -> List[str]:
    """
    CRITICAL: Recursively add ALL prerequisites for all courses in the plan.
    If a course requires a prerequisite, that prerequisite MUST be in the plan!
    This ensures prerequisites from other departments are included (e.g., WMNST 104 for AMST 104).
    """
    planned_set = {_normalize_course_code(c) for c in planned_courses}
    new_courses = list(planned_courses)
    added_this_iteration = True
    iterations = 0
    
    # Create normalized lookup for prereq_map
    norm_to_original: Dict[str, List[str]] = {}
    for orig_course in prereq_map.keys():
        orig_norm = _normalize_course_code(orig_course)
        if orig_norm not in norm_to_original:
            norm_to_original[orig_norm] = []
        norm_to_original[orig_norm].append(orig_course)
    
    while added_this_iteration and iterations < max_depth:
        iterations += 1
        added_this_iteration = False
        courses_to_check = list(new_courses)
        
        for course in courses_to_check:
            c_norm = _normalize_course_code(course)
            
            # Get prerequisites for this course
            prereqs = []
            if c_norm in norm_to_original:
                for orig_course in norm_to_original[c_norm]:
                    prereqs.extend(prereq_map.get(orig_course, []))
            else:
                prereqs = prereq_map.get(c_norm, [])
            
            # Also try base code matching
            if not prereqs:
                base_match = re.match(r"([A-Z]{2,})\s+(\d{1,3})", c_norm)
                if base_match:
                    base_code = f"{base_match.group(1)} {base_match.group(2)}"
                    for orig_course in prereq_map.keys():
                        orig_base_match = re.match(r"([A-Z]{2,})\s+(\d{1,3})", orig_course)
                        if orig_base_match and f"{orig_base_match.group(1)} {orig_base_match.group(2)}" == base_code:
                            prereqs.extend(prereq_map.get(orig_course, []))
            
            # Add prerequisites that aren't already in the plan
            for prereq in prereqs:
                prereq_norm = _normalize_course_code(prereq)
                if prereq_norm not in planned_set:
                    # Check if we already have a base code match
                    base_match = re.match(r"([A-Z]{2,})\s+(\d{1,3})", prereq_norm)
                    already_has_base = False
                    if base_match:
                        prereq_base = f"{base_match.group(1)} {base_match.group(2)}"
                        for existing_course in planned_set:
                            existing_base_match = re.match(r"([A-Z]{2,})\s+(\d{1,3})", existing_course)
                            if existing_base_match:
                                existing_base = f"{existing_base_match.group(1)} {existing_base_match.group(2)}"
                                if prereq_base == existing_base:
                                    already_has_base = True
                                    break
                    
                    if not already_has_base:
                        new_courses.append(prereq)
                        planned_set.add(prereq_norm)
                        added_this_iteration = True
    
    return new_courses


def _rebalance_schedule_in_place(
    scheduled: Dict[str, int],
    is_available_strict,
    max_credits_per_sem: int,
    min_credits_per_sem: int,
) -> None:
    """Even out credit load across the used semesters after the greedy pass.

    Moves courses from the most-loaded early semester to the least-loaded later
    semester, one at a time, as long as:
      - the destination has real capacity (credits + 3 <= max),
      - the moved course's prereqs remain satisfied at the destination sem
        (``is_available_strict`` returns True when the course is hypothetically
        at that sem),
      - moving it doesn't push total credits at the source below
        ``min_credits_per_sem`` when we could avoid that (soft guard).

    The pass is deliberately conservative: it only runs O(len(courses) *
    num_sems) iterations and stops when no beneficial move is available.
    Graduation timing (the last used sem) never extends.
    """
    if not scheduled:
        return

    def _sem_credits(target_sem: int) -> int:
        return sum(3 for _c, _s in scheduled.items() if _s == target_sem)

    last_sem = max(scheduled.values())
    if last_sem <= 1:
        return

    # Cap each pass to avoid pathological loops.
    for _ in range(len(scheduled) * 2):
        # Identify heaviest and lightest semester within [1, last_sem].
        loads = {s: _sem_credits(s) for s in range(1, last_sem + 1)}
        heaviest_sem = max(loads, key=lambda s: (loads[s], -s))
        lightest_sem = min(
            (s for s in loads if s > 1),
            key=lambda s: (loads[s], s),
            default=None,
        )
        if lightest_sem is None:
            return
        # Stop if already balanced within one course (3 credits).
        if loads[heaviest_sem] - loads[lightest_sem] <= 3:
            return
        # Only move forward in time: source must be strictly before destination.
        if heaviest_sem >= lightest_sem:
            return
        if loads[lightest_sem] + 3 > max_credits_per_sem:
            return

        # Pick a candidate from heaviest_sem that can legally move to
        # lightest_sem. Prefer the course that is most "flexible" (higher
        # level, fewer downstream blockers). We temporarily detach the
        # candidate from its current slot when checking availability.
        candidates = [c for c, s in scheduled.items() if s == heaviest_sem]
        candidates.sort(key=lambda c: (-_course_level(c), c))

        moved = False
        for cand in candidates:
            original_sem = scheduled[cand]
            # Temporarily remove so is_available_strict doesn't see it as
            # "same-sem or later" when evaluating its own prereqs.
            del scheduled[cand]
            ok = is_available_strict(cand, lightest_sem)
            scheduled[cand] = original_sem
            if not ok:
                continue
            # Ensure no course scheduled between heaviest_sem+1 and
            # lightest_sem lists this candidate as a prereq that must be
            # earlier. We check by temporarily moving and seeing if any
            # intermediate course becomes un-satisfiable.
            scheduled[cand] = lightest_sem
            any_broken = False
            for other_c, other_s in scheduled.items():
                if other_c == cand:
                    continue
                if other_s <= heaviest_sem:
                    continue
                if other_s > lightest_sem:
                    continue
                # Temporarily unset to force a fresh availability check.
                stash = scheduled.pop(other_c)
                good = is_available_strict(other_c, stash)
                scheduled[other_c] = stash
                if not good:
                    any_broken = True
                    break
            if any_broken:
                scheduled[cand] = original_sem
                continue
            moved = True
            break

        if not moved:
            return


def schedule_mixed_by_prereqs(
    planned_courses: List[str],
    prereq_map: Dict[str, List[str]],
    course_category: Dict[str, str],
    max_semesters: int = 8,
    max_credits_per_sem: int = 18,
    min_credits_per_sem: int = 15,
    completed_courses: Optional[List[str]] = None,
    apply_level_floor: bool = True,
) -> Tuple[Dict[str, int], nx.DiGraph]:
    """
    ``apply_level_floor``: when True, 400-level courses can't start before sem 5,
    300-level before sem 3, etc. This is the right behavior for a freshman-start
    roadmap. It must be OFF for transcript-based plans where the student is
    already a junior/senior, because their "Sem 1" is already late in their
    program and their remaining courses are nearly all 400-level.
    """

    # Normalize all planned courses
    planned_courses_norm = [_normalize_course_code(c) for c in planned_courses]
    planned_set = set(planned_courses_norm)
    remaining = set(planned_courses_norm)

    course_category = {
        _normalize_course_code(k): v for k, v in course_category.items()
    }
    
    # Create mapping from normalized to original for lookups
    norm_to_original_course = {_normalize_course_code(c): c for c in planned_courses}

    completed_set: set = set()
    completed_base_set: set = set()
    if completed_courses:
        for c in completed_courses:
            if not c:
                continue
            c_norm = _normalize_course_code(c)
            completed_set.add(c_norm)
            base_match = re.match(r"([A-Z]{2,})\s+(\d{1,3})", c_norm)
            if base_match:
                completed_base_set.add(
                    f"{base_match.group(1)} {base_match.group(2)}"
                )

    # Build prereq graph restricted to plan
    G = nx.DiGraph()
    for c in planned_courses_norm:
        G.add_node(c)
    
    # Create lookup maps for efficient prerequisite finding
    # Map: normalized course -> list of original course codes in prereq_map
    norm_to_original: Dict[str, List[str]] = {}
    for orig_course in prereq_map.keys():
        orig_norm = _normalize_course_code(orig_course)
        if orig_norm not in norm_to_original:
            norm_to_original[orig_norm] = []
        norm_to_original[orig_norm].append(orig_course)
    
    # Also create base code -> list of courses mapping for matching
    base_to_courses: Dict[str, List[str]] = {}
    for orig_course in prereq_map.keys():
        base_match = re.match(r"([A-Z]{2,})\s+(\d{1,3})", orig_course)
        if base_match:
            base_code = f"{base_match.group(1)} {base_match.group(2)}"
            if base_code not in base_to_courses:
                base_to_courses[base_code] = []
            base_to_courses[base_code].append(orig_course)
    
    for c in planned_courses_norm:
        # Get prerequisites for this course (check normalized version)
        prereqs = []
        
        # Strategy 1: Direct normalized lookup
        if c in norm_to_original:
            # Get prerequisites from all matching original course codes
            for orig_course in norm_to_original[c]:
                prereqs.extend(prereq_map.get(orig_course, []))
        
        # Strategy 2: Direct lookup (in case key is already normalized)
        if not prereqs:
            prereqs = prereq_map.get(c, [])
        
        # Strategy 3: Base code matching (e.g., "AMST 100" matches "AMST 100Y")
        if not prereqs:
            base_match = re.match(r"([A-Z]{2,})\s+(\d{1,3})", c)
            if base_match:
                base_code = f"{base_match.group(1)} {base_match.group(2)}"
                if base_code in base_to_courses:
                    for orig_course in base_to_courses[base_code]:
                        prereqs.extend(prereq_map.get(orig_course, []))
        
        # Remove duplicates while preserving order
        seen_prereqs = set()
        unique_prereqs = []
        for p in prereqs:
            p_norm = _normalize_course_code(p)
            if p_norm not in seen_prereqs:
                seen_prereqs.add(p_norm)
                unique_prereqs.append(p)
        
        for p in unique_prereqs:
            p_norm = _normalize_course_code(p)
            # Check if prerequisite is in plan (normalized) - EXACT MATCH
            if p_norm in planned_set:
                # Only add edge if it doesn't create a cycle (prerequisite should come before)
                if not G.has_edge(c, p_norm):  # Avoid reverse edge
                    G.add_edge(p_norm, c)
            else:
                # Also check base code match (e.g., "MATH 311" matches "MATH 311M")
                base_match = re.match(r"([A-Z]{2,})\s+(\d{1,3})", p_norm)
                if base_match:
                    prereq_base = f"{base_match.group(1)} {base_match.group(2)}"
                    for planned_course in planned_set:
                        planned_base_match = re.match(r"([A-Z]{2,})\s+(\d{1,3})", planned_course)
                        if planned_base_match:
                            planned_base = f"{planned_base_match.group(1)} {planned_base_match.group(2)}"
                            if prereq_base == planned_base:
                                # Only add edge if it doesn't create a cycle
                                if not G.has_edge(c, planned_course):  # Avoid reverse edge
                                    G.add_edge(planned_course, c)
                                break

    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Prerequisite cycle detected")

    scheduled: Dict[str, int] = {}

    def _prereq_alternatives(course: str) -> List[str]:
        """Return the flat list of prereq codes for a course (already OR-alternatives as stored in the CSV)."""
        return prereq_map.get(course, []) or []

    def _planned_match(prereq_norm: str, prereq_base: Optional[str]) -> Optional[str]:
        """Return the planned-course code that matches this prereq (exact or base-code), else None."""
        if prereq_norm in planned_set:
            return prereq_norm
        if prereq_base:
            for planned_course in planned_set:
                pm = re.match(r"([A-Z]{2,})\s+(\d{1,3})", planned_course)
                if pm and f"{pm.group(1)} {pm.group(2)}" == prereq_base:
                    return planned_course
        return None

    def _prereq_satisfiable_statically(course: str) -> bool:
        """True iff at least one prereq alternative is in the plan or completed. No-prereq courses are satisfiable."""
        alts = _prereq_alternatives(course)
        if not alts:
            return True
        for p in alts:
            p_norm = _normalize_course_code(p)
            bm = re.match(r"([A-Z]{2,})\s+(\d{1,3})", p_norm)
            p_base = f"{bm.group(1)} {bm.group(2)}" if bm else None
            if p_norm in completed_set or (p_base and p_base in completed_base_set):
                return True
            if _planned_match(p_norm, p_base) is not None:
                return True
        return False

    def is_available_strict(course: str, current_sem: int) -> bool:
        """
        Prereq rule (hybrid AND/OR) + universal course-level floor:
        - Course-level floor: a 400-level course cannot start in sem 1 even if the CSV
          lists no course prereqs (catches IST 495-style capstones).
        - Every listed prereq that IS in the plan must be scheduled in a strictly earlier
          semester. This catches same-semester violations (e.g., CMPSC 463 needs CMPSC 462
          placed in a prior semester; it cannot share sem 4).
        - At least one alternative (completed coursework, or in-plan-scheduled-earlier) must
          actually be satisfied. This handles OR expressions where some alternatives are out
          of the plan.
        Courses with no course-code prereqs are still gated by the level floor.
        """
        if apply_level_floor and current_sem < _min_sem_by_level(course):
            return False

        alts = _prereq_alternatives(course)
        if not alts:
            return True

        any_satisfied = False
        for p in alts:
            prereq_norm = _normalize_course_code(p)
            bm = re.match(r"([A-Z]{2,})\s+(\d{1,3})", prereq_norm)
            prereq_base = f"{bm.group(1)} {bm.group(2)}" if bm else None

            if prereq_norm in completed_set or (
                prereq_base and prereq_base in completed_base_set
            ):
                any_satisfied = True
                continue

            match = _planned_match(prereq_norm, prereq_base)
            if match is None:
                # Alternative isn't in the plan at all; not a candidate for satisfaction here.
                continue

            planned_sem = scheduled.get(match)
            if planned_sem is not None and planned_sem < current_sem:
                any_satisfied = True
            else:
                # The prereq IS in the plan but is either unscheduled or scheduled in the
                # same/later semester. Block so we wait for it to land earlier first.
                return False
        return any_satisfied

    # Purge courses that are fundamentally unsatisfiable (have prereqs, but none of the
    # alternatives exist in the plan or completed coursework). These would otherwise get
    # jammed into the graph in impossible semesters.
    unsatisfiable = {c for c in planned_courses_norm if not _prereq_satisfiable_statically(c)}
    if unsatisfiable:
        for c in unsatisfiable:
            remaining.discard(c)
            planned_set.discard(c)
            if G.has_node(c):
                G.remove_node(c)

    bucket_order = ["Prescribed", "Additional", "Option", "GenEd"]

    for sem in range(1, max_semesters + 1):
        sem_credits = 0
        made_progress = True
        iterations = 0
        max_iterations = 100  # Prevent infinite loops

        while made_progress and iterations < max_iterations:
            iterations += 1
            made_progress = False

            available = [
                c for c in remaining
                if is_available_strict(c, sem)
            ]

            if not available:
                # Respect prereqs: if nothing is legitimately available this semester,
                # end the semester (even if below min credits). Previously we bypassed
                # prereq checks to meet min credits, which caused impossible placements
                # (e.g., STAT 414 in semester 2 without its prereqs satisfied).
                break

            avail_by_bucket = {b: [] for b in bucket_order}
            for c in available:
                b = course_category.get(c, "Option")
                avail_by_bucket.setdefault(b, []).append(c)

            for b in bucket_order:
                avail_by_bucket[b].sort(
                    key=lambda c: (_course_level(c), -G.out_degree(c), c)
                )

            for b in bucket_order:
                # Continue adding courses until we meet minimum OR reach configured maximum.
                if sem_credits >= max_credits_per_sem:
                    break
                if not avail_by_bucket[b]:
                    continue

                c = avail_by_bucket[b][0]
                scheduled[c] = sem
                sem_credits += 3
                remaining.remove(c)
                made_progress = True
                
                # Stop if we've reached configured maximum.
                if sem_credits >= max_credits_per_sem:
                    break

        # If below minimum credits, only backfill with honestly-available courses.
        # Never bypass prereq checks just to hit a credit floor.
        if sem_credits < min_credits_per_sem and remaining:
            additional_available = [c for c in remaining if is_available_strict(c, sem)]
            needed_credits = min_credits_per_sem - sem_credits
            courses_to_add = min(len(additional_available), needed_credits // 3)
            for c in additional_available[:courses_to_add]:
                if sem_credits >= max_credits_per_sem:
                    break
                scheduled[c] = sem
                sem_credits += 3
                remaining.remove(c)

        if not remaining:
            break

    # If there are still remaining courses after all semesters, only place them in the
    # earliest semester where their prereqs are honestly satisfied AND there is room.
    # Any course whose prereqs never become satisfiable is dropped (it is excluded from
    # the roadmap rather than shown in an impossible position).
    if remaining:
        dropped_orphans: List[str] = []
        for course in list(remaining):
            placed = False
            for sem in range(1, max_semesters + 1):
                if not is_available_strict(course, sem):
                    continue
                sem_courses = [c for c, s in scheduled.items() if s == sem]
                sem_credits = len(sem_courses) * 3
                if sem_credits < max_credits_per_sem:
                    scheduled[course] = sem
                    remaining.remove(course)
                    placed = True
                    break
            if not placed:
                dropped_orphans.append(course)
                remaining.discard(course)
                if G.has_node(course):
                    G.remove_node(course)

    # #region rebalance pass (H3)
    # After the greedy scheduler completes, earlier semesters may be packed to
    # ``max_credits_per_sem`` while the final semester has only the handful of
    # courses that were waiting on prereq chains. A 9/3/18/12 or 18/18/6 layout
    # is technically valid but not a plan a student would follow.
    #
    # This pass redistributes courses forward (never backward, so prereqs stay
    # satisfied) while honoring three invariants:
    #   (1) Every course keeps landing in a semester where its prereqs are met.
    #   (2) No semester exceeds ``max_credits_per_sem``.
    #   (3) We do not PUSH past the current ``last_sem``; graduation timing
    #       stays the same. We only even out credits within the used range.
    try:
        _rebalance_schedule_in_place(
            scheduled=scheduled,
            is_available_strict=is_available_strict,
            max_credits_per_sem=max_credits_per_sem,
            min_credits_per_sem=min_credits_per_sem,
        )
    except Exception:
        # Rebalancing is a best-effort improvement pass; if anything goes wrong
        # keep the greedy schedule rather than fail the whole request.
        pass
    # #endregion

    return scheduled, G


@dataclass
class NodeStyle:
    facecolor: str
    edgecolor: str
    textcolor: str


STYLE_MAP = {
    "Prescribed": NodeStyle("#98FB98", "#2F6B2F", "#000000"),
    "Additional": NodeStyle("#B0E0E6", "#2C6A75", "#000000"),
    "Option":     NodeStyle("#D8BFD8", "#6B3F6B", "#000000"),
    "GenEd":      NodeStyle("#FFDAB9", "#8B5A2B", "#000000"),
}


class GraduationFlowchart:
    def __init__(self, courses_csv: str, majors_csv: str):
        self.majors_df = pd.read_csv(majors_csv)
        self.prereq_map = load_prereq_map(courses_csv)
        self.courses_csv_path = courses_csv
        self.courses_df = None  # Lazy load when needed

    # --- Helpers restored from previous version ---
    def _get_major_row(self, major_name: str) -> pd.Series:
        rows = self.majors_df[self.majors_df["Major Name"] == major_name]
        if rows.empty:
            raise ValueError(f"Major '{major_name}' not found in majors CSV.")
        return rows.iloc[0]

    def _option_courses_column(self, option_name: str) -> Optional[str]:
        if not option_name:
            return None
        name = option_name.strip().lower()
        if "data science" in name:
            return "Data Science Option Courses"
        if "general" in name:
            return "General Option Courses"
        # fallback: try matching by prefix
        for col in self.majors_df.columns:
            if "option" in col.lower() and name in col.lower():
                return col
        return None

    def _parse_gened_categories(self, gened_text: str) -> List[Tuple[str, int]]:
        if not isinstance(gened_text, str):
            return []
        pairs = []
        for lbl, n in GENED_CREDITS_RE.findall(gened_text):
            try:
                pairs.append((lbl.strip(), int(n)))
            except Exception:
                continue
        return pairs

    def _make_gened_nodes(self, categories: List[Tuple[str, int]]) -> Tuple[List[str], Dict[str, str]]:
        nodes, labels = [], {}
        for (cat, credits) in categories:
            count = int(math.ceil(credits / 3))
            key = re.sub(r"[^A-Za-z0-9]+", "_", cat).strip("_").upper() or "GENED"
            for i in range(count):
                nid = f"GENED__{key}__{i+1}"
                nodes.append(nid)
                labels[nid] = f"Gen Ed: {cat}"
        return nodes, labels

    def _dominant_dept(self, courses: List[str]) -> str:
        counts: Dict[str, int] = {}
        for c in courses:
            parts = c.split()
            if not parts:
                continue
            dept = parts[0].strip()
            if dept and dept.isalpha():
                counts[dept] = counts.get(dept, 0) + 1
        if not counts:
            return "OPT"
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    def _find_department_courses(self, prescribed_courses: List[str], row: pd.Series) -> List[str]:
        """
        When option courses column contains descriptive text instead of course codes,
        find actual courses from the major's department.
        This works for ALL majors universally.
        """
        if not prescribed_courses:
            # Try to get department from additional courses or major name
            additional_expr = str(row.get("Additional Courses", "") or "")
            additional_courses = extract_all_courses(additional_expr)
            if additional_courses:
                prescribed_courses = additional_courses[:1]  # Use first additional course
        
        if not prescribed_courses:
            return []
        
        # Get the department from prescribed courses (e.g., "AMST" from "AMST 491W")
        # Try multiple courses to find a valid department
        dept = None
        for course in prescribed_courses:
            parts = course.split()
            if len(parts) >= 2:
                potential_dept = parts[0].strip()
                # Validate it looks like a department code (2-8 uppercase letters)
                if potential_dept.isalpha() and potential_dept.isupper() and 2 <= len(potential_dept) <= 8:
                    dept = potential_dept
                    break
        
        if not dept:
            return []
        
        # Load courses CSV and find courses from this department
        try:
            if not hasattr(self, 'courses_df') or self.courses_df is None:
                if os.path.exists(self.courses_csv_path):
                    self.courses_df = pd.read_csv(self.courses_csv_path)
                else:
                    return []
            
            # Find all courses from this department
            dept_courses = []
            for _, course_row in self.courses_df.iterrows():
                course_code = str(course_row.get("course_number", "")).strip()
                # Match department code at the start (e.g., "AMST 100" matches "AMST")
                if course_code and course_code.startswith(dept + " "):
                    # Normalize the course code
                    normalized = _normalize_course_code(course_code)
                    if normalized and normalized not in dept_courses:
                        dept_courses.append(normalized)
            
            # Sort by course level (lower first) for better scheduling
            dept_courses.sort(key=lambda c: (_course_level(c), c))
            
            return dept_courses
        except Exception as e:
            # Silently fail - this is a fallback mechanism
            return []

    def _select_option_courses(self, option_courses: List[str], needed_credits: int, exclude: set) -> List[str]:
        n_needed = int(math.ceil(max(0, needed_credits) / 3))
        pool = [c for c in option_courses if c not in exclude]
        pool.sort(key=lambda c: (_course_level(c), c))
        return pool[:n_needed]

    def generate_flowchart(self, major_name: str, option_name: str = "General Option") -> io.BytesIO:
        row = self._get_major_row(major_name)

        # ---- Prescribed (specific courses) ----
        prescribed_expr = str(row.get("Prescribed Courses", "") or "")
        prescribed_courses = resolve_common_or_groups(extract_all_courses(prescribed_expr))

        # ---- Additional (extract ALL courses, handle "Select X credits from" lists) ----
        additional_expr = str(row.get("Additional Courses", "") or "")
        additional_courses_all = extract_all_courses(additional_expr)
        
        # Check if this is a "Select X credits from" type requirement
        # If so, treat these as option courses, not fixed courses
        is_selective_list = "select" in additional_expr.lower() and "from" in additional_expr.lower()
        
        # Extract ALL additional courses, not just the first one
        # For "OR" groups like "ACCTG 211 or ACCTG 201 & 202", take the first option
        # But extract all individual courses mentioned
        additional_courses = []
        additional_option_courses = []  # Courses from "Select X credits from" lists
        seen = set()
        
        for course in additional_courses_all:
            if course not in seen:
                seen.add(course)
                if is_selective_list:
                    # These are option courses, not fixed requirements
                    additional_option_courses.append(course)
                else:
                    # These are fixed requirements (e.g., "ACCTG 211 or ACCTG 201 & 202")
                    additional_courses.append(course)

        fixed_courses = []
        for c in prescribed_courses + additional_courses:
            if c not in fixed_courses:
                fixed_courses.append(c)

        # ---- Option-specific (extract option courses first) ----
        opt_col = self._option_courses_column(option_name or "")
        option_course_text = str(row.get(opt_col, "") or "") if opt_col else ""
        option_course_list = extract_all_courses(option_course_text)
        
        # If no option courses found in the column (e.g., descriptive text instead of course codes),
        # try to find courses from the major's department
        if not option_course_list:
            option_course_list = self._find_department_courses(prescribed_courses, row)

        # Determine dominant dept for option placeholder label (e.g., CMPSC)
        # For Computer Science majors, ALWAYS use CMPSC for electives, not MATH
        # Check prescribed courses first to determine the major's primary department
        prescribed_depts = set([c.split()[0] for c in prescribed_courses if " " in c])
        if "CMPSC" in prescribed_depts:
            # Computer Science major - use CMPSC for electives
            dominant_dept = "CMPSC"
        elif "DS" in prescribed_depts:  # Data Science major
            dominant_dept = "DS"
        else:
            # For other majors, use the most common department from prescribed courses only
            dominant_dept = self._dominant_dept(prescribed_courses) or "OPT"

        # ---- Calculate total credits needed ----
        credits_for_major = _parse_credits(row.get("Credits for Major", 0), default=0)
        total_credits_required = _parse_credits(row.get("Total Credits Required", 0), default=120)
        credits_for_gened = _parse_credits(row.get("Credits for General Education", 0), default=45)
        
        # Calculate fixed courses credits
        approx_fixed_credits = 3 * len(fixed_courses)
        remaining_major_credits = max(0, credits_for_major - approx_fixed_credits)
        
        # CRITICAL: Calculate free electives needed
        # Free electives = Total Credits - (Major Credits + GenEd Credits)
        major_and_gened_credits = credits_for_major + credits_for_gened
        free_elective_credits = max(0, total_credits_required - major_and_gened_credits)
        
        # Get available option courses (including courses from "Select X credits from" lists)
        # Excluding fixed courses that are already in prescribed/additional
        all_option_courses = list(option_course_list) + additional_option_courses
        available_option_courses = [c for c in all_option_courses if c not in fixed_courses]
        
        # Calculate how many option courses we need to satisfy the option credit requirements
        n_option_courses_needed = int(math.ceil(remaining_major_credits / 3))
        
        # Select actual option courses - ALWAYS prioritize actual courses over placeholders
        # Sort by level (lower first) for better scheduling
        available_option_courses.sort(key=lambda c: (_course_level(c), c))
        
        # CRITICAL: Select ALL available option courses up to the number needed
        # Never create placeholders if actual courses are available
        option_selected = []
        option_placeholders = []
        
        if len(available_option_courses) > 0:
            # Select actual courses - take as many as we need, up to what's available
            option_selected = available_option_courses[:min(n_option_courses_needed, len(available_option_courses))]
            
            # Calculate remaining credits after selecting actual courses
            used_credits = 3 * len(option_selected)
            still_need = max(0, remaining_major_credits - used_credits)
            
            # Only create placeholders if we've exhausted ALL available option courses AND still need more
            if still_need > 0 and len(option_selected) >= len(available_option_courses):
                # We've used all available option courses, create placeholders for remaining credits
                count = int(math.ceil(still_need / 3))
                for i in range(count):
                    option_placeholders.append(f"OPT__{dominant_dept}__{i+1}")
        else:
            # No option courses available - create placeholders for all needed credits
            count = n_option_courses_needed
            for i in range(count):
                option_placeholders.append(f"OPT__{dominant_dept}__{i+1}")

        # ---- Gen Ed (per-category) ----
        gened_text = str(row.get("General Education Requirements", "") or "")
        categories = self._parse_gened_categories(gened_text)
        if not categories:
            total_gened = credits_for_gened
            if total_gened == 0:
                # Default to 45 credits if not specified
                total_gened = 45
            categories = [("General Education", total_gened)]
        gened_nodes, gened_labels = self._make_gened_nodes(categories)
        
        # ---- Free Electives (if Total Credits > Major + GenEd) ----
        free_elective_nodes = []
        free_elective_labels = {}
        if free_elective_credits > 0:
            n_free_electives = int(math.ceil(free_elective_credits / 3))
            for i in range(n_free_electives):
                free_node = f"FREE__ELECTIVE__{i+1}"
                free_elective_nodes.append(free_node)
                free_elective_labels[free_node] = "Free Elective"
        
        # Calculate current planned courses (BEFORE adding prerequisites)
        planned_courses_initial = fixed_courses + option_selected + option_placeholders + gened_nodes + free_elective_nodes
        
        # CRITICAL: Add ALL prerequisites for ALL courses in the plan
        # If a course requires a prerequisite, that prerequisite MUST be in the plan!
        # This ensures prerequisites from other departments are included (e.g., WMNST 104 for AMST 104)
        planned_courses = add_all_prerequisites_to_plan(planned_courses_initial, self.prereq_map)
        
        # Calculate total planned credits (including prerequisites)
        total_planned_credits = len(planned_courses) * 3
        
        # Ensure we meet the Total Credits Required
        # If we're short, add more free electives
        if total_planned_credits < total_credits_required:
            needed_credits = total_credits_required - total_planned_credits
            needed_courses = int(math.ceil(needed_credits / 3))
            start_idx = len(free_elective_nodes) + 1
            for i in range(needed_courses):
                free_node = f"FREE__ELECTIVE__{start_idx + i}"
                free_elective_nodes.append(free_node)
                free_elective_labels[free_node] = "Free Elective"
                planned_courses.append(free_node)
            # Recalculate after adding courses
            total_planned_credits = len(planned_courses) * 3
        
        # Also ensure we have enough courses to fill 7-8 semesters (minimum 12 credits per semester)
        # Target: 7 semesters * 12 credits = 84 credits minimum
        min_total_credits = 7 * 12
        if total_planned_credits < min_total_credits:
            needed_credits = min_total_credits - total_planned_credits
            needed_courses = int(math.ceil(needed_credits / 3))
            start_idx = len(free_elective_nodes) + 1
            for i in range(needed_courses):
                free_node = f"FREE__ELECTIVE__{start_idx + i}"
                free_elective_nodes.append(free_node)
                free_elective_labels[free_node] = "Free Elective"
                planned_courses.append(free_node)

        # ---- Category map for mixed scheduling ----
        # Track which courses were in initial plan vs. prerequisites added
        initial_plan_set = {_normalize_course_code(c) for c in planned_courses_initial}
        course_category: Dict[str, str] = {}
        
        for c in prescribed_courses:
            if c in planned_courses:
                course_category[_normalize_course_code(c)] = "Prescribed"
        for c in additional_courses:
            if c in planned_courses:
                course_category[_normalize_course_code(c)] = "Additional"
        for c in option_selected:
            course_category[_normalize_course_code(c)] = "Option"
        for c in option_placeholders:
            course_category[_normalize_course_code(c)] = "Option"
        for c in gened_nodes:
            course_category[_normalize_course_code(c)] = "GenEd"
        for c in free_elective_nodes:
            course_category[_normalize_course_code(c)] = "GenEd"  # Treat free electives as GenEd for scheduling
        
        # Categorize prerequisites that were added (from other departments)
        # These are required courses but from other departments - treat as "Additional"
        for c in planned_courses:
            c_norm = _normalize_course_code(c)
            if c_norm not in course_category:
                # This is a prerequisite that was added - categorize as "Additional"
                course_category[c_norm] = "Additional"

        # ---- Mixed scheduling + prereq edges ----
        scheduled, prereq_G = schedule_mixed_by_prereqs(
            planned_courses,
            self.prereq_map,
            course_category=course_category,
            max_semesters=8,
            max_credits_per_sem=18,
            min_credits_per_sem=12  # Minimum 12 credits per semester
        )

        # ---- Labels & styles ----
        # IMPORTANT: Normalize all keys to match the normalized graph nodes
        node_type: Dict[str, str] = {}
        label_map: Dict[str, str] = {}

        for c in prescribed_courses:
            if c in planned_courses:
                c_norm = _normalize_course_code(c)
                node_type[c_norm] = "Prescribed"
                label_map[c_norm] = c

        for c in additional_courses:
            if c in planned_courses:
                c_norm = _normalize_course_code(c)
                node_type[c_norm] = "Additional"
                label_map[c_norm] = c

        for c in option_selected:
            c_norm = _normalize_course_code(c)
            node_type[c_norm] = "Option"
            label_map[c_norm] = c

        # Placeholders must mention "option" and the option name
        # Format: "{dept} {option_name} Option Elective" (e.g., "CMPSC General Option Elective")
        for c in option_placeholders:
            # Placeholders are already normalized (they're strings like "OPT__AMST__1")
            node_type[c] = "Option"
            # Clean up option_name: remove redundant "Option" suffix if present
            clean_option_name = option_name.strip()
            # Remove trailing " Option" or "Option" (case insensitive)
            if clean_option_name.lower().endswith(" option"):
                clean_option_name = clean_option_name[:-7].strip()
            elif clean_option_name.lower().endswith("option"):
                clean_option_name = clean_option_name[:-6].strip()
            # Handle special case: "Options" -> use generic "Elective" label
            if clean_option_name.lower() == "options" or clean_option_name.lower() == "":
                # If it's just "Options" or empty, use a generic label but still include "Option"
                label_map[c] = f"{dominant_dept} Option Elective"
            else:
                # Capitalize properly: "General", "Data Science", etc.
                # ALWAYS include "Option" in the label
                clean_option_name = clean_option_name.title()
                label_map[c] = f"{dominant_dept} {clean_option_name} Option Elective"

        for c in gened_nodes:
            # GenEd nodes are already normalized (they're strings like "GENED__GA__1")
            node_type[c] = "GenEd"
            label_map[c] = gened_labels.get(c, "Gen Ed")
        
        for c in free_elective_nodes:
            # Free elective nodes are already normalized (they're strings like "FREE__ELECTIVE__1")
            node_type[c] = "GenEd"
            label_map[c] = free_elective_labels.get(c, "Free Elective")
        
        # Label prerequisites that were added (from other departments)
        # These are required courses but from other departments - should be visible as "Additional"
        initial_plan_set_norm = {_normalize_course_code(c) for c in planned_courses_initial}
        for c in planned_courses:
            c_norm = _normalize_course_code(c)
            if c_norm not in node_type:
                # This is a prerequisite that was added - label as "Additional" (light blue)
                node_type[c_norm] = "Additional"
                label_map[c_norm] = c

        # ---- Draw ----
        fig = self._draw_semester_chart(
            major_name=major_name,
            option_name=option_name,
            scheduled=scheduled,
            graph=prereq_G,
            node_type=node_type,
            label_map=label_map,
            max_semesters=8,
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf

    def _draw_semester_chart(
        self,
        major_name: str,
        option_name: str,
        scheduled: Dict[str, int],
        graph: nx.DiGraph,
        node_type: Dict[str, str],
        label_map: Dict[str, str],
        max_semesters: int = 8,
    ):
        sem_nodes: Dict[int, List[str]] = {i: [] for i in range(1, max_semesters + 1)}
        # Ensure all graph nodes are accounted for
        all_graph_nodes = set(graph.nodes())
        all_scheduled_nodes = set(scheduled.keys())
        
        # Add all graph nodes to sem_nodes (they should all be in scheduled, but be safe)
        for n in graph.nodes():
            sem = scheduled.get(n, max_semesters)
            if sem < 1 or sem > max_semesters:
                sem = max_semesters
            if n not in sem_nodes[sem]:
                sem_nodes[sem].append(n)
        
        # Also add any scheduled nodes that might not be in graph (shouldn't happen, but be safe)
        for n in all_scheduled_nodes:
            if n not in all_graph_nodes:
                sem = scheduled.get(n, max_semesters)
                if sem < 1 or sem > max_semesters:
                    sem = max_semesters
                if n not in sem_nodes[sem]:
                    sem_nodes[sem].append(n)

        for sem in sem_nodes:
            sem_nodes[sem].sort(key=lambda c: (_course_level(c), label_map.get(c, c)))

        col_w = 3.0
        row_h = 0.58
        box_w = 2.6
        box_h = 0.40
        x0 = 0.6
        y_top = 0.8

        max_rows = max((len(v) for v in sem_nodes.values()), default=1)
        height = max(6.0, max_rows * row_h + 2.2)
        width = (max_semesters) * col_w + 1.2

        fig, ax = plt.subplots(figsize=(width, height))
        ax.set_axis_off()

        ax.text(
            0.0, 1.02,
            f"{major_name} — {option_name}",
            fontsize=16,
            fontweight="bold",
            ha="left",
            va="bottom",
            transform=ax.transAxes,
        )

        for sem in range(1, max_semesters + 1):
            ax.text(
                x0 + (sem - 1) * col_w,
                y_top,
                f"Semester {sem}",
                fontsize=10,
                fontweight="bold",
                ha="left",
                va="bottom",
            )

        node_rects: Dict[str, Tuple[float, float, float, float]] = {}

        for sem in range(1, max_semesters + 1):
            x = x0 + (sem - 1) * col_w
            nodes = sem_nodes[sem]
            for r, n in enumerate(nodes):
                y = y_top + 0.35 + r * row_h
                node_rects[n] = (x, y, box_w, box_h)

        # draw nodes
        for n in graph.nodes():
            if n not in node_rects:
                # Skip nodes that aren't scheduled (shouldn't happen, but be safe)
                continue
            x, y, w, h = node_rects[n]
            kind = node_type.get(n, "Option")
            style = STYLE_MAP.get(kind, STYLE_MAP["Option"])

            rect = FancyBboxPatch(
                (x, y),
                w, h,
                boxstyle="round,pad=0.02,rounding_size=0.05",
                linewidth=1.2,
                edgecolor=style.edgecolor,
                facecolor=style.facecolor,
                zorder=5,  # Nodes above background but below arrows
            )
            ax.add_patch(rect)

            ax.text(
                x + w / 2.0,
                y + h / 2.0,
                label_map.get(n, n),
                fontsize=8,
                ha="center",
                va="center",
                color=style.textcolor,
                wrap=True,
            )

        # edges between in-plan nodes only
        # Draw arrows for prerequisites - make them more visible
        edges_drawn = 0
        for u, v in graph.edges():
            # Check if both nodes have rectangles (are scheduled and visible)
            if u not in node_rects or v not in node_rects:
                # Skip edges where nodes aren't visible (shouldn't happen, but be safe)
                continue
                
            ur = node_rects[u]
            vr = node_rects[v]
            
            # Calculate arrow start and end points
            # Start from right edge of prerequisite course
            start = (ur[0] + ur[2], ur[1] + ur[3] / 2.0)
            # End at left edge of dependent course
            end = (vr[0], vr[1] + vr[3] / 2.0)
            
            # Get semesters for curvature calculation
            u_sem = scheduled.get(u, 0)
            v_sem = scheduled.get(v, 0)
            
            # Calculate curvature based on semester difference
            # More curve for same semester, less for different semesters
            if u_sem == v_sem:
                # Same semester - use more curve to avoid overlap
                rad = 0.4
            else:
                # Different semesters - use moderate curve
                rad = 0.2
            
            # Draw arrow with maximum visibility
            # CRITICAL: zorder must be HIGHER than nodes (zorder=5) to be visible!
            arrow = FancyArrowPatch(
                start,
                end,
                arrowstyle="->",  # Simple arrow style
                mutation_scale=30,  # Very large arrowhead for visibility
                linewidth=3.0,  # Very thick line for visibility
                color="#000000",  # Pure black
                connectionstyle=f"arc3,rad={rad}",  # Curved connection
                zorder=10,  # ABOVE nodes so arrows are visible!
                alpha=1.0,  # Fully opaque
            )
            ax.add_patch(arrow)
            edges_drawn += 1

        ax.set_xlim(0, x0 + max_semesters * col_w + 0.5)
        ax.set_ylim(y_top + max_rows * row_h + 1.5, 0)
        return fig


# --- Helper functions restored ---
def _normalize_course_token(dept_part: str, num_part: str) -> str:
    """Produce canonical course code.
    NITH-style: short dept (2-4 chars) → 'CS-111'
    PSU-style:  long dept (5+ chars)   → 'CMPSC 131'
    """
    dept = dept_part.split("/")[0].strip().upper()
    num = num_part.strip().upper()
    if len(dept) <= 4:
        return f"{dept}-{num}"
    return f"{dept} {num}"


def extract_all_courses(expr: str) -> List[str]:
    """Extract ALL course codes found in a string (robust to '...')."""
    if not isinstance(expr, str) or not expr.strip():
        return []
    found = []
    for m in COURSE_TOKEN_RE.finditer(expr):
        found.append(_normalize_course_token(m.group(1), m.group(2)))
    out, seen = [], set()
    for c in found:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def resolve_common_or_groups(courses: List[str]) -> List[str]:
    """
    Resolve the common CMPSC intro pair:
      (CMPSC 131,132) OR (CMPSC 121,122)
    If both sets appear, keep 131/132 and drop 121/122.
    """
    s = set(courses)
    has_131_132 = ("CMPSC 131" in s) and ("CMPSC 132" in s)
    has_121_122 = ("CMPSC 121" in s) and ("CMPSC 122" in s)
    if has_131_132 and has_121_122:
        drop = {"CMPSC 121", "CMPSC 122"}
        return [c for c in courses if c not in drop]
    return courses


def generate_semester_plan(
    major_name: str,
    completed_courses: List[str],
    unmet_requirements: List[dict],
    courses_csv_path: str,
    majors_csv_path: str,
    option_name: str = "General Option",
    credits_needed: Optional[float] = None,
    max_semesters: int = 8,
    max_credits_per_sem: int = 18,
    min_credits_per_sem: int = 15,
) -> dict:
    """
    Generate a semester-by-semester plan structure (not PNG).
    Returns: {
        "semesters": [
            {"semester": 1, "courses": ["CMPSC 131", ...], "credits": 15},
            ...
        ],
        "total_credits": 120,
        "prerequisites": {"CMPSC 132": ["CMPSC 131"], ...}
    }
    """
    flowchart = GraduationFlowchart(courses_csv_path, majors_csv_path)
    course_titles, course_credits = _build_course_metadata(courses_csv_path)
    
    # Get major requirements
    row = flowchart._get_major_row(major_name)
    
    # Extract prescribed courses
    prescribed_expr = str(row.get("Prescribed Courses", "") or "")
    prescribed_courses = resolve_common_or_groups(extract_all_courses(prescribed_expr))
    
    # Extract additional courses
    additional_expr = str(row.get("Additional Courses", "") or "")
    additional_courses_all = extract_all_courses(additional_expr)
    additional_courses_all = [
        c for c in additional_courses_all
        if _normalize_course_code(c) not in {_normalize_course_code(x) for x in prescribed_courses}
    ]

    # Build a broader course pool from all major row columns.
    # Many majors store requirement-bearing courses in option/summary/supporting columns.
    non_course_columns = {
        "Major Name", "Major Code", "Total Credits Required", "Credits for Major",
        "Credits for General Education", "Options", "General Education Requirements"
    }
    broad_major_pool: List[str] = []
    for col in row.index:
        if col in non_course_columns:
            continue
        try:
            value = str(row.get(col, "") or "")
        except Exception:
            value = ""
        if not value:
            continue
        for c in extract_all_courses(value):
            if c not in broad_major_pool:
                broad_major_pool.append(c)

    fixed_courses = []
    for course in prescribed_courses:
        if course not in fixed_courses:
            fixed_courses.append(course)
    
    # Extract option courses
    opt_col = flowchart._option_courses_column(option_name or "")
    option_course_text = str(row.get(opt_col, "") or "") if opt_col else ""
    option_course_list = extract_all_courses(option_course_text)
    
    completed_set = {_normalize_course_code(c) for c in completed_courses or []}
    fixed_remaining = [
        c for c in fixed_courses if _normalize_course_code(c) not in completed_set
    ]

    # Determine dominant department for option placeholders
    # Handle both hyphenated (CS-111 → "CS") and space (CMPSC 131 → "CMPSC")
    prescribed_depts = set()
    for c in prescribed_courses:
        parts = re.split(r"[-\s]+", c.strip())
        if parts:
            prescribed_depts.add(parts[0].upper())
    if "CMPSC" in prescribed_depts:
        dominant_dept = "CMPSC"
    elif "DS" in prescribed_depts:
        dominant_dept = "DS"
    elif "CS" in prescribed_depts:
        dominant_dept = "CS"
    else:
        dominant_dept = flowchart._dominant_dept(prescribed_courses) or "OPT"

    # Option course selection based on remaining major credits
    credits_for_major = _parse_credits(row.get("Credits for Major", 0), default=0)
    major_course_set = {
        _normalize_course_code(c)
        for c in (fixed_courses + additional_courses_all + option_course_list + broad_major_pool)
    }
    completed_major_courses = [
        c for c in completed_set if c in major_course_set
    ]
    completed_major_credits = 3 * len(completed_major_courses)
    fixed_remaining_credits = 3 * len(fixed_remaining)

    # Select additional courses to help satisfy major credits (instead of taking only one).
    additional_candidates = [
        c for c in additional_courses_all
        if _normalize_course_code(c) not in completed_set and c not in fixed_remaining
    ]
    additional_candidates.sort(key=lambda c: (_course_level(c), c))
    additional_selected: List[str] = []
    if credits_for_major > 0:
        remaining_after_fixed = max(0, credits_for_major - completed_major_credits - fixed_remaining_credits)
        max_additional_count = int(math.ceil(remaining_after_fixed / 3)) if remaining_after_fixed else 0
        if max_additional_count > 0:
            additional_selected = additional_candidates[:max_additional_count]

    selected_additional_credits = 3 * len(additional_selected)
    remaining_major_credits = max(
        0, credits_for_major - completed_major_credits - fixed_remaining_credits - selected_additional_credits
    )
    n_option_courses_needed = int(math.ceil(remaining_major_credits / 3)) if remaining_major_credits else 0

    available_option_courses = [
        c for c in option_course_list
        if _normalize_course_code(c) not in completed_set and c not in fixed_courses
    ]
    available_option_courses.sort(key=lambda c: (_course_level(c), c))

    option_selected = available_option_courses[:min(n_option_courses_needed, len(available_option_courses))]
    option_placeholders = []
    if n_option_courses_needed > len(option_selected):
        placeholder_count = n_option_courses_needed - len(option_selected)
        for i in range(placeholder_count):
            option_placeholders.append(f"OPT__{dominant_dept}__{i+1}")

    # If option list is sparse/empty, use the broad major pool as additional real-course candidates.
    if n_option_courses_needed > len(option_selected):
        deficit = n_option_courses_needed - len(option_selected)
        broad_candidates = [
            c for c in broad_major_pool
            if _normalize_course_code(c) not in completed_set
            and c not in fixed_remaining
            and c not in additional_selected
            and c not in option_selected
        ]
        broad_candidates.sort(key=lambda c: (_course_level(c), c))
        if deficit > 0 and broad_candidates:
            broad_pick = broad_candidates[:deficit]
            option_selected.extend(broad_pick)
            deficit = max(0, deficit - len(broad_pick))
            if deficit > 0:
                option_placeholders.extend([f"OPT__{dominant_dept}__X{i+1}" for i in range(deficit)])

    planned_courses = fixed_remaining + additional_selected + option_selected + option_placeholders

    # GenEd placeholders to reach remaining credits (if provided)
    gened_nodes = []
    gened_labels: Dict[str, str] = {}
    gened_text = str(row.get("General Education Requirements", "") or "")
    categories = flowchart._parse_gened_categories(gened_text)
    if not categories:
        total_gened = int(row.get("Credits for General Education", 0) or 0)
        if total_gened > 0:
            categories = [("General Education", total_gened)]

    if categories:
        gened_nodes, gened_labels = flowchart._make_gened_nodes(categories)

    if credits_needed is not None:
        planned_credits = 3 * len(planned_courses)
        remaining_for_gened = max(0, credits_needed - planned_credits)
        needed_gened_count = int(math.ceil(remaining_for_gened / 3)) if remaining_for_gened else 0
        if gened_nodes:
            gened_nodes = gened_nodes[:needed_gened_count]
            gened_labels = {k: gened_labels[k] for k in gened_nodes}
            # If detailed GenEd categories are fewer than needed, backfill with generic electives.
            if len(gened_nodes) < needed_gened_count:
                shortfall = needed_gened_count - len(gened_nodes)
                extras = [f"GENED__ELECTIVE__FILL__{i+1}" for i in range(shortfall)]
                gened_nodes.extend(extras)
                for k in extras:
                    gened_labels[k] = "Gen Ed Elective"
        elif needed_gened_count > 0:
            gened_nodes = [f"GENED__ELECTIVE__{i+1}" for i in range(needed_gened_count)]
            gened_labels = {k: "Gen Ed Elective" for k in gened_nodes}

    planned_courses.extend(gened_nodes)
    
    # Load prerequisites
    prereq_map = flowchart.prereq_map
    
    # Schedule courses
    course_category: Dict[str, str] = {}
    for c in prescribed_courses:
        if c in planned_courses:
            course_category[c] = "Prescribed"
    for c in additional_selected:
        if c in planned_courses:
            course_category[c] = "Additional"
    for c in option_selected:
        course_category[c] = "Option"
    for c in option_placeholders:
        course_category[c] = "Option"
    for c in gened_nodes:
        course_category[c] = "GenEd"

    # If the student already has completed coursework, this is a transcript-based
    # remaining-plan: their "Sem 1" is already their next semester, not the first
    # freshman semester, so the freshman-year level floor does not apply.
    apply_level_floor = not bool(completed_courses)
    scheduled, graph = schedule_mixed_by_prereqs(
        planned_courses=planned_courses,
        prereq_map=prereq_map,
        course_category=course_category,
        max_semesters=max_semesters,
        max_credits_per_sem=max_credits_per_sem,
        min_credits_per_sem=min_credits_per_sem,
        completed_courses=completed_courses,
        apply_level_floor=apply_level_floor,
    )
    
    # Build semester structure
    semesters = {}
    for course, sem in scheduled.items():
        if sem not in semesters:
            semesters[sem] = []
        semesters[sem].append(course)

    # Build prerequisites dict
    prerequisites = {}
    for course in planned_courses:
        if course in prereq_map:
            prerequisites[course] = prereq_map[course]

    # Penn State Abington plans should explicitly show the first-year seminar
    # for standard freshman-start plans. It is a 1-credit catalog course (PSU 1
    # in the current course CSV; PSU 101 is supported if future data uses it).
    completed_norm = {_normalize_course_code(c) for c in completed_courses or []}
    seminar_code = _first_year_seminar_code(course_titles)
    if seminar_code and not completed_norm:
        seminar_norm = _normalize_course_code(seminar_code)
        seminar_already_present = any(
            _normalize_course_code(c) == seminar_norm
            for sem_courses in semesters.values()
            for c in sem_courses
        )
        if not seminar_already_present:
            semesters.setdefault(1, []).append(seminar_norm)
            prerequisites[seminar_norm] = []
    
    # Format as list
    semester_list = []
    for sem in sorted(semesters.keys()):
        courses_in_sem = sorted(semesters[sem])
        credits = sum(course_credits.get(_normalize_course_code(c), 3) for c in courses_in_sem)
        semester_list.append({
            "semester": sem,
            "courses": courses_in_sem,
            "credits": credits
        })
    
    # Final universal backfill: guarantee requested credits are reachable.
    # This prevents truncated plans for majors with sparse explicit course-code coverage.
    if credits_needed is not None:
        target_total = int(math.ceil(float(credits_needed) / 3.0) * 3)
        current_total = sum(s["credits"] for s in semester_list)
        if current_total < target_total:
            remaining_credits = target_total - current_total
            extra_courses_needed = int(math.ceil(remaining_credits / 3.0))
            for i in range(extra_courses_needed):
                placeholder = f"ELECTIVE__FREE__{i+1}"
                placed = False
                for sem in range(1, max_semesters + 1):
                    sem_courses = semesters.get(sem, [])
                    sem_credits = len(sem_courses) * 3
                    if sem_credits < max_credits_per_sem:
                        semesters.setdefault(sem, []).append(placeholder)
                        placed = True
                        break
                if not placed:
                    # If every semester is full, place in final semester as last resort.
                    semesters.setdefault(max_semesters, []).append(placeholder)
                prerequisites[placeholder] = []
            # Rebuild semester_list after backfill
            semester_list = []
            for sem in sorted(semesters.keys()):
                courses_in_sem = sorted(semesters[sem])
                credits = sum(course_credits.get(_normalize_course_code(c), 3) for c in courses_in_sem)
                semester_list.append({
                    "semester": sem,
                    "courses": courses_in_sem,
                    "credits": credits
                })

    return {
        "semesters": semester_list,
        "total_credits": sum(s["credits"] for s in semester_list),
        "prerequisites": prerequisites,
        "course_titles": course_titles,
        "course_credits": course_credits,
    }
