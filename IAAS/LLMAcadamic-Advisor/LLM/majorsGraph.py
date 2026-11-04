from __future__ import annotations

import hashlib
import io
import os
import re

import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from fuzzywuzzy import fuzz

matplotlib.use("Agg")

from graduationFlowchart import format_planned_course_label, generate_semester_plan


class MajorGraph:
    """Generate a semester-plan prerequisite graph for a major (with on-disk cache)."""

    # Bump this when graph visuals change in a way that should invalidate all caches.
    _GRAPH_CODE_VERSION = "v2"

    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.major_names = self.df["Major Name"].tolist()
        self._csv_path = csv_path
        self._cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "major_graphs")
        os.makedirs(self._cache_dir, exist_ok=True)

    # ----- caching helpers -------------------------------------------------
    def _cache_version_token(self) -> str:
        """Build a short token that changes when the CSVs or the planner/graph code changes.
        Any edit to the data or logic invalidates the cache automatically."""
        parts = [self._GRAPH_CODE_VERSION]
        here = os.path.dirname(os.path.abspath(__file__))
        files_for_version = [
            self._csv_path,
            os.path.join(here, "nith_courses.csv"),
            os.path.join(here, "nith_majors.csv"),
            os.path.join(here, "processed_psu_courses.csv"),
            os.path.join(here, "graduationFlowchart.py"),
            os.path.join(here, "majorsGraph.py"),
        ]
        for fp in files_for_version:
            try:
                if os.path.exists(fp):
                    parts.append(f"{os.path.basename(fp)}:{int(os.path.getmtime(fp))}")
            except Exception:
                pass
        digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:10]
        return digest

    def _cache_path_for(self, major_name: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9]+", "_", major_name).strip("_") or "major"
        return os.path.join(self._cache_dir, f"{safe}.{self._cache_version_token()}.png")

    def _read_cache(self, path: str) -> io.BytesIO | None:
        try:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                with open(path, "rb") as f:
                    return io.BytesIO(f.read())
        except Exception:
            return None
        return None

    def _write_cache(self, path: str, buf: io.BytesIO) -> None:
        try:
            buf.seek(0)
            data = buf.read()
            tmp = path + ".tmp"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
            buf.seek(0)
        except Exception:
            pass

    def _prune_stale_cache(self) -> None:
        """Drop cached PNGs whose version token no longer matches the current one."""
        try:
            current = self._cache_version_token()
            for name in os.listdir(self._cache_dir):
                if not name.endswith(".png"):
                    continue
                # Expected filename: <major>.<version>.png
                parts = name.rsplit(".", 2)
                if len(parts) == 3 and parts[1] != current:
                    try:
                        os.remove(os.path.join(self._cache_dir, name))
                    except Exception:
                        pass
        except Exception:
            pass

    def _find_matching_major(self, query: str) -> str | None:
        if not query:
            return None
        query = query.lower().strip()
        best_match = None
        best_score = 0
        
        # First, try exact substring match (case-insensitive)
        for name in self.major_names:
            name_lower = name.lower()
            if query in name_lower or name_lower in query:
                # Extract the major name part (before comma or parentheses)
                major_part = name_lower.split(',')[0].split('(')[0].strip()
                query_clean = query.split(',')[0].split('(')[0].strip()
                if query_clean == major_part or query_clean in major_part or major_part in query_clean:
                    return name
        
        # If no exact match, try fuzzy matching with multiple strategies
        for name in self.major_names:
            name_lower = name.lower()
            # Extract just the major name (before comma)
            major_part = name_lower.split(',')[0].strip()
            query_clean = query.split(',')[0].strip()
            
            # Try multiple scoring methods
            score1 = fuzz.partial_ratio(query, name_lower)
            score2 = fuzz.partial_ratio(query_clean, major_part)
            score3 = fuzz.ratio(query_clean, major_part)
            score4 = fuzz.token_sort_ratio(query_clean, major_part)
            
            # Use the best score
            score = max(score1, score2, score3, score4)
            
            if score > best_score:
                best_score = score
                best_match = name
        
        # Lower threshold for better matching (was 60, now 50)
        if best_score < 50:
            return None
        return best_match

    def _display_course(self, course: str) -> str:
        return format_planned_course_label(course)

    def _has_meaningful_value(self, value) -> bool:
        if value is None:
            return False
        try:
            if pd.isna(value):
                return False
        except Exception:
            pass
        text = str(value).strip()
        if not text:
            return False
        if text.lower() in {"nan", "none", "n/a", "na"}:
            return False
        return True

    def _normalize_option_name(self, raw_name: str) -> str:
        name = re.sub(r"\s+", " ", str(raw_name or "").strip())
        if not name:
            return ""
        if "option" not in name.lower():
            name = f"{name} Option"
        return name

    def _get_major_options(self, major_name: str) -> list[str]:
        rows = self.df[self.df["Major Name"] == major_name]
        if rows.empty:
            return []
        row = rows.iloc[0]

        options: list[str] = []
        for col in self.df.columns:
            col_lower = col.lower()
            if "option courses" in col_lower or "option summary" in col_lower:
                if not self._has_meaningful_value(row.get(col)):
                    continue
                option_base = col.replace(" Option Courses", "").replace(" Option Summary", "").strip()
                option_name = self._normalize_option_name(option_base)
                if option_name and option_name not in options:
                    options.append(option_name)

        # Fallback to the generic "Options" text column when present.
        if not options and "Options" in row.index and self._has_meaningful_value(row.get("Options")):
            pieces = re.split(r",|/|;|\band\b", str(row.get("Options")))
            for p in pieces:
                cleaned = self._normalize_option_name(p)
                if cleaned and cleaned not in options:
                    options.append(cleaned)

        return options

    def _build_freshman_plan(self, actual_major_name: str, option_name: str = "General Option") -> dict:
        csv_dir = os.path.dirname(__file__)
        courses_csv = os.path.join(csv_dir, "nith_courses.csv")
        majors_csv = os.path.join(csv_dir, "nith_majors.csv")
        if not os.path.exists(courses_csv):
            courses_csv = os.path.join(csv_dir, "..", "LLM", "nith_courses.csv")
        if not os.path.exists(courses_csv):
            courses_csv = os.path.join(csv_dir, "processed_psu_courses.csv")
            if not os.path.exists(courses_csv):
                courses_csv = os.path.join(csv_dir, "..", "LLM", "processed_psu_courses.csv")

        if not os.path.exists(majors_csv):
            majors_csv = os.path.join(csv_dir, "..", "LLM", "nith_majors.csv")
        if not os.path.exists(majors_csv):
            majors_csv = os.path.join(csv_dir, "all_abington_majors_combined(5).csv")
            if not os.path.exists(majors_csv):
                majors_csv = os.path.join(csv_dir, "..", "LLM", "all_abington_majors_combined(5).csv")

        # Mirror the no-transcript freshman planning logic used by chat.
        # Use appropriate credits for NIT Hamirpur (160) vs others (120)
        is_nith = "NIT Hamirpur" in actual_major_name or "B.Tech" in actual_major_name
        credits_needed = 160 if is_nith else 120
        max_cr = 20 if is_nith else 16
        min_cr = 16 if is_nith else 12

        return generate_semester_plan(
            major_name=actual_major_name,
            completed_courses=[],
            unmet_requirements=[],
            courses_csv_path=courses_csv,
            majors_csv_path=majors_csv,
            option_name=option_name,
            credits_needed=credits_needed,
            max_semesters=8,
            max_credits_per_sem=max_cr,
            min_credits_per_sem=min_cr,
        )

    def _draw_plan_graph(
        self,
        actual_major_name: str,
        plan: dict,
        option_name: str = "General Option",
        include_title: bool = True,
    ) -> io.BytesIO:
        semesters = sorted(plan.get("semesters", []), key=lambda s: s.get("semester", 0))
        if not semesters:
            raise ValueError("No semesters were generated for this major.")

        G = nx.DiGraph()
        semester_courses = {}
        for sem_data in semesters:
            sem_num = int(sem_data.get("semester", 0))
            courses = sem_data.get("courses", []) or []
            semester_courses[sem_num] = courses
            for idx, course in enumerate(courses):
                G.add_node(course, semester=sem_num, order=idx)

        prereq_map = plan.get("prerequisites", {}) or {}
        for course, prereqs in prereq_map.items():
            if course not in G:
                continue
            for prereq in prereqs or []:
                if prereq in G:
                    G.add_edge(prereq, course)

        max_courses_in_sem = max((len(v) for v in semester_courses.values()), default=1)
        fig_w = max(16, len(semester_courses) * 3.2)
        fig_h = max(10, max_courses_in_sem * 0.9 + 4)
        plt.figure(figsize=(fig_w, fig_h))

        pos = {}
        for sem_num, courses in semester_courses.items():
            for idx, course in enumerate(courses):
                # Left-to-right by semester, top-to-bottom within semester
                pos[course] = (sem_num * 3.0, -idx * 1.1)

        color_map = plt.cm.get_cmap("tab20")
        node_colors = [color_map((G.nodes[n]["semester"] - 1) % 20) for n in G.nodes()]

        nx.draw_networkx_edges(
            G, pos,
            alpha=0.45,
            arrows=True,
            arrowsize=12,
            edge_color="#6b7280",
            width=1.2,
            connectionstyle="arc3,rad=0.05",
        )
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_shape="s",
            node_size=2500,
            linewidths=1.2,
            edgecolors="#1f2937",
        )
        labels = {n: self._display_course(n) for n in G.nodes()}
        nx.draw_networkx_labels(
            G, pos, labels=labels,
            font_size=9.5,
            font_weight="bold",
            font_color="#0b0f19",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.84, pad=0.22),
        )

        for sem_num in sorted(semester_courses.keys()):
            x = sem_num * 3.0
            plt.text(
                x,
                1.2,
                f"Semester {sem_num}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="#0f172a",
            )

        if include_title:
            plt.title(
                (
                    f"{actual_major_name} — {option_name}\n"
                    "Freshman-start 8-semester graduation roadmap "
                    "(arrows indicate prerequisite flow)"
                ),
                fontsize=13,
                fontweight="bold",
            )
        plt.axis("off")
        plt.tight_layout()

        out = io.BytesIO()
        plt.savefig(out, format="png", dpi=170, bbox_inches="tight")
        plt.close()
        out.seek(0)
        return out

    def _draw_all_option_graphs(self, actual_major_name: str, options: list[str]) -> io.BytesIO:
        num_options = len(options)
        fig, axes = plt.subplots(num_options, 1, figsize=(24, 12 * num_options))
        if num_options == 1:
            axes = [axes]

        for idx, option_name in enumerate(options):
            plan = self._build_freshman_plan(actual_major_name, option_name=option_name)
            graph_buf = self._draw_plan_graph(
                actual_major_name,
                plan,
                option_name=option_name,
                include_title=False,
            )
            graph_buf.seek(0)
            img = mpimg.imread(graph_buf, format="png")
            axes[idx].imshow(img)
            axes[idx].axis("off")
            axes[idx].set_title(
                (
                    f"{actual_major_name} - {option_name}\n"
                    "Freshman-start 8-semester plan | arrows show prerequisites"
                ),
                fontsize=14,
                fontweight="bold",
                pad=12,
            )

        plt.tight_layout()
        out = io.BytesIO()
        plt.savefig(out, format="png", dpi=150, bbox_inches="tight")
        plt.close()
        out.seek(0)
        return out

    def draw_major_graph(self, major_name: str, use_cache: bool = True) -> io.BytesIO:
        actual_major_name = self._find_matching_major(major_name)
        if not actual_major_name:
            raise ValueError(f"Could not find a matching major for '{major_name}'. Please try a different name.")

        cache_path = self._cache_path_for(actual_major_name)
        if use_cache:
            self._prune_stale_cache()
            cached = self._read_cache(cache_path)
            if cached is not None:
                return cached

        options = self._get_major_options(actual_major_name)
        if options:
            buf = self._draw_all_option_graphs(actual_major_name, options)
        else:
            plan = self._build_freshman_plan(actual_major_name, option_name="General Option")
            buf = self._draw_plan_graph(actual_major_name, plan, option_name="General Option")

        if use_cache:
            self._write_cache(cache_path, buf)
        return buf

    def clear_cache(self) -> int:
        """Delete all cached major graph PNGs. Returns count removed."""
        n = 0
        try:
            for name in os.listdir(self._cache_dir):
                if name.endswith(".png"):
                    try:
                        os.remove(os.path.join(self._cache_dir, name))
                        n += 1
                    except Exception:
                        pass
        except Exception:
            pass
        return n

