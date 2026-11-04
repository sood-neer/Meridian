import pandas as pd
import networkx as nx
import re
import sys
import os
import ast
import matplotlib.pyplot as plt


COURSE_CODE_RE = re.compile(r'\b([A-Z]{2,8}(?:/[A-Z]{2,8})*)\s*(\d{1,3})([A-Z]?)\b', re.IGNORECASE)

# Use built-in layout (no pygraphviz/graphviz system dependency required)
def _get_graph_layout(G):
    """Get node positions for drawing. Uses spring_layout (no external deps)."""
    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        return graphviz_layout(G, prog="dot")
    except ImportError:
        return nx.spring_layout(G, k=2, iterations=50)

# Robust text cleaning function
def clean_text(text):
    if pd.isna(text):
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.replace('\xa0', ' ').replace('xa0', ' ').replace('\u00a0', ' ')
    text = re.sub(r'[^A-Za-z0-9\s:,\.()\[\]/]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def normalize_course_code(code):
    """Return a canonical PSU course code while preserving suffix letters."""
    if not isinstance(code, str):
        code = str(code or "")
    text = code.replace('\xa0', ' ').replace('\u00a0', ' ').upper().strip()
    text = re.sub(r'\s+', ' ', text)
    match = COURSE_CODE_RE.search(text)
    if not match:
        return ""
    subject, number, suffix = match.groups()
    return f"{subject.upper()} {number}{(suffix or '').upper()}"


def _course_base_key(code):
    normalized = normalize_course_code(code)
    match = COURSE_CODE_RE.fullmatch(normalized)
    if not match:
        return None
    subject, number, _suffix = match.groups()
    return subject.upper(), number


def _build_catalog_index(catalog_codes):
    if (
        isinstance(catalog_codes, tuple)
        and len(catalog_codes) == 2
        and isinstance(catalog_codes[0], set)
        and isinstance(catalog_codes[1], dict)
    ):
        return catalog_codes
    exact = {normalize_course_code(c) for c in (catalog_codes or [])}
    exact.discard("")
    by_base = {}
    for code in exact:
        base = _course_base_key(code)
        if base:
            by_base.setdefault(base, []).append(code)
    for codes in by_base.values():
        codes.sort()
    return exact, by_base


def build_course_catalog_index(catalog_codes):
    return _build_catalog_index(catalog_codes)


def _resolve_catalog_code(code, catalog_codes=None):
    """Map a base-only extracted code to its unique catalog suffix variant.

    The source CSV sometimes has lossy extracted values like ``CMPSC 487`` even
    though the real catalog course is only ``CMPSC 487W``. If there is no exact
    base course and exactly one suffixed catalog match, use that real course.
    When multiple variants exist, keep the base code instead of guessing.
    """
    normalized = normalize_course_code(code)
    if not normalized:
        return ""
    exact, by_base = _build_catalog_index(catalog_codes)
    if not exact or normalized in exact:
        return normalized
    match = COURSE_CODE_RE.fullmatch(normalized)
    if not match:
        return normalized
    subject, number, suffix = match.groups()
    if suffix:
        return normalized
    candidates = by_base.get((subject.upper(), number), [])
    suffixed_candidates = [c for c in candidates if COURSE_CODE_RE.fullmatch(c).group(3)]
    if normalized not in exact and len(suffixed_candidates) == 1:
        return suffixed_candidates[0]
    return normalized


def _extract_course_codes(text, catalog_codes=None):
    cleaned = clean_text(text).upper()
    codes = []
    seen = set()
    for match in COURSE_CODE_RE.finditer(cleaned):
        subject, number, suffix = match.groups()
        code = _resolve_catalog_code(f"{subject} {number}{suffix or ''}", catalog_codes)
        if code and code not in seen:
            seen.add(code)
            codes.append(code)
    return codes


def _literal_list_text(value):
    if value is None or pd.isna(value):
        return ""
    if not isinstance(value, str):
        return str(value)
    raw = value.strip()
    if not raw or raw.lower() == "nan":
        return ""
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw
    if isinstance(parsed, (list, tuple, set)):
        return " ".join(str(item) for item in parsed)
    return str(parsed)


def parse_prerequisite_fields(raw_prereq_data, raw_course_numbers="", catalog_codes=None):
    """Parse enforced and concurrent prerequisites from CSV fields.

    ``prerequisite_data`` is the higher-fidelity source because it preserves
    suffixes and explicit concurrent markers. ``prerequisite_course_numbers`` is
    a lossy extracted fallback only; it must not be used to invent concurrent
    requirements by diffing against the direct field.
    """
    direct_text = _literal_list_text(raw_prereq_data)
    enforced, concurrent, _concurrent_logic, _prereq_logic = extract_prereq_and_concurrent(
        direct_text,
        catalog_codes=catalog_codes,
    )
    if enforced or concurrent:
        return enforced, concurrent

    fallback_text = _literal_list_text(raw_course_numbers)
    return _extract_course_codes(fallback_text, catalog_codes), []


def format_course_prerequisite_summary(row, catalog_codes=None):
    enforced, concurrent = parse_prerequisite_fields(
        row.get("prerequisite_data", ""),
        row.get("prerequisite_course_numbers", ""),
        catalog_codes=catalog_codes,
    )
    parts = []
    if enforced:
        parts.append(f"Prerequisites: {', '.join(sorted(enforced))}.")
    if concurrent:
        parts.append(f"Concurrent courses: {', '.join(sorted(concurrent))}.")
    return " ".join(parts)

# Extract prerequisites and concurrent with logical relationships
def extract_prereq_and_concurrent(text, catalog_codes=None):
    prereqs, concurrents, concurrent_logic, prereq_logic = [], [], [], []
    if not text:
        return prereqs, concurrents, concurrent_logic, prereq_logic

    text = clean_text(text).upper()

    if 'ENFORCED CONCURRENT AT ENROLLMENT:' in text:
        prereq_text, concurrent_text = text.split('ENFORCED CONCURRENT AT ENROLLMENT:')
    elif 'CONCURRENT COURSES:' in text:
        prereq_text, concurrent_text = text.split('CONCURRENT COURSES:')
    elif 'OR CONCURRENT:' in text or ' OR CONCURRENT ' in text:
        prereq_text, concurrent_text = re.split(r'OR CONCURRENT[: ]', text, maxsplit=1)
    else:
        prereq_text, concurrent_text = text, ""

    # Handle prereq logic (supporting OR)
    prereq_or_groups = re.split(r'\s+OR\s+', prereq_text)
    for group in prereq_or_groups:
        matches = _extract_course_codes(group, catalog_codes)
        if matches:
            if 'AND' in group:
                prereqs.extend(matches)
                prereq_logic.append(' and '.join(matches))
            elif len(matches) > 1:
                prereqs.extend(matches)
                prereq_logic.append(' or '.join(matches))
            else:
                prereqs.extend(matches)
                prereq_logic.append(matches[0])

    # Handle concurrent logic (supporting OR and AND)
    grouped = re.split(r'\s+OR\s+', concurrent_text)
    for g in grouped:
        if 'AND' in g:
            and_group = _extract_course_codes(g, catalog_codes)
            if and_group:
                concurrents.extend(and_group)
                concurrent_logic.append(' and '.join(and_group))
        else:
            or_course = _extract_course_codes(g, catalog_codes)
            if or_course:
                concurrents.extend(or_course)
                concurrent_logic.append(or_course[0])

    concurrent_set = set(concurrents)
    prereqs = [p for p in dict.fromkeys(prereqs) if p not in concurrent_set]
    return prereqs, list(dict.fromkeys(concurrents)), concurrent_logic, prereq_logic

def parse_course_codes_from_requirements(requirements_str):
    """Parse course codes from a requirements string."""
    if not requirements_str or pd.isna(requirements_str):
        return []

    return _extract_course_codes(requirements_str)


def get_major_course_codes(major_name):
    """Get all course codes associated with a major from the major requirements CSV."""
    try:
        df = pd.read_csv("all_abington_majors_combined(5).csv")
        # Find the major row
        major_row = df[df['Major Name'].str.contains(major_name, case=False, na=False, regex=False)]

        if major_row.empty:
            return None

        # Extract course codes from prescribed_courses and additional_courses
        course_codes = set()

        # Process prescribed courses
        prescribed = major_row['Prescribed Courses'].iloc[0]
        if not pd.isna(prescribed):
            course_codes.update(
                parse_course_codes_from_requirements(prescribed))

        # Process additional courses if they exist
        additional = major_row['Additional Courses'].iloc[0]
        if not pd.isna(additional):
            course_codes.update(
                parse_course_codes_from_requirements(additional))

        return list(course_codes) if course_codes else None
    except Exception as e:
        print(f"Error reading major requirements: {e}")
        return None

# Build course graph accurately from CSV
def build_course_graph(csv_file):
    df = pd.read_csv(csv_file)
    df.fillna("", inplace=True)
    G = nx.DiGraph()
    catalog_codes = [row["course_number"] for _, row in df.iterrows()]
    catalog_index = _build_catalog_index(catalog_codes)

    for _, row in df.iterrows():
        course = normalize_course_code(row["course_number"])
        prereq_data = clean_text(row["prerequisite_data"])

        G.add_node(course)

        prereqs, concurrents, _, _ = extract_prereq_and_concurrent(
            prereq_data,
            catalog_codes=catalog_index,
        )

        for prereq in prereqs:
            G.add_edge(prereq, course, type="prereq")

        for concurrent in concurrents:
            G.add_edge(concurrent, course, type="concurrent")

    return G

# Get course prerequisites with formatted logic

def get_course_prerequisites(course_code, G):
    course_code = normalize_course_code(course_code)
    if course_code not in G:
        return f"No course named '{course_code}' found."

    prereqs = [u for u, v, d in G.in_edges(course_code, data=True) if d.get("type") == "prereq"]
    concurrents = [u for u, v, d in G.in_edges(course_code, data=True) if d.get("type") == "concurrent"]

    df = pd.read_csv(csv_file_path)
    catalog_codes = [str(c) for c in df["course_number"].dropna().tolist()]
    catalog_index = _build_catalog_index(catalog_codes)
    row = df[df["course_number"].str.upper().str.strip() == course_code.upper()]

    enforced_prereqs = []
    concurrent_options = []

    if not row.empty:
        enforced_prereqs, concurrent_options = parse_prerequisite_fields(
            row.iloc[0].get("prerequisite_data", ""),
            row.iloc[0].get("prerequisite_course_numbers", ""),
            catalog_codes=catalog_index,
        )

        # Fallback to graph edges if lists are empty
        if not enforced_prereqs and prereqs:
            enforced_prereqs = prereqs
        if not concurrent_options and concurrents:
            concurrent_options = concurrents
    else:
        enforced_prereqs = prereqs
        concurrent_options = concurrents

    result = f"Prerequisites for {course_code}:\n"
    if enforced_prereqs:
        result += f"- Enforced prerequisite at enrollment: {', '.join(sorted(enforced_prereqs))}\n"
    else:
        result += "- Enforced prerequisite at enrollment: None\n"

    if concurrent_options:
        result += f"- Enforced concurrent at enrollment (can be taken together): {', '.join(sorted(concurrent_options))}\n"
    else:
        result += "- Enforced concurrent at enrollment: None\n"

    return result

# Visualize the course graph clearly
def visualize_graph(G):
    plt.figure(figsize=(16, 12))
    pos = _get_graph_layout(G)

    node_shapes = {node: ("d" if "OR_" in node else "o") for node in G.nodes()}
    node_colors = ["lightgreen" if shape == "d" else "lightblue" for shape in node_shapes.values()]

    for shape in set(node_shapes.values()):
        shaped_nodes = [node for node in G.nodes() if node_shapes[node] == shape]
        nx.draw_networkx_nodes(G, pos, nodelist=shaped_nodes,
                               node_shape=shape, node_size=2500,
                               node_color="lightgreen" if shape == "d" else "lightblue",
                               edgecolors="black")

    prereq_edges = [(u, v) for u, v, d in G.edges(data=True) if d["type"] == "prereq"]
    concurrent_edges = [(u, v) for u, v, d in G.edges(data=True) if d["type"] == "concurrent"]

    nx.draw_networkx_edges(G, pos, edgelist=prereq_edges, arrowsize=20, edge_color="black")
    nx.draw_networkx_edges(G, pos, edgelist=concurrent_edges, arrowsize=20,
                           edge_color="red", style="dashed")

    nx.draw_networkx_labels(G, pos, {node: G.nodes[node].get("label", node) for node in G.nodes()},
                            font_size=10, font_weight="bold")

    plt.title("Course Requirements Graph (Solid: Prerequisites, Dashed Red: Concurrent)")
    plt.axis("off")
    plt.show()

# Example usage - Use relative path that works on any system
import os
csv_file_path = os.path.join(os.path.dirname(__file__), "processed_psu_courses.csv")
course_graph = build_course_graph(csv_file_path)


'''
# Test cases
print(get_course_prerequisites("MATH 414", course_graph))
print(get_course_prerequisites("CMPSC 442", course_graph))
print(get_course_prerequisites("MATH 140", course_graph))
print(get_course_prerequisites("PHYS 212", course_graph))
print(get_course_prerequisites("MATH 141", course_graph))
print(get_course_prerequisites("CMPSC 462", course_graph))

# Visualize graph
# visualize_graph(course_graph)



# Print graph info
print("Number of nodes:", course_graph.number_of_nodes())
print("Number of edges:", course_graph.number_of_edges())
print("Sample edges:", list(course_graph.edges(data=True))[:100])

# Print first 5 dictionaries representing nodes and their edges
graph_dict = {node: list(course_graph.successors(node)) for node in list(course_graph.nodes())[:100]}
print("First 5 dictionaries:", graph_dict)


# Function to print hierarchy recursively
def print_hierarchy(G, node, level=0, visited=None, prefix=""):
    if visited is None:
        visited = set()

    if node in visited:
        return  # Avoid cycles

    visited.add(node)

    indent = "  " * level
    print(f"{indent}{prefix}{node}")

    successors = sorted(G.successors(node))
    for i, succ in enumerate(successors):
        new_prefix = "├── " if i < len(successors) - 1 else "└── "
        print_hierarchy(G, succ, level + 1, visited, new_prefix)

# Find root nodes (no prerequisites)
def find_roots(G):
    return [node for node in G.nodes() if G.in_degree(node) == 0]

roots = find_roots(course_graph)
print("Roots:", roots)

print("Course Prerequisite Hierarchy:")
for root in sorted(roots):
    print_hierarchy(course_graph, root)

# Visualize graph

def visualize_graph(G):
    plt.figure(figsize=(14, 10))
    pos = _get_graph_layout(G)

    # Define node colors and shapes
    node_colors = ["lightgreen" if "OR_" in node else "lightblue" for node in G.nodes()]
    node_shapes = {node: "d" if "OR_" in node else "o" for node in G.nodes()}

    for shape in set(node_shapes.values()):
        shaped_nodes = [node for node in G.nodes() if node_shapes[node] == shape]
        node_shape = "d" if shape == "d" else "o"
        nx.draw_networkx_nodes(G, pos, nodelist=shaped_nodes, node_size=3000 if shape == "o" else 2000,
                               node_color="lightblue" if shape == "o" else "lightgreen", node_shape=node_shape,
                               edgecolors="black")

    # Define edges by type
    prereq_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("type") == "prereq"]
    concurrent_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("type") == "concurrent"]

    nx.draw_networkx_edges(G, pos, edgelist=prereq_edges, arrowstyle="-|>", arrowsize=15,
                          edge_color="black", connectionstyle="arc3,rad=0.2")
    nx.draw_networkx_edges(G, pos, edgelist=concurrent_edges, style="dashed", edge_color="red",
                          arrowstyle="-|>", arrowsize=15, connectionstyle="arc3,rad=0.2")

    node_labels = {node: G.nodes[node].get("label", node) for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_weight="bold")

    plt.title("Course Graph: Solid=Prereq, Dashed=Concurrent")
    plt.axis("off")
    plt.show()

visualize_graph(course_graph)


'''
