"""
obsidian_parser.py — Obsidian Vault Ingestion for the personal RAG Pipeline
=========================================================================
Parses an Obsidian vault into semantically meaningful chunks ready for
embedding. Handles:
  - Daily notes with multiple back-to-back course lectures (heading-split)
  - Course-specific markdown notes (section-split)
  - Frontmatter (YAML) metadata extraction
  - Wikilink / backlink resolution
  - Tag extraction and propagation
  - Hierarchical context inheritance (parent headings cascade into children)

Output: List[Document] where each Document carries text + rich metadata
suitable for ChromaDB / BM25 dual indexing.

Usage:
    parser = ObsidianParser("/path/to/vault")
    documents = parser.parse_all()
    # Each document has .text, .metadata (dict), .doc_id (deterministic hash)
"""

import hashlib
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml


# ─────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Document:
    """A single chunk ready for embedding + vector store insertion."""
    text: str
    metadata: dict = field(default_factory=dict)
    doc_id: str = ""

    def __post_init__(self):
        if not self.doc_id:
            # Deterministic ID from content + source for dedup
            sig = f"{self.metadata.get('source_file', '')}::{self.text[:500]}"
            self.doc_id = hashlib.sha256(sig.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {"doc_id": self.doc_id, "text": self.text, "metadata": self.metadata}


# ─────────────────────────────────────────────────────────────────────
# Constants — adjust these to match YOUR vault conventions
# ─────────────────────────────────────────────────────────────────────

# Patterns that identify a course lecture inside a daily note.
# the author's daily notes have multiple course lectures back-to-back,
# separated by H2 or H3 headings like "## CS 251 — Lecture 14" or
# "## NLP Specialization — Week 3".
# Expand this list to match your actual heading conventions.
COURSE_HEADING_PATTERNS = [
    # Course codes: "# CS 251", "## DS 223", etc.
    r"^#{1,3}\s+(CS|DS|ENGS|BSDS|ECON)\s*\d{2,3}",
    # Course abbreviations (from vault scout): "# TSF", "# NLP L2", "# RL L3", etc.
    r"^#{1,3}\s+(TSF|NLP|RL|ML|MA|AI|NA|DV|DataViz|Dataviz|DATAVIZ|ArmHist|Econ)\b",
    # Full course names: "# Statistics 2", "# Numerical Analysis L5", etc.
    r"^#{1,3}\s+(Statistics|Numerical Analysis|Machine Learning|Linear Algebra|Calculus|Time Series|Data Visualization|Armenian History|Personal Finance|Economics|Capstone|Trust|Generative AI)\b",
    # "# Lecture N" or "## Lecture N:"
    r"^#{1,3}\s+Lecture\s+\d+",
    # Course + Lecture pattern: "# RL L3", "# TSF L12", "# NLP L4"
    r"^#{1,3}\s+\w+\s+L\d+",
    # Generic: heading with dash/colon + lecture/week: "# RL — Lecture 5"
    r"^#{1,3}\s+\w+\s*[-—:]\s+(?:Lecture|Week|Session|Lab|Tutorial|Intro|Quiz|Midterm|Final|PSS|HW)",
]

# Known course code → full name mapping (from the author's portfolio)
COURSE_MAP = {
    # Original AUA codes
    "CS 251": "Machine Learning & AI",
    "CS 246": "Reinforcement Learning",
    "CS 108": "Statistics & Inference",
    "CS 102": "Linear Algebra",
    "CS 104": "Calculus / Multivariable",
    "CS 112": "Numerical Methods",
    "CS2:101": "Natural Language Processing",
    "DS 110": "Statistics & Inference",
    "DS 116": "Data Visualization",
    "DS 120": "Programming & Engineering",
    "DS 205": "Databases & Data Engineering",
    "DS 206": "Business Intelligence & Analytics",
    "DS 223": "Marketing Analytics",
    "DS 232": "Generative AI",
    "BSDS 227": "Business Analytics for Data Science",
    "CS 362": "Time Series & Forecasting",
    "ENGS 101": "Calculus I",
    # Added from vault scout
    "ECON 101": "Economics",
    "ECON": "Economics",
}

# Map course names to knowledge domains (from the portfolio's 9 domains)
DOMAIN_MAP = {
    # Core DS/NLP domains (from portfolio)
    "Natural Language Processing": "nlp",
    "Machine Learning & AI": "ml",
    "Statistics & Inference": "stats",
    "Time Series & Forecasting": "ts",
    "Business Intelligence & Analytics": "biz",
    "Programming & Engineering": "prog",
    "Data Visualization": "viz",
    "Databases & Data Engineering": "db",
    "Calculus I": "math",
    "Calculus / Multivariable": "math",
    "Linear Algebra": "math",
    "Numerical Methods": "math",
    "Generative AI": "ml",
    "Marketing Analytics": "biz",
    "Reinforcement Learning": "ml",
    # Added from vault scout
    "Intro to AI": "ml",
    "Economics": "biz",
    "Personal Finance": "biz",
    "Trust": "general",
    "Armenian History": "general",
    "Capstone": "nlp",
    # Added this phase (course-tag calibration)
    "Data Structures & Algorithms": "prog",
    "Intro to CS": "prog",
    "Intro to Business": "biz",
    "Ethics": "general",
    "Physics & Chemistry": "general",
    "Business Analytics for Data Science": "biz",
    # Added: self-study tech books (Tech Books/<bucket>/ folders)
    "Cloud & DevOps": "swe",
    "Software Architecture": "swe",
    "Web Development": "swe",
    "Data Science (Books)": "ml",
    "Python Development": "prog",
}

# Folder-name → course name lookup.
# Used as a fallback in _detect_course_from_path when the folder name
# doesn't match any heading pattern (e.g. "Statistics & Inference" has '&'
# which confuses the regex engine, and some folder names use full titles
# with dashes/spaces that the heading patterns don't cover).
FOLDER_COURSE_MAP = {
    # Exact folder name (case-insensitive) -> canonical course name
    "statistics & inference":           "Statistics & Inference",
    "statistics and inference":         "Statistics & Inference",
    "natural language processing":      "Natural Language Processing",
    "machine learning & ai":            "Machine Learning & AI",
    "machine learning and ai":          "Machine Learning & AI",
    "machine learning":                 "Machine Learning & AI",
    "reinforcement learning":           "Reinforcement Learning",
    "time series & forecasting":        "Time Series & Forecasting",
    "time series and forecasting":      "Time Series & Forecasting",
    "time series forecasting":          "Time Series & Forecasting",
    "data visualization":               "Data Visualization",
    "numerical methods":                "Numerical Methods",
    "numerical analysis":               "Numerical Methods",
    "calculus / multivariable":         "Calculus / Multivariable",
    "calculus":                         "Calculus / Multivariable",
    "linear algebra":                   "Linear Algebra",
    "databases & data engineering":     "Databases & Data Engineering",
    "databases and data engineering":   "Databases & Data Engineering",
    "databases":                        "Databases & Data Engineering",
    "business intelligence & analytics":"Business Intelligence & Analytics",
    "business intelligence":            "Business Intelligence & Analytics",
    "marketing analytics":              "Marketing Analytics",
    "intro to ai":                      "Intro to AI",
    "introduction to ai":               "Intro to AI",
    "generative ai":                    "Generative AI",
    "economics":                        "Economics",
    "personal finance":                 "Personal Finance",
    "armenian history":                 "Armenian History",
    "trust":                            "Trust",
    "capstone":                         "Capstone",
    "programming & engineering":        "Programming & Engineering",
    "programming and engineering":      "Programming & Engineering",
    # --- Current Courses abbreviations (active course folders) ---
    "nlp":                              "Natural Language Processing",
    "rl":                               "Reinforcement Learning",
    "ma":                               "Marketing Analytics",
    "genai":                            "Generative AI",
    "gen ai":                           "Generative AI",
    "tsf":                              "Time Series & Forecasting",
    "dataviz":                          "Data Visualization",
    "dv":                               "Data Visualization",
    # --- 01-Passed leaf course folders (exact names, incl. suffixes) ---
    "ai (a+)":                          "Intro to AI",
    "ai":                               "Intro to AI",
    "ds116 (fixed pdfs)":               "Data Visualization",
    "intro to cs":                      "Intro to CS",
    "introduction to computer science": "Intro to CS",
    "data structures":                  "Data Structures & Algorithms",
    "data structures and algorithms":   "Data Structures & Algorithms",
    "data structures/algorithms":       "Data Structures & Algorithms",
    "intro to business":                "Intro to Business",
    "introduction to business":         "Intro to Business",
    "ethics":                           "Ethics",
    "physics and chemistry b":          "Physics & Chemistry",
    "physics and chemistry":            "Physics & Chemistry",
    "statistics probability":           "Statistics & Inference",  # parent: all children are stats
    # AUA course codes as folder names (some students name folders by code)
    "ds 116":   "Data Visualization",
    "ds 120":   "Programming & Engineering",
    "ds 205":   "Databases & Data Engineering",
    "ds 206":   "Business Intelligence & Analytics",
    "ds 223":   "Marketing Analytics",
    "ds 227":   "Business Analytics for Data Science",
    "ds 232":   "Generative AI",
    "ds 235":   "Generative AI",
    "cs 251":   "Machine Learning & AI",
    "cs 246":   "Reinforcement Learning",
    "cs 362":   "Time Series & Forecasting",
    "cs 108":   "Statistics & Inference",
    "econ 101": "Economics",
    # Self-study tech-book subfolders under Tech Books/ (folder name -> course)
    "cloud & devops":        "Cloud & DevOps",
    "software architecture": "Software Architecture",
    "web development":       "Web Development",
    "data science":          "Data Science (Books)",
    "python development":    "Python Development",
}

# Ordered keyword fallback for the messy, variant-heavy human folder names that
# exact matching can't enumerate ("Statistics (A+)", "Stat A(Vahe) Pass",
# "Statistics B Pass", "Business Analytics for ...", truncated names, typos).
# Most-specific FIRST. Matched as a case-insensitive substring against a single
# path component, walking leaf->root so the deepest (most specific) folder wins.
COURSE_KEYWORDS = [
    ("reinforcement",         "Reinforcement Learning"),
    ("natural language",      "Natural Language Processing"),
    ("generative",            "Generative AI"),
    ("marketing analytics",   "Marketing Analytics"),
    ("business intelligence", "Business Intelligence & Analytics"),
    ("business analytics",    "Business Intelligence & Analytics"),
    ("time series",           "Time Series & Forecasting"),
    ("machine learning",      "Machine Learning & AI"),
    ("data visualization",    "Data Visualization"),
    ("dataviz",               "Data Visualization"),
    ("data structures",       "Data Structures & Algorithms"),
    ("distributed systems",   "Databases & Data Engineering"),
    ("database",              "Databases & Data Engineering"),
    ("linear algebra",        "Linear Algebra"),
    ("numerical",             "Numerical Methods"),
    ("calculus",              "Calculus / Multivariable"),
    ("probability",           "Statistics & Inference"),
    ("statistic",             "Statistics & Inference"),   # statistics / statistical
    ("personal finance",      "Personal Finance"),
    ("intro to business",     "Intro to Business"),
    ("economics",             "Economics"),
    ("ethics",                "Ethics"),
    ("physics",               "Physics & Chemistry"),
    ("chemistry",             "Physics & Chemistry"),
    ("computer science",      "Intro to CS"),
]


def detect_course_from_path(parts: list[str]) -> dict:
    """Detect course from path components. Single source of truth used by the
    PDF and notebook loaders (and the recalibration pass).

    Order:
      1. Exact FOLDER_COURSE_MAP (root->leaf). Only leaf course folders are
         mapped, so organizational parents (e.g. "Prgoramming for Data Science",
         which holds ML/AI/TSF) do NOT clobber their children.
      2. Course-code regex (CS|DS|ENGS|BSDS|ECON ###).
      3. Keyword substring fallback (leaf->root, most-specific-first) for the
         many human folder-name variants exact matching can't enumerate.

    Returns {course_code, course_name, domain}; unknown/unknown/general if none.
    """
    # 1. exact folder-name match
    for part in parts:
        name = FOLDER_COURSE_MAP.get(part.lower().strip())
        if name:
            return {"course_code": name, "course_name": name,
                    "domain": DOMAIN_MAP.get(name, "general")}
    # 2. course-code regex
    for part in parts:
        m = re.search(r"(CS|DS|ENGS|BSDS|ECON)\s*\d{2,3}", part, re.IGNORECASE)
        if m:
            code = re.sub(r"\s+", " ",
                          re.sub(r"(\D)(\d)", r"\1 \2", m.group(0).upper())).strip()
            name = COURSE_MAP.get(code, code)
            return {"course_code": code, "course_name": name,
                    "domain": DOMAIN_MAP.get(name, "general")}
    # 3. keyword fallback, deepest folder first
    for part in reversed(parts):
        low = part.lower()
        for kw, name in COURSE_KEYWORDS:
            if kw in low:
                return {"course_code": name, "course_name": name,
                        "domain": DOMAIN_MAP.get(name, "general")}
    return {"course_code": "unknown", "course_name": "unknown", "domain": "general"}

# Chunk size constraints (in characters)
MIN_CHUNK_SIZE = 200       # Skip fragments smaller than this
MAX_CHUNK_SIZE = 3000      # Split sections larger than this
OVERLAP_SIZE = 150         # Overlap between split chunks


# ─────────────────────────────────────────────────────────────────────
# Frontmatter parser
# ─────────────────────────────────────────────────────────────────────

def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Extract YAML frontmatter and return (metadata_dict, remaining_content)."""
    if not content.startswith("---"):
        return {}, content
    end = content.find("---", 3)
    if end == -1:
        return {}, content
    try:
        fm = yaml.safe_load(content[3:end])
        if not isinstance(fm, dict):
            fm = {}
    except yaml.YAMLError:
        fm = {}
    body = content[end + 3:].lstrip("\n")
    return fm, body


# ─────────────────────────────────────────────────────────────────────
# Heading tree builder
# ─────────────────────────────────────────────────────────────────────

@dataclass
class HeadingNode:
    """A node in the heading tree — represents one section of a markdown file."""
    level: int
    title: str
    content: str  # text directly under this heading (before any child heading)
    children: list = field(default_factory=list)
    line_start: int = 0


def build_heading_tree(body: str) -> list[HeadingNode]:
    """
    Parse markdown body into a tree of HeadingNode.
    Top-level content (before any heading) becomes a level-0 node.
    """
    heading_re = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    lines = body.split("\n")
    # Find all heading positions
    headings = []
    for i, line in enumerate(lines):
        m = heading_re.match(line)
        if m:
            headings.append((i, len(m.group(1)), m.group(2).strip()))

    if not headings:
        # No headings — entire body is one chunk
        return [HeadingNode(level=0, title="(untitled)", content=body.strip(), line_start=0)]

    nodes = []
    # Content before first heading
    if headings[0][0] > 0:
        pre = "\n".join(lines[:headings[0][0]]).strip()
        if pre:
            nodes.append(HeadingNode(level=0, title="(preamble)", content=pre, line_start=0))

    for idx, (line_num, level, title) in enumerate(headings):
        # Content runs from the line after this heading to the line before the next heading
        start = line_num + 1
        end = headings[idx + 1][0] if idx + 1 < len(headings) else len(lines)
        content = "\n".join(lines[start:end]).strip()
        nodes.append(HeadingNode(level=level, title=title, content=content, line_start=line_num))

    return nodes


# ─────────────────────────────────────────────────────────────────────
# Course detection from headings
# ─────────────────────────────────────────────────────────────────────

def detect_course_from_heading(heading: str) -> Optional[str]:
    """Try to extract a course code or name from a heading string.
    Returns course code/name, or '__SKIP__' for non-course headings,
    or None if unrecognized."""
    h_stripped = heading.strip()
    h_lower = h_stripped.lower()

    # ── SKIP PATTERNS (non-academic headings) ──
    # These are detected first to prevent false matches (e.g., "Self S" matching "stat")
    skip_patterns = [
        r"^self\s*s",           # Self Study (55+ headings)
        r"^mapp\b",             # Music Appreciation
        r"^nutrition",          # Nutrition and Health
        r"^bohl\b",             # Basics of Healthy Lifestyle
        r"^healthy\s+lifestyle",
        r"^hl\s+l\d",           # "HL L2" etc — Health Lifestyle lectures
        r"^opera\b",            # Music/Opera notes
        r"^music\b",
        r"^baroque\b",
        r"^todo\b",             # Task lists
        r"^to[\s-]?do",
        r"^day\s+planner",
        r"^plan\b",
        r"^routine",
        r"^library\s+schedule",
        r"^genius\s+idea",
        r"^schedule\b",
        r"^pre-?study\b",       # Pre-study sessions (not a course)
        r"^sunday",             # Day-of-week notes
        r"^saturday",
        r"^tomorrow",
        r"^before\b",
        r"^after\b",
        r"^now\b",
        r"^main\s+tasks",
        r"^current\s+(study|focus|job)",
        r"^challenges\s+of",
        r"^what\s+should",
        r"^convo\s+with",
        r"^king'?s\s+sunday",
        r"^\.\.\.$",            # Placeholder headings
        # Added from real vault data (post-run analysis)
        r"^excalidraw\s+data",  # Raw Excalidraw JSON embedded in .md files (129x)
        r"^basics\s+of\s+healthy", # BoHL full-title variant (13x)
        r"^retro.futurist",     # Design/art one-off
        r"^vaporwave",
        r"^💰",                  # Finance recalculation personal notes
        r"^seatbelt\s+safety",  # One-off assignment
        r"^appeal\s+letter",    # Personal admin document
        r"^ucom\b",             # Phone plan research
        r"^comprehensive\s+report\s+on\s+arguments", # Essay
        r"^do\s+not\s+include", # Notes to self (12x "do not include this claude")
        r"^fall\s+\d{4}\s+course\s+schedule", # Schedule files
        r"^american\s+university\s+of\s+armenia", # Boilerplate header
        r"^formal\s+explanation\s+of\s+vegetation", # One-off
        r"^problem\s+\d+:\s+inclusionary", # One-off homework
        r"^licensing\s+&\s+legal",  # One-off business doc
    ]
    for pat in skip_patterns:
        if re.match(pat, h_lower):
            return "__SKIP__"

    # ── COURSE CODE MATCH: "CS 251", "DS 110", "ECON 101", etc. ──
    code_match = re.search(r"(CS|DS|ENGS|BSDS|ECON)\s*\d{2,3}", heading, re.IGNORECASE)
    if code_match:
        code = code_match.group(0).upper()
        # Normalize to exactly one space between letters and digits
        code = re.sub(r"\s+", " ", re.sub(r"(\D)(\d)", r"\1 \2", code)).strip()
        return code

    # ── ABBREVIATION MATCH (from vault scout data) ──
    # Ordered: longest first to prevent partial matches.
    # Each entry: (pattern, canonical course name)
    abbreviations = [
        # Time Series — "TSF", "TSF L12", "Time Series Forecasting", "Time Series L9"
        (r"^tsf\b", "Time Series & Forecasting"),
        (r"^time\s+series", "Time Series & Forecasting"),

        # Data Visualization — "DataViz", "Dataviz", "DV", "DATAVIZ", "DataViz L8", "DataV"
        (r"^data\s*viz", "Data Visualization"),
        (r"^datav\b", "Data Visualization"),
        (r"^dv\b", "Data Visualization"),

        # Numerical Analysis — "Numerical Analysis", "NA PSS6", "NA HW10", "NM L2"
        (r"^numerical\s+analysis", "Numerical Methods"),
        (r"^na\s+(pss|hw|midterm|l\d|pre)", "Numerical Methods"),
        (r"^na\b", "Numerical Methods"),
        (r"^nm\s+l\d", "Numerical Methods"),

        # NLP — "NLP", "NLP L2", "NLP HW", "NLP Final Lecture", "NLP PSS", "NLP Midterm"
        (r"^nlp\b", "Natural Language Processing"),

        # Machine Learning — "ML", "ML L6", "ML FINAL", "ML group"
        (r"^ml\b", "Machine Learning & AI"),

        # Reinforcement Learning — "RL", "RL L3", "RL midterm", "RL PSS", "RL Dynamic"
        (r"^rl\b", "Reinforcement Learning"),

        # Marketing Analytics — "MA", "MA:", "MA Quiz", "MA: CLV", "MA L1"
        (r"^ma[\s:]+", "Marketing Analytics"),
        (r"^ma$", "Marketing Analytics"),

        # Statistics — "Statistics", "Statistics 2", "Statistics:", "Stat ", "Statistics 2 L5"
        (r"^statistics?\b", "Statistics & Inference"),
        (r"^stat\s+(midterm|hw|pss|mid)", "Statistics & Inference"),
        (r"^stat\b", "Statistics & Inference"),

        # Armenian History — "ArmHist", "ArmHist2", "ArmHist 2", "ARMHist2",
        #                     "Armenian History", "History", "ArmHistory"
        (r"^armhist", "Armenian History"),
        (r"^armenian\s+history", "Armenian History"),
        (r"^armhistory", "Armenian History"),
        (r"^history\b", "Armenian History"),

        # AI (Intro to AI, separate from GenAI) — "AI", "AI:", "Ai:", "AI midterm"
        (r"^ai[\s:]", "Intro to AI"),
        (r"^ai$", "Intro to AI"),
        (r"^ai\s+(?:midterm|pss|problem|recap)", "Intro to AI"),

        # Calculus — "Calculus", "Calculus:", "Calculus 3:", "CALCULUS 3", "Calculus HW"
        (r"^calculus", "Calculus / Multivariable"),

        # Linear Algebra — "Linear Algebra"
        (r"^linear\s+algebra", "Linear Algebra"),

        # Economics — "Econ", "Economics", "ECON L2", "Introduction to Economics"
        (r"^econ", "Economics"),
        (r"^introduction\s+to\s+economics", "Economics"),

        # Personal Finance — "Personal Finance", "Personal finance", "PF", "IPF"
        (r"^personal\s+finance", "Personal Finance"),
        (r"^personal\s+f\b", "Personal Finance"),
        (r"^intro\s+to\s+personal\s+finance", "Personal Finance"),
        (r"^ipf\b", "Personal Finance"),
        (r"^pf\s+(game|quiz)", "Personal Finance"),
        (r"^investment\b", "Personal Finance"),

        # Capstone — "Capstone", "Capstone:", "Capstone meeting"
        (r"^capstone", "Capstone"),
        (r"^thesis\b", "Capstone"),

        # Trust — "Trust", "Trust."
        (r"^trust\b", "Trust"),

        # Generative AI — "Generative AI", "GenAI", "Decoding in LLMs" etc.
        (r"^generative\s+ai", "Generative AI"),
        (r"^genai\b", "Generative AI"),

        # Databases
        (r"^database", "Databases & Data Engineering"),
        (r"^sql\b", "Databases & Data Engineering"),

        # BI
        (r"^business\s+intelligence", "Business Intelligence & Analytics"),
        (r"^power\s+bi", "Business Intelligence & Analytics"),
        (r"^bi\b", "Business Intelligence & Analytics"),

        # Group project (NLP)
        (r"^group\s+project", "Natural Language Processing"),
        # Filename-specific patterns (for files whose names are the only course signal)
        (r"^sub.areas\s+ml", "Machine Learning & AI"),          # Sub_Areas ML Group Project.md
        (r"^main\s+gp\s+text", "Natural Language Processing"),  # main GP text...md (GP = NLP group project)
        (r"^nlp.presentation", "Natural Language Processing"),  # NLP_presentation_CODING...md
        (r"^claude\s+pf\b", "Personal Finance"),                # Claude PF Round2-3.md
        (r"^first\s+republic\s+arm", "Armenian History"),       # First Republic Armhist Summary.md
        (r"^taxi.r\b", "Data Visualization"),                   # taxi-r-full-explanation.md

        # Programming
        (r"^python\b", "Programming & Engineering"),
        (r"^fastapi\b", "Programming & Engineering"),
    ]

    for pattern, course_name in abbreviations:
        if re.match(pattern, h_lower):
            return course_name

    return None


def resolve_course_metadata(course_id: Optional[str]) -> dict:
    """Given a course code or name, return enriched metadata."""
    if not course_id:
        return {"course_code": "unknown", "course_name": "unknown", "domain": "general"}

    # If it's already a full name (from DOMAIN_MAP keys), use it
    if course_id in DOMAIN_MAP:
        return {
            "course_code": course_id,
            "course_name": course_id,
            "domain": DOMAIN_MAP[course_id],
        }

    # Otherwise look up the code
    name = COURSE_MAP.get(course_id, course_id)
    domain = DOMAIN_MAP.get(name, "general")
    return {"course_code": course_id, "course_name": name, "domain": domain}


# ─────────────────────────────────────────────────────────────────────
# Daily note detection
# ─────────────────────────────────────────────────────────────────────

DAILY_NOTE_PATTERNS = [
    # "28.05.2026" or "28-05-2026" (DD.MM.YYYY — the author's format)
    r"^\d{2}[-_.]\d{2}[-_.]\d{4}$",
    # "2026-05-28" or "2026.05.28" (YYYY-MM-DD — standard ISO)
    r"^\d{4}[-_.]\d{2}[-_.]\d{2}$",
    # "May 28, 2026" or "28 May 2026"
    r"^\w+\s+\d{1,2},?\s+\d{4}$",
    r"^\d{1,2}\s+\w+\s+\d{4}$",
]


def is_daily_note(filepath: Path) -> bool:
    """Detect if a file is a daily note based on its filename."""
    stem = filepath.stem
    for pattern in DAILY_NOTE_PATTERNS:
        if re.match(pattern, stem):
            return True
    # Also check if it's inside a "Daily Notes" or "daily" folder
    parts = [p.lower() for p in filepath.parts]
    return any(d in parts for d in ["daily notes", "daily", "dailies", "journal",
                                     "daily_study_notes", "09 - daily_study_notes"])


def extract_date_from_filename(filepath: Path) -> Optional[str]:
    """Try to extract an ISO date from the filename."""
    stem = filepath.stem
    # DD.MM.YYYY format (the author's vault)
    ddmmyyyy = re.search(r"(\d{2})[-_.](\d{2})[-_.](\d{4})", stem)
    if ddmmyyyy:
        dd, mm, yyyy = ddmmyyyy.group(1), ddmmyyyy.group(2), ddmmyyyy.group(3)
        return f"{yyyy}-{mm}-{dd}"
    # YYYY-MM-DD format (ISO)
    yyyymmdd = re.search(r"(\d{4})[-_.](\d{2})[-_.](\d{2})", stem)
    if yyyymmdd:
        return f"{yyyymmdd.group(1)}-{yyyymmdd.group(2)}-{yyyymmdd.group(3)}"
    return None


# ─────────────────────────────────────────────────────────────────────
# Wikilink + tag extraction
# ─────────────────────────────────────────────────────────────────────

def extract_wikilinks(text: str) -> list[str]:
    """Extract [[wikilinks]] and [[wikilinks|display text]]."""
    return re.findall(r"\[\[([^\]|]+?)(?:\|[^\]]+?)?\]\]", text)


def extract_tags(text: str) -> list[str]:
    """Extract #tags from text (not inside code blocks)."""
    # Simple approach — skip lines starting with ```
    tags = set()
    in_code = False
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if not in_code:
            tags.update(re.findall(r"(?:^|\s)#([a-zA-Z][\w/-]*)", line))
    return sorted(tags)


# ─────────────────────────────────────────────────────────────────────
# Text cleaning
# ─────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Clean markdown text for embedding — preserve structure, remove noise."""
    # Remove image embeds but keep alt text
    text = re.sub(r"!\[\[([^\]]*)\]\]", r"[image: \1]", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"[image: \1]", text)

    # Convert wikilinks to plain text
    text = re.sub(r"\[\[([^\]|]+?)\|([^\]]+?)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+?)\]\]", r"\1", text)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove HTML comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    return text.strip()


# ─────────────────────────────────────────────────────────────────────
# Chunk splitter (for oversized sections)
# ─────────────────────────────────────────────────────────────────────

def split_large_chunk(text: str, max_size: int = MAX_CHUNK_SIZE,
                      overlap: int = OVERLAP_SIZE) -> list[str]:
    """Split text that exceeds max_size into overlapping chunks.
    Splits on paragraph boundaries when possible, falls back to sentences."""
    if len(text) <= max_size:
        return [text]

    chunks = []
    paragraphs = text.split("\n\n")
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 > max_size:
            if current:
                chunks.append(current.strip())
                # Overlap: keep the last `overlap` chars
                current = current[-overlap:] + "\n\n" + para
            else:
                # Single paragraph exceeds max — split by sentences
                sentences = re.split(r"(?<=[.!?])\s+", para)
                for sent in sentences:
                    if len(current) + len(sent) + 1 > max_size:
                        if current:
                            chunks.append(current.strip())
                            current = current[-overlap:] + " " + sent
                        elif len(sent) > max_size:
                            # A single "sentence" longer than max_size (OCR
                            # walls, tables, minified text): window it — the
                            # old sent[:max_size] silently DROPPED the rest.
                            chunks.extend(fixed_window_chunks(sent, max_size, overlap))
                            current = ""
                        else:
                            current = sent
                    else:
                        current = current + " " + sent if current else sent
        else:
            current = current + "\n\n" + para if current else para

    if current.strip():
        chunks.append(current.strip())

    return chunks


# ─────────────────────────────────────────────────────────────────────
# Selectable chunking strategies (for oversized sections ONLY)
# ─────────────────────────────────────────────────────────────────────
# Strategy selection changes how a section that exceeds max_chunk_size is
# cut into sub-chunks — nothing else. Heading detection, context headers,
# metadata, and the doc_ids of sections that fit in one chunk are identical
# under every strategy, so re-ingesting with a different strategy re-chunks
# (new text → new doc_ids → swap playbook) only the long sections.

CHUNKING_STRATEGIES = ("heading", "fixed", "document", "none")

_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_TABLE_RE = re.compile(r"^\s*\|.*\|\s*$")
_LIST_RE = re.compile(r"^\s*([-*+]\s+|\d+[.)]\s+)")


def _document_blocks(text: str) -> list[str]:
    """Split text into document ELEMENTS: fenced code blocks, tables, lists,
    and paragraphs. An element is never cut internally — a table row can't
    end up in a different chunk than its header, a code fence stays whole."""
    blocks: list[str] = []
    cur: list[str] = []
    mode: str | None = None      # code | table | list | para
    in_fence = False

    def flush():
        nonlocal cur
        joined = "\n".join(cur).strip("\n")
        if joined.strip():
            blocks.append(joined)
        cur = []

    for ln in text.split("\n"):
        if in_fence:
            cur.append(ln)
            if _FENCE_RE.match(ln):
                in_fence = False
                flush()
                mode = None
            continue
        if _FENCE_RE.match(ln):
            flush()
            cur = [ln]
            in_fence = True
            mode = "code"
            continue
        if not ln.strip():
            # blank line ends the current element (indented list bodies keep
            # their blanks only via the continuation rule below)
            flush()
            mode = None
            continue
        if _TABLE_RE.match(ln):
            if mode != "table":
                flush()
                mode = "table"
            cur.append(ln)
            continue
        if _LIST_RE.match(ln):
            if mode != "list":
                flush()
                mode = "list"
            cur.append(ln)
            continue
        if mode == "list" and ln[:1] in (" ", "\t"):
            cur.append(ln)               # indented continuation of a list item
            continue
        if mode != "para":
            flush()
            mode = "para"
        cur.append(ln)
    if in_fence:                          # unterminated fence: keep what we have
        flush()
    else:
        flush()
    return blocks


def document_element_chunks(text: str, max_size: int, overlap: int) -> list[str]:
    """Document-aware packing: split into elements (code/table/list/paragraph),
    then greedily pack consecutive elements up to max_size without ever cutting
    inside one. A single element bigger than max_size falls back to the fixed
    sliding window (the only honest option left)."""
    text = text.strip()
    if len(text) <= max_size:
        return [text] if text else []
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for b in _document_blocks(text):
        if len(b) > max_size:
            if buf:
                chunks.append("\n\n".join(buf))
                buf, size = [], 0
            chunks.extend(fixed_window_chunks(b, max_size, overlap))
            continue
        if buf and size + len(b) + 2 > max_size:
            chunks.append("\n\n".join(buf))
            buf, size = [], 0
        buf.append(b)
        size += len(b) + 2
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


def fixed_window_chunks(text: str, max_size: int, overlap: int) -> list[str]:
    """Strict sliding window with whitespace-snapped cuts.

    Every chunk is ~max_size chars; the window advances by (end - overlap),
    cutting at the last newline/space in the final 40% of the window so words
    survive intact. Predictable chunk size/count regardless of the text's
    paragraph structure — for text where that structure is noise (OCR output,
    minified exports, wall-of-text notes) and the default 'heading' packing
    produces wildly uneven chunks. Always makes forward progress.
    """
    text = text.strip()
    if len(text) <= max_size:
        return [text] if text else []
    overlap = max(0, min(overlap, max_size // 2))   # sane stride whatever the config says
    chunks = []
    i, n = 0, len(text)
    while i < n:
        end = min(i + max_size, n)
        if end < n:
            floor = i + int(max_size * 0.6)
            cut = max(text.rfind("\n", floor, end), text.rfind(" ", floor, end))
            if cut > i:
                end = cut
        piece = text[i:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        i = max(end - overlap, i + 1)
    return chunks


def split_section(text: str, max_size: int, overlap: int,
                  strategy: str = "heading") -> list[str]:
    """Split one oversized section using the named strategy.

    'heading' (default) = split_large_chunk, the historical
    paragraph/sentence-boundary packing. 'fixed' = fixed_window_chunks.
    'document' = document_element_chunks — element-aware packing that never
    cuts inside a code fence, table, or list. 'none' = no splitting at all:
    the whole section stays one chunk regardless of size (embedding models
    truncate long inputs — use only for sources where splitting is worse).
    All loaders (markdown + PDF) route through here; `--chunking` on the CLI
    or the console's Ingest tab selects per run.
    """
    strategy = (strategy or "heading").lower()
    if strategy not in CHUNKING_STRATEGIES:
        raise ValueError(f"unknown chunking strategy {strategy!r}; "
                         f"expected one of {CHUNKING_STRATEGIES}")
    if strategy == "none":
        text = text.strip()
        return [text] if text else []
    if strategy == "fixed":
        return fixed_window_chunks(text, max_size, overlap)
    if strategy == "document":
        return document_element_chunks(text, max_size, overlap)
    return split_large_chunk(text, max_size, overlap)


# ─────────────────────────────────────────────────────────────────────
# Context header builder (hierarchical heading inheritance)
# ─────────────────────────────────────────────────────────────────────

def build_context_header(heading_stack: list[str], file_meta: dict) -> str:
    """
    Build a context prefix that gets prepended to every chunk.
    This is the KEY insight from NVIDIA's retrieval research:
    chunks without context lose meaning. A section titled "Results"
    is meaningless without knowing it's from "CS 362 > ARIMA Models > Results".

    The heading stack provides hierarchical context inheritance.
    """
    parts = []

    if file_meta.get("course_name") and file_meta["course_name"] != "unknown":
        parts.append(f"Course: {file_meta['course_name']}")
    if file_meta.get("date"):
        parts.append(f"Date: {file_meta['date']}")
    if heading_stack:
        parts.append("Section: " + " > ".join(heading_stack))

    if not parts:
        return ""
    return "[" + " | ".join(parts) + "]\n"


# ─────────────────────────────────────────────────────────────────────
# Parent sections (E2 small-to-big: match small chunks, generate from
# the enclosing section). Children carry parent_id in METADATA ONLY —
# chunk text is untouched, so doc_ids are unaffected by this feature.
# Parents live in a sidecar JSONL and are NEVER embedded.
# ─────────────────────────────────────────────────────────────────────

def section_spans(nodes: list) -> list[tuple[int, int]]:
    """
    For each node i return (start, end): the slice of `nodes` making up the
    FULL section rooted at i — i itself plus every following node with a
    deeper heading level (its subsections). Level-0 nodes (preamble) are
    standalone. The flat node list from build_heading_tree is depth-ordered,
    so a section ends at the next node whose level is <= its own.
    """
    spans = []
    for i, n in enumerate(nodes):
        if n.level == 0:
            spans.append((i, i + 1))
            continue
        j = i + 1
        while j < len(nodes) and (nodes[j].level == 0 or nodes[j].level > n.level):
            j += 1
        spans.append((i, j))
    return spans


def section_ancestors(nodes: list, i: int) -> list[int]:
    """Indices of enclosing-section roots for node i, nearest first."""
    out = []
    lvl = nodes[i].level
    j = i - 1
    while j >= 0 and lvl > 1:
        if 0 < nodes[j].level < lvl:
            out.append(j)
            lvl = nodes[j].level
        j -= 1
    return out


# ─────────────────────────────────────────────────────────────────────
# MAIN PARSER
# ─────────────────────────────────────────────────────────────────────

class ObsidianParser:
    """
    Parse an Obsidian vault into Document chunks for RAG ingestion.

    Architecture decision: heading-level splitting with context inheritance.
    Each H2/H3 section becomes its own chunk. The heading hierarchy is
    prepended as a context header so that "Results" in isolation becomes
    "[Course: Time Series | Section: ARIMA Models > Results]" — giving
    the embedding model the context it needs.
    """

    def __init__(self, vault_path: str, config: Optional[dict] = None):
        self.vault_path = Path(vault_path)
        if not self.vault_path.exists():
            raise FileNotFoundError(f"Vault not found: {vault_path}")

        self.config = config or {}
        self.min_chunk = self.config.get("min_chunk_size", MIN_CHUNK_SIZE)
        self.max_chunk = self.config.get("max_chunk_size", MAX_CHUNK_SIZE)
        self.overlap = self.config.get("overlap_size", OVERLAP_SIZE)
        # How OVERSIZED sections are split (see split_section): 'heading'
        # keeps the historical paragraph packing, 'fixed' = sliding window.
        self.chunking = str(self.config.get("chunking", "heading")).lower()
        if self.chunking not in CHUNKING_STRATEGIES:
            raise ValueError(f"parser.chunking must be one of "
                             f"{CHUNKING_STRATEGIES}, got {self.chunking!r}")
        # E2 parent-child: cap on a parent section's text. Parents above the
        # cap are not emitted (their children stand alone) so a child's text
        # is always CONTAINED in its parent — swapping never loses evidence.
        self.parent_max_chars = self.config.get("parent_max_chars", 5000)
        self.parents: dict[str, dict] = {}      # parent_id -> sidecar record

        # Directories to skip (any path depth). _Backups = timestamped
        # duplicate copies of agent skills (E1 verdict: never index backups —
        # near-duplicate floods aimed straight at fusion).
        self.skip_dirs = {".obsidian", ".trash", ".git", "node_modules",
                          ".smart-connections", ".obsidian-git", "_Backups"}
        self.skip_dirs.update(set(self.config.get("skip_dirs", [])))
        # Top-level vault trees to exclude (exact match on the FIRST path
        # component only — safer than skip_dirs for names like "Documents"
        # that could legitimately appear deeper inside course folders).
        # Session 8: the agent-project trees (Workspace1/Workspace2/Workspace3/...)
        # ballooned an unscoped parse 7K -> 88.7K chunks; the markdown corpus
        # stays scoped to coursework until the author decides otherwise.
        self.skip_roots = set(self.config.get("skip_roots", []))
        # Optional scoped parse: only .md files whose vault-relative posix path
        # contains this substring (case-insensitive). Powers the inbox md lane —
        # a scoped parse MUST also set a non-default output so it never
        # clobbers the full chunks.jsonl.
        inc = self.config.get("include_path") or None
        self.include_path = str(inc).lower() if inc else None

        # Stats
        self.stats = {
            "files_processed": 0,
            "files_skipped": 0,
            "daily_notes": 0,
            "course_notes": 0,
            "other_notes": 0,
            "total_chunks": 0,
            "skipped_sections": 0,
            "chunks_by_domain": {},
            "chunks_by_course": {},
        }

    # Filename stems/patterns that identify non-content files to skip entirely.
    # These are checked in discover_files before any parsing happens.
    SKIP_FILE_STEMS = {
        # Excalidraw files export as .md but contain only JSON drawing data
        # (they show up as "Excalidraw Data" headings — 129 junk chunks)
    }
    SKIP_FILE_SUFFIXES = (".excalidraw",)
    SKIP_FILE_STEM_PATTERNS = [
        r"\.excalidraw$",          # "MyDiagram.excalidraw.md"
        r"excalidraw",             # any file with excalidraw in the name
    ]

    def discover_files(self) -> list[Path]:
        """Find all markdown files in the vault, respecting skip rules."""
        files = []
        for f in self.vault_path.rglob("*.md"):
            # Skip hidden / system directories
            if any(part in self.skip_dirs for part in f.parts):
                continue
            # Skip excluded top-level trees (agent projects etc.)
            rel_parts = f.relative_to(self.vault_path).parts
            if rel_parts and rel_parts[0] in self.skip_roots:
                self.stats["files_skipped"] += 1
                continue
            # Scoped parse (inbox md lane): substring filter on the rel path
            if self.include_path and self.include_path not in \
                    f.relative_to(self.vault_path).as_posix().lower():
                self.stats["files_skipped"] += 1
                continue
            # Skip empty files
            if f.stat().st_size < 50:
                self.stats["files_skipped"] += 1
                continue
            # Skip Excalidraw drawing files (JSON masquerading as .md)
            stem_lower = f.stem.lower()
            if any(re.search(pat, stem_lower) for pat in self.SKIP_FILE_STEM_PATTERNS):
                self.stats["files_skipped"] += 1
                continue
            files.append(f)
        return sorted(files)

    def parse_file(self, filepath: Path) -> list[Document]:
        """Parse a single markdown file into Document chunks."""
        content = filepath.read_text(encoding="utf-8", errors="replace")
        frontmatter, body = parse_frontmatter(content)
        rel_path = str(filepath.relative_to(self.vault_path))

        # Base metadata from frontmatter + file info
        base_meta = {
            "source_file": rel_path,
            "filename": filepath.stem,
            "file_type": "daily_note" if is_daily_note(filepath) else "note",
            "vault_path": str(self.vault_path),
            "tags": extract_tags(body),
            "wikilinks": extract_wikilinks(body)[:20],  # cap for metadata size
        }

        # Merge frontmatter (user's YAML takes priority)
        if "tags" in frontmatter:
            fm_tags = frontmatter["tags"]
            if isinstance(fm_tags, list):
                base_meta["tags"] = list(set(base_meta["tags"] + fm_tags))
        if "course" in frontmatter:
            base_meta["frontmatter_course"] = frontmatter["course"]
        if "date" in frontmatter:
            base_meta["date"] = str(frontmatter["date"])
        elif is_daily_note(filepath):
            date = extract_date_from_filename(filepath)
            if date:
                base_meta["date"] = date

        # Route to appropriate parser
        if is_daily_note(filepath):
            self.stats["daily_notes"] += 1
            return self._parse_daily_note(body, base_meta)
        else:
            # Detect if it's a course-specific note from path or frontmatter
            course = self._detect_course_from_path(filepath, frontmatter)
            if course:
                self.stats["course_notes"] += 1
                base_meta.update(resolve_course_metadata(course))
            else:
                self.stats["other_notes"] += 1
            return self._parse_standard_note(body, base_meta)

    def _detect_course_from_path(self, filepath: Path, frontmatter: dict) -> Optional[str]:
        """Try to detect course from file path or frontmatter.

        Priority order:
          1. Frontmatter 'course' key
          2. Heading-pattern match on directory/filename parts
          3. FOLDER_COURSE_MAP exact lookup (catches full names with & / spaces)
        """
        # 1. Frontmatter wins
        if "course" in frontmatter:
            return str(frontmatter["course"])

        # 2. Try heading-pattern detection on each path part
        for part in filepath.parts:
            course = detect_course_from_heading(part)
            if course and course != "__SKIP__":
                return course

        # 3. Folder-name exact lookup (case-insensitive) — catches full-title
        #    folder names like "Statistics & Inference" that have special chars
        for part in filepath.parts:
            match = FOLDER_COURSE_MAP.get(part.lower().strip())
            if match:
                return match

        # 4. Check filename via heading detection
        course = detect_course_from_heading(filepath.stem)
        if course == "__SKIP__":
            return None
        return course

    # ---- E2 parent-child helpers ----

    def _section_text(self, nodes: list, s: int, e: int) -> str:
        """Full markdown of the section nodes[s:e]: headings + cleaned content.
        May include content of sub-headings that were skip-listed as chunks
        (todo lists etc.) — capped and harmless as generation context."""
        parts = []
        for n in nodes[s:e]:
            if n.level > 0:
                parts.append("#" * n.level + " " + n.title)
            if n.content and n.content.strip():
                parts.append(clean_text(n.content))
        return "\n\n".join(parts)

    def _parent_candidate(self, nodes: list, spans: list, i: int,
                          stacks: list, chunk_meta: dict):
        """
        For the chunk(s) emitted from node i, pick the OUTERMOST enclosing
        section whose full text fits parent_max_chars (section text only grows
        going outward, so the first overflow stops the walk). Returns
        (parent_id, text_len, sidecar_record) or None. The caller links it
        per-child and records it into self.parents only when actually used.
        """
        best = None
        for idx in [i] + section_ancestors(nodes, i):
            s, e = spans[idx]
            text = self._section_text(nodes, s, e)
            if len(text) > self.parent_max_chars:
                break
            best = (idx, text)
        if best is None:
            return None
        idx, text = best
        header = build_context_header(stacks[idx], chunk_meta)
        pid = hashlib.sha256(
            f"{chunk_meta.get('source_file', '')}::P{idx}::{text[:200]}".encode()
        ).hexdigest()[:16]
        record = {
            "parent_id": pid,
            "source_file": chunk_meta.get("source_file", ""),
            "heading": nodes[idx].title,
            "heading_path": " > ".join(stacks[idx]) if stacks[idx] else "",
            "text": header + text,
        }
        return pid, len(text), record

    def _maybe_link_parent(self, meta: dict, pinfo, child_text: str) -> None:
        """Attach parent_id when the parent is meaningfully bigger than the
        child (>=1.3x) — swapping a chunk for a same-sized parent is noise."""
        if pinfo and pinfo[1] >= int(1.3 * len(child_text)):
            meta["parent_id"] = pinfo[0]
            self.parents.setdefault(pinfo[0], pinfo[2])

    def _parse_daily_note(self, body: str, base_meta: dict) -> list[Document]:
        """
        Parse a daily note with multiple course lectures.

        Strategy: walk through heading nodes. When a heading matches a course
        pattern, start a new "lecture segment". All content under that heading
        (including sub-headings) belongs to that lecture until the next
        course-level heading.

        This handles the author's format:
            # RL L3
            ## Topic: Policy Gradient
            content...
            ---
            # NLP L4
            ## Attention Mechanism
            content...
        """
        nodes = build_heading_tree(body)
        spans = section_spans(nodes)
        stacks: list[list[str]] = []   # per-node heading-stack snapshots (E2)
        documents = []
        current_course = None
        current_course_meta = {}
        heading_stack = []  # for context inheritance
        skip_mode = False   # skip non-academic sections (Self S, MAPP, etc.)

        for i, node in enumerate(nodes):
            # Check if this heading starts a new course lecture
            # the author uses H1 for course separators, so check level >= 1
            detected = detect_course_from_heading(node.title) if node.level >= 1 else None

            if detected == "__SKIP__":
                # Non-academic section (Self Study, MAPP, Nutrition, etc.)
                skip_mode = True
                current_course = None
                current_course_meta = {}
                heading_stack = [node.title]
                stacks.append(list(heading_stack))
                continue

            if detected and detected != "__SKIP__" and node.level <= 3:
                # New course segment — exit skip mode
                skip_mode = False
                current_course = detected
                current_course_meta = resolve_course_metadata(detected)
                heading_stack = [node.title]
            elif node.level > 0:
                # Sub-heading within current course (or skipped section)
                while heading_stack and len(heading_stack) >= node.level:
                    heading_stack.pop()
                heading_stack.append(node.title)
            stacks.append(list(heading_stack))

            # Skip content in non-academic sections
            if skip_mode:
                self.stats["skipped_sections"] += 1
                continue

            if not node.content or len(node.content.strip()) < self.min_chunk:
                continue

            # Build chunk metadata
            chunk_meta = {**base_meta}
            chunk_meta["heading"] = node.title
            chunk_meta["heading_level"] = node.level
            chunk_meta["heading_path"] = " > ".join(heading_stack)
            if current_course_meta:
                chunk_meta.update(current_course_meta)

            # Build the chunk text with context header
            context = build_context_header(heading_stack, chunk_meta)
            cleaned = clean_text(node.content)

            # Split if too large
            sub_chunks = split_section(cleaned, self.max_chunk, self.overlap,
                                       self.chunking)
            pinfo = self._parent_candidate(nodes, spans, i, stacks, chunk_meta)

            for pi, chunk_text in enumerate(sub_chunks):
                if len(chunk_text.strip()) < self.min_chunk:
                    continue

                full_text = context + chunk_text
                meta = {**chunk_meta}
                if len(sub_chunks) > 1:
                    meta["chunk_part"] = f"{pi + 1}/{len(sub_chunks)}"
                self._maybe_link_parent(meta, pinfo, chunk_text)

                doc = Document(text=full_text, metadata=meta)
                documents.append(doc)
                self._update_stats(meta)

        return documents

    def _parse_standard_note(self, body: str, base_meta: dict) -> list[Document]:
        """
        Parse a standard (non-daily) note.
        Splits on H2/H3 boundaries with context inheritance.
        """
        nodes = build_heading_tree(body)
        spans = section_spans(nodes)
        stacks: list[list[str]] = []   # per-node heading-stack snapshots (E2)
        documents = []
        heading_stack = []

        for i, node in enumerate(nodes):
            # Maintain heading stack for context
            if node.level > 0:
                while heading_stack and len(heading_stack) >= node.level:
                    heading_stack.pop()
                heading_stack.append(node.title)
            stacks.append(list(heading_stack))

            if not node.content or len(node.content.strip()) < self.min_chunk:
                continue

            # If no course detected yet, try from this heading
            if base_meta.get("course_code") is None or base_meta.get("course_code") == "unknown":
                detected = detect_course_from_heading(node.title)
                if detected:
                    base_meta.update(resolve_course_metadata(detected))

            chunk_meta = {**base_meta}
            chunk_meta["heading"] = node.title
            chunk_meta["heading_level"] = node.level
            chunk_meta["heading_path"] = " > ".join(heading_stack)

            context = build_context_header(heading_stack, chunk_meta)
            cleaned = clean_text(node.content)

            sub_chunks = split_section(cleaned, self.max_chunk, self.overlap,
                                       self.chunking)
            pinfo = self._parent_candidate(nodes, spans, i, stacks, chunk_meta)

            for pi, chunk_text in enumerate(sub_chunks):
                if len(chunk_text.strip()) < self.min_chunk:
                    continue
                full_text = context + chunk_text
                meta = {**chunk_meta}
                if len(sub_chunks) > 1:
                    meta["chunk_part"] = f"{pi + 1}/{len(sub_chunks)}"
                self._maybe_link_parent(meta, pinfo, chunk_text)

                doc = Document(text=full_text, metadata=meta)
                documents.append(doc)
                self._update_stats(meta)

        return documents

    def _update_stats(self, meta: dict):
        """Track parsing statistics."""
        self.stats["total_chunks"] += 1
        domain = meta.get("domain", "general")
        self.stats["chunks_by_domain"][domain] = \
            self.stats["chunks_by_domain"].get(domain, 0) + 1
        course = meta.get("course_code", "unknown")
        self.stats["chunks_by_course"][course] = \
            self.stats["chunks_by_course"].get(course, 0) + 1

    def parse_all(self, verbose: bool = True) -> list[Document]:
        """
        Parse the entire vault. Returns all documents.

        This is the main entry point.
        """
        files = self.discover_files()
        if verbose:
            print(f"[parser] Found {len(files)} markdown files in {self.vault_path}")

        all_docs = []
        for filepath in files:
            try:
                docs = self.parse_file(filepath)
                all_docs.extend(docs)
                self.stats["files_processed"] += 1
            except Exception as e:
                print(f"[parser] ERROR processing {filepath}: {e}", file=sys.stderr)
                self.stats["files_skipped"] += 1

        if verbose:
            self._print_stats()

        return all_docs

    def _print_stats(self):
        """Print a summary of parsing results."""
        s = self.stats
        print(f"\n{'=' * 60}")
        print(f"  OBSIDIAN VAULT PARSING COMPLETE")
        print(f"{'=' * 60}")
        print(f"  Files processed:  {s['files_processed']}")
        print(f"  Files skipped:    {s['files_skipped']}")
        print(f"  Daily notes:      {s['daily_notes']}")
        print(f"  Course notes:     {s['course_notes']}")
        print(f"  Other notes:      {s['other_notes']}")
        print(f"  Total chunks:     {s['total_chunks']}")
        print(f"  Skipped sections: {s['skipped_sections']} (Self Study, MAPP, Nutrition, BoHL)")
        print(f"\n  Chunks by domain:")
        for domain, count in sorted(s["chunks_by_domain"].items(),
                                     key=lambda x: -x[1]):
            print(f"    {domain:.<30} {count}")
        print(f"\n  Chunks by course:")
        for course, count in sorted(s["chunks_by_course"].items(),
                                     key=lambda x: -x[1]):
            print(f"    {course:.<30} {count}")
        print(f"{'=' * 60}\n")

    def export_jsonl(self, documents: list[Document], output_path: str):
        """Export documents as JSONL for downstream processing."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for doc in documents:
                f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")
        print(f"[parser] Exported {len(documents)} documents to {path}")

    def export_parents_jsonl(self, output_path: str):
        """Export the E2 parent-section sidecar (NOT embedded, NOT indexed —
        consumed only by retrieval.parent_context at generation time)."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for rec in self.parents.values():
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[parser] Exported {len(self.parents)} parent sections to {path}")


# ─────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Parse Obsidian vault for RAG ingestion")
    ap.add_argument("vault_path", help="Path to Obsidian vault root")
    ap.add_argument("-o", "--output", default="chunks.jsonl",
                    help="Output JSONL file (default: chunks.jsonl)")
    ap.add_argument("--max-chunk", type=int, default=MAX_CHUNK_SIZE,
                    help=f"Max chunk size in chars (default: {MAX_CHUNK_SIZE})")
    ap.add_argument("--min-chunk", type=int, default=MIN_CHUNK_SIZE,
                    help=f"Min chunk size in chars (default: {MIN_CHUNK_SIZE})")
    ap.add_argument("--overlap", type=int, default=OVERLAP_SIZE,
                    help=f"Overlap between split chunks (default: {OVERLAP_SIZE})")
    ap.add_argument("--chunking", default="heading", choices=CHUNKING_STRATEGIES,
                    help="How oversized sections are split: 'heading' = "
                         "paragraph packing (default), 'fixed' = sliding window")
    ap.add_argument("--preview", type=int, default=0,
                    help="Print N sample chunks to stdout")
    ap.add_argument("--parents-out", default="data/parents_md.jsonl",
                    help="Sidecar JSONL for E2 parent sections "
                         "(default: data/parents_md.jsonl; '' disables)")
    ap.add_argument("--skip-roots", default="",
                    help="Comma-separated TOP-LEVEL vault folders to exclude "
                         "(e.g. \"Workspace1,Workspace2\" — agent trees, not coursework)")
    ap.add_argument("--include-path", default="",
                    help="Only parse .md files whose vault-relative path contains "
                         "this substring (scoped parse — ALWAYS pair with a "
                         "non-default --output so chunks.jsonl is never clobbered)")

    args = ap.parse_args()
    if args.include_path and args.output == "chunks.jsonl":
        ap.error("--include-path requires an explicit non-default --output "
                 "(a scoped parse would clobber the full chunks.jsonl)")

    config = {
        "max_chunk_size": args.max_chunk,
        "min_chunk_size": args.min_chunk,
        "overlap_size": args.overlap,
        "chunking": args.chunking,
        "skip_roots": [s.strip() for s in args.skip_roots.split(",") if s.strip()],
        "include_path": args.include_path or None,
    }

    parser = ObsidianParser(args.vault_path, config=config)
    docs = parser.parse_all()

    if args.preview > 0:
        print(f"\n--- SAMPLE CHUNKS (first {args.preview}) ---\n")
        for doc in docs[:args.preview]:
            print(f"[{doc.doc_id}] {doc.metadata.get('source_file', '?')}")
            print(f"  course: {doc.metadata.get('course_name', '?')}")
            print(f"  domain: {doc.metadata.get('domain', '?')}")
            print(f"  heading: {doc.metadata.get('heading_path', '?')}")
            print(f"  text: {doc.text[:200]}...")
            print()

    parser.export_jsonl(docs, args.output)
    if args.parents_out:
        parser.export_parents_jsonl(args.parents_out)


if __name__ == "__main__":
    main()
