"""
Structure Agent (v2 — Fine-tuned T5 / Bot 3)
---------------------------------------------
Loads the locally fine-tuned T5-base model from 'bot 3/' and converts
sanitised (GLiNER-anonymised) resume text into a strict, bias-free JSON
schema.

Architecture
------------
  sanitised_text
      │
      ▼
  preformat_resume()          ← normalises to training-data style
      │
      ▼
  Fine-tuned T5 model         ← attempts JSON generation
      │
      ├─ valid JSON? ──────────► return structured dict
      │
      └─ invalid / empty? ─────► rule_based_fallback()
                                    │
                                    └─► return structured dict

Why the fallback?
  The model checkpoint ('bot 3/best_checkpoint.ckpt') was saved at
  global_step=1000, epoch=0, with a teacher-forced cross-entropy loss of
  ~4.6 on training examples. That indicates the model has not fully
  converged and may not yet produce reliable JSON. The rule-based fallback
  guarantees the pipeline always returns a usable structured dict, while the
  T5 output is preferred whenever it is valid JSON — meaning re-training to
  lower loss will automatically improve results with no code changes needed.

Output Schema
-------------
{
    "total_years_experience": int | null,
    "technical_skills":       [str],
    "job_history":            [{"title": str, "duration_months": int}],
    "highest_degree":         "High School" | "Associate" | "Bachelor"
                            | "Master" | "PhD" | "None stated",
    "education":              [{"degree": str, "field": str, "institution": str,
                               "gpa": str, "score": str}],
    "experience":             [{"title": str, "company": str, "duration_months": int,
                               "date_range": str, "type": "Job" | "Internship"}],
    "work_experience_summary": {
        "total_months": int,
        "total_years": float,
        "jobs_count": int,
        "jobs_months": int,
        "internships_count": int,
        "internships_months": int,
        "roles": [{"title": str, "type": str, "duration_months": int, "date_range": str}]
    } | null
}
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
_CHECKPOINT_PATH = _HERE.parent.parent / "bot3.1" / "best_checkpoint.ckpt"
_BASE_MODEL_NAME  = "t5-base"

# ── Generation hyper-parameters ────────────────────────────────────────────────
_MAX_INPUT_TOKENS  = 768
_MAX_OUTPUT_TOKENS = 384
_NUM_BEAMS         = 4

# ── Subjective-phrase filter ───────────────────────────────────────────────────
_SUBJECTIVE_RE = re.compile(
    r"\b("
    r"passionate(ly)?|hard[- ]?work(ing|er)?|results?[- ]?driven|"
    r"visionary|innovative|dynamic|exceptional|accomplished|award[- ]?winning|"
    r"world[- ]?class|go[- ]?getter|thrives? under pressure|motivated|"
    r"highly motivated|extremely hardworking|deeply empathetic|resourceful|"
    r"creative|versatile|seasoned|self[- ]?taught|dedicated|committed|"
    r"driven|strategic|emotionally intelligent|unparalleled|polymath|"
    r"eager to make an impact|proven leadership|top[- ]?tier|obsessed with"
    r")\b",
    re.IGNORECASE,
)

# ── Section-header normalisation ───────────────────────────────────────────────
_SECTION_ALIASES: list[tuple[re.Pattern, str]] = [
    (re.compile(
        r"^\s*(technical\s+skills?|core\s+competencies?|skills?\s+acquired|"
        r"design\s+tools?|research\s+&\s+technical\s+skills?|"
        r"tech\s+stack|tech\s+expertise|tools?\s+&\s+(systems?|platforms?))\s*:?\s*$",
        re.I | re.M,
    ), "SKILLS"),
    (re.compile(
        r"^\s*(work\s+(history|experience)|professional\s+experience|"
        r"experience|career\s+history|leadership\s+history|"
        r"freelance\s+&\s+work|previous\s+experience)\s*:?\s*$",
        re.I | re.M,
    ), "EXPERIENCE"),
    (re.compile(
        r"^\s*(education|academic\s+background)\s*:?\s*$",
        re.I | re.M,
    ), "EDUCATION"),
]

# ── Lazy model singleton ────────────────────────────────────────────────────────
_model:     Optional[T5ForConditionalGeneration] = None
_tokenizer: Optional[T5Tokenizer] = None


def _load_model() -> tuple[T5ForConditionalGeneration, T5Tokenizer]:
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    if not _CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Fine-tuned checkpoint not found:\n  {_CHECKPOINT_PATH}"
        )

    print(f"[INFO] Loading tokeniser from '{_BASE_MODEL_NAME}' …")
    _tokenizer = T5Tokenizer.from_pretrained(_BASE_MODEL_NAME)

    print(f"[INFO] Loading fine-tuned model from:\n  {_CHECKPOINT_PATH}")
    ck = torch.load(str(_CHECKPOINT_PATH), map_location="cpu")
    # PyTorch-Lightning prefixes every key with 'model.' — strip it
    cleaned_sd = {
        k[len("model."):] if k.startswith("model.") else k: v
        for k, v in ck["state_dict"].items()
    }
    _model = T5ForConditionalGeneration.from_pretrained(
        _BASE_MODEL_NAME, ignore_mismatched_sizes=True
    )
    _model.load_state_dict(cleaned_sd, strict=False)
    _model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = _model.to(device)
    print(f"[INFO] Model ready on {device.upper()}.")
    return _model, _tokenizer


def unload_t5() -> None:
    """
    Release the T5 model from GPU/CPU memory so that the downstream evaluator
    (Phi-3.5-mini LoRA) can fit within the available VRAM.

    Call this after structure_resume() completes and before evaluate_resume().
    The model will be reloaded lazily on the next structure_resume() call.
    """
    import gc
    global _model, _tokenizer
    if _model is not None:
        del _model
        _model = None
    if _tokenizer is not None:
        del _tokenizer
        _tokenizer = None
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    print("[INFO] T5 model unloaded to free VRAM for evaluator.")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Pre-formatter
# ══════════════════════════════════════════════════════════════════════════════

def preformat_resume(text: str) -> str:
    """
    Normalise sanitised resume text to match the training-data style exactly.

    The model's training examples share a consistent layout:
      • Subjective / opinion phrases stripped
      • Section headers in ALL-CAPS (SKILLS / EXPERIENCE / EDUCATION)
      • One role per line with duration in parentheses
      • No more than one blank line between sections

    Matching this format at inference time reduces encoder confusion and
    improves extraction accuracy.
    """
    # 1. Strip subjective phrases
    text = _SUBJECTIVE_RE.sub(" ", text)

    # 2. Normalise section headers → canonical ALL-CAPS
    for pattern, canonical in _SECTION_ALIASES:
        text = pattern.sub(f"\n{canonical}\n", text)

    # 3. Collapse 3+ consecutive blank lines → 1 blank line
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 4. Strip trailing spaces on each line
    text = "\n".join(line.rstrip() for line in text.splitlines())

    return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — T5 inference
# ══════════════════════════════════════════════════════════════════════════════

def _run_t5(formatted_text: str) -> Optional[dict]:
    """
    Feed pre-formatted resume text into the fine-tuned T5 model.
    Returns a parsed dict if the output is valid JSON, else None.
    """
    try:
        model, tokenizer = _load_model()
    except FileNotFoundError as e:
        print(f"[WARN] Model not available: {e}")
        return None

    device = next(model.parameters()).device
    inputs = tokenizer(
        f"Extract JSON from this resume:\n{formatted_text}",
        return_tensors="pt",
        max_length=_MAX_INPUT_TOKENS,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=_MAX_OUTPUT_TOKENS,
            num_beams=_NUM_BEAMS,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    raw = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"[INFO] T5 raw output: {raw[:200]}")

    # Truncate at 'Tip:' — discard anything the model adds after the JSON
    tip_match = re.search(r"tip:", raw, re.IGNORECASE)
    if tip_match:
        raw = raw[: tip_match.start()]

    # Try to parse as JSON
    raw_clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    try:
        return json.loads(raw_clean)
    except json.JSONDecodeError:
        pass

    # Try to find a JSON object anywhere in the output
    match = re.search(r"\{.*\}", raw_clean, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None  # Model output was not usable JSON


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Rule-based fallback extractor
# ══════════════════════════════════════════════════════════════════════════════

# Degree keywords ordered from highest to lowest (first match wins)
_DEGREE_PATTERNS = [
    ("PhD",         re.compile(r"\b(ph\.?d|doctor(ate)?)\b", re.I)),
    ("Master",      re.compile(r"\b(master|m\.?s\.?|m\.?a\.?|mba|m\.?eng)\b", re.I)),
    ("Bachelor",    re.compile(r"\b(bachelor|b\.?s\.?|b\.?a\.?|b\.?eng|b\.?tech|b\.?comm)\b", re.I)),
    ("Associate",   re.compile(r"\bassociate\b", re.I)),
    ("High School", re.compile(r"\b(high\s+school|secondary|diploma)\b", re.I)),
]

# Duration extraction from role lines like "... (3 years)" / "(18 months)"
_DURATION_RE = re.compile(
    r"\((?:(?P<years>\d+(?:\.\d+)?)\s*years?)?"
    r"(?:\s*(?P<months_word>and\s+)?"
    r"(?P<months>\d+)\s*months?)?\)",
    re.I,
)

# Skills delimiters
_SKILL_SPLIT_RE = re.compile(r"[,;|•\t]+")




def _extract_duration_months(text: str) -> Optional[int]:
    """Parse a duration string like '3 years' / '18 months' / '2 years 6 months'."""
    m = _DURATION_RE.search(text)
    if not m:
        return None
    years  = float(m.group("years")  or 0)
    months = float(m.group("months") or 0)
    total  = int(years * 12 + months)
    return total if total > 0 else None


def _rule_based_fallback(formatted_text: str, sanitized_text: str) -> dict:
    """
    Parse the pre-formatted resume text using deterministic rules.
    This mirrors the patterns the model was trained on, making it a reliable
    fallback when the model's output is not valid JSON.
    Designed to work on real freeform resume text, not just training-data style.
    """
    lines = formatted_text.splitlines()
    full_text = formatted_text  # keep full text for cross-section searches

    result = {
        "total_years_experience": None,
        "technical_skills": [],
        "job_history": [],
        "highest_degree": "None stated",
    }

    # ── Section detection (fuzzy keyword matching) ────────────────────────────
    # Maps each line index to the section it belongs to
    SECTION_KEYWORDS = {
        "SKILLS": re.compile(
            r"(technical\s+skills?|skills?|competenc|proficienc|technologies|"
            r"tech\s+stack|tools?\s+&|core\s+skills?|areas\s+of\s+expertise)",
            re.I
        ),
        "EXPERIENCE": re.compile(
            r"(experience|employment|work\s+history|career|positions?|roles?|"
            r"professional\s+background|leadership\s+history|internship)",
            re.I
        ),
        "EDUCATION": re.compile(
            r"(education|academic|degree|qualif|university|college|school|"
            r"certif)",
            re.I
        ),
    }

    line_sections: list[str | None] = [None] * len(lines)
    current_section = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        # A "header" line: short (< 60 chars), no sentence punctuation, matches keyword
        if len(stripped) < 60 and not re.search(r"[.?!]", stripped):
            for sec_name, pattern in SECTION_KEYWORDS.items():
                if pattern.search(stripped):
                    # Make sure it's actually a heading (not a skill containing the word)
                    # Heuristic: heading lines rarely contain commas or pipes
                    if stripped.count(",") <= 1 and "|" not in stripped:
                        current_section = sec_name
                        break
        line_sections[i] = current_section

    skills_lines   = [lines[i] for i, s in enumerate(line_sections) if s == "SKILLS"]
    exp_lines      = [lines[i] for i, s in enumerate(line_sections) if s == "EXPERIENCE"]
    edu_lines      = [lines[i] for i, s in enumerate(line_sections) if s == "EDUCATION"]

    # ── 1. Technical skills ───────────────────────────────────────────────────
    # Known tech keywords to scan for when no explicit SKILLS section exists
    _TECH_KEYWORDS = re.compile(
        r"\b("
        r"python|java(?:script)?|typescript|c\+\+|c#|ruby|go|rust|kotlin|swift|php|scala|"
        r"react(?:\.js)?|angular|vue(?:\.js)?|next\.js|node(?:\.js)?|express(?:\.js)?|"
        r"django|flask|fastapi|spring|laravel|rails|"
        r"sql|mysql|postgresql|postgres|sqlite|mongodb|redis|elasticsearch|dynamodb|firebase|"
        r"html|css|sass|scss|tailwind|"
        r"aws|azure|gcp|docker|kubernetes|terraform|ansible|jenkins|github\s+actions|"
        r"git|linux|bash|powershell|"
        r"tensorflow|pytorch|keras|scikit[- ]learn|pandas|numpy|matplotlib|"
        r"graphql|rest(?:\s+api)?|grpc|kafka|rabbitmq|"
        r"figma|sketch|adobe\s+xd|photoshop|illustrator|"
        r"excel|power\s+bi|tableau|looker|"
        r"jira|confluence|trello|asana|notion|"
        r"selenium|cypress|playwright|jest|pytest|postman|junit"
        r")\b",
        re.I,
    )
    _COURSEWORK_RE = re.compile(r"relevant\s+coursework|extracurricular|gpa|grade", re.I)

    raw_skills: list[str] = []
    if skills_lines:
        # Dedicated SKILLS section found — use it directly
        for sl in skills_lines:
            raw_skills.extend(_SKILL_SPLIT_RE.split(sl))
    else:
        # No SKILLS section: scan all lines for comma-separated tech lists,
        # but skip lines that are clearly from education (coursework, GPA, etc.)
        for line in lines:
            stripped_line = line.strip()
            # Skip coursework / education annotation lines
            if _COURSEWORK_RE.search(stripped_line):
                continue
            parts = [p.strip() for p in stripped_line.split(",") if p.strip()]
            if len(parts) >= 3 and all(len(p) < 40 and "\n" not in p for p in parts):
                # Only accept this list if at least one item is a known tech keyword
                if any(_TECH_KEYWORDS.search(p) for p in parts):
                    raw_skills.extend(parts)

    # If still no skills found, mine tech keywords from bullet-point lines
    if not raw_skills:
        seen: set[str] = set()
        for line in lines:
            stripped_line = line.strip()
            if _COURSEWORK_RE.search(stripped_line):
                continue
            for m in _TECH_KEYWORDS.finditer(stripped_line):
                kw = m.group(0).strip()
                if kw.lower() not in seen:
                    seen.add(kw.lower())
                    raw_skills.append(kw)

    result["technical_skills"] = [
        s.strip() for s in raw_skills
        if s.strip()
        and len(s.strip()) > 1
        and not re.match(r"^\[.*\]$", s.strip())  # skip bare placeholders
    ]

    # ── 2. Job history ────────────────────────────────────────────────────────
    # Pattern A: "(3 years)" / "(18 months)" / "(2 years 6 months)"
    dur_paren_re = re.compile(
        r"\((?:(?P<y1>\d+(?:\.\d+)?)\s*years?)?(?:\s*(?:and\s+)?(?P<m1>\d+)\s*months?)?\)",
        re.I,
    )
    # Pattern B: named-month date range like "Jan 2019 – Mar 2022"
    date_range_re = re.compile(
        r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{4})"
        r"\s*[-–—to]+\s*"
        r"(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+)?(\d{4}|present|current|now)",
        re.I,
    )
    # Pattern C: YYYY-MM date range like "2023-06 - present" or "2023-01 - 2023-05"
    ym_range_re = re.compile(
        r"(\d{4})[\-/](\d{1,2})"
        r"\s*[-–—to]+\s*"
        r"(?:(\d{4})[\-/](\d{1,2})|(present|current|now))",
        re.I,
    )
    import datetime
    current_year  = datetime.datetime.now().year
    current_month = datetime.datetime.now().month

    def _months_from_range(m: re.Match) -> int | None:
        try:
            y1 = int(m.group(1))
            y2_raw = m.group(2).lower()
            y2 = current_year if y2_raw in ("present", "current", "now") else int(y2_raw)
            return max(0, (y2 - y1) * 12)
        except Exception:
            return None

    def _months_from_ym_range(m: re.Match) -> int | None:
        """Compute months between YYYY-MM ... YYYY-MM (or present)."""
        try:
            y1, mo1 = int(m.group(1)), int(m.group(2))
            if m.group(5):  # 'present' / 'current' / 'now'
                y2, mo2 = current_year, current_month
            else:
                y2, mo2 = int(m.group(3)), int(m.group(4))
            total = (y2 - y1) * 12 + (mo2 - mo1)
            return max(1, total)
        except Exception:
            return None

    search_lines = exp_lines if exp_lines else lines

    for line in search_lines:
        stripped = line.strip()
        if not stripped or len(stripped) < 5:
            continue

        dur = None

        # Try pattern A first (explicit parenthetical durations)
        m = dur_paren_re.search(stripped)
        if m:
            y = float(m.group("y1") or 0)
            mo = float(m.group("m1") or 0)
            total = int(y * 12 + mo)
            if total > 0:
                dur = total

        # Try pattern B (named-month ranges)
        if dur is None:
            m2 = date_range_re.search(stripped)
            if m2:
                dur = _months_from_range(m2)

        # Try pattern C (YYYY-MM ranges — common in modern/ATS resumes)
        if dur is None:
            m3 = ym_range_re.search(stripped)
            if m3:
                dur = _months_from_ym_range(m3)

        if dur and dur > 0:
            # Extract title: text before first pipe, date, comma, or bracket
            title_part = re.split(r"[\|,\[\(]|[-–—]\s*(?:\d{4}|[A-Z])", stripped)[0]
            title = title_part.strip().rstrip("-–—").strip()
            # Remove leftover placeholders and excess whitespace
            title = re.sub(r"\[.*?\]", "", title).strip()
            title = re.sub(r"\s{2,}", " ", title).strip()
            if title and len(title) > 2 and len(title) < 80:
                result["job_history"].append({
                    "title": title,
                    "duration_months": dur,
                })

    # ── 3. Highest degree ─────────────────────────────────────────────────────
    edu_text = " ".join(edu_lines) if edu_lines else full_text
    for degree_label, pattern in _DEGREE_PATTERNS:
        if pattern.search(edu_text):
            result["highest_degree"] = degree_label
            break

    # ── 3b. Education entries (structured list) ───────────────────────────────
    # Patterns to parse degree lines like:
    #   "Bachelor of Science in Computer Science, [UNIVERSITY]"
    #   "Master of Engineering – Data Science | MIT"
    _DEGREE_LINE_RE = re.compile(
        r"(?P<degree>ph\.?d|doctor(?:ate)?|master(?:'?s)?|bachelor(?:'?s)?|b\.?s\.?|b\.?a\.?|m\.?s\.?|m\.?a\.?|mba|m\.?eng|b\.?eng|b\.?tech|associate)"
        r"(?:[^,\|\n]*?(?:of|in|:)\s*(?P<field>[^,\|\n\[]{3,60}))?",
        re.I,
    )
    # GPA / CGPA pattern: "GPA: 3.8/4.0", "CGPA 8.5", "8.5/10"
    _GPA_RE = re.compile(
        r"(?:c?gpa|grade\s+point)\s*[:\-]?\s*(?P<gpa>[\d\.]+)(?:\s*/\s*(?P<gpa_max>[\d\.]+))?",
        re.I,
    )
    # Percentage pattern: "85%", "85.5 percent"
    _PCT_RE = re.compile(
        r"(?P<pct>[\d\.]+)\s*(?:%|percent)",
        re.I,
    )
    # Grade/class keywords: "First Class", "Distinction", "Cum Laude", "Merit"
    _GRADE_KW_RE = re.compile(
        r"(?P<grade>first\s+class|second\s+class|third\s+class|distinction|"
        r"merit|pass|cum\s+laude|magna\s+cum\s+laude|summa\s+cum\s+laude|"
        r"with\s+honors?|honours?)",
        re.I,
    )

    education_entries = []
    search_edu = edu_lines if edu_lines else lines
    # Build a lookup: for each degree-line we also scan the next few lines for GPA/score
    search_edu_indexed = list(enumerate(search_edu))

    for idx, line in search_edu_indexed:
        stripped = line.strip()
        if not stripped or len(stripped) < 5:
            continue
        m = _DEGREE_LINE_RE.search(stripped)
        if m:
            degree_raw = m.group("degree").strip()
            # Normalise degree label
            degree_norm = "None stated"
            for deg_label, deg_pat in _DEGREE_PATTERNS:
                if deg_pat.search(degree_raw):
                    degree_norm = deg_label
                    break
            field_raw = (m.group("field") or "").strip().rstrip(",|–—-").strip()
            # Try to find institution: text after last comma or pipe that isn't a placeholder
            inst_match = re.split(r"[,\|]", stripped)
            institution = ""
            for part in reversed(inst_match):
                part = part.strip()
                if part and not re.match(r"^\[.*\]$", part) and len(part) > 2:
                    if not _DEGREE_LINE_RE.search(part):
                        institution = part
                        break

            # ── GPA / score extraction ──────────────────────────────────────
            # Scan the degree line itself + up to 3 following lines
            gpa_str = ""
            score_str = ""
            scan_window = [stripped] + [
                search_edu[j].strip()
                for j in range(idx + 1, min(idx + 4, len(search_edu)))
            ]
            for scan_line in scan_window:
                if not gpa_str:
                    gm = _GPA_RE.search(scan_line)
                    if gm:
                        gpa_val = gm.group("gpa")
                        gpa_max = gm.group("gpa_max")
                        gpa_str = f"{gpa_val}/{gpa_max}" if gpa_max else gpa_val
                if not score_str:
                    pm = _PCT_RE.search(scan_line)
                    if pm:
                        score_str = f"{pm.group('pct')}%"
                if not score_str:
                    gkm = _GRADE_KW_RE.search(scan_line)
                    if gkm:
                        score_str = gkm.group("grade").title()
                if gpa_str and score_str:
                    break

            education_entries.append({
                "degree": degree_norm,
                "field": field_raw,
                "institution": institution,
                "gpa": gpa_str,
                "score": score_str,
            })
    result["education"] = education_entries

    # ── 3c. Structured experience entries ────────────────────────────────────
    # Rich list mirroring job_history but also capturing company + date_range + type.
    _COMPANY_SPLIT_RE = re.compile(r"[\|@]")
    _INTERN_RE = re.compile(
        r"\b(intern(?:ship)?|trainee|placement|apprentice|co[-\s]?op)\b", re.I
    )
    experience_entries = []
    for job in result["job_history"]:
        entry_type = "Internship" if _INTERN_RE.search(job["title"]) else "Job"
        experience_entries.append({
            "title": job["title"],
            "company": "",
            "duration_months": job["duration_months"],
            "date_range": "",
            "type": entry_type,
        })

    # Second pass: scan experience lines to fill in company + date_range
    for i, line in enumerate(search_lines):
        stripped = line.strip()
        if not stripped:
            continue
        # Match duration
        dur = None
        m = dur_paren_re.search(stripped)
        if m:
            y = float(m.group("y1") or 0)
            mo = float(m.group("m1") or 0)
            dur = int(y * 12 + mo) or None
        if dur is None:
            m2 = date_range_re.search(stripped)
            if m2:
                dur = _months_from_range(m2)
        if dur is None:
            m3 = ym_range_re.search(stripped)
            if m3:
                dur = _months_from_ym_range(m3)
        if not dur:
            continue

        # Find matching experience entry by duration
        for exp_entry in experience_entries:
            if exp_entry["duration_months"] == dur and not exp_entry["company"]:
                # Extract date_range string
                dr = date_range_re.search(stripped)
                dr2 = ym_range_re.search(stripped)
                if dr:
                    exp_entry["date_range"] = dr.group(0)
                elif dr2:
                    exp_entry["date_range"] = dr2.group(0)

                # Refine type using the full original line (catches "Intern" even if title was trimmed)
                if _INTERN_RE.search(stripped):
                    exp_entry["type"] = "Internship"

                # Extract company: text after first pipe/@ separator
                parts = _COMPANY_SPLIT_RE.split(stripped)
                if len(parts) >= 2:
                    company_candidate = parts[1].strip()
                    # Strip dates, durations and placeholders
                    company_candidate = re.sub(r"\[.*?\]", "", company_candidate)
                    company_candidate = date_range_re.sub("", company_candidate)
                    company_candidate = ym_range_re.sub("", company_candidate)
                    company_candidate = dur_paren_re.sub("", company_candidate)
                    company_candidate = company_candidate.strip().rstrip(",–—-").strip()
                    if company_candidate and len(company_candidate) > 1:
                        exp_entry["company"] = company_candidate
                break
    result["experience"] = experience_entries

    # ── 4. Total years experience ─────────────────────────────────────────────
    # Try: "X years of experience / expertise"
    exp_match = re.search(
        r"(\d+)\+?\s+years?\s+(?:of\s+)?(?:experience|expertise|work|industry)",
        sanitized_text, re.I
    )
    if exp_match:
        result["total_years_experience"] = int(exp_match.group(1))
    else:
        total_months = sum(j["duration_months"] for j in result["job_history"])
        if total_months > 0:
            result["total_years_experience"] = max(1, round(total_months / 12))

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════════════

def structure_resume(sanitized_text: str) -> dict:
    """
    Main entry point.

    Takes GLiNER-sanitised resume text and returns a structured JSON dict.
    Uses the fine-tuned T5 model when it produces valid JSON; falls back to
    deterministic rule-based extraction otherwise.

    Args:
        sanitized_text: GLiNER-anonymised resume text.
                        Names → [PERSON], dates → [DATE], orgs → [ORGANIZATION].

    Returns:
        Dict conforming to the StructuredResume schema.
    """
    # 1. Pre-format to match training-data style
    formatted = preformat_resume(sanitized_text)
    print(f"[INFO] Pre-formatted resume ({len(formatted)} chars):\n"
          f"{formatted[:400]}{'…' if len(formatted) > 400 else ''}\n")

    # 2. Try fine-tuned T5 model
    print("[INFO] Running fine-tuned T5 model …")
    t5_result = _run_t5(formatted)

    if t5_result is not None:
        print("[INFO] T5 produced valid JSON — using model output.")
        return _validate_and_fill(t5_result)

    # 3. Fallback to rule-based extraction
    print("[INFO] T5 output was not valid JSON — using rule-based fallback.")
    fallback = _rule_based_fallback(formatted, sanitized_text)
    return _validate_and_fill(fallback)


def _validate_and_fill(data: dict) -> dict:
    """Ensure all required fields exist and highest_degree is a valid enum value."""
    data.setdefault("total_years_experience", None)
    data.setdefault("technical_skills", [])
    data.setdefault("job_history", [])
    data.setdefault("highest_degree", "None stated")
    data.setdefault("education", [])
    data.setdefault("experience", [])

    valid_degrees = {"High School", "Associate", "Bachelor", "Master", "PhD", "None stated"}
    if data["highest_degree"] not in valid_degrees:
        data["highest_degree"] = "None stated"

    # Filter out fake skills (long sentences, tips, or template text)
    cleaned_skills = []
    for skill in data["technical_skills"]:
        s = str(skill).strip()
        if not s: continue
        if "tip:" in s.lower(): break
        cleaned_skills.append(s)

    data["technical_skills"] = cleaned_skills

    # Ensure experience entries have all keys
    for exp in data["experience"]:
        exp.setdefault("title", "")
        exp.setdefault("company", "")
        exp.setdefault("duration_months", 0)
        exp.setdefault("date_range", "")
        exp.setdefault("type", "Job")

    # Ensure education entries have all keys
    for edu in data["education"]:
        edu.setdefault("degree", "None stated")
        edu.setdefault("field", "")
        edu.setdefault("institution", "")
        edu.setdefault("gpa", "")
        edu.setdefault("score", "")

    # ── Compute work_experience_summary from experience[] ───────────────────────────
    # Fall back to job_history if experience[] is empty (e.g. T5 model output path)
    source = data["experience"] if data["experience"] else [
        {"title": j["title"], "type": "Job", "duration_months": j["duration_months"], "date_range": ""}
        for j in data["job_history"]
    ]

    if source:
        jobs = [e for e in source if e.get("type", "Job") != "Internship"]
        interns = [e for e in source if e.get("type", "Job") == "Internship"]
        total_months = sum(e["duration_months"] for e in source)
        jobs_months = sum(e["duration_months"] for e in jobs)
        interns_months = sum(e["duration_months"] for e in interns)
        data["work_experience_summary"] = {
            "total_months": total_months,
            "total_years": round(total_months / 12, 1),
            "jobs_count": len(jobs),
            "jobs_months": jobs_months,
            "internships_count": len(interns),
            "internships_months": interns_months,
            "roles": [
                {
                    "title": e["title"],
                    "type": e.get("type", "Job"),
                    "duration_months": e["duration_months"],
                    "date_range": e.get("date_range", ""),
                }
                for e in source
            ],
        }
    else:
        data["work_experience_summary"] = None

    return data


# ── CLI Test ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = """
    Extremely hardworking and passionate backend engineer with 7 years of experience.

    SKILLS
    Python, FastAPI, PostgreSQL, Redis, Docker, Kubernetes, AWS

    WORK HISTORY
    Senior Backend Engineer | TechCorp | Jan 2021 - Present (3 years)
    Backend Engineer | StartupXYZ | Mar 2019 – Jan 2021 (2 years)
    Software Intern | MegaCorp | Jun 2018 – Aug 2018 (3 months)

    EDUCATION
    Bachelor of Science in Computer Science, State University
    GPA: 3.8/4.0
    """

    print("=== Pre-formatted ===")
    print(preformat_resume(sample))
    print("\n=== Structured output ===")
    print(json.dumps(structure_resume(sample), indent=2))
