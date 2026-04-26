import gc
import os
import re
import shutil
import tempfile
from pathlib import Path

# Load .env from project root (two levels up from backend/)
from dotenv import load_dotenv
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import nest_asyncio

# Apply nest_asyncio to allow nested event loops which llama_parse requires.
nest_asyncio.apply()

from llama_parse import LlamaParse
from gliner import GLiNER
from structure_agent import structure_resume
from evaluator_agent import evaluate_resume


# ── Candidate Role Inference ────────────────────────────────────────────────
# Lightweight rule-based role classifier — no ML needed, runs instantly.

_ROLE_RULES: list[tuple[str, list[str]]] = [
    # (Role Label, [keywords that strongly signal this role])
    ("AI / ML Engineer",          ["machine learning", "deep learning", "tensorflow", "pytorch", "keras", "nlp", "llm", "neural network", "data science", "scikit", "transformers"]),
    ("Data Scientist",            ["data scientist", "statistical modeling", "r studio", "pandas", "numpy", "data analysis", "visualization", "tableau", "power bi", "looker", "a/b testing"]),
    ("Data Engineer",             ["data engineer", "etl", "pipeline", "spark", "hadoop", "airflow", "kafka", "dbt", "data warehouse", "redshift", "snowflake", "databricks"]),
    ("Backend Engineer",          ["backend", "api", "fastapi", "django", "flask", "spring", "node.js", "express", "graphql", "rest", "microservices", "postgresql", "mysql", "redis", "rabbitmq"]),
    ("Frontend Engineer",         ["frontend", "react", "vue", "angular", "next.js", "svelte", "html", "css", "sass", "tailwind", "webpack", "ui", "ux", "figma", "sketch"]),
    ("Full Stack Engineer",       ["full stack", "fullstack", "full-stack", "mern", "mean", "lamp"]),
    ("Mobile Developer",          ["android", "ios", "swift", "kotlin", "flutter", "react native", "mobile", "xcode"]),
    ("DevOps / Cloud Engineer",   ["devops", "ci/cd", "docker", "kubernetes", "terraform", "ansible", "aws", "azure", "gcp", "jenkins", "github actions", "infrastructure", "site reliability"]),
    ("Cybersecurity Engineer",    ["security", "penetration testing", "ethical hacking", "soc", "siem", "firewall", "vulnerability", "owasp", "cryptography", "iam", "zero trust"]),
    ("Embedded / Systems Engineer",["embedded", "firmware", "rtos", "c++", "assembly", "microcontroller", "fpga", "verilog", "uart", "spi", "i2c"]),
    ("UI / UX Designer",          ["ui design", "ux design", "user research", "wireframe", "prototype", "adobe xd", "figma", "usability", "interaction design"]),
    ("Product Manager",           ["product manager", "product management", "roadmap", "stakeholder", "agile", "scrum", "sprint", "user story", "okr", "go-to-market"]),
    ("QA / Test Engineer",        ["qa", "quality assurance", "test engineer", "selenium", "cypress", "playwright", "jest", "pytest", "junit", "test automation", "regression"]),
    ("Software Engineer",         ["software engineer", "software developer", "programming", "algorithms", "data structures", "object-oriented", "oop", "git"]),
]

# Title-to-role direct mapping (checked first for speed)
_TITLE_MAP: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(ml|machine\s*learning|ai|artificial\s*intelligence|data\s*scientist)\b", re.I), "AI / ML Engineer"),
    (re.compile(r"\bdata\s*engineer\b", re.I),            "Data Engineer"),
    (re.compile(r"\b(front[- ]?end|ui\s*developer|react\s*dev)\b", re.I), "Frontend Engineer"),
    (re.compile(r"\b(back[- ]?end)\b", re.I),             "Backend Engineer"),
    (re.compile(r"\bfull[- ]?stack\b", re.I),             "Full Stack Engineer"),
    (re.compile(r"\b(devops|cloud\s*engineer|sre|site\s*reliability)\b", re.I), "DevOps / Cloud Engineer"),
    (re.compile(r"\b(android|ios|mobile)\s*(developer|engineer)?\b", re.I), "Mobile Developer"),
    (re.compile(r"\b(security|cyber|pentest)\b", re.I),  "Cybersecurity Engineer"),
    (re.compile(r"\b(embedded|firmware|hardware)\b", re.I), "Embedded / Systems Engineer"),
    (re.compile(r"\b(product\s*manager|pm\b)", re.I),    "Product Manager"),
    (re.compile(r"\b(qa|test\s*engineer|quality)", re.I), "QA / Test Engineer"),
    (re.compile(r"\b(ux|ui)\s*(designer|design)\b", re.I), "UI / UX Designer"),
]


def infer_candidate_role(structured_data: dict) -> str:
    """
    Quickly infer what role/specialisation the candidate in the resume fits.
    Uses job titles first (fast path), then falls back to keyword scoring
    over technical skills and experience titles.
    Returns a human-readable role string like 'Backend Engineer'.
    """
    # Collect all text: job titles + skills
    titles = []
    for exp in structured_data.get("experience", []):
        if exp.get("title"):
            titles.append(exp["title"])
    for job in structured_data.get("job_history", []):
        if job.get("title"):
            titles.append(job["title"])

    # Fast path: direct title pattern match
    for title in titles:
        for pattern, role in _TITLE_MAP:
            if pattern.search(title):
                return role

    # Keyword scoring over skills + titles combined
    skills = structured_data.get("technical_skills", [])
    all_text = " ".join(titles + skills).lower()

    best_role = "Software Engineer"
    best_score = 0
    for role_label, keywords in _ROLE_RULES:
        score = sum(1 for kw in keywords if kw in all_text)
        if score > best_score:
            best_score = score
            best_role = role_label

    return best_role

# ── Optional: Tesseract OCR for image fallback ─────────────────────────────
try:
    from PIL import Image
    import pytesseract

    # Auto-detect Tesseract binary on Windows (check common install paths)
    _TESSERACT_WIN_PATHS = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for _tpath in _TESSERACT_WIN_PATHS:
        if os.path.exists(_tpath):
            pytesseract.pytesseract.tesseract_cmd = _tpath
            print(f"[INFO] Tesseract found at: {_tpath}")
            break
    else:
        print("[INFO] Tesseract not at default paths — relying on system PATH.")

    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False
    print("[WARN] pytesseract/Pillow not installed — image OCR fallback disabled.")

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def _ocr_image(filepath: str) -> str:
    """
    Extract text from an image file using Tesseract OCR.
    Requires the Tesseract binary to be installed separately:
      Windows: https://github.com/UB-Mannheim/tesseract/wiki
      macOS:   brew install tesseract
      Linux:   sudo apt install tesseract-ocr
    """
    if not _OCR_AVAILABLE:
        raise RuntimeError(
            "pytesseract and/or Pillow is not installed. "
            "Run: pip install pytesseract Pillow"
        )
    img = Image.open(filepath)
    # --psm 6: assume a single uniform block of text (good for most resume layouts)
    text = pytesseract.image_to_string(img, config="--psm 6")
    return text.strip()

app = FastAPI(title="Resume Scanner API")

# Allow all origins so CORS works even if the backend crashes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API key from .env (never hardcode secrets)
LLAMA_CLOUD_API_KEY = os.environ.get("LLAMA_CLOUD_API_KEY", "")

# Initialize GLiNER lazily to avoid OOM at startup
_gliner_model = None

def _get_gliner():
    global _gliner_model
    if _gliner_model is None:
        print("[INFO] Loading GLiNER model (first request) …")
        _gliner_model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
        print("[INFO] GLiNER model ready.")
    return _gliner_model


def _unload_gliner():
    """Free GLiNER model memory before loading T5 / other heavy models."""
    global _gliner_model
    if _gliner_model is not None:
        del _gliner_model
        _gliner_model = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        print("[INFO] GLiNER model unloaded to free memory.")


@app.post("/api/scan-resume")
async def scan_resume(file: UploadFile = File(...)):
    filename_lower = file.filename.lower()
    allowed_extensions = (".pdf", ".png", ".jpg", ".jpeg")
    if not filename_lower.endswith(allowed_extensions):
        raise HTTPException(status_code=400, detail="Only PDF, PNG, JPG, and JPEG files are supported.")

    ext = os.path.splitext(filename_lower)[1]

    # On Windows, NamedTemporaryFile must be closed before another process reads it.
    # Use delete=False, close it, then parse, then clean up manually.
    temp_filepath = None
    try:
        # Read content into memory first
        file_bytes = await file.read()

        # Write to a temp file (closed immediately after write so LlamaParse can open it on Windows)
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(file_bytes)
            temp_filepath = tmp.name

        print(f"[INFO] Temp file written: {temp_filepath}")

        parser = LlamaParse(
            api_key=LLAMA_CLOUD_API_KEY,
            result_type="markdown",  # "markdown" or "text"
        )

        parsed_documents = await parser.aload_data(temp_filepath)

        text = "\n\n".join([doc.text for doc in parsed_documents]).strip()

        # ── Image OCR fallback ────────────────────────────────────────────
        # LlamaParse may return empty/minimal text for image-based resumes.
        # If that happens, fall back to local Tesseract OCR.
        is_image = ext in _IMAGE_EXTENSIONS
        if is_image and not text:
            print("[INFO] LlamaParse returned empty for image — trying OCR fallback …")
            try:
                text = _ocr_image(temp_filepath)
                print(f"[INFO] OCR extracted {len(text)} chars.")
            except Exception as ocr_err:
                print(f"[WARN] OCR failed: {ocr_err}")

        if not text:
            raise HTTPException(
                status_code=500,
                detail="Could not extract text from the file. "
                       "For images, ensure the file is a clear, high-resolution scan.",
            )

        # ── Step 2: GLiNER anonymisation ──────────────────────────────────
        # Hide sensitive data to prevent AI bias in downstream models.
        # NOTE: "Date" is intentionally excluded from labels below.
        # Redacting all dates would erase employment date ranges (e.g. "Jan 2021 – Mar 2023"),
        # making it impossible for the structure agent to compute job durations.
        # Bias-sensitive dates (e.g. date of birth) are rare on modern resumes and
        # are usually co-located with Person/Location entities that ARE redacted.
        labels = [
            "Person", "Location", "Email", "Phone",
            "Address", "Organization", "Nationality", "Gender", "University"
        ]

        sanitized_lines = []
        for line in text.split('\n'):
            if not line.strip():
                sanitized_lines.append(line)
                continue

            entities = _get_gliner().predict_entities(line, labels, threshold=0.45)

            if not entities:
                sanitized_lines.append(line)
                continue

            # Sort spans by start position; drop overlapping spans (keep longest)
            entities_sorted = sorted(entities, key=lambda e: (e["start"], -(e["end"] - e["start"])))
            non_overlapping = []
            last_end = -1
            for ent in entities_sorted:
                if ent["start"] >= last_end:
                    non_overlapping.append(ent)
                    last_end = ent["end"]

            # Rebuild the line by substituting spans with placeholders
            result_chars = []
            cursor = 0
            for ent in non_overlapping:
                result_chars.append(line[cursor:ent["start"]])
                result_chars.append(f"[{ent['label'].upper()}]")
                cursor = ent["end"]
            result_chars.append(line[cursor:])
            sanitized_lines.append("".join(result_chars))

        sanitized_text = "\n".join(sanitized_lines)

        # ── Free GLiNER memory before loading T5 ──────────────────────────
        _unload_gliner()

        # ── Step 3: Structure Agent ────────────────────────────────────────
        # Converts sanitized text into strict, bias-free JSON.
        # Powered by few-shot examples from training_data.jsonl.
        print("[INFO] Running structure agent...")
        structured_data = structure_resume(sanitized_text)
        print(f"[INFO] Structured data: {structured_data}")

        # ── Step 4: Quick role inference ──────────────────────────────────
        candidate_role = infer_candidate_role(structured_data)
        print(f"[INFO] Inferred candidate role: {candidate_role}")

        return {
            "filename": file.filename,
            "sanitized_text": sanitized_text,
            "structured_data": structured_data,
            "candidate_role": candidate_role,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Parsing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Parsing error: {str(e)}")
    finally:
        if temp_filepath and os.path.exists(temp_filepath):
            os.remove(temp_filepath)
            print(f"[INFO] Temp file cleaned up: {temp_filepath}")


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Resume Scanner API is running!"}


@app.post("/api/quick-role")
async def quick_role(payload: dict = Body(...)):
    """
    Lightweight endpoint: given a pre-computed structured_data dict,
    returns the inferred candidate role instantly (no file parsing needed).
    """
    structured_data = payload.get("structured_data", {})
    role = infer_candidate_role(structured_data)
    return {"candidate_role": role}


@app.post("/api/evaluate")
async def evaluate_candidate(payload: dict = Body(...)):
    """
    Evaluate a candidate against a Job Description.
    Expects:
      - jd_text: str (the job description text)
      - structured_data: dict (the structured resume JSON from scan-resume)
    Returns:
      - scorecard, overall_score, recommendation
    """
    jd_text = payload.get("jd_text", "").strip()
    structured_data = payload.get("structured_data", {})

    if not jd_text:
        raise HTTPException(status_code=400, detail="jd_text is required.")
    if not structured_data:
        raise HTTPException(status_code=400, detail="structured_data is required.")

    # Build a minimal candidate profile for the evaluator
    candidate_profile = {
        "total_years_experience": structured_data.get("total_years_experience"),
        "technical_skills": structured_data.get("technical_skills", []),
        "job_history": structured_data.get("job_history", []),
        "highest_degree": structured_data.get("highest_degree", "None stated"),
    }

    print(f"[INFO] Evaluating candidate against JD: {jd_text[:100]}...")
    scorecard = evaluate_resume(jd_text, candidate_profile)
    print(f"[INFO] Evaluation result: overall={scorecard.get('overall_score')}, rec={scorecard.get('recommendation')}")

    return scorecard


@app.post("/api/suggest-skills")
async def suggest_skills(payload: dict = Body(...)):
    """
    Given a job role title, returns 6-8 typical required skills
    using Groq Llama 3.3 70B. Powers the JD Builder autocomplete.
    """
    role_title = payload.get("role_title", "").strip()
    if not role_title:
        raise HTTPException(status_code=400, detail="role_title is required.")

    try:
        from groq import Groq
        import json as _json

        GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
        client = Groq(api_key=GROQ_API_KEY)

        prompt = (
            f"List exactly 8 typical required technical skills or qualifications "
            f"for a '{role_title}' job position. "
            f"Return ONLY a JSON array of strings, nothing else. "
            f"Example: [\"Python\", \"REST APIs\", \"PostgreSQL\", \"Docker\", "
            f"\"System Design\", \"Git\", \"AWS\", \"Microservices\"]"
        )

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=200,
        )

        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        skills = _json.loads(raw)
        if not isinstance(skills, list):
            raise ValueError("Not a list")

        return {"skills": [str(s).strip() for s in skills if s]}

    except Exception as e:
        print(f"[WARN] suggest-skills failed: {e}")
        # Return generic fallback so the UI never breaks
        return {"skills": [
            "Communication Skills", "Problem Solving", "Team Collaboration",
            "5+ Years Experience", "Bachelor's Degree", "Leadership"
        ]}
