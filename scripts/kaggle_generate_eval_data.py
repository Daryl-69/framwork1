# ╔══════════════════════════════════════════════════════════════════╗
# ║  Bot 4 — Evaluator Training Data Generator (Kaggle Version)     ║
# ║  Reads t5_training_data.jsonl → calls Groq → saves scorecard    ║
# ╚══════════════════════════════════════════════════════════════════╝
#
# HOW TO USE ON KAGGLE:
# ─────────────────────
# 1. Create a new Kaggle Notebook
# 2. Add your dataset: Upload t5_training_data.jsonl as a Kaggle Dataset
#    (Datasets → New Dataset → upload the file → name it "ai-model-data")
# 3. Add your Groq key as a Kaggle Secret:
#    (Notebook → Add-ons → Secrets → Add → Name: GROQ_API_KEY, Value: your_key)
# 4. Paste this entire file into a single code cell and run it.
# 5. When done, the output file is at /kaggle/working/evaluator_training_data.jsonl
#    → Download it from the Output tab on the right side.
#
# NOTE: The 53 entries already generated locally are embedded below.
# The script will skip those line indices automatically.

# ── Cell 1: Install & Import ──────────────────────────────────────────────────

import subprocess
subprocess.run(["pip", "install", "groq", "-q"])

import json
import os
import re
import time
from pathlib import Path
from kaggle_secrets import UserSecretsClient

# ── Cell 2: Config ────────────────────────────────────────────────────────────

# ── GROQ KEYS ────────────────────────────────────────────────────
# Set GROQ_API_KEYS env var (comma-separated) or add keys to the list below.
# On Kaggle, use Secrets: Add-ons → Secrets → GROQ_API_KEYS
_env_keys = os.environ.get("GROQ_API_KEYS", os.environ.get("GROQ_API_KEY", ""))
GROQ_KEYS = [k.strip() for k in _env_keys.split(",") if k.strip()]
if not GROQ_KEYS:
    raise RuntimeError("No Groq API keys found. Set GROQ_API_KEYS env var.")
# ─────────────────────────────────────────────────────────────────

# Paths — adjust dataset name if yours is different
INPUT_FILE  = Path("/kaggle/input/ai-model-data/t5_training_data.jsonl")
OUTPUT_FILE = Path("/kaggle/working/evaluator_training_data.jsonl")

MODEL = "llama-3.3-70b-versatile"
DELAY = 2.5   # seconds between calls (lower = faster with multiple keys)

# Line indices already completed locally (skip these)
ALREADY_DONE = set(range(0, 53))  # lines 0-52 done locally

# ── Key rotator ───────────────────────────────────────────────────
_key_index = 0

def get_client():
    """Return a Groq client using the current key."""
    from groq import Groq
    return Groq(api_key=GROQ_KEYS[_key_index])

def rotate_key():
    """Switch to the next available key. Returns False if all keys exhausted."""
    global _key_index
    _key_index += 1
    if _key_index >= len(GROQ_KEYS):
        _key_index = 0  # wrap around
        return False  # full cycle done, all keys are rate-limited
    print(f"  [KEY ROTATE] Switching to key {_key_index + 1}/{len(GROQ_KEYS)}")
    return True

# ── Cell 3: Prompt ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a training-data generator for an AI resume-scoring system.
You will receive a candidate's resume (raw text + structured JSON).
Your job is to produce TWO things:

1. A realistic Job Description (JD) that PARTIALLY matches this candidate.
   - Include 5-8 required skills/qualifications.
   - Make sure 2-4 of them are things the candidate clearly has.
   - Make sure 2-4 of them are things the candidate does NOT have or has weak evidence for.
   - This ensures varied scores (not all 10s or all 0s).
   - The JD should sound like a real job posting (role title, company context, requirements).

2. A Scorecard scoring the candidate against YOUR generated JD.
   - For each required skill/qualification:
     - "skill": skill name
     - "score": integer 0-10
     - "justification": exactly 2 sentences. Purely factual, based only on the JSON.
   - Also: "overall_score" (average, 1 decimal), "recommendation":
     - "Strong Match"  if overall_score >= 7
     - "Partial Match" if overall_score >= 4
     - "Weak Match"    if overall_score < 4

Return ONLY this JSON object, no markdown fences, no extra text:
{
  "job_description": "...",
  "scorecard": [
    {"skill": "Python", "score": 8, "justification": "Two factual sentences."},
    ...
  ],
  "overall_score": 5.5,
  "recommendation": "Partial Match"
}

RULES: Scores must be integers 0-10. Justifications = exactly 2 sentences.
Do NOT give all high or all low scores. Aim for a MIX."""

# ── Cell 4: Helper functions ──────────────────────────────────────────────────

from groq import Groq

client = Groq(api_key=GROQ_KEYS[0])

def generate_one(resume_text: str, structured_json: dict) -> dict | None:
    candidate_json_str = json.dumps(structured_json, indent=2)
    user_prompt = f"""Here is the candidate's data:

--- RAW RESUME TEXT ---
{resume_text[:2000]}

--- STRUCTURED JSON ---
{candidate_json_str}

Generate the Job Description and Scorecard now. Return ONLY valid JSON."""

    try:
        client = get_client()  # uses current active key
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.8,
            max_tokens=2048,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        parsed = json.loads(raw)
        required = {"job_description", "scorecard", "overall_score", "recommendation"}
        if not required.issubset(parsed.keys()):
            print(f"  [WARN] Missing keys: {required - set(parsed.keys())}")
            return None
        for entry in parsed["scorecard"]:
            if not all(k in entry for k in ("skill", "score", "justification")):
                return None
            if not isinstance(entry["score"], (int, float)):
                entry["score"] = int(entry["score"])
        return parsed

    except json.JSONDecodeError as e:
        print(f"  [WARN] JSON parse error: {e}")
        return None
    except Exception as e:
        err = str(e)
        if "429" in err or "rate" in err.lower() or "quota" in err.lower():
            rotated = rotate_key()
            if rotated:
                print(f"  [RATE LIMIT] Key rotated → retrying…")
                time.sleep(2)
            else:
                # All keys exhausted — wait 60s then try again from key 0
                print(f"  [RATE LIMIT] All keys exhausted. Waiting 60s…")
                time.sleep(60)
            return None
        print(f"  [ERROR] {e}")
        return None


def format_example(resume_json, jd_text, scorecard, overall_score, recommendation):
    candidate_json_str = json.dumps(resume_json, indent=2)
    input_text = (
        "You are an objective scoring engine. Compare the candidate's JSON profile "
        "against the provided Job Description criteria. Assign a score from 0-10 for "
        "each required skill. Provide a 2-sentence purely factual justification for "
        "each score based only on the JSON data.\n\n"
        f"Job Description:\n{jd_text}\n\n"
        f"Candidate JSON Profile:\n{candidate_json_str}"
    )
    output_text = json.dumps({
        "scorecard": scorecard,
        "overall_score": overall_score,
        "recommendation": recommendation,
    })
    return {"input": input_text, "output": output_text}


def load_done_indices() -> set:
    done = set(ALREADY_DONE)  # start with locally-completed ones
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if "line_index" in obj:
                        done.add(obj["line_index"])
                except:
                    pass
    return done

# ── Cell 5: Main loop ─────────────────────────────────────────────────────────

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    all_lines = f.readlines()

total = len(all_lines)
done_indices = load_done_indices()
print(f"[INFO] Total resumes : {total}")
print(f"[INFO] Already done  : {len(done_indices)} (skipping)")
print(f"[INFO] Remaining     : {total - len(done_indices)}")
print(f"[INFO] Output        : {OUTPUT_FILE}")
print("=" * 60)

success = len(done_indices)
failed  = 0

for idx, raw_line in enumerate(all_lines):
    if idx in done_indices:
        continue

    try:
        data = json.loads(raw_line)
    except:
        print(f"[{idx+1}/{total}] Skipping — bad JSON")
        failed += 1
        continue

    resume_text = data.get("input_text", "").replace("Extract JSON from this resume:\n", "").strip()
    target_text = data.get("target_text", "{}")

    try:
        structured_json = json.loads(target_text)
    except:
        print(f"[{idx+1}/{total}] Skipping — bad target JSON")
        failed += 1
        continue

    if len(resume_text) < 50:
        print(f"[{idx+1}/{total}] Skipping — too sparse")
        failed += 1
        continue

    print(f"[{idx+1}/{total}] Generating…", end=" ", flush=True)

    result = None
    for attempt in range(3):
        result = generate_one(resume_text, structured_json)
        if result:
            break
        if attempt < 2:
            print(f"retry {attempt+2}…", end=" ", flush=True)
            time.sleep(DELAY * 2)

    if not result:
        print("FAILED")
        failed += 1
        continue

    example = format_example(
        resume_json   = structured_json,
        jd_text       = result["job_description"],
        scorecard     = result["scorecard"],
        overall_score = result["overall_score"],
        recommendation= result["recommendation"],
    )
    example["line_index"] = idx  # for crash-resume support

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(example, ensure_ascii=False) + "\n")

    success += 1
    print(f"OK  ({len(result['scorecard'])} skills, avg={result['overall_score']}, {result['recommendation']})")
    time.sleep(DELAY)

print("=" * 60)
print(f"[DONE] Success: {success} | Failed: {failed}")
print(f"[DONE] File: {OUTPUT_FILE}")
