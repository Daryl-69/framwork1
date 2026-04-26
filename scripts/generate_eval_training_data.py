"""
Generate Evaluator Agent (Bot 4) Training Data — Multi-Key Edition
===================================================================
Uses 7 Groq API keys with automatic rotation on rate limits.
Resumes are processed from where they left off (crash-safe).
Run: python scripts/generate_eval_training_data.py
"""

import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from groq import Groq

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE         = Path(__file__).parent
_PROJECT_ROOT = _HERE.parent
_INPUT_FILE   = _PROJECT_ROOT / "data" / "t5_training_data.jsonl"
_OUTPUT_FILE  = _PROJECT_ROOT / "data" / "evaluator_training_data.jsonl"

MODEL = "llama-3.3-70b-versatile"
DELAY = 1.5   # seconds between calls

# ── All Groq API Keys (comma-separated in .env) ──────────────────────────────
_GROQ_KEYS = [k.strip() for k in os.environ.get("GROQ_API_KEYS", os.environ.get("GROQ_API_KEY", "")).split(",") if k.strip()]
_key_idx = 0


def _get_client() -> Groq:
    return Groq(api_key=_GROQ_KEYS[_key_idx])


def _rotate_key():
    global _key_idx
    _key_idx = (_key_idx + 1) % len(_GROQ_KEYS)
    print(f"  [KEY {_key_idx + 1}/{len(_GROQ_KEYS)}] Switched.", end=" ", flush=True)


# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a training-data generator for an AI resume-scoring system.
You will receive a candidate's resume (raw text + structured JSON).
Your job is to produce TWO things:

1. A realistic Job Description (JD) that PARTIALLY matches this candidate.
   - Include 5-8 required skills/qualifications.
   - Make sure 2-4 of them are things the candidate clearly has.
   - Make sure 2-4 of them are things the candidate does NOT have or has weak evidence for.
   - This ensures varied scores (not all 10s or all 0s).
   - The JD should sound like a real job posting.

2. A Scorecard scoring the candidate against YOUR generated JD.
   - For each required skill: "skill", "score" (int 0-10), "justification" (exactly 2 factual sentences).
   - "overall_score": average of all scores, 1 decimal.
   - "recommendation": "Strong Match" (>=7), "Partial Match" (>=4), or "Weak Match" (<4).

Return ONLY this JSON, no markdown fences, no extra text:
{
  "job_description": "...",
  "scorecard": [{"skill": "Python", "score": 8, "justification": "Two factual sentences."}, ...],
  "overall_score": 5.5,
  "recommendation": "Partial Match"
}

RULES: Scores = integers 0-10. Justifications = exactly 2 sentences. Mix high and low scores."""


# ── Core functions ────────────────────────────────────────────────────────────

def load_existing_progress() -> set:
    done = set()
    if _OUTPUT_FILE.exists():
        with open(_OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if "line_index" in obj:
                        done.add(obj["line_index"])
                except json.JSONDecodeError:
                    continue
    return done


def generate_one(resume_text: str, structured_json: dict) -> dict | None:
    candidate_json_str = json.dumps(structured_json, indent=2)
    user_prompt = f"""Here is the candidate's data:

--- RAW RESUME TEXT ---
{resume_text[:2000]}

--- STRUCTURED JSON ---
{candidate_json_str}

Generate the Job Description and Scorecard now. Return ONLY valid JSON."""

    try:
        client = _get_client()
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
        print(f"  [WARN] JSON: {e}")
        return None
    except Exception as e:
        err = str(e)
        if "429" in err or "rate" in err.lower() or "quota" in err.lower():
            _rotate_key()
            time.sleep(2)
            return None
        print(f"  [ERROR] {e}")
        return None


def format_training_example(resume_json, jd_text, scorecard, overall_score, recommendation):
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not _INPUT_FILE.exists():
        print(f"[ERROR] Input not found: {_INPUT_FILE}")
        sys.exit(1)

    with open(_INPUT_FILE, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
    total = len(all_lines)

    done_indices = load_existing_progress()

    print(f"[INFO] Resumes     : {total}")
    print(f"[INFO] Already done: {len(done_indices)}")
    print(f"[INFO] Remaining   : {total - len(done_indices)}")
    print(f"[INFO] Keys        : {len(_GROQ_KEYS)} (rotating on rate limit)")
    print(f"[INFO] Output      : {_OUTPUT_FILE}")
    print("=" * 60)

    success = len(done_indices)
    failed  = 0

    for idx, raw_line in enumerate(all_lines):
        if idx in done_indices:
            continue

        try:
            data = json.loads(raw_line)
        except json.JSONDecodeError:
            print(f"[{idx+1}/{total}] Skipping — bad JSON")
            failed += 1
            continue

        resume_text = data.get("input_text", "").replace("Extract JSON from this resume:\n", "").strip()
        target_text = data.get("target_text", "{}")

        try:
            structured_json = json.loads(target_text)
        except json.JSONDecodeError:
            print(f"[{idx+1}/{total}] Skipping — bad target JSON")
            failed += 1
            continue

        if len(resume_text) < 50:
            print(f"[{idx+1}/{total}] Skipping — too sparse")
            failed += 1
            continue

        print(f"[{idx+1}/{total}] Generating...", end=" ", flush=True)

        result = None
        for attempt in range(len(_GROQ_KEYS) * 2):
            result = generate_one(resume_text, structured_json)
            if result:
                break
            time.sleep(DELAY)

        if not result:
            print("FAILED")
            failed += 1
            continue

        example = format_training_example(
            resume_json=structured_json,
            jd_text=result["job_description"],
            scorecard=result["scorecard"],
            overall_score=result["overall_score"],
            recommendation=result["recommendation"],
        )
        example["line_index"] = idx

        with open(_OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

        success += 1
        print(f"OK  ({len(result['scorecard'])} skills, avg={result['overall_score']}, {result['recommendation']})")
        time.sleep(DELAY)

    print("=" * 60)
    print(f"[DONE] Success: {success} | Failed: {failed}")
    print(f"[DONE] Output : {_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
