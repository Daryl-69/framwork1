"""
Evaluator Agent (Bot 4 — Fine-tuned Phi-3.5-mini LoRA)
------------------------------------------------------
Loads the fine-tuned Phi-3.5-mini LoRA adapter from 'bot4/' and generates
objective, rubric-based scorecards comparing a candidate's structured JSON
profile against a Job Description.

Architecture
------------
  structured_json + job_description_text
      │
      ▼
  _build_eval_prompt()   ← formats into Phi-3.5 chat template
      │
      ▼
  Fine-tuned Phi-3.5 LoRA model   ← generates scorecard JSON
      │
      ├─ valid JSON? ──────► return scorecard dict
      │
      └─ invalid? ──────────► return empty scorecard dict

Output Schema
-------------
{
    "scorecard": [
        {
            "skill": str,
            "score": int (0-10),
            "justification": str
        }
    ],
    "overall_score": float,
    "recommendation": "Strong Match" | "Moderate Match" | "Weak Match" | "No Match"
}
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
_ADAPTER_PATH = _HERE.parent.parent / "bot4"
_BASE_MODEL_ID = "microsoft/Phi-3.5-mini-instruct"

# ── System prompt (must match training) ────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an objective scoring engine. Compare the candidate's JSON profile "
    "against the provided Job Description criteria. Assign a score from 0-10 for "
    "each required skill. Provide a 2-sentence purely factual justification for "
    "each score based only on the JSON data. Return a valid JSON object with "
    "scorecard, overall_score, and recommendation fields."
)

# ── Lazy model singleton ────────────────────────────────────────────────────────
_model = None
_tokenizer = None
_load_attempted = False


def _load_model():
    """
    Load the base Phi-3.5-mini-instruct with 4-bit quantization and apply
    the LoRA adapter from bot4/.
    Returns (model, tokenizer) or (None, None) if loading fails.
    """
    global _model, _tokenizer, _load_attempted

    if _load_attempted:
        return _model, _tokenizer

    _load_attempted = True

    if not _ADAPTER_PATH.exists() or not (_ADAPTER_PATH / "adapter_config.json").exists():
        print(f"[WARN] Evaluator adapter not found at {_ADAPTER_PATH}")
        _load_attempted = False  # Reset so it can retry
        return None, None

    try:
        import sys
        if str(_HERE) not in sys.path:
            sys.path.insert(0, str(_HERE))
        from structure_agent import unload_t5
        
        print("[INFO] Unloading T5 to free VRAM before loading Evaluator...")
        unload_t5()

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        print(f"[INFO] Loading evaluator tokenizer from '{_BASE_MODEL_ID}' …")
        _tokenizer = AutoTokenizer.from_pretrained(_BASE_MODEL_ID, trust_remote_code=True)
        _tokenizer.pad_token = _tokenizer.eos_token

        # 4-bit quantization for 16GB GPU / CPU fallback
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cuda":
            # RTX 3050 Laptop = 4 GB VRAM — cap GPU at 3 GB, spill rest to CPU
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_cap = f"{max(gpu_mem - 1.0, 2.0):.0f}GiB"  # leave ~1 GB headroom
            print(f"[INFO] Loading evaluator with 4-bit quant (GPU cap {gpu_cap}) …")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                _BASE_MODEL_ID,
                quantization_config=bnb_config,
                device_map="auto",
                max_memory={0: gpu_cap, "cpu": "6GiB"},
                trust_remote_code=True,
                attn_implementation="eager",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
        else:
            print("[INFO] Loading evaluator on CPU (no quantization) …")
            base_model = AutoModelForCausalLM.from_pretrained(
                _BASE_MODEL_ID,
                device_map="cpu",
                trust_remote_code=True,
                torch_dtype=torch.float32,
            )

        print(f"[INFO] Applying LoRA adapter from {_ADAPTER_PATH} …")
        _model = PeftModel.from_pretrained(base_model, str(_ADAPTER_PATH))
        _model.eval()

        print(f"[INFO] Evaluator model ready on {device.upper()}.")
        return _model, _tokenizer

    except Exception as e:
        print(f"[WARN] Failed to load evaluator model: {e}")
        _model = None
        _tokenizer = None
        _load_attempted = False  # Reset so it retries
        return None, None


def _build_eval_prompt(jd_text: str, candidate_json: dict) -> str:
    """
    Build the Phi-3.5 chat-templated prompt matching the training format.
    """
    candidate_str = json.dumps(candidate_json, indent=2)
    user_content = (
        f"Job Description:\n{jd_text}\n\n"
        f"Candidate JSON Profile:\n{candidate_str}"
    )
    prompt = (
        f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
        f"<|user|>\n{user_content}<|end|>\n"
        f"<|assistant|>\n"
    )
    return prompt


def _parse_scorecard_json(raw_text: str) -> Optional[dict]:
    """
    Try to extract a valid scorecard JSON object from model output.
    Handles markdown fences, trailing text, etc.
    """
    # Strip markdown code fences
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    # Try direct parse
    try:
        result = json.loads(cleaned)
        if "scorecard" in result:
            return result
    except json.JSONDecodeError:
        pass

    # Try finding a JSON object in the output
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if "scorecard" in result:
                return result
        except json.JSONDecodeError:
            pass

    return None


def _run_evaluator_model(jd_text: str, candidate_json: dict) -> Optional[dict]:
    """
    Run the fine-tuned Phi-3.5 LoRA model to generate a scorecard.
    Returns parsed dict or None if generation/parsing fails.
    """
    model, tokenizer = _load_model()
    if model is None or tokenizer is None:
        return None

    try:
        import torch

        prompt = _build_eval_prompt(jd_text, candidate_json)
        device = next(model.parameters()).device

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1536,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=800,
                temperature=0.1,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                cache_implementation="dynamic",
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        print(f"[INFO] Evaluator raw output (first 300 chars): {response[:300]}")

        return _parse_scorecard_json(response)

    except Exception as e:
        print(f"[WARN] Evaluator model inference failed: {e}")
        return None


def _validate_scorecard(data: dict) -> dict:
    """
    Ensure the scorecard has the expected structure with all required fields.
    Handles three formats the LLM may return:
      - list of dicts: [{"skill": ..., "score": ..., "justification": ...}]
      - dict of skill->score: {"Python": 8, "Docker": 5}
      - list of strings: ["Python", "Docker"]
    """
    scorecard = data.get("scorecard", [])
    validated_scorecard = []

    if isinstance(scorecard, dict):
        for skill_name, score_val in scorecard.items():
            try:
                score_int = max(0, min(10, int(float(score_val))))
            except (TypeError, ValueError):
                score_int = 0
            validated_scorecard.append({
                "skill": str(skill_name),
                "score": score_int,
                "justification": "No justification provided.",
            })
    elif isinstance(scorecard, list):
        for item in scorecard:
            if isinstance(item, dict):
                try:
                    score_int = max(0, min(10, int(item.get("score", 0))))
                except (TypeError, ValueError):
                    score_int = 0
                validated_scorecard.append({
                    "skill": str(item.get("skill", "Unknown")),
                    "score": score_int,
                    "justification": str(item.get("justification", "No justification provided.")),
                })
            elif isinstance(item, str):
                validated_scorecard.append({
                    "skill": item,
                    "score": 0,
                    "justification": "Invalid format returned by model.",
                })

    # Compute overall_score if not provided or invalid
    overall = data.get("overall_score")
    if validated_scorecard and (overall is None or not isinstance(overall, (int, float))):
        overall = round(
            sum(s["score"] for s in validated_scorecard) / len(validated_scorecard), 1
        )

    # Determine recommendation based on overall score
    recommendation = data.get("recommendation", "")
    if not recommendation or recommendation not in {
        "Strong Match", "Moderate Match", "Weak Match", "No Match"
    }:
        if overall is not None:
            if overall >= 7.5:
                recommendation = "Strong Match"
            elif overall >= 5.0:
                recommendation = "Moderate Match"
            elif overall >= 2.5:
                recommendation = "Weak Match"
            else:
                recommendation = "No Match"
        else:
            recommendation = "No Match"

    return {
        "scorecard": validated_scorecard,
        "overall_score": overall or 0,
        "recommendation": recommendation,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_resume(jd_text: str, candidate_json: dict) -> dict:
    """
    Main entry point for the evaluator agent.

    Takes a Job Description text and a structured candidate JSON profile,
    and returns a scorecard comparing the candidate against the JD requirements.

    1. Tries the fine-tuned Phi-3.5-mini LoRA model first.
    2. Falls back to Groq Llama 3.3 70B API if the local model fails.

    Args:
        jd_text:        The job description text (from the JD Builder).
        candidate_json: The structured resume JSON (output of structure_resume).

    Returns:
        Dict with scorecard, overall_score, and recommendation fields.
    """
    print("[INFO] Running evaluator agent …")

    # 1. Try fine-tuned model
    result = _run_evaluator_model(jd_text, candidate_json)
    if result is not None:
        print("[INFO] Evaluator model produced valid scorecard.")
        return _validate_scorecard(result)

    # 2. Last resort: return empty scorecard
    print("[WARN] Evaluator model unavailable or output invalid — returning empty scorecard.")
    return {
        "scorecard": [],
        "overall_score": 0,
        "recommendation": "No Match",
    }
