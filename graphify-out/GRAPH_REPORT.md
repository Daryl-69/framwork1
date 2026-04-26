# Graph Report - 1_ai-model  (2026-04-26)

## Corpus Check
- 14 files · ~19,053 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 70 nodes · 68 edges · 14 communities detected
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 4 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]

## God Nodes (most connected - your core abstractions)
1. `structure_resume()` - 7 edges
2. `infer_candidate_role()` - 4 edges
3. `scan_resume()` - 4 edges
4. `preformat_resume()` - 4 edges
5. `_run_t5()` - 4 edges
6. `_rule_based_fallback()` - 4 edges
7. `_validate_and_fill()` - 4 edges
8. `main()` - 4 edges
9. `generate_one()` - 4 edges
10. `main()` - 4 edges

## Surprising Connections (you probably didn't know these)
- `scan_resume()` --calls--> `structure_resume()`  [INFERRED]
  resume_scanner\backend\main.py → resume_scanner\backend\structure_agent.py
- `preformat_resume()` --calls--> `main()`  [INFERRED]
  resume_scanner\backend\structure_agent.py → scripts\convert_dataset.py
- `_rule_based_fallback()` --calls--> `main()`  [INFERRED]
  resume_scanner\backend\structure_agent.py → scripts\convert_dataset.py
- `_validate_and_fill()` --calls--> `main()`  [INFERRED]
  resume_scanner\backend\structure_agent.py → scripts\convert_dataset.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.17
Nodes (15): main(), _extract_duration_months(), _load_model(), preformat_resume(), Structure Agent (v2 — Fine-tuned T5 / Bot 3) -----------------------------------, Normalise sanitised resume text to match the training-data style exactly.      T, Feed pre-formatted resume text into the fine-tuned T5 model.     Returns a parse, Parse a duration string like '3 years' / '18 months' / '2 years 6 months'. (+7 more)

### Community 1 - "Community 1"
Cohesion: 0.21
Nodes (10): infer_candidate_role(), _ocr_image(), quick_role(), Extract text from an image file using Tesseract OCR.     Requires the Tesseract, # NOTE: "Date" is intentionally excluded from labels below., Lightweight endpoint: given a pre-computed structured_data dict,     returns the, Given a job role title, returns 6-8 typical required skills     using Groq Llama, Quickly infer what role/specialisation the candidate in the resume fits.     Use (+2 more)

### Community 2 - "Community 2"
Cohesion: 0.28
Nodes (6): generate_one(), get_client(), # NOTE: The 53 entries already generated locally are embedded below., Return a Groq client using the current key., Switch to the next available key. Returns False if all keys exhausted., rotate_key()

### Community 3 - "Community 3"
Cohesion: 0.43
Nodes (7): format_training_example(), generate_one(), _get_client(), load_existing_progress(), main(), Generate Evaluator Agent (Bot 4) Training Data — Multi-Key Edition =============, _rotate_key()

### Community 4 - "Community 4"
Cohesion: 0.5
Nodes (2): load_and_format_dataset(), Load the evaluator JSONL and format each example into Phi-3.5's     chat templat

### Community 5 - "Community 5"
Cohesion: 1.0
Nodes (2): App(), useDebounce()

### Community 13 - "Community 13"
Cohesion: 1.0
Nodes (1): Normalise sanitised resume text to match the training-data style exactly.      T

### Community 14 - "Community 14"
Cohesion: 1.0
Nodes (1): Feed pre-formatted resume text into the fine-tuned T5 model.     Returns a parse

### Community 15 - "Community 15"
Cohesion: 1.0
Nodes (1): Parse a duration string like '3 years' / '18 months' / '2 years 6 months'.

### Community 16 - "Community 16"
Cohesion: 1.0
Nodes (1): Parse the pre-formatted resume text using deterministic rules.     This mirrors

### Community 17 - "Community 17"
Cohesion: 1.0
Nodes (1): Main entry point.      Takes GLiNER-sanitised resume text and returns a structur

### Community 18 - "Community 18"
Cohesion: 1.0
Nodes (1): Ensure all required fields exist and highest_degree is a valid enum value.

### Community 19 - "Community 19"
Cohesion: 1.0
Nodes (1): Frontend README

### Community 20 - "Community 20"
Cohesion: 1.0
Nodes (1): Backend Requirements

## Knowledge Gaps
- **25 isolated node(s):** `Quickly infer what role/specialisation the candidate in the resume fits.     Use`, `Extract text from an image file using Tesseract OCR.     Requires the Tesseract`, `Lightweight endpoint: given a pre-computed structured_data dict,     returns the`, `Given a job role title, returns 6-8 typical required skills     using Groq Llama`, `# NOTE: "Date" is intentionally excluded from labels below.` (+20 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 4`** (4 nodes): `install_packages()`, `load_and_format_dataset()`, `Load the evaluator JSONL and format each example into Phi-3.5's     chat templat`, `kaggle_train_evaluator.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 5`** (3 nodes): `App()`, `useDebounce()`, `App.jsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 13`** (1 nodes): `Normalise sanitised resume text to match the training-data style exactly.      T`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 14`** (1 nodes): `Feed pre-formatted resume text into the fine-tuned T5 model.     Returns a parse`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 15`** (1 nodes): `Parse a duration string like '3 years' / '18 months' / '2 years 6 months'.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 16`** (1 nodes): `Parse the pre-formatted resume text using deterministic rules.     This mirrors`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 17`** (1 nodes): `Main entry point.      Takes GLiNER-sanitised resume text and returns a structur`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 18`** (1 nodes): `Ensure all required fields exist and highest_degree is a valid enum value.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 19`** (1 nodes): `Frontend README`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 20`** (1 nodes): `Backend Requirements`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `structure_resume()` connect `Community 0` to `Community 1`?**
  _High betweenness centrality (0.093) - this node is a cross-community bridge._
- **Why does `scan_resume()` connect `Community 1` to `Community 0`?**
  _High betweenness centrality (0.081) - this node is a cross-community bridge._
- **What connects `Quickly infer what role/specialisation the candidate in the resume fits.     Use`, `Extract text from an image file using Tesseract OCR.     Requires the Tesseract`, `Lightweight endpoint: given a pre-computed structured_data dict,     returns the` to the rest of the system?**
  _25 weakly-connected nodes found - possible documentation gaps or missing edges._