---
aliases: [structure_agent.py]
---

# structure_agent.py

## Description
Structure Agent (v2 — Fine-tuned T5 / Bot 3) -----------------------------------

## Functions
### `_load_model()`

### `preformat_resume()`
> Normalise sanitised resume text to match the training-data style exactly.      T

### `_run_t5()`
> Feed pre-formatted resume text into the fine-tuned T5 model.     Returns a parse

### `_extract_duration_months()`
> Parse a duration string like '3 years' / '18 months' / '2 years 6 months'.

### `_rule_based_fallback()`
> Parse the pre-formatted resume text using deterministic rules.     This mirrors

### `structure_resume()`
> Main entry point.      Takes GLiNER-sanitised resume text and returns a structur

### `_validate_and_fill()`
> Ensure all required fields exist and highest_degree is a valid enum value.

## Dependencies
- [[convert_dataset]]
