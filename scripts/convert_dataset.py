import json
import os
import sys

# Add the backend folder to Python's path so we can import your existing logic
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'resume_scanner', 'backend'))
sys.path.append(backend_path)

from structure_agent import preformat_resume, _rule_based_fallback, _validate_and_fill

INPUT_FILE = r"D:\1_ai-model\bot 3\training_data (4).jsonl"
OUTPUT_FILE = r"D:\1_ai-model\data\t5_training_data.jsonl"
PREFIX = "Extract JSON from this resume:\n"

# We don't need all 10,000+ examples for a simple fine-tune. 
# 2,000 high-quality examples is the perfect sweet spot for T5-base.
MAX_EXAMPLES = 2000

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"File not found: {INPUT_FILE}")
        return

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    success_count = 0
    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        
        print("Starting conversion...")
        for line in fin:
            if success_count >= MAX_EXAMPLES:
                break
                
            try:
                data = json.loads(line)
                messages = data.get("messages", [])
                
                # Find the user message containing the resume
                user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
                
                # Check if it has the summarization prompt we expect
                if "Please summarize the following resume:" not in user_msg:
                    continue
                    
                # Extract the actual resume text by splitting off the prompt
                parts = user_msg.split("Please summarize the following resume:\n\n", 1)
                if len(parts) < 2:
                    continue
                    
                resume_text = parts[1].strip()
                
                if not resume_text:
                    continue
                    
                # Use your existing, excellent rule-based fallback to generate the "Target" JSON!
                formatted = preformat_resume(resume_text)
                extracted_data = _rule_based_fallback(formatted, resume_text)
                final_data = _validate_and_fill(extracted_data)
                
                # Filter out garbage: only save it if our rule-based engine found at least SOME skills or jobs
                if not final_data["technical_skills"] and not final_data["job_history"]:
                    continue 
                    
                target_json_str = json.dumps(final_data)
                
                # Format exactly how T5 expects it (No 'messages' or 'roles')
                t5_example = {
                    "input_text": f"{PREFIX}{resume_text}",
                    "target_text": target_json_str
                }
                
                fout.write(json.dumps(t5_example) + "\n")
                success_count += 1
                
                if success_count % 250 == 0:
                    print(f"Processed {success_count} valid examples...")
                    
            except Exception as e:
                # If a line is corrupted, just skip it
                continue

    print(f"Done! Saved {success_count} perfectly formatted T5 training examples to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
