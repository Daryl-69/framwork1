# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Kaggle Notebook: Fine-tune Phi-3.5-mini for Resume Evaluator Scorecard   ║
# ║  GPU: T4 16GB  |  Estimated training time: ~2-3 hours                     ║
# ║                                                                            ║
# ║  INSTRUCTIONS:                                                             ║
# ║  1. Create a new Kaggle notebook                                           ║
# ║  2. Enable GPU T4 x2 or P100 accelerator                                  ║
# ║  3. Add your dataset: daryl69/bot-evaluation                               ║
# ║  4. Paste this entire script into a single cell and run                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ============================================================================
# CELL 1: Install Dependencies
# ============================================================================
import subprocess
import sys

def install_packages():
    packages = [
        "transformers>=4.43.0",
        "peft>=0.12.0",
        "trl>=0.9.0",
        "bitsandbytes>=0.43.0",
        "accelerate>=0.33.0",
        "datasets>=2.20.0",
        "scipy",
    ]
    for pkg in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
    print("✅ All packages installed successfully!")

install_packages()

# ============================================================================
# CELL 2: Imports & Config
# ============================================================================
import json
import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer

# ── Paths ───────────────────────────────────────────────────────
INPUT_PATH  = "/kaggle/input/datasets/daryl69/bot-evaluation/evaluator_training_data.jsonl"
OUTPUT_DIR  = "/kaggle/working/evaluator-phi35-lora"
FINAL_DIR   = "/kaggle/working/evaluator-phi35-lora/final"
MODEL_ID    = "microsoft/Phi-3.5-mini-instruct"

# ── Verify input file exists ────────────────────────────────────
if not os.path.exists(INPUT_PATH):
    # Try alternate Kaggle path formats
    alt_paths = [
        "/kaggle/input/bot-evaluation/evaluator_training_data.jsonl",
        "/kaggle/input/daryl69/bot-evaluation/evaluator_training_data.jsonl",
    ]
    for alt in alt_paths:
        if os.path.exists(alt):
            INPUT_PATH = alt
            print(f"⚠️  Using alternate path: {INPUT_PATH}")
            break
    else:
        # List what's actually in /kaggle/input to help debug
        print("❌ Dataset file not found! Searching /kaggle/input/ ...")
        for root, dirs, files in os.walk("/kaggle/input/"):
            for f in files:
                print(f"   Found: {os.path.join(root, f)}")
        raise FileNotFoundError(f"Cannot find evaluator_training_data.jsonl")

print(f"✅ Input file found: {INPUT_PATH}")
print(f"📁 Output directory: {OUTPUT_DIR}")
print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

# ============================================================================
# CELL 3: Load & Format Dataset
# ============================================================================

SYSTEM_PROMPT = (
    "You are an objective scoring engine. Compare the candidate's JSON profile "
    "against the provided Job Description criteria. Assign a score from 0-10 for "
    "each required skill. Provide a 2-sentence purely factual justification for "
    "each score based only on the JSON data. Return a valid JSON object with "
    "scorecard, overall_score, and recommendation fields."
)

def load_and_format_dataset(path):
    """
    Load the evaluator JSONL and format each example into Phi-3.5's
    chat template:  <|system|>...<|end|><|user|>...<|end|><|assistant|>...<|end|>
    """
    examples = []
    skipped = 0

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                row = json.loads(line.strip())
                input_text  = row.get("input", "")
                output_text = row.get("output", "")

                if not input_text or not output_text:
                    skipped += 1
                    continue

                # Strip the system prompt from input if it's already embedded
                # (your data has it baked into the input field)
                clean_input = input_text
                if clean_input.startswith("You are an objective scoring engine."):
                    # Split off the system instruction, keep the JD + candidate part
                    parts = clean_input.split("\n\n", 1)
                    if len(parts) > 1:
                        clean_input = parts[1]

                # Build the chat-formatted text
                text = (
                    f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
                    f"<|user|>\n{clean_input}<|end|>\n"
                    f"<|assistant|>\n{output_text}<|end|>"
                )

                examples.append({"text": text})

            except json.JSONDecodeError:
                skipped += 1
                continue

    print(f"✅ Loaded {len(examples)} training examples (skipped {skipped})")
    return Dataset.from_list(examples)

dataset = load_and_format_dataset(INPUT_PATH)

# Quick sanity check — show the first example's length
sample_tokens = len(dataset[0]["text"].split())
print(f"📊 Sample word count: ~{sample_tokens} words")
print(f"📊 First example preview (first 200 chars):\n{dataset[0]['text'][:200]}...")

# ============================================================================
# CELL 4: Split into Train / Validation
# ============================================================================
split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset  = split["test"]

print(f"🔹 Training examples:   {len(train_dataset)}")
print(f"🔹 Validation examples: {len(eval_dataset)}")

# ============================================================================
# CELL 5: Load Model with 4-bit Quantization
# ============================================================================
print("⏳ Loading Phi-3.5-mini-instruct with 4-bit quantization...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager",   # T4 doesn't support Flash Attention 2
    torch_dtype=torch.float16,
)

model = prepare_model_for_kbit_training(model)
model.config.use_cache = False     # Required for gradient checkpointing

print(f"✅ Model loaded!")
print(f"💾 Model memory: {model.get_memory_footprint() / 1024**3:.2f} GB")

# ============================================================================
# CELL 6: Configure LoRA Adapters
# ============================================================================
lora_config = LoraConfig(
    r=16,                          # Rank — sweet spot for this dataset size
    lora_alpha=32,                 # Alpha = 2x rank is standard
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",   # Attention layers
        "gate_proj", "up_proj", "down_proj",        # MLP layers
    ],
)

# Count trainable parameters
from peft import get_peft_model
model_peft = get_peft_model(model, lora_config)
trainable, total = model_peft.get_nb_trainable_parameters()
print(f"🔧 Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

# ============================================================================
# CELL 7: Training Arguments (Optimized for T4 16GB)
# ============================================================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    # ── Epochs & Batching ───────────────────────────────────
    num_train_epochs=3,
    per_device_train_batch_size=1,       # Must be 1 for 16GB with long sequences
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,       # Effective batch size = 8

    # ── Memory Optimization ─────────────────────────────────
    gradient_checkpointing=True,         # Trades compute for VRAM
    fp16=True,                           # T4 supports fp16
    bf16=False,                          # T4 does NOT support bf16
    optim="paged_adamw_8bit",            # 8-bit optimizer saves ~2GB VRAM

    # ── Learning Rate ───────────────────────────────────────
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    max_grad_norm=0.3,

    # ── Logging & Saving ────────────────────────────────────
    logging_steps=10,
    logging_first_step=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,                  # Keep only last 2 checkpoints (save disk)
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    # ── Misc ────────────────────────────────────────────────
    report_to="none",                    # No wandb/tensorboard on Kaggle
    dataloader_pin_memory=False,
    remove_unused_columns=False,
)

# ============================================================================
# CELL 8: Create Trainer & Start Training
# ============================================================================
print("🚀 Starting training...")
print(f"   Epochs: {training_args.num_train_epochs}")
print(f"   Batch size: {training_args.per_device_train_batch_size} x {training_args.gradient_accumulation_steps} grad accum = effective {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"   Learning rate: {training_args.learning_rate}")
print(f"   Max sequence length: 3072 tokens")
print()

trainer = SFTTrainer(
    model=model_peft,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=lora_config,
    tokenizer=tokenizer,
    args=training_args,
    max_seq_length=3072,                 # Fits your longest examples
    dataset_text_field="text",
    packing=False,                       # Don't pack — each example is already long
)

# Run training
train_result = trainer.train()

# Print training metrics
print("\n" + "=" * 60)
print("📊 TRAINING RESULTS")
print("=" * 60)
metrics = train_result.metrics
print(f"   Total train loss:    {metrics.get('train_loss', 'N/A'):.4f}")
print(f"   Training runtime:    {metrics.get('train_runtime', 0):.0f} seconds")
print(f"   Samples/second:      {metrics.get('train_samples_per_second', 0):.2f}")

# ============================================================================
# CELL 9: Evaluate on Validation Set
# ============================================================================
print("\n⏳ Running evaluation...")
eval_metrics = trainer.evaluate()
print(f"   Eval loss: {eval_metrics.get('eval_loss', 'N/A'):.4f}")

# ============================================================================
# CELL 10: Save the Final Model
# ============================================================================
os.makedirs(FINAL_DIR, exist_ok=True)

# Save LoRA adapter weights (small, ~50-100 MB)
trainer.save_model(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)

print(f"\n✅ Model saved to: {FINAL_DIR}")
print(f"📦 Saved files:")
for f in os.listdir(FINAL_DIR):
    size_mb = os.path.getsize(os.path.join(FINAL_DIR, f)) / (1024 * 1024)
    print(f"   {f} ({size_mb:.1f} MB)")

# ============================================================================
# CELL 11: Test the Model — Generate a Sample Scorecard
# ============================================================================
print("\n" + "=" * 60)
print("🧪 TESTING: Generating a sample scorecard")
print("=" * 60)

model_peft.eval()

test_input = """Job Description:
Job Title: Full Stack Developer
Company: TechStartup Inc.
Requirements:
- 3+ years of experience in full stack development
- Proficiency in React.js and Node.js
- Experience with PostgreSQL or MongoDB
- Knowledge of REST API design
- Bachelor's degree in Computer Science

Candidate JSON Profile:
{
  "total_years_experience": 5,
  "technical_skills": ["JavaScript", "React", "Node.js", "MongoDB"],
  "job_history": [
    {
      "title": "Software Developer",
      "duration_months": 60
    }
  ],
  "highest_degree": "Bachelor"
}"""

test_prompt = (
    f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
    f"<|user|>\n{test_input}<|end|>\n"
    f"<|assistant|>\n"
)

inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")

with torch.no_grad():
    outputs = model_peft.generate(
        **inputs,
        max_new_tokens=1500,
        temperature=0.1,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"\n📋 Generated Scorecard:\n{response}")

# Try to parse as JSON to verify structure
try:
    parsed = json.loads(response)
    print(f"\n✅ Valid JSON! Overall score: {parsed.get('overall_score', 'N/A')}")
    print(f"   Recommendation: {parsed.get('recommendation', 'N/A')}")
    print(f"   Skills scored: {len(parsed.get('scorecard', []))}")
except json.JSONDecodeError as e:
    print(f"\n⚠️  Output is not valid JSON yet (this improves with more training): {e}")

# ============================================================================
# CELL 12: Download Instructions
# ============================================================================
print("\n" + "=" * 60)
print("📥 HOW TO DOWNLOAD YOUR MODEL")
print("=" * 60)
print("""
Option 1 — From Kaggle Output:
   1. Click 'Save Version' (top right) → 'Quick Save'
   2. Go to your notebook's Output tab
   3. Download the 'evaluator-phi35-lora/final/' folder

Option 2 — Zip and download:
   Run this in a new cell:
""")
print("   import shutil")
print(f"   shutil.make_archive('/kaggle/working/evaluator-model', 'zip', '{FINAL_DIR}')")
print("   # Then download /kaggle/working/evaluator-model.zip from Output tab")

print("\n" + "=" * 60)
print("✅ ALL DONE! Your evaluator model is trained and saved.")
print("=" * 60)
