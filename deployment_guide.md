# 🚀 Deployment Guide — Resume Scanner

## What's Already in the Repo (No Action Needed)

| Item | Path | Size | Notes |
|------|------|------|-------|
| Backend source code | `resume_scanner/backend/` | — | FastAPI server |
| Frontend source code | `resume_scanner/frontend/` | — | React + Vite |
| Bot4 LoRA adapter | `bot4/` | ~6 MB | Fine-tuned Phi-3.5 weights |
| Training data | `data/` | ~34 KB | JSONL files |
| Scripts | `scripts/` | — | Data generation, tests |
| `.env.example` | root | — | Template for API keys |

> [!IMPORTANT]
> You do **NOT** need to upload datasets or model adapters anywhere — they're already committed to GitHub.

---

## What's NOT in the Repo (Must Set Up on Each Machine)

These are gitignored on purpose:

| Item | Why Gitignored | What To Do |
|------|----------------|-----------|
| `.env` | Contains secret API keys | Copy `.env.example` → `.env`, fill in your real keys |
| `venv/` | Python virtual environment (~500 MB+) | Recreate with `pip install` |
| `node_modules/` | Node dependencies (~100 MB+) | Recreate with `npm install` |
| Base ML models | Phi-3.5-mini (~2.5 GB), GLiNER, T5 | **Auto-downloaded** from HuggingFace on first run |

---

## Step-by-Step: Deploy on a New Machine

### Prerequisites
- **Python 3.10+**
- **Node.js 18+**  
- **NVIDIA GPU** with ≥4 GB VRAM (for Bot4 evaluator) — or CPU-only (slower)
- **Git**

### 1. Clone the Repo
```bash
git clone https://github.com/Daryl-69/framwork1.git
cd framwork1
```

### 2. Create `.env` with Your API Keys
```bash
# Windows
copy .env.example .env

# Linux/Mac
cp .env.example .env
```
Then edit `.env` and fill in your real keys:
```env
LLAMA_CLOUD_API_KEY=llx-your-real-key-here
GROQ_API_KEY=gsk_your-real-key-here
GEMINI_API_KEY=AIzaSy-your-real-key-here
```

### 3. Set Up the Backend
```bash
cd resume_scanner/backend
python -m venv venv

# Activate venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 4. Set Up the Frontend
```bash
cd resume_scanner/frontend
npm install
```

### 5. Run Both Services
```bash
# Terminal 1 — Backend (from resume_scanner/backend/)
.\venv\Scripts\python.exe -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — Frontend (from resume_scanner/frontend/)
npm run dev
```

### 6. Open in Browser
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000

---

## What Happens on First Run (Auto-Downloads)

The first time the backend processes a request, it will **automatically download** these models from HuggingFace:

| Model | When Downloaded | Size | Used For |
|-------|----------------|------|----------|
| `urchade/gliner_small-v2.1` | First `/api/scan-resume` call | ~100 MB | Anonymising PII |
| T5 (via `structure_agent`) | First structuring call | ~900 MB | Converting resume text → JSON |
| `microsoft/Phi-3.5-mini-instruct` | First `/api/evaluate` call | ~2.5 GB (4-bit) | Evaluating candidate vs JD |

> [!NOTE]
> After the first download, models are cached in `~/.cache/huggingface/` and won't re-download.

---

## Cloud Deployment Options

If you want to deploy publicly (not just localhost):

### Option A: Railway / Render (Easiest, but no GPU)
- Works for the backend **without** Bot4 evaluator (CPU inference is very slow for Phi-3.5)
- The evaluator will return empty scorecards; everything else works
- Free tiers available

### Option B: Google Colab / Kaggle (Free GPU)
- Good for **testing** with Bot4 evaluator
- Not a permanent hosting solution

### Option C: Hugging Face Spaces (Free, limited GPU)
- Deploy as a Gradio/Streamlit app
- T4 GPU available on free tier (16 GB VRAM — more than enough)

### Option D: VPS with GPU (AWS, GCP, RunPod)
- Full control, persistent deployment
- Cheapest GPU instances: ~$0.20-0.50/hr
- Recommended: at least 8 GB VRAM

> [!TIP]
> For a college project demo, just **run it locally** on your machine — that's the simplest path. The backend + frontend are already running on your PC right now.

---

## TL;DR

**Q: Do I need to upload the datasets/models somewhere?**  
**A: No.** The LoRA adapter (`bot4/`) and training data (`data/`) are already in the GitHub repo. The large base models (Phi-3.5, GLiNER, T5) download automatically on first use. You only need to create a `.env` file with your API keys and run `pip install` + `npm install`.
