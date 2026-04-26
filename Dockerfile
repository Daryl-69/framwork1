# Resume Scanner Backend — Render Deployment
#
# This Dockerfile builds a production-ready image for the FastAPI backend.
# It pre-downloads the GLiNER model at build time so cold starts are fast.
# The T5 checkpoint is already in the repo (bot3.1/).
# Phi-3.5 will be downloaded on first /api/evaluate call (or Groq API fallback is used).

FROM python:3.11-slim

# System deps for PyTorch CPU + image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python deps
COPY resume_scanner/backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project (needed for bot3.1/, bot4/, data/)
COPY . .

# Set working directory to the backend
WORKDIR /app/resume_scanner/backend

# Pre-download GLiNER model at build time (avoid slow cold starts)
RUN python -c "from gliner import GLiNER; GLiNER.from_pretrained('urchade/gliner_small-v2.1'); print('[BUILD] GLiNER cached.')"

# Pre-download T5 tokenizer at build time
RUN python -c "from transformers import T5Tokenizer; T5Tokenizer.from_pretrained('t5-base'); print('[BUILD] T5 tokenizer cached.')"

# Expose port (Render sets PORT env var)
EXPOSE 10000

# Start command — Render injects PORT env var
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]
