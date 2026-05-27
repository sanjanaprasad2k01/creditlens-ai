# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.12-slim

# ── Metadata ──────────────────────────────────────────────────────────────────
LABEL maintainer="Sanjana Prasad <sanjana.prasad2023@utexas.edu>"
LABEL description="CreditLens AI — Production Credit Risk Scoring API"
LABEL version="1.0.0"

# ── Set working directory ─────────────────────────────────────────────────────
WORKDIR /app

# ── Install system dependencies ───────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# ── Copy requirements first (Docker layer caching) ────────────────────────────
# This layer only rebuilds when requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ─────────────────────────────────────────────────────
COPY app/        ./app/
COPY src/        ./src/
COPY artifacts/  ./artifacts/

# ── Expose API port ───────────────────────────────────────────────────────────
EXPOSE 8000

# ── Health check for container orchestration ──────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Start the API ─────────────────────────────────────────────────────────────
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]