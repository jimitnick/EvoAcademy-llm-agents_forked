# ── Stage 1: Build ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies (some packages need gcc for C extensions)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime ────────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="abhijith"
LABEL description="EvoAcademy — Evolutionary Algorithm Notebook API"

WORKDIR /app

# Copy installed Python packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source code
COPY app/ ./app/
COPY requirements.txt .
COPY .env.example .

# Create directories for runtime data
RUN mkdir -p storage .chroma_version_store

# Expose the FastAPI port
EXPOSE 8000

# Health check — hits the /health endpoint every 30s
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
