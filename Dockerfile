# ============================================================================
# Recognize AI — Nextcloud ExApp Dockerfile
#
# Build (CPU):  docker build -t recognize-ai .
# Build (GPU):  docker build --build-arg GPU=true -t recognize-ai-gpu .
# Run:          docker run -p 8000:8000 -v ./models:/app/models recognize-ai
#
# ExApp mode:   Deployed automatically by Nextcloud AppAPI
# ============================================================================

ARG GPU=false

# ---------------------------------------------------------------------------
# Base stage — shared between CPU and GPU
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

LABEL org.opencontainers.image.title="Recognize AI"
LABEL org.opencontainers.image.description="High-performance AI backend for Nextcloud Recognize"
LABEL org.opencontainers.image.source="https://github.com/pener/recognize-ai-backend"
LABEL org.opencontainers.image.vendor="pener"
LABEL io.nextcloud.app.id="recognize_ai"

WORKDIR /app

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# ---------------------------------------------------------------------------
# CPU variant
# ---------------------------------------------------------------------------
FROM base AS cpu-deps
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# GPU variant
# ---------------------------------------------------------------------------
FROM base AS gpu-deps
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir onnxruntime-gpu

# ---------------------------------------------------------------------------
# Final stage — pick CPU or GPU based on build arg
# ---------------------------------------------------------------------------
FROM ${GPU:+gpu-deps} AS with-gpu
FROM cpu-deps AS without-gpu

# This trick selects gpu-deps if GPU=true, cpu-deps otherwise
FROM ${GPU:+with-gpu}${GPU:-without-gpu} AS final

WORKDIR /app

# Copy Nextcloud app manifest
COPY appinfo/ /app/appinfo/

# Copy application code
COPY config.py main.py inference.py utils.py nc_app.py scanner.py ./

# Models directory (mount or copy externally)
RUN mkdir -p /app/models

# Configure for container environment
ENV RECOGNIZE_HOST=0.0.0.0
ENV RECOGNIZE_PORT=8000
ENV RECOGNIZE_MODELS_DIR=/app/models

# AppAPI will inject these at runtime:
# ENV APP_ID=recognize_ai
# ENV APP_SECRET=...
# ENV APP_PORT=8000
# ENV NEXTCLOUD_URL=...

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/health'); r.raise_for_status()" || exit 1

CMD ["python", "main.py"]
