FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps (install first for Docker cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# API code
COPY app.py .

# Demo UI
COPY demo/ ./demo/

# Trained model
COPY saved_model/ ./saved_model/

# Environment
ENV MODEL_DIR=/app/saved_model
ENV DEMO_DIR=/app/demo
ENV CONFIDENCE_THRESHOLD=0.65

# Non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
