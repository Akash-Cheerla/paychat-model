FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps (install first for Docker cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Server code
COPY eval/ ./eval/

# Trained model
COPY saved_model/ ./saved_model/

# Environment
ENV MODEL_DIR=/app/saved_model
ENV PORT=8001

# Non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8001/meta || exit 1

CMD ["python", "eval/eval_server.py"]
