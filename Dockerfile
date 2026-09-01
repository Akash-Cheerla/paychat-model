FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps (install first for Docker cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

# Server code
COPY app.py .
COPY conversation.py .
COPY guardrails.py .
COPY conv_classifier.py .

# Tester chat page, served at /chat
COPY demo/ ./demo/

# Trained models
COPY saved_model/ ./saved_model/
COPY conv_model/ ./conv_model/

# Environment
ENV MODEL_DIR=/app/saved_model
ENV PORT=8000
# Production runs the classifier. The default used to be 0 while it was still
# behind a flag, which meant a clean rebuild silently fell back to the rule
# layer unless the deploy config remembered to override it.
ENV PAYCHAT_CONV_CLASSIFIER=1

# Non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
