# VIBE – Veritas Integrity Behavior Engine
# Root-level Dockerfile — validated against Scaler pre-validation script

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY environment/ ./environment/
COPY app.py       .
COPY openenv.yaml .

# Copy dataset (CSV lives in data/ folder)
COPY data/        ./data/

# Expose the port declared in openenv.yaml
EXPOSE 7860

# Health check — validator pings /reset
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
