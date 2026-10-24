# Multi-Stage Production Container for Autonomous Road Traffic Sign Detection & Recognition
FROM python:3.11-slim

WORKDIR /app

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    KERAS_BACKEND=torch \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Install minimal OS dependencies for OpenCV, FFmpeg video decoding, and health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application codebase
COPY . .

# Ensure entrypoint script has execution permissions
RUN chmod +x entrypoint.sh

# Expose ports for Streamlit (8501) and FastAPI (8000)
EXPOSE 8501 8000

# Container healthcheck (checks whichever port is active)
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8501}/_stcore/health || curl -f http://localhost:${PORT:-8000}/health || exit 1

# Production entrypoint
ENTRYPOINT ["/bin/bash", "entrypoint.sh"]
