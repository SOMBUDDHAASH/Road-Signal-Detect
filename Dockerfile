# Multi-stage Dockerfile for Self-Hosted Traffic Sign Detection & Recognition System
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and video handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch torchvision yt-dlp mss

# Copy repository files
COPY . .

# Expose ports for Streamlit and FastAPI
EXPOSE 8501 8000

# Default entrypoint starts Streamlit dashboard
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
