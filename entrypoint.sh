#!/bin/bash
set -e

# Set defaults
SERVICE="${SERVICE:-streamlit}"
PORT="${PORT:-8501}"
HOST="${HOST:-0.0.0.0}"

export PYTHONUNBUFFERED=1
export KERAS_BACKEND="${KERAS_BACKEND:-torch}"

echo "========================================================"
echo "  Road Traffic Sign Detection & Recognition System"
echo "  Mode: ${SERVICE} | Host: ${HOST} | Port: ${PORT}"
echo "========================================================"

if [ "$SERVICE" = "api" ] || [ "$SERVICE" = "fastapi" ]; then
    API_PORT="${PORT:-8000}"
    echo "[DEPLOY] Launching FastAPI Production Microservice on ${HOST}:${API_PORT}..."
    exec uvicorn api:app --host "${HOST}" --port "${API_PORT}" --workers "${WORKERS:-2}"
elif [ "$SERVICE" = "all" ] || [ "$SERVICE" = "dual" ]; then
    echo "[DEPLOY] Launching dual services: FastAPI on 8000 and Streamlit on 8501..."
    uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1 &
    exec streamlit run app.py --server.port=8501 --server.address=0.0.0.0
else
    echo "[DEPLOY] Launching Streamlit Production Dashboard on ${HOST}:${PORT}..."
    exec streamlit run app.py --server.port="${PORT}" --server.address="${HOST}"
fi
