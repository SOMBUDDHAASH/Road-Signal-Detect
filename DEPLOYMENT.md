# Production Deployment & Operations Guide

**Autonomous Road Traffic Sign Detection & Recognition System (GTSRB & TT100K)**  
**Version**: 2.0.0 Production Release  

---

## 📌 Executive Overview

The perception system is architected for zero-downtime, multi-target deployment across local containers, edge compute platforms, virtual private servers (VPS), and serverless cloud container environments.

### Deployment Targets Supported
* **Docker & Docker Compose**: 1-command local or server deployment with automated health checks.
* **Streamlit Community Cloud & Hugging Face Spaces**: Instant serverless web dashboard deployment.
* **Render / Railway / Fly.io / Heroku**: Automated Git-push container deployments respecting dynamic `$PORT`.
* **Google Cloud Run / AWS ECS / Azure Container Apps**: Scalable serverless container microservices.
* **Linux Virtual Machines (Ubuntu / Debian)**: Systemd service daemon with Nginx reverse proxy and SSL.

---

## 🚀 1. Docker & Docker Compose Deployment (Recommended)

### 1.1 Multi-Service Deployment (Streamlit + FastAPI)
Launch both the interactive web dashboard (port `8501`) and the REST API microservice (port `8000`) simultaneously:

```bash
docker compose up --build -d
```

* **Streamlit UI**: `http://localhost:8501`
* **FastAPI Docs (Swagger UI)**: `http://localhost:8000/docs`
* **FastAPI Health Endpoint**: `http://localhost:8000/health`

To view streaming container logs:
```bash
docker compose logs -f
```

To stop all services:
```bash
docker compose down
```

---

### 1.2 Single Container Deployment (Docker CLI)

#### Option A: Deploy Streamlit Web Dashboard
```bash
docker build -t traffic-sign-vision .
docker run -d -p 8501:8501 --name traffic-web traffic-sign-vision
```

#### Option B: Deploy FastAPI Microservice
```bash
docker run -d -p 8000:8000 -e SERVICE=api -e PORT=8000 --name traffic-api traffic-sign-vision
```

---

## ☁️ 2. Cloud Platform Deployment

### 2.1 Hugging Face Spaces (Streamlit App)
1. Create a new Space on [Hugging Face](https://huggingface.co/spaces).
2. Select **Streamlit** SDK or **Docker**.
3. Push or connect this repository.
4. The space automatically detects `.streamlit/config.toml` and `app.py`.

### 2.2 Render / Railway / Cloud Run
The container's `entrypoint.sh` automatically detects the dynamic `$PORT` environment variable provided by cloud hosts.
1. Connect your GitHub repository to Render / Railway.
2. Select **Dockerfile** as the build runtime.
3. Configure environment variables (optional):
   * `SERVICE=streamlit` (for Web UI) or `SERVICE=api` (for REST API)
   * `KERAS_BACKEND=torch`
4. The cloud provider will automatically build the image and pass `$PORT`.

---

## 🖥️ 3. Linux VM / Bare-Metal Deployment (Systemd + Nginx)

### 3.1 Install Prerequisites
```bash
sudo apt update && sudo apt install -y python3-pip python3-venv git ffmpeg libgl1 libglib2.0-0 nginx curl
git clone https://github.com/SOMBUDDHAASH/Road-Signal-Detect.git /opt/traffic_sign
cd /opt/traffic_sign
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3.2 Create Systemd Services

#### Service 1: FastAPI Microservice (`/etc/systemd/system/traffic-api.service`)
```ini
[Unit]
Description=Traffic Sign Detection FastAPI Service
After=network.target

[Service]
User=www-data
WorkingDirectory=/opt/traffic_sign
Environment="PATH=/opt/traffic_sign/venv/bin"
Environment="KERAS_BACKEND=torch"
ExecStart=/opt/traffic_sign/venv/bin/uvicorn api:app --host 127.0.0.1 --port 8000 --workers 2
Restart=always

[Install]
WantedBy=multi-user.target
```

#### Service 2: Streamlit Dashboard (`/etc/systemd/system/traffic-web.service`)
```ini
[Unit]
Description=Traffic Sign Detection Streamlit Web Dashboard
After=network.target

[Service]
User=www-data
WorkingDirectory=/opt/traffic_sign
Environment="PATH=/opt/traffic_sign/venv/bin"
Environment="KERAS_BACKEND=torch"
ExecStart=/opt/traffic_sign/venv/bin/streamlit run app.py --server.port 8501 --server.address 127.0.0.1
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start services:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now traffic-api traffic-web
```

---

### 3.3 Nginx Reverse Proxy Configuration
Create `/etc/nginx/sites-available/traffic-sign`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Streamlit Web UI & WebSockets
    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }

    # FastAPI REST Microservice
    location /api/ {
        rewrite ^/api/(.*) /$1 break;
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

Enable site and restart Nginx:
```bash
sudo ln -s /etc/nginx/sites-available/traffic-sign /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl restart nginx
```

---

## ⚙️ 4. Environment Variables Reference

| Variable | Default | Allowed Values | Description |
| :--- | :--- | :--- | :--- |
| `SERVICE` | `streamlit` | `streamlit`, `api`, `fastapi`, `dual` | Defines which application layer starts inside the container. |
| `PORT` | `8501` (web) / `8000` (api) | Any valid port integer | Server port binding (honored dynamically by cloud providers). |
| `HOST` | `0.0.0.0` | IP string | Network interface binding address. |
| `KERAS_BACKEND` | `torch` | `torch`, `tensorflow` | Backend engine powering `.keras` deep learning model execution. |
| `WORKERS` | `2` | Integer $\ge 1$ | Number of concurrent Uvicorn worker processes for FastAPI. |
| `RELOAD` | `false` | `true`, `false` | Enables hot-reload in development (keep `false` in production). |

---

## 🩺 5. Health Checks & Verification

* **FastAPI Service Health**:
  ```bash
  curl -f http://localhost:8000/health
  # Response: {"status":"healthy","service":"traffic-sign-pipeline","version":"2.0.0",...}
  ```
* **Streamlit UI Health**:
  ```bash
  curl -f http://localhost:8501/_stcore/health
  # Response: ok
  ```
* **Automated Unit & Integration Tests**:
  ```bash
  pytest tests/ -v
  # 68 passed (100% pass rate)
  ```
