import os
import sys
import time
import socket
import webbrowser
import subprocess
from pathlib import Path


def get_base_dir() -> Path:
    """Get the base project directory whether running as script or frozen exe."""
    if getattr(sys, "frozen", False):
        # Running as compiled PyInstaller executable
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def main():
    base_dir = get_base_dir()
    os.chdir(base_dir)

    print("\n" + "=" * 76)
    print("   AUTONOMOUS ROAD TRAFFIC SIGN DETECTION & RECOGNITION PLATFORM")
    print("   Lead Architect: Sombuddha Ash | GTSRB 43 Classes & TT100K 221 Classes")
    print("=" * 76 + "\n")

    print("[1/4] Verifying project structure and perception weights...")
    yolo_weight = base_dir / "weights" / "detection" / "best.pt"
    cls_weight = base_dir / "weights" / "classification" / "classifier.pt"

    if yolo_weight.exists():
        print(f"  [OK] Member B YOLOv8 Detector found: {yolo_weight.name}")
    else:
        print("  [WARN] Member B YOLO weights not detected; will use robust geometric fallback.")

    if cls_weight.exists():
        print(f"  [OK] Member C Deep CNN Classifier found: {cls_weight.name}")
    else:
        print("  [WARN] Member C weights not detected; using multi-backend fallback.")

    port = 8501
    while is_port_in_use(port):
        print(f"  [INFO] Port {port} is in use, trying {port + 1}...")
        port += 1

    app_url = f"http://localhost:{port}"
    print(f"\n[2/4] Selected dashboard port: {port}")
    print(f"[3/4] Preparing browser launch for: {app_url}")

    # Launch browser after a brief delay so the server has initialized
    def open_browser():
        time.sleep(2.5)
        try:
            webbrowser.open(app_url)
        except Exception:
            pass

    import threading
    t = threading.Thread(target=open_browser, daemon=True)
    t.start()

    print(f"[4/4] Starting Streamlit Perception Dashboard...")
    print("-" * 76)
    print(f"  Access Dashboard : {app_url}")
    print(f"  FastAPI Docs     : http://localhost:8000/docs (if running api.py)")
    print(f"  Stop Application : Press Ctrl+C in this terminal")
    print("-" * 76 + "\n")

    # Locate python executable
    python_exe = sys.executable
    if getattr(sys, "frozen", False):
        # Look for system python or venv python
        venv_py = base_dir / "venv" / "Scripts" / "python.exe"
        if venv_py.exists():
            python_exe = str(venv_py)
        else:
            python_exe = "python"

    cmd = [
        python_exe,
        "-m",
        "streamlit",
        "run",
        str(base_dir / "app.py"),
        "--server.port",
        str(port),
        "--server.headless",
        "false",
        "--browser.serverAddress",
        "localhost",
        "--server.enableCORS",
        "false",
        "--server.enableXsrfProtection",
        "false",
    ]

    try:
        proc = subprocess.run(cmd)
        sys.exit(proc.returncode)
    except KeyboardInterrupt:
        print("\n[STOP] Shutting down ADAS Dashboard gracefully. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Failed to launch Streamlit: {e}")
        print("Fallback: Run `streamlit run app.py` directly from PowerShell/CMD.")
        sys.exit(1)


if __name__ == "__main__":
    main()
