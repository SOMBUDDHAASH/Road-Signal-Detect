"""
Cross-platform Self-Hosted Launcher for Road / Traffic Sign Detection System.
Provides a simple interactive terminal menu for any new user.
"""

import sys
import os
import subprocess


def print_banner():
    print("""
========================================================================
     TRAFFIC SIGN DETECTION & RECOGNITION (GTSRB BENCHMARK)
               Member D Production Integration Pipeline
========================================================================
    """)


def main():
    print_banner()
    print("Select an option to run:")
    print("  [1] Launch Interactive Web Dashboard (Streamlit UI)")
    print("  [2] Run Continuous Live Stream (Camera / Video / Dashcam)")
    print("  [3] Run Automated Test Suite (pytest)")
    print("  [4] Run REST API Backend (FastAPI)")
    print("  [5] Generate Synthetic Driving Dashcam Video")
    print("  [0] Exit")
    print()

    choice = input("Enter choice [1-5, 0]: ").strip()

    if choice == "1":
        print("\nStarting Streamlit Dashboard...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    elif choice == "2":
        source = input("\nEnter source (0 for webcam, or path to .mp4 video) [default: 0]: ").strip() or "0"
        mode = input("Enter pipeline mode (heuristic / production) [default: production]: ").strip() or "production"
        cmd = [sys.executable, "-m", "src.live_feed", "--source", source, "--mode", mode]
        subprocess.run(cmd)
    elif choice == "3":
        print("\nRunning pytest...")
        subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])
    elif choice == "4":
        print("\nStarting FastAPI REST service on http://127.0.0.1:8000...")
        subprocess.run([sys.executable, "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", "8000", "--reload"])
    elif choice == "5":
        print("\nGenerating simulated driving video...")
        subprocess.run([sys.executable, "scripts/generate_driving_simulation.py"])
    elif choice == "0":
        print("Goodbye!")
        sys.exit(0)
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
