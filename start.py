"""Cross-Platform 1-Click Interactive Menu Launcher.
Launch with:
  python start.py
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

def menu():
    while True:
        print("\n" + "=" * 65)
        print("  ROAD TRAFFIC SIGN DETECTION & RECOGNITION SYSTEM")
        print("  Integrated Multimodal Vision Platform (Member A, B, C, D)")
        print("=" * 65)
        print(" [1] Launch Interactive Web Dashboard (Streamlit)")
        print(" [2] Launch REST API Service (FastAPI / Uvicorn)")
        print(" [3] Run Automated Test Suite (Pytest)")
        print(" [4] Process Demo Still Image (CLI)")
        print(" [5] Process Demo Driving Video (CLI)")
        print(" [6] Run Full Data Preprocessing Pipeline (Member A)")
        print(" [7] Exit")
        print("=" * 65)

        choice = input("Enter choice (1-7): ").strip()

        if choice == "1":
            print("\nLaunching Streamlit Dashboard...")
            subprocess.run([sys.executable, "-m", "streamlit", "run", str(PROJECT_ROOT / "app.py")])
        elif choice == "2":
            print("\nLaunching FastAPI Server at http://127.0.0.1:8000 ...")
            subprocess.run([sys.executable, "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", "8000", "--reload"])
        elif choice == "3":
            print("\nRunning Pytest Suite...")
            subprocess.run([sys.executable, "-m", "pytest", str(PROJECT_ROOT / "tests"), "-v"])
        elif choice == "4":
            print("\nProcessing Demo Still Image...")
            subprocess.run([sys.executable, str(PROJECT_ROOT / "cli.py"), "--source", "image"])
        elif choice == "5":
            print("\nProcessing Demo Video...")
            subprocess.run([sys.executable, str(PROJECT_ROOT / "cli.py"), "--source", "video"])
        elif choice == "6":
            print("\nRunning Data Pipeline...")
            subprocess.run([sys.executable, str(PROJECT_ROOT / "run_pipeline.py"), "--skip-download", "--full"])
        elif choice == "7":
            print("Exiting.")
            break
        else:
            print("Invalid option. Please enter 1 to 7.")

if __name__ == "__main__":
    menu()
