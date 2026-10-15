"""Master Pipeline Runner.
Supports running:
  1. Member A Data Pipeline (download -> clean -> split -> augment -> EDA)
  2. Full Integration Verification (data -> detection -> classification -> pipeline smoke test)

Usage:
  python run_pipeline.py [--skip-download] [--target 500] [--full]
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_pipeline.download_data import download_or_link_dataset
from src.data_pipeline.preprocess import clean_and_extract_crops
from src.data_pipeline.split_data import split_and_save_data
from src.data_pipeline.augment import balance_and_augment
from src.data_pipeline.eda import generate_eda_charts
from src.config import SAMPLES_DIR


def run_pipeline(skip_download: bool = False, target: int = 500, full_test: bool = False):
    print("=" * 65)
    print(" ROAD TRAFFIC SIGN DETECTION & RECOGNITION — INTEGRATED PIPELINE")
    print("=" * 65)

    # Step 0: Download / Link Data
    if not skip_download:
        print("\n--- [Step 0/4] Downloading or Linking GTSRB Dataset ---")
        download_or_link_dataset()
    else:
        print("\n--- [Step 0/4] Skipping dataset download (--skip-download enabled) ---")

    # Step 1: Preprocess & Clean
    print("\n--- [Step 1/4] Preprocessing, ROI Cropping & Cleaning ---")
    clean_and_extract_crops()

    # Step 2: Group/Stratified Split
    print("\n--- [Step 2/4] Train / Validation / Test Splitting ---")
    split_and_save_data()

    # Step 3: Augmentation & Balancing (No Horizontal Flips)
    print(f"\n--- [Step 3/4] Data Augmentation & Balancing (Target: {target}/class) ---")
    balance_and_augment(target_per_class=target)

    # Step 4: Exploratory Data Analysis
    print("\n--- [Step 4/4] Exploratory Data Analysis (EDA Charts) ---")
    generate_eda_charts()

    print("\n" + "=" * 65)
    print(" [DATA PIPELINE COMPLETE] Processed data in data/processed/, charts in results/plots/")
    print("=" * 65)

    if full_test:
        print("\n--- [Step 5] Running End-to-End Perception Smoke Test ---")
        from src.integration.pipeline import TrafficSignPipeline
        pipeline = TrafficSignPipeline()

        sample_img = SAMPLES_DIR / "class_14_sample_1.png"
        if sample_img.exists():
            res = pipeline.process_image(str(sample_img))
            print(f"[TEST PASS] Successfully ran pipeline on sample image!")
            print(f"  Detections: {len(res.detections)} | Latency: {res.inference_time_ms} ms")
            if res.detections:
                det = res.detections[0]
                print(f"  Detected: {det.final_class_name} (Class {det.final_class_id}) | Conf: {det.final_confidence*100:.2f}%")
        else:
            print("[INFO] Sample image not found, skipping visual check.")

    print("\nSystem is fully integrated and ready to run.")
    print("  Launch Web App:   streamlit run app.py")
    print("  Launch REST API:  uvicorn api:app --reload")
    print("  CLI Runner:       python cli.py --source image --input <path>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete data and perception pipeline")
    parser.add_argument("--skip-download", action="store_true", help="Skip dataset download step")
    parser.add_argument("--target", type=int, default=500, help="Target images per class for augmentation")
    parser.add_argument("--full", action="store_true", help="Run end-to-end perception smoke test after data steps")
    args = parser.parse_args()

    run_pipeline(skip_download=args.skip_download, target=args.target, full_test=args.full)
