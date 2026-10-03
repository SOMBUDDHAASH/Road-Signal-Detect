"""
Evaluation Script for Member C (Classification Lead).
Lead: Member C (Sneha Chakraborty) • Branch: feature/classification

Evaluates the GTSRB classifier model against ground-truth benchmark samples.
Calculates Top-1 accuracy, Top-5 accuracy, and per-category performance.
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.C_classification.classifier import StandaloneGTSRBClassifier
from src.dataset.benchmark_loader import BenchmarkDataLoader
from src.gtsrb_classes import get_sign_category, SignCategory, get_class_name


def evaluate_classifier(model_path: str = None, num_samples: int = 30):
    classifier = StandaloneGTSRBClassifier(model_path=model_path)
    if not classifier.is_ready:
        print("[Error] Classifier model could not be loaded.")
        return

    loader = BenchmarkDataLoader(str(ROOT_DIR / "data"))
    sample_dir = ROOT_DIR / "data" / "samples"
    image_files = sorted(list(sample_dir.glob("*.png")))[:num_samples]

    if not image_files:
        print("[Error] No sample images found in data/samples.")
        return

    total = 0
    top1_correct = 0
    top5_correct = 0
    cat_stats = {cat.value: {"total": 0, "correct": 0} for cat in SignCategory}

    print(f"\n[Evaluation] Evaluating classifier on {len(image_files)} benchmark samples...")
    print(f"{'Filename':<14} | {'Ground Truth':<30} | {'Prediction':<30} | {'Conf':<6} | Match")
    print("-" * 92)

    for img_path in image_files:
        gt = loader.get_ground_truth(img_path.name)
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        if gt:
            # Crop to ROI if specified
            crop = img[gt.roi_y1:gt.roi_y2, gt.roi_x1:gt.roi_x2]
            if crop.size == 0:
                crop = img
            gt_id = gt.class_id
            gt_cat = gt.category
            gt_name = gt.class_name
        else:
            crop = img
            gt_id = -1
            gt_cat = "Unknown"
            gt_name = "Unknown"

        res = classifier.classify(crop)

        total += 1
        is_match = (res.class_id == gt_id)
        if is_match:
            top1_correct += 1

        top_ids = [k[0] for k in res.top_k] if res.top_k else [res.class_id]
        if gt_id in top_ids:
            top5_correct += 1

        if gt_cat in cat_stats:
            cat_stats[gt_cat]["total"] += 1
            if is_match:
                cat_stats[gt_cat]["correct"] += 1

        match_str = "[OK]" if is_match else "[MISMATCH]"
        print(f"{img_path.name:<14} | {gt_name[:28]:<30} | {res.class_name[:28]:<30} | {res.confidence:5.1%} | {match_str}")

    if total == 0:
        print("[Warning] No valid samples could be evaluated.")
        return

    top1_acc = (top1_correct / total) * 100.0
    top5_acc = (top5_correct / total) * 100.0

    print("\n" + "="*60)
    print("             GTSRB CLASSIFIER EVALUATION REPORT")
    print("="*60)
    print(f"Total Evaluated Samples : {total}")
    print(f"Top-1 Accuracy          : {top1_acc:.2f}% ({top1_correct}/{total})")
    print(f"Top-5 Accuracy          : {top5_acc:.2f}% ({top5_correct}/{total})")
    print("-" * 60)
    print("Performance by Category:")
    for cat_name, stats in cat_stats.items():
        if stats["total"] > 0:
            c_acc = (stats["correct"] / stats["total"]) * 100.0
            print(f"  - {cat_name:<16s}: {c_acc:6.2f}% ({stats['correct']}/{stats['total']})")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GTSRB Classifier")
    parser.add_argument("--model", type=str, default=None, help="Path to classifier.pt")
    parser.add_argument("--samples", type=int, default=30, help="Number of test samples to evaluate")
    args = parser.parse_args()

    evaluate_classifier(model_path=args.model, num_samples=args.samples)
