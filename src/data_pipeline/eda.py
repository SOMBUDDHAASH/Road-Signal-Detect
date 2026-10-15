"""Exploratory Data Analysis (EDA) Module (Member A)
Generates visualizations for class distributions, samples grid, and dataset statistics.
"""

import os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.config import DATA_PROCESSED_DIR, RESULTS_DIR, CLASS_NAMES, NUM_CLASSES


def generate_eda_charts():
    """Generate and save EDA plots."""
    plots_dir = RESULTS_DIR / "plots"
    os.makedirs(plots_dir, exist_ok=True)

    train_path = DATA_PROCESSED_DIR / "train.npz"
    balanced_path = DATA_PROCESSED_DIR / "train_balanced.npz"

    if not train_path.exists():
        print("[EDA] train.npz not found. Skipping EDA charts.")
        return

    data = np.load(train_path)
    X, y = data["X"], data["y"]

    counts_raw = Counter(y)

    counts_bal = None
    if balanced_path.exists():
        data_bal = np.load(balanced_path)
        counts_bal = Counter(data_bal["y"])

    # 1. Class Distribution Chart
    plt.figure(figsize=(14, 6))
    classes = list(range(NUM_CLASSES))
    raw_freqs = [counts_raw.get(c, 0) for c in classes]

    if counts_bal:
        bal_freqs = [counts_bal.get(c, 0) for c in classes]
        width = 0.4
        plt.bar([c - width/2 for c in classes], raw_freqs, width=width, label="Raw Split", color="steelblue")
        plt.bar([c + width/2 for c in classes], bal_freqs, width=width, label="Augmented & Balanced", color="coral")
    else:
        plt.bar(classes, raw_freqs, color="steelblue", label="Training Samples")

    plt.title("GTSRB Class Distribution Across 43 Traffic Sign Categories", fontsize=14)
    plt.xlabel("Class ID (0 - 42)", fontsize=12)
    plt.ylabel("Number of Images", fontsize=12)
    plt.xticks(classes, rotation=90)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    dist_plot_path = plots_dir / "class_distribution.png"
    plt.savefig(dist_plot_path, dpi=150)
    plt.close()
    print(f"[EDA] Class distribution chart saved: {dist_plot_path}")

    # 2. Canonical Sample Grid (Sample of all classes present)
    plt.figure(figsize=(15, 12))
    cols = 7
    rows = (NUM_CLASSES + cols - 1) // cols

    for cid in range(NUM_CLASSES):
        plt.subplot(rows, cols, cid + 1)
        indices = np.where(y == cid)[0]
        if len(indices) > 0:
            sample_img = X[indices[0]]
            plt.imshow(sample_img)
        else:
            plt.imshow(np.zeros((32, 32, 3), dtype=np.uint8))

        name = CLASS_NAMES.get(cid, str(cid))
        if len(name) > 16:
            name = name[:14] + "..."
        plt.title(f"{cid}: {name}", fontsize=8)
        plt.axis("off")

    plt.suptitle("GTSRB Canonical Class Sign Gallery (43 Classes)", fontsize=16)
    plt.tight_layout()

    gallery_plot_path = plots_dir / "class_gallery.png"
    plt.savefig(gallery_plot_path, dpi=150)
    plt.close()
    print(f"[EDA] Class gallery chart saved: {gallery_plot_path}")


if __name__ == "__main__":
    generate_eda_charts()
